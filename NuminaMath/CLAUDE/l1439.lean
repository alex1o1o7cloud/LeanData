import Mathlib

namespace NUMINAMATH_CALUDE_quilt_square_transformation_l1439_143961

theorem quilt_square_transformation (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 12 := by
sorry

end NUMINAMATH_CALUDE_quilt_square_transformation_l1439_143961


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l1439_143976

/-- Two parabolas with different vertices, where each parabola's vertex lies on the other parabola -/
structure TwoParabolas where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  p : ℝ
  q : ℝ
  h_diff_vertices : x₁ ≠ x₂
  h_vertex_on_other₁ : y₂ = p * (x₂ - x₁)^2 + y₁
  h_vertex_on_other₂ : y₁ = q * (x₁ - x₂)^2 + y₂

/-- The sum of the leading coefficients of two parabolas with the described properties is zero -/
theorem sum_of_coefficients_is_zero (tp : TwoParabolas) : tp.p + tp.q = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l1439_143976


namespace NUMINAMATH_CALUDE_system_solution_unique_l1439_143942

theorem system_solution_unique :
  ∃! (x y : ℚ),
    1 / (2 - x + 2 * y) - 1 / (x + 2 * y - 1) = 2 ∧
    1 / (2 - x + 2 * y) - 1 / (1 - x - 2 * y) = 4 ∧
    x = 11 / 6 ∧
    y = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1439_143942


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1439_143968

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Ensure positive side lengths
  (a^2 + b^2 = c^2) →        -- Pythagorean theorem (right-angled triangle)
  (a^2 + b^2 + c^2 = 2450) → -- Sum of squares condition
  (b = a + 7) →              -- One leg is 7 units longer
  c = 35 := by               -- Conclusion: hypotenuse length is 35
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1439_143968


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1439_143941

def rahul_age_after_6_years : ℕ := 26
def years_until_rahul_age : ℕ := 6
def deepak_current_age : ℕ := 8

theorem rahul_deepak_age_ratio :
  let rahul_current_age := rahul_age_after_6_years - years_until_rahul_age
  (rahul_current_age : ℚ) / deepak_current_age = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1439_143941


namespace NUMINAMATH_CALUDE_intersection_equals_subset_implies_union_equals_set_l1439_143917

theorem intersection_equals_subset_implies_union_equals_set 
  (M P : Set α) (h : M ∩ P = P) : M ∪ P = M := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_subset_implies_union_equals_set_l1439_143917


namespace NUMINAMATH_CALUDE_unique_configuration_l1439_143969

/-- A configuration of n points in the plane with associated real numbers. -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points in the plane. -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- Predicate stating that three points are not collinear. -/
def nonCollinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop := sorry

/-- The configuration satisfies the area condition for all triples of points. -/
def satisfiesAreaCondition (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i < j → j < k →
    triangleArea (config.points i) (config.points j) (config.points k) =
    config.r i + config.r j + config.r k

/-- The configuration satisfies the non-collinearity condition for all triples of points. -/
def satisfiesNonCollinearityCondition (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    nonCollinear (config.points i) (config.points j) (config.points k)

/-- The main theorem stating that 4 is the only integer greater than 3 satisfying the conditions. -/
theorem unique_configuration :
  ∀ (n : ℕ), n > 3 →
  (∃ (config : PointConfiguration n),
    satisfiesAreaCondition config ∧
    satisfiesNonCollinearityCondition config) →
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_configuration_l1439_143969


namespace NUMINAMATH_CALUDE_hasans_plates_l1439_143931

/-- Proves the number of plates initially in Hasan's box -/
theorem hasans_plates
  (plate_weight : ℕ)
  (max_weight_oz : ℕ)
  (removed_plates : ℕ)
  (h1 : plate_weight = 10)
  (h2 : max_weight_oz = 20 * 16)
  (h3 : removed_plates = 6) :
  (max_weight_oz + removed_plates * plate_weight) / plate_weight = 38 := by
  sorry

end NUMINAMATH_CALUDE_hasans_plates_l1439_143931


namespace NUMINAMATH_CALUDE_expression_equality_l1439_143913

theorem expression_equality (w : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 1.00 / Real.sqrt w = 2.650793650793651) → 
  w = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1439_143913


namespace NUMINAMATH_CALUDE_divisibility_by_53_l1439_143929

theorem divisibility_by_53 (n : ℕ) : 53 ∣ (10^(n+3) + 17) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_53_l1439_143929


namespace NUMINAMATH_CALUDE_nba_games_total_l1439_143940

theorem nba_games_total (bulls_wins heat_wins knicks_wins : ℕ) : 
  bulls_wins = 70 →
  heat_wins = bulls_wins + 5 →
  knicks_wins = 2 * heat_wins →
  bulls_wins + heat_wins + knicks_wins = 295 := by
  sorry

end NUMINAMATH_CALUDE_nba_games_total_l1439_143940


namespace NUMINAMATH_CALUDE_apple_probability_l1439_143972

def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def chosen_apples : ℕ := 3

theorem apple_probability :
  (Nat.choose red_apples chosen_apples +
   Nat.choose green_apples chosen_apples +
   (Nat.choose red_apples 2 * Nat.choose green_apples 1) +
   (Nat.choose green_apples 2 * Nat.choose red_apples 1)) /
  Nat.choose total_apples chosen_apples = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_probability_l1439_143972


namespace NUMINAMATH_CALUDE_circle_chord_length_l1439_143926

theorem circle_chord_length (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0 → (∃ x₀ y₀ : ℝ, x₀ + y₀ + 4 = 0 ∧ 
    (x - x₀)^2 + (y - y₀)^2 = 2^2)) → 
  a = -7 := by
sorry


end NUMINAMATH_CALUDE_circle_chord_length_l1439_143926


namespace NUMINAMATH_CALUDE_rulers_placed_l1439_143983

theorem rulers_placed (initial_rulers final_rulers : ℕ) (h : final_rulers = initial_rulers + 14) :
  final_rulers - initial_rulers = 14 := by
  sorry

end NUMINAMATH_CALUDE_rulers_placed_l1439_143983


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1439_143967

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1439_143967


namespace NUMINAMATH_CALUDE_vector_at_negative_two_l1439_143953

/-- A parameterized line in 2D space. -/
structure ParameterizedLine where
  vector : ℝ → (ℝ × ℝ)

/-- Given conditions for the parameterized line. -/
def line_conditions (L : ParameterizedLine) : Prop :=
  L.vector 1 = (2, 5) ∧ L.vector 4 = (5, -7)

/-- The theorem stating the vector at t = -2 given the conditions. -/
theorem vector_at_negative_two
  (L : ParameterizedLine)
  (h : line_conditions L) :
  L.vector (-2) = (-1, 17) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_two_l1439_143953


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l1439_143944

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.65

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 12

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * (num_tshirts : ℝ)

/-- Theorem: The total cost of Carrie's t-shirt purchase is $115.80 -/
theorem carries_tshirt_purchase : total_cost = 115.80 := by
  sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l1439_143944


namespace NUMINAMATH_CALUDE_one_meeting_l1439_143948

/-- Represents a boy moving on a circular track -/
structure Boy where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The problem setup -/
def circularTrackProblem (circumference : ℝ) (boy1 boy2 : Boy) : Prop :=
  circumference > 0 ∧
  boy1.speed = 6 ∧
  boy2.speed = 10 ∧
  boy1.direction ≠ boy2.direction

/-- The number of meetings between the two boys -/
def numberOfMeetings (circumference : ℝ) (boy1 boy2 : Boy) : ℕ := sorry

/-- The theorem stating that the boys meet exactly once -/
theorem one_meeting (circumference : ℝ) (boy1 boy2 : Boy) 
  (h : circularTrackProblem circumference boy1 boy2) : 
  numberOfMeetings circumference boy1 boy2 = 1 := by sorry

end NUMINAMATH_CALUDE_one_meeting_l1439_143948


namespace NUMINAMATH_CALUDE_least_addition_for_multiple_of_five_l1439_143910

theorem least_addition_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (879 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (879 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_multiple_of_five_l1439_143910


namespace NUMINAMATH_CALUDE_largest_power_of_five_factor_l1439_143960

-- Define factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the sum of factorials
def sum_of_factorials : ℕ := factorial 77 + factorial 78 + factorial 79

-- Define the function to count factors of 5
def count_factors_of_five (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + if x % 5 = 0 then 1 else 0) 0

-- Theorem statement
theorem largest_power_of_five_factor :
  ∃ (n : ℕ), n = 18 ∧ 5^n ∣ sum_of_factorials ∧ ¬(5^(n+1) ∣ sum_of_factorials) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_five_factor_l1439_143960


namespace NUMINAMATH_CALUDE_yellow_paint_theorem_l1439_143938

/-- Represents the ratio of paints in the mixture -/
structure PaintRatio :=
  (blue : ℚ)
  (yellow : ℚ)
  (white : ℚ)

/-- Calculates the amount of yellow paint needed given the amount of white paint and the ratio -/
def yellow_paint_amount (ratio : PaintRatio) (white_amount : ℚ) : ℚ :=
  (ratio.yellow / ratio.white) * white_amount

/-- Theorem stating that given the specific ratio and white paint amount, 
    the yellow paint amount should be 9 quarts -/
theorem yellow_paint_theorem (ratio : PaintRatio) (white_amount : ℚ) :
  ratio.blue = 4 ∧ ratio.yellow = 3 ∧ ratio.white = 5 ∧ white_amount = 15 →
  yellow_paint_amount ratio white_amount = 9 := by
  sorry


end NUMINAMATH_CALUDE_yellow_paint_theorem_l1439_143938


namespace NUMINAMATH_CALUDE_raccoon_lock_problem_l1439_143991

theorem raccoon_lock_problem :
  ∀ (x : ℝ),
  let first_lock_time := 5
  let second_lock_time := x * first_lock_time - 3
  let both_locks_time := 5 * second_lock_time
  both_locks_time = 60 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_raccoon_lock_problem_l1439_143991


namespace NUMINAMATH_CALUDE_inequality_solution_l1439_143951

theorem inequality_solution (x : ℝ) : 
  (12 * x^3 + 24 * x^2 - 75 * x - 3) / ((3 * x - 4) * (x + 5)) < 6 ↔ -5 < x ∧ x < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1439_143951


namespace NUMINAMATH_CALUDE_trail_mix_composition_l1439_143956

/-- The weight of peanuts used in the trail mix -/
def peanuts : ℚ := 0.16666666666666666

/-- The weight of raisins used in the trail mix -/
def raisins : ℚ := 0.08333333333333333

/-- The total weight of the trail mix -/
def total_mix : ℚ := 0.4166666666666667

/-- The weight of chocolate chips used in the trail mix -/
def chocolate_chips : ℚ := total_mix - (peanuts + raisins)

theorem trail_mix_composition :
  chocolate_chips = 0.1666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_composition_l1439_143956


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1439_143992

open Real

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_pos : ∀ x, x > 0 → f x > 0) 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_eq : ∀ x, x > 0 → deriv f (a / x) = x / f x) :
  ∃ b : ℝ, b > 0 ∧ ∀ x, x > 0 → f x = a^(1 - a/b) * x^(a/b) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1439_143992


namespace NUMINAMATH_CALUDE_diamond_four_three_l1439_143906

/-- Diamond operation: a ◇ b = 4a + 3b - ab + a² + b² -/
def diamond (a b : ℝ) : ℝ := 4*a + 3*b - a*b + a^2 + b^2

theorem diamond_four_three : diamond 4 3 = 38 := by sorry

end NUMINAMATH_CALUDE_diamond_four_three_l1439_143906


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1439_143909

theorem matrix_N_satisfies_conditions :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![2, -1, 6; 3, 4, 0; -1, 1, -3]
  let i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
  let j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
  let k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]
  N * i = !![1; 2; -5] + !![1; 1; 4] ∧
  N * j = !![-1; 4; 1] ∧
  N * k = !![6; 0; -3] :=
by sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1439_143909


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l1439_143916

/-- Given point A(1, 0) and line l: y = 2x - 4, with point R on line l such that
    vector RA equals vector AP, prove that the trajectory of point P is y = 2x -/
theorem trajectory_of_point_P (R P : ℝ × ℝ) :
  (∃ a : ℝ, R = (a, 2 * a - 4)) →  -- R is on line l: y = 2x - 4
  (R.1 - 1, R.2) = (P.1 - 1, P.2) →  -- vector RA = vector AP
  P.2 = 2 * P.1 :=  -- trajectory of P is y = 2x
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l1439_143916


namespace NUMINAMATH_CALUDE_jane_initial_crayons_l1439_143946

/-- The number of crayons Jane started with -/
def initial_crayons : ℕ := sorry

/-- The number of crayons eaten by the hippopotamus -/
def eaten_crayons : ℕ := 7

/-- The number of crayons Jane ended with -/
def final_crayons : ℕ := 80

/-- Theorem stating that Jane started with 87 crayons -/
theorem jane_initial_crayons : initial_crayons = 87 := by
  sorry

end NUMINAMATH_CALUDE_jane_initial_crayons_l1439_143946


namespace NUMINAMATH_CALUDE_concert_revenue_l1439_143937

/-- Calculate the total revenue from concert ticket sales --/
theorem concert_revenue (ticket_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
  (first_group : ℕ) (second_group : ℕ) (total_people : ℕ) :
  ticket_price = 20 →
  first_discount = 0.4 →
  second_discount = 0.15 →
  first_group = 10 →
  second_group = 20 →
  total_people = 45 →
  (first_group * ticket_price * (1 - first_discount) +
   second_group * ticket_price * (1 - second_discount) +
   (total_people - first_group - second_group) * ticket_price) = 760 := by
sorry

end NUMINAMATH_CALUDE_concert_revenue_l1439_143937


namespace NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l1439_143984

theorem mrs_hilt_pizza_slices :
  ∀ (num_pizzas : ℕ) (slices_per_pizza : ℕ),
    num_pizzas = 5 →
    slices_per_pizza = 12 →
    num_pizzas * slices_per_pizza = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l1439_143984


namespace NUMINAMATH_CALUDE_coefficient_of_minus_two_ab_l1439_143974

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℤ) (x : String) : ℤ := m

/-- Given monomial -2ab, prove its coefficient is -2 -/
theorem coefficient_of_minus_two_ab :
  coefficient (-2) "ab" = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_minus_two_ab_l1439_143974


namespace NUMINAMATH_CALUDE_sandy_younger_than_molly_l1439_143922

theorem sandy_younger_than_molly (sandy_age molly_age : ℕ) : 
  sandy_age = 63 → 
  sandy_age * 9 = molly_age * 7 → 
  molly_age - sandy_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sandy_younger_than_molly_l1439_143922


namespace NUMINAMATH_CALUDE_hearty_beads_count_l1439_143988

/-- The number of packages of blue beads -/
def blue_packages : ℕ := 4

/-- The number of packages of red beads -/
def red_packages : ℕ := 5

/-- The number of packages of green beads -/
def green_packages : ℕ := 2

/-- The number of beads in each blue package -/
def blue_beads_per_package : ℕ := 30

/-- The number of beads in each red package -/
def red_beads_per_package : ℕ := 45

/-- The number of additional beads in each green package compared to a blue package -/
def green_extra_beads : ℕ := 15

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * blue_beads_per_package + 
                        red_packages * red_beads_per_package + 
                        green_packages * (blue_beads_per_package + green_extra_beads)

theorem hearty_beads_count : total_beads = 435 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l1439_143988


namespace NUMINAMATH_CALUDE_ghee_mixture_quantity_l1439_143914

theorem ghee_mixture_quantity (original_ghee_percent : Real) 
                               (original_vanaspati_percent : Real)
                               (original_palm_oil_percent : Real)
                               (added_ghee : Real)
                               (added_palm_oil : Real)
                               (final_vanaspati_percent : Real) :
  original_ghee_percent = 0.55 →
  original_vanaspati_percent = 0.35 →
  original_palm_oil_percent = 0.10 →
  added_ghee = 15 →
  added_palm_oil = 5 →
  final_vanaspati_percent = 0.30 →
  ∃ (original_quantity : Real),
    original_quantity = 120 ∧
    original_vanaspati_percent * original_quantity = 
      final_vanaspati_percent * (original_quantity + added_ghee + added_palm_oil) :=
by sorry

end NUMINAMATH_CALUDE_ghee_mixture_quantity_l1439_143914


namespace NUMINAMATH_CALUDE_arithmetic_sum_l1439_143993

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 3 = 13 →
  a 1 = 2 →
  a 4 + a 5 + a 6 = 42 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l1439_143993


namespace NUMINAMATH_CALUDE_lollipop_count_l1439_143975

theorem lollipop_count (total_cost : ℝ) (single_cost : ℝ) (count : ℕ) : 
  total_cost = 90 →
  single_cost = 0.75 →
  (count : ℝ) * single_cost = total_cost →
  count = 120 := by
sorry

end NUMINAMATH_CALUDE_lollipop_count_l1439_143975


namespace NUMINAMATH_CALUDE_square_minus_twice_plus_nine_equals_eleven_l1439_143954

theorem square_minus_twice_plus_nine_equals_eleven :
  let a : ℝ := 2 / (Real.sqrt 3 - 1)
  a^2 - 2*a + 9 = 11 := by sorry

end NUMINAMATH_CALUDE_square_minus_twice_plus_nine_equals_eleven_l1439_143954


namespace NUMINAMATH_CALUDE_or_false_implies_both_false_l1439_143934

theorem or_false_implies_both_false (p q : Prop) : 
  (¬p ∨ ¬q) → (¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_or_false_implies_both_false_l1439_143934


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1439_143936

/-- The line equation as a function of m, x, and y -/
def line_equation (m x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The theorem stating that the line passes through (-2, 3) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m (-2) 3 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1439_143936


namespace NUMINAMATH_CALUDE_tan_triple_angle_l1439_143915

theorem tan_triple_angle (α : Real) (P : ℝ × ℝ) :
  α > 0 ∧ α < π / 2 →  -- α is acute
  P.1 = 2 * (Real.cos (280 * π / 180))^2 →  -- x-coordinate of P
  P.2 = Real.sin (20 * π / 180) →  -- y-coordinate of P
  Real.tan (3 * α) = Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_tan_triple_angle_l1439_143915


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l1439_143904

theorem cos_alpha_plus_5pi_12 (α : Real) (h : Real.sin (α - π/12) = 1/3) :
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l1439_143904


namespace NUMINAMATH_CALUDE_number_equals_five_l1439_143982

theorem number_equals_five (N x : ℝ) (h1 : N / (4 + 1/x) = 1) (h2 : x = 1) : N = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_five_l1439_143982


namespace NUMINAMATH_CALUDE_meeting_time_calculation_l1439_143970

-- Define the speeds of the two people
def v₁ : ℝ := 6
def v₂ : ℝ := 4

-- Define the time difference in reaching the final destination
def time_difference : ℝ := 10

-- Define the theorem to prove
theorem meeting_time_calculation (t₁ : ℝ) :
  v₂ * t₁ = v₁ * (t₁ - time_difference) → t₁ = 30 :=
by sorry

end NUMINAMATH_CALUDE_meeting_time_calculation_l1439_143970


namespace NUMINAMATH_CALUDE_num_keepers_is_correct_l1439_143989

/-- The number of keepers in a caravan with hens, goats, and camels. -/
def num_keepers : ℕ :=
  let num_hens : ℕ := 50
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let hen_feet : ℕ := 2
  let goat_feet : ℕ := 4
  let camel_feet : ℕ := 4
  let keeper_feet : ℕ := 2
  let total_animal_feet : ℕ := num_hens * hen_feet + num_goats * goat_feet + num_camels * camel_feet
  let total_animal_heads : ℕ := num_hens + num_goats + num_camels
  let extra_feet : ℕ := 224
  15

theorem num_keepers_is_correct : num_keepers = 15 := by
  sorry

#eval num_keepers

end NUMINAMATH_CALUDE_num_keepers_is_correct_l1439_143989


namespace NUMINAMATH_CALUDE_evaluate_expression_l1439_143908

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) :
  y * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1439_143908


namespace NUMINAMATH_CALUDE_adam_tshirts_correct_l1439_143985

/-- The number of t-shirts Adam initially took out -/
def adam_tshirts : ℕ := 20

/-- The total number of clothing items donated -/
def total_donated : ℕ := 126

/-- The number of Adam's friends who donated -/
def friends_donating : ℕ := 3

/-- The number of pants Adam took out -/
def adam_pants : ℕ := 4

/-- The number of jumpers Adam took out -/
def adam_jumpers : ℕ := 4

/-- The number of pajama sets Adam took out -/
def adam_pajamas : ℕ := 4

/-- Theorem stating that the number of t-shirts Adam initially took out is correct -/
theorem adam_tshirts_correct : 
  (adam_pants + adam_jumpers + 2 * adam_pajamas + adam_tshirts) / 2 + 
  friends_donating * (adam_pants + adam_jumpers + 2 * adam_pajamas + adam_tshirts) = 
  total_donated :=
sorry

end NUMINAMATH_CALUDE_adam_tshirts_correct_l1439_143985


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_less_than_1000_l1439_143918

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem power_of_two_less_than_1000 :
  (¬ ∃ n : ℕ, 2^n < 1000) ↔ (∀ n : ℕ, 2^n ≥ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_less_than_1000_l1439_143918


namespace NUMINAMATH_CALUDE_f_2021_value_l1439_143955

-- Define the set A
def A : Set ℚ := {x : ℚ | x ≠ -1 ∧ x ≠ 0}

-- Define the function property
def has_property (f : A → ℝ) : Prop :=
  ∀ x : A, f x + f ⟨1 + 1 / x, sorry⟩ = (1/2) * Real.log (abs (x : ℝ))

-- State the theorem
theorem f_2021_value (f : A → ℝ) (h : has_property f) :
  f ⟨2021, sorry⟩ = (1/2) * Real.log 2021 := by sorry

end NUMINAMATH_CALUDE_f_2021_value_l1439_143955


namespace NUMINAMATH_CALUDE_binary_101_is_5_l1439_143901

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₂ -/
def binary_101 : List Bool := [true, false, true]

/-- Theorem stating that the decimal representation of 101₂ is 5 -/
theorem binary_101_is_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_is_5_l1439_143901


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1439_143986

/-- Represents a pyramid with a square base and isosceles right triangle lateral faces -/
structure Pyramid where
  base_side_length : ℝ
  is_square_base : base_side_length = 2
  is_isosceles_right_triangle_faces : True

/-- Represents a cube inside the pyramid -/
structure InsideCube where
  edge_length : ℝ
  vertex_at_base_center : True
  three_vertices_touch_faces : True

/-- The volume of the cube inside the pyramid is 1 -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : InsideCube) : c.edge_length ^ 3 = 1 := by
  sorry

#check cube_volume_in_pyramid

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1439_143986


namespace NUMINAMATH_CALUDE_race_distance_proof_l1439_143903

/-- The total distance Jesse and Mia each need to run in a week-long race. -/
def total_distance : ℝ := 48

theorem race_distance_proof (jesse_first_three : ℝ) (jesse_day_four : ℝ) (mia_first_four : ℝ) (final_three_avg : ℝ) :
  jesse_first_three = 3 * (2/3) →
  jesse_day_four = 10 →
  mia_first_four = 4 * 3 →
  final_three_avg = 6 →
  total_distance = jesse_first_three + jesse_day_four + (3 * 2 * final_three_avg) / 2 :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l1439_143903


namespace NUMINAMATH_CALUDE_probability_two_females_one_male_l1439_143981

theorem probability_two_females_one_male (total : ℕ) (females : ℕ) (males : ℕ) (chosen : ℕ) :
  total = females + males →
  total = 8 →
  females = 5 →
  males = 3 →
  chosen = 3 →
  (Nat.choose females 2 * Nat.choose males 1 : ℚ) / Nat.choose total chosen = 15 / 28 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_females_one_male_l1439_143981


namespace NUMINAMATH_CALUDE_original_average_weight_l1439_143959

theorem original_average_weight 
  (original_count : ℕ) 
  (new_boy_weight : ℝ) 
  (average_increase : ℝ) : 
  original_count = 5 →
  new_boy_weight = 40 →
  average_increase = 1 →
  (original_count : ℝ) * ((original_count : ℝ) * average_increase + new_boy_weight) / 
    (original_count + 1) - new_boy_weight = 34 := by
  sorry

end NUMINAMATH_CALUDE_original_average_weight_l1439_143959


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1439_143924

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ 3 * x^2 - x + m = 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1439_143924


namespace NUMINAMATH_CALUDE_z_to_twelve_equals_one_l1439_143933

theorem z_to_twelve_equals_one :
  let z : ℂ := (Real.sqrt 3 - Complex.I) / 2
  z^12 = 1 := by sorry

end NUMINAMATH_CALUDE_z_to_twelve_equals_one_l1439_143933


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l1439_143947

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 * x + (15 * x + 17) ^ (1/3)) ^ (1/3)

-- State the theorem
theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, f x = 18 ∧ x = 387 := by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l1439_143947


namespace NUMINAMATH_CALUDE_tangent_at_one_two_tangent_through_one_one_l1439_143980

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- Theorem for the tangent line at (1, 2)
theorem tangent_at_one_two :
  ∃ (m b : ℝ), (∀ x y, y = m * x + b ↔ 2 * x - y = 0) ∧
  f 1 = 2 ∧ f' 1 = m := by sorry

-- Theorem for the tangent lines through (1, 1)
theorem tangent_through_one_one :
  ∃ (x₀ : ℝ), (x₀ = 0 ∨ x₀ = 2) ∧
  (∀ x y, y = 1 ↔ x₀ = 0 ∧ y = f x₀ + f' x₀ * (x - x₀)) ∧
  (∀ x y, 4 * x - y - 3 = 0 ↔ x₀ = 2 ∧ y = f x₀ + f' x₀ * (x - x₀)) ∧
  f x₀ + f' x₀ * (1 - x₀) = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_at_one_two_tangent_through_one_one_l1439_143980


namespace NUMINAMATH_CALUDE_pyramid_height_equal_volume_l1439_143977

theorem pyramid_height_equal_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 6 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 6.48 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equal_volume_l1439_143977


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_div_4_l1439_143997

def binary_number : ℕ := 3789 -- 111001001101₂ in decimal

theorem remainder_of_binary_number_div_4 :
  binary_number % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_div_4_l1439_143997


namespace NUMINAMATH_CALUDE_count_rectangles_l1439_143958

/-- The number of checkered rectangles containing exactly one gray cell -/
def num_rectangles (total_gray_cells : ℕ) (blue_cells : ℕ) (red_cells : ℕ) 
  (rectangles_per_blue : ℕ) (rectangles_per_red : ℕ) : ℕ :=
  blue_cells * rectangles_per_blue + red_cells * rectangles_per_red

/-- Theorem stating the number of checkered rectangles containing exactly one gray cell -/
theorem count_rectangles : 
  num_rectangles 40 36 4 4 8 = 176 := by
  sorry

end NUMINAMATH_CALUDE_count_rectangles_l1439_143958


namespace NUMINAMATH_CALUDE_enlarged_lawn_area_l1439_143907

theorem enlarged_lawn_area (initial_width : ℝ) (initial_area : ℝ) (new_width : ℝ) :
  initial_width = 8 →
  initial_area = 640 →
  new_width = 16 →
  let length : ℝ := initial_area / initial_width
  let new_area : ℝ := length * new_width
  new_area = 1280 := by
  sorry

end NUMINAMATH_CALUDE_enlarged_lawn_area_l1439_143907


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1439_143932

theorem modular_congruence_solution (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (98 * n) % 103 = 33 % 103 → n % 103 = 87 % 103 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1439_143932


namespace NUMINAMATH_CALUDE_pencil_length_after_sharpening_l1439_143966

/-- Calculates the final length of a pencil after sharpening on four consecutive days. -/
def final_pencil_length (initial_length : ℕ) (day1 day2 day3 day4 : ℕ) : ℕ :=
  initial_length - (day1 + day2 + day3 + day4)

/-- Theorem stating that given specific initial length and sharpening amounts, the final pencil length is 36 inches. -/
theorem pencil_length_after_sharpening :
  final_pencil_length 50 2 3 4 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_after_sharpening_l1439_143966


namespace NUMINAMATH_CALUDE_mike_total_games_l1439_143923

/-- The number of video games Mike had initially -/
def total_games : ℕ := sorry

/-- The number of non-working games -/
def non_working_games : ℕ := 8

/-- The price of each working game in dollars -/
def price_per_game : ℕ := 7

/-- The total amount earned from selling working games in dollars -/
def total_earned : ℕ := 56

/-- Theorem stating that the total number of video games Mike had initially is 16 -/
theorem mike_total_games : total_games = 16 := by sorry

end NUMINAMATH_CALUDE_mike_total_games_l1439_143923


namespace NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_point_l1439_143927

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  per_sheet : ℚ
  sitting_fee : ℚ

/-- Calculates the total cost for a given number of sheets -/
def total_cost (company : PhotoCompany) (sheets : ℚ) : ℚ :=
  company.per_sheet * sheets + company.sitting_fee

/-- John's Photo World pricing -/
def johns_photo_world : PhotoCompany :=
  { per_sheet := 2.75, sitting_fee := 125 }

/-- Sam's Picture Emporium pricing -/
def sams_picture_emporium : PhotoCompany :=
  { per_sheet := 1.50, sitting_fee := 140 }

/-- Theorem stating that the two companies charge the same for 12 sheets -/
theorem same_price_at_12_sheets :
  total_cost johns_photo_world 12 = total_cost sams_picture_emporium 12 :=
by sorry

/-- Theorem stating that 12 is the unique number of sheets where prices are equal -/
theorem unique_equal_price_point (sheets : ℚ) :
  total_cost johns_photo_world sheets = total_cost sams_picture_emporium sheets ↔ sheets = 12 :=
by sorry

end NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_point_l1439_143927


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_product_l1439_143943

theorem consecutive_even_numbers_product : 
  ∃! (a b c : ℕ), 
    (b = a + 2 ∧ c = b + 2) ∧ 
    (a % 2 = 0) ∧
    (800000 ≤ a * b * c) ∧ 
    (a * b * c < 900000) ∧
    (a * b * c % 10 = 2) ∧
    (a = 94 ∧ b = 96 ∧ c = 98) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_product_l1439_143943


namespace NUMINAMATH_CALUDE_oldest_youngest_sum_l1439_143950

def age_problem (a b c d : ℕ) : Prop :=
  a + b + c + d = 100 ∧
  a = 32 ∧
  a + b = 3 * (c + d) ∧
  c = d + 3

theorem oldest_youngest_sum (a b c d : ℕ) 
  (h : age_problem a b c d) : 
  max a (max b (max c d)) + min a (min b (min c d)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_oldest_youngest_sum_l1439_143950


namespace NUMINAMATH_CALUDE_response_rate_percentage_l1439_143902

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : 
  responses_needed = 900 → questionnaires_mailed = 1500 → 
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l1439_143902


namespace NUMINAMATH_CALUDE_divisors_in_range_l1439_143925

theorem divisors_in_range (m a b : ℕ) (hm : 0 < m) (ha : m^2 < a) (hb : m^2 < b) 
  (ha_upper : a < m^2 + m) (hb_upper : b < m^2 + m) (hab : a ≠ b) : 
  ∀ d : ℕ, m^2 < d → d < m^2 + m → d ∣ (a * b) → d = a ∨ d = b := by
sorry

end NUMINAMATH_CALUDE_divisors_in_range_l1439_143925


namespace NUMINAMATH_CALUDE_expression_zero_l1439_143999

theorem expression_zero (a b c : ℝ) (h : c = b + 2) :
  b = -2 ∧ c = 0 → (a - (b + c)) - ((a + c) - b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_zero_l1439_143999


namespace NUMINAMATH_CALUDE_dereks_score_l1439_143900

/-- Given a basketball team's score and the performance of other players, 
    calculate Derek's score. -/
theorem dereks_score 
  (total_score : ℕ) 
  (other_players : ℕ) 
  (avg_score_others : ℕ) 
  (h1 : total_score = 65) 
  (h2 : other_players = 8) 
  (h3 : avg_score_others = 5) : 
  total_score - (other_players * avg_score_others) = 25 := by
sorry

end NUMINAMATH_CALUDE_dereks_score_l1439_143900


namespace NUMINAMATH_CALUDE_oreo_cheesecake_solution_l1439_143919

def oreo_cheesecake_problem (graham_boxes_bought : ℕ) (oreo_packets_bought : ℕ) 
  (graham_boxes_per_cake : ℕ) (graham_boxes_leftover : ℕ) : ℕ :=
  let cakes_made := (graham_boxes_bought - graham_boxes_leftover) / graham_boxes_per_cake
  oreo_packets_bought / cakes_made

theorem oreo_cheesecake_solution :
  oreo_cheesecake_problem 14 15 2 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_oreo_cheesecake_solution_l1439_143919


namespace NUMINAMATH_CALUDE_triangular_to_square_ratio_l1439_143978

/-- A polyhedron with only triangular and square faces -/
structure Polyhedron :=
  (triangular_faces : ℕ)
  (square_faces : ℕ)

/-- Property that no two faces of the same type share an edge -/
def no_same_type_edge_sharing (p : Polyhedron) : Prop :=
  ∀ (edge : ℕ), (∃! square_face : ℕ, square_face ≤ p.square_faces) ∧
                (∃! triangular_face : ℕ, triangular_face ≤ p.triangular_faces)

theorem triangular_to_square_ratio (p : Polyhedron) 
  (h : no_same_type_edge_sharing p) (h_pos : p.square_faces > 0) : 
  (p.triangular_faces : ℚ) / p.square_faces = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangular_to_square_ratio_l1439_143978


namespace NUMINAMATH_CALUDE_three_faces_colored_count_l1439_143979

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  totalSmallCubes : ℕ
  smallCubesPerEdge : ℕ

/-- Calculates the number of small cubes with exactly three faces colored -/
def threeFacesColored (c : CutCube) : ℕ := 8

/-- Theorem: In a cube cut into 216 equal smaller cubes, 
    the number of small cubes with exactly three faces colored is 8 -/
theorem three_faces_colored_count :
  ∀ (c : CutCube), c.totalSmallCubes = 216 → threeFacesColored c = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_faces_colored_count_l1439_143979


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l1439_143930

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1_bounds : 1 < a 1 ∧ a 1 < 3)
  (h_a3 : a 3 = 4)
  (b : ℕ → ℝ)
  (h_b_def : ∀ n, b n = 2^(a n)) :
  (∃ r, ∀ n, b (n + 1) = r * b n) ∧
  (b 1 < b 2) ∧
  (b 2 > 4) ∧
  (b 2 * b 4 = 256) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l1439_143930


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1439_143939

theorem complex_equation_solution :
  ∀ (x y : ℝ), (1 + x * Complex.I) * (1 - 2 * Complex.I) = y → x = 2 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1439_143939


namespace NUMINAMATH_CALUDE_mangoes_quantity_l1439_143957

/-- The quantity of mangoes purchased by Harkamal -/
def mangoes_kg : ℕ := sorry

/-- The price of grapes per kg -/
def grapes_price : ℕ := 70

/-- The price of mangoes per kg -/
def mangoes_price : ℕ := 45

/-- The quantity of grapes purchased in kg -/
def grapes_kg : ℕ := 8

/-- The total amount paid -/
def total_paid : ℕ := 965

theorem mangoes_quantity :
  grapes_kg * grapes_price + mangoes_kg * mangoes_price = total_paid ∧
  mangoes_kg = 9 := by sorry

end NUMINAMATH_CALUDE_mangoes_quantity_l1439_143957


namespace NUMINAMATH_CALUDE_shaded_area_is_zero_l1439_143996

/-- Rectangle JKLM with given dimensions and points -/
structure Rectangle where
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  C : ℝ × ℝ
  B : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The conditions of the rectangle as given in the problem -/
def rectangle_conditions (r : Rectangle) : Prop :=
  r.J = (0, 0) ∧
  r.K = (4, 0) ∧
  r.L = (4, 5) ∧
  r.M = (0, 5) ∧
  r.C = (1.5, 5) ∧
  r.B = (4, 4) ∧
  r.E = r.J ∧
  r.F = r.M

/-- The area of the shaded region formed by the intersection of CF and BE -/
def shaded_area (r : Rectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 0 -/
theorem shaded_area_is_zero (r : Rectangle) (h : rectangle_conditions r) : 
  shaded_area r = 0 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_zero_l1439_143996


namespace NUMINAMATH_CALUDE_draw_with_replacement_l1439_143987

/-- The number of items to choose from -/
def n : ℕ := 15

/-- The number of times we draw -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from n items -/
def num_lists (n k : ℕ) : ℕ := n^k

theorem draw_with_replacement :
  num_lists n k = 50625 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_replacement_l1439_143987


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1439_143952

def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_value :
  ∃ (x : ℝ), x > 0 ∧ dot_product (vector_a x) (vector_b x) = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1439_143952


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1439_143905

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3*x + y = 5) 
  (h2 : x + 3*y = 6) : 
  10*x^2 + 13*x*y + 10*y^2 = 97 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1439_143905


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1439_143998

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 3 / (x + 1) ≥ 2 * Real.sqrt 3 - 1 :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, x + 3 / (x + 1) = 2 * Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1439_143998


namespace NUMINAMATH_CALUDE_expression_value_l1439_143964

theorem expression_value
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023*c*d - (a + b)/20 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1439_143964


namespace NUMINAMATH_CALUDE_area_of_absolute_value_graph_l1439_143973

/-- The area enclosed by the graph of |x| + |3y| = 9 is 54 square units -/
theorem area_of_absolute_value_graph : 
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ |x| + |3 * y|
  ∃ S : Set (ℝ × ℝ), S = {p : ℝ × ℝ | f p = 9} ∧ MeasureTheory.volume S = 54 := by
  sorry

end NUMINAMATH_CALUDE_area_of_absolute_value_graph_l1439_143973


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l1439_143965

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem gcd_consecutive_b_terms (n : ℕ) : Nat.gcd (b n) (b (n + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l1439_143965


namespace NUMINAMATH_CALUDE_inequality_proof_l1439_143911

theorem inequality_proof (a b x : ℝ) (h : a ≠ b) :
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1439_143911


namespace NUMINAMATH_CALUDE_samantha_more_heads_prob_l1439_143963

def fair_coin_prob : ℚ := 1/2
def biased_coin_prob1 : ℚ := 3/5
def biased_coin_prob2 : ℚ := 2/3

def coin_set := (fair_coin_prob, biased_coin_prob1, biased_coin_prob2)

def prob_more_heads (coins : ℚ × ℚ × ℚ) : ℚ :=
  sorry

theorem samantha_more_heads_prob :
  prob_more_heads coin_set = 436/225 :=
sorry

end NUMINAMATH_CALUDE_samantha_more_heads_prob_l1439_143963


namespace NUMINAMATH_CALUDE_complex_point_location_l1439_143971

theorem complex_point_location (z : ℂ) : 
  (2 + Complex.I) * z = Complex.abs (1 - 2 * Complex.I) →
  Real.sign (z.re) > 0 ∧ Real.sign (z.im) < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_point_location_l1439_143971


namespace NUMINAMATH_CALUDE_inequality_proof_l1439_143920

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (3*x^2 - x) / (1 + x^2) + (3*y^2 - y) / (1 + y^2) + (3*z^2 - z) / (1 + z^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1439_143920


namespace NUMINAMATH_CALUDE_remainder_problem_l1439_143994

theorem remainder_problem (x : ℤ) (h : x % 82 = 5) : (x + 17) % 41 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1439_143994


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1439_143990

theorem decimal_point_problem : ∃! (x : ℝ), x > 0 ∧ 10000 * x = 9 / x ∧ x = 0.03 := by sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1439_143990


namespace NUMINAMATH_CALUDE_airplane_distance_difference_l1439_143935

/-- The difference in distance traveled by an airplane flying without wind for 4 hours
    and against a 20 km/h wind for 3 hours, given that the airplane's windless speed is a km/h. -/
theorem airplane_distance_difference (a : ℝ) : 
  4 * a - (3 * (a - 20)) = a + 60 := by
  sorry

end NUMINAMATH_CALUDE_airplane_distance_difference_l1439_143935


namespace NUMINAMATH_CALUDE_unique_natural_solution_l1439_143945

theorem unique_natural_solution :
  ∃! (x y : ℕ), 3 * x + 7 * y = 23 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_natural_solution_l1439_143945


namespace NUMINAMATH_CALUDE_prob_second_white_given_first_white_l1439_143949

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the state of the bag -/
structure BagState where
  white : ℕ
  black : ℕ

/-- The initial state of the bag -/
def initial_bag : BagState := ⟨3, 2⟩

/-- The probability of drawing a white ball given the bag state -/
def prob_white (bag : BagState) : ℚ :=
  bag.white / (bag.white + bag.black)

/-- The probability of drawing a specific color given the bag state -/
def prob_draw (bag : BagState) (color : BallColor) : ℚ :=
  match color with
  | BallColor.White => prob_white bag
  | BallColor.Black => 1 - prob_white bag

/-- The new bag state after drawing a ball of a given color -/
def draw_ball (bag : BagState) (color : BallColor) : BagState :=
  match color with
  | BallColor.White => ⟨bag.white - 1, bag.black⟩
  | BallColor.Black => ⟨bag.white, bag.black - 1⟩

theorem prob_second_white_given_first_white :
  prob_draw (draw_ball initial_bag BallColor.White) BallColor.White = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_white_given_first_white_l1439_143949


namespace NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l1439_143995

theorem sum_of_square_roots_lower_bound (a b c d e : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l1439_143995


namespace NUMINAMATH_CALUDE_problem_statement_l1439_143962

theorem problem_statement (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1439_143962


namespace NUMINAMATH_CALUDE_power_of_three_squared_l1439_143921

theorem power_of_three_squared : 3^2 = 9 := by sorry

end NUMINAMATH_CALUDE_power_of_three_squared_l1439_143921


namespace NUMINAMATH_CALUDE_no_divisor_3_mod_4_and_unique_solution_l1439_143912

theorem no_divisor_3_mod_4_and_unique_solution : 
  (∀ x : ℤ, ∀ d : ℤ, d ∣ (x^2 + 1) → d % 4 ≠ 3) ∧ 
  (∀ x y : ℕ, x^2 - y^3 = 7 ↔ x = 23 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_no_divisor_3_mod_4_and_unique_solution_l1439_143912


namespace NUMINAMATH_CALUDE_bees_flew_in_l1439_143928

theorem bees_flew_in (initial_bees final_bees : ℕ) (h : initial_bees ≤ final_bees) :
  final_bees - initial_bees = final_bees - initial_bees :=
by sorry

end NUMINAMATH_CALUDE_bees_flew_in_l1439_143928
