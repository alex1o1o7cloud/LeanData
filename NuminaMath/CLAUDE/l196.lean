import Mathlib

namespace sufficient_but_not_necessary_l196_19637

theorem sufficient_but_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by
  sorry

end sufficient_but_not_necessary_l196_19637


namespace rectangular_box_width_is_correct_boxes_fit_in_wooden_box_l196_19627

/-- The width of rectangular boxes that fit in a wooden box -/
def rectangular_box_width : ℝ :=
  let wooden_box_length : ℝ := 800  -- 8 m in cm
  let wooden_box_width : ℝ := 700   -- 7 m in cm
  let wooden_box_height : ℝ := 600  -- 6 m in cm
  let box_length : ℝ := 8
  let box_height : ℝ := 6
  let max_boxes : ℕ := 1000000
  7  -- Width of rectangular boxes in cm

theorem rectangular_box_width_is_correct : rectangular_box_width = 7 := by
  sorry

/-- The volume of the wooden box in cubic centimeters -/
def wooden_box_volume : ℝ :=
  800 * 700 * 600

/-- The volume of a single rectangular box in cubic centimeters -/
def single_box_volume (w : ℝ) : ℝ :=
  8 * w * 6

/-- The total volume of all rectangular boxes -/
def total_boxes_volume (w : ℝ) : ℝ :=
  1000000 * single_box_volume w

theorem boxes_fit_in_wooden_box :
  total_boxes_volume rectangular_box_width ≤ wooden_box_volume := by
  sorry

end rectangular_box_width_is_correct_boxes_fit_in_wooden_box_l196_19627


namespace walking_biking_time_difference_l196_19689

/-- Proves that the difference between walking and biking time is 4 minutes -/
theorem walking_biking_time_difference :
  let blocks : ℕ := 6
  let walk_time_per_block : ℚ := 1
  let bike_time_per_block : ℚ := 20 / 60
  (blocks * walk_time_per_block) - (blocks * bike_time_per_block) = 4 := by
  sorry

end walking_biking_time_difference_l196_19689


namespace unique_function_satisfying_equations_l196_19639

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_function_satisfying_equations (f : RealFunction) :
  (∀ x : ℝ, f (x + 1) = 1 + f x) ∧
  (∀ x : ℝ, f (x^4 - x^2) = f x^4 - f x^2) →
  ∀ x : ℝ, f x = x :=
by
  sorry

end unique_function_satisfying_equations_l196_19639


namespace average_temperature_problem_l196_19656

/-- The average temperature problem -/
theorem average_temperature_problem 
  (temp_mon : ℝ) 
  (temp_tue : ℝ) 
  (temp_wed : ℝ) 
  (temp_thu : ℝ) 
  (temp_fri : ℝ) 
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : temp_mon = 42)
  (h3 : temp_fri = 10) :
  (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 40 := by
  sorry


end average_temperature_problem_l196_19656


namespace odot_examples_l196_19666

def odot (a b : ℚ) : ℚ := a * (a + b) - 1

theorem odot_examples :
  (odot 3 (-2) = 2) ∧ (odot (-2) (odot 3 5) = -43) := by
  sorry

end odot_examples_l196_19666


namespace welders_left_l196_19687

/-- Proves that 12 welders left the project given the initial conditions and remaining time. -/
theorem welders_left (initial_welders : ℕ) (initial_days : ℝ) (remaining_days : ℝ) : 
  initial_welders = 36 →
  initial_days = 3 →
  remaining_days = 3.0000000000000004 →
  (initial_welders - (initial_welders - remaining_days * initial_welders / (initial_days + remaining_days - 1))) = 12 := by
sorry

end welders_left_l196_19687


namespace cube_less_than_triple_l196_19623

theorem cube_less_than_triple : ∃! x : ℤ, x^3 < 3*x :=
by
  -- Proof goes here
  sorry

end cube_less_than_triple_l196_19623


namespace initial_pills_count_l196_19665

/-- The number of pills Tony takes in the first two days -/
def pills_first_two_days : ℕ := 2 * 3 * 2

/-- The number of pills Tony takes in the next three days -/
def pills_next_three_days : ℕ := 1 * 3 * 3

/-- The number of pills Tony takes on the sixth day -/
def pills_sixth_day : ℕ := 2

/-- The number of pills left in the bottle after Tony's treatment -/
def pills_left : ℕ := 27

/-- The total number of pills Tony took during his treatment -/
def total_pills_taken : ℕ := pills_first_two_days + pills_next_three_days + pills_sixth_day

/-- Theorem: The initial number of pills in the bottle is 50 -/
theorem initial_pills_count : total_pills_taken + pills_left = 50 := by
  sorry

end initial_pills_count_l196_19665


namespace negation_of_implication_l196_19659

theorem negation_of_implication (a b c : ℝ) : 
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
sorry

end negation_of_implication_l196_19659


namespace equation_solution_l196_19643

theorem equation_solution (a : ℝ) (x : ℝ) : 
  a = 3 → (5 * a - x = 13 ↔ x = 2) := by sorry

end equation_solution_l196_19643


namespace right_triangle_sides_l196_19690

/-- A right-angled triangle with sides in arithmetic progression and area 216 cm² -/
structure RightTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The sides form an arithmetic progression
  arith_prog : ∃ (d : ℝ), b = a + d ∧ c = b + d
  -- The triangle is right-angled (Pythagorean theorem)
  right_angle : a^2 + b^2 = c^2
  -- The area of the triangle is 216
  area : a * b / 2 = 216

theorem right_triangle_sides (t : RightTriangle) : t.a = 18 ∧ t.b = 24 ∧ t.c = 30 := by
  sorry

end right_triangle_sides_l196_19690


namespace simplify_and_evaluate_expression_l196_19650

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 2
  (1 + x) / (1 - x) / (x - 2 * x / (1 - x)) = - (Real.sqrt 2 + 2) / 2 := by
  sorry

end simplify_and_evaluate_expression_l196_19650


namespace candy_store_spend_l196_19684

def weekly_allowance : ℚ := 3/2

def arcade_spend (allowance : ℚ) : ℚ := (3/5) * allowance

def toy_store_spend (remaining : ℚ) : ℚ := (1/3) * remaining

theorem candy_store_spend :
  let remaining_after_arcade := weekly_allowance - arcade_spend weekly_allowance
  let remaining_after_toy := remaining_after_arcade - toy_store_spend remaining_after_arcade
  remaining_after_toy = 2/5 := by sorry

end candy_store_spend_l196_19684


namespace derivative_of_exp_neg_x_l196_19688

theorem derivative_of_exp_neg_x (x : ℝ) : deriv (fun x => Real.exp (-x)) x = -Real.exp (-x) := by
  sorry

end derivative_of_exp_neg_x_l196_19688


namespace lcm_15_18_l196_19632

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end lcm_15_18_l196_19632


namespace sum_of_H_and_J_l196_19605

theorem sum_of_H_and_J : ∃ (H J K L : ℕ),
  H ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  J ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  K ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  L ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  H ≠ J ∧ H ≠ K ∧ H ≠ L ∧ J ≠ K ∧ J ≠ L ∧ K ≠ L ∧
  (H : ℚ) / J - (K : ℚ) / L = 5 / 6 →
  H + J = 7 :=
sorry

end sum_of_H_and_J_l196_19605


namespace custom_mult_comm_custom_mult_comm_complex_l196_19696

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating the commutativity of the custom multiplication -/
theorem custom_mult_comm (a b : ℝ) : custom_mult a b = custom_mult b a := by
  sorry

/-- Theorem stating the commutativity of the custom multiplication with a complex expression -/
theorem custom_mult_comm_complex (a b c : ℝ) : custom_mult a (b - c) = custom_mult (b - c) a := by
  sorry

end custom_mult_comm_custom_mult_comm_complex_l196_19696


namespace R_properties_l196_19680

noncomputable def R (x : ℝ) : ℝ :=
  x^2 + 1/x^2 + (1-x)^2 + 1/(1-x)^2 + x^2/(1-x)^2 + (x-1)^2/x^2

theorem R_properties :
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = R (1/x)) ∧
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = R (1-x)) ∧
  ¬ (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = c) :=
by sorry

end R_properties_l196_19680


namespace hex_lattice_equilateral_triangles_l196_19610

/-- Represents a point in a 2D hexagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the hexagonal lattice -/
def HexagonalLattice : Type := List LatticePoint

/-- Calculates the distance between two points -/
def distance (p1 p2 : LatticePoint) : ℝ := sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : LatticePoint) : Bool := sorry

/-- Counts the number of equilateral triangles in the lattice -/
def countEquilateralTriangles (lattice : HexagonalLattice) : Nat := sorry

/-- The hexagonal lattice with 7 points -/
def hexLattice : HexagonalLattice := sorry

theorem hex_lattice_equilateral_triangles :
  countEquilateralTriangles hexLattice = 6 := by sorry

end hex_lattice_equilateral_triangles_l196_19610


namespace johns_annual_profit_l196_19621

/-- Calculates the annual profit from subletting an apartment --/
def annual_profit (num_subletters : ℕ) (subletter_rent : ℕ) (apartment_rent : ℕ) : ℕ :=
  (num_subletters * subletter_rent * 12) - (apartment_rent * 12)

/-- Theorem: John's annual profit from subletting his apartment is $3600 --/
theorem johns_annual_profit :
  annual_profit 3 400 900 = 3600 := by
  sorry

end johns_annual_profit_l196_19621


namespace similar_triangle_longest_side_l196_19641

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (h_triangle : a = 5 ∧ b = 12 ∧ c = 13) 
  (h_perimeter : ∃ k : ℝ, k > 0 ∧ k * (a + b + c) = 150) :
  ∃ s : ℝ, s = 65 ∧ s = max (k * a) (max (k * b) (k * c)) :=
sorry

end similar_triangle_longest_side_l196_19641


namespace relay_race_time_reduction_l196_19693

theorem relay_race_time_reduction (T : ℝ) (T1 T2 T3 T4 T5 : ℝ) :
  T > 0 ∧ T1 > 0 ∧ T2 > 0 ∧ T3 > 0 ∧ T4 > 0 ∧ T5 > 0 ∧
  T = T1 + T2 + T3 + T4 + T5 ∧
  T1/2 + T2 + T3 + T4 + T5 = 0.95 * T ∧
  T1 + T2/2 + T3 + T4 + T5 = 0.9 * T ∧
  T1 + T2 + T3/2 + T4 + T5 = 0.88 * T ∧
  T1 + T2 + T3 + T4/2 + T5 = 0.85 * T →
  T1 + T2 + T3 + T4 + T5/2 = 0.92 * T := by
sorry

end relay_race_time_reduction_l196_19693


namespace min_value_expression_l196_19638

/-- The minimum value of (s+5-3|cos t|)^2 + (s-2|sin t|)^2 is 2, where s and t are real numbers. -/
theorem min_value_expression : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (s t : ℝ), (s + 5 - 3 * |Real.cos t|)^2 + (s - 2 * |Real.sin t|)^2 ≥ m :=
by sorry

end min_value_expression_l196_19638


namespace polynomial_modulus_bound_l196_19655

theorem polynomial_modulus_bound (a b c d : ℂ) 
  (ha : Complex.abs a = 1) (hb : Complex.abs b = 1) 
  (hc : Complex.abs c = 1) (hd : Complex.abs d = 1) : 
  ∃ z : ℂ, Complex.abs z = 1 ∧ 
    Complex.abs (a * z^3 + b * z^2 + c * z + d) ≥ Real.sqrt 6 := by
  sorry

end polynomial_modulus_bound_l196_19655


namespace equivalence_of_functional_equations_l196_19634

theorem equivalence_of_functional_equations (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x * y + x + y) = f (x * y) + f x + f y) := by
  sorry

end equivalence_of_functional_equations_l196_19634


namespace line_l_equation_line_l_prime_equation_l196_19661

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (-1, 2)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the symmetry point
def sym_point : ℝ × ℝ := (1, -1)

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    (l₁ x y ∧ l₂ x y → (x, y) = M) → 
    (∀ (a b : ℝ), perp_line a b → (y - M.2) = m * (x - M.1)) → 
    (x - 2 * y + 5 = 0) :=
sorry

-- Theorem for the equation of line l′
theorem line_l_prime_equation :
  ∀ (x y : ℝ),
    (∃ (x' y' : ℝ), l₁ x' y' ∧ 
      x' = 2 * sym_point.1 - x ∧ 
      y' = 2 * sym_point.2 - y) →
    (3 * x + 4 * y + 7 = 0) :=
sorry

end line_l_equation_line_l_prime_equation_l196_19661


namespace suraj_innings_l196_19629

/-- The number of innings Suraj played before the last one -/
def n : ℕ := sorry

/-- Suraj's average before the last innings -/
def A : ℚ := sorry

/-- Suraj's new average after the last innings -/
def new_average : ℚ := 28

/-- The increase in Suraj's average after the last innings -/
def average_increase : ℚ := 8

/-- The runs Suraj scored in the last innings -/
def last_innings_runs : ℕ := 140

theorem suraj_innings : 
  (n : ℚ) * A + last_innings_runs = (n + 1) * new_average ∧
  new_average = A + average_increase ∧
  n = 14 := by sorry

end suraj_innings_l196_19629


namespace total_good_vegetables_l196_19682

def carrots_day1 : ℕ := 23
def carrots_day2 : ℕ := 47
def rotten_carrots_day1 : ℕ := 10
def rotten_carrots_day2 : ℕ := 15

def tomatoes_day1 : ℕ := 34
def tomatoes_day2 : ℕ := 50
def rotten_tomatoes_day1 : ℕ := 5
def rotten_tomatoes_day2 : ℕ := 7

def cucumbers_day1 : ℕ := 42
def cucumbers_day2 : ℕ := 38
def rotten_cucumbers_day1 : ℕ := 7
def rotten_cucumbers_day2 : ℕ := 12

theorem total_good_vegetables :
  (carrots_day1 - rotten_carrots_day1) + (carrots_day2 - rotten_carrots_day2) +
  (tomatoes_day1 - rotten_tomatoes_day1) + (tomatoes_day2 - rotten_tomatoes_day2) +
  (cucumbers_day1 - rotten_cucumbers_day1) + (cucumbers_day2 - rotten_cucumbers_day2) = 178 :=
by sorry

end total_good_vegetables_l196_19682


namespace three_cubes_volume_l196_19668

theorem three_cubes_volume (s₁ s₂ : ℝ) (h₁ : s₁ > 0) (h₂ : s₂ > 0) : 
  6 * (s₁ + s₂)^2 = 864 → 2 * s₁^3 + s₂^3 = 1728 := by
  sorry

end three_cubes_volume_l196_19668


namespace binomial_7_4_l196_19672

theorem binomial_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_l196_19672


namespace mo_hot_chocolate_consumption_l196_19663

/-- Represents the drinking habits of Mo --/
structure MoDrinkingHabits where
  rainyDayHotChocolate : ℚ
  nonRainyDayTea : ℕ
  totalCups : ℕ
  teaMoreThanHotChocolate : ℕ
  rainyDays : ℕ

/-- Theorem stating Mo's hot chocolate consumption on rainy mornings --/
theorem mo_hot_chocolate_consumption (mo : MoDrinkingHabits)
  (h1 : mo.nonRainyDayTea = 3)
  (h2 : mo.totalCups = 20)
  (h3 : mo.teaMoreThanHotChocolate = 10)
  (h4 : mo.rainyDays = 2)
  (h5 : (7 - mo.rainyDays) * mo.nonRainyDayTea + mo.rainyDays * mo.rainyDayHotChocolate = mo.totalCups)
  (h6 : (7 - mo.rainyDays) * mo.nonRainyDayTea = mo.rainyDays * mo.rainyDayHotChocolate + mo.teaMoreThanHotChocolate) :
  mo.rainyDayHotChocolate = 5/2 := by
  sorry

end mo_hot_chocolate_consumption_l196_19663


namespace max_daily_revenue_l196_19602

-- Define the sales price function
def P (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 70
  else 0

-- Define the daily sales volume function
def Q (t : ℕ) : ℝ := -t + 40

-- Define the daily sales revenue function
def R (t : ℕ) : ℝ := P t * Q t

-- Theorem statement
theorem max_daily_revenue :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ R t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → R t ≤ 1125) ∧
  (R 25 = 1125) :=
sorry

end max_daily_revenue_l196_19602


namespace min_root_of_negated_quadratic_l196_19658

theorem min_root_of_negated_quadratic (p : ℝ) (r₁ r₂ : ℝ) :
  (∀ x, (x - 19) * (x - 83) = p ↔ x = r₁ ∨ x = r₂) →
  (∃ x, (x - r₁) * (x - r₂) = -p) →
  (∀ x, (x - r₁) * (x - r₂) = -p → x ≥ -19) ∧
  (∃ x, (x - r₁) * (x - r₂) = -p ∧ x = -19) :=
by sorry

end min_root_of_negated_quadratic_l196_19658


namespace smallest_three_digit_palindrome_non_five_digit_palindrome_product_l196_19697

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function that checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem statement -/
theorem smallest_three_digit_palindrome_non_five_digit_palindrome_product :
  isThreeDigitPalindrome 131 ∧
  ¬(isFiveDigitPalindrome (131 * 103)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 131 → isFiveDigitPalindrome (n * 103) :=
sorry

end smallest_three_digit_palindrome_non_five_digit_palindrome_product_l196_19697


namespace polynomial_simplification_l196_19653

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) =
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 := by
  sorry

end polynomial_simplification_l196_19653


namespace midpoint_of_intersections_l196_19606

/-- The line equation y = x - 3 -/
def line_eq (x y : ℝ) : Prop := y = x - 3

/-- The parabola equation y^2 = 2x -/
def parabola_eq (x y : ℝ) : Prop := y^2 = 2*x

/-- A point (x, y) is on both the line and the parabola -/
def intersection_point (x y : ℝ) : Prop := line_eq x y ∧ parabola_eq x y

/-- There exist two distinct intersection points -/
axiom two_intersections : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  x₁ ≠ x₂ ∧ intersection_point x₁ y₁ ∧ intersection_point x₂ y₂

theorem midpoint_of_intersections : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    intersection_point x₁ y₁ ∧ 
    intersection_point x₂ y₂ ∧ 
    ((x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 1) :=
sorry

end midpoint_of_intersections_l196_19606


namespace max_third_term_is_16_l196_19642

/-- An arithmetic sequence of four positive integers -/
structure ArithmeticSequence :=
  (a : ℕ+) -- First term
  (d : ℕ+) -- Common difference
  (sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50)
  (third_term_even : Even (a + 2*d))

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value for the third term is 16 -/
theorem max_third_term_is_16 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 16 :=
sorry

end max_third_term_is_16_l196_19642


namespace circle_diameter_ratio_l196_19686

/-- Given two circles A and B, where A is inside B, this theorem proves the diameter of A
    given the diameter of B, the distance between centers, and the ratio of areas. -/
theorem circle_diameter_ratio (dB : ℝ) (d : ℝ) (r : ℝ) : 
  dB = 20 →  -- Diameter of circle B
  d = 4 →    -- Distance between centers
  r = 5 →    -- Ratio of shaded area to area of circle A
  ∃ (dA : ℝ), dA = 2 * Real.sqrt (50 / 3) ∧ 
    (π * (dA / 2)^2) * (1 + r) = π * (dB / 2)^2 ∧ 
    d ≤ (dB - dA) / 2 :=
by sorry

end circle_diameter_ratio_l196_19686


namespace second_quadrant_necessary_not_sufficient_for_obtuse_l196_19674

/-- An angle is in the second quadrant if it's between 90° and 180° exclusive. -/
def is_in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

/-- An angle is obtuse if it's between 90° and 180° exclusive. -/
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α, is_obtuse α → is_in_second_quadrant α) ∧
  (∃ α, is_in_second_quadrant α ∧ ¬is_obtuse α) :=
by sorry

end second_quadrant_necessary_not_sufficient_for_obtuse_l196_19674


namespace parallel_lines_m_value_l196_19671

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- Given two lines l₁ and l₂, prove that if they are parallel, then m = -2 -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, m * x + 2 * y - 3 = 0 ↔ 3 * x + (m - 1) * y + m - 6 = 0) →
  m = -2 :=
by sorry

end parallel_lines_m_value_l196_19671


namespace translated_line_y_intercept_l196_19633

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line horizontally and vertically -/
def translateLine (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    yIntercept := l.yIntercept - dy + l.slope * dx }

/-- The original line y = x -/
def originalLine : Line :=
  { slope := 1,
    yIntercept := 0 }

/-- The translated line -/
def translatedLine : Line :=
  translateLine originalLine 3 (-2)

theorem translated_line_y_intercept :
  translatedLine.yIntercept = -5 := by
  sorry

end translated_line_y_intercept_l196_19633


namespace max_triangle_side_length_l196_19657

theorem max_triangle_side_length :
  ∀ a b c : ℕ,
    a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different integer side lengths
    a + b + c = 30 →        -- Perimeter is 30 units
    a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
    a + b > c ∧ b + c > a ∧ a + c > b → -- Triangle inequality
    a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 -- Maximum side length is 14
  := by sorry

end max_triangle_side_length_l196_19657


namespace billy_already_ahead_l196_19667

def billy_miles : List ℝ := [2, 3, 0, 4, 1, 0]
def tiffany_miles : List ℝ := [1.5, 0, 2.5, 2.5, 3, 0]

theorem billy_already_ahead : 
  (billy_miles.sum > tiffany_miles.sum) ∧ 
  (billy_miles.length = tiffany_miles.length) := by
  sorry

end billy_already_ahead_l196_19667


namespace parallelogram_angles_l196_19631

/-- Represents the angles of a parallelogram -/
structure ParallelogramAngles where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

/-- Properties of a parallelogram with one angle 50° less than the other -/
def is_valid_parallelogram (p : ParallelogramAngles) : Prop :=
  p.angle1 = p.angle3 ∧
  p.angle2 = p.angle4 ∧
  p.angle1 + p.angle2 = 180 ∧
  p.angle2 = p.angle1 + 50

/-- Theorem: The angles of a parallelogram with one angle 50° less than the other are 65°, 115°, 65°, and 115° -/
theorem parallelogram_angles :
  ∃ (p : ParallelogramAngles), is_valid_parallelogram p ∧
    p.angle1 = 65 ∧ p.angle2 = 115 ∧ p.angle3 = 65 ∧ p.angle4 = 115 :=
by
  sorry

end parallelogram_angles_l196_19631


namespace expression_simplification_l196_19604

theorem expression_simplification (x y : ℝ) (n : Nat) (h1 : x > 0) (h2 : y > 0) (h3 : x ≠ y) (h4 : n = 2 ∨ n = 3 ∨ n = 4) :
  let r := (x^2 + y^2) / (2*x*y)
  (((Real.sqrt (r + 1) - Real.sqrt (r - 1))^n - (Real.sqrt (r + 1) + Real.sqrt (r - 1))^n) /
   ((Real.sqrt (r + 1) - Real.sqrt (r - 1))^n + (Real.sqrt (r + 1) + Real.sqrt (r - 1))^n)) =
  (y^n - x^n) / (y^n + x^n) := by
  sorry

end expression_simplification_l196_19604


namespace log_equation_solution_l196_19615

theorem log_equation_solution (a : ℝ) (h1 : a > 1) 
  (h2 : Real.log a / Real.log 5 + Real.log a / Real.log 3 = 
        (Real.log a / Real.log 5) * (Real.log a / Real.log 3)) : 
  a = 15 := by
  sorry

end log_equation_solution_l196_19615


namespace john_and_sarah_money_l196_19640

theorem john_and_sarah_money (john_money : ℚ) (sarah_money : ℚ)
  (h1 : john_money = 5 / 8)
  (h2 : sarah_money = 7 / 16) :
  john_money + sarah_money = 1.0625 := by
sorry

end john_and_sarah_money_l196_19640


namespace eggplant_seed_distribution_l196_19651

theorem eggplant_seed_distribution (total_seeds : ℕ) (num_pots : ℕ) (seeds_in_last_pot : ℕ) :
  total_seeds = 10 →
  num_pots = 4 →
  seeds_in_last_pot = 1 →
  ∃ (seeds_per_pot : ℕ),
    seeds_per_pot * (num_pots - 1) + seeds_in_last_pot = total_seeds ∧
    seeds_per_pot = 3 :=
by sorry

end eggplant_seed_distribution_l196_19651


namespace cubic_roots_sum_of_squares_reciprocals_l196_19618

theorem cubic_roots_sum_of_squares_reciprocals (p q r : ℂ) : 
  p^3 - 15*p^2 + 26*p + 3 = 0 →
  q^3 - 15*q^2 + 26*q + 3 = 0 →
  r^3 - 15*r^2 + 26*r + 3 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 766/9 := by
sorry

end cubic_roots_sum_of_squares_reciprocals_l196_19618


namespace expression_evaluation_l196_19677

theorem expression_evaluation :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 - 1/3) = -13 := by
  sorry

end expression_evaluation_l196_19677


namespace solution_set_equivalence_l196_19664

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the set {x ∈ ℝ | |f(x)| < 1} is equal to the open interval (0, 3). -/
theorem solution_set_equivalence (f : ℝ → ℝ) 
    (h_increasing : ∀ x y, x < y → f x < f y)
    (h_f_0 : f 0 = -1)
    (h_f_3 : f 3 = 1) :
    {x : ℝ | |f x| < 1} = Set.Ioo 0 3 := by
  sorry


end solution_set_equivalence_l196_19664


namespace triangle_area_specific_l196_19692

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

theorem triangle_area_specific : 
  ∀ (a b c : ℝ) (A B C : ℝ),
    b = 2 →
    c = 2 * Real.sqrt 2 →
    C = π / 4 →
    triangle_area a b c A B C = Real.sqrt 3 + 1 := by
  sorry

end triangle_area_specific_l196_19692


namespace incorrect_expression_l196_19644

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) : 
  y / (2 * x - y) ≠ 6 / 1 := by
  sorry

end incorrect_expression_l196_19644


namespace perfect_square_trinomial_l196_19648

/-- If x^2 + mx + 16 is a perfect square trinomial, then m = ±8 -/
theorem perfect_square_trinomial (m : ℝ) : 
  (∀ x, ∃ k, x^2 + m*x + 16 = k^2) → m = 8 ∨ m = -8 := by
  sorry

end perfect_square_trinomial_l196_19648


namespace digit_difference_in_base_d_l196_19611

/-- Given two digits X and Y in base d > 8, if XY + XX = 182 in base d, then X - Y = d - 8 in base d -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : Fin d) (h_d : d > 8) 
  (h_sum : d * X.val + Y.val + d * X.val + X.val = d^2 + 8*d + 2) : 
  X.val - Y.val = d - 8 := by
  sorry

end digit_difference_in_base_d_l196_19611


namespace square_flags_count_l196_19669

theorem square_flags_count (total_fabric : ℕ) (square_size wide_size tall_size : ℕ × ℕ)
  (wide_count tall_count : ℕ) (fabric_left : ℕ) :
  total_fabric = 1000 →
  square_size = (4, 4) →
  wide_size = (5, 3) →
  tall_size = (3, 5) →
  wide_count = 20 →
  tall_count = 10 →
  fabric_left = 294 →
  ∃ (square_count : ℕ),
    square_count = 16 ∧
    square_count * (square_size.1 * square_size.2) +
    wide_count * (wide_size.1 * wide_size.2) +
    tall_count * (tall_size.1 * tall_size.2) +
    fabric_left = total_fabric :=
by sorry

end square_flags_count_l196_19669


namespace four_digit_sum_problem_l196_19694

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def to_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

theorem four_digit_sum_problem (a b c d : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  to_number a b c d + to_number d c b a = 11990 →
  (a = 1 ∧ b = 9 ∧ c = 9 ∧ d = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 9 ∧ d = 1) :=
by sorry

end four_digit_sum_problem_l196_19694


namespace imaginary_unit_cube_l196_19636

theorem imaginary_unit_cube (i : ℂ) (hi : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end imaginary_unit_cube_l196_19636


namespace remainder_sum_l196_19670

theorem remainder_sum (c d : ℤ) 
  (hc : c % 100 = 78) 
  (hd : d % 150 = 123) : 
  (c + d) % 50 = 1 := by
sorry

end remainder_sum_l196_19670


namespace marys_max_take_home_pay_l196_19662

/-- Calculates Mary's take-home pay after taxes and insurance premium -/
def marys_take_home_pay (max_hours : ℕ) (regular_rate : ℚ) (overtime_rates : List ℚ) 
  (social_security_rate : ℚ) (medicare_rate : ℚ) (insurance_premium : ℚ) : ℚ :=
  sorry

/-- Theorem stating Mary's take-home pay for maximum hours worked -/
theorem marys_max_take_home_pay : 
  marys_take_home_pay 70 8 [1.25, 1.5, 1.75, 2] (8/100) (2/100) 50 = 706 := by
  sorry

end marys_max_take_home_pay_l196_19662


namespace equation_result_l196_19624

theorem equation_result : 300 * 2 + (12 + 4) * 1 / 8 = 602 := by
  sorry

end equation_result_l196_19624


namespace codes_lost_with_no_leading_zeros_l196_19660

/-- The number of digits in each code -/
def code_length : ℕ := 5

/-- The number of possible digits (0 to 9) -/
def digit_options : ℕ := 10

/-- The number of non-zero digits (1 to 9) -/
def non_zero_digits : ℕ := 9

/-- Calculates the total number of possible codes -/
def total_codes : ℕ := digit_options ^ code_length

/-- Calculates the number of codes without leading zeros -/
def codes_without_leading_zeros : ℕ := non_zero_digits * (digit_options ^ (code_length - 1))

/-- The theorem to be proved -/
theorem codes_lost_with_no_leading_zeros :
  total_codes - codes_without_leading_zeros = 10000 := by
  sorry


end codes_lost_with_no_leading_zeros_l196_19660


namespace two_digit_primes_with_rearranged_digits_and_square_difference_l196_19617

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def digits_rearranged (a b : ℕ) : Prop :=
  (a / 10 = b % 10) ∧ (a % 10 = b / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem two_digit_primes_with_rearranged_digits_and_square_difference :
  ∀ a b : ℕ,
    is_two_digit_prime a ∧
    is_two_digit_prime b ∧
    digits_rearranged a b ∧
    is_perfect_square (a - b) →
    (a = 73 ∧ b = 37) ∨ (a = 37 ∧ b = 73) :=
by sorry

end two_digit_primes_with_rearranged_digits_and_square_difference_l196_19617


namespace parabola_vertex_on_x_axis_l196_19698

/-- A parabola with equation y = x^2 - 10x + d + 4 has its vertex on the x-axis if and only if d = 21 -/
theorem parabola_vertex_on_x_axis (d : ℝ) : 
  (∃ x : ℝ, x^2 - 10*x + d + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 - 10*y + d + 4 ≥ x^2 - 10*x + d + 4) ↔ 
  d = 21 := by
sorry

end parabola_vertex_on_x_axis_l196_19698


namespace square_areas_equal_l196_19679

/-- Represents the configuration of squares and circles -/
structure SquareCircleConfig where
  circle_radius : ℝ
  num_small_squares : ℕ

/-- Calculates the area of the larger square -/
def larger_square_area (config : SquareCircleConfig) : ℝ :=
  4 * config.circle_radius ^ 2

/-- Calculates the total area of the smaller squares -/
def total_small_squares_area (config : SquareCircleConfig) : ℝ :=
  config.num_small_squares * (2 * config.circle_radius) ^ 2

/-- Theorem stating that the area of the larger square is equal to the total area of the smaller squares -/
theorem square_areas_equal (config : SquareCircleConfig) 
    (h1 : config.circle_radius = 3)
    (h2 : config.num_small_squares = 4) : 
  larger_square_area config = total_small_squares_area config ∧ 
  larger_square_area config = 144 := by
  sorry

#eval larger_square_area { circle_radius := 3, num_small_squares := 4 }
#eval total_small_squares_area { circle_radius := 3, num_small_squares := 4 }

end square_areas_equal_l196_19679


namespace tom_program_duration_l196_19603

/-- Represents the duration of a combined BS and Ph.D. program -/
structure ProgramDuration where
  bs : ℕ
  phd : ℕ

/-- Calculates the time taken to complete a program given the standard duration and a completion factor -/
def completionTime (d : ProgramDuration) (factor : ℚ) : ℚ :=
  factor * (d.bs + d.phd)

theorem tom_program_duration :
  let standard_duration : ProgramDuration := { bs := 3, phd := 5 }
  let completion_factor : ℚ := 3/4
  completionTime standard_duration completion_factor = 6 := by sorry

end tom_program_duration_l196_19603


namespace monotonic_f_implies_m_range_inequality_implies_a_range_l196_19607

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x + m * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a * x - 3

theorem monotonic_f_implies_m_range (m : ℝ) :
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) → m ≤ -1 := by sorry

theorem inequality_implies_a_range (a : ℝ) :
  (∀ x, x > 0 → 2 * (f 0 x) ≥ g a x) → a ≤ 4 := by sorry

end monotonic_f_implies_m_range_inequality_implies_a_range_l196_19607


namespace investment_equation_l196_19683

/-- Proves the equation for the investment problem -/
theorem investment_equation (x : ℝ) (h : x > 0) : (106960 / (x + 500)) - (50760 / x) = 20 := by
  sorry

end investment_equation_l196_19683


namespace equation_solution_set_l196_19600

theorem equation_solution_set : 
  {x : ℝ | x > 0 ∧ x^(Real.log x / Real.log 10) = x^4 / 1000} = {10, 1000} := by
  sorry

end equation_solution_set_l196_19600


namespace line_slope_l196_19676

/-- A line that returns to its original position after moving 4 units left and 1 unit up has a slope of -1/4 -/
theorem line_slope (l : ℝ → ℝ) (b : ℝ) (h : ∀ x, l x = l (x + 4) - 1) : 
  ∃ k, k = -1/4 ∧ ∀ x, l x = k * x + b := by
sorry

end line_slope_l196_19676


namespace blue_markers_count_l196_19673

theorem blue_markers_count (total : ℝ) (red : ℝ) (blue : ℝ) : 
  total = 64.0 → red = 41.0 → blue = total - red → blue = 23.0 := by
  sorry

end blue_markers_count_l196_19673


namespace parabola_coefficient_l196_19649

/-- Given a parabola y = ax^2 + bx + c with vertex (q, 2q) and y-intercept (0, -3q), where q ≠ 0, 
    the value of b is 10. -/
theorem parabola_coefficient (a b c q : ℝ) : q ≠ 0 →
  (∀ x y, y = a * x^2 + b * x + c ↔ 
    (x = q ∧ y = 2*q) ∨ (x = 0 ∧ y = -3*q)) →
  b = 10 := by sorry

end parabola_coefficient_l196_19649


namespace problem_solution_l196_19601

theorem problem_solution (x y n : ℝ) : 
  x = 3 → y = 1 → n = x - y^(x-y) → n = 2 := by sorry

end problem_solution_l196_19601


namespace log_equality_implies_ratio_one_l196_19681

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 18 ∧ 
       Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) : 
  p / q = 1 := by
  sorry

end log_equality_implies_ratio_one_l196_19681


namespace calculation_proof_l196_19635

theorem calculation_proof : (-1) * (-4) + 3^2 / (7 - 4) = 7 := by
  sorry

end calculation_proof_l196_19635


namespace prob_shortest_diagonal_21_sided_l196_19620

/-- The number of sides in the regular polygon -/
def n : ℕ := 21

/-- The number of shortest diagonals in a regular n-sided polygon -/
def num_shortest_diagonals (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The probability of randomly selecting one of the shortest diagonals
    from all the diagonals of a regular n-sided polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (total_diagonals n : ℚ)

theorem prob_shortest_diagonal_21_sided :
  prob_shortest_diagonal n = 10 / 189 := by
  sorry

end prob_shortest_diagonal_21_sided_l196_19620


namespace coin_game_theorem_l196_19646

/-- Represents a pile of coins -/
structure CoinPile :=
  (count : ℕ)
  (hcount : count ≥ 2015)

/-- Represents the state of the three piles -/
structure GameState :=
  (pile1 : CoinPile)
  (pile2 : CoinPile)
  (pile3 : CoinPile)

/-- The polynomial f(x) = x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + 1 -/
def f (x : ℕ) : ℕ := x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + 1

/-- Represents a valid operation on the piles -/
inductive Operation
  | SplitEven (i : Fin 3) : Operation
  | RemoveOdd (i : Fin 3) : Operation

/-- Applies an operation to a game state -/
def applyOperation (state : GameState) (op : Operation) : GameState :=
  sorry

/-- Checks if a game state has reached the goal -/
def hasReachedGoal (state : GameState) : Prop :=
  ∃ (i : Fin 3), state.pile1.count ≥ 2017^2017 ∨ 
                 state.pile2.count ≥ 2017^2017 ∨ 
                 state.pile3.count ≥ 2017^2017

/-- The main theorem to prove -/
theorem coin_game_theorem (a b c : ℕ) 
  (ha : a ≥ 2015) (hb : b ≥ 2015) (hc : c ≥ 2015) :
  (∃ (ops : List Operation), 
    hasReachedGoal (ops.foldl applyOperation 
      { pile1 := ⟨a, ha⟩, pile2 := ⟨b, hb⟩, pile3 := ⟨c, hc⟩ })) ↔ 
  (f 2 = 2017 ∧ f 1 = 7) :=
sorry

end coin_game_theorem_l196_19646


namespace smallest_sum_of_sequence_l196_19625

theorem smallest_sum_of_sequence (P Q R S : ℤ) : 
  P > 0 → Q > 0 → R > 0 →  -- P, Q, R are positive integers
  (R - Q = Q - P) →  -- P, Q, R form an arithmetic sequence
  (R * R = Q * S) →  -- Q, R, S form a geometric sequence
  (R = (4 * Q) / 3) →  -- R/Q = 4/3
  (∀ P' Q' R' S' : ℤ, 
    P' > 0 → Q' > 0 → R' > 0 → 
    (R' - Q' = Q' - P') → 
    (R' * R' = Q' * S') → 
    (R' = (4 * Q') / 3) → 
    P + Q + R + S ≤ P' + Q' + R' + S') →
  P + Q + R + S = 171 :=
by sorry

end smallest_sum_of_sequence_l196_19625


namespace password_config_exists_l196_19699

/-- A password configuration is represented by a list of integers, 
    where each integer represents the count of a distinct character. -/
def PasswordConfig := List Nat

/-- The number of combinations for a given password configuration -/
def numCombinations (config : PasswordConfig) : Nat :=
  Nat.factorial 5 / (config.map Nat.factorial).prod

/-- Theorem: There exists a 5-character password configuration 
    that results in exactly 20 different combinations -/
theorem password_config_exists : ∃ (config : PasswordConfig), 
  config.sum = 5 ∧ numCombinations config = 20 := by
  sorry

end password_config_exists_l196_19699


namespace perpendicular_vectors_x_value_l196_19613

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 3]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define perpendicularity condition
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 0 + u 1 * v 1 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular a (b x) → x = -3/2 := by
  sorry

end perpendicular_vectors_x_value_l196_19613


namespace exists_square_farther_than_V_l196_19609

/-- Represents a square on the board --/
structure Square where
  x : Fin 19
  y : Fin 19

/-- Defines the movement of the dragon --/
def dragonMove (s : Square) : Set Square :=
  { t | (t.x = s.x + 4 ∧ t.y = s.y + 1) ∨
        (t.x = s.x + 4 ∧ t.y = s.y - 1) ∨
        (t.x = s.x - 4 ∧ t.y = s.y + 1) ∨
        (t.x = s.x - 4 ∧ t.y = s.y - 1) ∨
        (t.x = s.x + 1 ∧ t.y = s.y + 4) ∨
        (t.x = s.x + 1 ∧ t.y = s.y - 4) ∨
        (t.x = s.x - 1 ∧ t.y = s.y + 4) ∨
        (t.x = s.x - 1 ∧ t.y = s.y - 4) }

/-- Draconian distance between two squares --/
def draconianDistance (s t : Square) : ℕ :=
  sorry

/-- Corner square --/
def C : Square :=
  { x := 0, y := 0 }

/-- Diagonally adjacent square to C --/
def V : Square :=
  { x := 1, y := 1 }

/-- Main theorem --/
theorem exists_square_farther_than_V :
  ∃ X : Square, draconianDistance C X > draconianDistance C V :=
sorry

end exists_square_farther_than_V_l196_19609


namespace sufficient_not_necessary_condition_l196_19622

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1 → (a - 1) * (b - 1) > 0) ∧
  ¬(∀ a b : ℝ, (a - 1) * (b - 1) > 0 → a > 1 ∧ b > 1) :=
sorry

end sufficient_not_necessary_condition_l196_19622


namespace polynomial_primes_theorem_l196_19608

def is_valid_polynomial (Q : ℤ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, Q x = a * x^2 + b * x + c

def satisfies_condition (Q : ℤ → ℤ) : Prop :=
  ∃ (p₁ p₂ p₃ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    |Q p₁| = 11 ∧ |Q p₂| = 11 ∧ |Q p₃| = 11

def is_solution (Q : ℤ → ℤ) : Prop :=
  (∀ x, Q x = 11) ∨
  (∀ x, Q x = x^2 - 13*x + 11) ∨
  (∀ x, Q x = 2*x^2 - 32*x + 67) ∨
  (∀ x, Q x = 11*x^2 - 77*x + 121)

theorem polynomial_primes_theorem :
  ∀ Q : ℤ → ℤ, is_valid_polynomial Q → satisfies_condition Q → is_solution Q :=
sorry

end polynomial_primes_theorem_l196_19608


namespace number_of_schools_l196_19619

theorem number_of_schools (n : ℕ) : n = 22 :=
  -- Define the total number of students
  let total_students := 4 * n
  -- Define Alex's rank
  let alex_rank := 2 * n
  -- Define the ranks of Alex's teammates
  let jordan_rank := 45
  let kim_rank := 73
  let lee_rank := 98
  -- State the conditions
  have h1 : alex_rank < jordan_rank := by sorry
  have h2 : alex_rank < kim_rank := by sorry
  have h3 : alex_rank < lee_rank := by sorry
  have h4 : total_students = 2 * alex_rank - 1 := by sorry
  have h5 : alex_rank ≤ 49 := by sorry
  -- Prove that n = 22
  sorry

#check number_of_schools

end number_of_schools_l196_19619


namespace min_sum_squares_l196_19695

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + 3 * x₂ + 4 * x₃ = 72) (h_rel : x₁ = 3 * x₂) :
  x₁^2 + x₂^2 + x₃^2 ≥ 347.04 := by
  sorry

end min_sum_squares_l196_19695


namespace janet_paperclips_used_l196_19675

/-- The number of paper clips Janet used during the day -/
def paperclips_used (initial : ℕ) (found : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) (final : ℕ) : ℕ :=
  initial + found - given_per_friend * num_friends - final

/-- Theorem stating that Janet used 64 paper clips during the day -/
theorem janet_paperclips_used :
  paperclips_used 85 20 5 3 26 = 64 := by
  sorry

end janet_paperclips_used_l196_19675


namespace shooter_hit_rate_l196_19691

theorem shooter_hit_rate (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (1 - (1 - p)^4) = 80/81) : p = 2/3 := by
  sorry

end shooter_hit_rate_l196_19691


namespace limit_of_a_sequence_l196_19645

def a (n : ℕ) : ℚ := (9 - n^3) / (1 + 2*n^3)

theorem limit_of_a_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
by sorry

end limit_of_a_sequence_l196_19645


namespace car_speed_problem_l196_19652

/-- Proves that the speed at which a car travels 1 kilometer in 12 seconds less time
    than it does at 60 km/h is 50 km/h. -/
theorem car_speed_problem (v : ℝ) : 
  (1000 / (60000 / 3600) - 1000 / (v / 3600) = 12) → v = 50 := by
  sorry

end car_speed_problem_l196_19652


namespace events_B_C_complementary_l196_19616

-- Define the sample space (faces of the cube)
def S : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {n ∈ S | n % 2 = 1}
def B : Set Nat := {n ∈ S | n ≤ 3}
def C : Set Nat := {n ∈ S | n ≥ 4}

-- Theorem statement
theorem events_B_C_complementary : B ∪ C = S ∧ B ∩ C = ∅ :=
sorry

end events_B_C_complementary_l196_19616


namespace a_approximation_l196_19614

/-- For large x, the value of a that makes (a * x) / (0.5x - 406) closest to 3 is approximately 1.5 -/
theorem a_approximation (x : ℝ) (hx : x > 3000) :
  ∃ (a : ℝ), ∀ (ε : ℝ), ε > 0 → 
    ∃ (δ : ℝ), δ > 0 ∧ 
      ∀ (y : ℝ), y > x → 
        |((a * y) / (0.5 * y - 406) - 3)| < ε ∧ 
        |a - 1.5| < δ :=
sorry

end a_approximation_l196_19614


namespace star_seven_three_l196_19678

-- Define the ⋆ operation
def star (x y : ℤ) : ℤ := 2 * x - 4 * y

-- State the theorem
theorem star_seven_three : star 7 3 = 2 := by sorry

end star_seven_three_l196_19678


namespace student_calculation_error_l196_19612

theorem student_calculation_error (x : ℤ) : 
  (x + 5) - (x - (-5)) = 10 :=
sorry

end student_calculation_error_l196_19612


namespace molecular_weight_calculation_l196_19685

theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 960) 
  (h2 : num_moles = 5) : 
  total_weight / num_moles = 192 := by
sorry

end molecular_weight_calculation_l196_19685


namespace equation_solution_l196_19647

theorem equation_solution (a b : ℕ+) :
  2 * a ^ 2 = 3 * b ^ 3 ↔ ∃ k : ℕ+, a = 18 * k ^ 3 ∧ b = 6 * k ^ 2 := by
sorry

end equation_solution_l196_19647


namespace f_geq_f0_range_of_a_l196_19630

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem 1: f(x) ≥ f(0) for all x
theorem f_geq_f0 : ∀ x : ℝ, f x ≥ f 0 := by sorry

-- Theorem 2: Given 2f(x) ≥ f(a+1) for all x, the range of a is [-4.5, 1.5]
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * f x ≥ f (a + 1)) → -4.5 ≤ a ∧ a ≤ 1.5 := by sorry

end f_geq_f0_range_of_a_l196_19630


namespace tripled_minus_six_l196_19626

theorem tripled_minus_six (x : ℝ) : 3 * x - 6 = 15 → x = 7 := by
  sorry

end tripled_minus_six_l196_19626


namespace first_term_is_two_l196_19654

/-- A sequence of 5 terms where the differences between consecutive terms form an arithmetic sequence -/
def ArithmeticSequenceOfDifferences (a : Fin 5 → ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : Fin 3, a (i + 1) - a i = d + i

theorem first_term_is_two (a : Fin 5 → ℕ) 
  (h1 : a 1 = 4) 
  (h2 : a 2 = 7)
  (h3 : a 3 = 11)
  (h4 : a 4 = 16)
  (h5 : ArithmeticSequenceOfDifferences a) : 
  a 0 = 2 := by
  sorry

end first_term_is_two_l196_19654


namespace fraction_difference_l196_19628

theorem fraction_difference : (7 : ℚ) / 4 - (2 : ℚ) / 3 = (13 : ℚ) / 12 := by
  sorry

end fraction_difference_l196_19628
