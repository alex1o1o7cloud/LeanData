import Mathlib

namespace NUMINAMATH_CALUDE_c_absolute_value_l96_9640

def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem c_absolute_value (a b c : ℤ) :
  (∀ d : ℤ, d ≠ 1 → d ∣ a → d ∣ b → d ∣ c → False) →
  g a b c (3 + I) = 0 →
  |c| = 142 := by sorry

end NUMINAMATH_CALUDE_c_absolute_value_l96_9640


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l96_9602

/-- Two vectors in ℝ² are orthogonal if their dot product is zero -/
def orthogonal (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The first vector -/
def v : ℝ × ℝ := (3, 4)

/-- The second vector -/
def w (x : ℝ) : ℝ × ℝ := (x, -7)

/-- The theorem stating that the vectors are orthogonal when x = 28/3 -/
theorem vectors_orthogonal : orthogonal v (w (28/3)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l96_9602


namespace NUMINAMATH_CALUDE_distance_range_for_intersecting_circles_l96_9624

-- Define the radii of the two circles
def r₁ : ℝ := 3
def r₂ : ℝ := 5

-- Define the property of intersection
def intersecting (d : ℝ) : Prop := d < r₁ + r₂ ∧ d > abs (r₁ - r₂)

-- Theorem statement
theorem distance_range_for_intersecting_circles (d : ℝ) 
  (h : intersecting d) : 2 < d ∧ d < 8 := by sorry

end NUMINAMATH_CALUDE_distance_range_for_intersecting_circles_l96_9624


namespace NUMINAMATH_CALUDE_rumor_spread_l96_9693

theorem rumor_spread (n : ℕ) : (∃ m : ℕ, (3^(m+1) - 1) / 2 ≥ 1000 ∧ ∀ k < m, (3^(k+1) - 1) / 2 < 1000) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_rumor_spread_l96_9693


namespace NUMINAMATH_CALUDE_divisibility_implies_zero_product_l96_9653

theorem divisibility_implies_zero_product (p q r : ℝ) : 
  (∀ x, ∃ k, x^4 + 6*x^3 + 4*p*x^2 + 2*q*x + r = k * (x^3 + 4*x^2 + 2*x + 1)) →
  (p + q) * r = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_zero_product_l96_9653


namespace NUMINAMATH_CALUDE_petya_bonus_points_l96_9630

def calculate_bonus (final_score : ℕ) : ℕ :=
  if final_score < 1000 then
    (final_score * 20) / 100
  else if final_score < 2000 then
    200 + ((final_score - 1000) * 30) / 100
  else
    200 + 300 + ((final_score - 2000) * 50) / 100

theorem petya_bonus_points :
  calculate_bonus 2370 = 685 := by sorry

end NUMINAMATH_CALUDE_petya_bonus_points_l96_9630


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l96_9656

theorem solve_equation_and_evaluate (x : ℝ) : 
  2*x - 7 = 8*x - 1 → 5*(x - 3) = -20 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l96_9656


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l96_9665

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  manSpeed : ℝ  -- Speed of the man in still water (km/h)
  streamSpeed : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.manSpeed + s.streamSpeed else s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the man's speed in still water is 8 km/h. -/
theorem man_speed_in_still_water 
  (s : SwimmerSpeeds)
  (h1 : effectiveSpeed s true = 30 / 3)  -- Downstream condition
  (h2 : effectiveSpeed s false = 18 / 3) -- Upstream condition
  : s.manSpeed = 8 := by
  sorry

#check man_speed_in_still_water

end NUMINAMATH_CALUDE_man_speed_in_still_water_l96_9665


namespace NUMINAMATH_CALUDE_total_groom_time_l96_9676

def poodle_groom_time : ℕ := 30
def terrier_groom_time : ℕ := poodle_groom_time / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

theorem total_groom_time :
  num_poodles * poodle_groom_time + num_terriers * terrier_groom_time = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_l96_9676


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l96_9629

theorem max_value_trig_expression (a b c : ℝ) :
  (∃ (θ : ℝ), ∀ (φ : ℝ), a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ) ≥ 
                         a * Real.cos φ + b * Real.sin φ + c * Real.sin (2 * φ)) →
  (∃ (θ : ℝ), a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ) = Real.sqrt (2 * (a^2 + b^2 + c^2))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l96_9629


namespace NUMINAMATH_CALUDE_square_area_with_corner_circles_l96_9679

theorem square_area_with_corner_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by sorry

end NUMINAMATH_CALUDE_square_area_with_corner_circles_l96_9679


namespace NUMINAMATH_CALUDE_law_school_applicants_l96_9621

theorem law_school_applicants (PS : ℕ) (GPA_high : ℕ) (not_PS_GPA_low : ℕ) (PS_GPA_high : ℕ) :
  PS = 15 →
  GPA_high = 20 →
  not_PS_GPA_low = 10 →
  PS_GPA_high = 5 →
  PS + GPA_high - PS_GPA_high + not_PS_GPA_low = 40 :=
by sorry

end NUMINAMATH_CALUDE_law_school_applicants_l96_9621


namespace NUMINAMATH_CALUDE_quadratic_y_values_order_l96_9695

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  f (-2) = 1 ∧ (∀ x, f x ≤ f (-2))

/-- Theorem stating the relationship between y-values of specific points -/
theorem quadratic_y_values_order (f : ℝ → ℝ) (y₁ y₂ y₃ : ℝ)
  (hf : QuadraticFunction f)
  (h1 : f 1 = y₁)
  (h2 : f (-1) = y₂)
  (h3 : f (-4) = y₃) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_y_values_order_l96_9695


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l96_9645

/-- Hexagon ABCDEF with given side lengths -/
structure Hexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  AF : ℝ

/-- The perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.AF

/-- Theorem: The perimeter of the given hexagon is 7 + √10 -/
theorem hexagon_perimeter : 
  ∀ (h : Hexagon), 
  h.AB = 1 → h.BC = 1 → h.CD = 2 → h.DE = 2 → h.EF = 1 → h.AF = Real.sqrt 10 →
  perimeter h = 7 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l96_9645


namespace NUMINAMATH_CALUDE_nested_function_evaluation_l96_9692

-- Define P and Q functions
def P (x : ℝ) : ℝ := 3 * (x ^ (1/3))
def Q (x : ℝ) : ℝ := x ^ 3

-- State the theorem
theorem nested_function_evaluation :
  P (Q (P (Q (P (Q 4))))) = 108 :=
sorry

end NUMINAMATH_CALUDE_nested_function_evaluation_l96_9692


namespace NUMINAMATH_CALUDE_spring_mass_for_32cm_l96_9607

/-- Represents the relationship between spring length and mass -/
def spring_length (initial_length : ℝ) (extension_rate : ℝ) (mass : ℝ) : ℝ :=
  initial_length + extension_rate * mass

/-- Theorem: For a spring with initial length 18 cm and extension rate 2 cm/kg,
    a length of 32 cm corresponds to a mass of 7 kg -/
theorem spring_mass_for_32cm :
  spring_length 18 2 7 = 32 :=
by sorry

end NUMINAMATH_CALUDE_spring_mass_for_32cm_l96_9607


namespace NUMINAMATH_CALUDE_compare_expressions_compare_square_roots_l96_9606

-- Problem 1
theorem compare_expressions (x y : ℝ) : x^2 + y^2 + 1 > 2*(x + y - 1) := by
  sorry

-- Problem 2
theorem compare_square_roots (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) :
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_compare_square_roots_l96_9606


namespace NUMINAMATH_CALUDE_three_minus_one_point_two_repeating_l96_9673

/-- The decimal representation of 1.2 repeating -/
def one_point_two_repeating : ℚ := 11 / 9

/-- Proof that 3 - 1.2 repeating equals 16/9 -/
theorem three_minus_one_point_two_repeating :
  3 - one_point_two_repeating = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_three_minus_one_point_two_repeating_l96_9673


namespace NUMINAMATH_CALUDE_largest_frog_weight_l96_9672

theorem largest_frog_weight (S L : ℝ) 
  (h1 : L = 10 * S) 
  (h2 : L = S + 108) : 
  L = 120 := by
sorry

end NUMINAMATH_CALUDE_largest_frog_weight_l96_9672


namespace NUMINAMATH_CALUDE_locus_of_midpoint_is_circle_l96_9616

/-- Given a circle with center O and radius R, and a point P inside the circle,
    we rotate a right angle around P. The legs of the right angle intersect
    the circle at points A and B. This theorem proves that the locus of the
    midpoint of chord AB is a circle. -/
theorem locus_of_midpoint_is_circle
  (O : ℝ × ℝ)  -- Center of the circle
  (R : ℝ)      -- Radius of the circle
  (P : ℝ × ℝ)  -- Point inside the circle
  (h_R_pos : R > 0)  -- R is positive
  (h_P_inside : dist P O < R)  -- P is inside the circle
  (A B : ℝ × ℝ)  -- Points on the circle
  (h_A_on_circle : dist A O = R)  -- A is on the circle
  (h_B_on_circle : dist B O = R)  -- B is on the circle
  (h_right_angle : (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0)  -- ∠APB is a right angle
  : ∃ (C : ℝ × ℝ) (r : ℝ),
    let a := dist P O
    let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- Midpoint of AB
    C = (a / 2, 0) ∧ r = (1 / 2) * Real.sqrt (2 * R^2 - a^2) ∧
    dist M C = r :=
by sorry

end NUMINAMATH_CALUDE_locus_of_midpoint_is_circle_l96_9616


namespace NUMINAMATH_CALUDE_subset_intersection_condition_l96_9603

theorem subset_intersection_condition (n : ℕ) (h : n ≥ 4) :
  (∀ (S : Finset (Finset (Fin n))) (h_card : S.card = n) 
    (h_subsets : ∀ s ∈ S, s.card = 3),
    ∃ (s1 s2 : Finset (Fin n)), s1 ∈ S ∧ s2 ∈ S ∧ s1 ≠ s2 ∧ (s1 ∩ s2).card = 1) ↔
  n % 4 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_subset_intersection_condition_l96_9603


namespace NUMINAMATH_CALUDE_distribution_schemes_correct_l96_9612

/-- The number of ways to distribute 5 volunteers to 3 different Olympic venues,
    with at least one volunteer assigned to each venue. -/
def distributionSchemes : ℕ := 150

/-- Theorem stating that the number of distribution schemes is correct. -/
theorem distribution_schemes_correct : distributionSchemes = 150 := by sorry

end NUMINAMATH_CALUDE_distribution_schemes_correct_l96_9612


namespace NUMINAMATH_CALUDE_minimum_radios_l96_9661

theorem minimum_radios (n d : ℕ) (h1 : d > 0) : 
  (3 * (d / n / 3) + (n - 3) * (d / n + 12) - d = 108) →
  (∀ m : ℕ, m < n → ¬(3 * (d / m / 3) + (m - 3) * (d / m + 12) - d = 108)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_minimum_radios_l96_9661


namespace NUMINAMATH_CALUDE_find_a_l96_9687

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then a^x else -a^(-x)

-- State the theorem
theorem find_a : 
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ 
  (∀ x, f a (-x) = -(f a x)) ∧  -- odd function property
  (∀ x > 0, f a x = a^x) ∧      -- definition for x > 0
  f a (Real.log 2 / Real.log (1/2)) = -3 ∧ -- given condition
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l96_9687


namespace NUMINAMATH_CALUDE_election_votes_proof_l96_9664

theorem election_votes_proof (total_votes : ℕ) 
  (h1 : (total_votes : ℝ) * 0.8 * 0.55 + 2520 = (total_votes : ℝ) * 0.8) : 
  total_votes = 7000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_proof_l96_9664


namespace NUMINAMATH_CALUDE_parallel_line_theorem_l96_9674

/-- A line parallel to another line with a given y-intercept -/
def parallel_line_with_y_intercept (a b c : ℝ) (y_intercept : ℝ) : Prop :=
  ∃ k : ℝ, (k ≠ 0) ∧ 
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ a * x + b * y + k = 0) ∧
  (a * 0 + b * y_intercept + k = 0)

/-- The equation x + y + 1 = 0 represents the line parallel to x + y + 4 = 0 with y-intercept -1 -/
theorem parallel_line_theorem :
  parallel_line_with_y_intercept 1 1 4 (-1) →
  ∀ x y : ℝ, x + y + 1 = 0 ↔ parallel_line_with_y_intercept 1 1 4 (-1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_l96_9674


namespace NUMINAMATH_CALUDE_stripe_difference_l96_9638

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The total number of stripes on Olga's shoes -/
def olga_total_stripes : ℕ := 2 * olga_stripes_per_shoe

/-- The total number of stripes on Hortense's shoes -/
def hortense_total_stripes : ℕ := 2 * olga_total_stripes

/-- The total number of stripes on all their shoes -/
def total_stripes : ℕ := 22

/-- The number of stripes on Rick's shoes -/
def rick_total_stripes : ℕ := total_stripes - olga_total_stripes - hortense_total_stripes

theorem stripe_difference : olga_total_stripes - rick_total_stripes = 2 := by
  sorry

end NUMINAMATH_CALUDE_stripe_difference_l96_9638


namespace NUMINAMATH_CALUDE_lamp_probability_l96_9609

/-- The number of red lamps -/
def num_red_lamps : ℕ := 4

/-- The number of blue lamps -/
def num_blue_lamps : ℕ := 4

/-- The total number of lamps -/
def total_lamps : ℕ := num_red_lamps + num_blue_lamps

/-- The number of lamps turned on -/
def num_on_lamps : ℕ := 4

/-- The probability of the leftmost lamp being blue and on, and the rightmost lamp being red and off -/
theorem lamp_probability : 
  (num_red_lamps : ℚ) * (num_blue_lamps : ℚ) * (Nat.choose (total_lamps - 2) (num_on_lamps - 1)) / 
  ((Nat.choose total_lamps num_red_lamps) * (Nat.choose total_lamps num_on_lamps)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lamp_probability_l96_9609


namespace NUMINAMATH_CALUDE_symmetric_graph_phi_l96_9649

/-- Given a function f and a real number φ, proves that if the graph of y = f(x + φ) 
    is symmetric about x = 0 and |φ| ≤ π/2, then φ = π/6 -/
theorem symmetric_graph_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x : ℝ, f x = 2 * Real.sin (x + π/3)) →
  (∀ x : ℝ, f (x + φ) = f (-x + φ)) →
  |φ| ≤ π/2 →
  φ = π/6 := by
  sorry

#check symmetric_graph_phi

end NUMINAMATH_CALUDE_symmetric_graph_phi_l96_9649


namespace NUMINAMATH_CALUDE_direct_variation_with_constant_l96_9627

/-- A function that varies directly as x with an additional constant term -/
def f (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem stating that if f(3) = 9 and f(4) = 12, then f(-5) = -15 -/
theorem direct_variation_with_constant 
  (k c : ℝ) 
  (h1 : f k c 3 = 9) 
  (h2 : f k c 4 = 12) : 
  f k c (-5) = -15 := by
  sorry

#check direct_variation_with_constant

end NUMINAMATH_CALUDE_direct_variation_with_constant_l96_9627


namespace NUMINAMATH_CALUDE_nabla_computation_l96_9639

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_computation : nabla (nabla 1 3) 2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_nabla_computation_l96_9639


namespace NUMINAMATH_CALUDE_fraction_comparison_l96_9690

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a / b > (a + 1) / (b + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l96_9690


namespace NUMINAMATH_CALUDE_samantha_birth_year_l96_9654

-- Define the year of the first AMC 8
def first_amc8_year : ℕ := 1985

-- Define the frequency of AMC 8 (every 2 years)
def amc8_frequency : ℕ := 2

-- Define Samantha's age when she took the fourth AMC 8
def samantha_age_fourth_amc8 : ℕ := 12

-- Function to calculate the year of the nth AMC 8
def nth_amc8_year (n : ℕ) : ℕ :=
  first_amc8_year + (n - 1) * amc8_frequency

-- Theorem to prove Samantha's birth year
theorem samantha_birth_year :
  nth_amc8_year 4 - samantha_age_fourth_amc8 = 1981 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_l96_9654


namespace NUMINAMATH_CALUDE_topsoil_cost_l96_9675

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 8

/-- Theorem: The cost of 8 cubic yards of topsoil is $1728 -/
theorem topsoil_cost : 
  volume_in_cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l96_9675


namespace NUMINAMATH_CALUDE_range_of_f_l96_9633

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc (-8 : ℝ) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l96_9633


namespace NUMINAMATH_CALUDE_percentage_difference_l96_9671

theorem percentage_difference (n : ℝ) (x y : ℝ) 
  (h1 : n = 160) 
  (h2 : x > y) 
  (h3 : (x / 100) * n - (y / 100) * n = 24) : 
  x - y = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l96_9671


namespace NUMINAMATH_CALUDE_max_perimeter_triangle_is_isosceles_l96_9643

/-- Given a fixed base length and a fixed angle at one vertex, 
    the triangle with maximum perimeter is isosceles. -/
theorem max_perimeter_triangle_is_isosceles 
  (b : ℝ) 
  (β : ℝ) 
  (h1 : b > 0) 
  (h2 : 0 < β ∧ β < π) : 
  ∃ (a c : ℝ), 
    a > 0 ∧ c > 0 ∧
    a = c ∧
    ∀ (a' c' : ℝ), 
      a' > 0 → c' > 0 → 
      a' + b + c' ≤ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_triangle_is_isosceles_l96_9643


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l96_9670

theorem ceiling_floor_product (y : ℝ) : 
  y > 0 → ⌈y⌉ * ⌊y⌋ = 72 → 8 < y ∧ y < 9 := by
sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l96_9670


namespace NUMINAMATH_CALUDE_dessert_distribution_l96_9663

theorem dessert_distribution (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) :
  mini_cupcakes = 14 →
  donut_holes = 12 →
  students = 13 →
  (mini_cupcakes + donut_holes) % students = 0 →
  (mini_cupcakes + donut_holes) / students = 2 :=
by sorry

end NUMINAMATH_CALUDE_dessert_distribution_l96_9663


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l96_9698

theorem square_difference_divided_by_nine : (110^2 - 95^2) / 9 = 3075 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l96_9698


namespace NUMINAMATH_CALUDE_min_sum_with_product_and_even_constraint_l96_9667

theorem min_sum_with_product_and_even_constraint (a b : ℤ) : 
  a * b = 72 → Even a → (∀ (x y : ℤ), x * y = 72 → Even x → a + b ≤ x + y) → a + b = -38 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_product_and_even_constraint_l96_9667


namespace NUMINAMATH_CALUDE_correct_average_points_l96_9637

/-- Represents Melissa's basketball season statistics -/
structure BasketballSeason where
  totalGames : ℕ
  totalPoints : ℕ
  wonGames : ℕ
  averagePointDifference : ℕ

/-- Calculates the average points scored in won and lost games -/
def calculateAveragePoints (season : BasketballSeason) : ℕ × ℕ :=
  sorry

/-- Theorem stating the correct average points for won and lost games -/
theorem correct_average_points (season : BasketballSeason) 
  (h1 : season.totalGames = 20)
  (h2 : season.totalPoints = 400)
  (h3 : season.wonGames = 8)
  (h4 : season.averagePointDifference = 15) :
  calculateAveragePoints season = (29, 14) := by
  sorry

end NUMINAMATH_CALUDE_correct_average_points_l96_9637


namespace NUMINAMATH_CALUDE_product_cde_eq_1000_l96_9617

theorem product_cde_eq_1000 
  (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.5) :
  c * d * e = 1000 := by
sorry

end NUMINAMATH_CALUDE_product_cde_eq_1000_l96_9617


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l96_9681

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l96_9681


namespace NUMINAMATH_CALUDE_storage_to_total_ratio_l96_9615

def total_planks : ℕ := 200
def friends_planks : ℕ := 20
def store_planks : ℕ := 30

def parents_planks : ℕ := total_planks / 2

def storage_planks : ℕ := total_planks - parents_planks - friends_planks - store_planks

theorem storage_to_total_ratio :
  (storage_planks : ℚ) / total_planks = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_storage_to_total_ratio_l96_9615


namespace NUMINAMATH_CALUDE_infinitely_many_primes_6k_plus_1_l96_9610

theorem infinitely_many_primes_6k_plus_1 :
  ∀ S : Finset Nat, (∀ p ∈ S, Prime p ∧ ∃ k, p = 6*k + 1) →
  ∃ q, Prime q ∧ (∃ m, q = 6*m + 1) ∧ q ∉ S :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_6k_plus_1_l96_9610


namespace NUMINAMATH_CALUDE_function_b_increasing_on_negative_reals_l96_9628

/-- The function f(x) = 1 - 1/x is increasing on the interval (-∞,0) -/
theorem function_b_increasing_on_negative_reals :
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → (1 - 1/x) < (1 - 1/y) := by
sorry

end NUMINAMATH_CALUDE_function_b_increasing_on_negative_reals_l96_9628


namespace NUMINAMATH_CALUDE_elenas_garden_tulips_l96_9604

/-- Represents Elena's garden with lilies and tulips. -/
structure Garden where
  lilies : ℕ
  tulips : ℕ
  lily_petals : ℕ
  tulip_petals : ℕ
  total_petals : ℕ

/-- Theorem stating the number of tulips in Elena's garden. -/
theorem elenas_garden_tulips (g : Garden)
  (h1 : g.lilies = 8)
  (h2 : g.lily_petals = 6)
  (h3 : g.tulip_petals = 3)
  (h4 : g.total_petals = 63)
  (h5 : g.total_petals = g.lilies * g.lily_petals + g.tulips * g.tulip_petals) :
  g.tulips = 5 := by
  sorry


end NUMINAMATH_CALUDE_elenas_garden_tulips_l96_9604


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l96_9694

-- First expression
theorem simplify_expression_1 (a b : ℝ) :
  4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b := by sorry

-- Second expression
theorem simplify_expression_2 (m : ℝ) :
  3 * m / 2 - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l96_9694


namespace NUMINAMATH_CALUDE_garden_area_l96_9611

-- Define the rectangle garden
def rectangular_garden (length width : ℝ) := length * width

-- Theorem statement
theorem garden_area : rectangular_garden 12 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l96_9611


namespace NUMINAMATH_CALUDE_hockey_league_games_l96_9619

/-- The number of games played in a hockey league season -/
def number_of_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

theorem hockey_league_games :
  number_of_games 17 10 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l96_9619


namespace NUMINAMATH_CALUDE_tangent_plane_and_normal_line_at_point_A_l96_9683

-- Define the elliptic paraboloid
def elliptic_paraboloid (x y z : ℝ) : Prop := z = 2 * x^2 + y^2

-- Define the point A
def point_A : ℝ × ℝ × ℝ := (1, -1, 3)

-- Define the tangent plane equation
def tangent_plane (x y z : ℝ) : Prop := 4 * x - 2 * y - z - 3 = 0

-- Define the normal line equations
def normal_line (x y z : ℝ) : Prop :=
  (x - 1) / 4 = (y + 1) / (-2) ∧ (y + 1) / (-2) = (z - 3) / (-1)

-- Theorem statement
theorem tangent_plane_and_normal_line_at_point_A :
  ∀ x y z : ℝ,
  elliptic_paraboloid x y z →
  (x, y, z) = point_A →
  tangent_plane x y z ∧ normal_line x y z :=
sorry

end NUMINAMATH_CALUDE_tangent_plane_and_normal_line_at_point_A_l96_9683


namespace NUMINAMATH_CALUDE_total_students_l96_9614

theorem total_students (general : ℕ) (biology : ℕ) (chemistry : ℕ) (math : ℕ) (arts : ℕ) 
  (physics : ℕ) (history : ℕ) (literature : ℕ) : 
  general = 30 →
  biology = 2 * general →
  chemistry = general + 10 →
  math = (3 * (general + biology + chemistry)) / 5 →
  arts * 20 / 100 = general →
  physics = general + chemistry - 5 →
  history = (3 * general) / 4 →
  literature = history + 15 →
  general + biology + chemistry + math + arts + physics + history + literature = 484 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l96_9614


namespace NUMINAMATH_CALUDE_circle_centers_distance_l96_9608

-- Define the circles
def circle_O₁ : ℝ := 3
def circle_O₂ : ℝ := 7

-- Define the condition of at most one common point
def at_most_one_common_point (d : ℝ) : Prop :=
  d ≥ circle_O₁ + circle_O₂ ∨ d ≤ abs (circle_O₁ - circle_O₂)

-- State the theorem
theorem circle_centers_distance (d : ℝ) :
  at_most_one_common_point d → d ≠ 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l96_9608


namespace NUMINAMATH_CALUDE_mikes_video_games_l96_9685

theorem mikes_video_games :
  ∀ (total_games working_games nonworking_games : ℕ) 
    (price_per_game total_earnings : ℕ),
  nonworking_games = 8 →
  price_per_game = 7 →
  total_earnings = 56 →
  working_games * price_per_game = total_earnings →
  total_games = working_games + nonworking_games →
  total_games = 16 := by
sorry

end NUMINAMATH_CALUDE_mikes_video_games_l96_9685


namespace NUMINAMATH_CALUDE_quartic_polynomial_property_l96_9657

def Q (x : ℝ) (e f : ℝ) : ℝ := 3 * x^4 + 24 * x^3 + e * x^2 + f * x + 16

theorem quartic_polynomial_property (e f : ℝ) :
  (∀ r₁ r₂ r₃ r₄ : ℝ, Q r₁ e f = 0 ∧ Q r₂ e f = 0 ∧ Q r₃ e f = 0 ∧ Q r₄ e f = 0 →
    (-24 / 12 = e / 3) ∧
    (-24 / 12 = 3 + 24 + e + f + 16) ∧
    (e / 3 = 3 + 24 + e + f + 16)) →
  f = -39 := by
sorry

end NUMINAMATH_CALUDE_quartic_polynomial_property_l96_9657


namespace NUMINAMATH_CALUDE_power_of_integer_for_3150_l96_9659

theorem power_of_integer_for_3150 (a : ℕ) (h1 : a = 14) 
  (h2 : ∀ k < a, ∃ n : ℕ, 3150 * k = n ^ 2 → False) : 
  ∃ n : ℕ, 3150 * a = n ^ 2 :=
sorry

end NUMINAMATH_CALUDE_power_of_integer_for_3150_l96_9659


namespace NUMINAMATH_CALUDE_line_circle_intersection_l96_9684

/-- The line kx - 2y + 1 = 0 always intersects the circle x^2 + (y-1)^2 = 1 for any real k -/
theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), k * x - 2 * y + 1 = 0 ∧ x^2 + (y - 1)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l96_9684


namespace NUMINAMATH_CALUDE_complex_number_location_l96_9691

theorem complex_number_location :
  ∀ (z : ℂ), (2 + I) * z = -I →
  (z.re < 0 ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l96_9691


namespace NUMINAMATH_CALUDE_perfect_square_condition_l96_9677

/-- A quadratic trinomial ax^2 + bx + c is a perfect square if there exists a real number r such that ax^2 + bx + c = (√a * x + r)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

/-- The main theorem: if x^2 - mx + 16 is a perfect square trinomial, then m = 8 or m = -8 -/
theorem perfect_square_condition (m : ℝ) :
  is_perfect_square_trinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l96_9677


namespace NUMINAMATH_CALUDE_distinct_hexagon_colorings_l96_9631

-- Define the number of disks and colors
def num_disks : ℕ := 6
def num_blue : ℕ := 3
def num_red : ℕ := 2
def num_green : ℕ := 1

-- Define the symmetry group of a hexagon
def hexagon_symmetries : ℕ := 12

-- Define the function to calculate the number of distinct colorings
def distinct_colorings : ℕ :=
  let total_colorings := (num_disks.choose num_blue) * ((num_disks - num_blue).choose num_red)
  let fixed_points_identity := total_colorings
  let fixed_points_reflection := 3 * (3 * 2 * 1)  -- 3 reflections, each with 6 fixed points
  let fixed_points_rotation := 0  -- 120° and 240° rotations have no fixed points
  (fixed_points_identity + fixed_points_reflection + fixed_points_rotation) / hexagon_symmetries

-- Theorem statement
theorem distinct_hexagon_colorings :
  distinct_colorings = 13 :=
sorry

end NUMINAMATH_CALUDE_distinct_hexagon_colorings_l96_9631


namespace NUMINAMATH_CALUDE_max_value_f_range_of_a_l96_9605

-- Define the function f(x) = x^2 - 2ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Theorem 1: Maximum value of f(x) when a = 1 and x ∈ [-1, 2]
theorem max_value_f (x : ℝ) (h : x ∈ Set.Icc (-1) 2) : 
  f 1 x ≤ 5 :=
sorry

-- Theorem 2: Range of a when f(x) ≥ a for x ∈ [-1, +∞)
theorem range_of_a (a : ℝ) :
  (∀ x ≥ -1, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_range_of_a_l96_9605


namespace NUMINAMATH_CALUDE_total_spent_is_2094_l96_9635

def apple_price : ℚ := 40
def pear_price : ℚ := 50
def orange_price : ℚ := 30
def grape_price : ℚ := 60

def apple_quantity : ℚ := 14
def pear_quantity : ℚ := 18
def orange_quantity : ℚ := 10
def grape_quantity : ℚ := 8

def apple_discount : ℚ := 0.1
def pear_discount : ℚ := 0.05
def orange_discount : ℚ := 0.15
def grape_discount : ℚ := 0

def total_spent : ℚ := 
  apple_quantity * apple_price * (1 - apple_discount) +
  pear_quantity * pear_price * (1 - pear_discount) +
  orange_quantity * orange_price * (1 - orange_discount) +
  grape_quantity * grape_price * (1 - grape_discount)

theorem total_spent_is_2094 : total_spent = 2094 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_2094_l96_9635


namespace NUMINAMATH_CALUDE_thomas_escalator_problem_l96_9689

/-- Thomas's escalator problem -/
theorem thomas_escalator_problem 
  (l : ℝ) -- length of the escalator
  (v : ℝ) -- speed of the escalator when working
  (r : ℝ) -- Thomas's running speed
  (w : ℝ) -- Thomas's walking speed
  (h1 : l / (v + r) = 15) -- Thomas runs down moving escalator in 15 seconds
  (h2 : l / (v + w) = 30) -- Thomas walks down moving escalator in 30 seconds
  (h3 : l / r = 20) -- Thomas runs down broken escalator in 20 seconds
  : l / w = 60 := by
  sorry

end NUMINAMATH_CALUDE_thomas_escalator_problem_l96_9689


namespace NUMINAMATH_CALUDE_simplify_expression_l96_9634

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 360 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l96_9634


namespace NUMINAMATH_CALUDE_rectangle_triangle_count_l96_9680

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A rectangle defined by 6 points -/
structure Rectangle :=
  (A B C D E F : Point)

/-- A triangle defined by 3 points -/
structure Triangle :=
  (p1 p2 p3 : Point)

/-- Count the number of triangles with one vertex at a given point -/
def countTriangles (R : Rectangle) (p : Point) : ℕ :=
  sorry

theorem rectangle_triangle_count (R : Rectangle) :
  (countTriangles R R.A = 9) ∧ (countTriangles R R.F = 9) :=
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_count_l96_9680


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l96_9636

theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_radius := 4 * r
  let new_height := 3 * h
  (π * new_radius^2 * new_height) / (π * r^2 * h) = 48 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l96_9636


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l96_9642

/-- The maximum mark for paper I -/
def maximum_mark : ℕ := 186

/-- The passing percentage as a rational number -/
def passing_percentage : ℚ := 35 / 100

/-- The marks scored by the candidate -/
def scored_marks : ℕ := 42

/-- The marks by which the candidate failed -/
def failing_margin : ℕ := 23

/-- Theorem stating the maximum mark for paper I -/
theorem paper_I_maximum_mark :
  (↑maximum_mark * passing_percentage).floor = scored_marks + failing_margin :=
sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l96_9642


namespace NUMINAMATH_CALUDE_camp_attendance_l96_9650

theorem camp_attendance (total_lawrence : ℕ) (stayed_home : ℕ) (went_to_camp : ℕ)
  (h1 : total_lawrence = 1538832)
  (h2 : stayed_home = 644997)
  (h3 : went_to_camp = 893835)
  (h4 : total_lawrence = stayed_home + went_to_camp) :
  0 = went_to_camp - (total_lawrence - stayed_home) :=
by sorry

end NUMINAMATH_CALUDE_camp_attendance_l96_9650


namespace NUMINAMATH_CALUDE_sum_base6_100_l96_9623

/-- Converts a number from base 10 to base 6 -/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 6 to base 10 -/
def fromBase6 (n : ℕ) : ℕ := sorry

/-- Sum of numbers from 1 to n in base 6 -/
def sumBase6 (n : ℕ) : ℕ := sorry

theorem sum_base6_100 : sumBase6 100 = toBase6 (fromBase6 6110) := by sorry

end NUMINAMATH_CALUDE_sum_base6_100_l96_9623


namespace NUMINAMATH_CALUDE_clients_equal_cars_l96_9669

theorem clients_equal_cars (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_cars = 18)
  (h2 : selections_per_client = 3)
  (h3 : selections_per_car = 3) :
  (num_cars * selections_per_car) / selections_per_client = num_cars :=
by sorry

end NUMINAMATH_CALUDE_clients_equal_cars_l96_9669


namespace NUMINAMATH_CALUDE_sams_price_per_sheet_l96_9662

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  price_per_sheet : ℝ
  sitting_fee : ℝ

/-- Calculates the total cost for a given number of sheets -/
def total_cost (company : PhotoCompany) (sheets : ℝ) : ℝ :=
  company.price_per_sheet * sheets + company.sitting_fee

/-- Proves that Sam's Picture Emporium charges $1.50 per sheet -/
theorem sams_price_per_sheet :
  let johns := PhotoCompany.mk 2.75 125
  let sams := PhotoCompany.mk x 140
  total_cost johns 12 = total_cost sams 12 →
  x = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_sams_price_per_sheet_l96_9662


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_greatest_integer_with_gcd_six_exists_l96_9646

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 132 :=
by sorry

theorem greatest_integer_with_gcd_six_exists : ∃ n : ℕ, n = 132 ∧ n < 150 ∧ Nat.gcd n 18 = 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_greatest_integer_with_gcd_six_exists_l96_9646


namespace NUMINAMATH_CALUDE_corn_row_length_l96_9648

/-- Calculates the length of a row of seeds in feet, given the space per seed and the number of seeds. -/
def row_length_in_feet (space_per_seed_inches : ℕ) (num_seeds : ℕ) : ℚ :=
  (space_per_seed_inches * num_seeds : ℚ) / 12

/-- Theorem stating that a row with 80 seeds, each requiring 18 inches of space, is 120 feet long. -/
theorem corn_row_length :
  row_length_in_feet 18 80 = 120 := by
  sorry

#eval row_length_in_feet 18 80

end NUMINAMATH_CALUDE_corn_row_length_l96_9648


namespace NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l96_9699

theorem fourth_term_is_one_tenth (a : ℕ → ℚ) :
  (∀ n : ℕ, a n = 2 / (n^2 + n)) →
  a 4 = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l96_9699


namespace NUMINAMATH_CALUDE_min_value_a_l96_9647

theorem min_value_a (a : ℕ) (h : 17 ∣ (50^2023 + a)) : 
  ∀ b : ℕ, (17 ∣ (50^2023 + b)) → a ≤ b → 18 ≤ a := by
sorry

end NUMINAMATH_CALUDE_min_value_a_l96_9647


namespace NUMINAMATH_CALUDE_fifth_square_area_l96_9620

theorem fifth_square_area (s : ℝ) (h : s + 5 = 11) : s^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fifth_square_area_l96_9620


namespace NUMINAMATH_CALUDE_die_expected_value_l96_9626

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The strategy for two rolls -/
def strategy2 (d : Die) : Bool :=
  d.val ≥ 4

/-- The strategy for three rolls -/
def strategy3 (d : Die) : Bool :=
  d.val ≥ 5

/-- Expected value of a single roll -/
def E1 : ℚ := 3.5

/-- Expected value with two opportunities to roll -/
def E2 : ℚ := 4.25

/-- Expected value with three opportunities to roll -/
def E3 : ℚ := 14/3

theorem die_expected_value :
  (E2 = 4.25) ∧ (E3 = 14/3) := by
  sorry

end NUMINAMATH_CALUDE_die_expected_value_l96_9626


namespace NUMINAMATH_CALUDE_percentage_of_160_to_50_l96_9622

theorem percentage_of_160_to_50 : ∀ (x y : ℝ), x = 160 ∧ y = 50 → (x / y) * 100 = 320 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_160_to_50_l96_9622


namespace NUMINAMATH_CALUDE_inequality_solution_l96_9625

theorem inequality_solution (x : ℝ) :
  (x^2 - 9) / (x^2 - 1) > 0 ↔ x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l96_9625


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l96_9618

theorem sphere_volume_from_surface_area :
  ∀ (R : ℝ), 
  R > 0 →
  4 * π * R^2 = 24 * π →
  (4 / 3) * π * R^3 = 8 * Real.sqrt 6 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l96_9618


namespace NUMINAMATH_CALUDE_court_cases_dismissed_l96_9696

theorem court_cases_dismissed (total_cases : ℕ) 
  (remaining_cases : ℕ) (innocent_cases : ℕ) (delayed_cases : ℕ) (guilty_cases : ℕ) :
  total_cases = 17 →
  remaining_cases = innocent_cases + delayed_cases + guilty_cases →
  innocent_cases = 2 * (remaining_cases / 3) →
  delayed_cases = 1 →
  guilty_cases = 4 →
  total_cases - remaining_cases = 2 := by
sorry

end NUMINAMATH_CALUDE_court_cases_dismissed_l96_9696


namespace NUMINAMATH_CALUDE_probability_three_primes_l96_9651

def num_dice : ℕ := 5
def num_primes : ℕ := 3
def prob_prime : ℚ := 2/5

theorem probability_three_primes :
  (num_dice.choose num_primes : ℚ) * prob_prime^num_primes * (1 - prob_prime)^(num_dice - num_primes) = 720/3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_l96_9651


namespace NUMINAMATH_CALUDE_expression_evaluation_l96_9697

theorem expression_evaluation : (10 + 1/3) + (-11.5) + (-10 - 1/3) - 4.5 = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l96_9697


namespace NUMINAMATH_CALUDE_log_base_three_squared_l96_9600

theorem log_base_three_squared (m : ℝ) (b : ℝ) (h : 3^m = b) : 
  Real.log b / Real.log (3^2) = m / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_base_three_squared_l96_9600


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l96_9601

def indistinguishable_distributions (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem five_balls_three_boxes : 
  indistinguishable_distributions 5 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l96_9601


namespace NUMINAMATH_CALUDE_sibling_count_product_l96_9632

/-- Represents a family with a given number of boys and girls -/
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

/-- Represents a sibling in a family -/
structure Sibling :=
  (family : Family)
  (isBoy : Bool)

/-- Counts the number of sisters a sibling has -/
def sisterCount (s : Sibling) : ℕ :=
  s.family.girls - if s.isBoy then 0 else 1

/-- Counts the number of brothers a sibling has -/
def brotherCount (s : Sibling) : ℕ :=
  s.family.boys - if s.isBoy then 1 else 0

theorem sibling_count_product (f : Family) (h : Sibling) (henry : Sibling)
    (henry_sisters : sisterCount henry = 4)
    (henry_brothers : brotherCount henry = 7)
    (h_family : h.family = f)
    (henry_family : henry.family = f)
    (h_girl : h.isBoy = false)
    (henry_boy : henry.isBoy = true) :
    sisterCount h * brotherCount h = 28 := by
  sorry

end NUMINAMATH_CALUDE_sibling_count_product_l96_9632


namespace NUMINAMATH_CALUDE_existence_condition_l96_9688

variable {M : Type u}
variable (A B C : Set M)

theorem existence_condition :
  (∃ X : Set M, (X ∪ A) \ (X ∩ B) = C) ↔ 
  ((A ∩ Bᶜ ∩ Cᶜ = ∅) ∧ (Aᶜ ∩ B ∩ C = ∅)) := by sorry

end NUMINAMATH_CALUDE_existence_condition_l96_9688


namespace NUMINAMATH_CALUDE_factorization_equality_l96_9666

theorem factorization_equality (a b : ℝ) : (2*a - b)^2 + 8*a*b = (2*a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l96_9666


namespace NUMINAMATH_CALUDE_two_n_squared_lt_three_to_n_l96_9652

theorem two_n_squared_lt_three_to_n (n : ℕ+) : 2 * n.val ^ 2 < 3 ^ n.val := by sorry

end NUMINAMATH_CALUDE_two_n_squared_lt_three_to_n_l96_9652


namespace NUMINAMATH_CALUDE_power_function_decreasing_m_l96_9644

/-- A power function y = ax^b where a and b are constants and x > 0 -/
def isPowerFunction (y : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, x > 0 → y x = a * x ^ b

/-- A decreasing function on (0, +∞) -/
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₂ < f x₁

theorem power_function_decreasing_m (m : ℝ) :
  isPowerFunction (fun x => (m^2 - 2*m - 2) * x^(-4*m - 2)) →
  isDecreasingOn (fun x => (m^2 - 2*m - 2) * x^(-4*m - 2)) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_m_l96_9644


namespace NUMINAMATH_CALUDE_tangent_circles_slope_l96_9613

/-- Definition of circle w1 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 10*x - 20*y - 77 = 0

/-- Definition of circle w2 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 20*y + 193 = 0

/-- Definition of a line y = mx -/
def line (m x y : ℝ) : Prop := y = m * x

/-- Definition of internal tangency -/
def internallyTangent (x y r : ℝ) : Prop := (x - 5)^2 + (y - 10)^2 = (8 - r)^2

/-- Definition of external tangency -/
def externallyTangent (x y r : ℝ) : Prop := (x + 5)^2 + (y - 10)^2 = (r + 12)^2

/-- Main theorem -/
theorem tangent_circles_slope : 
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (m' : ℝ), m' > 0 → 
    (∃ (x y r : ℝ), line m' x y ∧ internallyTangent x y r ∧ externallyTangent x y r) 
    → m' ≥ m) ∧ 
  m^2 = 81/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_slope_l96_9613


namespace NUMINAMATH_CALUDE_second_digit_is_seven_l96_9678

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The second digit of a three-digit number satisfying the given condition is 7 -/
theorem second_digit_is_seven (n : ThreeDigitNumber) :
  100 * n.a + 10 * n.b + n.c - (n.a + n.b + n.c) = 261 → n.b = 7 := by
  sorry

#check second_digit_is_seven

end NUMINAMATH_CALUDE_second_digit_is_seven_l96_9678


namespace NUMINAMATH_CALUDE_max_midpoints_on_circle_l96_9668

/-- A regular n-gon with n ≥ 3 -/
structure RegularNGon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The set of midpoints of all sides and diagonals of a regular n-gon -/
def midpoints (ngon : RegularNGon) : Set (ℝ × ℝ) :=
  sorry

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of points from a set that lie on a given circle -/
def pointsOnCircle (S : Set (ℝ × ℝ)) (c : Circle) : ℕ :=
  sorry

/-- The maximum number of points from a set that lie on any circle -/
def maxPointsOnCircle (S : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem: The maximum number of marked midpoints that lie on the same circle is n -/
theorem max_midpoints_on_circle (ngon : RegularNGon) :
    maxPointsOnCircle (midpoints ngon) = ngon.n :=
  sorry

end NUMINAMATH_CALUDE_max_midpoints_on_circle_l96_9668


namespace NUMINAMATH_CALUDE_stars_per_student_l96_9660

theorem stars_per_student (total_students : ℕ) (total_stars : ℕ) 
  (h1 : total_students = 124) 
  (h2 : total_stars = 372) : 
  total_stars / total_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_stars_per_student_l96_9660


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l96_9658

theorem geometric_sequence_ratio (a : ℝ) (r : ℝ) (h1 : a > 0) (h2 : r > 0) :
  a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r) →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l96_9658


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l96_9682

/-- Calculates the total number of students in three grades given stratified sampling information -/
def totalStudents (sampleSize : ℕ) (firstGradeSample : ℕ) (thirdGradeSample : ℕ) (secondGradeTotal : ℕ) : ℕ :=
  let secondGradeSample := sampleSize - firstGradeSample - thirdGradeSample
  sampleSize * (secondGradeTotal / secondGradeSample)

/-- The total number of students in three grades is 900 given the stratified sampling information -/
theorem stratified_sampling_theorem (sampleSize : ℕ) (firstGradeSample : ℕ) (thirdGradeSample : ℕ) (secondGradeTotal : ℕ)
  (h1 : sampleSize = 45)
  (h2 : firstGradeSample = 20)
  (h3 : thirdGradeSample = 10)
  (h4 : secondGradeTotal = 300) :
  totalStudents sampleSize firstGradeSample thirdGradeSample secondGradeTotal = 900 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l96_9682


namespace NUMINAMATH_CALUDE_gym_floor_area_per_person_l96_9686

theorem gym_floor_area_per_person :
  ∀ (base height : ℝ) (num_students : ℕ),
    base = 9 →
    height = 8 →
    num_students = 24 →
    (base * height) / num_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_gym_floor_area_per_person_l96_9686


namespace NUMINAMATH_CALUDE_P_infimum_and_no_minimum_l96_9641

/-- The function P : ℝ² → ℝ defined by P(X₁, X₂) = X₁² + (1 - X₁X₂)² -/
def P : ℝ × ℝ → ℝ := fun (X₁, X₂) ↦ X₁^2 + (1 - X₁ * X₂)^2

theorem P_infimum_and_no_minimum :
  (∀ ε > 0, ∃ (X₁ X₂ : ℝ), P (X₁, X₂) < ε) ∧
  (¬∃ (X₁ X₂ : ℝ), ∀ (Y₁ Y₂ : ℝ), P (X₁, X₂) ≤ P (Y₁, Y₂)) := by
  sorry

end NUMINAMATH_CALUDE_P_infimum_and_no_minimum_l96_9641


namespace NUMINAMATH_CALUDE_max_sum_property_terms_l96_9655

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a given point -/
def evaluate (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The property that P(n+1) = P(n) + P(n-1) -/
def hasSumProperty (P : QuadraticPolynomial) (n : ℕ) : Prop :=
  evaluate P (n + 1 : ℝ) = evaluate P n + evaluate P (n - 1 : ℝ)

/-- The main theorem: maximum number of terms with sum property is 2 -/
theorem max_sum_property_terms (P : QuadraticPolynomial) :
  (∃ (S : Finset ℕ), (∀ n ∈ S, n ≥ 2 ∧ hasSumProperty P n) ∧ S.card > 2) → False :=
sorry

end NUMINAMATH_CALUDE_max_sum_property_terms_l96_9655
