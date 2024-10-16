import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_l2089_208977

theorem quadratic_roots_greater_than_one (a : ℝ) :
  a ≠ -1 →
  (∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1) ↔
  -16/7 < a ∧ a < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_l2089_208977


namespace NUMINAMATH_CALUDE_no_consecutive_integers_without_real_solutions_l2089_208907

theorem no_consecutive_integers_without_real_solutions :
  ¬ ∃ (b c : ℕ), 
    (c = b + 1) ∧ 
    (∀ x : ℝ, x^2 + b*x + c ≠ 0) ∧ 
    (∀ x : ℝ, x^2 + c*x + b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_without_real_solutions_l2089_208907


namespace NUMINAMATH_CALUDE_rice_purchase_amount_l2089_208917

/-- The price of rice in cents per pound -/
def rice_price : ℚ := 75

/-- The price of beans in cents per pound -/
def bean_price : ℚ := 35

/-- The total weight of rice and beans in pounds -/
def total_weight : ℚ := 30

/-- The total cost in cents -/
def total_cost : ℚ := 1650

/-- The amount of rice purchased in pounds -/
def rice_amount : ℚ := 15

theorem rice_purchase_amount :
  ∃ (bean_amount : ℚ),
    rice_amount + bean_amount = total_weight ∧
    rice_price * rice_amount + bean_price * bean_amount = total_cost :=
sorry

end NUMINAMATH_CALUDE_rice_purchase_amount_l2089_208917


namespace NUMINAMATH_CALUDE_only_dog_owners_l2089_208996

/-- The number of people who own only dogs -/
def D : ℕ := sorry

/-- The number of people who own only cats -/
def C : ℕ := 10

/-- The number of people who own only snakes -/
def S : ℕ := sorry

/-- The number of people who own only cats and dogs -/
def CD : ℕ := 5

/-- The number of people who own only cats and snakes -/
def CS : ℕ := sorry

/-- The number of people who own only dogs and snakes -/
def DS : ℕ := sorry

/-- The number of people who own cats, dogs, and snakes -/
def CDS : ℕ := 3

/-- The total number of pet owners -/
def total_pet_owners : ℕ := 59

/-- The total number of snake owners -/
def total_snake_owners : ℕ := 29

theorem only_dog_owners : D = 15 := by
  have h1 : D + C + S + CD + CS + DS + CDS = total_pet_owners := by sorry
  have h2 : S + CS + DS + CDS = total_snake_owners := by sorry
  sorry

end NUMINAMATH_CALUDE_only_dog_owners_l2089_208996


namespace NUMINAMATH_CALUDE_taxi_growth_equation_l2089_208966

def initial_taxis : ℕ := 11720
def final_taxis : ℕ := 13116
def years : ℕ := 2

theorem taxi_growth_equation (x : ℝ) : 
  (initial_taxis : ℝ) * (1 + x)^years = final_taxis ↔ 
  x = ((final_taxis : ℝ) / initial_taxis)^(1 / years : ℝ) - 1 :=
by sorry

end NUMINAMATH_CALUDE_taxi_growth_equation_l2089_208966


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2089_208989

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2089_208989


namespace NUMINAMATH_CALUDE_cube_division_l2089_208906

theorem cube_division (original_size : ℝ) (num_divisions : ℕ) (num_painted : ℕ) :
  original_size = 3 →
  num_divisions ^ 3 = 27 →
  num_painted = 26 →
  ∃ (smaller_size : ℝ),
    smaller_size = 1 ∧
    num_divisions * smaller_size = original_size :=
by sorry

end NUMINAMATH_CALUDE_cube_division_l2089_208906


namespace NUMINAMATH_CALUDE_joans_cat_kittens_l2089_208980

/-- The number of kittens Joan has now -/
def total_kittens : ℕ := 10

/-- The number of kittens Joan got from her friends -/
def kittens_from_friends : ℕ := 2

/-- The number of kittens Joan's cat had -/
def cat_kittens : ℕ := total_kittens - kittens_from_friends

theorem joans_cat_kittens : cat_kittens = 8 := by
  sorry

end NUMINAMATH_CALUDE_joans_cat_kittens_l2089_208980


namespace NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l2089_208956

theorem sum_of_ten_consecutive_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ ∃ m : ℕ, 10 * n^2 + 10 * n + 85 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l2089_208956


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2089_208905

theorem quadratic_equation_solution :
  let f (x : ℂ) := 2 * (5 * x^2 + 4 * x + 3) - 6
  let g (x : ℂ) := -3 * (2 - 4 * x)
  ∀ x : ℂ, f x = g x ↔ x = (1 + Complex.I * Real.sqrt 14) / 5 ∨ x = (1 - Complex.I * Real.sqrt 14) / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2089_208905


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2089_208927

theorem rectangle_area_increase 
  (l w : ℝ) 
  (hl : l > 0) 
  (hw : w > 0) : 
  let new_length := 1.3 * l
  let new_width := 1.15 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.495 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2089_208927


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2089_208999

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (1 + i)
  Complex.im z = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2089_208999


namespace NUMINAMATH_CALUDE_solve_for_a_l2089_208909

theorem solve_for_a (x a : ℚ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2089_208909


namespace NUMINAMATH_CALUDE_selection_methods_l2089_208948

theorem selection_methods (boys girls : ℕ) (tasks : ℕ) : 
  boys = 5 → girls = 4 → tasks = 3 →
  (Nat.choose boys 2 * Nat.choose girls 1 + Nat.choose boys 1 * Nat.choose girls 2) * Nat.factorial tasks = 420 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_l2089_208948


namespace NUMINAMATH_CALUDE_tournament_games_played_23_teams_l2089_208990

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- Calculates the number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties,
    22 games must be played before a winner can be declared. -/
theorem tournament_games_played_23_teams :
  ∀ (t : Tournament), t.num_teams = 23 → t.no_ties = true →
  games_played t = 22 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_played_23_teams_l2089_208990


namespace NUMINAMATH_CALUDE_C_younger_than_A_l2089_208957

-- Define variables for ages
variable (A B C : ℕ)

-- Define the condition from the problem
def age_condition (A B C : ℕ) : Prop := A + B = B + C + 12

-- Theorem to prove
theorem C_younger_than_A (h : age_condition A B C) : A = C + 12 := by
  sorry

end NUMINAMATH_CALUDE_C_younger_than_A_l2089_208957


namespace NUMINAMATH_CALUDE_cubic_monomial_exists_l2089_208922

/-- A cubic monomial with variables x and y and a negative coefficient exists. -/
theorem cubic_monomial_exists : ∃ (a : ℝ) (i j : ℕ), 
  a < 0 ∧ i + j = 3 ∧ (λ (x y : ℝ) => a * x^i * y^j) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_monomial_exists_l2089_208922


namespace NUMINAMATH_CALUDE_smallest_band_size_l2089_208982

theorem smallest_band_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 6 = 5 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 6 ∧ 
  (∀ m : ℕ, m > 0 → m % 6 = 5 → m % 5 = 4 → m % 7 = 6 → m ≥ n) ∧
  n = 119 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_band_size_l2089_208982


namespace NUMINAMATH_CALUDE_triangle_formation_l2089_208950

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (5, 6, 10), (1, 2, 3)]

theorem triangle_formation :
  ∃! (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l2089_208950


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2089_208946

-- Define the parabola and hyperbola
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf : ∃ (x y : ℝ), parabola x y ∧ hyperbola x y a b ∧ (x, y) ≠ parabola_focus) 
  (hperp : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola x₁ y₁ ∧ hyperbola x₁ y₁ a b ∧
    parabola x₂ y₂ ∧ hyperbola x₂ y₂ a b ∧
    (x₁ + x₂) * (1 - x₁) + (y₁ + y₂) * (-y₁) = 0) :
  2 * a = 2 * Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2089_208946


namespace NUMINAMATH_CALUDE_rational_sums_and_products_l2089_208991

-- Define the property of being rational
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Main theorem
theorem rational_sums_and_products (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : IsRational (x * y))
  (hyz : IsRational (y * z))
  (hzx : IsRational (z * x)) :
  (IsRational (x^2 + y^2 + z^2)) ∧
  (IsRational (x^3 + y^3 + z^3) → IsRational x ∧ IsRational y ∧ IsRational z) := by
  sorry


end NUMINAMATH_CALUDE_rational_sums_and_products_l2089_208991


namespace NUMINAMATH_CALUDE_number_and_remainder_l2089_208963

theorem number_and_remainder : ∃ x : ℤ, 2 * x - 3 = 7 ∧ x % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_and_remainder_l2089_208963


namespace NUMINAMATH_CALUDE_statement_equivalence_l2089_208931

theorem statement_equivalence (P Q : Prop) :
  (Q → ¬P) ↔ (P → ¬Q) := by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l2089_208931


namespace NUMINAMATH_CALUDE_new_average_after_dropping_lowest_l2089_208923

def calculate_new_average (num_tests : ℕ) (original_average : ℚ) (lowest_score : ℚ) : ℚ :=
  ((num_tests : ℚ) * original_average - lowest_score) / ((num_tests : ℚ) - 1)

theorem new_average_after_dropping_lowest
  (num_tests : ℕ)
  (original_average : ℚ)
  (lowest_score : ℚ)
  (h1 : num_tests = 4)
  (h2 : original_average = 35)
  (h3 : lowest_score = 20) :
  calculate_new_average num_tests original_average lowest_score = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_after_dropping_lowest_l2089_208923


namespace NUMINAMATH_CALUDE_composition_ratio_l2089_208951

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 1
def g (x : ℝ) : ℝ := 2 * x + 5

-- State the theorem
theorem composition_ratio :
  (g (f (g 3))) / (f (g (f 3))) = 69 / 206 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l2089_208951


namespace NUMINAMATH_CALUDE_alison_initial_stamps_l2089_208937

/-- Represents the number of stamps each person has -/
structure StampCollection where
  anna : ℕ
  alison : ℕ
  jeff : ℕ

/-- The initial stamp collection -/
def initial : StampCollection where
  anna := 37
  alison := 26  -- This is what we want to prove
  jeff := 31

/-- The final stamp collection after exchanges -/
def final : StampCollection where
  anna := 50
  alison := initial.alison / 2
  jeff := initial.jeff + 1

theorem alison_initial_stamps :
  initial.anna + initial.alison / 2 = final.anna := by sorry

#check alison_initial_stamps

end NUMINAMATH_CALUDE_alison_initial_stamps_l2089_208937


namespace NUMINAMATH_CALUDE_geometric_roots_poly_n_value_l2089_208994

/-- A polynomial of degree 4 with four distinct real roots in geometric progression -/
structure GeometricRootsPoly where
  m : ℝ
  n : ℝ
  p : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  geometric : ∃ (a r : ℝ), ∀ i, roots i = a * r ^ i.val
  is_root : ∀ i, roots i ^ 4 + m * roots i ^ 3 + n * roots i ^ 2 + p * roots i + 256 = 0

/-- The theorem stating that n = -32 for such polynomials -/
theorem geometric_roots_poly_n_value (poly : GeometricRootsPoly) : poly.n = -32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_roots_poly_n_value_l2089_208994


namespace NUMINAMATH_CALUDE_system_solution_implies_m_zero_l2089_208900

theorem system_solution_implies_m_zero (x y m : ℝ) :
  (2 * x + 3 * y = 4) →
  (3 * x + 2 * y = 2 * m - 3) →
  (x + y = 1 / 5) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_m_zero_l2089_208900


namespace NUMINAMATH_CALUDE_correct_withdrawal_amount_withdrawal_amount_2016_l2089_208969

/-- Calculates the amount that can be withdrawn after a given number of years
    for a fixed-term deposit with annual compound interest. -/
def withdrawal_amount (initial_deposit : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  initial_deposit * (1 + interest_rate) ^ years

/-- Theorem stating the correct withdrawal amount after 14 years -/
theorem correct_withdrawal_amount (a : ℝ) (r : ℝ) :
  withdrawal_amount a r 14 = a * (1 + r)^14 := by
  sorry

/-- The number of years between January 1, 2002 and January 1, 2016 -/
def years_between_2002_and_2016 : ℕ := 14

/-- Theorem proving the correct withdrawal amount on January 1, 2016 -/
theorem withdrawal_amount_2016 (a : ℝ) (r : ℝ) :
  withdrawal_amount a r years_between_2002_and_2016 = a * (1 + r)^14 := by
  sorry

end NUMINAMATH_CALUDE_correct_withdrawal_amount_withdrawal_amount_2016_l2089_208969


namespace NUMINAMATH_CALUDE_daniels_initial_noodles_l2089_208916

/-- Represents the number of noodles Daniel had initially -/
def initial_noodles : ℕ := sorry

/-- Represents the number of noodles Daniel gave away -/
def noodles_given_away : ℕ := 12

/-- Represents the number of noodles Daniel has now -/
def remaining_noodles : ℕ := 54

/-- Theorem stating that Daniel's initial number of noodles was 66 -/
theorem daniels_initial_noodles : 
  initial_noodles = noodles_given_away + remaining_noodles := by
  sorry

end NUMINAMATH_CALUDE_daniels_initial_noodles_l2089_208916


namespace NUMINAMATH_CALUDE_tank_emptying_time_l2089_208926

/-- Represents the time (in minutes) it takes to empty a water tank -/
def empty_tank_time (initial_fill : ℚ) (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  initial_fill / (empty_rate - fill_rate)

theorem tank_emptying_time :
  let initial_fill : ℚ := 4/5
  let fill_pipe_rate : ℚ := 1/10
  let empty_pipe_rate : ℚ := 1/6
  empty_tank_time initial_fill fill_pipe_rate empty_pipe_rate = 12 := by
sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l2089_208926


namespace NUMINAMATH_CALUDE_point_line_plane_relation_l2089_208986

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (lies_in : Line → Plane → Prop)

-- Define the set membership and subset relations
variable (mem : Point → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem point_line_plane_relation 
  (A : Point) (b : Line) (β : Plane) 
  (h1 : lies_on A b) 
  (h2 : lies_in b β) :
  mem A b ∧ subset b β := by
  sorry

end NUMINAMATH_CALUDE_point_line_plane_relation_l2089_208986


namespace NUMINAMATH_CALUDE_number_division_multiplication_l2089_208985

theorem number_division_multiplication (x : ℚ) : x = 5.5 → (x / 6) * 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_division_multiplication_l2089_208985


namespace NUMINAMATH_CALUDE_x_24_equals_one_l2089_208964

theorem x_24_equals_one (x : ℂ) (h : x + 1/x = -Real.sqrt 3) : x^24 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_24_equals_one_l2089_208964


namespace NUMINAMATH_CALUDE_complex_magnitude_l2089_208947

theorem complex_magnitude (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = 2*i + (9 - 3*i) / (1 + i) →
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2089_208947


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l2089_208983

def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem age_ratio_in_two_years :
  man_age_in_two_years / son_age_in_two_years = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l2089_208983


namespace NUMINAMATH_CALUDE_function_equivalence_l2089_208979

theorem function_equivalence (x : ℝ) : (x^3 + x) / (x^2 + 1) = x := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l2089_208979


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2089_208919

theorem triangle_abc_properties (A B C : Real) (h : Real) :
  A + B + C = Real.pi →
  A + B = 3 * C →
  2 * Real.sin (A - C) = Real.sin B →
  h * 5 / 2 = Real.sin C * Real.sin A * Real.sin B * 25 →
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧ h = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2089_208919


namespace NUMINAMATH_CALUDE_indeterminate_magnitude_l2089_208961

/-- Given two approximate numbers A and B, prove that their relative magnitude cannot be determined. -/
theorem indeterminate_magnitude (A B : ℝ) (hA : 3.55 ≤ A ∧ A < 3.65) (hB : 3.595 ≤ B ∧ B < 3.605) :
  ¬(A > B ∨ A = B ∨ A < B) := by
  sorry

#check indeterminate_magnitude

end NUMINAMATH_CALUDE_indeterminate_magnitude_l2089_208961


namespace NUMINAMATH_CALUDE_max_area_cyclic_quadrilateral_l2089_208974

/-- The maximum area of a cyclic quadrilateral with side lengths 1, 4, 7, and 8 is 18 -/
theorem max_area_cyclic_quadrilateral :
  let a : ℝ := 1
  let b : ℝ := 4
  let c : ℝ := 7
  let d : ℝ := 8
  let s : ℝ := (a + b + c + d) / 2
  let area : ℝ := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))
  area = 18 := by sorry

end NUMINAMATH_CALUDE_max_area_cyclic_quadrilateral_l2089_208974


namespace NUMINAMATH_CALUDE_fish_length_theorem_l2089_208987

theorem fish_length_theorem (x : ℚ) :
  (1 / 3 : ℚ) * x + (1 / 4 : ℚ) * x + 3 = x → x = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fish_length_theorem_l2089_208987


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l2089_208918

def U : Finset ℕ := {1, 3, 5, 9}
def A : Finset ℕ := {1, 3, 9}
def B : Finset ℕ := {1, 9}

theorem complement_of_union_equals_five : (U \ (A ∪ B)) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l2089_208918


namespace NUMINAMATH_CALUDE_lcm_product_implies_hcf_l2089_208978

theorem lcm_product_implies_hcf (x y : ℕ+) 
  (h1 : Nat.lcm x y = 600) 
  (h2 : x * y = 18000) : 
  Nat.gcd x y = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_implies_hcf_l2089_208978


namespace NUMINAMATH_CALUDE_summer_grain_scientific_notation_l2089_208933

def summer_grain_production : ℝ := 11534000000

/-- Converts a number to scientific notation with a specified number of significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem summer_grain_scientific_notation :
  to_scientific_notation summer_grain_production 4 = (1.153, 8) :=
sorry

end NUMINAMATH_CALUDE_summer_grain_scientific_notation_l2089_208933


namespace NUMINAMATH_CALUDE_assignments_count_l2089_208976

/-- The number of assignments graded per hour initially -/
def initial_rate : ℕ := 6

/-- The number of assignments graded per hour after the change -/
def changed_rate : ℕ := 8

/-- The number of hours spent grading at the initial rate -/
def initial_hours : ℕ := 2

/-- The number of hours saved compared to the original plan -/
def hours_saved : ℕ := 3

/-- The total number of assignments in the batch -/
def total_assignments : ℕ := 84

/-- Theorem stating that the total number of assignments is 84 -/
theorem assignments_count :
  ∃ (x : ℕ), 
    (initial_rate * x = total_assignments) ∧ 
    (initial_rate * initial_hours + changed_rate * (x - initial_hours - hours_saved) = total_assignments) := by
  sorry

end NUMINAMATH_CALUDE_assignments_count_l2089_208976


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2089_208902

def solution_set : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}

theorem inequality_solution_set :
  ∀ x : ℝ, (x + 2 ≠ 0) → ((x - 3) / (x + 2) ≤ 0 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2089_208902


namespace NUMINAMATH_CALUDE_third_altitude_values_l2089_208984

/-- Triangle with two known altitudes and an integer third altitude -/
structure TriangleWithAltitudes where
  /-- First known altitude -/
  h₁ : ℝ
  /-- Second known altitude -/
  h₂ : ℝ
  /-- Third altitude (integer) -/
  h₃ : ℤ
  /-- Condition that first altitude is 4 -/
  h₁_eq : h₁ = 4
  /-- Condition that second altitude is 12 -/
  h₂_eq : h₂ = 12

/-- Theorem stating the possible values of the third altitude -/
theorem third_altitude_values (t : TriangleWithAltitudes) :
  t.h₃ = 4 ∨ t.h₃ = 5 :=
sorry

end NUMINAMATH_CALUDE_third_altitude_values_l2089_208984


namespace NUMINAMATH_CALUDE_one_match_among_withdrawn_l2089_208973

/-- Represents a table tennis tournament with special conditions -/
structure TableTennisTournament where
  n : ℕ  -- Total number of players
  total_matches : ℕ  -- Total number of matches played
  withdrawn_players : ℕ  -- Number of players who withdrew
  matches_per_withdrawn : ℕ  -- Number of matches each withdrawn player played
  hwithdrawncond : withdrawn_players = 3
  hmatchescond : matches_per_withdrawn = 2
  htotalcond : total_matches = 50

/-- The number of matches played among the withdrawn players -/
def matches_among_withdrawn (t : TableTennisTournament) : ℕ := 
  (t.withdrawn_players * t.matches_per_withdrawn - 
   t.total_matches + (t.n - t.withdrawn_players).choose 2) / 2

/-- Theorem stating that exactly one match was played among the withdrawn players -/
theorem one_match_among_withdrawn (t : TableTennisTournament) : 
  matches_among_withdrawn t = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_match_among_withdrawn_l2089_208973


namespace NUMINAMATH_CALUDE_pyramid_division_volumes_l2089_208992

/-- Right quadrangular pyramid with inscribed prism and dividing plane -/
structure PyramidWithPrism where
  /-- Side length of the pyramid's base -/
  a : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- Side length of the prism's base -/
  b : ℝ
  /-- Height of the prism -/
  h₀ : ℝ
  /-- Condition: The side length of the pyramid's base is 8√2 -/
  ha : a = 8 * Real.sqrt 2
  /-- Condition: The height of the pyramid is 4 -/
  hh : h = 4
  /-- Condition: The side length of the prism's base is 2 -/
  hb : b = 2
  /-- Condition: The height of the prism is 1 -/
  hh₀ : h₀ = 1

/-- Theorem stating the volumes of the parts divided by plane γ -/
theorem pyramid_division_volumes (p : PyramidWithPrism) :
  ∃ (v₁ v₂ : ℝ), v₁ = 512 / 15 ∧ v₂ = 2048 / 15 ∧
  v₁ + v₂ = (1 / 3) * p.a^2 * p.h :=
sorry

end NUMINAMATH_CALUDE_pyramid_division_volumes_l2089_208992


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2089_208970

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ + a₄ = 120 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2089_208970


namespace NUMINAMATH_CALUDE_staircase_steps_l2089_208945

/-- Represents the number of toothpicks used in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

/-- The number of toothpicks used in a 3-step staircase -/
def three_step_toothpicks : ℕ := 27

/-- The target number of toothpicks -/
def target_toothpicks : ℕ := 270

theorem staircase_steps :
  ∃ (n : ℕ), toothpicks n = target_toothpicks ∧ n = 12 :=
sorry

end NUMINAMATH_CALUDE_staircase_steps_l2089_208945


namespace NUMINAMATH_CALUDE_calculation_product_l2089_208959

theorem calculation_product (x : ℤ) (h : x - 9 - 12 = 24) : (x + 8 - 11) * 24 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_calculation_product_l2089_208959


namespace NUMINAMATH_CALUDE_least_perimeter_of_triangle_l2089_208993

theorem least_perimeter_of_triangle (a b c : ℕ) : 
  a = 24 → b = 51 → c > 0 → a + b > c → a + c > b → b + c > a → 
  ∀ x : ℕ, (x > 0 ∧ a + b > x ∧ a + x > b ∧ b + x > a) → a + b + c ≤ a + b + x →
  a + b + c = 103 := by sorry

end NUMINAMATH_CALUDE_least_perimeter_of_triangle_l2089_208993


namespace NUMINAMATH_CALUDE_f_extrema_l2089_208935

def f (p q x : ℝ) : ℝ := x^3 - p*x^2 - q*x

theorem f_extrema (p q : ℝ) :
  (f p q 1 = 0) →
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, f p q x ≤ f p q x₁) ∧ (∀ x : ℝ, f p q x ≥ f p q x₂) ∧ 
                 (f p q x₁ = 4/27) ∧ (f p q x₂ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l2089_208935


namespace NUMINAMATH_CALUDE_employee_pays_216_l2089_208975

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the store markup percentage
def store_markup : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount : ℝ := 0.10

-- Calculate the retail price
def retail_price : ℝ := wholesale_cost * (1 + store_markup)

-- Calculate the employee's final price
def employee_price : ℝ := retail_price * (1 - employee_discount)

-- Theorem to prove
theorem employee_pays_216 : employee_price = 216 := by sorry

end NUMINAMATH_CALUDE_employee_pays_216_l2089_208975


namespace NUMINAMATH_CALUDE_complex_number_real_part_l2089_208958

theorem complex_number_real_part : 
  ∀ (z : ℂ) (a : ℝ), 
  (z / (2 + a * Complex.I) = 2 / (1 + Complex.I)) → 
  (z.im = -3) → 
  (z.re = 1) := by
sorry

end NUMINAMATH_CALUDE_complex_number_real_part_l2089_208958


namespace NUMINAMATH_CALUDE_min_wires_for_unit_cube_l2089_208972

/-- Represents a piece of wire with a given length -/
structure Wire where
  length : ℕ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℕ
  numEdges : ℕ := 12
  numVertices : ℕ := 8

def availableWires : List Wire := [
  { length := 1 },
  { length := 2 },
  { length := 3 },
  { length := 4 },
  { length := 5 },
  { length := 6 },
  { length := 7 }
]

def targetCube : Cube := { edgeLength := 1 }

/-- Returns the minimum number of wire pieces needed to form the cube -/
def minWiresForCube (wires : List Wire) (cube : Cube) : ℕ := sorry

theorem min_wires_for_unit_cube :
  minWiresForCube availableWires targetCube = 4 := by sorry

end NUMINAMATH_CALUDE_min_wires_for_unit_cube_l2089_208972


namespace NUMINAMATH_CALUDE_celia_receives_171_spiders_l2089_208995

/-- Represents the number of stickers of each type Célia has -/
structure StickerCount where
  butterfly : ℕ
  shark : ℕ
  snake : ℕ
  parakeet : ℕ
  monkey : ℕ

/-- Represents the conversion rates between different types of stickers -/
structure ConversionRates where
  butterfly_to_shark : ℕ
  snake_to_parakeet : ℕ
  monkey_to_spider : ℕ
  parakeet_to_spider : ℕ
  shark_to_parakeet : ℕ

/-- Calculates the total number of spider stickers Célia can receive -/
def total_spider_stickers (count : StickerCount) (rates : ConversionRates) : ℕ :=
  sorry

/-- Theorem stating that Célia can receive 171 spider stickers -/
theorem celia_receives_171_spiders (count : StickerCount) (rates : ConversionRates) 
    (h1 : count.butterfly = 4)
    (h2 : count.shark = 5)
    (h3 : count.snake = 3)
    (h4 : count.parakeet = 6)
    (h5 : count.monkey = 6)
    (h6 : rates.butterfly_to_shark = 3)
    (h7 : rates.snake_to_parakeet = 3)
    (h8 : rates.monkey_to_spider = 4)
    (h9 : rates.parakeet_to_spider = 3)
    (h10 : rates.shark_to_parakeet = 2) :
    total_spider_stickers count rates = 171 :=
  sorry

end NUMINAMATH_CALUDE_celia_receives_171_spiders_l2089_208995


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l2089_208998

theorem apple_pear_equivalence :
  ∀ (apple_value pear_value : ℚ),
    (3/4 * 16 * apple_value = 12 * pear_value) →
    (2/3 * 9 * apple_value = 6 * pear_value) := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l2089_208998


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2089_208949

def b (n : ℕ) : ℕ := (n + 2).factorial - n^2

theorem max_gcd_consecutive_terms :
  (∃ k : ℕ, Nat.gcd (b k) (b (k + 1)) = 5) ∧
  (∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2089_208949


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l2089_208971

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = focus + t • (Q - focus) ∨ Q = focus + t • (P - focus)

-- Define the theorem
theorem parabola_intersection_length 
  (P Q : ℝ × ℝ) 
  (h_P : parabola P.1 P.2) 
  (h_Q : parabola Q.1 Q.2) 
  (h_line : line_through_focus P Q) 
  (h_sum : P.1 + Q.1 = 9) : 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 11 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l2089_208971


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2089_208939

def is_valid_representation (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 9

def has_at_least_two_of_each (n : ℕ) : Prop :=
  (n.digits 10).count 4 ≥ 2 ∧ (n.digits 10).count 9 ≥ 2

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 4 = 0 ∧
    m % 9 = 0 ∧
    is_valid_representation m ∧
    has_at_least_two_of_each m ∧
    (∀ k : ℕ, k > 0 ∧ k % 4 = 0 ∧ k % 9 = 0 ∧ is_valid_representation k ∧ has_at_least_two_of_each k → m ≤ k) ∧
    last_four_digits m = 9494 :=
  by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2089_208939


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2089_208924

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x + 2) * (10*x + 6) * (5*x + 5) = 960 * k) ∧
  (∀ (m : ℤ), m > 960 → ¬(∀ (y : ℤ), Odd y → ∃ (l : ℤ), (10*y + 2) * (10*y + 6) * (5*y + 5) = m * l)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2089_208924


namespace NUMINAMATH_CALUDE_simplify_expression_l2089_208934

theorem simplify_expression (a : ℝ) : (1 : ℝ) * (2 * a) * (3 * a^2) * (4 * a^3) * (5 * a^4) = 120 * a^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2089_208934


namespace NUMINAMATH_CALUDE_greatest_c_value_l2089_208941

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_c_value_l2089_208941


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2089_208901

/-- The volume of a right rectangular prism with face areas 6, 8, and 12 square inches is 24 cubic inches. -/
theorem right_rectangular_prism_volume (l w h : ℝ) 
  (area1 : l * w = 6)
  (area2 : w * h = 8)
  (area3 : l * h = 12) :
  l * w * h = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2089_208901


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_negative_300_degrees_to_radians_l2089_208904

theorem degree_to_radian_conversion (angle_in_degrees : ℝ) : 
  angle_in_degrees * (π / 180) = angle_in_degrees * π / 180 := by sorry

theorem negative_300_degrees_to_radians : 
  -300 * (π / 180) = -5 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_negative_300_degrees_to_radians_l2089_208904


namespace NUMINAMATH_CALUDE_square_difference_and_product_l2089_208914

theorem square_difference_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 81)
  (h2 : x * y = 15) : 
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_and_product_l2089_208914


namespace NUMINAMATH_CALUDE_difference_is_integer_l2089_208932

/-- A linear function from ℝ to ℝ -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  map : ℝ → ℝ := fun x ↦ a * x + b
  increasing : 0 < a

/-- Two linear functions with the integer property -/
structure IntegerPropertyFunctions where
  f : LinearFunction
  g : LinearFunction
  integer_property : ∀ x : ℝ, Int.floor (f.map x) = f.map x ↔ Int.floor (g.map x) = g.map x

/-- The main theorem -/
theorem difference_is_integer (funcs : IntegerPropertyFunctions) :
  ∀ x : ℝ, ∃ n : ℤ, funcs.f.map x - funcs.g.map x = n :=
sorry

end NUMINAMATH_CALUDE_difference_is_integer_l2089_208932


namespace NUMINAMATH_CALUDE_problem_solution_l2089_208981

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 4| + |x - 4|

-- Theorem statement
theorem problem_solution :
  -- Part 1: Solution set of f(x) ≥ 10
  (∀ x, f x ≥ 10 ↔ x ∈ Set.Iic (-10/3) ∪ Set.Ici 2) ∧
  -- Part 2: Minimum value of f(x) is 6
  (∃ x, f x = 6 ∧ ∀ y, f y ≥ f x) ∧
  -- Part 3: Inequality for positive real numbers a, b, c
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 →
    1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2089_208981


namespace NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l2089_208930

theorem hua_luogeng_birthday_factorization :
  (1163 : ℕ).Prime ∧ ¬(16424 : ℕ).Prime :=
by
  have h : 19101112 = 1163 * 16424 := by rfl
  sorry

end NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l2089_208930


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l2089_208943

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := num_points.choose 2

/-- The number of ways to select the required number of chords -/
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

/-- The number of ways to choose points that form a convex quadrilateral -/
def convex_quadrilaterals : ℕ := num_points.choose 4

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quadrilaterals / ways_to_select_chords

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l2089_208943


namespace NUMINAMATH_CALUDE_sum_of_inscribed_circle_areas_l2089_208929

/-- Given a triangle ABC with sides a, b, c, and an inscribed circle of radius r,
    prove that the sum of the areas of four inscribed circles
    (one in the original triangle and three in the smaller triangles formed by
    tangents parallel to the sides) is equal to π r² · (a² + b² + c²) / s²,
    where s is the semi-perimeter of the triangle. -/
theorem sum_of_inscribed_circle_areas
  (a b c r : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (s : ℝ) (h_s : s = (a + b + c) / 2)
  (h_inradius : r = s / ((a + b + c) / 2)) :
  let original_circle_area := π * r^2
  let smaller_circles_area := π * r^2 * ((s - a)^2 + (s - b)^2 + (s - c)^2) / s^2
  original_circle_area + smaller_circles_area = π * r^2 * (a^2 + b^2 + c^2) / s^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_circle_areas_l2089_208929


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l2089_208938

/-- Given two right circular cylinders with radii r₁ and r₂, and heights h₁ and h₂,
    prove that if the volume of the second is twice the first and r₂ = 1.1 * r₁,
    then h₂ ≈ 1.65 * h₁ -/
theorem cylinder_height_relationship (r₁ r₂ h₁ h₂ : ℝ) 
  (volume_relation : π * r₂^2 * h₂ = 2 * π * r₁^2 * h₁)
  (radius_relation : r₂ = 1.1 * r₁)
  (h₁_pos : h₁ > 0) (r₁_pos : r₁ > 0) :
  ∃ ε > 0, abs (h₂ / h₁ - 200 / 121) < ε :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l2089_208938


namespace NUMINAMATH_CALUDE_digit_2023_of_11_26_l2089_208968

/-- The repeating decimal representation of 11/26 -/
def repeating_decimal : List Nat := [4, 2, 3, 0, 7, 6]

/-- The length of the repeating decimal -/
def repeat_length : Nat := repeating_decimal.length

theorem digit_2023_of_11_26 : 
  -- The 2023rd digit past the decimal point in 11/26
  List.get! repeating_decimal ((2023 - 1) % repeat_length) = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_2023_of_11_26_l2089_208968


namespace NUMINAMATH_CALUDE_g_derivative_at_5_l2089_208915

-- Define the function g
def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

-- State the theorem
theorem g_derivative_at_5 : 
  (deriv g) 5 = 26 := by sorry

end NUMINAMATH_CALUDE_g_derivative_at_5_l2089_208915


namespace NUMINAMATH_CALUDE_train_length_l2089_208911

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 25 → speed * time * (1000 / 3600) = 833.25 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2089_208911


namespace NUMINAMATH_CALUDE_parking_savings_l2089_208967

-- Define the weekly and monthly rental rates
def weekly_rate : ℕ := 10
def monthly_rate : ℕ := 42

-- Define the number of weeks and months in a year
def weeks_per_year : ℕ := 52
def months_per_year : ℕ := 12

-- Define the yearly cost for weekly and monthly rentals
def yearly_cost_weekly : ℕ := weekly_rate * weeks_per_year
def yearly_cost_monthly : ℕ := monthly_rate * months_per_year

-- Theorem: The difference in yearly cost between weekly and monthly rentals is $16
theorem parking_savings : yearly_cost_weekly - yearly_cost_monthly = 16 := by
  sorry

end NUMINAMATH_CALUDE_parking_savings_l2089_208967


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2089_208920

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

-- Define the translation
def translate_left : ℝ := 3
def translate_up : ℝ := 2

-- Define parabola C after translation
def parabola_C (x : ℝ) : ℝ := original_parabola (x + translate_left) + translate_up

-- Define the symmetric parabola
def symmetric_parabola (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

-- Theorem statement
theorem parabola_symmetry :
  ∀ x : ℝ, parabola_C (-x) = symmetric_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2089_208920


namespace NUMINAMATH_CALUDE_triangular_pyramid_surface_area_l2089_208960

/-- A triangular pyramid with given base and side areas -/
structure TriangularPyramid where
  base_area : ℝ
  side_area : ℝ

/-- The surface area of a triangular pyramid -/
def surface_area (tp : TriangularPyramid) : ℝ :=
  tp.base_area + 3 * tp.side_area

/-- Theorem: The surface area of a triangular pyramid with base area 3 and side area 6 is 21 -/
theorem triangular_pyramid_surface_area :
  ∃ (tp : TriangularPyramid), tp.base_area = 3 ∧ tp.side_area = 6 ∧ surface_area tp = 21 := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_surface_area_l2089_208960


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l2089_208962

/-- Jake's current weight in pounds -/
def jakes_weight : ℕ := 196

/-- Jake's sister's weight in pounds -/
def sisters_weight : ℕ := (jakes_weight - 8) / 2

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℕ := jakes_weight + sisters_weight

/-- Theorem stating that the combined weight of Jake and his sister is 290 pounds -/
theorem jake_and_sister_weight : combined_weight = 290 := by
  sorry

/-- Lemma stating that if Jake loses 8 pounds, he will weigh twice as much as his sister -/
lemma jake_twice_sister_weight : jakes_weight - 8 = 2 * sisters_weight := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l2089_208962


namespace NUMINAMATH_CALUDE_matrix_equation_implies_even_dimension_l2089_208942

theorem matrix_equation_implies_even_dimension (n : ℕ+) :
  (∃ (A B : Matrix (Fin n) (Fin n) ℝ), 
    Matrix.det A ≠ 0 ∧ 
    Matrix.det B ≠ 0 ∧ 
    A * B - B * A = B ^ 2 * A) → 
  Even n := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_implies_even_dimension_l2089_208942


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l2089_208910

theorem greatest_three_digit_number : ∃ n : ℕ, 
  (n ≤ 999) ∧ 
  (n ≥ 100) ∧ 
  (∃ k : ℕ, n = 8 * k - 1) ∧ 
  (∃ m : ℕ, n = 7 * m + 4) ∧ 
  (∀ x : ℕ, x ≤ 999 ∧ x ≥ 100 ∧ (∃ a : ℕ, x = 8 * a - 1) ∧ (∃ b : ℕ, x = 7 * b + 4) → x ≤ n) ∧
  n = 967 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l2089_208910


namespace NUMINAMATH_CALUDE_homework_problem_l2089_208921

theorem homework_problem (p t : ℕ) : 
  p > 0 → 
  t > 0 → 
  p ≥ 15 → 
  3 * p - 5 ≥ 20 → 
  p * t = (3 * p - 5) * (t - 3) → 
  p * t = 100 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l2089_208921


namespace NUMINAMATH_CALUDE_min_value_expression_l2089_208965

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 5) : x^2 + y^2 + 2*z^2 - x^2*y^2*z ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2089_208965


namespace NUMINAMATH_CALUDE_iphone_case_cost_percentage_l2089_208940

/-- Proves that the percentage of the case cost relative to the phone cost is 20% --/
theorem iphone_case_cost_percentage :
  let phone_cost : ℝ := 1000
  let monthly_contract_cost : ℝ := 200
  let case_cost_percentage : ℝ → ℝ := λ x => x / 100 * phone_cost
  let headphones_cost : ℝ → ℝ := λ x => (1 / 2) * case_cost_percentage x
  let total_yearly_cost : ℝ → ℝ := λ x => 
    phone_cost + 12 * monthly_contract_cost + case_cost_percentage x + headphones_cost x
  ∃ x : ℝ, total_yearly_cost x = 3700 ∧ x = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_iphone_case_cost_percentage_l2089_208940


namespace NUMINAMATH_CALUDE_compound_line_chart_optimal_l2089_208928

/-- Represents different types of statistical charts -/
inductive StatisticalChart
  | Bar
  | Pie
  | Line
  | Scatter
  | CompoundLine

/-- Represents the requirements for the chart -/
structure ChartRequirements where
  numStudents : Nat
  showComparison : Bool
  showChangesOverTime : Bool

/-- Determines if a chart type is optimal for given requirements -/
def isOptimalChart (chart : StatisticalChart) (req : ChartRequirements) : Prop :=
  chart = StatisticalChart.CompoundLine ∧
  req.numStudents = 2 ∧
  req.showComparison = true ∧
  req.showChangesOverTime = true

/-- Theorem stating that a compound line chart is optimal for the given scenario -/
theorem compound_line_chart_optimal (req : ChartRequirements) :
  req.numStudents = 2 →
  req.showComparison = true →
  req.showChangesOverTime = true →
  isOptimalChart StatisticalChart.CompoundLine req :=
by sorry

end NUMINAMATH_CALUDE_compound_line_chart_optimal_l2089_208928


namespace NUMINAMATH_CALUDE_equation_solutions_l2089_208953

theorem equation_solutions :
  (∃ x : ℚ, (3 - x) / (x + 4) = 1 / 2 ∧ x = 2 / 3) ∧
  (∃ x : ℚ, x / (x - 1) - 2 * x / (3 * x - 3) = 1 ∧ x = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2089_208953


namespace NUMINAMATH_CALUDE_salary_decrease_l2089_208903

theorem salary_decrease (initial_salary : ℝ) (cut1 cut2 cut3 : ℝ) 
  (h1 : cut1 = 0.08) (h2 : cut2 = 0.14) (h3 : cut3 = 0.18) :
  1 - (1 - cut1) * (1 - cut2) * (1 - cut3) = 1 - (0.92 * 0.86 * 0.82) := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_l2089_208903


namespace NUMINAMATH_CALUDE_unique_solution_system_l2089_208936

theorem unique_solution_system (x y z : ℝ) : 
  (x + y = 2 ∧ x * y - z^2 = 1) → (x = 1 ∧ y = 1 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2089_208936


namespace NUMINAMATH_CALUDE_katrina_andy_earnings_l2089_208997

theorem katrina_andy_earnings (t : ℝ) : 
  (t - 4 : ℝ) * (3 * t - 10) = (3 * t - 12) * (t - 3) → t = 4 := by
  sorry

end NUMINAMATH_CALUDE_katrina_andy_earnings_l2089_208997


namespace NUMINAMATH_CALUDE_shopping_mall_purchase_l2089_208913

/-- Represents the shopping mall's purchase of products A and B -/
structure ProductPurchase where
  cost_price_A : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ
  profit_margin_B : ℝ
  total_units : ℕ
  total_cost : ℝ

/-- Theorem stating the correct number of units purchased for each product -/
theorem shopping_mall_purchase (p : ProductPurchase)
  (h1 : p.cost_price_A = 40)
  (h2 : p.selling_price_A = 60)
  (h3 : p.selling_price_B = 80)
  (h4 : p.profit_margin_B = 0.6)
  (h5 : p.total_units = 50)
  (h6 : p.total_cost = 2200) :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = p.total_units ∧
    units_A * p.cost_price_A + units_B * (p.selling_price_B / (1 + p.profit_margin_B)) = p.total_cost ∧
    units_A = 30 ∧
    units_B = 20 := by
  sorry


end NUMINAMATH_CALUDE_shopping_mall_purchase_l2089_208913


namespace NUMINAMATH_CALUDE_tangency_quadrilateral_area_is_1_6_l2089_208952

/-- An isosceles trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Area of the trapezoid -/
  trapezoidArea : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- The circle is inscribed in the trapezoid -/
  isInscribed : Bool

/-- The area of the quadrilateral formed by the points of tangency -/
def tangencyQuadrilateralArea (t : InscribedCircleTrapezoid) : ℝ := sorry

/-- Theorem: The area of the tangency quadrilateral is 1.6 -/
theorem tangency_quadrilateral_area_is_1_6 (t : InscribedCircleTrapezoid) 
  (h1 : t.radius = 1) 
  (h2 : t.trapezoidArea = 5) 
  (h3 : t.isIsosceles = true) 
  (h4 : t.isInscribed = true) : 
  tangencyQuadrilateralArea t = 1.6 := by sorry

end NUMINAMATH_CALUDE_tangency_quadrilateral_area_is_1_6_l2089_208952


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l2089_208925

theorem fraction_product_theorem : 
  (7 : ℚ) / 4 * 8 / 14 * 28 / 16 * 24 / 36 * 49 / 35 * 40 / 25 * 63 / 42 * 32 / 48 = 56 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l2089_208925


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l2089_208944

theorem functional_inequality_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) ≤ (1/2) * (f x + f y)) : 
  ∃ a c : ℝ, (∀ x : ℝ, x ≠ 0 → f x = c) ∧ (f 0 = a) ∧ (a ≤ c) := by
  sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l2089_208944


namespace NUMINAMATH_CALUDE_probability_both_odd_l2089_208955

def m : ℕ := 7
def n : ℕ := 9

def is_odd (k : ℕ) : Prop := k % 2 = 1

def count_odd (k : ℕ) : ℕ := (k + 1) / 2

theorem probability_both_odd : 
  (count_odd m * count_odd n : ℚ) / (m * n : ℚ) = 20 / 63 := by sorry

end NUMINAMATH_CALUDE_probability_both_odd_l2089_208955


namespace NUMINAMATH_CALUDE_complement_of_union_l2089_208908

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {5, 6, 7}

theorem complement_of_union : (U \ (M ∪ N)) = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2089_208908


namespace NUMINAMATH_CALUDE_rectangle_area_l2089_208954

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2089_208954


namespace NUMINAMATH_CALUDE_yuna_candies_l2089_208912

theorem yuna_candies (initial_candies remaining_candies : ℕ) 
  (h1 : initial_candies = 23)
  (h2 : remaining_candies = 7) :
  initial_candies - remaining_candies = 16 := by
  sorry

end NUMINAMATH_CALUDE_yuna_candies_l2089_208912


namespace NUMINAMATH_CALUDE_radio_dealer_profit_l2089_208988

theorem radio_dealer_profit (n d : ℕ) (h_d_pos : d > 0) : 
  (3 * (d / n / 3) + (n - 3) * (d / n + 10) - d = 100) → n ≥ 13 :=
by
  sorry

end NUMINAMATH_CALUDE_radio_dealer_profit_l2089_208988
