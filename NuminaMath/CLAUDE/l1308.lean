import Mathlib

namespace family_probability_l1308_130887

theorem family_probability :
  let p_boy : ℝ := 1/2
  let p_girl : ℝ := 1/2
  let num_children : ℕ := 4
  p_boy + p_girl = 1 →
  (1 : ℝ) - (p_boy ^ num_children + p_girl ^ num_children) = 7/8 :=
by sorry

end family_probability_l1308_130887


namespace m_div_60_eq_483840_l1308_130843

/-- The smallest positive integer that is a multiple of 60 and has exactly 96 positive integral divisors -/
def m : ℕ := sorry

/-- m is a multiple of 60 -/
axiom m_multiple_of_60 : 60 ∣ m

/-- m has exactly 96 positive integral divisors -/
axiom m_divisors_count : (Finset.filter (· ∣ m) (Finset.range m)).card = 96

/-- m is the smallest such number -/
axiom m_smallest : ∀ k : ℕ, k < m → ¬(60 ∣ k ∧ (Finset.filter (· ∣ k) (Finset.range k)).card = 96)

/-- The main theorem -/
theorem m_div_60_eq_483840 : m / 60 = 483840 := sorry

end m_div_60_eq_483840_l1308_130843


namespace vector_expression_l1308_130812

theorem vector_expression (a b c : ℝ × ℝ) :
  a = (1, 2) →
  a + b = (0, 3) →
  c = (1, 5) →
  c = 2 • a + b := by
sorry

end vector_expression_l1308_130812


namespace smallest_number_with_same_prime_factors_l1308_130813

def alice_number : ℕ := 72

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_number_with_same_prime_factors :
  ∃ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → k ≤ m :=
by sorry

end smallest_number_with_same_prime_factors_l1308_130813


namespace inclined_plane_angle_theorem_l1308_130864

/-- 
Given a system with two blocks connected by a cord over a frictionless pulley,
where one block of mass m is on a frictionless inclined plane and the other block
of mass M is hanging vertically, this theorem proves that if M = 1.5 * m and the
acceleration of the system is g/3, then the angle θ of the inclined plane
satisfies sin θ = 2/3.
-/
theorem inclined_plane_angle_theorem 
  (m : ℝ) 
  (M : ℝ) 
  (g : ℝ) 
  (θ : ℝ) 
  (h_mass : M = 1.5 * m) 
  (h_accel : m * g / 3 = m * g * (1 - Real.sin θ)) : 
  Real.sin θ = 2 / 3 := by
  sorry

end inclined_plane_angle_theorem_l1308_130864


namespace tangent_line_equation_l1308_130852

/-- The function f(x) = x³ - 3x² + x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 1

theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ 2*x + y - 1 = 0) ∧
    (m = f' 1) ∧
    (f 1 = m*1 + b) :=
sorry

end tangent_line_equation_l1308_130852


namespace yanna_apples_kept_l1308_130828

def apples_kept (initial : ℕ) (given_to_zenny : ℕ) (given_to_andrea : ℕ) : ℕ :=
  initial - given_to_zenny - given_to_andrea

theorem yanna_apples_kept :
  apples_kept 60 18 6 = 36 := by
  sorry

end yanna_apples_kept_l1308_130828


namespace cancellable_fractions_characterization_l1308_130849

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def cancellable_fraction (n d : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit d ∧
  ∃ (a b c : ℕ), 0 < a ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 < c ∧
    n = 10 * a + b ∧ d = 10 * b + c ∧ n * c = a * d

def valid_fractions : Set (ℕ × ℕ) :=
  {(19, 95), (16, 64), (11, 11), (26, 65), (22, 22), (33, 33),
   (49, 98), (44, 44), (55, 55), (66, 66), (77, 77), (88, 88), (99, 99)}

theorem cancellable_fractions_characterization :
  {p : ℕ × ℕ | cancellable_fraction p.1 p.2} = valid_fractions := by sorry

end cancellable_fractions_characterization_l1308_130849


namespace expression_simplification_l1308_130892

theorem expression_simplification (a : ℝ) :
  (a - (2*a - 1) / a) / ((1 - a^2) / (a^2 + a)) = a + 1 :=
by sorry

end expression_simplification_l1308_130892


namespace min_cans_correct_l1308_130845

/-- The volume of soda in a single can (in ounces) -/
def can_volume : ℝ := 15

/-- The conversion factor from liters to ounces -/
def liter_to_ounce : ℝ := 33.814

/-- The required volume of soda (in liters) -/
def required_volume : ℝ := 3.8

/-- The minimum number of cans required to provide at least the required volume of soda -/
def min_cans : ℕ := 9

/-- Theorem stating that the minimum number of cans required to provide at least
    the required volume of soda is 9 -/
theorem min_cans_correct :
  ∀ n : ℕ, (n : ℝ) * can_volume ≥ required_volume * liter_to_ounce → n ≥ min_cans :=
by sorry

end min_cans_correct_l1308_130845


namespace temperature_difference_l1308_130881

def lowest_temp : ℤ := -4
def highest_temp : ℤ := 5

theorem temperature_difference : highest_temp - lowest_temp = 9 := by
  sorry

end temperature_difference_l1308_130881


namespace omega_is_abc_l1308_130898

theorem omega_is_abc (ω a b c x y z : ℝ) 
  (distinct : ω ≠ a ∧ ω ≠ b ∧ ω ≠ c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (eq1 : x + y + z = 1)
  (eq2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (eq3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (eq4 : a^4 * x + b^4 * y + c^4 * z = ω^4) :
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end omega_is_abc_l1308_130898


namespace square_and_sqrt_identities_l1308_130857

theorem square_and_sqrt_identities :
  (1001 : ℕ)^2 = 1002001 ∧
  (1001001 : ℕ)^2 = 1002003002001 ∧
  (1002003004005004003002001 : ℕ).sqrt = 1001001001001 := by
  sorry

end square_and_sqrt_identities_l1308_130857


namespace brothers_money_distribution_l1308_130863

/-- Represents the money distribution among four brothers -/
structure MoneyDistribution where
  john : ℕ
  william : ℕ
  charles : ℕ
  thomas : ℕ

/-- Checks if the given money distribution satisfies all conditions -/
def satisfies_conditions (d : MoneyDistribution) : Prop :=
  d.john + 2 = d.william - 2 ∧
  d.john + 2 = 2 * d.charles ∧
  d.john + 2 = d.thomas / 2 ∧
  d.john + d.william + d.charles + d.thomas = 45

/-- Checks if the given money distribution can be represented with 6 coins -/
def can_be_represented_with_six_coins (d : MoneyDistribution) : Prop :=
  ∃ (j1 j2 w1 w2 c t : ℕ),
    j1 + j2 = d.john ∧
    w1 + w2 = d.william ∧
    c = d.charles ∧
    t = d.thomas

/-- The main theorem stating the unique solution for the brothers' money distribution -/
theorem brothers_money_distribution :
  ∃! (d : MoneyDistribution),
    satisfies_conditions d ∧
    can_be_represented_with_six_coins d ∧
    d.john = 8 ∧ d.william = 12 ∧ d.charles = 5 ∧ d.thomas = 20 :=
by
  sorry

end brothers_money_distribution_l1308_130863


namespace triangle_side_range_l1308_130846

theorem triangle_side_range (a b c : ℝ) : 
  (|a - 3| + (b - 7)^2 = 0) →
  (c ≥ a ∧ c ≥ b) →
  (c < a + b) →
  (7 ≤ c ∧ c < 10) :=
sorry

end triangle_side_range_l1308_130846


namespace original_price_calculation_l1308_130895

theorem original_price_calculation (initial_price : ℚ) : 
  (initial_price * (1 + 10/100) * (1 - 20/100) = 2) → initial_price = 25/11 := by
  sorry

end original_price_calculation_l1308_130895


namespace coordinate_sum_theorem_l1308_130829

theorem coordinate_sum_theorem (g : ℝ → ℝ) (h : g 4 = 7) :
  ∃ (x y : ℝ), 3 * y = 2 * g (3 * x) + 6 ∧ x + y = 8 := by
  sorry

end coordinate_sum_theorem_l1308_130829


namespace sum_of_x_and_y_is_ten_l1308_130870

theorem sum_of_x_and_y_is_ten (x y : ℝ) (h1 : x = 25 / y) (h2 : x^2 + y^2 = 50) : x + y = 10 := by
  sorry

end sum_of_x_and_y_is_ten_l1308_130870


namespace present_ages_of_deepak_and_rajat_l1308_130804

-- Define the present ages as variables
variable (R D Ra : ℕ)

-- Define the conditions
def present_age_ratio : Prop := R / D = 4 / 3 ∧ Ra / D = 5 / 3
def rahul_future_age : Prop := R + 4 = 32
def rajat_future_age : Prop := Ra + 7 = 50

-- State the theorem
theorem present_ages_of_deepak_and_rajat 
  (h1 : present_age_ratio R D Ra)
  (h2 : rahul_future_age R)
  (h3 : rajat_future_age Ra) :
  D = 21 ∧ Ra = 43 := by
  sorry

end present_ages_of_deepak_and_rajat_l1308_130804


namespace sample_size_equals_surveyed_parents_l1308_130821

/-- Represents a school survey about students' daily activities -/
structure SchoolSurvey where
  total_students : ℕ
  surveyed_parents : ℕ
  sleep_6_to_7_hours_percentage : ℚ
  homework_3_to_4_hours_percentage : ℚ

/-- The sample size of a school survey is equal to the number of surveyed parents -/
theorem sample_size_equals_surveyed_parents (survey : SchoolSurvey) 
  (h1 : survey.total_students = 1800)
  (h2 : survey.surveyed_parents = 1000)
  (h3 : survey.sleep_6_to_7_hours_percentage = 70/100)
  (h4 : survey.homework_3_to_4_hours_percentage = 28/100) :
  survey.surveyed_parents = 1000 := by
  sorry

#check sample_size_equals_surveyed_parents

end sample_size_equals_surveyed_parents_l1308_130821


namespace remainder_sum_l1308_130893

theorem remainder_sum (D : ℕ) (h1 : D > 0) (h2 : 242 % D = 11) (h3 : 698 % D = 18) :
  940 % D = 29 := by
  sorry

end remainder_sum_l1308_130893


namespace volume_of_specific_polyhedron_l1308_130805

/-- A polygon in the figure --/
inductive Polygon
| EquilateralTriangle
| Square
| RegularHexagon

/-- The figure consisting of multiple polygons --/
structure Figure where
  polygons : List Polygon
  triangleSideLength : ℝ
  squareSideLength : ℝ
  hexagonSideLength : ℝ

/-- The polyhedron formed by folding the figure --/
structure Polyhedron where
  figure : Figure

/-- Calculate the volume of the polyhedron --/
def calculateVolume (p : Polyhedron) : ℝ :=
  sorry

/-- The theorem stating that the volume of the specific polyhedron is 8 --/
theorem volume_of_specific_polyhedron :
  let fig : Figure := {
    polygons := [Polygon.EquilateralTriangle, Polygon.EquilateralTriangle, Polygon.EquilateralTriangle,
                 Polygon.Square, Polygon.Square, Polygon.Square,
                 Polygon.RegularHexagon],
    triangleSideLength := 2,
    squareSideLength := 2,
    hexagonSideLength := 1
  }
  let poly : Polyhedron := { figure := fig }
  calculateVolume poly = 8 :=
sorry

end volume_of_specific_polyhedron_l1308_130805


namespace sum_of_squares_in_ratio_l1308_130810

theorem sum_of_squares_in_ratio (x y z : ℝ) : 
  x + y + z = 9 ∧ y = 2*x ∧ z = 4*x → x^2 + y^2 + z^2 = 1701 / 49 := by
  sorry

end sum_of_squares_in_ratio_l1308_130810


namespace unique_solution_l1308_130848

theorem unique_solution : ∃! x : ℝ, x + x^2 + 15 = 96 := by
  sorry

end unique_solution_l1308_130848


namespace arithmetic_geometric_mean_inequality_l1308_130825

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end arithmetic_geometric_mean_inequality_l1308_130825


namespace absolute_value_of_five_minus_e_l1308_130844

-- Define e as a constant approximation
def e : ℝ := 2.71828

-- State the theorem
theorem absolute_value_of_five_minus_e : |5 - e| = 2.28172 := by sorry

end absolute_value_of_five_minus_e_l1308_130844


namespace cubic_sum_theorem_l1308_130896

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 3) 
  (h3 : a * b * c = 1) : 
  a^3 + b^3 + c^3 = 5 := by sorry

end cubic_sum_theorem_l1308_130896


namespace binomial_12_3_l1308_130858

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by sorry

end binomial_12_3_l1308_130858


namespace probability_equals_fraction_l1308_130888

def num_forks : ℕ := 8
def num_spoons : ℕ := 5
def num_knives : ℕ := 7
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def pieces_removed : ℕ := 4

def probability_two_forks_one_spoon_one_knife : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1) /
  Nat.choose total_silverware pieces_removed

theorem probability_equals_fraction :
  probability_two_forks_one_spoon_one_knife = 196 / 969 := by
  sorry

end probability_equals_fraction_l1308_130888


namespace equal_roots_condition_l1308_130872

theorem equal_roots_condition (x m : ℝ) : 
  (x * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x / m → 
  (∃ (a : ℝ), ∀ (x : ℝ), x * (x - 2) - (m + 2) = (x - 2) * (m - 2) * (x / m) → x = a) →
  m = -3/2 := by
sorry

end equal_roots_condition_l1308_130872


namespace initial_alcohol_percentage_l1308_130851

/-- Given a 40-liter solution of alcohol and water, prove that the initial percentage
    of alcohol is 5% if adding 4.5 liters of alcohol and 5.5 liters of water
    results in a 50-liter solution that is 13% alcohol. -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (final_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_alcohol = 4.5)
  (h3 : added_water = 5.5)
  (h4 : final_volume = initial_volume + added_alcohol + added_water)
  (h5 : final_percentage = 13)
  (h6 : final_percentage / 100 * final_volume = 
        initial_volume * (initial_percentage / 100) + added_alcohol) :
  initial_percentage = 5 :=
by sorry

end initial_alcohol_percentage_l1308_130851


namespace journey_distance_on_foot_l1308_130801

theorem journey_distance_on_foot 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_on_foot : ℝ) 
  (speed_on_bicycle : ℝ) 
  (h1 : total_distance = 80) 
  (h2 : total_time = 7) 
  (h3 : speed_on_foot = 8) 
  (h4 : speed_on_bicycle = 16) :
  ∃ (distance_on_foot : ℝ),
    distance_on_foot = 32 ∧
    ∃ (distance_on_bicycle : ℝ),
      distance_on_foot + distance_on_bicycle = total_distance ∧
      distance_on_foot / speed_on_foot + distance_on_bicycle / speed_on_bicycle = total_time :=
by sorry

end journey_distance_on_foot_l1308_130801


namespace choose_four_from_nine_l1308_130890

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 → k = 4 → Nat.choose n k = 126 := by
  sorry

end choose_four_from_nine_l1308_130890


namespace smallest_number_l1308_130823

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def binary : List Nat := [1, 0, 1, 0, 1, 1]
def ternary : List Nat := [1, 2, 1, 0]
def octal : List Nat := [1, 1, 0]
def duodecimal : List Nat := [6, 8]

theorem smallest_number : 
  to_decimal ternary 3 ≤ to_decimal binary 2 ∧
  to_decimal ternary 3 ≤ to_decimal octal 8 ∧
  to_decimal ternary 3 ≤ to_decimal duodecimal 12 := by
  sorry

end smallest_number_l1308_130823


namespace financial_equation_solution_l1308_130850

theorem financial_equation_solution (g t p : ℂ) : 
  3 * g * p - t = 9000 ∧ g = 3 ∧ t = 3 + 75 * Complex.I → 
  p = 1000 + 1/3 + 8 * Complex.I + 1/3 * Complex.I := by
  sorry

end financial_equation_solution_l1308_130850


namespace arithmetic_sequence_sum_ratio_l1308_130822

theorem arithmetic_sequence_sum_ratio (a₁ d : ℚ) : 
  let S : ℕ → ℚ := λ n => n * a₁ + n * (n - 1) / 2 * d
  (S 3) / (S 7) = 1 / 3 → (S 6) / (S 7) = 17 / 21 := by
  sorry

end arithmetic_sequence_sum_ratio_l1308_130822


namespace shoe_price_increase_l1308_130820

theorem shoe_price_increase (regular_price : ℝ) (h : regular_price > 0) :
  let sale_price := regular_price * (1 - 0.2)
  (regular_price - sale_price) / sale_price * 100 = 25 := by
sorry

end shoe_price_increase_l1308_130820


namespace average_age_of_students_average_age_proof_l1308_130866

theorem average_age_of_students (total_students : Nat) 
  (group1_count : Nat) (group1_avg : Nat) 
  (group2_count : Nat) (group2_avg : Nat)
  (last_student_age : Nat) : Nat :=
  let total_age := group1_count * group1_avg + group2_count * group2_avg + last_student_age
  total_age / total_students

theorem average_age_proof :
  average_age_of_students 15 8 14 6 16 17 = 15 := by
  sorry

end average_age_of_students_average_age_proof_l1308_130866


namespace train_travel_times_l1308_130899

theorem train_travel_times
  (usual_speed_A : ℝ)
  (usual_speed_B : ℝ)
  (distance_XM : ℝ)
  (h1 : usual_speed_A > 0)
  (h2 : usual_speed_B > 0)
  (h3 : distance_XM > 0)
  (h4 : usual_speed_B * 2 = usual_speed_A * 3) :
  let t : ℝ := 180
  let current_speed_A : ℝ := (6 / 7) * usual_speed_A
  let time_XM_reduced : ℝ := distance_XM / current_speed_A
  let time_XM_usual : ℝ := distance_XM / usual_speed_A
  let time_XY_A : ℝ := 3 * time_XM_usual
  let time_XY_B : ℝ := 810
  (time_XM_reduced = time_XM_usual + 30) ∧
  (time_XM_usual = t) ∧
  (time_XY_B = 1.5 * time_XY_A) := by
  sorry

end train_travel_times_l1308_130899


namespace function_decomposition_l1308_130891

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ),
    (∀ x, f x = g x + h x) ∧
    (∀ x, g x = g (-x)) ∧
    (∀ x, h (1 + x) = h (1 - x)) := by
  sorry

end function_decomposition_l1308_130891


namespace sin_A_value_l1308_130838

theorem sin_A_value (A : Real) (h1 : 0 < A) (h2 : A < Real.pi / 2) (h3 : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := by
  sorry

end sin_A_value_l1308_130838


namespace binary_sum_equals_852_l1308_130876

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def num1 : List Bool := [true, true, true, true, true, true, true, true, true]
def num2 : List Bool := [true, false, true, false, true, false, true, false, true]

theorem binary_sum_equals_852 : 
  binary_to_decimal num1 + binary_to_decimal num2 = 852 := by
sorry

end binary_sum_equals_852_l1308_130876


namespace cube_in_pyramid_volume_l1308_130800

/-- A pyramid with a square base and isosceles right triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (is_square_base : base_side > 0)
  (lateral_faces_isosceles_right : True)

/-- A cube placed inside a pyramid -/
structure CubeInPyramid :=
  (pyramid : Pyramid)
  (bottom_on_base : True)
  (top_touches_midpoints : True)

/-- The volume of a cube -/
def cube_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

/-- Theorem: The volume of the cube in the given pyramid configuration is 1 -/
theorem cube_in_pyramid_volume 
  (p : Pyramid) 
  (c : CubeInPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : c.pyramid = p) : 
  ∃ (edge_length : ℝ), cube_volume edge_length = 1 :=
sorry

end cube_in_pyramid_volume_l1308_130800


namespace perpendicular_implies_parallel_necessary_not_sufficient_l1308_130847

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the specific lines and planes
variable (l m : Line) (α β : Plane)

-- State the theorem
theorem perpendicular_implies_parallel_necessary_not_sufficient 
  (h1 : perp_plane l α) 
  (h2 : subset m β) :
  (∀ α β, parallel α β → perp l m) ∧ 
  (∃ α β, perp l m ∧ ¬ parallel α β) := by
  sorry

end perpendicular_implies_parallel_necessary_not_sufficient_l1308_130847


namespace parabola_vertex_specific_parabola_vertex_l1308_130814

/-- The vertex of a parabola in the form y = a(x-h)^2 + k is (h, k) --/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  ∃! (x₀ y₀ : ℝ), (∀ x, f x ≥ f x₀) ∧ f x₀ = y₀ ∧ (x₀, y₀) = (h, k) :=
sorry

/-- The vertex of the parabola y = -2(x-3)^2 - 2 is (3, -2) --/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ -2 * (x - 3)^2 - 2
  ∃! (x₀ y₀ : ℝ), (∀ x, f x ≥ f x₀) ∧ f x₀ = y₀ ∧ (x₀, y₀) = (3, -2) :=
sorry

end parabola_vertex_specific_parabola_vertex_l1308_130814


namespace range_of_p_l1308_130811

-- Define the sequence a_n
def a (n : ℕ+) : ℝ := (-1 : ℝ)^(n.val - 1) * (2 * n.val - 1)

-- Define the sum S_n
def S (n : ℕ+) : ℝ := (-1 : ℝ)^(n.val - 1) * n.val

-- Theorem statement
theorem range_of_p (p : ℝ) :
  (∀ n : ℕ+, (a (n + 1) - p) * (a n - p) < 0) ↔ -3 < p ∧ p < 1 := by
  sorry

end range_of_p_l1308_130811


namespace function_inequality_implies_a_bound_l1308_130877

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → Real.log (1 + x) ≥ a * x / (1 + x)) →
  a ≤ 1 := by
sorry

end function_inequality_implies_a_bound_l1308_130877


namespace min_b_value_l1308_130806

theorem min_b_value (a c b : ℕ+) (h1 : a < c) (h2 : c < b)
  (h3 : ∃! p : ℝ × ℝ, 3 * p.1 + p.2 = 3000 ∧ 
    p.2 = |p.1 - a.val| + |p.1 - c.val| + |p.1 - b.val|) :
  ∀ b' : ℕ+, (∃ a' c' : ℕ+, a' < c' ∧ c' < b' ∧
    ∃! p : ℝ × ℝ, 3 * p.1 + p.2 = 3000 ∧ 
    p.2 = |p.1 - a'.val| + |p.1 - c'.val| + |p.1 - b'.val|) → 
  9 ≤ b'.val := by
  sorry

end min_b_value_l1308_130806


namespace binomial_18_9_l1308_130862

theorem binomial_18_9 (h1 : Nat.choose 16 7 = 11440) 
                      (h2 : Nat.choose 16 8 = 12870) 
                      (h3 : Nat.choose 16 9 = 11440) : 
  Nat.choose 18 9 = 48620 := by
  sorry

end binomial_18_9_l1308_130862


namespace payroll_after_layoffs_l1308_130856

/-- Represents the company's employee structure and payroll --/
structure Company where
  total_employees : Nat
  employees_2000 : Nat
  employees_2500 : Nat
  employees_3000 : Nat
  bonus_2000 : Nat
  health_benefit_2500 : Nat
  retirement_benefit_3000 : Nat

/-- Calculates the remaining employees after a layoff --/
def layoff (employees : Nat) (percentage : Nat) : Nat :=
  employees - (employees * percentage / 100)

/-- Applies the first round of layoffs and benefit changes --/
def first_round (c : Company) : Company :=
  { c with
    employees_2000 := layoff c.employees_2000 20,
    employees_2500 := layoff c.employees_2500 25,
    employees_3000 := layoff c.employees_3000 15,
    bonus_2000 := 400,
    health_benefit_2500 := 300 }

/-- Applies the second round of layoffs and benefit changes --/
def second_round (c : Company) : Company :=
  { c with
    employees_2000 := layoff c.employees_2000 10,
    employees_2500 := layoff c.employees_2500 15,
    employees_3000 := layoff c.employees_3000 5,
    retirement_benefit_3000 := 480 }

/-- Calculates the total payroll after both rounds of layoffs --/
def total_payroll (c : Company) : Nat :=
  c.employees_2000 * (2000 + c.bonus_2000) +
  c.employees_2500 * (2500 + c.health_benefit_2500) +
  c.employees_3000 * (3000 + c.retirement_benefit_3000)

/-- The initial company state --/
def initial_company : Company :=
  { total_employees := 450,
    employees_2000 := 150,
    employees_2500 := 200,
    employees_3000 := 100,
    bonus_2000 := 500,
    health_benefit_2500 := 400,
    retirement_benefit_3000 := 600 }

theorem payroll_after_layoffs :
  total_payroll (second_round (first_round initial_company)) = 893200 := by
  sorry

end payroll_after_layoffs_l1308_130856


namespace locus_of_centroid_l1308_130827

/-- The locus of the centroid of a triangle formed by specific points on a line and parabola -/
theorem locus_of_centroid (k : ℝ) (x y : ℝ) : 
  -- Line l: y = k(x - 2)
  (∀ x, y = k * (x - 2)) →
  -- Parabola: y = x^2 + 2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ k * (x₁ - 2) = x₁^2 + 2 ∧ k * (x₂ - 2) = x₂^2 + 2) →
  -- Conditions on k
  (k < 4 - 2 * Real.sqrt 6 ∨ k > 4 + 2 * Real.sqrt 6) →
  k ≠ 0 →
  -- Point P conditions
  (∃ x₀ y₀, y₀ = (12 * k) / (k - 4) ∧ x₀ = 12 / (k - 4) + 2) →
  -- Centroid G(x, y)
  (x = (4 / (k - 4)) + 4/3 ∧ y = (4 * k) / (k - 4)) →
  -- Locus equation
  12 * x - 3 * y - 4 = 0 ∧ 
  4 - (4/3) * Real.sqrt 6 < y ∧ 
  y < 4 + (4/3) * Real.sqrt 6 ∧ 
  y ≠ 4 :=
by sorry

end locus_of_centroid_l1308_130827


namespace angle_measure_in_special_triangle_l1308_130880

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
  h_sum_angles : A + B + C = π
  h_area : S = (1/2) * b * c * Real.sin A

-- Theorem statement
theorem angle_measure_in_special_triangle (t : Triangle) 
  (h : (t.b + t.c)^2 - t.a^2 = 4 * Real.sqrt 3 * t.S) : 
  t.A = π/3 := by
  sorry

end angle_measure_in_special_triangle_l1308_130880


namespace find_n_l1308_130819

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 9 ∧ n = 2 := by
  sorry

end find_n_l1308_130819


namespace cleaning_time_calculation_l1308_130867

/-- Represents the cleaning schedule for a person -/
structure CleaningSchedule where
  vacuuming : Nat × Nat  -- (minutes per day, days per week)
  dusting : Nat × Nat
  sweeping : Nat × Nat
  deep_cleaning : Nat × Nat

/-- Calculates the total cleaning time in minutes per week -/
def totalCleaningTime (schedule : CleaningSchedule) : Nat :=
  schedule.vacuuming.1 * schedule.vacuuming.2 +
  schedule.dusting.1 * schedule.dusting.2 +
  schedule.sweeping.1 * schedule.sweeping.2 +
  schedule.deep_cleaning.1 * schedule.deep_cleaning.2

/-- Converts minutes to hours and minutes -/
def minutesToHoursAndMinutes (minutes : Nat) : Nat × Nat :=
  (minutes / 60, minutes % 60)

/-- Aron's cleaning schedule -/
def aronSchedule : CleaningSchedule :=
  { vacuuming := (30, 3)
    dusting := (20, 2)
    sweeping := (15, 4)
    deep_cleaning := (45, 1) }

/-- Ben's cleaning schedule -/
def benSchedule : CleaningSchedule :=
  { vacuuming := (40, 2)
    dusting := (25, 3)
    sweeping := (20, 5)
    deep_cleaning := (60, 1) }

theorem cleaning_time_calculation :
  let aronTime := totalCleaningTime aronSchedule
  let benTime := totalCleaningTime benSchedule
  let aronHoursMinutes := minutesToHoursAndMinutes aronTime
  let benHoursMinutes := minutesToHoursAndMinutes benTime
  let timeDifference := benTime - aronTime
  let timeDifferenceHoursMinutes := minutesToHoursAndMinutes timeDifference
  aronHoursMinutes = (3, 55) ∧
  benHoursMinutes = (5, 15) ∧
  timeDifferenceHoursMinutes = (1, 20) := by
  sorry

end cleaning_time_calculation_l1308_130867


namespace exactly_three_correct_deliveries_l1308_130831

def n : ℕ := 5

-- The probability of exactly k successes in n trials
def probability (k : ℕ) : ℚ :=
  (n.choose k * (n - k).factorial) / n.factorial

-- The main theorem
theorem exactly_three_correct_deliveries : probability 3 = 1 / 12 := by
  sorry

end exactly_three_correct_deliveries_l1308_130831


namespace tan_three_pi_fourth_l1308_130835

theorem tan_three_pi_fourth : Real.tan (3 * π / 4) = -1 := by
  sorry

end tan_three_pi_fourth_l1308_130835


namespace fish_price_calculation_l1308_130860

theorem fish_price_calculation (discount_rate : ℝ) (discounted_price : ℝ) (package_weight : ℝ) : 
  discount_rate = 0.6 →
  discounted_price = 4.5 →
  package_weight = 0.75 →
  (discounted_price / package_weight) / (1 - discount_rate) = 15 := by
sorry

end fish_price_calculation_l1308_130860


namespace square_difference_equals_square_l1308_130826

theorem square_difference_equals_square (x : ℝ) : (10 - x)^2 = x^2 ↔ x = 5 := by
  sorry

end square_difference_equals_square_l1308_130826


namespace vikkis_take_home_pay_is_correct_l1308_130886

/-- Calculates Vikki's take-home pay after all deductions --/
def vikkis_take_home_pay (hours_worked : ℕ) (hourly_rate : ℚ) 
  (federal_tax_rate_low : ℚ) (federal_tax_rate_high : ℚ) (federal_tax_threshold : ℚ)
  (state_tax_rate : ℚ) (retirement_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : ℚ :=
  let gross_earnings := hours_worked * hourly_rate
  let federal_tax_low := min federal_tax_threshold gross_earnings * federal_tax_rate_low
  let federal_tax_high := max 0 (gross_earnings - federal_tax_threshold) * federal_tax_rate_high
  let state_tax := gross_earnings * state_tax_rate
  let retirement := gross_earnings * retirement_rate
  let insurance := gross_earnings * insurance_rate
  let total_deductions := federal_tax_low + federal_tax_high + state_tax + retirement + insurance + union_dues
  gross_earnings - total_deductions

/-- Theorem stating that Vikki's take-home pay is $328.48 --/
theorem vikkis_take_home_pay_is_correct : 
  vikkis_take_home_pay 42 12 (15/100) (22/100) 300 (7/100) (6/100) (3/100) 5 = 328.48 := by
  sorry

end vikkis_take_home_pay_is_correct_l1308_130886


namespace pencil_count_l1308_130803

/-- Given the ratio of pens to pencils and their difference, calculate the number of pencils -/
theorem pencil_count (x : ℕ) (h1 : 6 * x = 5 * x + 6) : 6 * x = 36 := by
  sorry

#check pencil_count

end pencil_count_l1308_130803


namespace opposite_of_negative_2023_l1308_130833

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end opposite_of_negative_2023_l1308_130833


namespace organization_growth_l1308_130883

/-- Represents the number of people in the organization at year k -/
def people_count (k : ℕ) : ℕ :=
  if k = 0 then 30
  else 3 * people_count (k - 1) - 20

/-- The number of leaders in the organization each year -/
def num_leaders : ℕ := 10

/-- The initial number of people in the organization -/
def initial_people : ℕ := 30

theorem organization_growth :
  people_count 10 = 1180990 :=
sorry

end organization_growth_l1308_130883


namespace max_quartets_correct_max_quartets_5x5_l1308_130884

def max_quartets (m n : ℕ) : ℕ :=
  if m % 2 = 0 ∧ n % 2 = 0 then
    m * n / 4
  else if (m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0) then
    m * (n - 1) / 4
  else
    (m * (n - 1) - 2) / 4

theorem max_quartets_correct (m n : ℕ) :
  max_quartets m n = 
    if m % 2 = 0 ∧ n % 2 = 0 then
      m * n / 4
    else if (m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0) then
      m * (n - 1) / 4
    else
      (m * (n - 1) - 2) / 4 :=
by sorry

theorem max_quartets_5x5 : max_quartets 5 5 = 5 :=
by sorry

end max_quartets_correct_max_quartets_5x5_l1308_130884


namespace cake_recipe_difference_l1308_130854

theorem cake_recipe_difference (flour_required sugar_required sugar_added : ℕ) :
  flour_required = 9 →
  sugar_required = 6 →
  sugar_added = 4 →
  flour_required - (sugar_required - sugar_added) = 7 := by
sorry

end cake_recipe_difference_l1308_130854


namespace multiply_fractions_l1308_130855

theorem multiply_fractions : 12 * (1 / 15) * 30 = 24 := by
  sorry

end multiply_fractions_l1308_130855


namespace total_rooms_to_paint_l1308_130842

/-- Proves that the total number of rooms to be painted is 9 -/
theorem total_rooms_to_paint (hours_per_room : ℕ) (rooms_painted : ℕ) (hours_remaining : ℕ) : 
  hours_per_room = 8 → rooms_painted = 5 → hours_remaining = 32 →
  rooms_painted + (hours_remaining / hours_per_room) = 9 :=
by
  sorry

#check total_rooms_to_paint

end total_rooms_to_paint_l1308_130842


namespace square_plus_square_l1308_130839

theorem square_plus_square (x : ℝ) : x^2 + x^2 = 2 * x^2 := by
  sorry

end square_plus_square_l1308_130839


namespace friend_lunch_cost_l1308_130882

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 1 → friend_cost = total / 2 + difference / 2 → friend_cost = 8 := by
sorry

end friend_lunch_cost_l1308_130882


namespace no_solution_for_system_l1308_130894

theorem no_solution_for_system :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) := by
  sorry

end no_solution_for_system_l1308_130894


namespace folded_paper_perimeter_ratio_l1308_130853

/-- Represents the dimensions of a rectangular piece of paper --/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular piece of paper --/
def perimeter (p : PaperDimensions) : ℝ :=
  2 * (p.length + p.width)

/-- Represents the paper after folding and cutting --/
structure FoldedPaper where
  original : PaperDimensions
  flap : PaperDimensions
  largest_rectangle : PaperDimensions

/-- The theorem to be proved --/
theorem folded_paper_perimeter_ratio 
  (paper : FoldedPaper) 
  (h1 : paper.original.length = 6 ∧ paper.original.width = 6)
  (h2 : paper.flap.length = 3 ∧ paper.flap.width = 3)
  (h3 : paper.largest_rectangle.length = 6 ∧ paper.largest_rectangle.width = 4.5) :
  perimeter paper.flap / perimeter paper.largest_rectangle = 4 / 5 := by
  sorry

end folded_paper_perimeter_ratio_l1308_130853


namespace simplify_and_rationalize_l1308_130859

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 42 := by
  sorry

end simplify_and_rationalize_l1308_130859


namespace difference_c_minus_a_l1308_130824

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 50 := by
sorry

end difference_c_minus_a_l1308_130824


namespace class_ratio_proof_l1308_130837

/-- Given a class of students, prove that the ratio of boys in Grade A to girls in Grade B is 2:9. -/
theorem class_ratio_proof (S : ℚ) (G : ℚ) (B : ℚ) 
  (h1 : (1/3) * G = (1/4) * S) 
  (h2 : S = B + G) 
  (h3 : (2/5) * B > 0) 
  (h4 : (3/5) * G > 0) : 
  ((2/5) * B) / ((3/5) * G) = 2/9 := by
  sorry

end class_ratio_proof_l1308_130837


namespace rhombus_diagonal_length_l1308_130889

/-- Proves that in a rhombus with one diagonal of 62 meters and an area of 2480 square meters,
    the length of the other diagonal is 80 meters. -/
theorem rhombus_diagonal_length (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
    (h1 : d1 = 62) 
    (h2 : area = 2480) 
    (h3 : area = (d1 * d2) / 2) : d2 = 80 := by
  sorry

end rhombus_diagonal_length_l1308_130889


namespace bryce_raisins_l1308_130879

theorem bryce_raisins : ∃ (bryce carter : ℕ), 
  bryce = carter + 8 ∧ 
  carter = bryce / 3 ∧ 
  bryce = 12 := by
sorry

end bryce_raisins_l1308_130879


namespace integer_between_sqrt_2n_and_sqrt_5n_l1308_130875

theorem integer_between_sqrt_2n_and_sqrt_5n (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, Real.sqrt (2 * n) < k ∧ k < Real.sqrt (5 * n) := by
  sorry

end integer_between_sqrt_2n_and_sqrt_5n_l1308_130875


namespace smallest_prime_perimeter_triangle_l1308_130878

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
    a = 5 →
    is_prime a →
    is_prime b →
    is_prime c →
    is_prime (a + b + c) →
    triangle_inequality a b c →
    is_scalene a b c →
    ∀ p q r : ℕ,
      p = 5 →
      is_prime p →
      is_prime q →
      is_prime r →
      is_prime (p + q + r) →
      triangle_inequality p q r →
      is_scalene p q r →
      a + b + c ≤ p + q + r →
    a + b + c = 23 :=
sorry

end smallest_prime_perimeter_triangle_l1308_130878


namespace salt_solution_weight_l1308_130861

/-- 
Given a salt solution with initial concentration of 10% and final concentration of 30%,
this theorem proves that if 28.571428571428573 kg of pure salt is added,
the initial weight of the solution was 100 kg.
-/
theorem salt_solution_weight 
  (initial_concentration : Real) 
  (final_concentration : Real)
  (added_salt : Real) 
  (initial_weight : Real) :
  initial_concentration = 0.10 →
  final_concentration = 0.30 →
  added_salt = 28.571428571428573 →
  initial_concentration * initial_weight + added_salt = 
    final_concentration * (initial_weight + added_salt) →
  initial_weight = 100 := by
  sorry

#check salt_solution_weight

end salt_solution_weight_l1308_130861


namespace sequence_range_l1308_130865

theorem sequence_range (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 12 * S n = 4 * a (n + 1) + 5^n - 13) →
  (∀ n : ℕ, S n ≤ S 4) →
  (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) →
  13/48 ≤ a 1 ∧ a 1 ≤ 59/64 := by
  sorry

end sequence_range_l1308_130865


namespace sum_of_m_and_n_l1308_130818

theorem sum_of_m_and_n (m n : ℝ) (h : |m - 2| + |n - 6| = 0) : m + n = 8 := by
  sorry

end sum_of_m_and_n_l1308_130818


namespace perfect_cube_units_digits_l1308_130868

theorem perfect_cube_units_digits : 
  ∃! (s : Finset Nat), 
    (∀ d ∈ s, d < 10) ∧ 
    (∀ n : ℤ, ∃ d ∈ s, (n ^ 3) % 10 = d) ∧
    s.card = 10 :=
by sorry

end perfect_cube_units_digits_l1308_130868


namespace simple_interest_years_l1308_130817

/-- Calculates the number of years for which a sum was put at simple interest, given the principal amount and the additional interest earned with a 1% rate increase. -/
def calculateYears (principal : ℚ) (additionalInterest : ℚ) : ℚ :=
  (100 * additionalInterest) / principal

theorem simple_interest_years :
  let principal : ℚ := 2300
  let additionalInterest : ℚ := 69
  calculateYears principal additionalInterest = 3 := by
  sorry

end simple_interest_years_l1308_130817


namespace minimum_reciprocal_sum_l1308_130834

theorem minimum_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ 1 / 5 := by
  sorry

end minimum_reciprocal_sum_l1308_130834


namespace integer_difference_l1308_130807

theorem integer_difference (x y : ℤ) : 
  x = 32 → y = 5 * x + 2 → y - x = 130 := by
  sorry

end integer_difference_l1308_130807


namespace basketball_three_pointers_l1308_130874

/-- Represents the number of 3-point shots in a basketball game -/
def three_point_shots (total_points total_shots : ℕ) : ℕ :=
  sorry

/-- The number of 3-point shots is 4 when the total points is 26 and total shots is 11 -/
theorem basketball_three_pointers :
  three_point_shots 26 11 = 4 :=
sorry

end basketball_three_pointers_l1308_130874


namespace simons_treasures_l1308_130809

def sand_dollars : ℕ := 10

def sea_glass (sand_dollars : ℕ) : ℕ := 3 * sand_dollars

def seashells (sea_glass : ℕ) : ℕ := 5 * sea_glass

def total_treasures (sand_dollars sea_glass seashells : ℕ) : ℕ :=
  sand_dollars + sea_glass + seashells

theorem simons_treasures :
  total_treasures sand_dollars (sea_glass sand_dollars) (seashells (sea_glass sand_dollars)) = 190 := by
  sorry

end simons_treasures_l1308_130809


namespace tic_tac_toe_tie_probability_l1308_130840

theorem tic_tac_toe_tie_probability (john_win_prob martha_win_prob : ℚ) 
  (h1 : john_win_prob = 4 / 9)
  (h2 : martha_win_prob = 5 / 12) :
  1 - (john_win_prob + martha_win_prob) = 5 / 36 := by
  sorry

end tic_tac_toe_tie_probability_l1308_130840


namespace matrix_transformation_and_eigenvalues_l1308_130832

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 2, 2]

theorem matrix_transformation_and_eigenvalues :
  -- 1) A transforms (1, 2) to (7, 6)
  A.mulVec ![1, 2] = ![7, 6] ∧
  -- 2) The eigenvalues of A are -1 and 4
  (A.charpoly.roots.toFinset = {-1, 4}) ∧
  -- 3) [3, -2] is an eigenvector for λ = -1
  (A.mulVec ![3, -2] = (-1 : ℝ) • ![3, -2]) ∧
  -- 4) [1, 1] is an eigenvector for λ = 4
  (A.mulVec ![1, 1] = (4 : ℝ) • ![1, 1]) := by
sorry


end matrix_transformation_and_eigenvalues_l1308_130832


namespace lentil_dishes_count_l1308_130830

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_lentils : ℕ)
  (beans_seitan : ℕ)
  (tempeh_lentils : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)
  (only_tempeh : ℕ)

/-- The conditions of the vegan restaurant menu -/
def menu_conditions (m : VeganMenu) : Prop :=
  m.total_dishes = 20 ∧
  m.beans_lentils = 3 ∧
  m.beans_seitan = 4 ∧
  m.tempeh_lentils = 2 ∧
  m.only_beans = 2 * m.only_tempeh ∧
  m.only_seitan = 3 * m.only_tempeh ∧
  m.total_dishes = m.beans_lentils + m.beans_seitan + m.tempeh_lentils +
                   m.only_beans + m.only_seitan + m.only_lentils + m.only_tempeh

/-- The theorem stating that the number of dishes with lentils is 10 -/
theorem lentil_dishes_count (m : VeganMenu) :
  menu_conditions m → m.only_lentils + m.beans_lentils + m.tempeh_lentils = 10 :=
by
  sorry


end lentil_dishes_count_l1308_130830


namespace positive_difference_of_average_l1308_130871

theorem positive_difference_of_average (y : ℝ) : 
  (50 + y) / 2 = 35 → |50 - y| = 30 := by
  sorry

end positive_difference_of_average_l1308_130871


namespace soup_feeding_theorem_l1308_130816

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) : ℕ :=
  let cans_for_children := (children_fed + can_capacity.children - 1) / can_capacity.children
  let remaining_cans := total_cans - cans_for_children
  remaining_cans * can_capacity.adults

/-- The main theorem to be proved -/
theorem soup_feeding_theorem (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) :
  total_cans = 8 →
  can_capacity.adults = 5 →
  can_capacity.children = 10 →
  children_fed = 20 →
  remaining_adults_fed total_cans can_capacity children_fed = 30 := by
  sorry

#check soup_feeding_theorem

end soup_feeding_theorem_l1308_130816


namespace expression_equality_l1308_130815

theorem expression_equality : 
  (1/2)⁻¹ + 4 * Real.cos (60 * π / 180) - |-3| + Real.sqrt 9 - (-2023)^0 + (-1)^(2023-1) = 4 := by
  sorry

end expression_equality_l1308_130815


namespace lines_perpendicular_l1308_130808

/-- A line passing through a point (1, 1) with equation 2x - ay - 1 = 0 -/
def line_l1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - a * p.2 - 1 = 0 ∧ p = (1, 1)}

/-- A line with equation x + 2y = 0 -/
def line_l2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2 * p.2 = 0}

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ m1 m2 : ℝ, (∀ p ∈ l1, ∀ q ∈ l1, p ≠ q → (p.2 - q.2) = m1 * (p.1 - q.1)) ∧
                (∀ p ∈ l2, ∀ q ∈ l2, p ≠ q → (p.2 - q.2) = m2 * (p.1 - q.1)) ∧
                m1 * m2 = -1

theorem lines_perpendicular :
  ∃ a : ℝ, perpendicular (line_l1 a) line_l2 :=
sorry

end lines_perpendicular_l1308_130808


namespace trinomial_square_equality_l1308_130869

theorem trinomial_square_equality : 
  15^2 + 3^2 + 1^2 + 2*(15*3) + 2*(15*1) + 2*(3*1) = (15 + 3 + 1)^2 := by
  sorry

end trinomial_square_equality_l1308_130869


namespace cos_equality_problem_l1308_130836

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (812 * π / 180) → n = 88 := by
  sorry

end cos_equality_problem_l1308_130836


namespace urea_formation_moles_l1308_130802

-- Define the chemical species
inductive ChemicalSpecies
| CarbonDioxide
| Ammonia
| Urea
| Water

-- Define a structure for chemical reactions
structure ChemicalReaction where
  reactants : List (ChemicalSpecies × ℚ)
  products : List (ChemicalSpecies × ℚ)

-- Define the urea formation reaction
def ureaFormationReaction : ChemicalReaction :=
  { reactants := [(ChemicalSpecies.CarbonDioxide, 1), (ChemicalSpecies.Ammonia, 2)]
  , products := [(ChemicalSpecies.Urea, 1), (ChemicalSpecies.Water, 1)] }

-- Define a function to calculate the moles of product formed
def molesOfProductFormed (reaction : ChemicalReaction) (limitingReactant : ChemicalSpecies) (molesOfLimitingReactant : ℚ) (product : ChemicalSpecies) : ℚ :=
  sorry -- Implementation details omitted

-- Theorem statement
theorem urea_formation_moles :
  molesOfProductFormed ureaFormationReaction ChemicalSpecies.CarbonDioxide 1 ChemicalSpecies.Urea = 1 :=
sorry

end urea_formation_moles_l1308_130802


namespace sequence_properties_l1308_130885

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := a n + n^2 - 1

def b_relation (n : ℕ) (a b : ℕ → ℝ) : Prop :=
  3^n * b (n+1) = (n+1) * a (n+1) - n * a n

theorem sequence_properties (a b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, S n a = a n + n^2 - 1) →
  (∀ n, b_relation n a b) →
  b 1 = 3 →
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, b n = (4*n - 1) / 3^(n-1)) ∧
  (∀ n, T n = 15/2 - (4*n + 5) / (2 * 3^(n-1))) ∧
  (∀ n > 3, T n ≥ 7) ∧
  (T 3 < 7) :=
by sorry

end sequence_properties_l1308_130885


namespace correct_mean_calculation_l1308_130873

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 →
  initial_mean = 150 →
  incorrect_value = 135 →
  correct_value = 165 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = 151 := by
  sorry

end correct_mean_calculation_l1308_130873


namespace lighting_effect_improves_l1308_130897

theorem lighting_effect_improves (a b m : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
  a / b < (a + m) / (b + m) := by
  sorry

end lighting_effect_improves_l1308_130897


namespace no_triangle_from_tangent_line_l1308_130841

/-- Given a line ax + by + c = 0 (where a, b, and c are positive) tangent to the circle x^2 + y^2 = 2,
    there does not exist a triangle with side lengths a, b, and c. -/
theorem no_triangle_from_tangent_line (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_tangent : ∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 = 2) :
  ¬ ∃ (A B C : ℝ × ℝ), 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = c ∧
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = a ∧
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = b :=
by sorry

end no_triangle_from_tangent_line_l1308_130841
