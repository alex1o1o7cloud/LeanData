import Mathlib

namespace NUMINAMATH_CALUDE_cube_face_perimeter_l2198_219822

-- Define the volume of the cube
def cube_volume : ℝ := 216

-- Define the function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Define the function to calculate the perimeter of a square given its side length
def square_perimeter (side : ℝ) : ℝ := 4 * side

-- Theorem statement
theorem cube_face_perimeter :
  square_perimeter (side_length cube_volume) = 24 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l2198_219822


namespace NUMINAMATH_CALUDE_lagaan_collection_l2198_219839

/-- The total amount of lagaan collected from a village, given the payment of one farmer and their land proportion. -/
theorem lagaan_collection (farmer_payment : ℝ) (farmer_land_proportion : ℝ) 
  (h1 : farmer_payment = 480) 
  (h2 : farmer_land_proportion = 0.23255813953488372 / 100) : 
  (farmer_payment / farmer_land_proportion) = 206400000 := by
  sorry

end NUMINAMATH_CALUDE_lagaan_collection_l2198_219839


namespace NUMINAMATH_CALUDE_luke_good_games_l2198_219805

def budget : ℕ := 100
def price_a : ℕ := 15
def price_b : ℕ := 8
def price_c : ℕ := 6
def num_a : ℕ := 3
def num_b : ℕ := 5
def sold_games : ℕ := 2
def sold_price : ℕ := 12
def broken_a : ℕ := 3
def broken_b : ℕ := 2

def remaining_budget : ℕ := budget - (num_a * price_a + num_b * price_b) + (sold_games * sold_price)

def num_c : ℕ := remaining_budget / price_c

theorem luke_good_games : 
  (num_a - broken_a) + (num_b - broken_b) + num_c = 9 :=
sorry

end NUMINAMATH_CALUDE_luke_good_games_l2198_219805


namespace NUMINAMATH_CALUDE_distance_between_points_l2198_219848

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 17)
  let p2 : ℝ × ℝ := (10, 3)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  distance = 2 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2198_219848


namespace NUMINAMATH_CALUDE_jones_family_puzzle_l2198_219871

/-- Represents a 4-digit number where one digit appears three times and another once -/
structure LicensePlate where
  digits : Fin 4 → Nat
  pattern : ∃ (a b : Nat), (∀ i, digits i = a ∨ digits i = b) ∧
                           (∃ j, digits j ≠ digits ((j + 1) % 4))

/-- Mr. Jones' family setup -/
structure JonesFamily where
  license : LicensePlate
  children_ages : Finset Nat
  jones_age : Nat
  h1 : children_ages.card = 8
  h2 : 12 ∈ children_ages
  h3 : ∀ age ∈ children_ages, license.digits 0 * 1000 + license.digits 1 * 100 + 
                               license.digits 2 * 10 + license.digits 3 % age = 0
  h4 : jones_age = license.digits 1 * 10 + license.digits 0

theorem jones_family_puzzle (family : JonesFamily) : 11 ∉ family.children_ages := by
  sorry

end NUMINAMATH_CALUDE_jones_family_puzzle_l2198_219871


namespace NUMINAMATH_CALUDE_inequality_chain_l2198_219835

theorem inequality_chain (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x + y = 1) :
  x < 2*x*y ∧ 2*x*y < (x + y)/2 ∧ (x + y)/2 < y := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l2198_219835


namespace NUMINAMATH_CALUDE_initial_students_count_l2198_219837

/-- The number of students who got off the bus at the first stop -/
def students_off : ℕ := 3

/-- The number of students remaining on the bus after the first stop -/
def students_remaining : ℕ := 7

/-- The initial number of students on the bus -/
def initial_students : ℕ := students_remaining + students_off

theorem initial_students_count : initial_students = 10 := by sorry

end NUMINAMATH_CALUDE_initial_students_count_l2198_219837


namespace NUMINAMATH_CALUDE_income_comparison_l2198_219808

/-- Given that Mart's income is 30% more than Tim's income and 78% of Juan's income,
    prove that Tim's income is 40% less than Juan's income. -/
theorem income_comparison (tim mart juan : ℝ) 
  (h1 : mart = 1.3 * tim) 
  (h2 : mart = 0.78 * juan) : 
  tim = 0.6 * juan := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l2198_219808


namespace NUMINAMATH_CALUDE_geometric_sum_7_terms_l2198_219807

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_7_terms :
  let a : ℚ := 1/2
  let r : ℚ := -1/3
  let n : ℕ := 7
  geometric_sum a r n = 547/1458 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_7_terms_l2198_219807


namespace NUMINAMATH_CALUDE_three_numbers_with_square_sums_l2198_219811

theorem three_numbers_with_square_sums : ∃ (a b c : ℕ+), 
  (∃ (x : ℕ), (a + b + c : ℕ) = x^2) ∧
  (∃ (y : ℕ), (a + b : ℕ) = y^2) ∧
  (∃ (z : ℕ), (b + c : ℕ) = z^2) ∧
  (∃ (w : ℕ), (a + c : ℕ) = w^2) ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_with_square_sums_l2198_219811


namespace NUMINAMATH_CALUDE_percentage_of_120_to_50_l2198_219842

theorem percentage_of_120_to_50 : (120 : ℝ) / 50 * 100 = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_50_l2198_219842


namespace NUMINAMATH_CALUDE_distinct_book_selections_l2198_219825

theorem distinct_book_selections (n k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  Nat.choose n k = 455 := by
  sorry

end NUMINAMATH_CALUDE_distinct_book_selections_l2198_219825


namespace NUMINAMATH_CALUDE_angle_properties_l2198_219815

/-- Given an angle α whose terminal side passes through the point (sin(5π/6), cos(5π/6)),
    prove that α is in the fourth quadrant and the smallest positive angle with the same
    terminal side as α is 5π/3 -/
theorem angle_properties (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (5 * Real.pi / 6) ∧
                    r * Real.sin α = Real.sin (5 * Real.pi / 6)) →
  (Real.cos α > 0 ∧ Real.sin α < 0) ∧
  (∀ β : Real, β > 0 ∧ Real.cos β = Real.cos α ∧ Real.sin β = Real.sin α → β ≥ 5 * Real.pi / 3) ∧
  (Real.cos (5 * Real.pi / 3) = Real.cos α ∧ Real.sin (5 * Real.pi / 3) = Real.sin α) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l2198_219815


namespace NUMINAMATH_CALUDE_ceiling_sqrt_180_l2198_219893

theorem ceiling_sqrt_180 : ⌈Real.sqrt 180⌉ = 14 := by
  have h : 13 < Real.sqrt 180 ∧ Real.sqrt 180 < 14 := by sorry
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_180_l2198_219893


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2198_219887

/-- The eccentricity of an ellipse with given conditions is between 0 and 2√5/5 -/
theorem ellipse_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt (1 - b^2 / a^2)
  let l := {p : ℝ × ℝ | p.2 = 1/2 * (p.1 + a)}
  let C₁ := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  let C₂ := {p : ℝ × ℝ | p.1^2 + p.2^2 = b^2}
  (∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ l ∩ C₂ ∧ q ∈ l ∩ C₂) →
  0 < e ∧ e < 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2198_219887


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l2198_219876

/-- Given a line with equation y = -1/2 * x + 1, prove that (2, -1) is a valid direction vector. -/
theorem direction_vector_of_line (x y : ℝ) :
  y = -1/2 * x + 1 →
  ∃ (t : ℝ), (x + 2*t, y - t) = (x, y) :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l2198_219876


namespace NUMINAMATH_CALUDE_total_lemons_l2198_219886

/-- The number of lemons each person has -/
structure LemonCounts where
  levi : ℕ
  jayden : ℕ
  eli : ℕ
  ian : ℕ

/-- The conditions of the lemon problem -/
def lemon_problem (c : LemonCounts) : Prop :=
  c.levi = 5 ∧
  c.jayden = c.levi + 6 ∧
  c.jayden * 3 = c.eli ∧
  c.eli * 2 = c.ian

/-- The theorem stating the total number of lemons -/
theorem total_lemons (c : LemonCounts) :
  lemon_problem c → c.levi + c.jayden + c.eli + c.ian = 115 := by
  sorry

end NUMINAMATH_CALUDE_total_lemons_l2198_219886


namespace NUMINAMATH_CALUDE_sum_and_double_l2198_219872

theorem sum_and_double : 2 * (2/20 + 3/30 + 4/40) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l2198_219872


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_neg_two_l2198_219840

theorem tan_pi_minus_alpha_neg_two (α : ℝ) (h : Real.tan (π - α) = -2) :
  (Real.cos (2 * π - α) + 2 * Real.cos ((3 * π) / 2 - α)) /
  (Real.sin (π - α) - Real.sin (-π / 2 - α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_neg_two_l2198_219840


namespace NUMINAMATH_CALUDE_wand_price_l2198_219817

theorem wand_price (price : ℝ) (original_price : ℝ) : 
  price = 12 → price = (1/8) * original_price → original_price = 96 := by
sorry

end NUMINAMATH_CALUDE_wand_price_l2198_219817


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2198_219870

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2198_219870


namespace NUMINAMATH_CALUDE_tom_program_duration_l2198_219873

def combined_program_duration (bs_duration ph_d_duration : ℕ) : ℕ :=
  bs_duration + ph_d_duration

def accelerated_duration (total_duration : ℕ) (acceleration_factor : ℚ) : ℚ :=
  (total_duration : ℚ) * acceleration_factor

theorem tom_program_duration :
  let bs_duration : ℕ := 3
  let ph_d_duration : ℕ := 5
  let acceleration_factor : ℚ := 3 / 4
  let total_duration := combined_program_duration bs_duration ph_d_duration
  accelerated_duration total_duration acceleration_factor = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_program_duration_l2198_219873


namespace NUMINAMATH_CALUDE_raffle_tickets_sold_l2198_219899

theorem raffle_tickets_sold (ticket_price : ℚ) (total_donations : ℚ) (total_raised : ℚ) :
  ticket_price = 2 →
  total_donations = 50 →
  total_raised = 100 →
  (total_raised - total_donations) / ticket_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_raffle_tickets_sold_l2198_219899


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2198_219833

theorem imaginary_part_of_complex_fraction (z : ℂ) : 
  z = (Complex.I : ℂ) / (1 + 2 * Complex.I) → Complex.im z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2198_219833


namespace NUMINAMATH_CALUDE_inequality_proof_l2198_219862

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + c + a)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2198_219862


namespace NUMINAMATH_CALUDE_equation_solution_l2198_219836

theorem equation_solution (x : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 3*x)) + Real.sqrt (3 + Real.sqrt (1 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 10 + 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2198_219836


namespace NUMINAMATH_CALUDE_eighth_box_books_l2198_219838

theorem eighth_box_books (total_books : ℕ) (num_boxes : ℕ) (books_per_box : ℕ) 
  (h1 : total_books = 800)
  (h2 : num_boxes = 8)
  (h3 : books_per_box = 105) :
  total_books - (num_boxes - 1) * books_per_box = 65 := by
  sorry

end NUMINAMATH_CALUDE_eighth_box_books_l2198_219838


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l2198_219853

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 95 → speed2 = 60 → (speed1 + speed2) / 2 = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l2198_219853


namespace NUMINAMATH_CALUDE_tv_price_increase_l2198_219800

theorem tv_price_increase (x : ℝ) : 
  (1 + 0.3) * (1 + x) = 1 + 0.5600000000000001 ↔ x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l2198_219800


namespace NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l2198_219803

theorem infinite_solutions_diophantine_equation (n : ℤ) :
  ∃ (S : Set (ℕ × ℕ × ℕ)), Set.Infinite S ∧
    ∀ (x y z : ℕ), (x, y, z) ∈ S → (x^2 : ℤ) + y^2 - z^2 = n :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l2198_219803


namespace NUMINAMATH_CALUDE_system_solution_l2198_219802

theorem system_solution (x y m : ℝ) : 
  (3 * x + 2 * y = 4 * m - 5 ∧ 
   2 * x + 3 * y = m ∧ 
   x + y = 2) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2198_219802


namespace NUMINAMATH_CALUDE_constant_vertex_l2198_219889

/-- The function f(x) = a^(x-2) + 1 always passes through the point (2, 2) for a > 0 and a ≠ 1 -/
theorem constant_vertex (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_vertex_l2198_219889


namespace NUMINAMATH_CALUDE_angela_december_sleep_l2198_219816

/-- The number of hours Angela slept every night in December -/
def december_sleep_hours : ℝ := sorry

/-- The number of hours Angela slept every night in January -/
def january_sleep_hours : ℝ := 8.5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The number of days in January -/
def january_days : ℕ := 31

/-- The additional hours of sleep Angela got in January compared to December -/
def additional_sleep : ℝ := 62

theorem angela_december_sleep :
  december_sleep_hours = 6.5 :=
by
  sorry

end NUMINAMATH_CALUDE_angela_december_sleep_l2198_219816


namespace NUMINAMATH_CALUDE_investment_comparison_l2198_219882

def initial_aa : ℝ := 150
def initial_bb : ℝ := 120
def initial_cc : ℝ := 100

def year1_aa_change : ℝ := 1.15
def year1_bb_change : ℝ := 0.70
def year1_cc_change : ℝ := 1.00

def year2_aa_change : ℝ := 0.85
def year2_bb_change : ℝ := 1.20
def year2_cc_change : ℝ := 1.00

def year3_aa_change : ℝ := 1.10
def year3_bb_change : ℝ := 0.95
def year3_cc_change : ℝ := 1.05

def final_aa : ℝ := initial_aa * year1_aa_change * year2_aa_change * year3_aa_change
def final_bb : ℝ := initial_bb * year1_bb_change * year2_bb_change * year3_bb_change
def final_cc : ℝ := initial_cc * year1_cc_change * year2_cc_change * year3_cc_change

theorem investment_comparison : final_bb < final_cc ∧ final_cc < final_aa :=
sorry

end NUMINAMATH_CALUDE_investment_comparison_l2198_219882


namespace NUMINAMATH_CALUDE_gcd_of_128_144_512_l2198_219828

theorem gcd_of_128_144_512 : Nat.gcd 128 (Nat.gcd 144 512) = 16 := by sorry

end NUMINAMATH_CALUDE_gcd_of_128_144_512_l2198_219828


namespace NUMINAMATH_CALUDE_kevin_distance_after_six_hops_l2198_219863

/-- Kevin's hopping journey on a number line -/
def kevin_hop (total_distance : ℚ) (first_hop_fraction : ℚ) (subsequent_hop_fraction : ℚ) (num_hops : ℕ) : ℚ :=
  let first_hop := first_hop_fraction * total_distance
  let remaining_distance := total_distance - first_hop
  let subsequent_hops := remaining_distance * (1 - (1 - subsequent_hop_fraction) ^ (num_hops - 1))
  first_hop + subsequent_hops

/-- The theorem stating the distance Kevin has hopped after six hops -/
theorem kevin_distance_after_six_hops :
  kevin_hop 2 (1/4) (2/3) 6 = 1071/243 := by
  sorry

end NUMINAMATH_CALUDE_kevin_distance_after_six_hops_l2198_219863


namespace NUMINAMATH_CALUDE_max_value_is_six_range_of_m_l2198_219821

-- Define the problem setup
def problem_setup (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 6

-- Define the maximum value function
def max_value (a b c : ℝ) : ℝ := a + 2*b + c

-- Theorem for the maximum value
theorem max_value_is_six (a b c : ℝ) (h : problem_setup a b c) :
  ∃ (M : ℝ), (∀ (a' b' c' : ℝ), problem_setup a' b' c' → max_value a' b' c' ≤ M) ∧
             M = 6 :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x + m| ≥ 6) ↔ (m ≥ 7 ∨ m ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_max_value_is_six_range_of_m_l2198_219821


namespace NUMINAMATH_CALUDE_running_preference_related_to_gender_l2198_219854

/-- Represents the contingency table for students liking running --/
structure RunningPreference where
  total_students : Nat
  boys : Nat
  girls : Nat
  girls_like_running : Nat
  boys_dont_like_running : Nat

/-- Calculates the K^2 value for the contingency table --/
def calculate_k_squared (pref : RunningPreference) : Rat :=
  let boys_like_running := pref.boys - pref.boys_dont_like_running
  let girls_dont_like_running := pref.girls - pref.girls_like_running
  let N := pref.total_students
  let a := boys_like_running
  let b := pref.boys_dont_like_running
  let c := pref.girls_like_running
  let d := girls_dont_like_running
  (N * (a * d - b * c)^2 : Rat) / ((a + c) * (b + d) * (a + b) * (c + d))

/-- Theorem stating that the K^2 value exceeds the critical value --/
theorem running_preference_related_to_gender (pref : RunningPreference) 
  (h1 : pref.total_students = 200)
  (h2 : pref.boys = 120)
  (h3 : pref.girls = 80)
  (h4 : pref.girls_like_running = 30)
  (h5 : pref.boys_dont_like_running = 50)
  (critical_value : Rat := 6635 / 1000) :
  calculate_k_squared pref > critical_value := by
  sorry

end NUMINAMATH_CALUDE_running_preference_related_to_gender_l2198_219854


namespace NUMINAMATH_CALUDE_largest_int_less_100_rem_4_div_7_l2198_219859

theorem largest_int_less_100_rem_4_div_7 : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_int_less_100_rem_4_div_7_l2198_219859


namespace NUMINAMATH_CALUDE_m_range_l2198_219818

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, |x - m| < 1 ↔ 1/3 < x ∧ x < 1/2) → 
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2198_219818


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2198_219841

theorem quadratic_equation_result (y : ℂ) : 
  3 * y^2 + 2 * y + 1 = 0 → (6 * y + 5)^2 = -7 + 12 * Complex.I * Real.sqrt 2 ∨ 
                              (6 * y + 5)^2 = -7 - 12 * Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2198_219841


namespace NUMINAMATH_CALUDE_power_division_equality_l2198_219847

theorem power_division_equality : 8^15 / 64^3 = 8^9 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l2198_219847


namespace NUMINAMATH_CALUDE_tangent_line_x_squared_at_one_l2198_219806

/-- The equation of the tangent line to y = x^2 at x = 1 is y = 2x - 1 -/
theorem tangent_line_x_squared_at_one :
  let f (x : ℝ) := x^2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2*x - 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_x_squared_at_one_l2198_219806


namespace NUMINAMATH_CALUDE_cubic_roots_l2198_219852

theorem cubic_roots : 
  ∀ x : ℝ, x^3 + 3*x^2 - 6*x - 8 = 0 ↔ x = -1 ∨ x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_l2198_219852


namespace NUMINAMATH_CALUDE_sum_max_min_xy_xz_yz_l2198_219858

/-- Given real numbers x, y, and z satisfying 5(x + y + z) = x^2 + y^2 + z^2,
    the sum of the maximum value of xy + xz + yz and 10 times its minimum value is 150. -/
theorem sum_max_min_xy_xz_yz (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  ∃ (N n : ℝ),
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 →
      a * b + a * c + b * c ≤ N ∧ n ≤ a * b + a * c + b * c) ∧
    N + 10 * n = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_max_min_xy_xz_yz_l2198_219858


namespace NUMINAMATH_CALUDE_three_solutions_inequality_l2198_219826

theorem three_solutions_inequality (a : ℝ) : 
  (∃! (s : Finset ℕ), s.card = 3 ∧ 
    (∀ x : ℕ, x ∈ s ↔ (x > 0 ∧ 3 * (x - 1) < 2 * (x + a) - 5))) ↔ 
  (5/2 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_three_solutions_inequality_l2198_219826


namespace NUMINAMATH_CALUDE_car_sale_percentage_increase_l2198_219877

theorem car_sale_percentage_increase 
  (P : ℝ) 
  (discount : ℝ) 
  (profit : ℝ) 
  (buying_price : ℝ) 
  (selling_price : ℝ) :
  discount = 0.1 →
  profit = 0.62000000000000014 →
  buying_price = P * (1 - discount) →
  selling_price = P * (1 + profit) →
  (selling_price - buying_price) / buying_price = 0.8000000000000002 :=
by sorry

end NUMINAMATH_CALUDE_car_sale_percentage_increase_l2198_219877


namespace NUMINAMATH_CALUDE_f_range_is_1_2_5_l2198_219801

def f (x : Int) : Int := x^2 + 1

def domain : Set Int := {-1, 0, 1, 2}

theorem f_range_is_1_2_5 : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_f_range_is_1_2_5_l2198_219801


namespace NUMINAMATH_CALUDE_ratio_of_bases_l2198_219860

/-- An isosceles trapezoid circumscribed about a circle -/
structure CircumscribedTrapezoid where
  /-- The longer base of the trapezoid -/
  AD : ℝ
  /-- The shorter base of the trapezoid -/
  BC : ℝ
  /-- The ratio of AN to NM, where N is the intersection of AM and the circle -/
  k : ℝ
  /-- AD is longer than BC -/
  h_AD_gt_BC : AD > BC
  /-- The trapezoid is isosceles -/
  h_isosceles : True
  /-- The trapezoid is circumscribed about a circle -/
  h_circumscribed : True
  /-- The circle touches one of the non-parallel sides -/
  h_touches_side : True
  /-- AM intersects the circle at N -/
  h_AM_intersects : True

/-- The ratio of the longer base to the shorter base in a circumscribed isosceles trapezoid -/
theorem ratio_of_bases (t : CircumscribedTrapezoid) : t.AD / t.BC = 8 * t.k - 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_bases_l2198_219860


namespace NUMINAMATH_CALUDE_barbara_tuna_packs_l2198_219813

/-- The number of tuna packs Barbara bought -/
def tuna_packs : ℕ := sorry

/-- The price of each tuna pack in dollars -/
def tuna_price : ℚ := 2

/-- The number of water bottles Barbara bought -/
def water_bottles : ℕ := 4

/-- The price of each water bottle in dollars -/
def water_price : ℚ := (3 : ℚ) / 2

/-- The amount spent on different goods in dollars -/
def different_goods_cost : ℚ := 40

/-- The total amount Barbara paid in dollars -/
def total_paid : ℚ := 56

theorem barbara_tuna_packs : 
  tuna_packs = 5 ∧ 
  (tuna_packs : ℚ) * tuna_price + (water_bottles : ℚ) * water_price + different_goods_cost = total_paid :=
sorry

end NUMINAMATH_CALUDE_barbara_tuna_packs_l2198_219813


namespace NUMINAMATH_CALUDE_min_value_theorem_l2198_219830

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  2/x + 9/(1-2*x) ≥ 25 ∧ ∃ y ∈ Set.Ioo 0 (1/2), 2/y + 9/(1-2*y) = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2198_219830


namespace NUMINAMATH_CALUDE_circus_cages_l2198_219820

theorem circus_cages (n : ℕ) (ways : ℕ) (h1 : n = 6) (h2 : ways = 240) :
  ∃ x : ℕ, x = 3 ∧ (n! / x! = ways) :=
by sorry

end NUMINAMATH_CALUDE_circus_cages_l2198_219820


namespace NUMINAMATH_CALUDE_square_roots_problem_l2198_219857

theorem square_roots_problem (a : ℝ) : 
  (∃ x : ℝ, (3 * a + 2)^2 = x ∧ (a + 14)^2 = x) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2198_219857


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2198_219804

-- Problem 1
theorem problem_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((x^2 - 2*x) / (x^2 - 4*x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2198_219804


namespace NUMINAMATH_CALUDE_investment_partnership_profit_share_l2198_219856

/-- Investment partnership problem -/
theorem investment_partnership_profit_share
  (invest_A invest_B invest_C invest_D : ℚ)
  (total_profit : ℚ)
  (h1 : invest_A = 3 * invest_B)
  (h2 : invest_B = 2/3 * invest_C)
  (h3 : invest_D = 1/2 * invest_A)
  (h4 : total_profit = 19900) :
  invest_B / (invest_A + invest_B + invest_C + invest_D) * total_profit = 2842.86 := by
sorry


end NUMINAMATH_CALUDE_investment_partnership_profit_share_l2198_219856


namespace NUMINAMATH_CALUDE_circle_center_l2198_219843

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

theorem circle_center : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 4) ∧ h = 2 ∧ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l2198_219843


namespace NUMINAMATH_CALUDE_wax_spilled_amount_l2198_219832

/-- The amount of wax spilled before use -/
def wax_spilled (car_wax SUV_wax initial_wax remaining_wax : ℕ) : ℕ :=
  initial_wax - (car_wax + SUV_wax) - remaining_wax

/-- Theorem stating that the amount of wax spilled is 2 ounces -/
theorem wax_spilled_amount :
  wax_spilled 3 4 11 2 = 2 := by sorry

end NUMINAMATH_CALUDE_wax_spilled_amount_l2198_219832


namespace NUMINAMATH_CALUDE_worker_c_time_l2198_219849

/-- The time taken by worker c to complete the work alone, given the conditions -/
def time_c (time_abc time_a time_b : ℚ) : ℚ :=
  1 / (1 / time_abc - 1 / time_a - 1 / time_b)

/-- Theorem stating that under given conditions, worker c takes 18 days to finish the work alone -/
theorem worker_c_time (time_abc time_a time_b : ℚ) 
  (h_abc : time_abc = 4)
  (h_a : time_a = 12)
  (h_b : time_b = 9) :
  time_c time_abc time_a time_b = 18 := by
  sorry

#eval time_c 4 12 9

end NUMINAMATH_CALUDE_worker_c_time_l2198_219849


namespace NUMINAMATH_CALUDE_chocolate_theorem_l2198_219827

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The factor by which Alix's chocolates exceed Nick's -/
def alix_factor : ℕ := 3

/-- The number of chocolates mom took from Alix -/
def mom_took : ℕ := 5

/-- The difference in chocolates between Alix and Nick after mom took some -/
def chocolate_difference : ℕ := 15

theorem chocolate_theorem :
  (alix_factor * nick_chocolates - mom_took) - nick_chocolates = chocolate_difference := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l2198_219827


namespace NUMINAMATH_CALUDE_stock_price_decrease_l2198_219875

theorem stock_price_decrease (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_2006 := initial_price * 1.3
  let price_after_2007 := price_after_2006 * 1.2
  let decrease_percentage := (price_after_2007 - initial_price) / price_after_2007 * 100
  ∃ ε > 0, abs (decrease_percentage - 35.9) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l2198_219875


namespace NUMINAMATH_CALUDE_bells_lcm_l2198_219883

def bell_intervals : List ℕ := [18, 24, 30, 35]

theorem bells_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm 18 24) 30) 35 = 2520 := by sorry

end NUMINAMATH_CALUDE_bells_lcm_l2198_219883


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2198_219898

/-- The focal length of a hyperbola with equation x²- y²/4 = 1 is 2√5 -/
theorem hyperbola_focal_length : 
  let h : Set ((ℝ × ℝ) → Prop) := {f | ∃ (x y : ℝ), f (x, y) ↔ x^2 - y^2/4 = 1}
  ∃ (f : (ℝ × ℝ) → Prop), f ∈ h ∧ 
    (∃ (a b c : ℝ), a^2 = 1 ∧ b^2 = 4 ∧ c^2 = a^2 + b^2 ∧ 2*c = 2*Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2198_219898


namespace NUMINAMATH_CALUDE_remaining_oranges_l2198_219823

/-- The number of oranges Michaela needs to get full -/
def michaela_oranges : ℕ := 45

/-- The number of oranges Cassandra needs to get full -/
def cassandra_oranges : ℕ := 5 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 520

/-- The number of oranges remaining after Michaela and Cassandra have eaten until full -/
theorem remaining_oranges : total_oranges - (michaela_oranges + cassandra_oranges) = 250 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l2198_219823


namespace NUMINAMATH_CALUDE_equal_diagonal_polygon_l2198_219819

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : ℕ
  is_convex : Bool
  diagonals_equal : Bool

/-- Definition of a quadrilateral -/
def is_quadrilateral (p : ConvexPolygon) : Prop :=
  p.vertices = 4

/-- Definition of a pentagon -/
def is_pentagon (p : ConvexPolygon) : Prop :=
  p.vertices = 5

/-- Main theorem -/
theorem equal_diagonal_polygon (F : ConvexPolygon) 
  (h1 : F.vertices ≥ 4) 
  (h2 : F.is_convex = true) 
  (h3 : F.diagonals_equal = true) : 
  is_quadrilateral F ∨ is_pentagon F :=
sorry

end NUMINAMATH_CALUDE_equal_diagonal_polygon_l2198_219819


namespace NUMINAMATH_CALUDE_martha_troubleshooting_time_l2198_219891

/-- The total time Martha spent on router troubleshooting activities -/
def total_time (router_time hold_time yelling_time : ℕ) : ℕ :=
  router_time + hold_time + yelling_time

/-- Theorem stating the total time Martha spent on activities -/
theorem martha_troubleshooting_time :
  ∃ (router_time hold_time yelling_time : ℕ),
    router_time = 10 ∧
    hold_time = 6 * router_time ∧
    yelling_time = hold_time / 2 ∧
    total_time router_time hold_time yelling_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_martha_troubleshooting_time_l2198_219891


namespace NUMINAMATH_CALUDE_longitude_latitude_unique_identification_l2198_219844

/-- A point on the Earth's surface --/
structure EarthPoint where
  longitude : Real
  latitude : Real

/-- Function to determine if a description can uniquely identify a point --/
def canUniquelyIdentify (description : EarthPoint → Prop) : Prop :=
  ∀ (p1 p2 : EarthPoint), description p1 → description p2 → p1 = p2

/-- Theorem stating that longitude and latitude can uniquely identify a point --/
theorem longitude_latitude_unique_identification :
  canUniquelyIdentify (λ p : EarthPoint => p.longitude = 118 ∧ p.latitude = 40) :=
sorry

end NUMINAMATH_CALUDE_longitude_latitude_unique_identification_l2198_219844


namespace NUMINAMATH_CALUDE_x_seventh_plus_64x_squared_l2198_219868

theorem x_seventh_plus_64x_squared (x : ℝ) (h : x^3 + 4*x = 8) : x^7 + 64*x^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_x_seventh_plus_64x_squared_l2198_219868


namespace NUMINAMATH_CALUDE_non_congruent_triangle_count_l2198_219831

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The set of 10 points in the problem -/
def problem_points : Finset Point2D := sorry

/-- Predicate to check if three points form a triangle -/
def is_triangle (p q r : Point2D) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def are_congruent (t1 t2 : Point2D × Point2D × Point2D) : Prop := sorry

/-- The set of all possible triangles formed by the problem points -/
def all_triangles : Finset (Point2D × Point2D × Point2D) := sorry

/-- The set of non-congruent triangles -/
def non_congruent_triangles : Finset (Point2D × Point2D × Point2D) := sorry

theorem non_congruent_triangle_count :
  Finset.card non_congruent_triangles = 12 := by sorry

end NUMINAMATH_CALUDE_non_congruent_triangle_count_l2198_219831


namespace NUMINAMATH_CALUDE_remainder_theorem_l2198_219809

theorem remainder_theorem (N : ℤ) : N % 13 = 3 → N % 39 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2198_219809


namespace NUMINAMATH_CALUDE_circle_center_l2198_219865

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + y^2 - 8*y - 48 = 0

/-- The center of a circle given by its coordinates -/
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (x + h)^2 + (y - k)^2

theorem circle_center :
  is_center (-3) 4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2198_219865


namespace NUMINAMATH_CALUDE_johns_age_problem_l2198_219810

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem johns_age_problem :
  ∃! x : ℕ, x > 0 ∧ is_perfect_square (x - 5) ∧ is_perfect_cube (x + 3) ∧ x = 69 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_problem_l2198_219810


namespace NUMINAMATH_CALUDE_train_speed_l2198_219894

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 10) :
  length / time = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2198_219894


namespace NUMINAMATH_CALUDE_apples_count_l2198_219888

/-- The number of apples in the market -/
def apples : ℕ := sorry

/-- The number of oranges in the market -/
def oranges : ℕ := sorry

/-- The number of bananas in the market -/
def bananas : ℕ := sorry

/-- There are 27 more apples than oranges -/
axiom apples_oranges_diff : apples = oranges + 27

/-- There are 11 more oranges than bananas -/
axiom oranges_bananas_diff : oranges = bananas + 11

/-- The total number of fruits is 301 -/
axiom total_fruits : apples + oranges + bananas = 301

/-- The number of apples in the market is 122 -/
theorem apples_count : apples = 122 := by sorry

end NUMINAMATH_CALUDE_apples_count_l2198_219888


namespace NUMINAMATH_CALUDE_julie_weed_hours_l2198_219814

/-- Represents Julie's landscaping business earnings --/
def julie_earnings (weed_hours : ℕ) : ℕ :=
  let mowing_rate : ℕ := 4
  let weed_rate : ℕ := 8
  let mowing_hours : ℕ := 25
  2 * (mowing_rate * mowing_hours + weed_rate * weed_hours)

/-- Proves that Julie spent 3 hours pulling weeds in September --/
theorem julie_weed_hours : 
  ∃ (weed_hours : ℕ), julie_earnings weed_hours = 248 ∧ weed_hours = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_julie_weed_hours_l2198_219814


namespace NUMINAMATH_CALUDE_trigonometric_equation_l2198_219861

theorem trigonometric_equation (x : ℝ) :
  2 * Real.cos x - 5 * Real.sin x = 2 →
  Real.sin x + 2 * Real.cos x = 2 ∨ Real.sin x + 2 * Real.cos x = -62/29 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l2198_219861


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2198_219884

def initial_apples : ℕ := sorry

def apples_to_students : ℕ := 30
def number_of_pies : ℕ := 7
def apples_per_pie : ℕ := 8

theorem cafeteria_apples : initial_apples = 86 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2198_219884


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l2198_219812

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l2198_219812


namespace NUMINAMATH_CALUDE_anna_weekly_salary_l2198_219846

/-- Represents a worker's salary information -/
structure WorkerSalary where
  daysWorkedPerWeek : ℕ
  missedDays : ℕ
  deductionAmount : ℚ

/-- Calculates the usual weekly salary of a worker -/
def usualWeeklySalary (w : WorkerSalary) : ℚ :=
  (w.deductionAmount / w.missedDays) * w.daysWorkedPerWeek

theorem anna_weekly_salary :
  let anna : WorkerSalary := {
    daysWorkedPerWeek := 5,
    missedDays := 2,
    deductionAmount := 985
  }
  usualWeeklySalary anna = 2462.5 := by
  sorry

end NUMINAMATH_CALUDE_anna_weekly_salary_l2198_219846


namespace NUMINAMATH_CALUDE_rice_weight_in_pounds_l2198_219897

/-- Given rice divided equally into 4 containers, with each container having 33 ounces,
    and 1 pound being equal to 16 ounces, the total weight of rice in pounds is 8.25. -/
theorem rice_weight_in_pounds :
  let num_containers : ℕ := 4
  let ounces_per_container : ℚ := 33
  let ounces_per_pound : ℚ := 16
  let total_ounces : ℚ := num_containers * ounces_per_container
  let total_pounds : ℚ := total_ounces / ounces_per_pound
  total_pounds = 8.25 := by sorry

end NUMINAMATH_CALUDE_rice_weight_in_pounds_l2198_219897


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l2198_219834

/-- The percentage of liquid X in solution A -/
def percent_X_in_A : ℝ := 1.464

/-- The percentage of liquid X in solution B -/
def percent_X_in_B : ℝ := 1.8

/-- The weight of solution A in grams -/
def weight_A : ℝ := 500

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in the resulting mixture -/
def percent_X_in_mixture : ℝ := 1.66

theorem liquid_X_percentage :
  percent_X_in_A * weight_A / 100 + percent_X_in_B * weight_B / 100 =
  percent_X_in_mixture * (weight_A + weight_B) / 100 := by
  sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l2198_219834


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2198_219869

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (h1 : a > b) (h2 : b > 0) : 
  a + b > 2 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2198_219869


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2198_219855

theorem algebraic_expression_equality (x : ℝ) : 
  3 * x^2 - 4 * x = 6 → 6 * x^2 - 8 * x - 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2198_219855


namespace NUMINAMATH_CALUDE_four_digit_sum_l2198_219866

/-- Given four distinct non-zero digits, the sum of all four-digit numbers formed using these digits without repetition is 73,326 if and only if the digits are 1, 2, 3, and 5. -/
theorem four_digit_sum (a b c d : ℕ) : 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →  -- non-zero digits
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- distinct digits
  (6 * (a + b + c + d) * 1111 = 73326) →  -- sum condition
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l2198_219866


namespace NUMINAMATH_CALUDE_supplementary_angle_theorem_l2198_219851

theorem supplementary_angle_theorem (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_theorem_l2198_219851


namespace NUMINAMATH_CALUDE_outfit_combinations_l2198_219885

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 6

/-- The number of different types of clothing items -/
def num_items : ℕ := 4

/-- The number of valid outfit combinations -/
def valid_combinations : ℕ := num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3)

theorem outfit_combinations :
  valid_combinations = 360 :=
sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2198_219885


namespace NUMINAMATH_CALUDE_rectangle_area_l2198_219829

/-- The area of a rectangle with length thrice its breadth and perimeter 104 meters is 507 square meters. -/
theorem rectangle_area (breadth length perimeter area : ℝ) : 
  length = 3 * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 104 →
  area = length * breadth →
  area = 507 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2198_219829


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l2198_219874

/-- Proves that under given conditions, 35% of seeds in the second plot germinate -/
theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 15/100 →
  total_germination_rate = 23/100 →
  (germination_rate_plot1 * seeds_plot1 + 
   (total_germination_rate * (seeds_plot1 + seeds_plot2) - germination_rate_plot1 * seeds_plot1)) / seeds_plot2 = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l2198_219874


namespace NUMINAMATH_CALUDE_special_function_inequality_l2198_219892

/-- A function f: ℝ → ℝ satisfying specific conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  f_diff : Differentiable ℝ f
  f_even : ∀ x, f (x + 1) = f (-x + 1)
  f_derivative_sign : ∀ x, (x - 1) * (deriv f x) < 0

/-- Theorem stating that for a SpecialFunction f, if x₁ < x₂ and x₁ + x₂ > 2, then f(x₁) > f(x₂) -/
theorem special_function_inequality (f : SpecialFunction) (x₁ x₂ : ℝ) 
  (h_lt : x₁ < x₂) (h_sum : x₁ + x₂ > 2) : f.f x₁ > f.f x₂ := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l2198_219892


namespace NUMINAMATH_CALUDE_first_group_size_l2198_219864

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℝ := 28

/-- The number of men in the second group -/
def men_second_group : ℝ := 20

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℝ := 22.4

/-- The work done by a group is inversely proportional to the time taken -/
axiom work_time_inverse_proportion {men days : ℝ} : men * days = (men_second_group * days_second_group)

theorem first_group_size : ∃ (men : ℝ), men * days_first_group = men_second_group * days_second_group ∧ men = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l2198_219864


namespace NUMINAMATH_CALUDE_total_spent_equals_1150_l2198_219879

-- Define the quantities of toys
def elder_action_figures : ℕ := 60
def younger_action_figures : ℕ := 3 * elder_action_figures
def cars : ℕ := 20
def stuffed_animals : ℕ := 10

-- Define the prices of toys
def elder_action_figure_price : ℕ := 5
def younger_action_figure_price : ℕ := 4
def car_price : ℕ := 3
def stuffed_animal_price : ℕ := 7

-- Define the total cost function
def total_cost : ℕ :=
  elder_action_figures * elder_action_figure_price +
  younger_action_figures * younger_action_figure_price +
  cars * car_price +
  stuffed_animals * stuffed_animal_price

-- Theorem statement
theorem total_spent_equals_1150 : total_cost = 1150 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_1150_l2198_219879


namespace NUMINAMATH_CALUDE_solution_implies_value_l2198_219824

theorem solution_implies_value (a b : ℝ) : 
  (-a * 3 - b = 5 - 2 * 3) → (3 - 6 * a - 2 * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_value_l2198_219824


namespace NUMINAMATH_CALUDE_number_equation_l2198_219880

theorem number_equation (x : ℝ) : 3550 - (x / 20.04) = 3500 ↔ x = 1002 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2198_219880


namespace NUMINAMATH_CALUDE_rectangle_area_l2198_219896

/-- The area of a rectangle with length 2 and width 4 is 8 -/
theorem rectangle_area : ∀ (length width area : ℝ), 
  length = 2 → width = 4 → area = length * width → area = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2198_219896


namespace NUMINAMATH_CALUDE_intersecting_lines_m_value_l2198_219850

/-- Given three lines that intersect at a single point, prove that the value of m is -22/7 -/
theorem intersecting_lines_m_value (x y : ℚ) :
  y = 4 * x - 8 ∧
  y = -3 * x + 9 ∧
  y = 2 * x + m →
  m = -22 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_m_value_l2198_219850


namespace NUMINAMATH_CALUDE_unique_power_of_two_with_prepended_digit_l2198_219895

theorem unique_power_of_two_with_prepended_digit : 
  ∃! n : ℕ, 
    (∃ k : ℕ, n = 2^k) ∧ 
    (∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ ∃ m : ℕ, 10 * n + d = 2^m) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_power_of_two_with_prepended_digit_l2198_219895


namespace NUMINAMATH_CALUDE_cashew_nut_purchase_l2198_219890

/-- Prove that given the conditions of the nut purchase problem, the number of kilos of cashew nuts bought is 3. -/
theorem cashew_nut_purchase (cashew_price peanut_price peanut_amount total_weight avg_price : ℝ) 
  (h1 : cashew_price = 210)
  (h2 : peanut_price = 130)
  (h3 : peanut_amount = 2)
  (h4 : total_weight = 5)
  (h5 : avg_price = 178) :
  (total_weight - peanut_amount) = 3 := by
  sorry


end NUMINAMATH_CALUDE_cashew_nut_purchase_l2198_219890


namespace NUMINAMATH_CALUDE_jar_weight_percentage_l2198_219881

theorem jar_weight_percentage (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (h1 : jar_weight = 0.2 * (jar_weight + full_beans_weight))
  (h2 : 0.5 * full_beans_weight = full_beans_weight / 2) :
  (jar_weight + full_beans_weight / 2) / (jar_weight + full_beans_weight) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_jar_weight_percentage_l2198_219881


namespace NUMINAMATH_CALUDE_tangent_segment_equality_tangent_line_distance_equality_l2198_219867

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a tangent line to a circle
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  sorry

-- Define the distance between two lines
def line_distance (l1 l2 : Line) : ℝ :=
  sorry

-- Define the point of tangency
def point_of_tangency (l : Line) (c : Circle) : Point :=
  sorry

theorem tangent_segment_equality (c1 c2 : Circle) (l1 l2 l3 l4 : Line) 
  (h1 : is_tangent l1 c1) (h2 : is_tangent l2 c1)
  (h3 : is_tangent l3 c2) (h4 : is_tangent l4 c2) :
  let p1 := point_of_tangency l1 c1
  let p2 := point_of_tangency l2 c1
  let p3 := point_of_tangency l3 c2
  let p4 := point_of_tangency l4 c2
  distance p1 p3 = distance p2 p4 :=
sorry

theorem tangent_line_distance_equality (c1 c2 : Circle) (l1 l2 l3 l4 : Line)
  (h1 : is_tangent l1 c1) (h2 : is_tangent l2 c1)
  (h3 : is_tangent l3 c2) (h4 : is_tangent l4 c2) :
  line_distance l1 l3 = line_distance l2 l4 :=
sorry

end NUMINAMATH_CALUDE_tangent_segment_equality_tangent_line_distance_equality_l2198_219867


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2198_219878

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (x - 2 + 1/x)^4) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = 70 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2198_219878


namespace NUMINAMATH_CALUDE_snake_diet_l2198_219845

/-- The number of birds each snake eats per day in a forest ecosystem -/
def birds_per_snake (beetles_per_bird : ℕ) (snakes_per_jaguar : ℕ) (num_jaguars : ℕ) (total_beetles : ℕ) : ℕ :=
  (total_beetles / beetles_per_bird) / (num_jaguars * snakes_per_jaguar)

/-- Theorem stating that each snake eats 3 birds per day in the given ecosystem -/
theorem snake_diet :
  birds_per_snake 12 5 6 1080 = 3 := by
  sorry

#eval birds_per_snake 12 5 6 1080

end NUMINAMATH_CALUDE_snake_diet_l2198_219845
