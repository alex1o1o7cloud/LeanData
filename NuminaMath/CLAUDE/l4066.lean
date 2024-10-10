import Mathlib

namespace clerical_percentage_after_reduction_l4066_406642

/-- Represents a department in the company -/
structure Department where
  total : Nat
  clerical_fraction : Rat
  reduction : Rat

/-- Calculates the number of clerical staff in a department after reduction -/
def clerical_after_reduction (d : Department) : Rat :=
  (d.total : Rat) * d.clerical_fraction * (1 - d.reduction)

/-- The company structure with its departments -/
structure Company where
  dept_a : Department
  dept_b : Department
  dept_c : Department

/-- The specific company instance from the problem -/
def company_x : Company :=
  { dept_a := { total := 4000, clerical_fraction := 1/4, reduction := 1/4 },
    dept_b := { total := 6000, clerical_fraction := 1/6, reduction := 1/10 },
    dept_c := { total := 2000, clerical_fraction := 1/8, reduction := 0 } }

/-- Total number of employees in the company -/
def total_employees : Nat := 12000

/-- Theorem stating the percentage of clerical staff after reductions -/
theorem clerical_percentage_after_reduction :
  (clerical_after_reduction company_x.dept_a +
   clerical_after_reduction company_x.dept_b +
   clerical_after_reduction company_x.dept_c) /
  (total_employees : Rat) * 100 = 1900 / 12000 * 100 := by
  sorry

end clerical_percentage_after_reduction_l4066_406642


namespace range_of_m_l4066_406683

theorem range_of_m (x m : ℝ) : 
  (m > 0) →
  (∀ x, (|1 - (x - 1) / 3| ≤ 2) → (x^2 - 2*x + 1 - m^2 ≤ 0)) →
  (∃ x, (|1 - (x - 1) / 3| > 2) ∧ (x^2 - 2*x + 1 - m^2 ≤ 0)) →
  (0 < m ∧ m ≤ 3) :=
by sorry

end range_of_m_l4066_406683


namespace hydrogen_atoms_in_compound_l4066_406614

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  carbon : Nat
  hydrogen : Nat
  oxygen : Nat

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : Real
  hydrogen : Real
  oxygen : Real

/-- Calculates the molecular weight of a compound -/
def molecularWeight (formula : MolecularFormula) (weights : AtomicWeights) : Real :=
  formula.carbon * weights.carbon + formula.hydrogen * weights.hydrogen + formula.oxygen * weights.oxygen

/-- Theorem stating that the value of y in C6HyO7 is 8 for a molecular weight of 192 g/mol -/
theorem hydrogen_atoms_in_compound (weights : AtomicWeights) 
    (h_carbon : weights.carbon = 12.01)
    (h_hydrogen : weights.hydrogen = 1.01)
    (h_oxygen : weights.oxygen = 16.00) :
  ∃ y : Nat, y = 8 ∧ 
    molecularWeight { carbon := 6, hydrogen := y, oxygen := 7 } weights = 192 := by
  sorry

end hydrogen_atoms_in_compound_l4066_406614


namespace probability_three_heads_in_eight_tosses_l4066_406619

def coin_tosses : ℕ := 8
def heads_count : ℕ := 3

theorem probability_three_heads_in_eight_tosses :
  (Nat.choose coin_tosses heads_count) / (2 ^ coin_tosses) = 7 / 32 :=
by sorry

end probability_three_heads_in_eight_tosses_l4066_406619


namespace a_alone_finish_time_l4066_406603

/-- Represents the time taken by A alone to finish the job -/
def time_a : ℝ := 16

/-- Represents the time taken by A and B together to finish the job -/
def time_ab : ℝ := 40

/-- Represents the number of days A and B worked together -/
def days_together : ℝ := 10

/-- Represents the number of days A worked alone after B left -/
def days_a_alone : ℝ := 12

/-- Theorem stating that given the conditions, A alone can finish the job in 16 days -/
theorem a_alone_finish_time :
  (1 / time_a + 1 / time_ab) * days_together + (1 / time_a) * days_a_alone = 1 :=
sorry

end a_alone_finish_time_l4066_406603


namespace solution_set_implies_a_equals_two_existence_implies_m_greater_than_five_l4066_406624

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1
theorem solution_set_implies_a_equals_two :
  (∀ x, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∃ a, ∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∀ a, (∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2) :=
sorry

-- Theorem 2
theorem existence_implies_m_greater_than_five :
  (∃ x m, f 2 x + f 2 (x + 5) < m) →
  (∀ m, (∃ x, f 2 x + f 2 (x + 5) < m) → m > 5) :=
sorry

end solution_set_implies_a_equals_two_existence_implies_m_greater_than_five_l4066_406624


namespace solve_equation_l4066_406645

theorem solve_equation (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 := by
  sorry

end solve_equation_l4066_406645


namespace multiplication_value_proof_l4066_406658

theorem multiplication_value_proof : 
  let number : ℝ := 5.5
  let divisor : ℝ := 6
  let result : ℝ := 11
  let multiplier : ℝ := 12
  (number / divisor) * multiplier = result :=
by sorry

end multiplication_value_proof_l4066_406658


namespace nearest_town_distance_l4066_406660

theorem nearest_town_distance (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (d > 7 ∧ d < 8) :=
by
  sorry

end nearest_town_distance_l4066_406660


namespace cubic_root_ratio_l4066_406651

theorem cubic_root_ratio (p q r s : ℝ) (h : p ≠ 0) :
  (∀ x, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) →
  r / s = -5 / 12 := by
sorry

end cubic_root_ratio_l4066_406651


namespace intersection_of_A_and_B_l4066_406639

-- Define sets A and B
def A : Set ℝ := {x | x ≤ 7}
def B : Set ℝ := {x | x > 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 7} := by
  sorry

end intersection_of_A_and_B_l4066_406639


namespace coordinate_system_proof_l4066_406688

def M (m : ℝ) : ℝ × ℝ := (m - 2, 2 * m - 7)
def N (n : ℝ) : ℝ × ℝ := (n, 3)

theorem coordinate_system_proof :
  (∀ m : ℝ, M m = (m - 2, 2 * m - 7)) ∧
  (∀ n : ℝ, N n = (n, 3)) →
  (∀ m : ℝ, (M m).2 = 0 → m = 7/2 ∧ M m = (3/2, 0)) ∧
  (∀ m : ℝ, |m - 2| = |2 * m - 7| → m = 5 ∨ m = 3) ∧
  (∀ m n : ℝ, (M m).1 = (N n).1 ∧ |(M m).2 - (N n).2| = 2 → n = 4 ∨ n = 2) :=
by sorry

end coordinate_system_proof_l4066_406688


namespace negative_square_cubed_l4066_406678

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l4066_406678


namespace negation_equivalence_l4066_406644

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 2 < 0) ↔ (∀ x : ℝ, x^2 + x - 2 ≥ 0) := by
  sorry

end negation_equivalence_l4066_406644


namespace customized_bowling_ball_volume_l4066_406648

/-- The volume of a customized bowling ball -/
theorem customized_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 10
  let small_hole_diameter : ℝ := 2
  let large_hole_diameter : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let small_hole_volume := π * (small_hole_diameter / 2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2)^2 * hole_depth
  let total_hole_volume := 2 * small_hole_volume + 2 * large_hole_volume
  sphere_volume - total_hole_volume = 2239 * π :=
by sorry

end customized_bowling_ball_volume_l4066_406648


namespace lcm_hcf_problem_l4066_406685

/-- Given two positive integers with specific LCM and HCF, prove that if one number is 210, the other is 517 -/
theorem lcm_hcf_problem (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 47) (h3 : A = 210) : B = 517 := by
  sorry

end lcm_hcf_problem_l4066_406685


namespace triangle_inequality_theorem_l4066_406668

/-- Triangle inequality theorem for a triangle with side lengths a, b, c, and perimeter s -/
theorem triangle_inequality_theorem (a b c s : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = s) :
  (13/27 * s^2 ≤ a^2 + b^2 + c^2 + 4*a*b*c/s ∧ a^2 + b^2 + c^2 + 4*a*b*c/s < s^2/2) ∧
  (s^2/4 < a*b + b*c + c*a - 2*a*b*c/s ∧ a*b + b*c + c*a - 2*a*b*c/s ≤ 7/27 * s^2) := by
  sorry

end triangle_inequality_theorem_l4066_406668


namespace sqrt_sum_fractions_equals_sqrt29_over_4_l4066_406654

theorem sqrt_sum_fractions_equals_sqrt29_over_4 :
  Real.sqrt (9 / 36 + 25 / 16) = Real.sqrt 29 / 4 := by
  sorry

end sqrt_sum_fractions_equals_sqrt29_over_4_l4066_406654


namespace c_properties_l4066_406670

-- Define the given conditions
axiom sqrt_ab : ∃ a b : ℝ, Real.sqrt (a * b) = 99 * Real.sqrt 2
axiom sqrt_abc_nat : ∃ a b c : ℝ, ∃ n : ℕ, Real.sqrt (a * b * c) = n

-- Theorem to prove
theorem c_properties :
  ∃ a b c : ℝ,
  (∀ n : ℕ, Real.sqrt (a * b * c) = n) →
  (c ≠ Real.sqrt 2) ∧
  (∃ k : ℕ, c = 2 * k^2) ∧
  (∃ e : ℕ, e % 2 = 0 ∧ ¬(∀ n : ℕ, Real.sqrt (a * b * e) = n)) ∧
  (∀ m : ℕ, ∃ c' : ℝ, c' ≠ c ∧ ∀ n : ℕ, Real.sqrt (a * b * c') = n) :=
by
  sorry

end c_properties_l4066_406670


namespace unique_solution_for_exponential_equation_l4066_406671

theorem unique_solution_for_exponential_equation :
  ∀ a n : ℕ+, 3^(n : ℕ) = (a : ℕ)^2 - 16 → a = 5 ∧ n = 2 :=
by sorry

end unique_solution_for_exponential_equation_l4066_406671


namespace wire_service_reporters_l4066_406625

theorem wire_service_reporters (total : ℝ) 
  (country_x country_y country_z : ℝ)
  (xy_overlap yz_overlap xz_overlap xyz_overlap : ℝ)
  (finance environment social : ℝ)
  (h_total : total > 0)
  (h_x : country_x = 0.3 * total)
  (h_y : country_y = 0.2 * total)
  (h_z : country_z = 0.15 * total)
  (h_xy : xy_overlap = 0.05 * total)
  (h_yz : yz_overlap = 0.03 * total)
  (h_xz : xz_overlap = 0.02 * total)
  (h_xyz : xyz_overlap = 0.01 * total)
  (h_finance : finance = 0.1 * total)
  (h_environment : environment = 0.07 * total)
  (h_social : social = 0.05 * total) :
  (total - (country_x + country_y + country_z - xy_overlap - yz_overlap - xz_overlap + xyz_overlap) - 
   (finance + environment + social)) / total = 0.27 := by
sorry

end wire_service_reporters_l4066_406625


namespace sanxingdui_jinsha_visitor_l4066_406616

/-- Represents the four people in the problem -/
inductive Person : Type
  | A | B | C | D

/-- Represents the two archaeological sites -/
inductive Site : Type
  | Sanxingdui
  | Jinsha

/-- Predicate to represent if a person visited a site -/
def visited (p : Person) (s : Site) : Prop := sorry

/-- Predicate to represent if a person is telling the truth -/
def telling_truth (p : Person) : Prop := sorry

theorem sanxingdui_jinsha_visitor :
  (∃! p : Person, ∀ s : Site, visited p s) →
  (∃! p : Person, ¬telling_truth p) →
  (¬visited Person.A Site.Sanxingdui ∧ ¬visited Person.A Site.Jinsha) →
  (visited Person.B Site.Sanxingdui ↔ visited Person.A Site.Sanxingdui) →
  (visited Person.C Site.Jinsha ↔ visited Person.B Site.Jinsha) →
  (∀ s : Site, visited Person.D s → ¬visited Person.B s) →
  (∀ s : Site, visited Person.C s) :=
sorry

end sanxingdui_jinsha_visitor_l4066_406616


namespace solve_q_l4066_406605

theorem solve_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by sorry

end solve_q_l4066_406605


namespace complex_modulus_three_fourths_minus_three_i_l4066_406675

theorem complex_modulus_three_fourths_minus_three_i :
  Complex.abs (3/4 - 3*I) = (3 * Real.sqrt 17) / 4 := by
  sorry

end complex_modulus_three_fourths_minus_three_i_l4066_406675


namespace triangle_inequality_l4066_406657

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l4066_406657


namespace object_distance_in_one_hour_l4066_406661

/-- Proves that an object traveling at 3 feet per second will cover 10800 feet in one hour. -/
theorem object_distance_in_one_hour 
  (speed : ℝ) 
  (seconds_per_hour : ℕ) 
  (h1 : speed = 3) 
  (h2 : seconds_per_hour = 3600) : 
  speed * seconds_per_hour = 10800 := by
  sorry

end object_distance_in_one_hour_l4066_406661


namespace halfway_fraction_l4066_406674

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) :
  (a + b) / 2 = 19/24 := by
  sorry

end halfway_fraction_l4066_406674


namespace min_value_expression_min_value_achieved_l4066_406694

theorem min_value_expression (x y z : ℝ) 
  (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hz : -2 < z ∧ z < 2) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) ≥ 2 :=
sorry

theorem min_value_achieved (x y z : ℝ) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end min_value_expression_min_value_achieved_l4066_406694


namespace dandelions_to_grandmother_value_l4066_406697

/-- The number of dandelion puffs Caleb gave to his grandmother -/
def dandelions_to_grandmother (total : ℕ) (to_mom : ℕ) (to_sister : ℕ) (to_dog : ℕ) 
  (num_friends : ℕ) (to_each_friend : ℕ) : ℕ :=
  total - (to_mom + to_sister + to_dog + num_friends * to_each_friend)

theorem dandelions_to_grandmother_value : 
  dandelions_to_grandmother 40 3 3 2 3 9 = 5 := by sorry

end dandelions_to_grandmother_value_l4066_406697


namespace parabola_line_intersection_ratio_l4066_406689

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line with a slope angle -/
structure Line where
  angle : ℝ

/-- Focal distance ratio for a parabola intersected by a line -/
def focal_distance_ratio (parabola : Parabola) (line : Line) : ℝ :=
  sorry

theorem parabola_line_intersection_ratio 
  (parabola : Parabola) 
  (line : Line) 
  (h_angle : line.angle = 2*π/3) -- 120° in radians
  (A B : Point)
  (h_A : A.y^2 = 2*parabola.p*A.x ∧ A.x > 0 ∧ A.y > 0) -- A in first quadrant
  (h_B : B.y^2 = 2*parabola.p*B.x ∧ B.x > 0 ∧ B.y < 0) -- B in fourth quadrant
  : focal_distance_ratio parabola line = 1/3 := by
  sorry

end parabola_line_intersection_ratio_l4066_406689


namespace square_floor_tiles_l4066_406636

theorem square_floor_tiles (black_tiles : ℕ) (h : black_tiles = 57) :
  ∃ (side_length : ℕ),
    (2 * side_length - 1 = black_tiles) ∧
    (side_length * side_length = 841) :=
by sorry

end square_floor_tiles_l4066_406636


namespace large_bottle_price_calculation_l4066_406604

-- Define the variables
def large_bottles : ℕ := 1300
def small_bottles : ℕ := 750
def small_bottle_price : ℚ := 138 / 100
def average_price : ℚ := 17034 / 10000

-- Define the theorem
theorem large_bottle_price_calculation :
  ∃ (large_price : ℚ),
    (large_bottles * large_price + small_bottles * small_bottle_price) / (large_bottles + small_bottles) = average_price ∧
    abs (large_price - 189 / 100) < 1 / 100 := by
  sorry

end large_bottle_price_calculation_l4066_406604


namespace illumination_configurations_count_l4066_406686

/-- The number of different ways to illuminate n traffic lights, each with three possible states. -/
def illumination_configurations (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of different ways to illuminate n traffic lights,
    each with three possible states, is 3^n. -/
theorem illumination_configurations_count (n : ℕ) :
  illumination_configurations n = 3^n :=
by sorry

end illumination_configurations_count_l4066_406686


namespace distance_sum_bounds_l4066_406628

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is between 17 and 18 -/
theorem distance_sum_bounds (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 0) → D = (3, 4) → 
  17 < Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ∧
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) < 18 :=
by sorry

end distance_sum_bounds_l4066_406628


namespace trig_problem_l4066_406659

theorem trig_problem (θ φ : Real) 
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧ 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ 
  Real.cos θ = -(Real.sqrt 5 / 5) ∧ 
  Real.cos φ = -(Real.sqrt 2 / 10) := by
  sorry

end trig_problem_l4066_406659


namespace new_library_capacity_l4066_406695

theorem new_library_capacity 
  (M : ℚ) -- Millicent's total number of books
  (H : ℚ) -- Harold's total number of books
  (h1 : H = (1 : ℚ) / 2 * M) -- Harold has 1/2 as many books as Millicent
  (h2 : (1 : ℚ) / 3 * H + (1 : ℚ) / 2 * M > 0) -- New home's capacity is positive
  : ((1 : ℚ) / 3 * H + (1 : ℚ) / 2 * M) / M = (2 : ℚ) / 3 := by
  sorry

end new_library_capacity_l4066_406695


namespace good_array_probability_l4066_406650

def is_good_array (a b c d : Int) : Prop :=
  a ∈ ({-1, 0, 1} : Set Int) ∧
  b ∈ ({-1, 0, 1} : Set Int) ∧
  c ∈ ({-1, 0, 1} : Set Int) ∧
  d ∈ ({-1, 0, 1} : Set Int) ∧
  a + b ≠ c + d ∧
  a + b ≠ a + c ∧
  a + b ≠ b + d ∧
  c + d ≠ a + c ∧
  c + d ≠ b + d ∧
  a + c ≠ b + d

def total_arrays : Nat := 3^4

def good_arrays : Nat := 16

theorem good_array_probability :
  (good_arrays : ℚ) / total_arrays = 16 / 81 :=
sorry

end good_array_probability_l4066_406650


namespace megan_spelling_problems_l4066_406682

/-- The number of spelling problems Megan had to solve -/
def spelling_problems (math_problems : ℕ) (problems_per_hour : ℕ) (total_hours : ℕ) : ℕ :=
  problems_per_hour * total_hours - math_problems

theorem megan_spelling_problems :
  spelling_problems 36 8 8 = 28 := by
  sorry

end megan_spelling_problems_l4066_406682


namespace sum_of_10th_degree_polynomials_l4066_406687

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- A polynomial is of 10th degree -/
def is_10th_degree (p : Polynomial ℝ) : Prop := degree p = 10

theorem sum_of_10th_degree_polynomials (p q : Polynomial ℝ) 
  (hp : is_10th_degree p) (hq : is_10th_degree q) : 
  degree (p + q) ≤ 10 := by sorry

end sum_of_10th_degree_polynomials_l4066_406687


namespace find_number_l4066_406627

theorem find_number : ∃! x : ℤ, x - 254 + 329 = 695 ∧ x = 620 := by sorry

end find_number_l4066_406627


namespace investment_problem_l4066_406601

/-- Given a sum P invested at a rate R for 20 years, if investing at a rate (R + 10)%
    yields Rs. 3000 more in interest, then P = 1500. -/
theorem investment_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  P * (R + 10) * 20 / 100 = P * R * 20 / 100 + 3000 →
  P = 1500 := by
sorry

end investment_problem_l4066_406601


namespace greatest_four_digit_divisible_l4066_406600

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_four_digit_divisible (p : ℕ) 
  (h1 : is_four_digit p)
  (h2 : is_four_digit (reverse_digits p))
  (h3 : p % 63 = 0)
  (h4 : (reverse_digits p) % 63 = 0)
  (h5 : p % 19 = 0) :
  p ≤ 5985 ∧ (∀ q : ℕ, 
    is_four_digit q → 
    is_four_digit (reverse_digits q) → 
    q % 63 = 0 → 
    (reverse_digits q) % 63 = 0 → 
    q % 19 = 0 → 
    q ≤ p) :=
by sorry

end greatest_four_digit_divisible_l4066_406600


namespace first_term_of_arithmetic_sequence_l4066_406630

/-- An arithmetic sequence with first term a and common difference d -/
structure ArithmeticSequence where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a + (n - 1) * seq.d)

/-- Theorem: If the sum of the first 100 terms is 800 and the sum of the next 100 terms is 7500,
    then the first term of the arithmetic sequence is -24.835 -/
theorem first_term_of_arithmetic_sequence
  (seq : ArithmeticSequence)
  (h1 : sum_n_terms seq 100 = 800)
  (h2 : sum_n_terms seq 200 - sum_n_terms seq 100 = 7500) :
  seq.a = -4967 / 200 :=
sorry

end first_term_of_arithmetic_sequence_l4066_406630


namespace weight_difference_E_D_l4066_406618

/-- Given the weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D -/
theorem weight_difference_E_D (w_A w_B w_C w_D w_E : ℝ) : w_E - w_D = 3 :=
  by
  have h1 : (w_A + w_B + w_C) / 3 = 84 := by sorry
  have h2 : (w_A + w_B + w_C + w_D) / 4 = 80 := by sorry
  have h3 : (w_B + w_C + w_D + w_E) / 4 = 79 := by sorry
  have h4 : w_A = 75 := by sorry
  sorry

#check weight_difference_E_D

end weight_difference_E_D_l4066_406618


namespace function_property_l4066_406679

theorem function_property (f : ℕ+ → ℕ+) 
  (h1 : f 1 ≠ 1)
  (h2 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1) :
  f 2015 = 2016 := by
  sorry

end function_property_l4066_406679


namespace x_minus_y_equals_two_l4066_406662

theorem x_minus_y_equals_two (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_sq_eq : x^2 - y^2 = 20) : 
  x - y = 2 := by
  sorry

end x_minus_y_equals_two_l4066_406662


namespace largest_even_multiple_of_15_under_500_l4066_406621

theorem largest_even_multiple_of_15_under_500 : ∃ n : ℕ, 
  n * 15 = 480 ∧ 
  480 % 2 = 0 ∧ 
  480 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 % 2 = 0 → m * 15 ≤ 480 :=
by sorry

end largest_even_multiple_of_15_under_500_l4066_406621


namespace impossibleCubePlacement_l4066_406684

/-- A type representing the vertices of a cube --/
inductive CubeVertex
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

/-- A function type representing a placement of numbers on the cube vertices --/
def CubePlacement := CubeVertex → Nat

/-- Predicate to check if two vertices are adjacent on a cube --/
def adjacent : CubeVertex → CubeVertex → Prop :=
  sorry

/-- Predicate to check if a number is in the valid range and not divisible by 13 --/
def validNumber (n : Nat) : Prop :=
  1 ≤ n ∧ n ≤ 245 ∧ n % 13 ≠ 0

/-- Predicate to check if two numbers have a common divisor greater than 1 --/
def hasCommonDivisor (a b : Nat) : Prop :=
  ∃ (d : Nat), d > 1 ∧ d ∣ a ∧ d ∣ b

theorem impossibleCubePlacement :
  ¬∃ (p : CubePlacement),
    (∀ v, validNumber (p v)) ∧
    (∀ v1 v2, v1 ≠ v2 → p v1 ≠ p v2) ∧
    (∀ v1 v2, adjacent v1 v2 → hasCommonDivisor (p v1) (p v2)) ∧
    (∀ v1 v2, ¬adjacent v1 v2 → ¬hasCommonDivisor (p v1) (p v2)) :=
by
  sorry


end impossibleCubePlacement_l4066_406684


namespace concentric_circles_ratio_l4066_406602

theorem concentric_circles_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = 1 / Real.sqrt 5 := by
  sorry

end concentric_circles_ratio_l4066_406602


namespace sqrt_equality_implies_specific_values_l4066_406631

theorem sqrt_equality_implies_specific_values (a b : ℕ) :
  0 < a → 0 < b → a < b →
  Real.sqrt (2 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 2 ∧ b = 5 := by
  sorry

end sqrt_equality_implies_specific_values_l4066_406631


namespace prob_no_adjacent_standing_ten_people_l4066_406615

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The number of people seated around the table. -/
def numPeople : ℕ := 10

/-- The total number of possible outcomes when flipping n fair coins. -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The probability of no two adjacent people standing when n people flip fair coins. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / totalOutcomes n

theorem prob_no_adjacent_standing_ten_people :
  noAdjacentStandingProb numPeople = 123 / 1024 := by
  sorry

#eval noAdjacentStandingProb numPeople

end prob_no_adjacent_standing_ten_people_l4066_406615


namespace part_one_part_two_l4066_406612

-- Define the functions f and g
def f (x : ℝ) := |x - 2|
def g (m : ℝ) (x : ℝ) := -|x + 3| + m

-- Part I
theorem part_one (m : ℝ) : 
  (∀ x, g m x ≥ 0 ↔ -5 ≤ x ∧ x ≤ -1) → m = 2 := by sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f x > g m x) → m < 5 := by sorry

end part_one_part_two_l4066_406612


namespace max_value_implies_t_equals_one_l4066_406676

def f (t : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - t|

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f t x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 3, f t x = 2) →
  t = 1 := by
sorry

end max_value_implies_t_equals_one_l4066_406676


namespace equal_coefficients_implies_n_seven_l4066_406638

theorem equal_coefficients_implies_n_seven (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
sorry

end equal_coefficients_implies_n_seven_l4066_406638


namespace lcm_852_1491_l4066_406665

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end lcm_852_1491_l4066_406665


namespace work_completion_l4066_406608

theorem work_completion (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 17 →
  absent_men = 8 →
  final_days = 21 →
  ∃ (original_men : ℕ),
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 42 :=
by sorry

end work_completion_l4066_406608


namespace binomial_10_choose_3_l4066_406649

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l4066_406649


namespace statement_b_false_statement_c_false_l4066_406640

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := |x - y + 3|

-- Statement B is false
theorem statement_b_false :
  ¬ (∀ x y : ℝ, 3 * (star x y) = star (3 * x + 3) (3 * y + 3)) :=
sorry

-- Statement C is false
theorem statement_c_false :
  ¬ (∀ x : ℝ, star x (-3) = x) :=
sorry

end statement_b_false_statement_c_false_l4066_406640


namespace f1_properties_f2_properties_f3_properties_f4_properties_l4066_406699

-- Function 1: y = 4 - x^2 for |x| ≤ 2
def f1 (x : ℝ) := 4 - x^2

-- Function 2: y = 0.5(x^2 + x|x| + 4)
def f2 (x : ℝ) := 0.5 * (x^2 + x * |x| + 4)

-- Function 3: y = (x^3 - x) / |x|
noncomputable def f3 (x : ℝ) := (x^3 - x) / |x|

-- Function 4: y = (x - 2)|x|
def f4 (x : ℝ) := (x - 2) * |x|

-- Theorem for function 1
theorem f1_properties (x : ℝ) (h : |x| ≤ 2) :
  f1 x ≤ 4 ∧ f1 0 = 4 ∧ f1 2 = f1 (-2) := by sorry

-- Theorem for function 2
theorem f2_properties (x : ℝ) :
  (x ≥ 0 → f2 x = x^2 + 2) ∧ (x < 0 → f2 x = 2) := by sorry

-- Theorem for function 3
theorem f3_properties (x : ℝ) (h : x ≠ 0) :
  (x > 0 → f3 x = x^2 - 1) ∧ (x < 0 → f3 x = -x^2 + 1) := by sorry

-- Theorem for function 4
theorem f4_properties (x : ℝ) :
  (x ≥ 0 → f4 x = x^2 - 2*x) ∧ (x < 0 → f4 x = -x^2 + 2*x) := by sorry

end f1_properties_f2_properties_f3_properties_f4_properties_l4066_406699


namespace jane_final_score_l4066_406652

/-- Calculates the final score in a card game --/
def final_score (rounds : ℕ) (points_per_win : ℕ) (points_lost : ℕ) : ℕ :=
  rounds * points_per_win - points_lost

/-- Theorem: Jane's final score in the card game --/
theorem jane_final_score :
  let rounds : ℕ := 8
  let points_per_win : ℕ := 10
  let points_lost : ℕ := 20
  final_score rounds points_per_win points_lost = 60 := by
  sorry


end jane_final_score_l4066_406652


namespace purple_tile_cost_l4066_406664

-- Define the problem parameters
def wall1_width : ℝ := 5
def wall1_height : ℝ := 8
def wall2_width : ℝ := 7
def wall2_height : ℝ := 8
def tiles_per_sqft : ℝ := 4
def turquoise_tile_cost : ℝ := 13
def savings : ℝ := 768

-- Calculate total area and number of tiles
def total_area : ℝ := wall1_width * wall1_height + wall2_width * wall2_height
def total_tiles : ℝ := total_area * tiles_per_sqft

-- Calculate costs
def turquoise_total_cost : ℝ := total_tiles * turquoise_tile_cost
def purple_total_cost : ℝ := turquoise_total_cost - savings

-- Theorem to prove
theorem purple_tile_cost : purple_total_cost / total_tiles = 11 := by
  sorry

end purple_tile_cost_l4066_406664


namespace customers_left_l4066_406632

theorem customers_left (initial : ℕ) (first_leave_percent : ℚ) (second_leave_percent : ℚ) : 
  initial = 36 → 
  first_leave_percent = 1/2 → 
  second_leave_percent = 3/10 → 
  ⌊(initial - ⌊initial * first_leave_percent⌋) - ⌊(initial - ⌊initial * first_leave_percent⌋) * second_leave_percent⌋⌋ = 13 := by
  sorry

end customers_left_l4066_406632


namespace increasing_quadratic_function_m_bound_l4066_406646

/-- Given that f(x) = -x^2 + mx is an increasing function on (-∞, 1], prove that m ≥ 2 -/
theorem increasing_quadratic_function_m_bound 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = -x^2 + m*x) 
  (h2 : ∀ x y, x < y → x ≤ 1 → y ≤ 1 → f x < f y) : 
  m ≥ 2 := by
  sorry

end increasing_quadratic_function_m_bound_l4066_406646


namespace smallest_square_with_property_l4066_406669

theorem smallest_square_with_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * n) % 10 ≠ 0 ∧ 
  (n * n) ≥ 121 ∧
  ∃ m : ℕ, m > 0 ∧ (n * n) / 100 = m * m ∧
  ∀ k : ℕ, k > 0 → (k * k) % 10 ≠ 0 → (k * k) < (n * n) → 
    ¬(∃ j : ℕ, j > 0 ∧ (k * k) / 100 = j * j) :=
by sorry

end smallest_square_with_property_l4066_406669


namespace math_problem_l4066_406637

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_sum : 3 * m + 2 * n = 225) :
  (gcd m n = 15 → m + n = 105) ∧ (lcm m n = 45 → m + n = 90) := by
  sorry

end math_problem_l4066_406637


namespace arithmetic_sequence_common_difference_l4066_406680

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The common difference of an arithmetic sequence is 3 given S_2 = 4 and S_4 = 20 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.S 2 = 4) 
  (h2 : seq.S 4 = 20) : 
  seq.a 2 - seq.a 1 = 3 := by
sorry

end arithmetic_sequence_common_difference_l4066_406680


namespace product_c_remaining_amount_l4066_406617

/-- Calculate the remaining amount to be paid for a product -/
def remaining_amount (cost deposit discount_rate tax_rate : ℝ) : ℝ :=
  let discounted_price := cost * (1 - discount_rate)
  let total_price := discounted_price * (1 + tax_rate)
  total_price - deposit

/-- Theorem: The remaining amount to be paid for Product C is $3,610 -/
theorem product_c_remaining_amount :
  remaining_amount 3800 380 0 0.05 = 3610 := by
  sorry

end product_c_remaining_amount_l4066_406617


namespace common_root_condition_l4066_406681

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) ↔ (m = 2021 ∨ m = -2021) := by
  sorry

end common_root_condition_l4066_406681


namespace no_roots_below_x0_l4066_406634

theorem no_roots_below_x0 (a b c d x₀ : ℝ) 
  (h1 : ∀ x ≥ x₀, x^2 + a*x + b > 0)
  (h2 : ∀ x ≥ x₀, x^2 + c*x + d > 0) :
  ∀ x > x₀, x^2 + (a+c)/2 * x + (b+d)/2 > 0 :=
by sorry

end no_roots_below_x0_l4066_406634


namespace necklace_diamonds_l4066_406607

theorem necklace_diamonds (total_necklaces : ℕ) (diamonds_type1 diamonds_type2 : ℕ) (total_diamonds : ℕ) :
  total_necklaces = 20 →
  diamonds_type1 = 2 →
  diamonds_type2 = 5 →
  total_diamonds = 79 →
  ∃ (x y : ℕ), x + y = total_necklaces ∧ 
                diamonds_type1 * x + diamonds_type2 * y = total_diamonds ∧
                y = 13 :=
by sorry

end necklace_diamonds_l4066_406607


namespace cos_pi_third_minus_alpha_l4066_406606

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) :
  Real.cos (π / 3 - α) = 3 / 5 := by sorry

end cos_pi_third_minus_alpha_l4066_406606


namespace evaluate_complex_expression_l4066_406647

theorem evaluate_complex_expression :
  let N := (Real.sqrt (Real.sqrt 10 + 3) - Real.sqrt (Real.sqrt 10 - 3)) / 
           Real.sqrt (Real.sqrt 10 + 2) - 
           Real.sqrt (6 - 4 * Real.sqrt 2)
  N = 1 + Real.sqrt 2 := by
  sorry

end evaluate_complex_expression_l4066_406647


namespace diminishing_allocation_solution_l4066_406693

/-- Represents the diminishing allocation problem with four terms -/
structure DiminishingAllocation where
  /-- The first term of the geometric sequence -/
  b : ℝ
  /-- The diminishing allocation ratio -/
  a : ℝ
  /-- The total amount to be distributed -/
  m : ℝ

/-- Conditions for the diminishing allocation problem -/
def validDiminishingAllocation (da : DiminishingAllocation) : Prop :=
  da.b > 0 ∧ da.a > 0 ∧ da.a < 1 ∧ da.m > 0 ∧
  da.b * (1 - da.a)^2 = 80 ∧
  da.b * (1 - da.a) + da.b * (1 - da.a)^3 = 164 ∧
  da.b + 80 + 164 = da.m

/-- Theorem stating the solution to the diminishing allocation problem -/
theorem diminishing_allocation_solution (da : DiminishingAllocation) 
  (h : validDiminishingAllocation da) : da.a = 0.2 ∧ da.m = 369 := by
  sorry

end diminishing_allocation_solution_l4066_406693


namespace floor_ceiling_sum_l4066_406653

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.005 : ℝ)⌉ = 5 := by
  sorry

end floor_ceiling_sum_l4066_406653


namespace cube_root_of_negative_eight_l4066_406677

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 ↔ x = -2 := by
  sorry

end cube_root_of_negative_eight_l4066_406677


namespace businessmen_drinking_none_l4066_406611

theorem businessmen_drinking_none (total : ℕ) (coffee tea soda coffee_tea tea_soda coffee_soda all_three : ℕ) : 
  total = 30 ∧ 
  coffee = 15 ∧ 
  tea = 12 ∧ 
  soda = 8 ∧ 
  coffee_tea = 7 ∧ 
  tea_soda = 3 ∧ 
  coffee_soda = 2 ∧ 
  all_three = 1 → 
  total - (coffee + tea + soda - coffee_tea - tea_soda - coffee_soda + all_three) = 6 := by
sorry

end businessmen_drinking_none_l4066_406611


namespace problem_solution_l4066_406641

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 10.5 := by
  sorry

end problem_solution_l4066_406641


namespace apple_expense_calculation_l4066_406673

/-- Proves that the amount spent on apples is the difference between the total amount and the sum of other expenses and remaining money. -/
theorem apple_expense_calculation (total amount_oranges amount_candy amount_left : ℕ) 
  (h1 : total = 95)
  (h2 : amount_oranges = 14)
  (h3 : amount_candy = 6)
  (h4 : amount_left = 50) :
  total - (amount_oranges + amount_candy + amount_left) = 25 :=
by sorry

end apple_expense_calculation_l4066_406673


namespace quadratic_inequality_relationship_l4066_406692

theorem quadratic_inequality_relationship (x : ℝ) :
  (x^2 - 5*x + 6 > 0 → x > 3) ∧ ¬(x > 3 → x^2 - 5*x + 6 > 0) :=
by sorry

end quadratic_inequality_relationship_l4066_406692


namespace factorization_problem_1_factorization_problem_2_l4066_406666

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  3 * x^2 - 12 * x * y + 12 * y^2 = 3 * (x - 2 * y)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l4066_406666


namespace shaded_area_in_square_l4066_406635

/-- The area of a shaded region within a square, where two congruent right triangles
    are removed from opposite corners. -/
theorem shaded_area_in_square (side : ℝ) (triangle_side : ℝ)
    (h_side : side = 30)
    (h_triangle : triangle_side = 20) :
    side * side - 2 * (1/2 * triangle_side * triangle_side) = 500 := by
  sorry

end shaded_area_in_square_l4066_406635


namespace solutions_not_real_root_loci_l4066_406633

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℂ) : ℂ := x^2 + a*x + 1

-- Theorem for the interval of a where solutions are not real
theorem solutions_not_real (a : ℝ) :
  (∀ x : ℂ, quadratic a x = 0 → x.im ≠ 0) ↔ a ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

-- Define the ellipse
def ellipse (z : ℂ) : Prop := 4 * z.re^2 + z.im^2 = 4

-- Theorem for the loci of roots
theorem root_loci (a : ℝ) (z : ℂ) :
  a ∈ Set.Ioo (-2 : ℝ) 2 →
  (quadratic a z = 0 ↔ (ellipse z ∧ z ≠ -1 ∧ z ≠ 1)) :=
sorry

end solutions_not_real_root_loci_l4066_406633


namespace farmer_earnings_example_l4066_406663

/-- Calculates a farmer's earnings from egg sales over a given number of weeks -/
def farmer_earnings (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  let total_eggs := num_chickens * eggs_per_chicken * num_weeks
  let dozens := total_eggs / 12
  dozens * price_per_dozen

theorem farmer_earnings_example : farmer_earnings 46 6 3 8 = 552 := by
  sorry

end farmer_earnings_example_l4066_406663


namespace twelfth_root_of_unity_l4066_406620

open Complex

theorem twelfth_root_of_unity : 
  let z : ℂ := (Complex.tan (π / 6) + I) / (Complex.tan (π / 6) - I)
  z = exp (I * π / 3) ∧ z^12 = 1 := by sorry

end twelfth_root_of_unity_l4066_406620


namespace inequality_solution_l4066_406626

def k : ℝ := 0.5

def inequality (θ x : ℝ) : Prop :=
  x^2 * Real.sin θ - k*x*(1 - x) + (1 - x)^2 * Real.cos θ ≥ 0

def solution_set : Set ℝ :=
  {θ | 0 ≤ θ ∧ θ ≤ 2*Real.pi ∧ ∀ x, 0 ≤ x ∧ x ≤ 1 → inequality θ x}

theorem inequality_solution :
  solution_set = {θ | (0 ≤ θ ∧ θ ≤ Real.pi/12) ∨ (23*Real.pi/12 ≤ θ ∧ θ ≤ 2*Real.pi)} :=
by sorry

end inequality_solution_l4066_406626


namespace power_multiplication_l4066_406609

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l4066_406609


namespace clock_hands_angle_at_1_10_clock_hands_angle_at_1_10_is_25_l4066_406656

/-- The angle between clock hands at 1:10 -/
theorem clock_hands_angle_at_1_10 : ℝ := by
  -- Define constants
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let minutes_passed : ℕ := 10

  -- Define speeds (degrees per minute)
  let hour_hand_speed : ℝ := total_degrees / (total_hours * 60)
  let minute_hand_speed : ℝ := total_degrees / 60

  -- Define initial positions at 1:00
  let initial_hour_hand_position : ℝ := 30
  let initial_minute_hand_position : ℝ := 0

  -- Calculate final positions at 1:10
  let final_hour_hand_position : ℝ := initial_hour_hand_position + hour_hand_speed * minutes_passed
  let final_minute_hand_position : ℝ := initial_minute_hand_position + minute_hand_speed * minutes_passed

  -- Calculate the angle between hands
  let angle_between_hands : ℝ := final_minute_hand_position - final_hour_hand_position

  -- Prove that the angle is 25°
  sorry

/-- The theorem states that the angle between the hour and minute hands at 1:10 is 25° -/
theorem clock_hands_angle_at_1_10_is_25 : clock_hands_angle_at_1_10 = 25 := by
  sorry

end clock_hands_angle_at_1_10_clock_hands_angle_at_1_10_is_25_l4066_406656


namespace derivative_of_f_l4066_406655

-- Define the function f(x) = (2 + x³)²
def f (x : ℝ) : ℝ := (2 + x^3)^2

-- State the theorem that the derivative of f(x) is 2(2 + x³) · 3x
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2 * (2 + x^3) * 3 * x := by sorry

end derivative_of_f_l4066_406655


namespace simplify_expression_l4066_406643

theorem simplify_expression (m n : ℝ) : 
  4 * m * n^3 * (2 * m^2 - 3/4 * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := by
  sorry

end simplify_expression_l4066_406643


namespace remaining_quantities_l4066_406629

theorem remaining_quantities (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 5 ∧ 
  total_avg = 10 ∧ 
  subset = 3 ∧ 
  subset_avg = 4 ∧ 
  remaining_avg = 19 →
  total - subset = 2 :=
by sorry

end remaining_quantities_l4066_406629


namespace line_through_points_l4066_406667

/-- A line passing through two points (1,2) and (5,14) has equation y = ax + b. This theorem proves that a - b = 4. -/
theorem line_through_points (a b : ℝ) : 
  (2 = a * 1 + b) → (14 = a * 5 + b) → a - b = 4 := by
  sorry

end line_through_points_l4066_406667


namespace area_of_five_arranged_triangles_l4066_406623

/-- The area covered by five equilateral triangles arranged in a specific way -/
theorem area_of_five_arranged_triangles : 
  let side_length : ℝ := 2 * Real.sqrt 3
  let single_triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let number_of_triangles : ℕ := 5
  let effective_triangles : ℝ := 4
  effective_triangles * single_triangle_area = 12 * Real.sqrt 3 :=
by sorry

end area_of_five_arranged_triangles_l4066_406623


namespace smallest_b_value_l4066_406696

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, a'.val - k.val = 8 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val * k.val) = 16) :=
by sorry

end smallest_b_value_l4066_406696


namespace rosa_initial_flowers_l4066_406622

theorem rosa_initial_flowers (flowers_from_andre : ℝ) (total_flowers : ℕ) :
  flowers_from_andre = 90.0 →
  total_flowers = 157 →
  total_flowers - Int.floor flowers_from_andre = 67 := by
  sorry

end rosa_initial_flowers_l4066_406622


namespace inequality_proof_l4066_406672

theorem inequality_proof (a b c k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 1) :
  (a^(k+1) / b^k : ℚ) + (b^(k+1) / c^k : ℚ) + (c^(k+1) / a^k : ℚ) ≥ 
  (a^k / b^(k-1) : ℚ) + (b^k / c^(k-1) : ℚ) + (c^k / a^(k-1) : ℚ) := by
  sorry

end inequality_proof_l4066_406672


namespace intersection_of_A_and_B_l4066_406610

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l4066_406610


namespace find_certain_number_l4066_406698

theorem find_certain_number : ∃ x : ℤ, x - 5 = 4 ∧ x = 9 := by
  sorry

end find_certain_number_l4066_406698


namespace sculpture_cost_yuan_l4066_406690

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_namibian : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_yuan : ℚ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_namibian : ℚ := 160

/-- Theorem stating that the cost of the sculpture in Chinese yuan is 160 -/
theorem sculpture_cost_yuan :
  (sculpture_cost_namibian / usd_to_namibian) * usd_to_yuan = 160 := by
  sorry

end sculpture_cost_yuan_l4066_406690


namespace circle_area_tripled_l4066_406613

theorem circle_area_tripled (r m : ℝ) (h : r > 0) (h' : m > 0) : 
  π * (r + m)^2 = 3 * (π * r^2) → r = (m * (1 + Real.sqrt 3)) / 2 := by
  sorry

end circle_area_tripled_l4066_406613


namespace zoo_cost_l4066_406691

theorem zoo_cost (goat_price : ℕ) (goat_count : ℕ) (llama_price_increase : ℚ) : 
  goat_price = 400 →
  goat_count = 3 →
  llama_price_increase = 1/2 →
  (goat_count * goat_price + 
   2 * goat_count * (goat_price + goat_price * llama_price_increase)) = 4800 := by
sorry

end zoo_cost_l4066_406691
