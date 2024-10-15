import Mathlib

namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1192_119214

theorem solve_exponential_equation :
  ∃ y : ℝ, (3 : ℝ) ^ (y + 2) = 27 ^ y ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1192_119214


namespace NUMINAMATH_CALUDE_place_value_ratio_l1192_119293

def number : ℚ := 56439.2071

theorem place_value_ratio : 
  (10000 : ℚ) * (number - number.floor) * 10 = (number.floor % 100000 - number.floor % 10000) / 10 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l1192_119293


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1192_119269

theorem triangle_angle_c (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = Real.pi) (h5 : B = Real.pi / 4)
  (h6 : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = c * Real.sqrt 2 ∧
    a / (Real.sin C) = b / (Real.sin A) ∧ b / (Real.sin B) = c / (Real.sin A)) :
  C = 7 * Real.pi / 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1192_119269


namespace NUMINAMATH_CALUDE_division_problem_l1192_119209

theorem division_problem (d : ℕ) (h : 23 = d * 4 + 3) : d = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1192_119209


namespace NUMINAMATH_CALUDE_max_m_value_l1192_119208

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  (∀ m : ℝ, a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m*a*b ≥ 0) →
  (∃ m : ℝ, m = 3/2 ∧ a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m*a*b = 0 ∧
    ∀ m' : ℝ, m' > m → a + b/2 + Real.sqrt (a^2/2 + 2*b^2) - m'*a*b < 0) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1192_119208


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1192_119207

theorem smaller_number_proof (S L : ℕ) 
  (h1 : L - S = 2468) 
  (h2 : L = 8 * S + 27) : 
  S = 349 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1192_119207


namespace NUMINAMATH_CALUDE_vector_equality_l1192_119259

-- Define the triangle ABC and point D
variable (A B C D : ℝ × ℝ)

-- Define vectors
def AB : ℝ × ℝ := B - A
def AC : ℝ × ℝ := C - A
def AD : ℝ × ℝ := D - A
def BC : ℝ × ℝ := C - B
def BD : ℝ × ℝ := D - B
def DC : ℝ × ℝ := C - D

-- State the theorem
theorem vector_equality (h : BD = 3 • DC) : AD = (1/4) • AB + (3/4) • AC := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l1192_119259


namespace NUMINAMATH_CALUDE_multiple_of_six_last_digit_l1192_119216

theorem multiple_of_six_last_digit (n : ℕ) : 
  n ≥ 85670 ∧ n < 85680 ∧ n % 6 = 0 → n = 85676 := by sorry

end NUMINAMATH_CALUDE_multiple_of_six_last_digit_l1192_119216


namespace NUMINAMATH_CALUDE_calcium_chloride_formation_l1192_119264

/-- Represents a chemical reaction --/
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String
  ratio1 : ℚ
  ratio2 : ℚ

/-- Calculates the moles of product formed in a chemical reaction --/
def calculate_product_moles (r : Reaction) (moles_reactant1 : ℚ) (moles_reactant2 : ℚ) : ℚ :=
  min (moles_reactant1 / r.ratio1) (moles_reactant2 / r.ratio2)

/-- Theorem: Given 4 moles of HCl and 2 moles of CaCO3, 2 moles of CaCl2 are formed --/
theorem calcium_chloride_formation :
  let r : Reaction := {
    reactant1 := "CaCO3",
    reactant2 := "HCl",
    product1 := "CaCl2",
    product2 := "CO2",
    product3 := "H2O",
    ratio1 := 1,
    ratio2 := 2
  }
  calculate_product_moles r 2 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calcium_chloride_formation_l1192_119264


namespace NUMINAMATH_CALUDE_probability_of_winning_reward_l1192_119285

/-- The number of different types of blessing cards -/
def num_card_types : ℕ := 3

/-- The number of red envelopes Xiao Ming has -/
def num_envelopes : ℕ := 4

/-- The probability of winning the reward -/
def win_probability : ℚ := 4/9

/-- Theorem stating the probability of winning the reward -/
theorem probability_of_winning_reward :
  (num_card_types = 3) →
  (num_envelopes = 4) →
  (win_probability = 4/9) := by
  sorry

end NUMINAMATH_CALUDE_probability_of_winning_reward_l1192_119285


namespace NUMINAMATH_CALUDE_cynthia_gallons_proof_l1192_119258

def pool_capacity : ℕ := 105
def num_trips : ℕ := 7
def caleb_gallons : ℕ := 7

theorem cynthia_gallons_proof :
  ∃ (cynthia_gallons : ℕ),
    cynthia_gallons * num_trips + caleb_gallons * num_trips = pool_capacity ∧
    cynthia_gallons = 8 := by
  sorry

end NUMINAMATH_CALUDE_cynthia_gallons_proof_l1192_119258


namespace NUMINAMATH_CALUDE_crystal_mass_problem_l1192_119204

theorem crystal_mass_problem (a b : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  (0.04 * a / 4 = 0.05 * b / 3) → 
  ((a + 20) / (b + 20) = 1.5) → 
  (a = 100 ∧ b = 60) := by
  sorry

end NUMINAMATH_CALUDE_crystal_mass_problem_l1192_119204


namespace NUMINAMATH_CALUDE_factorization_proof_l1192_119203

theorem factorization_proof (x : ℝ) : 72 * x^2 + 108 * x + 36 = 36 * (2 * x + 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1192_119203


namespace NUMINAMATH_CALUDE_solve_class_problem_l1192_119212

def class_problem (N : ℕ) : Prop :=
  ∃ (taqeesha_score : ℕ),
    N > 1 ∧
    (77 * (N - 1) + taqeesha_score) / N = 78 ∧
    N - 1 = 16

theorem solve_class_problem :
  ∃ (N : ℕ), class_problem N ∧ N = 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_class_problem_l1192_119212


namespace NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l1192_119210

theorem max_tickets_jane_can_buy (ticket_cost : ℕ) (service_charge : ℕ) (budget : ℕ) :
  ticket_cost = 15 →
  service_charge = 10 →
  budget = 120 →
  ∃ (n : ℕ), n = 7 ∧ 
    n * ticket_cost + service_charge ≤ budget ∧
    ∀ (m : ℕ), m * ticket_cost + service_charge ≤ budget → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l1192_119210


namespace NUMINAMATH_CALUDE_probability_at_least_six_heads_in_eight_flips_probability_at_least_six_heads_in_eight_flips_proof_l1192_119219

/-- The probability of getting at least 6 heads in 8 flips of a fair coin -/
theorem probability_at_least_six_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting at least 6 heads in 8 flips of a fair coin is 37/256 -/
theorem probability_at_least_six_heads_in_eight_flips_proof :
  probability_at_least_six_heads_in_eight_flips = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_six_heads_in_eight_flips_probability_at_least_six_heads_in_eight_flips_proof_l1192_119219


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1192_119267

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1192_119267


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1192_119238

theorem exponent_multiplication (x : ℝ) (a b : ℕ) :
  x^a * x^b = x^(a + b) := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1192_119238


namespace NUMINAMATH_CALUDE_inequality_proof_l1192_119292

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x^2 + y^2 + z^2 = 2*(x*y + y*z + z*x)) : 
  (x + y + z) / 3 ≥ (2*x*y*z)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1192_119292


namespace NUMINAMATH_CALUDE_driving_time_calculation_l1192_119272

/-- 
Given a trip with the following conditions:
1. The total trip duration is 15 hours
2. The time stuck in traffic is twice the driving time
This theorem proves that the driving time is 5 hours
-/
theorem driving_time_calculation (total_time : ℝ) (driving_time : ℝ) (traffic_time : ℝ) 
  (h1 : total_time = 15)
  (h2 : traffic_time = 2 * driving_time)
  (h3 : total_time = driving_time + traffic_time) :
  driving_time = 5 := by
sorry

end NUMINAMATH_CALUDE_driving_time_calculation_l1192_119272


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l1192_119270

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := ((x/2)^α) / (x-1)

noncomputable def g (α : ℝ) : ℝ := (deriv^[4] (f α)) 2

theorem leading_coefficient_of_g (α : ℝ) : 
  ∃ (p : Polynomial ℝ), (∀ x, g x = p.eval x) ∧ p.leadingCoeff = 1/16 :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l1192_119270


namespace NUMINAMATH_CALUDE_at_least_one_hit_probability_l1192_119250

theorem at_least_one_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.7) (h2 : p2 = 0.8) :
  p1 + p2 - p1 * p2 = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hit_probability_l1192_119250


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l1192_119277

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_ground_time (y t : ℝ) : 
  y = -9.8 * t^2 + 5.6 * t + 10 →
  y = 0 →
  t = 131 / 98 := by sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l1192_119277


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l1192_119239

theorem alcohol_mixture_percentage :
  ∀ x : ℝ,
  (x / 100) * 8 + (12 / 100) * 2 = (22.4 / 100) * 10 →
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l1192_119239


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1192_119247

theorem divisible_by_nine (A : Nat) : A < 10 → (7000 + 100 * A + 46) % 9 = 0 ↔ A = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1192_119247


namespace NUMINAMATH_CALUDE_binomial_20_18_l1192_119223

theorem binomial_20_18 : Nat.choose 20 18 = 190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_18_l1192_119223


namespace NUMINAMATH_CALUDE_exist_sequence_l1192_119294

/-- Sum of digits of a positive integer -/
def S (m : ℕ+) : ℕ := sorry

/-- Product of digits of a positive integer -/
def P (m : ℕ+) : ℕ := sorry

/-- For any positive integer n, there exist positive integers a₁, a₂, ..., aₙ
    satisfying the required conditions -/
theorem exist_sequence (n : ℕ+) : 
  ∃ (a : Fin n → ℕ+), 
    (∀ i j : Fin n, i < j → S (a i) < S (a j)) ∧ 
    (∀ i : Fin n, S (a i) = P (a ((i + 1) % n))) := by
  sorry

end NUMINAMATH_CALUDE_exist_sequence_l1192_119294


namespace NUMINAMATH_CALUDE_volleyball_managers_l1192_119232

theorem volleyball_managers (num_teams : ℕ) (people_per_team : ℕ) (num_employees : ℕ) : 
  num_teams = 6 → people_per_team = 5 → num_employees = 7 →
  num_teams * people_per_team - num_employees = 23 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_managers_l1192_119232


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1192_119280

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 1 < x ∧ x < 3) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1192_119280


namespace NUMINAMATH_CALUDE_first_term_value_l1192_119236

/-- A geometric sequence with five terms -/
def GeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 36 = b * r ∧ c = 36 * r ∧ 144 = c * r

/-- The first term of the geometric sequence is 9/4 -/
theorem first_term_value (a b c : ℝ) (h : GeometricSequence a b c) : a = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_value_l1192_119236


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1192_119222

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}

-- Define the set B
def B : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1192_119222


namespace NUMINAMATH_CALUDE_tangent_segment_length_is_3cm_l1192_119225

/-- An isosceles triangle with a base of 12 cm and a height of 8 cm -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  isIsosceles : base = 12 ∧ height = 8

/-- A circle inscribed in the isosceles triangle -/
structure InscribedCircle (t : IsoscelesTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  isInscribed : True  -- This is a placeholder for the inscribed circle condition

/-- A tangent line parallel to the base of the triangle -/
structure ParallelTangent (t : IsoscelesTriangle) (c : InscribedCircle t) where
  point : ℝ × ℝ
  isParallel : True  -- This is a placeholder for the parallel condition
  isTangent : True   -- This is a placeholder for the tangent condition

/-- The length of the segment of the tangent line between the sides of the triangle -/
def tangentSegmentLength (t : IsoscelesTriangle) (c : InscribedCircle t) (l : ParallelTangent t c) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem stating that the length of the tangent segment is 3 cm -/
theorem tangent_segment_length_is_3cm (t : IsoscelesTriangle) (c : InscribedCircle t) (l : ParallelTangent t c) :
  tangentSegmentLength t c l = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_segment_length_is_3cm_l1192_119225


namespace NUMINAMATH_CALUDE_parallelogram_other_vertices_y_sum_l1192_119290

/-- A parallelogram with two opposite vertices at (2,15) and (8,-2) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (2, 15)
  v2 : ℝ × ℝ := (8, -2)
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  is_parallelogram : True  -- We assume this is a valid parallelogram

/-- The sum of y-coordinates of the other two vertices is 13 -/
theorem parallelogram_other_vertices_y_sum (p : Parallelogram) : 
  (p.v3).2 + (p.v4).2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_other_vertices_y_sum_l1192_119290


namespace NUMINAMATH_CALUDE_pipe_c_fill_time_l1192_119268

/-- The time (in minutes) it takes for pipe a to fill the tank -/
def time_a : ℝ := 20

/-- The time (in minutes) it takes for pipe b to fill the tank -/
def time_b : ℝ := 20

/-- The time (in minutes) it takes for pipe c to fill the tank -/
def time_c : ℝ := 30

/-- The proportion of solution r in the tank after 3 minutes -/
def proportion_r : ℝ := 0.25

/-- The time (in minutes) after which we measure the proportion of solution r -/
def measure_time : ℝ := 3

theorem pipe_c_fill_time :
  (time_c = 30) ∧
  (measure_time * (1 / time_a + 1 / time_b + 1 / time_c) * (1 / time_c) /
   (measure_time * (1 / time_a + 1 / time_b + 1 / time_c)) = proportion_r) :=
sorry

end NUMINAMATH_CALUDE_pipe_c_fill_time_l1192_119268


namespace NUMINAMATH_CALUDE_sum_product_difference_l1192_119235

theorem sum_product_difference (x y : ℝ) : 
  x + y = 500 → x * y = 22000 → y - x = -402.5 := by
sorry

end NUMINAMATH_CALUDE_sum_product_difference_l1192_119235


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1192_119253

theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- a_1 = 1
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by            
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1192_119253


namespace NUMINAMATH_CALUDE_salt_calculation_l1192_119257

/-- Calculates the amount of salt obtained from seawater evaporation -/
def salt_from_seawater (volume : ℝ) (salt_percentage : ℝ) : ℝ :=
  volume * salt_percentage * 1000

/-- Proves that 2 liters of seawater with 20% salt content yields 400 ml of salt -/
theorem salt_calculation :
  salt_from_seawater 2 0.20 = 400 := by
  sorry

end NUMINAMATH_CALUDE_salt_calculation_l1192_119257


namespace NUMINAMATH_CALUDE_exam_students_count_l1192_119283

/-- Proves that given the conditions of the exam results, the total number of students is 14 -/
theorem exam_students_count (total_average : ℝ) (excluded_count : ℕ) (excluded_average : ℝ) (remaining_average : ℝ)
  (h1 : total_average = 65)
  (h2 : excluded_count = 5)
  (h3 : excluded_average = 20)
  (h4 : remaining_average = 90) :
  ∃ (n : ℕ), n = 14 ∧ 
    (n : ℝ) * total_average = 
      ((n - excluded_count) : ℝ) * remaining_average + (excluded_count : ℝ) * excluded_average :=
by sorry

end NUMINAMATH_CALUDE_exam_students_count_l1192_119283


namespace NUMINAMATH_CALUDE_product_of_five_integers_l1192_119246

theorem product_of_five_integers (E F G H I : ℕ) 
  (sum_condition : E + F + G + H + I = 110)
  (equality_condition : (E : ℚ) / 2 = (F : ℚ) / 3 ∧ 
                        (F : ℚ) / 3 = G * 4 ∧ 
                        G * 4 = H * 2 ∧ 
                        H * 2 = I - 5) : 
  (E : ℚ) * F * G * H * I = 623400000 / 371293 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_integers_l1192_119246


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l1192_119206

theorem first_number_in_ratio (A B : ℕ) (h1 : A > 0) (h2 : B > 0) : 
  A * 4 = B * 5 → lcm A B = 80 → A = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l1192_119206


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1192_119224

/-- Given two vectors a and b in ℝ², prove that if a = (1, -2) and b = (3, x) are perpendicular, then x = 3/2. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, -2) →
  b = (3, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1192_119224


namespace NUMINAMATH_CALUDE_basketball_win_rate_l1192_119273

theorem basketball_win_rate (total_games : ℕ) (first_segment : ℕ) (won_first : ℕ) (target_rate : ℚ) : 
  total_games = 130 →
  first_segment = 70 →
  won_first = 60 →
  target_rate = 3/4 →
  ∃ (x : ℕ), x = 38 ∧ 
    (won_first + x : ℚ) / total_games = target_rate ∧
    x ≤ total_games - first_segment :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l1192_119273


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1192_119298

/-- For an ellipse with given eccentricity and focal length, prove the length of its minor axis. -/
theorem ellipse_minor_axis_length 
  (e : ℝ) -- eccentricity
  (f : ℝ) -- focal length
  (h_e : e = 1/2)
  (h_f : f = 2) :
  ∃ (minor_axis : ℝ), minor_axis = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1192_119298


namespace NUMINAMATH_CALUDE_cyclist_speed_is_25_l1192_119233

/-- The speed of the motorcyclist in km/h -/
def V_M : ℝ := 50

/-- The system of equations for the cyclist and motorcyclist problem -/
def equations (x y : ℝ) : Prop :=
  (20 / x - 20 / V_M = y) ∧ (70 - 8 / 3 * x = V_M * (7 / 15 - y))

/-- Theorem stating that x = 25 km/h satisfies the system of equations for some y -/
theorem cyclist_speed_is_25 : ∃ y : ℝ, equations 25 y := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_is_25_l1192_119233


namespace NUMINAMATH_CALUDE_simplify_expression_l1192_119254

theorem simplify_expression (a : ℝ) (h1 : a^2 - 1 ≠ 0) (h2 : a ≠ 0) :
  (1 / (a + 1) + 1 / (a^2 - 1)) / (a / (a^2 - 2*a + 1)) = (a - 1) / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1192_119254


namespace NUMINAMATH_CALUDE_min_detectors_for_cross_l1192_119284

/-- The size of the board --/
def boardSize : Nat := 5

/-- The number of cells in the cross pattern --/
def crossSize : Nat := 5

/-- The number of possible positions for the cross on the board --/
def possiblePositions : Nat := 3 * 3

/-- Function to calculate the number of possible detector states --/
def detectorStates (n : Nat) : Nat := 2^n

/-- Theorem stating the minimum number of detectors needed --/
theorem min_detectors_for_cross :
  ∃ (n : Nat), (n = 4) ∧ 
  (∀ (k : Nat), detectorStates k ≥ possiblePositions → k ≥ n) ∧
  (detectorStates n ≥ possiblePositions) := by
  sorry

end NUMINAMATH_CALUDE_min_detectors_for_cross_l1192_119284


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_twenty_l1192_119200

/-- The sum of an arithmetic sequence with 5 terms, where the first term is 2 and the last term is 6 -/
def arithmetic_sum : ℕ :=
  let n := 5  -- number of days
  let a₁ := 2 -- first day's distance
  let aₙ := 6 -- last day's distance
  n * (a₁ + aₙ) / 2

/-- The theorem states that the arithmetic sum defined above equals 20 -/
theorem arithmetic_sum_equals_twenty : arithmetic_sum = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_twenty_l1192_119200


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l1192_119244

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
  n ≤ 99986420 :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l1192_119244


namespace NUMINAMATH_CALUDE_min_shortest_side_is_12_l1192_119299

/-- Represents a triangle with integer side lengths and given altitudes -/
structure Triangle where
  -- Side lengths
  AB : ℕ
  BC : ℕ
  CA : ℕ
  -- Altitude lengths
  AD : ℕ
  BE : ℕ
  CF : ℕ
  -- Conditions
  altitude_AD : AD = 3
  altitude_BE : BE = 4
  altitude_CF : CF = 5
  -- Area equality conditions
  area_eq_1 : BC * AD = CA * BE
  area_eq_2 : CA * BE = AB * CF

/-- The minimum possible length of the shortest side of the triangle -/
def min_shortest_side (t : Triangle) : ℕ := min t.AB (min t.BC t.CA)

/-- Theorem stating the minimum possible length of the shortest side is 12 -/
theorem min_shortest_side_is_12 (t : Triangle) : min_shortest_side t = 12 := by
  sorry

#check min_shortest_side_is_12

end NUMINAMATH_CALUDE_min_shortest_side_is_12_l1192_119299


namespace NUMINAMATH_CALUDE_largest_number_l1192_119217

-- Define the function to convert a number from any base to decimal (base 10)
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def binary : List Nat := [1, 1, 1, 1, 1]
def ternary : List Nat := [1, 2, 2, 1]
def quaternary : List Nat := [2, 1, 3]
def octal : List Nat := [6, 5]

-- State the theorem
theorem largest_number :
  to_decimal quaternary 4 = 54 ∧
  to_decimal quaternary 4 > to_decimal binary 2 ∧
  to_decimal quaternary 4 > to_decimal ternary 3 ∧
  to_decimal quaternary 4 > to_decimal octal 8 :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1192_119217


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l1192_119255

theorem positive_integer_solutions_of_inequality :
  ∀ x : ℕ+, 9 - 3 * (x : ℝ) > 0 ↔ x = 1 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l1192_119255


namespace NUMINAMATH_CALUDE_rational_function_value_l1192_119287

-- Define the polynomials p and q
def p (a b x : ℝ) : ℝ := x * (a * x + b)
def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

-- State the theorem
theorem rational_function_value
  (a b : ℝ)
  (h1 : p a b 1 / q 1 = -1)
  (h2 : a + b = 1/4) :
  p a b (-1) / q (-1) = (a - b) / 4 := by
sorry

end NUMINAMATH_CALUDE_rational_function_value_l1192_119287


namespace NUMINAMATH_CALUDE_cube_equality_solution_l1192_119256

theorem cube_equality_solution : ∃! (N : ℕ), N > 0 ∧ 12^3 * 30^3 = 20^3 * N^3 :=
by
  use 18
  sorry

end NUMINAMATH_CALUDE_cube_equality_solution_l1192_119256


namespace NUMINAMATH_CALUDE_inequality_preservation_l1192_119296

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 1 > b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1192_119296


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1192_119226

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def valid_pairs : ℕ := 4

/-- Definition of a valid pair (m,n) -/
def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ n ∧ n % 2 = 1 ∧ m^2 - n^2 = 120

/-- Theorem stating that there are exactly 4 valid pairs -/
theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = valid_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1192_119226


namespace NUMINAMATH_CALUDE_function_not_monotonic_iff_m_gt_four_l1192_119263

/-- A function f(x) = mln(x+1) + x^2 - mx is not monotonic on (1, +∞) iff m > 4 -/
theorem function_not_monotonic_iff_m_gt_four (m : ℝ) :
  (∃ (x y : ℝ), 1 < x ∧ x < y ∧
    (m * Real.log (x + 1) + x^2 - m * x ≤ m * Real.log (y + 1) + y^2 - m * y ∧
     m * Real.log (y + 1) + y^2 - m * y ≤ m * Real.log (x + 1) + x^2 - m * x)) ↔
  m > 4 :=
by sorry


end NUMINAMATH_CALUDE_function_not_monotonic_iff_m_gt_four_l1192_119263


namespace NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l1192_119213

/-- The fraction halfway between two fractions is their average -/
def halfway (a b : ℚ) : ℚ := (a + b) / 2

/-- The fraction halfway between 1/4 and 1/7 is 11/56 -/
theorem halfway_between_one_fourth_and_one_seventh :
  halfway (1/4) (1/7) = 11/56 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l1192_119213


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1192_119229

-- Problem 1
theorem factorization_problem_1 (a : ℝ) : a^3 - 16*a = a*(a+4)*(a-4) := by sorry

-- Problem 2
theorem factorization_problem_2 (x : ℝ) : (x-2)*(x-4)+1 = (x-3)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1192_119229


namespace NUMINAMATH_CALUDE_max_value_of_product_l1192_119266

theorem max_value_of_product (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 5*y < 105) :
  ∃ (max : ℝ), max = 4287.5 ∧ xy*(105 - 2*x - 5*y) ≤ max ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 5*y₀ < 105 ∧ x₀*y₀*(105 - 2*x₀ - 5*y₀) = max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_product_l1192_119266


namespace NUMINAMATH_CALUDE_lee_test_probability_l1192_119205

theorem lee_test_probability (p_physics : ℝ) (p_chem_given_no_physics : ℝ) 
  (h1 : p_physics = 5/8)
  (h2 : p_chem_given_no_physics = 2/3) :
  (1 - p_physics) * p_chem_given_no_physics = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_lee_test_probability_l1192_119205


namespace NUMINAMATH_CALUDE_quadratic_roots_l1192_119262

theorem quadratic_roots : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, x * (x - 2) = 2 - x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1192_119262


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l1192_119291

theorem ceiling_floor_sum : ⌈(5/4 : ℝ)⌉ + ⌊-(5/4 : ℝ)⌋ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l1192_119291


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1192_119265

theorem right_rectangular_prism_volume
  (face_area1 face_area2 face_area3 : ℝ)
  (h1 : face_area1 = 6.5)
  (h2 : face_area2 = 8)
  (h3 : face_area3 = 13)
  : ∃ (l w h : ℝ),
    l * w = face_area1 ∧
    w * h = face_area2 ∧
    l * h = face_area3 ∧
    l * w * h = 26 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1192_119265


namespace NUMINAMATH_CALUDE_sum_gcf_lcm_18_30_45_l1192_119240

theorem sum_gcf_lcm_18_30_45 : 
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  A + B = 93 := by
sorry

end NUMINAMATH_CALUDE_sum_gcf_lcm_18_30_45_l1192_119240


namespace NUMINAMATH_CALUDE_sum_of_smallest_solutions_l1192_119260

noncomputable def smallest_solutions (x : ℝ) : Prop :=
  x > 2017 ∧ 
  (Real.cos (9*x))^5 + (Real.cos x)^5 = 
    32 * (Real.cos (5*x))^5 * (Real.cos (4*x))^5 + 
    5 * (Real.cos (9*x))^2 * (Real.cos x)^2 * (Real.cos (9*x) + Real.cos x)

theorem sum_of_smallest_solutions :
  ∃ (x₁ x₂ : ℝ), 
    smallest_solutions x₁ ∧ 
    smallest_solutions x₂ ∧ 
    x₁ < x₂ ∧
    (∀ (y : ℝ), smallest_solutions y → y ≥ x₂ ∨ y = x₁) ∧
    x₁ + x₂ = 4064 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smallest_solutions_l1192_119260


namespace NUMINAMATH_CALUDE_no_valid_n_l1192_119237

theorem no_valid_n : ¬ ∃ (n : ℕ), 
  n > 0 ∧ 
  (1000 ≤ n / 5) ∧ (n / 5 ≤ 9999) ∧ 
  (1000 ≤ 5 * n) ∧ (5 * n ≤ 9999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l1192_119237


namespace NUMINAMATH_CALUDE_m_is_fengli_fengli_condition_l1192_119215

/-- Definition of a Fengli number -/
def is_fengli (n : ℤ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

/-- M is a Fengli number -/
theorem m_is_fengli (x y : ℕ) :
  is_fengli (x^2 + 2*x*y + 2*y^2) :=
sorry

/-- Theorem about the value of m for p to be a Fengli number -/
theorem fengli_condition (x y : ℕ) (m : ℤ) (h : x > y) (h' : y > 0) :
  is_fengli (4*x^2 + m*x*y + 2*y^2 - 10*y + 25) ↔ m = 4 ∨ m = -4 :=
sorry

end NUMINAMATH_CALUDE_m_is_fengli_fengli_condition_l1192_119215


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_negative_l1192_119245

/-- A cubic function with a linear term -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A function has two extreme points if its derivative has two distinct real roots -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

theorem extreme_points_imply_a_negative (a : ℝ) :
  has_two_extreme_points a → a < 0 := by sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_negative_l1192_119245


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1192_119248

theorem remainder_divisibility (N : ℤ) : N % 13 = 5 → N % 39 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1192_119248


namespace NUMINAMATH_CALUDE_train_length_calculation_l1192_119261

/-- Given a train that crosses a platform and a signal pole, calculate its length. -/
theorem train_length_calculation
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : platform_length = 200)
  (h2 : platform_crossing_time = 45)
  (h3 : pole_crossing_time = 30)
  : ∃ (train_length : ℝ),
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time ∧
    train_length = 400 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1192_119261


namespace NUMINAMATH_CALUDE_focus_to_asymptote_distance_l1192_119243

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define one asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Statement: The distance from the focus to the asymptote is √3
theorem focus_to_asymptote_distance :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola x y ∧ asymptote x y ∧
  (Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_focus_to_asymptote_distance_l1192_119243


namespace NUMINAMATH_CALUDE_fib_150_mod_5_l1192_119289

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_5 : fib 150 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_5_l1192_119289


namespace NUMINAMATH_CALUDE_delta_value_l1192_119278

theorem delta_value : ∃ Δ : ℂ, (4 * (-3) = Δ^2 + 3) ∧ (Δ = Complex.I * Real.sqrt 15 ∨ Δ = -Complex.I * Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l1192_119278


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1192_119227

/-- Given a rectangle with area 540 square centimeters, if its length is decreased by 15% and
    its width is increased by 20%, the new area will be 550.8 square centimeters. -/
theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 540) : 
  (L * 0.85) * (W * 1.2) = 550.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1192_119227


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l1192_119230

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 18

/-- Predicate to check if a number has exactly 3 digits in base 7 -/
def has_three_digits_base_7 (n : ℕ) : Prop :=
  7^2 ≤ n ∧ n < 7^3

theorem largest_three_digit_square_base_7 :
  M = 18 ∧
  has_three_digits_base_7 (M^2) ∧
  ∀ n : ℕ, n > M → ¬has_three_digits_base_7 (n^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l1192_119230


namespace NUMINAMATH_CALUDE_fraction_problem_l1192_119281

theorem fraction_problem (f : ℚ) : f * 20 + 7 = 17 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1192_119281


namespace NUMINAMATH_CALUDE_red_joker_probability_l1192_119286

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (standard_cards : ℕ)
  (red_jokers : ℕ)
  (black_jokers : ℕ)

/-- Definition of our specific modified deck -/
def our_deck : ModifiedDeck :=
  { total_cards := 54,
    standard_cards := 52,
    red_jokers := 1,
    black_jokers := 1 }

/-- The probability of drawing a specific card from a deck -/
def probability_of_draw (deck : ModifiedDeck) (specific_cards : ℕ) : ℚ :=
  specific_cards / deck.total_cards

theorem red_joker_probability :
  probability_of_draw our_deck our_deck.red_jokers = 1 / 54 := by
  sorry


end NUMINAMATH_CALUDE_red_joker_probability_l1192_119286


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l1192_119241

theorem greatest_x_with_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 12 18) = 180) → x ≤ 180 ∧ ∃ y : ℕ, y = 180 ∧ Nat.lcm y (Nat.lcm 12 18) = 180 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l1192_119241


namespace NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l1192_119275

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_equals_two_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → A m ∩ B = {4}) ∧
  (∃ m : ℝ, m ≠ 2 ∧ A m ∩ B = {4}) :=
sorry

end NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l1192_119275


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l1192_119249

theorem rectangular_plot_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 24 →
  length = 2 * width →
  2 * (length + width) = perimeter →
  length * width = 32 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l1192_119249


namespace NUMINAMATH_CALUDE_abc_is_right_triangle_l1192_119271

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Parabola y^2 = 4x -/
def on_parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Line passes through a point -/
def line_passes_through (l : Line) (p : Point) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  let ab_slope := (t.b.y - t.a.y) / (t.b.x - t.a.x)
  let ac_slope := (t.c.y - t.a.y) / (t.c.x - t.a.x)
  ab_slope * ac_slope = -1

theorem abc_is_right_triangle (a b c : Point) (h1 : a.x = 1 ∧ a.y = 2)
    (h2 : on_parabola b) (h3 : on_parabola c)
    (h4 : line_passes_through (Line.mk b c) (Point.mk 5 (-2))) :
    is_right_triangle (Triangle.mk a b c) := by
  sorry

end NUMINAMATH_CALUDE_abc_is_right_triangle_l1192_119271


namespace NUMINAMATH_CALUDE_problem_statement_l1192_119242

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 2) :
  (∀ x : ℝ, (1/a^2 + 4/b^2 ≥ |2*x - 1| - |x - 1|) → (-9/2 ≤ x ∧ x ≤ 9/2)) ∧
  ((1/a + 1/b) * (a^5 + b^5) ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1192_119242


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1192_119202

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let first_part_runs := game.first_part_run_rate * game.first_part_overs
  let remaining_runs := game.target_runs - first_part_runs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_part_overs = 10)
  (h3 : game.first_part_run_rate = 6.2)
  (h4 : game.target_runs = 282) :
  required_run_rate game = 5.5 := by
  sorry

#eval required_run_rate { 
  total_overs := 50, 
  first_part_overs := 10, 
  first_part_run_rate := 6.2, 
  target_runs := 282 
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1192_119202


namespace NUMINAMATH_CALUDE_divisible_by_225_l1192_119228

theorem divisible_by_225 (n : ℕ) : ∃ k : ℤ, 16^n - 15*n - 1 = 225*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_225_l1192_119228


namespace NUMINAMATH_CALUDE_volume_of_one_gram_l1192_119251

/-- Given a substance where 1 cubic meter has a mass of 100 kg, 
    prove that 1 gram of this substance has a volume of 10 cubic centimeters. -/
theorem volume_of_one_gram (substance_mass : ℝ) (substance_volume : ℝ) 
  (h1 : substance_mass = 100) 
  (h2 : substance_volume = 1) 
  (h3 : (1 : ℝ) = 1000 * (1 / 1000)) -- 1 kg = 1000 g
  (h4 : (1 : ℝ) = 1000000 * (1 / 1000000)) -- 1 m³ = 1,000,000 cm³
  : (1 / 1000) / (substance_mass / substance_volume) = 10 * (1 / 1000000) := by
  sorry

end NUMINAMATH_CALUDE_volume_of_one_gram_l1192_119251


namespace NUMINAMATH_CALUDE_odd_function_property_l1192_119288

-- Define an odd function f from ℝ to ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+2) = -1/f(x)
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -1 / f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_prop : has_property f) : 
  f 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l1192_119288


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1192_119234

theorem polynomial_expansion (x : ℝ) : 
  (5*x^2 + 3*x - 4) * (2*x^3 + x^2 - x + 1) = 
  10*x^5 + 11*x^4 - 10*x^3 - 2*x^2 + 7*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1192_119234


namespace NUMINAMATH_CALUDE_min_adventurers_l1192_119220

/-- Represents a group of adventurers with their gem possessions -/
structure AdventurerGroup where
  rubies : Finset Nat
  emeralds : Finset Nat
  sapphires : Finset Nat
  diamonds : Finset Nat

/-- The conditions for the adventurer group -/
def validGroup (g : AdventurerGroup) : Prop :=
  (g.rubies.card = 4) ∧
  (g.emeralds.card = 10) ∧
  (g.sapphires.card = 6) ∧
  (g.diamonds.card = 14) ∧
  (∀ a ∈ g.rubies, (a ∈ g.emeralds ∨ a ∈ g.diamonds) ∧ ¬(a ∈ g.emeralds ∧ a ∈ g.diamonds)) ∧
  (∀ a ∈ g.emeralds, (a ∈ g.rubies ∨ a ∈ g.sapphires) ∧ ¬(a ∈ g.rubies ∧ a ∈ g.sapphires))

/-- The theorem stating the minimum number of adventurers -/
theorem min_adventurers (g : AdventurerGroup) (h : validGroup g) :
  (g.rubies ∪ g.emeralds ∪ g.sapphires ∪ g.diamonds).card ≥ 18 := by
  sorry


end NUMINAMATH_CALUDE_min_adventurers_l1192_119220


namespace NUMINAMATH_CALUDE_hiring_probability_l1192_119282

theorem hiring_probability (n m k : ℕ) (hn : n = 5) (hm : m = 3) (hk : k = 2) :
  let total_combinations := Nat.choose n m
  let favorable_combinations := total_combinations - Nat.choose (n - k) m
  (favorable_combinations : ℚ) / total_combinations = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_hiring_probability_l1192_119282


namespace NUMINAMATH_CALUDE_unique_n_for_consecutive_prime_products_l1192_119252

def x (n : ℕ) : ℕ := 2 * n + 49

def is_product_of_two_distinct_primes_with_same_difference (m : ℕ) : Prop :=
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ ∃ (d : ℕ), m = p * q ∧ q - p = d

theorem unique_n_for_consecutive_prime_products : 
  ∃! (n : ℕ), n > 0 ∧ 
    is_product_of_two_distinct_primes_with_same_difference (x n) ∧
    is_product_of_two_distinct_primes_with_same_difference (x (n + 1)) ∧
    n = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_n_for_consecutive_prime_products_l1192_119252


namespace NUMINAMATH_CALUDE_product_not_always_greater_l1192_119231

theorem product_not_always_greater : ∃ (a b : ℝ), a * b ≤ a ∨ a * b ≤ b := by
  sorry

end NUMINAMATH_CALUDE_product_not_always_greater_l1192_119231


namespace NUMINAMATH_CALUDE_ways_to_sum_3060_l1192_119218

/-- Represents the number of ways to write a given number as the sum of twos and threes -/
def waysToSum (n : ℕ) : ℕ := sorry

/-- The target number we want to represent -/
def targetNumber : ℕ := 3060

/-- Theorem stating that there are 511 ways to write 3060 as the sum of twos and threes -/
theorem ways_to_sum_3060 : waysToSum targetNumber = 511 := by sorry

end NUMINAMATH_CALUDE_ways_to_sum_3060_l1192_119218


namespace NUMINAMATH_CALUDE_third_shift_members_l1192_119276

theorem third_shift_members (first_shift : ℕ) (second_shift : ℕ) (first_participation : ℚ)
  (second_participation : ℚ) (third_participation : ℚ) (total_participation : ℚ)
  (h1 : first_shift = 60)
  (h2 : second_shift = 50)
  (h3 : first_participation = 20 / 100)
  (h4 : second_participation = 40 / 100)
  (h5 : third_participation = 10 / 100)
  (h6 : total_participation = 24 / 100) :
  ∃ (third_shift : ℕ),
    (first_shift * first_participation + second_shift * second_participation + third_shift * third_participation) /
    (first_shift + second_shift + third_shift) = total_participation ∧
    third_shift = 40 := by
  sorry

end NUMINAMATH_CALUDE_third_shift_members_l1192_119276


namespace NUMINAMATH_CALUDE_divisor_degree_l1192_119201

-- Define the degrees of the polynomials
def deg_dividend : ℕ := 15
def deg_quotient : ℕ := 9
def deg_remainder : ℕ := 4

-- Theorem statement
theorem divisor_degree :
  ∀ (deg_divisor : ℕ),
    deg_dividend = deg_divisor + deg_quotient ∧
    deg_remainder < deg_divisor →
    deg_divisor = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisor_degree_l1192_119201


namespace NUMINAMATH_CALUDE_fraction_of_sales_for_ingredients_l1192_119211

/-- Proves that the fraction of sales used to buy ingredients is 3/5 -/
theorem fraction_of_sales_for_ingredients
  (num_pies : ℕ)
  (price_per_pie : ℚ)
  (amount_remaining : ℚ)
  (h1 : num_pies = 200)
  (h2 : price_per_pie = 20)
  (h3 : amount_remaining = 1600) :
  (num_pies * price_per_pie - amount_remaining) / (num_pies * price_per_pie) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_sales_for_ingredients_l1192_119211


namespace NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l1192_119221

theorem no_real_solution_for_sqrt_equation :
  ¬∃ t : ℝ, Real.sqrt (49 - t^2) + 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l1192_119221


namespace NUMINAMATH_CALUDE_valid_pairs_count_l1192_119295

def count_valid_pairs : ℕ := by sorry

theorem valid_pairs_count :
  count_valid_pairs = 8 :=
by
  have h1 : ∀ a b : ℕ+, (a : ℝ) + 2 / (b : ℝ) = 17 * ((1 : ℝ) / a + 2 * b) →
            (a : ℕ) + b ≤ 150 →
            (a : ℕ) = 17 * b := by sorry
  
  have h2 : ∀ b : ℕ+, b ≤ 8 → (17 * b : ℕ) + b ≤ 150 := by sorry
  
  have h3 : ∀ b : ℕ+, b > 8 → (17 * b : ℕ) + b > 150 := by sorry
  
  sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l1192_119295


namespace NUMINAMATH_CALUDE_store_gross_profit_l1192_119297

theorem store_gross_profit (purchase_price : ℝ) (initial_markup_percent : ℝ) (price_decrease_percent : ℝ) : 
  purchase_price = 210 →
  initial_markup_percent = 25 →
  price_decrease_percent = 20 →
  let original_selling_price := purchase_price / (1 - initial_markup_percent / 100)
  let discounted_price := original_selling_price * (1 - price_decrease_percent / 100)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 14 := by
sorry

end NUMINAMATH_CALUDE_store_gross_profit_l1192_119297


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1192_119279

/-- The distance between the vertices of the hyperbola x²/64 - y²/81 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/64 - y^2/81 = 1
  ∃ x₁ x₂ : ℝ, h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1192_119279


namespace NUMINAMATH_CALUDE_animus_tower_spiders_l1192_119274

/-- The number of spiders hired for the Animus Tower project -/
def spiders_hired (total_workers beavers_hired : ℕ) : ℕ :=
  total_workers - beavers_hired

/-- Theorem stating the number of spiders hired for the Animus Tower project -/
theorem animus_tower_spiders :
  spiders_hired 862 318 = 544 := by
  sorry

end NUMINAMATH_CALUDE_animus_tower_spiders_l1192_119274
