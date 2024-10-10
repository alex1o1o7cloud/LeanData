import Mathlib

namespace circle_number_placement_l2790_279093

theorem circle_number_placement :
  ∃ (a₁ b₁ c₁ d₁ e₁ a₂ b₂ c₂ d₂ e₂ : ℕ),
    (1 ≤ a₁ ∧ a₁ ≤ 9) ∧ (1 ≤ b₁ ∧ b₁ ≤ 9) ∧ (1 ≤ c₁ ∧ c₁ ≤ 9) ∧ (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ e₁ ∧ e₁ ≤ 9) ∧
    (1 ≤ a₂ ∧ a₂ ≤ 9) ∧ (1 ≤ b₂ ∧ b₂ ≤ 9) ∧ (1 ≤ c₂ ∧ c₂ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (1 ≤ e₂ ∧ e₂ ≤ 9) ∧
    b₁ - d₁ = 2 ∧ d₁ - a₁ = 3 ∧ a₁ - c₁ = 1 ∧
    b₂ - d₂ = 2 ∧ d₂ - a₂ = 3 ∧ a₂ - c₂ = 1 ∧
    a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ a₁ ≠ e₁ ∧
    b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ b₁ ≠ e₁ ∧
    c₁ ≠ d₁ ∧ c₁ ≠ e₁ ∧
    d₁ ≠ e₁ ∧
    a₂ ≠ b₂ ∧ a₂ ≠ c₂ ∧ a₂ ≠ d₂ ∧ a₂ ≠ e₂ ∧
    b₂ ≠ c₂ ∧ b₂ ≠ d₂ ∧ b₂ ≠ e₂ ∧
    c₂ ≠ d₂ ∧ c₂ ≠ e₂ ∧
    d₂ ≠ e₂ ∧
    (a₁ ≠ a₂ ∨ b₁ ≠ b₂ ∨ c₁ ≠ c₂ ∨ d₁ ≠ d₂ ∨ e₁ ≠ e₂) :=
by sorry

end circle_number_placement_l2790_279093


namespace min_sum_of_constrained_integers_l2790_279020

theorem min_sum_of_constrained_integers (x y : ℕ) 
  (h1 : x - y < 1)
  (h2 : 2 * x - y > 2)
  (h3 : x < 5) :
  ∃ (a b : ℕ), a + b = 6 ∧ 
    (∀ (x' y' : ℕ), x' - y' < 1 → 2 * x' - y' > 2 → x' < 5 → x' + y' ≥ 6) := by
  sorry

end min_sum_of_constrained_integers_l2790_279020


namespace modulus_of_complex_product_l2790_279012

theorem modulus_of_complex_product : ∃ (z : ℂ), z = (Complex.I - 2) * (2 * Complex.I + 1) ∧ Complex.abs z = 5 := by
  sorry

end modulus_of_complex_product_l2790_279012


namespace tetrahedron_relationships_l2790_279007

/-- Properties of a tetrahedron with inscribed and face-touching spheres -/
structure Tetrahedron where
  ρ : ℝ  -- radius of inscribed sphere
  ρ₁ : ℝ  -- radius of sphere touching face opposite to A
  ρ₂ : ℝ  -- radius of sphere touching face opposite to B
  ρ₃ : ℝ  -- radius of sphere touching face opposite to C
  ρ₄ : ℝ  -- radius of sphere touching face opposite to D
  m₁ : ℝ  -- length of altitude from A to opposite face
  m₂ : ℝ  -- length of altitude from B to opposite face
  m₃ : ℝ  -- length of altitude from C to opposite face
  m₄ : ℝ  -- length of altitude from D to opposite face
  ρ_pos : 0 < ρ
  ρ₁_pos : 0 < ρ₁
  ρ₂_pos : 0 < ρ₂
  ρ₃_pos : 0 < ρ₃
  ρ₄_pos : 0 < ρ₄
  m₁_pos : 0 < m₁
  m₂_pos : 0 < m₂
  m₃_pos : 0 < m₃
  m₄_pos : 0 < m₄

/-- Theorem about relationships in a tetrahedron -/
theorem tetrahedron_relationships (t : Tetrahedron) :
  (2 / t.ρ = 1 / t.ρ₁ + 1 / t.ρ₂ + 1 / t.ρ₃ + 1 / t.ρ₄) ∧
  (1 / t.ρ = 1 / t.m₁ + 1 / t.m₂ + 1 / t.m₃ + 1 / t.m₄) ∧
  (1 / t.ρ₁ = -1 / t.m₁ + 1 / t.m₂ + 1 / t.m₃ + 1 / t.m₄) := by
  sorry

end tetrahedron_relationships_l2790_279007


namespace isosceles_triangle_proof_l2790_279025

/-- Represents the sides of an isosceles triangle --/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle_proof :
  let rope_length : ℝ := 20
  let triangle1 : IsoscelesTriangle := { base := 8, leg := 6 }
  let triangle2 : IsoscelesTriangle := { base := 4, leg := 8 }
  
  -- Part 1
  (triangle1.base + 2 * triangle1.leg = rope_length) ∧
  (triangle1.base - triangle1.leg = 2) ∧
  (is_valid_triangle triangle1.base triangle1.leg triangle1.leg) ∧
  
  -- Part 2
  (triangle2.base + 2 * triangle2.leg = rope_length) ∧
  (is_valid_triangle triangle2.base triangle2.leg triangle2.leg) :=
by sorry

end isosceles_triangle_proof_l2790_279025


namespace find_c_l2790_279032

theorem find_c (p q : ℝ → ℝ) (c : ℝ) 
  (hp : ∀ x, p x = 3 * x - 8)
  (hq : ∀ x, q x = 4 * x - c)
  (h_pq3 : p (q 3) = 14) :
  c = 14 / 3 := by
sorry

end find_c_l2790_279032


namespace min_value_floor_sum_l2790_279066

theorem min_value_floor_sum (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0),
    ⌊(x + y + z) / w⌋ + ⌊(y + z + w) / x⌋ + ⌊(z + w + x) / y⌋ + ⌊(w + x + y) / z⌋ = 9 ∧
    ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
      ⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ ≥ 9 :=
by sorry

end min_value_floor_sum_l2790_279066


namespace reciprocal_sum_theorem_l2790_279017

theorem reciprocal_sum_theorem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 3 * x * y + 2) :
  1 / x + 1 / y = 3 := by
sorry

end reciprocal_sum_theorem_l2790_279017


namespace g_of_2_eq_0_l2790_279016

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4

-- State the theorem
theorem g_of_2_eq_0 : g 2 = 0 := by sorry

end g_of_2_eq_0_l2790_279016


namespace arithmetic_sequence_common_difference_l2790_279011

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ) (h1 : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 7 = 22)
  (h2 : arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 10 = 40) :
  d = 3 := by
  sorry

end arithmetic_sequence_common_difference_l2790_279011


namespace math_expressions_evaluation_l2790_279046

theorem math_expressions_evaluation :
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) →
  (∀ (x : ℝ), x ≥ 0 → (Real.sqrt x) ^ 2 = x) →
  (∀ (x y : ℝ), y ≠ 0 → Real.sqrt (x / y) = Real.sqrt x / Real.sqrt y) →
  (Real.sqrt 5 * Real.sqrt 15 - Real.sqrt 12 = 3 * Real.sqrt 3) ∧
  ((Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1) ∧
  ((Real.sqrt 20 + 5) / Real.sqrt 5 = 2 + Real.sqrt 5) := by
  sorry

end math_expressions_evaluation_l2790_279046


namespace geometric_sequence_property_l2790_279030

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geo : GeometricSequence a q)
  (h_cond : 2 * a 5 = a 3 - a 4)
  (n m : ℕ)
  (h_terms : a 1 = 4 * Real.sqrt (a n * a m)) :
  n + m = 6 := by
sorry

end geometric_sequence_property_l2790_279030


namespace fifth_term_of_sequence_l2790_279050

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence 3 (4*y) 5 = 768 * y^4 := by
  sorry

end fifth_term_of_sequence_l2790_279050


namespace pentagon_angle_measure_l2790_279049

theorem pentagon_angle_measure (a b c d e : ℝ) : 
  -- Pentagon angles sum to 540 degrees
  a + b + c + d + e = 540 ∧
  -- Four angles are congruent
  a = b ∧ b = c ∧ c = d ∧
  -- The fifth angle is 50 degrees more than each of the other angles
  e = a + 50 →
  -- The measure of the fifth angle is 148 degrees
  e = 148 :=
by sorry

end pentagon_angle_measure_l2790_279049


namespace multiplication_error_correction_l2790_279080

theorem multiplication_error_correction (N : ℝ) (x : ℝ) : 
  (((N * x - N / 5) / (N * x)) * 100 = 93.33333333333333) → x = 3 :=
by
  sorry

end multiplication_error_correction_l2790_279080


namespace pink_highlighters_l2790_279009

theorem pink_highlighters (total : ℕ) (yellow : ℕ) (blue : ℕ) (h1 : yellow = 2) (h2 : blue = 4) (h3 : total = 12) :
  total - yellow - blue = 6 := by
  sorry

end pink_highlighters_l2790_279009


namespace grid_toothpicks_l2790_279079

/-- Calculates the total number of toothpicks in a grid with internal lines -/
def total_toothpicks (length width spacing : ℕ) : ℕ :=
  let vertical_lines := length / spacing + 1 + length % spacing
  let horizontal_lines := width / spacing + 1 + width % spacing
  vertical_lines * width + horizontal_lines * length

/-- Proves that a grid of 50x40 toothpicks with internal lines every 10 toothpicks uses 4490 toothpicks -/
theorem grid_toothpicks : total_toothpicks 50 40 10 = 4490 := by
  sorry

end grid_toothpicks_l2790_279079


namespace system_inequalities_solution_l2790_279033

theorem system_inequalities_solution (x : ℝ) :
  (5 / (x + 3) ≥ 1 ∧ x^2 + x - 2 ≥ 0) ↔ ((-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)) :=
by sorry

end system_inequalities_solution_l2790_279033


namespace trio_songs_count_l2790_279086

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  hanna : Nat
  mary : Nat
  alina : Nat
  tina : Nat

/-- Calculates the total number of songs sung by the trios -/
def totalSongs (counts : SongCounts) : Nat :=
  (counts.hanna + counts.mary + counts.alina + counts.tina) / 3

/-- Theorem stating the conditions and the result to be proved -/
theorem trio_songs_count (counts : SongCounts) 
  (hanna_most : counts.hanna = 7 ∧ counts.hanna > counts.alina ∧ counts.hanna > counts.tina)
  (mary_least : counts.mary = 4 ∧ counts.mary < counts.alina ∧ counts.mary < counts.tina)
  (alina_tina_between : counts.alina > 4 ∧ counts.alina < 7 ∧ counts.tina > 4 ∧ counts.tina < 7)
  : totalSongs counts = 7 := by
  sorry

end trio_songs_count_l2790_279086


namespace line_tangent_to_circle_l2790_279071

/-- The line x + y = 2k is tangent to the circle x^2 + y^2 = 4k if and only if k = 2 -/
theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, x + y = 2 * k → x^2 + y^2 = 4 * k) ↔ k = 2 := by
  sorry

end line_tangent_to_circle_l2790_279071


namespace call_center_problem_l2790_279048

theorem call_center_problem (team_a_agents : ℚ) (team_b_agents : ℚ) 
  (team_a_calls_per_agent : ℚ) (team_b_calls_per_agent : ℚ) :
  team_a_agents = (5 / 8) * team_b_agents →
  team_a_calls_per_agent = (2 / 5) * team_b_calls_per_agent →
  let total_calls := team_a_agents * team_a_calls_per_agent + team_b_agents * team_b_calls_per_agent
  (team_b_agents * team_b_calls_per_agent) / total_calls = 8 / 9 := by
  sorry

end call_center_problem_l2790_279048


namespace jackies_break_duration_l2790_279096

/-- Represents Jackie's push-up performance --/
structure PushupPerformance where
  pushups_per_10sec : ℕ
  pushups_per_minute_with_breaks : ℕ
  num_breaks : ℕ

/-- Calculates the duration of each break in seconds --/
def break_duration (perf : PushupPerformance) : ℕ :=
  let pushups_per_minute := perf.pushups_per_10sec * 6
  let total_break_time := (pushups_per_minute - perf.pushups_per_minute_with_breaks) * (10 / perf.pushups_per_10sec)
  total_break_time / perf.num_breaks

/-- Theorem: Jackie's break duration is 8 seconds --/
theorem jackies_break_duration :
  let jackie : PushupPerformance := ⟨5, 22, 2⟩
  break_duration jackie = 8 := by
  sorry

end jackies_break_duration_l2790_279096


namespace total_corn_harvest_l2790_279064

-- Define the cornfield properties
def johnson_field : ℝ := 1
def johnson_yield : ℝ := 80
def johnson_period : ℝ := 2

def smith_field : ℝ := 2
def smith_yield_factor : ℝ := 2

def brown_field : ℝ := 1.5
def brown_yield : ℝ := 50
def brown_period : ℝ := 3

def taylor_field : ℝ := 0.5
def taylor_yield : ℝ := 30
def taylor_period : ℝ := 1

def total_months : ℝ := 6

-- Define the theorem
theorem total_corn_harvest :
  let johnson_total := (total_months / johnson_period) * johnson_yield
  let smith_total := (total_months / johnson_period) * (smith_field * smith_yield_factor * johnson_yield)
  let brown_total := (total_months / brown_period) * (brown_field * brown_yield)
  let taylor_total := (total_months / taylor_period) * taylor_yield
  johnson_total + smith_total + brown_total + taylor_total = 1530 := by
  sorry

end total_corn_harvest_l2790_279064


namespace complex_number_in_third_quadrant_l2790_279005

/-- The complex number z = i² + i³ corresponds to a point in the third quadrant of the complex plane -/
theorem complex_number_in_third_quadrant :
  let z : ℂ := Complex.I^2 + Complex.I^3
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l2790_279005


namespace farmers_wheat_estimate_l2790_279018

/-- The farmer's wheat harvest problem -/
theorem farmers_wheat_estimate (total_harvest : ℕ) (extra_bushels : ℕ) 
  (h1 : total_harvest = 48781)
  (h2 : extra_bushels = 684) :
  total_harvest - extra_bushels = 48097 := by
  sorry

end farmers_wheat_estimate_l2790_279018


namespace rope_section_length_l2790_279036

theorem rope_section_length 
  (total_length : ℝ) 
  (art_fraction : ℝ) 
  (friend_fraction : ℝ) 
  (num_sections : ℕ) :
  total_length = 50 →
  art_fraction = 1/5 →
  friend_fraction = 1/2 →
  num_sections = 10 →
  let remaining_after_art := total_length * (1 - art_fraction)
  let remaining_after_friend := remaining_after_art * (1 - friend_fraction)
  remaining_after_friend / num_sections = 2 := by
sorry

end rope_section_length_l2790_279036


namespace sequence_sum_inequality_l2790_279075

theorem sequence_sum_inequality (a : ℕ → ℝ) (S : ℕ → ℝ) (x : ℝ) :
  a 1 = 1 →
  (∀ n, 2 * a (n + 1) = a n) →
  (∀ n : ℕ, ∀ t ∈ Set.Icc (-1 : ℝ) 1, x^2 + t*x + 1 > S n) →
  x ∈ Set.Iic (((-1:ℝ) - Real.sqrt 5) / 2) ∪ Set.Ici ((1 + Real.sqrt 5) / 2) :=
by sorry

end sequence_sum_inequality_l2790_279075


namespace reciprocal_opposite_sum_l2790_279031

theorem reciprocal_opposite_sum (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : 2*c + 2*d - 3*a*b = -3 := by
  sorry

end reciprocal_opposite_sum_l2790_279031


namespace red_balls_in_box_l2790_279014

theorem red_balls_in_box (total_balls : ℕ) (prob_red : ℚ) (num_red : ℕ) : 
  total_balls = 6 → 
  prob_red = 1/3 → 
  (num_red : ℚ) / total_balls = prob_red → 
  num_red = 2 := by
sorry

end red_balls_in_box_l2790_279014


namespace mary_stickers_l2790_279099

theorem mary_stickers (front_page : ℕ) (other_pages : ℕ) (pages : ℕ) (remaining : ℕ) :
  front_page = 3 →
  other_pages = 7 →
  pages = 6 →
  remaining = 44 →
  front_page + other_pages * pages + remaining = 89 := by
  sorry

end mary_stickers_l2790_279099


namespace sin_18_cos_12_plus_cos_18_sin_12_l2790_279023

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end sin_18_cos_12_plus_cos_18_sin_12_l2790_279023


namespace vector_magnitude_proof_l2790_279065

def vector_problem (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  let sum := (a.1 + b.1, a.2 + b.2)
  dot_product = 10 ∧ 
  magnitude sum = 5 * Real.sqrt 2 ∧ 
  a = (2, 1) →
  magnitude b = 5

theorem vector_magnitude_proof : 
  ∀ (a b : ℝ × ℝ), vector_problem a b :=
sorry

end vector_magnitude_proof_l2790_279065


namespace max_value_theorem_l2790_279059

/-- The function f(x) = x^3 + x -/
def f (x : ℝ) : ℝ := x^3 + x

/-- The theorem stating the maximum value of a√(1 + b^2) -/
theorem max_value_theorem (a b : ℝ) (h : f (a^2) + f (2 * b^2 - 3) = 0) :
  ∃ (M : ℝ), M = (5 * Real.sqrt 2) / 4 ∧ a * Real.sqrt (1 + b^2) ≤ M :=
sorry

end max_value_theorem_l2790_279059


namespace distance_A_proof_l2790_279043

/-- The distance that runner A can run, given the conditions of the problem -/
def distance_A : ℝ := 224

theorem distance_A_proof (time_A time_B beat_distance : ℝ) 
  (h1 : time_A = 28)
  (h2 : time_B = 32)
  (h3 : beat_distance = 32)
  (h4 : distance_A / time_A * time_B = distance_A + beat_distance) : 
  distance_A = 224 := by sorry

end distance_A_proof_l2790_279043


namespace rachel_age_2009_l2790_279038

/-- Rachel's age at the end of 2004 -/
def rachel_age_2004 : ℝ := 47.5

/-- Rachel's uncle's age at the end of 2004 -/
def uncle_age_2004 : ℝ := 3 * rachel_age_2004

/-- The sum of Rachel's and her uncle's birth years -/
def birth_years_sum : ℕ := 3818

/-- The year for which we're calculating Rachel's age -/
def target_year : ℕ := 2009

/-- The base year from which we're calculating -/
def base_year : ℕ := 2004

theorem rachel_age_2009 :
  rachel_age_2004 + (target_year - base_year) = 52.5 ∧
  rachel_age_2004 = uncle_age_2004 / 3 ∧
  (base_year - rachel_age_2004) + (base_year - uncle_age_2004) = birth_years_sum :=
by sorry

end rachel_age_2009_l2790_279038


namespace remainder_problem_l2790_279001

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 5 := by
  sorry

end remainder_problem_l2790_279001


namespace junk_mail_calculation_l2790_279026

/-- Calculates the total number of junk mail pieces per block -/
def total_junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) : ℕ :=
  houses_per_block * mail_per_house

/-- Theorem stating that the total junk mail per block is 640 -/
theorem junk_mail_calculation :
  total_junk_mail_per_block 20 32 = 640 := by
  sorry

end junk_mail_calculation_l2790_279026


namespace largest_power_of_two_dividing_difference_l2790_279028

theorem largest_power_of_two_dividing_difference (n : ℕ) : 
  n = 18^5 - 14^5 → ∃ k : ℕ, 2^k = 64 ∧ 2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k :=
by sorry

end largest_power_of_two_dividing_difference_l2790_279028


namespace solve_equation_l2790_279042

theorem solve_equation : ∃ (A : ℕ), A < 10 ∧ A * 100 + 72 - 23 = 549 :=
by
  -- The proof goes here
  sorry

end solve_equation_l2790_279042


namespace cereal_difference_theorem_l2790_279035

/-- Represents the probability of eating unsweetened cereal -/
def p_unsweetened : ℚ := 3/5

/-- Represents the probability of eating sweetened cereal -/
def p_sweetened : ℚ := 2/5

/-- Number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- Expected difference between days of eating unsweetened and sweetened cereal -/
def expected_difference : ℚ := days_in_year * (p_unsweetened - p_sweetened)

theorem cereal_difference_theorem : 
  expected_difference = 73 := by sorry

end cereal_difference_theorem_l2790_279035


namespace origin_and_point_opposite_sides_l2790_279098

/-- Determines if two points are on opposite sides of a line -/
def areOnOppositeSides (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) < 0

theorem origin_and_point_opposite_sides :
  areOnOppositeSides 0 0 2 1 (-6) 2 1 := by
  sorry

end origin_and_point_opposite_sides_l2790_279098


namespace octagon_arc_length_l2790_279003

/-- The length of an arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (side_length : ℝ) (h : side_length = 5) :
  let circumference := 2 * Real.pi * side_length
  let arc_length := circumference / 8
  arc_length = 1.25 * Real.pi := by
  sorry

end octagon_arc_length_l2790_279003


namespace eight_students_pairing_l2790_279089

theorem eight_students_pairing :
  (Nat.factorial 8) / ((Nat.factorial 4) * (2^4)) = 105 := by
  sorry

end eight_students_pairing_l2790_279089


namespace kelly_snacks_total_weight_l2790_279040

theorem kelly_snacks_total_weight 
  (peanuts_weight : ℝ) 
  (raisins_weight : ℝ) 
  (h1 : peanuts_weight = 0.1)
  (h2 : raisins_weight = 0.4) : 
  peanuts_weight + raisins_weight = 0.5 := by
sorry

end kelly_snacks_total_weight_l2790_279040


namespace factorization_equality_l2790_279037

theorem factorization_equality (x : ℝ) : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := by
  sorry

end factorization_equality_l2790_279037


namespace trigonometric_identity_l2790_279019

theorem trigonometric_identity (α φ : ℝ) : 
  Real.cos α ^ 2 + Real.cos φ ^ 2 + Real.cos (α + φ) ^ 2 - 
  2 * Real.cos α * Real.cos φ * Real.cos (α + φ) = 1 := by
  sorry

end trigonometric_identity_l2790_279019


namespace complex_triple_solution_l2790_279006

theorem complex_triple_solution (x y z : ℂ) :
  (x + y)^3 + (y + z)^3 + (z + x)^3 - 3*(x + y)*(y + z)*(z + x) = 0 →
  x^2*(y + z) + y^2*(z + x) + z^2*(x + y) = 0 →
  x + y + z = 0 ∧ x*y*z = 0 := by
  sorry

end complex_triple_solution_l2790_279006


namespace solve_equation_l2790_279087

theorem solve_equation (y : ℝ) : (45 / 75 = Real.sqrt (3 * y / 75)) → y = 9 := by
  sorry

end solve_equation_l2790_279087


namespace inversion_preserves_angle_l2790_279097

-- Define a type for geometric objects (circles or lines)
inductive GeometricObject
  | Circle : ℝ → ℝ → ℝ → GeometricObject  -- center_x, center_y, radius
  | Line : ℝ → ℝ → ℝ → GeometricObject    -- a, b, c for ax + by + c = 0

-- Define the inversion transformation
def inversion (center : ℝ × ℝ) (k : ℝ) (obj : GeometricObject) : GeometricObject :=
  sorry

-- Define the angle between two geometric objects
def angle_between (obj1 obj2 : GeometricObject) : ℝ :=
  sorry

-- State the theorem
theorem inversion_preserves_angle (center : ℝ × ℝ) (k : ℝ) (obj1 obj2 : GeometricObject) :
  angle_between obj1 obj2 = angle_between (inversion center k obj1) (inversion center k obj2) :=
  sorry

end inversion_preserves_angle_l2790_279097


namespace sam_distance_l2790_279081

/-- Proves that Sam drove 160 miles given the conditions of the problem -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
  (h1 : marguerite_distance = 120)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 160 :=
by sorry

end sam_distance_l2790_279081


namespace not_sufficient_for_congruence_l2790_279044

/-- Two triangles are congruent -/
def triangles_congruent (A B C D E F : Point) : Prop := sorry

/-- The measure of an angle -/
def angle_measure (A B C : Point) : ℝ := sorry

/-- The length of a line segment -/
def segment_length (A B : Point) : ℝ := sorry

/-- Theorem: Given ∠A = ∠F, ∠B = ∠E, and AC = DE, it's not sufficient to determine 
    the congruence of triangles ABC and DEF -/
theorem not_sufficient_for_congruence 
  (A B C D E F : Point) 
  (h1 : angle_measure A B C = angle_measure F E D)
  (h2 : angle_measure B A C = angle_measure E F D)
  (h3 : segment_length A C = segment_length D E) :
  ¬ (triangles_congruent A B C D E F) := by sorry

end not_sufficient_for_congruence_l2790_279044


namespace polynomial_divisibility_l2790_279070

theorem polynomial_divisibility (a b c d : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ ka kb kc kd : ℤ, a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) := by
sorry

end polynomial_divisibility_l2790_279070


namespace function_minimum_and_tangent_line_l2790_279047

/-- The function f(x) = (x-1)(x-a)^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)^2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := (x - a) * (3*x - a - 2)

theorem function_minimum_and_tangent_line 
  (h₁ : ∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, f (-2) 0 ≤ f (-2) x) :
  (a = -2) ∧ 
  (∃ xp : ℝ, xp ≠ 1 ∧ f' (-2) xp = f' (-2) 1 ∧ 
    (9 : ℝ) * xp - f (-2) xp + 23 = 0) := by
  sorry


end function_minimum_and_tangent_line_l2790_279047


namespace hyperbola_focus_equation_l2790_279000

/-- Given a hyperbola of the form x²/m - y² = 1 with one focus at (-2√2, 0),
    prove that m = 7 -/
theorem hyperbola_focus_equation (m : ℝ) : 
  (∃ (x y : ℝ), x^2 / m - y^2 = 1) →  -- hyperbola equation
  ((-2 * Real.sqrt 2, 0) : ℝ × ℝ) ∈ {(x, y) | x^2 / m - y^2 = 1} →  -- focus condition
  m = 7 := by
sorry

end hyperbola_focus_equation_l2790_279000


namespace arithmetic_mean_of_three_numbers_l2790_279034

theorem arithmetic_mean_of_three_numbers (a b c : ℕ) (h : a = 25 ∧ b = 41 ∧ c = 50) : 
  (a + b + c) / 3 = 116 / 3 := by
sorry

end arithmetic_mean_of_three_numbers_l2790_279034


namespace binomial_expansion_properties_l2790_279056

theorem binomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ = 1 ∧ a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -1) :=
by sorry

end binomial_expansion_properties_l2790_279056


namespace drawer_pull_cost_l2790_279092

/-- Given the conditions of Amanda's kitchen upgrade, prove the cost of each drawer pull. -/
theorem drawer_pull_cost (num_knobs : ℕ) (cost_per_knob : ℚ) (num_pulls : ℕ) (total_cost : ℚ) :
  num_knobs = 18 →
  cost_per_knob = 5/2 →
  num_pulls = 8 →
  total_cost = 77 →
  (total_cost - num_knobs * cost_per_knob) / num_pulls = 4 := by
  sorry

end drawer_pull_cost_l2790_279092


namespace price_tags_offer_advantages_l2790_279088

/-- Represents a product in a store -/
structure Product where
  name : String
  price : ℝ

/-- Represents a store with a collection of products -/
structure Store where
  products : List Product
  has_price_tags : Bool

/-- Represents the advantages of using price tags -/
structure PriceTagAdvantages where
  simplifies_purchase : Bool
  reduces_personnel_requirement : Bool
  provides_advertising : Bool
  increases_trust : Bool

/-- Theorem stating that attaching price tags to all products offers advantages -/
theorem price_tags_offer_advantages (store : Store) (h : store.has_price_tags = true) :
  ∃ (advantages : PriceTagAdvantages),
    advantages.simplifies_purchase ∧
    advantages.reduces_personnel_requirement ∧
    advantages.provides_advertising ∧
    advantages.increases_trust :=
  sorry

end price_tags_offer_advantages_l2790_279088


namespace digits_of_2_12_times_5_8_l2790_279022

theorem digits_of_2_12_times_5_8 : 
  (Nat.log 10 (2^12 * 5^8) + 1 : ℕ) = 10 := by
  sorry

end digits_of_2_12_times_5_8_l2790_279022


namespace coat_price_l2790_279083

theorem coat_price (W : ℝ) (h1 : 2*W - 1.9*W = 4) : 1.9*W = 76 := by
  sorry

end coat_price_l2790_279083


namespace even_function_sum_of_angles_l2790_279060

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_sum_of_angles (θ φ : ℝ) :
  IsEven (fun x ↦ Real.cos (x + θ) + Real.sqrt 2 * Real.sin (x + φ)) →
  0 < θ ∧ θ < π / 2 →
  0 < φ ∧ φ < π / 2 →
  Real.cos θ = Real.sqrt 6 / 3 * Real.sin φ →
  θ + φ = 7 * π / 12 := by
  sorry

end even_function_sum_of_angles_l2790_279060


namespace sin_675_degrees_l2790_279084

theorem sin_675_degrees :
  Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_675_degrees_l2790_279084


namespace normal_dist_probability_l2790_279061

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (hσ : σ > 0) : Type := Unit

-- Define the probability function
def P (X : normal_dist 1 σ hσ) (a b : ℝ) : ℝ := sorry

-- Define our theorem
theorem normal_dist_probability 
  (σ : ℝ) (hσ : σ > 0) (X : normal_dist 1 σ hσ) 
  (h : P X 0 1 = 0.4) : P X 0 2 = 0.8 := by sorry

end normal_dist_probability_l2790_279061


namespace solution_in_quadrant_I_l2790_279004

/-- A point (x, y) lies in Quadrant I if both x and y are positive -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The system of equations -/
def system_equations (k x y : ℝ) : Prop :=
  2 * x - y = 5 ∧ k * x^2 + y = 4

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, system_equations k x y ∧ in_quadrant_I x y) ↔ k > 0 :=
sorry

end solution_in_quadrant_I_l2790_279004


namespace tangent_perpendicular_implies_a_equals_one_l2790_279002

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Theorem statement
theorem tangent_perpendicular_implies_a_equals_one (a : ℝ) :
  (f_deriv a 1 * (-1/4) = -1) → a = 1 := by
  sorry

end tangent_perpendicular_implies_a_equals_one_l2790_279002


namespace bedroom_doors_count_l2790_279072

theorem bedroom_doors_count : 
  ∀ (outside_doors bedroom_doors : ℕ) 
    (outside_door_cost bedroom_door_cost total_cost : ℚ),
  outside_doors = 2 →
  outside_door_cost = 20 →
  bedroom_door_cost = outside_door_cost / 2 →
  total_cost = 70 →
  outside_doors * outside_door_cost + bedroom_doors * bedroom_door_cost = total_cost →
  bedroom_doors = 3 := by
sorry

end bedroom_doors_count_l2790_279072


namespace spider_dressing_theorem_l2790_279053

def spider_dressing_orders (n : ℕ) : ℚ :=
  (Nat.factorial (3 * n) * (4 ^ n)) / (6 ^ n)

theorem spider_dressing_theorem (n : ℕ) (hn : n = 8) :
  spider_dressing_orders n = (Nat.factorial (3 * n) * (4 ^ n)) / (6 ^ n) :=
by sorry

end spider_dressing_theorem_l2790_279053


namespace roof_dimension_difference_l2790_279055

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 676 →
  length - width = 39 := by
sorry

end roof_dimension_difference_l2790_279055


namespace imaginary_part_of_complex_expression_l2790_279052

theorem imaginary_part_of_complex_expression :
  Complex.im (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 - Complex.I)^2) = -1 := by
  sorry

end imaginary_part_of_complex_expression_l2790_279052


namespace isosceles_triangle_sides_l2790_279057

/-- Given a rope of length 18 cm forming an isosceles triangle with one side of 5 cm,
    the length of the other two sides can be either 5 cm or 6.5 cm. -/
theorem isosceles_triangle_sides (rope_length : ℝ) (given_side : ℝ) : 
  rope_length = 18 → given_side = 5 → 
  ∃ (other_side : ℝ), (other_side = 5 ∨ other_side = 6.5) ∧ 
  ((2 * other_side + given_side = rope_length) ∨ 
   (2 * given_side + other_side = rope_length)) := by
sorry

end isosceles_triangle_sides_l2790_279057


namespace max_cables_cut_specific_case_l2790_279095

/-- Represents a computer network -/
structure ComputerNetwork where
  total_computers : ℕ
  initial_cables : ℕ
  initial_clusters : ℕ
  final_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut in a computer network -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.initial_cables - (network.total_computers - network.final_clusters)

/-- Theorem stating the maximum number of cables that can be cut in the given scenario -/
theorem max_cables_cut_specific_case :
  let network := ComputerNetwork.mk 200 345 1 8
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut (ComputerNetwork.mk 200 345 1 8)

end max_cables_cut_specific_case_l2790_279095


namespace wizard_elixir_combinations_l2790_279073

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals available. -/
def num_crystals : ℕ := 6

/-- The number of herbs incompatible with one specific crystal. -/
def incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end wizard_elixir_combinations_l2790_279073


namespace ab_value_l2790_279078

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l2790_279078


namespace sad_children_count_l2790_279024

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) :
  total = 60 →
  happy = 30 →
  neither = 20 →
  total - (happy + neither) = 10 :=
by
  sorry

end sad_children_count_l2790_279024


namespace special_tetrahedron_ratio_bounds_l2790_279090

/-- Represents a tetrahedron with specific edge length properties -/
structure SpecialTetrahedron where
  -- The edge lengths
  a : ℝ
  b : ℝ
  -- Conditions on edge lengths
  h_positive : 0 < a ∧ 0 < b
  h_pa_eq_pb : true  -- Represents PA = PB = a
  h_pc_eq_sides : true  -- Represents PC = AB = BC = CA = b
  h_a_lt_b : a < b

/-- The ratio a/b in a special tetrahedron is bounded -/
theorem special_tetrahedron_ratio_bounds (t : SpecialTetrahedron) :
  Real.sqrt (2 - Real.sqrt 3) < t.a / t.b ∧ t.a / t.b < 1 := by
  sorry


end special_tetrahedron_ratio_bounds_l2790_279090


namespace largest_number_l2790_279013

def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def A : Nat := toDecimal [5, 8] 9
def B : Nat := toDecimal [0, 1, 2] 6
def C : Nat := toDecimal [0, 0, 0, 1] 4
def D : Nat := toDecimal [1, 1, 1, 1, 1] 2

theorem largest_number : 
  B > A ∧ B > C ∧ B > D := by sorry

end largest_number_l2790_279013


namespace square_perimeter_when_area_equals_side_l2790_279074

theorem square_perimeter_when_area_equals_side : ∀ s : ℝ,
  s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end square_perimeter_when_area_equals_side_l2790_279074


namespace sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_proof_l2790_279076

/-- The sum of the exterior angles of a pentagon is 360 degrees. -/
theorem sum_exterior_angles_pentagon : ℝ :=
  360

/-- A pentagon has 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the exterior angles of any polygon with n sides. -/
def sum_exterior_angles (n : ℕ) : ℝ := 360

theorem sum_exterior_angles_pentagon_proof :
  sum_exterior_angles pentagon_sides = sum_exterior_angles_pentagon :=
by sorry

end sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_proof_l2790_279076


namespace candidate_percentage_l2790_279008

theorem candidate_percentage (passing_marks total_marks : ℕ) 
  (first_candidate_marks second_candidate_marks : ℕ) : 
  passing_marks = 160 →
  first_candidate_marks = passing_marks - 40 →
  second_candidate_marks = passing_marks + 20 →
  second_candidate_marks = total_marks * 30 / 100 →
  first_candidate_marks * 100 / total_marks = 20 :=
by
  sorry

end candidate_percentage_l2790_279008


namespace cos_C_value_angle_C_measure_l2790_279051

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem cos_C_value (abc : Triangle) 
  (h1 : Real.sin abc.A = 5/13) 
  (h2 : Real.cos abc.B = 3/5) : 
  Real.cos abc.C = -16/65 := by sorry

-- Part 2
theorem angle_C_measure (abc : Triangle) 
  (h : ∃ p : ℝ, (Real.tan abc.A)^2 + p * (Real.tan abc.A + 1) + 1 = 0 ∧ 
                (Real.tan abc.B)^2 + p * (Real.tan abc.B + 1) + 1 = 0) : 
  abc.C = 3 * Real.pi / 4 := by sorry

end cos_C_value_angle_C_measure_l2790_279051


namespace diane_honey_harvest_l2790_279010

/-- Diane's honey harvest problem -/
theorem diane_honey_harvest 
  (last_year_harvest : ℕ) 
  (harvest_increase : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : harvest_increase = 6085) :
  last_year_harvest + harvest_increase = 8564 :=
by sorry

end diane_honey_harvest_l2790_279010


namespace polynomial_factorization_l2790_279029

theorem polynomial_factorization (x y z : ℝ) : 
  x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 = 
  (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) := by
  sorry

end polynomial_factorization_l2790_279029


namespace pastry_solution_l2790_279069

/-- Represents the number of pastries each person has -/
structure Pastries where
  calvin : ℕ
  phoebe : ℕ
  frank : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def pastry_problem (p : Pastries) : Prop :=
  p.grace = 30 ∧
  p.calvin > p.frank ∧
  p.phoebe > p.frank ∧
  p.calvin = p.grace - 5 ∧
  p.phoebe = p.grace - 5 ∧
  p.calvin + p.phoebe + p.frank + p.grace = 97

/-- The theorem stating the solution to the pastry problem -/
theorem pastry_solution (p : Pastries) (h : pastry_problem p) :
  p.calvin - p.frank = 8 ∧ p.phoebe - p.frank = 8 := by
  sorry

end pastry_solution_l2790_279069


namespace morning_ribbons_l2790_279094

theorem morning_ribbons (initial : ℕ) (afternoon : ℕ) (remaining : ℕ) : 
  initial = 38 → afternoon = 16 → remaining = 8 → initial - afternoon - remaining = 14 := by
  sorry

end morning_ribbons_l2790_279094


namespace hyperbola_asymptote_perpendicular_l2790_279063

/-- Given a hyperbola 4x^2 - y^2 = 1, the value of t for which one of its asymptotes
    is perpendicular to the line tx + y + 1 = 0 is ±1/2 -/
theorem hyperbola_asymptote_perpendicular (x y t : ℝ) : 
  (4 * x^2 - y^2 = 1) → 
  (∃ (m : ℝ), (y = m * x ∨ y = -m * x) ∧ 
              (m * (-1/t) = -1 ∨ (-m) * (-1/t) = -1)) → 
  (t = 1/2 ∨ t = -1/2) :=
sorry

end hyperbola_asymptote_perpendicular_l2790_279063


namespace ellipse_major_axis_length_l2790_279015

/-- The length of the major axis of an ellipse with given foci and tangent to y-axis -/
theorem ellipse_major_axis_length : ∀ (f₁ f₂ : ℝ × ℝ),
  f₁ = (10, 25) →
  f₂ = (50, 65) →
  ∃ (x : ℝ), -- point where ellipse is tangent to y-axis
  (∀ (p : ℝ × ℝ), p.1 = 0 → dist p f₁ + dist p f₂ ≥ dist (0, x) f₁ + dist (0, x) f₂) →
  dist (0, x) f₁ + dist (0, x) f₂ = 10 * Real.sqrt 117 :=
by sorry


end ellipse_major_axis_length_l2790_279015


namespace razorback_revenue_per_shirt_l2790_279091

/-- Razorback t-shirt shop sales data -/
structure TShirtSales where
  total_shirts : ℕ
  game_shirts : ℕ
  game_revenue : ℕ

/-- Calculate the revenue per t-shirt -/
def revenue_per_shirt (sales : TShirtSales) : ℚ :=
  sales.game_revenue / sales.game_shirts

/-- Theorem: The revenue per t-shirt is $98 -/
theorem razorback_revenue_per_shirt :
  let sales : TShirtSales := {
    total_shirts := 163,
    game_shirts := 89,
    game_revenue := 8722
  }
  revenue_per_shirt sales = 98 := by
  sorry

end razorback_revenue_per_shirt_l2790_279091


namespace worker_wage_problem_l2790_279027

/-- Represents the daily wages and work days of three workers -/
structure WorkerData where
  a_wage : ℚ
  b_wage : ℚ
  c_wage : ℚ
  a_days : ℕ
  b_days : ℕ
  c_days : ℕ

/-- The theorem statement for the worker wage problem -/
theorem worker_wage_problem (data : WorkerData) 
  (h_ratio : data.a_wage / 3 = data.b_wage / 4 ∧ data.b_wage / 4 = data.c_wage / 5)
  (h_days : data.a_days = 6 ∧ data.b_days = 9 ∧ data.c_days = 4)
  (h_total : data.a_wage * data.a_days + data.b_wage * data.b_days + data.c_wage * data.c_days = 1850) :
  data.c_wage = 125 := by
  sorry

end worker_wage_problem_l2790_279027


namespace sandy_correct_sums_l2790_279082

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 50)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums ≤ total_sums ∧
    (marks_per_correct : ℤ) * correct_sums - 
    (marks_per_incorrect : ℤ) * (total_sums - correct_sums) = total_marks ∧
    correct_sums = 22 := by
  sorry

#check sandy_correct_sums

end sandy_correct_sums_l2790_279082


namespace or_true_iff_not_and_not_false_l2790_279054

theorem or_true_iff_not_and_not_false (p q : Prop) :
  (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
sorry

end or_true_iff_not_and_not_false_l2790_279054


namespace weight_loss_program_l2790_279021

def initial_weight : ℕ := 250
def weeks_phase1 : ℕ := 4
def loss_per_week_phase1 : ℕ := 3
def weeks_phase2 : ℕ := 8
def loss_per_week_phase2 : ℕ := 2

theorem weight_loss_program (w : ℕ) :
  w = initial_weight - (weeks_phase1 * loss_per_week_phase1 + weeks_phase2 * loss_per_week_phase2) →
  w = 222 :=
by sorry

end weight_loss_program_l2790_279021


namespace linear_function_slope_l2790_279067

/-- Given a linear function y = 2x - kx + 1 and two distinct points on its graph,
    if the product of differences is negative, then k > 2 -/
theorem linear_function_slope (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : 
  y₁ = 2*x₁ - k*x₁ + 1 →
  y₂ = 2*x₂ - k*x₂ + 1 →
  x₁ ≠ x₂ →
  (x₁ - x₂) * (y₁ - y₂) < 0 →
  k > 2 := by
sorry

end linear_function_slope_l2790_279067


namespace age_ratio_problem_l2790_279045

/-- Aaron's current age -/
def aaron_age : ℕ := sorry

/-- Beth's current age -/
def beth_age : ℕ := sorry

/-- The number of years until their age ratio is 3:2 -/
def years_until_ratio : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_problem :
  (aaron_age - 4 = 2 * (beth_age - 4)) ∧
  (aaron_age - 6 = 3 * (beth_age - 6)) →
  years_until_ratio = 24 ∧
  (aaron_age + years_until_ratio) * 2 = 3 * (beth_age + years_until_ratio) :=
by sorry

end age_ratio_problem_l2790_279045


namespace emmys_journey_l2790_279068

theorem emmys_journey (total_length : ℚ) 
  (h1 : total_length / 4 + 30 + total_length / 6 = total_length) : 
  total_length = 360 / 7 := by
sorry

end emmys_journey_l2790_279068


namespace complex_modulus_product_l2790_279041

theorem complex_modulus_product : 
  Complex.abs ((5 * Real.sqrt 3 - 5 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 20 * Real.sqrt 6 := by
  sorry

end complex_modulus_product_l2790_279041


namespace seven_mult_five_equals_34_l2790_279085

/-- Custom multiplication operation -/
def custom_mult (A B : ℝ) : ℝ := (A + 2*B) * (A - B)

/-- Theorem stating that 7 * 5 = 34 under the custom multiplication -/
theorem seven_mult_five_equals_34 : custom_mult 7 5 = 34 := by
  sorry

end seven_mult_five_equals_34_l2790_279085


namespace marks_remaining_money_l2790_279077

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - num_items * item_cost

/-- Proves that Mark has $35 left after buying books -/
theorem marks_remaining_money :
  remaining_money 85 10 5 = 35 := by
  sorry

end marks_remaining_money_l2790_279077


namespace complex_power_problem_l2790_279039

theorem complex_power_problem (z : ℂ) (i : ℂ) (h1 : i^2 = -1) (h2 : z * (1 - i) = 1 + i) : z^2016 = 1 := by
  sorry

end complex_power_problem_l2790_279039


namespace linear_system_no_solution_l2790_279058

/-- A system of two linear equations in two variables -/
structure LinearSystem (a : ℝ) :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)
  (h1 : ∀ x y, eq1 x y = a * x + 2 * y - 3)
  (h2 : ∀ x y, eq2 x y = 2 * x + a * y - 2)

/-- The system has no solution -/
def NoSolution (s : LinearSystem a) : Prop :=
  ∀ x y, ¬(s.eq1 x y = 0 ∧ s.eq2 x y = 0)

theorem linear_system_no_solution (a : ℝ) :
  (∃ s : LinearSystem a, NoSolution s) → a = 2 ∨ a = -2 := by
  sorry

end linear_system_no_solution_l2790_279058


namespace divisible_by_27_l2790_279062

theorem divisible_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  ∃ k : ℤ, x + y + z = 27 * k := by
sorry

end divisible_by_27_l2790_279062
