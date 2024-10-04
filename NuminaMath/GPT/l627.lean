import Mathlib
import Mathlib.
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Commute
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Pointwise
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Sequences
import Mathlib.Algebra.Vector
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Determinant
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Geometry
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Lucas
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import MeasureTheory

namespace system_solution_l627_627557

theorem system_solution (a x y : ℝ) :
  (3 * (x - a) ^ 2 + y = 2 - a) ∧ (y ^ 2 + ((x - 2) / (| x | - 2)) ^ 2 = 1) ↔
  (a = 5 / 3 ∧ x = 4 / 3 ∧ y = 0) :=
by sorry

end system_solution_l627_627557


namespace sum_expression_l627_627581

noncomputable def T : ℝ := ∑ n in Finset.range 4900, (1 / (Real.sqrt (n+1 + Real.sqrt ((n+1)^2 - 1))))

theorem sum_expression (a b c : ℕ) (ha : a = 98) (hb : b = 70) (hc : c = 2) (hT : T = a + b * Real.sqrt c) : a + b + c = 170 :=
by
  sorry

end sum_expression_l627_627581


namespace annulus_divide_l627_627310

theorem annulus_divide (r : ℝ) (h₁ : 2 < 14) (h₂ : 2 > 0) (h₃ : 14 > 0)
    (h₄ : π * 196 - π * r^2 = π * r^2 - π * 4) : r = 10 := 
sorry

end annulus_divide_l627_627310


namespace circle_order_l627_627825

noncomputable def radius_A : ℝ := Real.sqrt 10
noncomputable def radius_B : ℝ := 10 / 2
noncomputable def radius_C : ℝ := Real.sqrt 25

theorem circle_order (h₁ : radius_A = Real.sqrt 10)
                     (h₂ : 2 * radius_B * Real.pi = 10 * Real.pi)
                     (h₃ : radius_C * radius_C * Real.pi = 25 * Real.pi) :
  radius_A < radius_B ∧ radius_B = radius_C :=
by
  have rA : radius_A = Real.sqrt 10 := h₁
  have rB : radius_B = 5 := by 
    rwa [mul_comm, mul_assoc] at h₂
    exact h₂ / (2 * Real.pi)
  have rC : radius_C = 5 := by
    rwa [mul_comm, mul_assoc] at h₃
    exact Real.sqrt (25) 

  have rA_lt_rB : radius_A < radius_B :=
    by 
      rw [rA, rB]
      exact Real.sqrt_pos.mpr (by norm_num)
      norm_num
    
  have rB_eq_rC : radius_B = radius_C :=
    by 
      rw [rB, rC]
      exact eq.refl 5
    
  exact ⟨rA_lt_rB, rB_eq_rC⟩
  sorry

end circle_order_l627_627825


namespace sputnik_dog_name_l627_627249

theorem sputnik_dog_name (date correct_name : String) (launched_satellite : date = "November 3, 1957") 
(sputnik2 : launched_satellite = "Sputnik 2") (dog_sent_in_space : String) 
(first_animal_in_space : dog_sent_in_space = "Dog"): dog_sent_in_space = correct_name :=
by
  -- Given the conditions
  have date_correct : date = "November 3, 1957" := sorry
  have satellite_correct : launched_satellite = "Sputnik 2" := sorry
  have animal_correct : dog_sent_in_space = "Dog" := sorry
  
  -- Show the result
  exact sorry

end sputnik_dog_name_l627_627249


namespace find_y_l627_627449

theorem find_y (k p y : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) 
  (h : (y - 2 * k)^2 - (y - 3 * k)^2 = 4 * k^2 - p) : 
  y = -(p + k^2) / (2 * k) :=
sorry

end find_y_l627_627449


namespace gender_matching_probability_l627_627266

def SchoolA : Type := {a // a = "m" ∨ a = "m" ∨ a = "f"}
def SchoolB : Type := {b // b = "m" ∨ b = "f" ∨ b = "f"}

noncomputable def matching_gender_probability : ℚ :=
  -- Defining sets of males and females in each school
  let males_in_a := {"m", "m"} in
  let females_in_a := {"f"} in
  let males_in_b := {"m"} in
  let females_in_b := {"f", "f"} in

  -- Counting the gender matches
  let male_pairs := (males_in_a.card : ℕ) * males_in_b.card in
  let female_pairs := (females_in_a.card : ℕ) * females_in_b.card in

  -- Total possible pairs and favorable pairs
  let total_pairs := (males_in_a.card + females_in_a.card) * (males_in_b.card + females_in_b.card) in
  let favorable_pairs := male_pairs + female_pairs in

  -- Probability calculation
  (favorable_pairs : ℚ) / (total_pairs : ℚ)

theorem gender_matching_probability : matching_gender_probability = 4/9 := by sorry

end gender_matching_probability_l627_627266


namespace sin_2_angle_BAD_l627_627262

noncomputable def leg_length : ℝ := 2
noncomputable def AC : ℝ := 2 * real.sqrt 2
noncomputable def perimeter_ABC : ℝ := 4 + 2 * real.sqrt 2
noncomputable def DA : ℝ := 3
noncomputable def CD : ℝ := 1

theorem sin_2_angle_BAD :
  let BAD := sorry in -- The actual calculation/geometry to define BAD
  let sin_2_BAD := real.sin (2 * BAD) in
  sin_2_BAD = 14 / 25 :=
by
  sorry -- Proof of the theorem

end sin_2_angle_BAD_l627_627262


namespace det_modulo_matrix_l627_627064

noncomputable def matrix_100x100 : Matrix (Fin 100) (Fin 100) ℕ :=
λ i j => (i : ℕ) * (j : ℕ)

theorem det_modulo_matrix :
  (matrix.det matrix_100x100 : ℤ) % 101 = 1 :=
sorry

end det_modulo_matrix_l627_627064


namespace ratio_of_expenditure_l627_627305

variable (A B AE BE : ℕ)

theorem ratio_of_expenditure (h1 : A = 2000) 
    (h2 : A / B = 5 / 4) 
    (h3 : A - AE = 800) 
    (h4: B - BE = 800) :
    AE / BE = 3 / 2 := by
  sorry

end ratio_of_expenditure_l627_627305


namespace train_b_overtakes_train_a_in_120_minutes_l627_627360

/-- Definition of the trains' speeds and the head start of train A -/
def train_a_speed : ℝ := 60
def train_b_speed : ℝ := 80
def head_start_minutes : ℝ := 40
def head_start_hours : ℝ := head_start_minutes / 60
def initial_distance_a : ℝ := train_a_speed * head_start_hours
def relative_speed : ℝ := train_b_speed - train_a_speed
def time_to_overtake_hours : ℝ := initial_distance_a / relative_speed
def time_to_overtake_minutes : ℝ := time_to_overtake_hours * 60

/-- Main theorem stating that Train B will overtake Train A in 120 minutes. -/
theorem train_b_overtakes_train_a_in_120_minutes : time_to_overtake_minutes = 120 := 
by
  sorry

end train_b_overtakes_train_a_in_120_minutes_l627_627360


namespace incorrect_statement_min_value_l627_627595

variable (x y : ℝ)

def conditions := x > 0 ∧ y > 0 ∧ x + 2 * y = 3

theorem incorrect_statement_min_value :
  conditions x y → ¬(∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2 * y = 3 → min (sqrt x + sqrt (2 * y)) = 2) :=
by
  sorry

end incorrect_statement_min_value_l627_627595


namespace problem_statement_l627_627529

theorem problem_statement (x a b : ℝ) (h1 : x^2 + 4 * x + 4 / x + 1 / x^2 = 34) 
(h2 : ∃ a b : ℝ, x = a + real.sqrt b ∧ a > 0 ∧ b > 0) : a + b = 11 :=
sorry

end problem_statement_l627_627529


namespace median_first_twenty_positive_integers_l627_627720

theorem median_first_twenty_positive_integers : 
  let s := {i | 1 ≤ i ∧ i ≤ 20} in
  let sorted_s := List.range' 1 20 in
  (sorted_s[9] + sorted_s[10]) / 2 = 10.5 :=
by {
let s := {i | 1 ≤ i ∧ i ≤ 20},
let sorted_s := List.range' 1 20, -- range' starts at 1 and goes up to 20 (exclusive), thus takes numbers 1 to 20
have h1 : sorted_s[9] = 10 := by rfl,
have h2 : sorted_s[10] = 11 := by rfl,
have h3 : (sorted_s[9] + sorted_s[10]) = 21 := by rw [h1, h2]; rfl,
show (sorted_s[9] + sorted_s[10]) / 2 = 10.5, by rw [h3]; norm_num
}

end median_first_twenty_positive_integers_l627_627720


namespace non_congruent_triangles_count_l627_627960

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627960


namespace find_sum_of_smallest_multiples_l627_627523

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l627_627523


namespace bricks_needed_l627_627003

-- Definitions for dimensions of the brick and wall
def brick_length : ℤ := 25
def brick_width : ℤ := 11
def brick_height : ℤ := 6

def wall_length : ℤ := 800
def wall_height : ℤ := 100
def wall_thickness : ℤ := 5

-- Volume calculations
def volume_wall : ℤ := wall_length * wall_height * wall_thickness
def volume_brick : ℤ := brick_length * brick_width * brick_height

-- Calculation of the number of bricks
def number_of_bricks : ℤ := (volume_wall + volume_brick - 1) / volume_brick  -- + (volume_brick - 1) for ceiling division

theorem bricks_needed :
  number_of_bricks = 243 :=
by
  rw [volume_wall, volume_brick, number_of_bricks]
  norm_num
  sorry

end bricks_needed_l627_627003


namespace gcd_180_450_l627_627090

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l627_627090


namespace find_a_l627_627905

noncomputable def S (n : ℕ) (a : ℤ) : ℤ := 2 ^ n + a
def a1 (a : ℤ) : ℤ := S 1 a
def a2 (a : ℤ) : ℤ := S 2 a - S 1 a
def a3 (a : ℤ) : ℤ := S 3 a - S 2 a

theorem find_a (a : ℤ) : (a1 a) * (a2 a) = a3 a → a = -1 :=
by
  intros h
  sorry

end find_a_l627_627905


namespace remarkable_two_digit_numbers_count_l627_627243

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def has_four_distinct_divisors (n : ℕ) : Prop :=
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ a ≠ b ∧ n = a * b

theorem remarkable_two_digit_numbers_count : 
  (finset.filter (λ n, has_four_distinct_divisors n) (finset.filter two_digit (finset.range 100))).card = 30 :=
by
  sorry

end remarkable_two_digit_numbers_count_l627_627243


namespace marks_in_mathematics_l627_627831

-- Define the marks obtained in each subject and the average
def marks_in_english : ℕ := 86
def marks_in_physics : ℕ := 82
def marks_in_chemistry : ℕ := 87
def marks_in_biology : ℕ := 85
def average_marks : ℕ := 85
def number_of_subjects : ℕ := 5

-- The theorem to prove the marks in Mathematics
theorem marks_in_mathematics : ℕ :=
  let sum_of_marks := average_marks * number_of_subjects
  let sum_of_known_marks := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology
  sum_of_marks - sum_of_known_marks

-- The expected result that we need to prove
example : marks_in_mathematics = 85 := by
  -- skip the proof
  sorry

end marks_in_mathematics_l627_627831


namespace non_congruent_triangles_count_l627_627963

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627963


namespace find_slope_and_sum_l627_627063

-- Define the conditions:
def is_parallelogram (a b c d : ℚ × ℚ) : Prop :=
  (a.1 = b.1) ∧ (c.1 = d.1) ∧ (b.2 = c.2) ∧ (a.2 = d.2) ∧ (a.1 ≠ c.1)

-- Define the given points:
def A : ℚ × ℚ := (5, 20)
def B : ℚ × ℚ := (5, 50)
def C : ℚ × ℚ := (20, 100)
def D : ℚ × ℚ := (20, 70)

-- Define the line passing through the origin and cutting the parallelogram into two congruent parts:
noncomputable def is_congruent_cut (p1 p2 : ℚ × ℚ) : Prop :=
  (p1.1 = 5) ∧ (p2.1 = 20) ∧
  ∃ a : ℚ, p1.2 = 20 + a ∧ p2.2 = 100 - a ∧
  (20 + a) / 5 = (100 - a) / 20

-- State the main problem:
theorem find_slope_and_sum : 
  is_parallelogram A B C D →
  ∃ p1 p2 : ℚ × ℚ, is_congruent_cut p1 p2 ∧ 
  let m := 40 in
  let n := 9 in
  m + n = 49 :=
by 
  sorry

end find_slope_and_sum_l627_627063


namespace number_of_non_congruent_triangles_perimeter_18_l627_627945

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627945


namespace incorrect_judgment_l627_627464

variable (p q : Prop)
variable (hyp_p : p = (3 + 3 = 5))
variable (hyp_q : q = (5 > 2))

theorem incorrect_judgment : 
  (¬ (p ∧ q) ∧ ¬p) = false :=
by
  sorry

end incorrect_judgment_l627_627464


namespace line_through_P_perpendicular_l627_627081

theorem line_through_P_perpendicular 
  (P : ℝ × ℝ) (a b c : ℝ) (hP : P = (-1, 3)) (hline : a = 1 ∧ b = -2 ∧ c = 3) :
  ∃ (a' b' c' : ℝ), (a' * P.1 + b' * P.2 + c' = 0) ∧ (a = b' ∧ b = -a') ∧ (a' = 2 ∧ b' = 1 ∧ c' = -1) := 
by
  use 2, 1, -1
  sorry

end line_through_P_perpendicular_l627_627081


namespace dan_present_age_l627_627002

theorem dan_present_age : ∃ x : ℕ, (x + 18 = 8 * (x - 3)) ∧ x = 6 :=
by
  -- We skip the proof steps
  sorry

end dan_present_age_l627_627002


namespace pencils_bought_l627_627427

theorem pencils_bought (cindi_spent : ℕ) (cost_per_pencil : ℕ) 
  (cindi_pencils : ℕ) 
  (marcia_pencils : ℕ) 
  (donna_pencils : ℕ) :
  cindi_spent = 30 → 
  cost_per_pencil = 1/2 → 
  cindi_pencils = cindi_spent / cost_per_pencil → 
  marcia_pencils = 2 * cindi_pencils → 
  donna_pencils = 3 * marcia_pencils → 
  donna_pencils + marcia_pencils = 480 := 
by
  sorry

end pencils_bought_l627_627427


namespace value_of_a7_l627_627198

-- Define an arithmetic sequence
structure ArithmeticSeq (a : Nat → ℤ) :=
  (d : ℤ)
  (a_eq : ∀ n, a (n+1) = a n + d)

-- Lean statement of the equivalent proof problem
theorem value_of_a7 (a : ℕ → ℤ) (H : ArithmeticSeq a) :
  (2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0) → a 7 = 4 * H.d :=
by
  sorry

end value_of_a7_l627_627198


namespace number_of_non_congruent_triangles_perimeter_18_l627_627944

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627944


namespace complex_power_24_eq_one_l627_627418

theorem complex_power_24_eq_one (z : ℂ) (h : z = (1 - complex.i) / real.sqrt 2) :
  z^24 = 1 := by
  sorry

end complex_power_24_eq_one_l627_627418


namespace isosceles_triangle_vertex_angle_l627_627204

theorem isosceles_triangle_vertex_angle (base_angle : ℝ) (h_base : base_angle = 40) :
  ∑ angles in list of three base angles = 180 :
  vertex_angle base_angle = 100 :=
by
  sorry

end isosceles_triangle_vertex_angle_l627_627204


namespace part1_part2_l627_627135

variable (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := real.log (x - 2) < 0

theorem part1 (h : a = 1) : (2 < x ∧ x < 3) ↔ (p a x ∧ q x) := 
by { sorry }

theorem part2 (h : ∀ x, p a x → q x ∧ ∃ x, q x ∧ ¬p a x) : 1 ≤ a ∧ a ≤ 2 := 
by { sorry }

end part1_part2_l627_627135


namespace fourth_person_height_l627_627700

theorem fourth_person_height:
  ∃ (H : ℤ), let person1 := H,
                 person2 := H + 2,
                 person3 := H + 4,
                 person4 := H + 10
             in (person1 + person2 + person3 + person4) / 4 = 79 → person4 = 85 :=
begin
  sorry
end

end fourth_person_height_l627_627700


namespace non_congruent_triangles_count_l627_627964

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627964


namespace count_remarkable_two_digit_numbers_l627_627240

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def remarkable (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ p1 ≠ p2 ∧ n = p1 * p2

def two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

theorem count_remarkable_two_digit_numbers : 
  { n : ℕ | two_digit n ∧ remarkable n }.to_finset.card = 30 := 
by
  sorry

end count_remarkable_two_digit_numbers_l627_627240


namespace complex_number_real_implies_values_of_a_l627_627531

theorem complex_number_real_implies_values_of_a (a : ℝ) :
  ((a + 1) + (a^2 - 1) * complex.I).im = 0 → a = 1 ∨ a = -1 :=
by 
  sorry

end complex_number_real_implies_values_of_a_l627_627531


namespace sin_double_angle_l627_627881

variable (α : ℝ)
hypothesis (h0 : α ∈ Ioo 0 (π / 2))
hypothesis (h1 : Real.sin α = 3 / 5)

theorem sin_double_angle : Real.sin (2 * α) = 24 / 25 :=
by
  sorry

end sin_double_angle_l627_627881


namespace bob_catch_john_time_alice_meet_john_time_l627_627564

/-- John's speed in mph. --/
def john_speed : ℝ := 4

/-- Bob's speed in mph. --/
def bob_speed : ℝ := 7

/-- Alice's speed in mph. --/
def alice_speed : ℝ := 3

/-- Initial distance between John and Bob. --/
def initial_distance_bob_john : ℝ := 2

/-- Initial distance between John and Alice. --/
def initial_distance_alice_john : ℝ := 2

theorem bob_catch_john_time :
  (initial_distance_bob_john / (bob_speed - john_speed)) * 60 = 40 :=
by
  sorry

theorem alice_meet_john_time :
  (initial_distance_alice_john / (john_speed + alice_speed)) * 60 ≈ 17 :=
by
  -- Floating-point comparison
  sorry

end bob_catch_john_time_alice_meet_john_time_l627_627564


namespace find_line_through_M_and_parallel_l627_627849
-- Lean code to represent the proof problem

def M : Prop := ∃ (x y : ℝ), 3 * x + 4 * y - 5 = 0 ∧ 2 * x - 3 * y + 8 = 0 

def line_parallel : Prop := ∃ (m b : ℝ), 2 * m + b = 0

theorem find_line_through_M_and_parallel :
  M → line_parallel → ∃ (a b c : ℝ), (a = 2) ∧ (b = 1) ∧ (c = 0) :=
by
  intros hM hLineParallel
  sorry

end find_line_through_M_and_parallel_l627_627849


namespace count_ordered_pairs_l627_627510

def log_eq_power (b a : ℝ) (k : ℕ) : Prop :=
  (Real.log a / Real.log b) ^ k = Real.log (a ^ k) / Real.log b

theorem count_ordered_pairs : 
  (∃ s : Finset (ℝ × ℕ), (∀ (a : ℝ) (b : ℕ), (a ∈ s ∧ b ∈ s) ↔ (0 < a ∧ 1 ≤ b ∧ b ≤ 100 ∧ log_eq_power b a 2023)) ∧ s.card = 300) :=
by
  sorry

end count_ordered_pairs_l627_627510


namespace Tyler_age_l627_627590

variable (T B S : ℕ) -- Assuming ages are non-negative integers

theorem Tyler_age (h1 : T = B - 3) (h2 : T + B + S = 25) (h3 : S = B + 2) : T = 6 := by
  sorry

end Tyler_age_l627_627590


namespace minimum_rounds_to_conclude_tournament_l627_627015

def num_of_teams : ℕ := 32
def losses_for_elimination : ℕ := 3

theorem minimum_rounds_to_conclude_tournament 
  (num_of_teams : ℕ) 
  (losses_for_elimination : ℕ) 
  (pairing : list (ℕ × ℕ)) 
  (skip_team_if_odd : bool) 
  (is_eliminated : ℕ → ℕ → bool) 
  (tournament_status : ℕ → bool) 
  (final_team : ℕ)
  (minimum_rounds : ℕ) :
    num_of_teams = 32 →
    losses_for_elimination = 3 →
    (∀ n, tournament_status n ↔ final_team = 1 ∧ ∀ t, t ≠ final_team → is_eliminated t 3) →
    minimum_rounds = 9 := 
by
  sorry

end minimum_rounds_to_conclude_tournament_l627_627015


namespace minimum_shift_symmetric_l627_627067

def f (x : ℝ) : ℝ := (√3) * Real.cos (x / 2) + Real.sin (x / 2)

theorem minimum_shift_symmetric (m : ℝ) (h₀ : m > 0) :
  (∀ x : ℝ, f x = f (-x + m)) ↔ m = (4 * Real.pi) / 3 := sorry

end minimum_shift_symmetric_l627_627067


namespace unit_direction_vector_of_line_l627_627845

theorem unit_direction_vector_of_line (x y : ℝ):
  (y = (3/4) * x - 1) ->
  (∃ d : ℝ × ℝ, (d = (4/5, 3/5) ∨ d = (-4/5, -3/5)) ∧ ∥d∥ = 1) :=
by
  sorry

end unit_direction_vector_of_line_l627_627845


namespace minimum_distance_ln_circle_l627_627473

noncomputable def f (x : ℝ) : ℝ := Real.log x

def circle (e : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - (e + 1/e)) ^ 2 + p.2 ^ 2 = 1}

theorem minimum_distance_ln_circle (e : ℝ) (h : 1 < e) :
  let P := {p : ℝ × ℝ | ∃ (m : ℝ), m > 0 ∧ p = (m, Real.log m)},
      Q := circle e in 
    ∀ p ∈ P, ∀ q ∈ Q, 
    dist p q >= (Real.sqrt (e^2 + 1) - e) / e ∧ 
    (∃ p ∈ P, ∃ q ∈ Q, dist p q = (Real.sqrt (e^2 + 1) - e) / e) :=
  sorry

end minimum_distance_ln_circle_l627_627473


namespace difference_max_min_area_l627_627841

open Int

def perimeter_condition (l w : ℕ) : Prop := 2 * l + 2 * w = 60

def area (l w : ℕ) : ℕ := l * w

theorem difference_max_min_area : 
  (∀ (l w : ℕ), perimeter_condition l w → (area 15 15 - area 1 29 = 196)) :=
by exists_intro [15, 15, 1, 29]
   sorry

end difference_max_min_area_l627_627841


namespace sum_of_valid_a_l627_627108

theorem sum_of_valid_a :
  let relevant_as := {a : ℤ | a >= -3 ∧ a ≠ -1 ∧ a + 4 ∣ 6}
  (relevant_as.sum id) = -5 :=
by
  sorry

end sum_of_valid_a_l627_627108


namespace smallest_prime_dividing_sum_l627_627726

theorem smallest_prime_dividing_sum (h1 : 2 ^ 14 % 2 = 0) (h2 : 7 ^ 9 % 2 = 1) (h3 : (2 ^ 14 + 7 ^ 9) % 2 = 1) : ∃ p : ℕ, p.prime ∧ p ∣ (2 ^ 14 + 7 ^ 9) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (2 ^ 14 + 7 ^ 9) → p ≤ q :=
by
  sorry

end smallest_prime_dividing_sum_l627_627726


namespace problem_solution_l627_627129

def proposition_p (a : Line) (α β : Plane) : Prop :=
  (a ∥ α) → (α ∥ β) → (a ∥ β)

def proposition_q (a : ℝ) : Prop :=
  let radius := (Real.sqrt 6 * a) / 4
  let surface_area := 4 * Real.pi * radius^2
  surface_area = (3 * Real.pi * a^2) / 2

theorem problem_solution (a : Line) (α β : Plane) (length_a : ℝ) :
  ¬ proposition_p a α β ∧ proposition_q length_a → (proposition_p a α β ∨ proposition_q length_a) :=
by
  intro h
  cases h with h₁ h₂
  -- Since ¬p is true and q is true, p ∨ q must be true
  exact Or.inr h₂

end problem_solution_l627_627129


namespace least_number_of_cubes_l627_627000

def gcd (a b : ℕ) : ℕ := nat.gcd a b

noncomputable def volume_of_block (l w h : ℕ) : ℕ := l * w * h

noncomputable def volume_of_cube (side : ℕ) : ℕ := side * side * side

theorem least_number_of_cubes :
  ∃ n : ℕ, let side_length := gcd (gcd 15 30) 75 in
  n = volume_of_block 15 30 75 / volume_of_cube side_length ∧
  side_length = 15 ∧ n = 10 :=
by
  apply Exists.intro 10 sorry

end least_number_of_cubes_l627_627000


namespace frog_jump_positions_l627_627230

theorem frog_jump_positions (p q d : ℕ) (hpq : Nat.gcd p q = 1) (h_frog_returns : ∃ n w : ℤ, 
 (n*p - w*q = 0)) (hd : d < p + q) : ∃ x y : ℤ, ∃ k m : ℕ, 
  (k*p - m*q = x) ∧ (k*p - m*q = y) ∧ (x ≠ y) ∧ (Int.nat_abs (x - y) = d) := 
begin
  sorry
end

end frog_jump_positions_l627_627230


namespace quadrilateral_area_eq_triangle_area_l627_627053

noncomputable def AcuteAngledTriangle (A B C : Type) := sorry -- Define the structure of an acute-angled triangle ABC
noncomputable def OnBC (E F : Type) (B C : Type) := sorry -- Define points E and F on side BC
noncomputable def EqualAngles (BAE CAF : Type) := sorry -- Define the condition for equal angles ∠BAE = ∠CAF
noncomputable def PerpendicularFoot (F M N : Type) (AB AC : Type) := sorry -- Define M and N as feet of perpendiculars from F to AB and AC
noncomputable def ExtendAEtoCircumcircle (A E D : Type) (CircleABC : Type) := sorry -- Define the extension of AE to intersect circumcircle at D
noncomputable def Area (Quadrilateral Triangle : Type) := sorry -- Define a function to calculate area

theorem quadrilateral_area_eq_triangle_area
  {ABC E F A M N D : Type}
  (hABC : AcuteAngledTriangle A B C)
  (hOnBC : OnBC E F B C)
  (hEqualAngles : EqualAngles (∠BAE) (∠CAF))
  (hPerpendicularFoot : PerpendicularFoot F M N AB AC)
  (hExtendAE : ExtendAEtoCircumcircle A E D (circumcircle A B C))
  : Area (Quadrilateral A M D N) = Area (Triangle A B C) :=
sorry

end quadrilateral_area_eq_triangle_area_l627_627053


namespace find_xyz_l627_627435

theorem find_xyz :
  ∃ (x y z : ℤ), z^x = y^(3 * x) ∧ 2^z = 8 * 8^x ∧ x + y + z = 20 ∧
  x = 2 ∧ y = 9 ∧ z = 9 :=
begin
  sorry
end

end find_xyz_l627_627435


namespace prob_black_yellow_green_l627_627192

theorem prob_black_yellow_green 
  (P : Set → ℚ)
  (A B C D : Set)
  (h_disjoint : ∀ (X Y : Set), X ≠ Y → Disjoint X Y)
  (h_total : P(A ∪ B ∪ C ∪ D) = 1)
  (h_A : P(A) = 1/3)
  (h_BC : P(B ∪ C) = 5/12)
  (h_CD : P(C ∪ D) = 5/12) :
  P(B) = 1/4 ∧ P(C) = 1/6 ∧ P(D) = 1/4 := 
sorry

end prob_black_yellow_green_l627_627192


namespace infinitely_many_common_elements_l627_627362

open Nat

-- Define the sequences (a_n) and (b_n)
def a : ℕ → ℤ
| 0       := 2
| 1       := 14
| (n + 2) := 14 * a (n + 1) + a n

def b : ℕ → ℤ
| 0       := 2
| 1       := 14
| (n + 2) := 6 * b (n + 1) - b n

-- Problem statement: There are infinitely many integers that appear in both sequences
theorem infinitely_many_common_elements : ∃ᶠ n in at_top, a n = b n :=
sorry

end infinitely_many_common_elements_l627_627362


namespace number_of_digits_product_l627_627579

theorem number_of_digits_product {Q : ℕ} (h : Q = 1000000001 * 10000000007) : nat.digits 10 Q = 20 :=
sorry

end number_of_digits_product_l627_627579


namespace distance_relationship_l627_627553

theorem distance_relationship
    (r d h : ℝ)
    (A B C D E F O : Point)
    (Circle : O → Real → Set Point)
    (Diameter : A B → ℝ)
    (Tangent : Point → Set Point)
    (H1 : Diameter A B = 2 * r)
    (H2 : OnTangent A C (Tangent A))
    (H3 : D = Extend B C (Tangent B))
    (H4 : E = Extend C B (Tangent B))
    (H5 : BeExtended : BE = DC)
    (H6 : Distance E B Tangent d)
    (H7 : Distance E A Diameter h)
    : h = sqrt (r^2 - d^2) := 
sorry

end distance_relationship_l627_627553


namespace least_number_to_add_for_divisibility_by_11_l627_627731

theorem least_number_to_add_for_divisibility_by_11 : ∃ k : ℕ, 11002 + k ≡ 0 [MOD 11] ∧ k = 9 := by
  sorry

end least_number_to_add_for_divisibility_by_11_l627_627731


namespace equivalent_expr_l627_627671

theorem equivalent_expr (a y : ℝ) (ha : a ≠ 0) (hy : y ≠ a ∧ y ≠ -a) :
  ( (a / (a + y) + y / (a - y)) / ( y / (a + y) - a / (a - y)) ) = -1 :=
by
  sorry

end equivalent_expr_l627_627671


namespace fraction_of_satisfactory_grades_is_four_fifths_l627_627193

theorem fraction_of_satisfactory_grades_is_four_fifths : 
  let A := 8
  let B := 7
  let C := 5
  let D := 4
  let F := 6
  let total_satisfactory := A + B + C + D
  let total_students := total_satisfactory + F
  let satisfactory_fraction := total_satisfactory / total_students.toRat
  satisfactory_fraction = (4 / 5 : ℚ) :=
by
  let A := 8
  let B := 7
  let C := 5
  let D := 4
  let F := 6
  let total_satisfactory := A + B + C + D
  let total_students := total_satisfactory + F
  let satisfactory_fraction := total_satisfactory / total_students.toRat
  have : satisfactory_fraction = (4 / 5 : ℚ) := sorry
  exact this

end fraction_of_satisfactory_grades_is_four_fifths_l627_627193


namespace time_for_train_to_pass_man_l627_627001

theorem time_for_train_to_pass_man :
  ∀ (train_length : ℕ) (train_speed_kmph : ℕ) (man_speed_kmph : ℕ),
  train_length = 165 →
  train_speed_kmph = 60 →
  man_speed_kmph = 6 →
  let relative_speed_mps := ((train_speed_kmph + man_speed_kmph) * 1000) / 3600 in
  let time_seconds := train_length / relative_speed_mps in
  time_seconds ≈ 9 :=
by sorry

end time_for_train_to_pass_man_l627_627001


namespace muffins_equation_l627_627814

def remaining_muffins : ℕ := 48
def total_muffins : ℕ := 83
def initially_baked_muffins : ℕ := 35

theorem muffins_equation : initially_baked_muffins + remaining_muffins = total_muffins :=
  by
    -- Skipping the proof here
    sorry

end muffins_equation_l627_627814


namespace x_plus_q_eq_2q_minus_3_l627_627527

theorem x_plus_q_eq_2q_minus_3 (x q : ℝ) (h1: |x + 3| = q) (h2: x > -3) :
  x + q = 2q - 3 :=
sorry

end x_plus_q_eq_2q_minus_3_l627_627527


namespace combinations_of_balls_and_hats_l627_627793

def validCombinations (b h : ℕ) : Prop :=
  6 * b + 4 * h = 100 ∧ h ≥ 2

theorem combinations_of_balls_and_hats : 
  (∃ (n : ℕ), n = 8 ∧ (∀ b h : ℕ, validCombinations b h → validCombinations b h)) :=
by
  sorry

end combinations_of_balls_and_hats_l627_627793


namespace reflected_ray_equation_l627_627800

variable (P Q : ℝ × ℝ)
variable (L : ℝ → ℝ → Prop)

def equation_of_line (A B : ℝ × ℝ) : Prop :=
∃ a b c : ℝ, a ≠ 0 → b ≠ 0 → a * A.1 + b * A.2 + c = 0 ∧ a * B.1 + b * B.2 + c = 0

theorem reflected_ray_equation :
  P = (1, 1) →
  L = (λ x y, x + y = -1) →
  Q = (2, 3) →
  equation_of_line (-2, -2) Q := 
sorry

end reflected_ray_equation_l627_627800


namespace sequence_odd_l627_627293

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 2 = 7 ∧ ∀ n ≥ 2, -1 / 2 < (a (n + 1) - a n ^ 2 / a (n - 1)) ∧ (a (n + 1) - a n ^ 2 / a (n - 1)) ≤ 1 / 2

theorem sequence_odd (a : ℕ → ℤ) (h : sequence a) : ∀ n > 1, a n % 2 = 1 :=
by
  sorry

end sequence_odd_l627_627293


namespace lacy_correct_percentage_l627_627543

theorem lacy_correct_percentage (x : ℕ) : 
  let total_problems := 6 * x,
      missed_problems := 2 * x,
      correct_problems := total_problems - missed_problems,
      correct_ratio := correct_problems / total_problems,
      correct_percentage := correct_ratio * 100 in 
  correct_percentage = (66.67 : ℝ) := by
  sorry

end lacy_correct_percentage_l627_627543


namespace equation_of_perpendicular_line_intersection_l627_627670

theorem equation_of_perpendicular_line_intersection  :
  ∃ (x y : ℝ), 4 * x + 2 * y + 5 = 0 ∧ 3 * x - 2 * y + 9 = 0 ∧ 
               (∃ (m : ℝ), m = 2 ∧ 4 * x - 2 * y + 11 = 0) := 
sorry

end equation_of_perpendicular_line_intersection_l627_627670


namespace non_congruent_triangles_with_perimeter_18_l627_627984

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627984


namespace concurrency_of_lines_l627_627030

noncomputable def acute_triangle (A B C : Point) : Prop :=
  ∠ BAC < 90 ∧ ∠ ABC < 90 ∧ ∠ ACB < 90

noncomputable def point_D_condition (A B C D : Point) : Prop :=
  ∠ BAD = ∠ DAC

noncomputable def point_E_condition (A C D E : Point) : Prop :=
  E ∈ interval A C ∧ ∠ ADE = ∠ DCB

noncomputable def point_F_condition (A B D F : Point) : Prop :=
  F ∈ interval A B ∧ ∠ ADF = ∠ DBC

noncomputable def point_X_condition (A C X B : Point) : Prop :=
  X ∈ line AC ∧ CX = BX

noncomputable def circumcenter (P Q R : Point) : Point := sorry

noncomputable def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

theorem concurrency_of_lines (A B C D E F X O1 O2 : Point) 
  (h_triangle : acute_triangle A B C) 
  (h_AB_AC: dist A B > dist A C)
  (h_D: point_D_condition A B C D) 
  (h_E: point_E_condition A C D E) 
  (h_F: point_F_condition A B D F) 
  (h_X: point_X_condition A C X B) 
  (h_O1: O1 = circumcenter A D C) 
  (h_O2: O2 = circumcenter D X E) : 
  are_concurrent (line B C) (line E F) (line O1 O2) :=
sorry

end concurrency_of_lines_l627_627030


namespace find_a_for_perpendicular_lines_l627_627928

theorem find_a_for_perpendicular_lines (a : ℝ) 
    (h_perpendicular : 2 * a + (-1) * (3 - a) = 0) :
    a = 1 :=
by
  sorry

end find_a_for_perpendicular_lines_l627_627928


namespace problem_sin_cos_ineq_max_m_inequality_l627_627147

theorem problem_sin_cos_ineq (ω φ : ℝ) (hω : ω > 0) (hφ : -π/2 < φ ∧ φ < π/2)
  (P : (ℝ × ℝ)) (hP : P = (0, 1)) (hT : ∃ T > 0, ∀ (t : ℝ), sin (ω * (t + T) + φ) + 1 = sin (ω * t + φ) + 1) :
  let f := λ x : ℝ, sin (ω * x + φ) + 1 in
  ω = 2 ∧ φ = 0 ∧ f = (λ x : ℝ, sin (2 * x) + 1) :=
begin
  sorry
end

-- Maximum value of m such that h(x) = sqrt(2) * sin(2x - π/4) is monotonic in (0, m).
theorem max_m_inequality (m : ℝ) :
  let g := λ x : ℝ, sin (2 * x) + cos (2 * x) - 1 in
  (∀ x, 0 < x ∧ x < m → ∀ y, 0 < y ∧ y < m → (sqrt 2 * sin (2 * (x - π/4) + π/4)) ≤ (sqrt 2 * sin (2 * (y - π/4) + π/4))) → m = 3 * π / 8 :=
begin
  sorry
end

end problem_sin_cos_ineq_max_m_inequality_l627_627147


namespace num_three_digit_perfect_cubes_divisible_by_16_l627_627162

def is_three_digit (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = x

def is_divisible_by_16 (x : ℕ) : Prop :=
  x % 16 = 0

theorem num_three_digit_perfect_cubes_divisible_by_16 :
  {x : ℕ // is_three_digit x ∧ is_perfect_cube x ∧ is_divisible_by_16 x}.card = 1 :=
by sorry

end num_three_digit_perfect_cubes_divisible_by_16_l627_627162


namespace students_not_picked_correct_l627_627698

-- Define the total number of students and the number of students picked for the team
def total_students := 17
def students_picked := 3 * 4

-- Define the number of students who didn't get picked based on the conditions
noncomputable def students_not_picked : ℕ := total_students - students_picked

-- The theorem stating the problem
theorem students_not_picked_correct : students_not_picked = 5 := 
by 
  sorry

end students_not_picked_correct_l627_627698


namespace non_congruent_triangles_with_perimeter_18_l627_627985

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627985


namespace log_problem_l627_627885

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_problem :
  let x := (log_base 8 2) ^ (log_base 2 8)
  log_base 3 x = -3 :=
by
  sorry

end log_problem_l627_627885


namespace total_boys_l627_627356

noncomputable def total_boys_camp : ℕ := 550

theorem total_boys (T : ℝ) (h1 : 0.20 * T = (0.20 * T)) 
  (h2 : 0.30 * (0.20 * T) = 0.30 * (0.20 * T))
  (h3 : 0.70 * (0.20 * T) = 77) : T = 550 := by
suffices : T = 77 / 0.14
  by rw this
  ring_nf
  sorry

end total_boys_l627_627356


namespace num_ways_to_color_board_l627_627827

def is_valid_coloring (board : Array (Array Bool)) : Prop :=
  board.size = 4 ∧ (∀ row, row < 4 → Array.count (λ b, b) (board[row]) = 2) ∧
  (∀ col, col < 4 → Array.count (λ b, b) ([board[i][col] | i] : Array Bool) = 2)

noncomputable def number_of_valid_colorings : Nat :=
  -- This is a placeholder for the actual number of ways, which is given as 90
  90

theorem num_ways_to_color_board : number_of_valid_colorings = 90 :=
by
  sorry

end num_ways_to_color_board_l627_627827


namespace sqrt_sum_of_fractions_l627_627346

theorem sqrt_sum_of_fractions :
  let a := (25 / 49 : ℚ)
  let b := (16 / 81 : ℚ)
  (Real.sqrt (a + b) = 53 / 63) :=
by
  let a := (25 / 49 : ℚ)
  let b := (16 / 81 : ℚ)
  have h : a + b = 2809 / 3969 := sorry
  rw h
  have h2 : Real.sqrt (2809 / 3969) = 53 / 63 := sorry
  rw h2
  exact rfl

end sqrt_sum_of_fractions_l627_627346


namespace smallest_prime_2_pow_14_plus_7_pow_9_is_7_l627_627729
noncomputable def smallest_prime_dividing_2_pow_14_plus_7_pow_9 : ℕ :=
if prime 2 ∧ (2 ∣ (2^14 + 7^9)) then 2 else 
if prime 3 ∧ (3 ∣ (2^14 + 7^9)) then 3 else 
if prime 5 ∧ (5 ∣ (2^14 + 7^9)) then 5 else
if prime 7 ∧ (7 ∣ (2^14 + 7^9)) then 7 else 0 -- considering 0 indicates none of the above primes divide the sum

theorem smallest_prime_2_pow_14_plus_7_pow_9_is_7 :
  smallest_prime_dividing_2_pow_14_plus_7_pow_9 = 7 := by
  sorry

end smallest_prime_2_pow_14_plus_7_pow_9_is_7_l627_627729


namespace meeting_probability_l627_627026

def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

def arrival_conditions (x y z : ℝ) : Prop :=
  z ≤ x ∧ z ≤ y ∧ abs (x - y) ≤ 0.5 ∧ abs (x - z) ≤ 0.5 ∧ abs (y - z) ≤ 0.5

def valid_probability_space_3d (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z, in_interval x → in_interval y → in_interval z → 0 ≤ f x y z ∧ f x y z ≤ 1

def integrable (f : ℝ → ℝ → ℝ → ℝ) :=
  ∫∫∫ (x y z : ℝ) in H, f x y z = 1

noncomputable def volume_indicator (x y z : ℝ) : ℝ :=
  if arrival_conditions x y z then 1 else 0

theorem meeting_probability :
  (∫∫∫ (x y z : ℝ) in (0:ℝ)..2, (0:ℝ)..2, (0:ℝ)..2, volume_indicator x y z) / (2 * 2 * 2) = 1 / 64 :=
by
  sorry

end meeting_probability_l627_627026


namespace domain_of_sqrt_fun_l627_627287

def f (x : ℝ) : ℝ := sqrt (x - 4)

theorem domain_of_sqrt_fun : { x : ℝ | ∃ y : ℝ, f x = y } = { x : ℝ | x ≥ 4 } :=
sorry

end domain_of_sqrt_fun_l627_627287


namespace min_distinct_abs_diff_l627_627065

theorem min_distinct_abs_diff (a : ℕ → ℕ) (h_distinct : function.injective a)
  (h_sum_distinct : finset.card (finset.image (λ (ij : fin (20) × fin (20)), a ij.1 + a ij.2) (finset.univ.product finset.univ)) = 201) :
  finset.card (finset.image (λ (ij : fin (20) × fin (20)), (a ij.1 - a ij.2).natAbs) (finset.univ.off_diag)) ≥ 100 := by
sorry

end min_distinct_abs_diff_l627_627065


namespace general_formula_a_general_formula_b_max_value_m_l627_627482

-- Conditions setup
variable (a : ℕ → ℤ) (b : ℕ → ℝ) (S : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ)
variable (n m : ℕ)
variable (h1 : ∀ n, a n = 3 * n - 4)
variable (h2 : ∀ n, 4 * S n = b n ^ 2 + 2 * b n - 3)
variable (h3 : ∀ n > 0, b n = 2 * n + 1)
variable (h4 : ∀ n, c n = 1 / ((2 * a n + 5) * b n))
variable (h5 : ∀ n, T n = (1 / 6) * (1 - 1 / (2 * T n + 1)))

-- Lean statements to prove
theorem general_formula_a :
  ∀ n, a n = 3 * n - 4 :=
by
  intro n
  exact h1 n

theorem general_formula_b :
  ∀ n, b n = 2 * n + 1 :=
by
  intro n
  exact h3 n (by linarith)

theorem max_value_m (h : ∀ n > 0, (T n / T (n + 1) ≥ a m / a (m + 1))) :
  m ≤ 6 :=
sorry

end general_formula_a_general_formula_b_max_value_m_l627_627482


namespace non_congruent_triangles_with_perimeter_18_l627_627986

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627986


namespace find_c_l627_627128

variable {x a b c : ℝ}

-- Conditions:
def eq1 : Prop := ∀ x, (x + a) * (x + b) = x^2 + c * x + 12
def b_value : Prop := b = 4
def sum_ab : Prop := a + b = 6

-- Theorem statement:
theorem find_c (h1 : eq1) (h2 : b_value) (h3 : sum_ab) : c = 6 := 
  sorry

end find_c_l627_627128


namespace rahul_matches_played_l627_627359

theorem rahul_matches_played
  (current_avg runs_today new_avg : ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 69)
  (h3 : new_avg = 54)
  : ∃ m : ℕ, ((51 * m + 69) / (m + 1) = 54) ∧ (m = 5) :=
by
  sorry

end rahul_matches_played_l627_627359


namespace shift_line_left_2_units_l627_627183

theorem shift_line_left_2_units :
  ∀ (x y : ℝ),
    (y = -2 * x + 1) →
    (∃ x' y', x' = x + 2 ∧ y = -2 * x' + 1) →
    y = -2 * x - 3 :=
by
  intro x y h₁ h₂
  obtain ⟨x', y', h₃, h₄⟩ := h₂
  subst h₃
  rw h₄ at h₁
  rw [←h₁, ←add_assoc]
  sorry

end shift_line_left_2_units_l627_627183


namespace ajax_weight_after_exercise_l627_627042

theorem ajax_weight_after_exercise :
  ∀ (initial_weight_kg : ℕ) (conversion_rate : ℝ) (daily_exercise_hours : ℕ) (exercise_loss_rate : ℝ) (days_in_week : ℕ) (weeks : ℕ),
    initial_weight_kg = 80 →
    conversion_rate = 2.2 →
    daily_exercise_hours = 2 →
    exercise_loss_rate = 1.5 →
    days_in_week = 7 →
    weeks = 2 →
    initial_weight_kg * conversion_rate - daily_exercise_hours * exercise_loss_rate * (days_in_week * weeks) = 134 :=
by
  intros
  sorry

end ajax_weight_after_exercise_l627_627042


namespace max_pressure_theorem_l627_627631

variables (R V₀ T₀ a b c : ℝ)
variable (c2_lt_a2_b2 : c^2 < a^2 + b^2)

def max_pressure :=
  R * T₀ / V₀ * ((a * (a^2 + b^2 - c^2).sqrt + b * c) / (b * (a^2 + b^2 - c^2).sqrt - a * c))

theorem max_pressure_theorem :
  ∃ P_max, P_max = max_pressure R V₀ T₀ a b c ∧
    ∀ (V T : ℝ), 
    (V / V₀ - a)^2 + (T / T₀ - b)^2 = c^2 → 
    ∀ (P : ℝ), 
    P = R * T / V → 
    P ≤ max_pressure R V₀ T₀ a b c :=
sorry

end max_pressure_theorem_l627_627631


namespace absolute_value_equation_solution_l627_627111

theorem absolute_value_equation_solution (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) ↔
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨ 
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨ 
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by
  sorry

end absolute_value_equation_solution_l627_627111


namespace part_one_part_two_part_three_l627_627495

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x - 1 - x - a * x^2

theorem part_one {x : ℝ} : f x 0 ≥ 0 :=
sorry

theorem part_two {a : ℝ} : (∀ x ≥ 0, f x a ≥ 0) ↔ a ∈ Set.Iic (1 / 2) :=
sorry

theorem part_three {x : ℝ} (h : x > 0) : (Real.exp x - 1) * Real.log (x + 1) > x^2 :=
sorry

end part_one_part_two_part_three_l627_627495


namespace multiple_rel_to_carol_l627_627689

-- Definitions
def bob_age := 16
def carol_age := 50
def sum_ages := 66
def carol_age_rel_to_bob (m : Nat) := carol_age = m * bob_age + 2

-- Theorem statement
theorem multiple_rel_to_carol : ∃ m, carol_age_rel_to_bob m ∧ m = 3 := by
  have h1 : carol_age = 50 := rfl
  have h2 : bob_age = 16 := rfl
  have h3 : carol_age = 16 * 3 + 2 := by
    calc
      50 = 48 + 2 : by norm_num
       ... = 16 * 3 + 2 : by norm_num
  exact ⟨3, h3, rfl⟩

end multiple_rel_to_carol_l627_627689


namespace product_of_six_distinct_divisible_by_perfect_square_l627_627271

theorem product_of_six_distinct_divisible_by_perfect_square :
  ∀ (s : Finset ℕ), (s ⊆ (Finset.range 10).map (Nat.succ)) ∧ s.card = 6 →
  ∃ k : ℕ, 1 < k ∧ k * k ∣ s.prod id :=
by
  sorry

end product_of_six_distinct_divisible_by_perfect_square_l627_627271


namespace range_of_a_l627_627483

theorem range_of_a (a : ℝ) (x : ℝ) : (∃ x, x^2 - a*x - a ≤ -3) → (a ≤ -6 ∨ a ≥ 2) :=
sorry

end range_of_a_l627_627483


namespace smallest_prime_2_pow_14_plus_7_pow_9_is_7_l627_627728
noncomputable def smallest_prime_dividing_2_pow_14_plus_7_pow_9 : ℕ :=
if prime 2 ∧ (2 ∣ (2^14 + 7^9)) then 2 else 
if prime 3 ∧ (3 ∣ (2^14 + 7^9)) then 3 else 
if prime 5 ∧ (5 ∣ (2^14 + 7^9)) then 5 else
if prime 7 ∧ (7 ∣ (2^14 + 7^9)) then 7 else 0 -- considering 0 indicates none of the above primes divide the sum

theorem smallest_prime_2_pow_14_plus_7_pow_9_is_7 :
  smallest_prime_dividing_2_pow_14_plus_7_pow_9 = 7 := by
  sorry

end smallest_prime_2_pow_14_plus_7_pow_9_is_7_l627_627728


namespace find_line_equation_l627_627904

theorem find_line_equation
  (P : ℝ × ℝ) (l1 l2 : ℝ → ℝ → Prop)
  (hP : P = (3, 1))
  (hl1 : ∀ x y : ℝ, l1 x y ↔ x + y + 1 = 0)
  (hl2 : ∀ x y : ℝ, l2 x y ↔ x + y + 6 = 0)
  (h_length : ∀ A B C D : ℝ × ℝ, l1 A.1 A.2 ∧ l2 B.1 B.2 →
    l1 C.1 C.2 ∧ l2 D.1 D.2 → (real.dist A B = 5 ∨ real.dist C D = 5)) :
  ∃ m b : ℝ, (m = -1 ∧ ∀ x y : ℝ, (y = m * x + b) ↔ y = -x + 4) :=
begin
  sorry
end

end find_line_equation_l627_627904


namespace remainder_when_divided_l627_627454

theorem remainder_when_divided (k : ℕ) (h_pos : 0 < k) (h_rem : 80 % k = 8) : 150 % (k^2) = 69 := by 
  sorry

end remainder_when_divided_l627_627454


namespace log_base_3_of_27_l627_627804

theorem log_base_3_of_27 :
  let r := log 10 (3^3)
  let p := log 2 (27^4)
  log 3 27 = 3 :=
by
  sorry

end log_base_3_of_27_l627_627804


namespace reflections_on_circumcircle_l627_627576

variables {A B C H : Type} [triangle ABC] [orthocenter H ABC]

theorem reflections_on_circumcircle (H_in_triangle : H ∈ interior_triangle ABC) 
  (H_reflections_ABC : reflections H ABC = reflections_on_circumcircle_of ABC) : 
  ∀ H', H' ∈ reflections H ABC → H' ∈ circumcircle ABC :=
sorry

end reflections_on_circumcircle_l627_627576


namespace exists_dividing_line_l627_627320

theorem exists_dividing_line (points : Finset (ℝ × ℝ)) (h_collinear : ∀ {a b c d : ℝ × ℝ}, a ≠ b → b ≠ c → c ≠ d → a ≠ d → 
    AffineIndependent ℝ ![a, b, c, d])
  (blue_points red_points : Finset (ℝ × ℝ))
  (h_total : points.card = 1988)
  (h_blue : blue_points.card = 1788)
  (h_red : red_points.card = 200)
  (h_disjoint : Disjoint blue_points red_points)
  (h_union : blue_points ∪ red_points = points) :
  ∃ l : ℝ × ℝ → ℝ, (set.count (λ p, l p < 0) blue_points = 894 ∧ set.count (λ p, l p < 0) red_points = 100) :=
by
  sorry

end exists_dividing_line_l627_627320


namespace minimum_value_expression_l627_627683

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 5) : 
  x^2 + y^2 + 2 * z^2 - x^2 * y^2 * z ≥ -6 := 
begin
  sorry
end

end minimum_value_expression_l627_627683


namespace find_angle_C_find_area_ABC_l627_627539

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Conditions
def condition1 : Prop := a * (Real.sin A - Real.sin B) + b * Real.sin B = c * Real.sin C
def condition2 : Prop := a^2 + b^2 = 6*(a + b) - 18

-- Problems to be proved
theorem find_angle_C (h : condition1) : C = π / 3 := 
sorry

theorem find_area_ABC (h1 : condition1) (h2 : condition2) (hC : C = π / 3) : 
  let area := (3 * 3 * Real.sqrt 3) / 4 in
  area = 9 * Real.sqrt 3 / 4 :=
sorry

end find_angle_C_find_area_ABC_l627_627539


namespace Donna_and_Marcia_total_pencils_l627_627423

def DonnaPencils (CindiPencils MarciaPencils DonnaPencils : ℕ) : Prop :=
  DonnaPencils = 3 * MarciaPencils

def MarciaPencils (CindiPencils MarciaPencils : ℕ) : Prop :=
  MarciaPencils = 2 * CindiPencils

def CindiPencils (CindiSpent CindiPencilCost CindiPencils : ℕ) : Prop :=
  CindiPencils = CindiSpent / CindiPencilCost

theorem Donna_and_Marcia_total_pencils (CindiSpent CindiPencilCost : ℕ) (DonnaPencils MarciaPencils CindiPencils : ℕ)
  (hCindi : CindiPencils CindiSpent CindiPencilCost CindiPencils)
  (hMarcia : MarciaPencils CindiPencils MarciaPencils)
  (hDonna : DonnaPencils CindiPencils MarciaPencils DonnaPencils) :
  DonnaPencils + MarciaPencils = 480 := 
sorry

end Donna_and_Marcia_total_pencils_l627_627423


namespace equilateral_A1C1E1_l627_627629

variables {A B C D E F A₁ B₁ C₁ D₁ E₁ F₁ : Type*}

-- Defining the convex hexagon and the equilateral triangles.
def is_convex_hexagon (A B C D E F : Type*) : Prop := sorry

def is_equilateral (P Q R : Type*) : Prop := sorry

-- Given conditions
variable (h_hexagon : is_convex_hexagon A B C D E F)
variable (h_eq_triangles :
  is_equilateral A B C₁ ∧ is_equilateral B C D₁ ∧ is_equilateral C D E₁ ∧
  is_equilateral D E F₁ ∧ is_equilateral E F A₁ ∧ is_equilateral F A B₁)
variable (h_B1D1F1 : is_equilateral B₁ D₁ F₁)

-- Statement to be proved
theorem equilateral_A1C1E1 :
  is_equilateral A₁ C₁ E₁ :=
sorry

end equilateral_A1C1E1_l627_627629


namespace daily_wage_of_C_l627_627349

theorem daily_wage_of_C :
  ∀ (a b c : ℝ), (a : b : c) = 3 : 4 : 5 ∧
  6 * a + 9 * b + 4 * c = 1628 →
  c = 110 :=
by 
  sorry

end daily_wage_of_C_l627_627349


namespace units_digit_of_45_pow_125_plus_7_pow_87_l627_627005

theorem units_digit_of_45_pow_125_plus_7_pow_87 :
  (45 ^ 125 + 7 ^ 87) % 10 = 8 :=
by
  -- sorry to skip the proof
  sorry

end units_digit_of_45_pow_125_plus_7_pow_87_l627_627005


namespace hundredth_number_is_112_l627_627878

-- Definitions
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m * m * m = n

def non_square_non_cube_seq : ℕ → ℕ
| 0     => 1
| (n+1) => if h : is_perfect_square (non_square_non_cube_seq n + 1) ∨ is_perfect_cube (non_square_non_cube_seq n + 1) 
           then non_square_non_cube_seq (n+1)
           else non_square_non_cube_seq n + 1

-- Theorem
theorem hundredth_number_is_112 : non_square_non_cube_seq 99 = 112 :=
sorry

end hundredth_number_is_112_l627_627878


namespace abs_neg_two_l627_627658

theorem abs_neg_two : abs (-2) = 2 := 
by 
  sorry

end abs_neg_two_l627_627658


namespace number_of_non_congruent_triangles_perimeter_18_l627_627947

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627947


namespace proof_intersection_l627_627154

-- Define set A
def A : set ℤ := {x | x^2 - 4 * x ≤ 0}

-- Define set B
def B : set ℤ := {y | ∃ (m : ℤ), m ∈ A ∧ y = m^2}

-- Prove that A ∩ B = {0, 1, 4}
theorem proof_intersection : A ∩ B = {0, 1, 4} :=
by
  sorry

end proof_intersection_l627_627154


namespace perfect_square_proof_l627_627344

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem perfect_square_proof :
  isPerfectSquare (factorial 22 * factorial 23 * factorial 24 / 12) :=
sorry

end perfect_square_proof_l627_627344


namespace sphere_radius_from_cube_surface_area_l627_627484

-- Given S : Surface area of an inscribed cube
def radius_of_sphere (S : ℝ) : ℝ :=
  1 / 4 * real.sqrt(2 * S)

-- Theorem to prove that the radius of the sphere is equal to 1/4 * sqrt(2S) given the surface area S of the inscribed cube
theorem sphere_radius_from_cube_surface_area
  (S : ℝ) (hS : S > 0) :
  radius_of_sphere S = (1 / 4) * real.sqrt(2 * S) :=
by
  -- proof omitted
  sorry

end sphere_radius_from_cube_surface_area_l627_627484


namespace linear_function_incorrect_conclusion_C_l627_627873

theorem linear_function_incorrect_conclusion_C :
  ∀ (x y : ℝ), (y = -2 * x + 4) → ¬(∃ x, y = 0 ∧ (x = 0 ∧ y = 4)) := by
  sorry

end linear_function_incorrect_conclusion_C_l627_627873


namespace plane_distance_against_wind_l627_627799

theorem plane_distance_against_wind :
  ∀ (D : ℕ),
  (
    let speed_with_wind := 200 in 
    let speed_against_wind := 160 in
    let distance_with_wind := 400 in
    (distance_with_wind / speed_with_wind = D / speed_against_wind)
  ) →
  D = 320 :=
by
  intros D h
  sorry

end plane_distance_against_wind_l627_627799


namespace find_phi_and_range_l627_627144

theorem find_phi_and_range :
  ∀ (f g : ℝ → ℝ) (x : ℝ) (ϕ : ℝ),
  (0 < ϕ ∧ ϕ < π) ∧
  (∀ x, f x = 1/2 * Real.cos (2 * x - ϕ)) ∧
  (f (π/6) = 1/2) ∧
  (g = λ x, f (2 * x)) →
  (ϕ = π / 3) ∧
  (∀ x, g x = 1/2 * Real.cos (4 * x - π / 3)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ π / 4 → -1/4 ≤ g x ∧ g x ≤ 1/2) :=
by
  intros f g x ϕ
  intro H
  cases H with H_ϕ H_f
  cases H_f with H_ϕ_bounds H_f_def
  cases H_f_def with H_f_pass H_g_def
  split
  { sorry },  -- Proof of ϕ = π / 3
  split
  { sorry },  -- Proof that g x = 1 / 2 * cos (4 * x - π / 3)
  { sorry }   -- Proof of the range [-1/4, 1/2] for x in [0, π / 4]

end find_phi_and_range_l627_627144


namespace Roy_school_days_l627_627646

theorem Roy_school_days (hours_per_day : ℕ) (days_missed : ℕ) (total_hours : ℕ) (h1 : hours_per_day = 2) (h2 : days_missed = 2) (h3 : total_hours = 6) : 
  (total_hours / hours_per_day) + days_missed = 5 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end Roy_school_days_l627_627646


namespace chord_bisected_at_P_midpoints_trajectory_with_slope_2_midpoints_trajectory_with_tangent_through_A_midpoints_trajectory_product_of_slopes_l627_627906

-- Definition of ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Proof Problem 1: Equation of the line containing the chord bisected at point P
theorem chord_bisected_at_P :
  ∀ x y : ℝ, ∃ a b c : ℝ, ellipse x y → a * x + b * y = c ∧ a = 2 ∧ b = 4 ∧ c = 3 := by
    sorry

-- Proof Problem 2: Trajectory equation of the midpoints of the chords with slope 2
theorem midpoints_trajectory_with_slope_2 :
  ∀ x y : ℝ, ellipse x y → -real.sqrt 2 < x ∧ x < real.sqrt 2  → x + 4 * y = 0 := by
    sorry

-- Proof Problem 3: Trajectory equation of the midpoints of the chords intercepted by tangent passing through A(2, 1)
theorem midpoints_trajectory_with_tangent_through_A :
  ∀ x y : ℝ, ellipse x y → x^2 - 2*x + 2*y^2 - 2*y = 0 := by
    sorry

-- Proof Problem 4: Trajectory equation of midpoints of PQ
theorem midpoints_trajectory_product_of_slopes :
  ∀ x y : ℝ, ellipse x y → x^2 + 2*y^2 = 1 := by
    sorry

end chord_bisected_at_P_midpoints_trajectory_with_slope_2_midpoints_trajectory_with_tangent_through_A_midpoints_trajectory_product_of_slopes_l627_627906


namespace remove_chairs_l627_627371

theorem remove_chairs (chairs_per_row total_chairs attendance : ℕ) 
  (h_chairs_per_row : chairs_per_row = 15) 
  (h_total_chairs : total_chairs = 300) 
  (h_attendance : attendance = 180) :
  ∃ chairs_to_remove, chairs_to_remove = 105 :=
by 
  use 105
  have h1: (total_chairs / chairs_per_row : ℕ) = 20, 
  {
    rw [h_chairs_per_row, h_total_chairs],
    norm_num,
  },
  have h2: ((attendance + chairs_per_row - 1) / chairs_per_row : ℕ) = 13, 
  {
    rw h_chairs_per_row,
    norm_num,
  },
  have h3: (13 * chairs_per_row : ℕ) = 195,
  {
    rw h_chairs_per_row,
    norm_num,
  },
  show total_chairs - 195 = 105,
  {
    rw h_total_chairs,
    norm_num,
  }

end remove_chairs_l627_627371


namespace n_pow_m_l627_627175

theorem n_pow_m (
  m n : ℤ 
  (h : ∀ x : ℂ, x^2 - 3 * x + m = (x - 1) * (x + n))
) : n^m = 4 := by
  sorry

end n_pow_m_l627_627175


namespace measure_exactly_85_liters_l627_627036

section MilkMeasurement

-- Given conditions
variable (container_of_milk : Type)
variable [HasZero container_of_milk]
variable (milk_bottle : container_of_milk → ℕ)
variable (one_liter : container_of_milk)

-- Noncomputable assumptions, since we cannot compute the exact operations
noncomputable def measure_milk (b1 b2 b3 : ℕ) : Prop :=
  let initial_milk := 1 in
  let steps := [1, 1, 2, 4, 8] in
  let second_measurement := initial_milk + (steps.sum) in
  let repeated_measurement := [second_measurement, second_measurement, 2 * second_measurement] in
  let total_milk := second_measurement + (repeated_measurement.sum) in
  total_milk = 85

-- The main theorem assuming we have three bottles with one having 1 liter of milk
theorem measure_exactly_85_liters :
  ∃ (b1 b2 b3 : ℕ), measure_milk b1 b2 b3 :=
sorry

end MilkMeasurement

end measure_exactly_85_liters_l627_627036


namespace line_AC_eq_l627_627922

/-
Given:
1. A parabola defined by y^2 = 2 * p * x,
2. Axis of symmetry x = -1,
3. Focus F,
4. Points A, B, C on the parabola such that the vectors FA, FB, FC form an arithmetic sequence,
5. Point B is below the x-axis,
6. FA + FB + FC = 0,

Prove: The equation of line AC is 2x - y - 1 = 0.
-/

noncomputable def parabola (p : ℝ) : ℝ → ℝ → Prop := λ x y, y^2 = 2 * p * x

def axis_of_symmetry : ℝ := -1

def focus (p : ℝ) : ℝ × ℝ := (p / 2 + 1, 0)

def is_arithmetic_sequence (v1 v2 v3 : ℝ) : Prop := v1 + v3 = 2 * v2

def vectors_sum_zero (FA FB FC : ℝ × ℝ) : Prop := FA.1 + FB.1 + FC.1 = 0 ∧ FA.2 + FB.2 + FC.2 = 0

-- Problem statement in Lean 4
theorem line_AC_eq {p : ℝ}
    (h1 : parabola p x1 y1)
    (h2 : parabola p x2 y2)
    (h3 : parabola p x3 y3)
    (h4 : axis_of_symmetry = -1)
    (h5 : focus p = F)
    (h6 : is_arithmetic_sequence (dist F A) (dist F B) (dist F C))
    (h7 : y2 < 0)
    (h8 : vectors_sum_zero (FA) (FB) (FC)) :
  ∃ m b, (m = 2 ∧ b = -1) ∧ ∀ x y, (y = m * x + b) ↔ (2 * x - y - 1 = 0) :=
sorry

end line_AC_eq_l627_627922


namespace exists_three_distinct_integers_in_A_l627_627453

noncomputable def A (m n : ℤ) : Set ℤ := { x^2 + m * x + n | x : ℤ }

theorem exists_three_distinct_integers_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a = b * c :=
by
  sorry

end exists_three_distinct_integers_in_A_l627_627453


namespace problem_statement_l627_627308
noncomputable def a (n : ℕ) : ℝ := 2 ^ (n - 1)

def a1 := 1
def q := 2
def S3 := 7

def condition_arithmetic_seq (a1 a2 a3 : ℝ) : Prop :=
  (a1 + 3, 3 * a2, a3 + 4).is_arithmetic_sequence

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem problem_statement :
  (is_geometric_sequence a q) →
  (S3 = 7) →
  condition_arithmetic_seq a1 (a 2) (a 3) →
  ∀ n, a n = 2 ^ (n - 1) ∧ (∑ k in finset.range n, (1 / ((log 2 (a k) + 1) * (log 2 (a (k+1)) + 1)))) = n / (n + 1)
:= by sorry

end problem_statement_l627_627308


namespace tension_max_is_zero_l627_627034

noncomputable def tension_max_angle 
    (m : ℝ) (L : ℝ) (g : ℝ) (θ₀ : ℝ) (T₀ : ℝ) 
    (h : 0 < θ₀ ∧ θ₀ < real.pi / 2) : ℝ :=
    0

theorem tension_max_is_zero 
    (m : ℝ) (L : ℝ) (g : ℝ) (θ₀ : ℝ) (T₀ : ℝ) 
    (h : 0 < θ₀ ∧ θ₀ < real.pi / 2) :
    tension_max_angle m L g θ₀ T₀ h = 0 := 
sorry

end tension_max_is_zero_l627_627034


namespace points_lie_on_circle_l627_627202

-- Definitions for the points and circles described
variables {A B C H A0 B0 C0 A1 A2 B1 B2 C1 C2 : Point}
variables {ωa ωb ωc : Circle}

-- Conditions provided in the problem
def acute_angled_triangle (A B C H : Point) : Prop := 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  internal_angle A B C < π/2 ∧ internal_angle B C A < π/2 ∧ internal_angle C A B < π/2

def orthocenter (A B C H : Point) : Prop :=
  ∃ (ha hb hc : Point), ha ∈ perpendicular_lines_through A ∧ 
                        ha ∈ line B C ∧ hb ∈ perpendicular_lines_through B ∧ 
                        hb ∈ line A C ∧ hc ∈ perpendicular_lines_through C ∧ 
                        hc ∈ line A B ∧ 
                        H = intersection_point ha hb hc

def midpoint (P1 P2 M : Point) : Prop :=
  M = (P1 + P2) / 2

def circle_passing_through (P1 P2 P3 : Point) (ω : Circle) : Prop :=
  ∀ (P : Point), P ∈ ω ↔ dist P1 ω.radius ∧ dist P2 ω.radius ∧ dist P3 ω.radius

def circle_intersects (ω : Circle) (L : Line) (P1 P2 : Point) : Prop :=
  ∃ (I1 I2 : Point), I1 ∈ ω ∧ I2 ∈ ω ∧ I1 ∈ L ∧ I2 ∈ L ∧ I1 ≠ I2 ∧ (P1 = I1 ∨ P1 = I2) ∧ (P2 = I1 ∨ P2 = I2)

-- Theorem statement
theorem points_lie_on_circle (ABC_type: acute_angled_triangle A B C H)
  (H_orth: orthocenter A B C H) (A0_mid: midpoint B C A0)
  (B0_mid: midpoint C A B0) (C0_mid: midpoint A B C0)
  (ωa_def: circle_passing_through H A0 ωa)
  (ωb_def: circle_passing_through H B0 ωb)
  (ωc_def: circle_passing_through H C0 ωc)
  (A1A2_inter: circle_intersects ωa (line B C) A1 A2)
  (B1B2_inter: circle_intersects ωb (line A C) B1 B2)
  (C1C2_inter: circle_intersects ωc (line A B) C1 C2) : 
  ∃ (O : Point) (R : ℝ), ∀ (P : Point), (P = A1 ∨ P = A2 ∨ P = B1 ∨ P = B2 ∨ P = C1 ∨ P = C2) → dist O P = R :=
sorry

end points_lie_on_circle_l627_627202


namespace part_one_part_two_l627_627110

noncomputable def complex_number (m : ℝ) : ℂ :=
  complex.mk (m^2 + 5 * m + 6) (m^2 - 2 * m - 15) 

theorem part_one (m : ℝ) : 
  (complex_number m = (2 : ℂ) - (12 : ℂ) * complex.i) -> 
  m = -1 :=
by
  sorry

theorem part_two (m : ℝ) : 
  (complex_number m).re = 0 ∧ (complex_number m).im ≠ 0 -> 
  m = -2 :=
by
  sorry

end part_one_part_two_l627_627110


namespace unique_triangles_count_l627_627956

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627956


namespace non_congruent_triangles_with_perimeter_18_l627_627976

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627976


namespace area_increase_l627_627300

open Nat

def perimeter (l w : ℕ) : ℕ := 2 * (l + w)
def area (l w : ℕ) : ℕ := l * w

theorem area_increase (l w l' w' : ℕ) :
  perimeter l w = 40 →
  area l w ≤ 40 →
  l + w = 20 →
  perimeter l' w' = 44 →
  l' + w' = 22 →
  ∃ ΔA, ΔA ∈ {2, 4, 21, 36, 38} ∧ ΔA = area l' w' - area l w :=
begin
  sorry
end

end area_increase_l627_627300


namespace k_inv_h_l627_627402

variable {α β : Type}
variable (h : α → β)
variable (k : β → α)
variable (h_inv : β → α)
variable (k_inv : α → β)

-- Hypotheses as conditions
hypothesis (H1 : ∀ x : α, h_inv (k x) = 3 * x - 1)

-- Question to prove
theorem k_inv_h : k_inv (h 4) = 5 / 3 :=
by
  -- Proof will be filled in here
  sorry

end k_inv_h_l627_627402


namespace find_value_l627_627158

variable (x y z : ℕ)

-- Condition: x / 4 = y / 3 = z / 2
def ratio_condition := x / 4 = y / 3 ∧ y / 3 = z / 2

-- Theorem: Given the ratio condition, prove that (x - y + 3z) / x = 7 / 4.
theorem find_value (h : ratio_condition x y z) : (x - y + 3 * z) / x = 7 / 4 := 
  by sorry

end find_value_l627_627158


namespace combination_20_choose_3_eq_1140_l627_627206

theorem combination_20_choose_3_eq_1140 :
  (Nat.choose 20 3) = 1140 := 
by sorry

end combination_20_choose_3_eq_1140_l627_627206


namespace fraction_operation_l627_627819

theorem fraction_operation : (3 / 5 - 1 / 10 + 2 / 15 = 19 / 30) :=
by
  sorry

end fraction_operation_l627_627819


namespace half_angle_of_cone_half_angle_for_s_075_l627_627805

-- Definition for the specific gravity s and the properties associated with it.
def specific_gravity (s : ℝ) : Prop := (0 < s) ∧ (s < 1)

-- Main theorem stating the half-angle of the cone.
theorem half_angle_of_cone (s : ℝ) (h : specific_gravity s) : 
  ∃ (x : ℝ), (cos x = (sqrt (1 + 8 * s) - 1) / 2) ∧ (0 < x) ∧ (x < π / 2) := 
sorry

-- Corollary for a specific s value.
theorem half_angle_for_s_075 : 
  ∃ (x : ℝ), (cos x = (sqrt (1 + 8 * 0.75) - 1) / 2) ∧ (0 < x) ∧ (x < π / 2) :=
sorry

end half_angle_of_cone_half_angle_for_s_075_l627_627805


namespace minimum_value_of_a_l627_627151

noncomputable def inequality_valid_for_all_x (a : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → x + a * Real.log x - x^a + 1 / Real.exp x ≥ 0

theorem minimum_value_of_a : ∃ a, inequality_valid_for_all_x a ∧ a = -Real.exp 1 := sorry

end minimum_value_of_a_l627_627151


namespace gcd_of_180_and_450_l627_627082

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l627_627082


namespace excircles_on_circumsphere_implies_isogonal_isogonal_implies_excircles_on_circumsphere_l627_627009

theorem excircles_on_circumsphere_implies_isogonal {T : Tetrahedron} (h : ∀ I, I ∈ excircle_centers T → I ∈ circumsphere T) : isogonal T := sorry

theorem isogonal_implies_excircles_on_circumsphere {T : Tetrahedron} (h : isogonal T) : ∀ I, I ∈ excircle_centers T → I ∈ circumsphere T := sorry

end excircles_on_circumsphere_implies_isogonal_isogonal_implies_excircles_on_circumsphere_l627_627009


namespace higher_concentration_acid_solution_l627_627020

theorem higher_concentration_acid_solution (x : ℝ) (h1 : 2 * (8 / 100 : ℝ) = 1.2 * (x / 100) + 0.8 * (5 / 100)) : x = 10 :=
sorry

end higher_concentration_acid_solution_l627_627020


namespace playground_area_l627_627676

theorem playground_area (B : ℕ) (L : ℕ) (playground_area : ℕ) 
  (h1 : L = 8 * B) 
  (h2 : L = 240) 
  (h3 : playground_area = (1 / 6) * (L * B)) : 
  playground_area = 1200 :=
by
  sorry

end playground_area_l627_627676


namespace numerator_denominator_added_l627_627341

theorem numerator_denominator_added (n : ℕ) : (3 + n) / (5 + n) = 9 / 11 → n = 6 :=
by
  sorry

end numerator_denominator_added_l627_627341


namespace primes_sum_product_condition_l627_627156

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem primes_sum_product_condition (m n p : ℕ) (hm : is_prime m) (hn : is_prime n) (hp : is_prime p)  
  (h : m * n * p = 5 * (m + n + p)) : 
  m^2 + n^2 + p^2 = 78 :=
sorry

end primes_sum_product_condition_l627_627156


namespace find_a_b_l627_627596

-- Defining the sequence {a_n}
def sequence (a b : ℤ) : ℕ → ℤ
| 0     := 0
| 1     := a
| 2     := b
| (n+2) := (sequence a b (n+1) + sequence a b n) / 2

-- Defining the summation of the sequence up to n
def S (a b : ℤ) (n : ℕ) : ℤ := ∑ k in Finset.range n, sequence a b k

theorem find_a_b (a b : ℤ) (hS : ∀ n, S a b (n+1) = 4) : a = 6 ∧ b = -3 :=
sorry

end find_a_b_l627_627596


namespace shaded_region_area_l627_627444

/-
  Given:
    The rectangular plot with vertices:
      A = (0,0)
      B = (40,0)
      C = (40,20)
      D = (0,20)
    The shaded polygon with vertices:
      P1 = (0,0)
      P2 = (20,0)
      P3 = (40,10)
      P4 = (40,20)
      P5 = (10,20)
  Prove:
    The area of the shaded region is 600 square units.
-/

def area_shaded_region (A B C D P1 P2 P3 P4 P5 : (ℝ × ℝ)) : ℝ :=
  let trapezoid_area := 1 / 2 * (20 + 30) * 20
  let triangle_area := 1 / 2 * 10 * 20
  trapezoid_area + triangle_area

theorem shaded_region_area :
  let A := (0,0)
  let B := (40,0)
  let C := (40,20)
  let D := (0,20)
  let P1 := (0,0)
  let P2 := (20,0)
  let P3 := (40,10)
  let P4 := (40,20)
  let P5 := (10,20)
  area_shaded_region A B C D P1 P2 P3 P4 P5 = 600 :=
by 
  sorry

end shaded_region_area_l627_627444


namespace last_digit_of_7_power_7_power_7_l627_627337

theorem last_digit_of_7_power_7_power_7 : (7 ^ (7 ^ 7)) % 10 = 3 :=
by
  sorry

end last_digit_of_7_power_7_power_7_l627_627337


namespace rectangle_area_is_16_l627_627600

theorem rectangle_area_is_16
  (A B C D M N : Type)
  (s : ℝ) -- s represents the side length
  (h1 : AM = MC)
  (h2 : AM = 1)
  (h3 : MB = 2)
  (h4 : angle (A M C) = 45) :
  area_of_rectangle s = 16 := by
sorry

end rectangle_area_is_16_l627_627600


namespace sum_mod_13_l627_627736

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l627_627736


namespace points_on_line_l627_627763

-- Step 1: Definitions based on the conditions
def coords (n : ℕ) : ℝ × ℝ
| 0     := (1, -1)
| (n+1) := let (a_n, b_n) := coords n in (a_n * b_n, _)

def line_eq (x y : ℝ) : Prop := 2 * x + y = 1

-- Step 2: Proof statement
theorem points_on_line : 
  ∀ n : ℕ, n > 0 → line_eq (fst (coords n)) (snd (coords n)) :=
by
  sorry

end points_on_line_l627_627763


namespace smallest_angle_cos_identity_l627_627098

theorem smallest_angle_cos_identity :
  ∃ θ : ℝ, (0 < θ ∧ θ ≤ 90) ∧ 
    cos θ = cos (15 * real.pi / 180) - sin (20 * real.pi / 180) + 
            cos (65 * real.pi / 180) + sin (85 * real.pi / 180) ∧ 
    θ = 60 * real.pi / 180 :=
by
  sorry

end smallest_angle_cos_identity_l627_627098


namespace college_period_length_l627_627747

theorem college_period_length :
  ∀ (start_time end_time : Nat) (total_periods : Nat) (break_duration : Nat),
  start_time = 600 → end_time = 1000 → total_periods = 5 → break_duration = 5 →
  let total_minutes := (end_time - start_time)
  let break_time := (total_periods - 1) * break_duration
  let teaching_time := total_minutes - break_time
  teaching_time / total_periods = 40 :=
by
  intros start_time end_time total_periods break_duration
  intros h_start h_end h_periods h_break
  let total_minutes := (end_time - start_time)
  let break_time := (total_periods - 1) * break_duration
  let teaching_time := total_minutes - break_time
  calc
    teaching_time / total_periods = sorry

end college_period_length_l627_627747


namespace meals_neither_kosher_vegan_l627_627611

theorem meals_neither_kosher_vegan : 
  ∀ (total vegan kosher both : ℕ), 
  total = 30 → vegan = 7 → kosher = 8 → both = 3 → 
  (total - (vegan + kosher - both)) = 18 :=
by
  intros total vegan kosher both h_total h_vegan h_kosher h_both
  rw [h_total, h_vegan, h_kosher, h_both]
  rfl

end meals_neither_kosher_vegan_l627_627611


namespace arithmetic_sequence_sum_l627_627469

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 3 = 7) (h2 : a 5 + a 7 = 26) :
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((2 * n + 1)^2 - 1)) ∧
  (∀ n, T n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_sum_l627_627469


namespace find_n_l627_627892

theorem find_n (n : ℤ) (h1 : n > 4)
  (h2 : ∀ (x y : ℂ), (2 * n * (complex.sqrt y) - n) = (-1)^(n - 3) * 2 * n * (n - 1) * (n - 2) * y) :
  n = 6 :=
by sorry

end find_n_l627_627892


namespace find_x_l627_627465

def diamond_op (a b : ℝ) : ℝ := a / b

theorem find_x (x : ℝ) (h : 504 / (8 / x) = 36) : x = 4 / 7 :=
by
  sorry

end find_x_l627_627465


namespace sum_mod_13_l627_627735

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l627_627735


namespace find_second_dimension_of_smaller_box_l627_627199

def volume_large_box : ℕ := 12 * 14 * 16
def volume_small_box (x : ℕ) : ℕ := 3 * x * 2
def max_small_boxes : ℕ := 64

theorem find_second_dimension_of_smaller_box (x : ℕ) : volume_large_box = max_small_boxes * volume_small_box x → x = 7 :=
by
  intros h
  unfold volume_large_box at h
  unfold volume_small_box at h
  sorry

end find_second_dimension_of_smaller_box_l627_627199


namespace probability_divisor_of_8_is_half_l627_627781

theorem probability_divisor_of_8_is_half :
  let outcomes := (1 : ℕ) :: (2 : ℕ) :: (3 : ℕ) :: (4 : ℕ) :: (5 : ℕ) :: (6 : ℕ) :: (7 : ℕ) :: (8 : ℕ) :: []
  let divisors_of_8 := [ 1, 2, 4, 8 ]
  let favorable_outcomes := list.filter (λ x, x ∣ 8) outcomes
  let favorable_probability := (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)
  favorable_probability = (1 / 2 : ℚ) := by
  sorry

end probability_divisor_of_8_is_half_l627_627781


namespace cookies_left_correct_l627_627608

def cookies_left (cookies_per_dozen : ℕ) (flour_per_dozen_lb : ℕ) (bag_count : ℕ) (flour_per_bag_lb : ℕ) (cookies_eaten : ℕ) : ℕ :=
  let total_flour_lb := bag_count * flour_per_bag_lb
  let total_cookies := (total_flour_lb / flour_per_dozen_lb) * cookies_per_dozen
  total_cookies - cookies_eaten

theorem cookies_left_correct :
  cookies_left 12 2 4 5 15 = 105 :=
by sorry

end cookies_left_correct_l627_627608


namespace least_number_of_shoes_l627_627250

theorem least_number_of_shoes (num_inhabitants : ℕ) 
  (one_legged_percentage : ℚ) 
  (barefooted_proportion : ℚ) 
  (h_num_inhabitants : num_inhabitants = 10000) 
  (h_one_legged_percentage : one_legged_percentage = 0.05) 
  (h_barefooted_proportion : barefooted_proportion = 0.5) : 
  ∃ (shoes_needed : ℕ), shoes_needed = 10000 := 
by
  sorry

end least_number_of_shoes_l627_627250


namespace probability_sum_multiple_of_3_l627_627321

theorem probability_sum_multiple_of_3 (n : ℕ) :
  let total_outcomes := 6^3 in
  let valid_outcomes := ∑ i in finset.range 6, ∑ j in finset.range 6, ∑ k in finset.range 6, 
    if ((i+1 + j+1 + k+1) % 3 = 0) then 1 else 0 in
  (valid_outcomes.to_real / total_outcomes.to_real = 1 / 8) :=
by
  sorry

end probability_sum_multiple_of_3_l627_627321


namespace non_congruent_triangles_with_perimeter_18_l627_627982

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627982


namespace probability_of_rolling_divisor_of_8_l627_627775

open_locale classical

-- Predicate: a number n is a divisor of 8
def is_divisor_of_8 (n : ℕ) : Prop := n ∣ 8

-- The total number of outcomes when rolling an 8-sided die
def total_outcomes : ℕ := 8

-- The probability of rolling a divisor of 8 on a fair 8-sided die
theorem probability_of_rolling_divisor_of_8 (is_fair_die : true) :
  (| {n | is_divisor_of_8 n} ∩ {1, 2, 3, 4, 5, 6, 7, 8} | : ℕ) / total_outcomes = 1 / 2 :=
by
  sorry

end probability_of_rolling_divisor_of_8_l627_627775


namespace min_func_value_l627_627440

theorem min_func_value (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
  ( √2 * Real.sin x - 3 * Real.tan y )^2 + ( √2 * Real.cos x - 3 * Real.cot y )^2 = 8 :=
sorry

end min_func_value_l627_627440


namespace simplify_sqrt_expression_l627_627405

theorem simplify_sqrt_expression :
  2 * Real.sqrt 12 - Real.sqrt 27 - (Real.sqrt 3 * Real.sqrt (1 / 9)) = (2 * Real.sqrt 3) / 3 := 
by
  sorry

end simplify_sqrt_expression_l627_627405


namespace option_D_is_correct_l627_627741

variable (a b : ℝ)

theorem option_D_is_correct :
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^2 + 3 * a ≠ 4 * a^2) ∧
  ((a + 2) * (a - 2) ≠ a^2 - 2) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end option_D_is_correct_l627_627741


namespace six_times_product_plus_one_equals_seven_pow_sixteen_l627_627818

theorem six_times_product_plus_one_equals_seven_pow_sixteen :
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := 
  sorry

end six_times_product_plus_one_equals_seven_pow_sixteen_l627_627818


namespace range_of_a_for_monotonicity_l627_627145

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * Real.log x + x

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a_for_monotonicity :
  (∀ a : ℝ, is_monotonically_increasing_on (λ x, (1 / 2) * x^2 - a * Real.log x + x) (Set.Ici 1) ↔ a ≤ 2) :=
by simp; sorry

end range_of_a_for_monotonicity_l627_627145


namespace find_k_multiple_l627_627351

theorem find_k_multiple (a b k : ℕ) (h1 : a = b + 5) (h2 : a + b = 13) 
  (h3 : 3 * (a + 7) = k * (b + 7)) : k = 4 := sorry

end find_k_multiple_l627_627351


namespace find_perpendicular_line_l627_627079

-- Define the point P
structure Point where
  x : ℤ
  y : ℤ

-- Define the line
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- The given problem conditions
def P : Point := { x := -1, y := 3 }

def given_line : Line := { a := 1, b := -2, c := 3 }

def perpendicular_line (line : Line) (point : Point) : Line :=
  ⟨ -line.b, line.a, -(line.a * point.y - line.b * point.x) ⟩

-- Theorem statement to prove
theorem find_perpendicular_line :
  perpendicular_line given_line P = { a := 2, b := 1, c := -1 } :=
by
  sorry

end find_perpendicular_line_l627_627079


namespace parabola_unique_intersection_x_axis_l627_627921

theorem parabola_unique_intersection_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ ∀ y, y^2 - 6*y + m = 0 → y = x) → m = 9 :=
by
  sorry

end parabola_unique_intersection_x_axis_l627_627921


namespace remainder_of_product_mod_17_l627_627724

theorem remainder_of_product_mod_17 :
  (5007 * 5008 * 5009 * 5010 * 5011) % 17 = 0 := 
by
  -- Apply the modulus to the known products
  have h1 : 5007 % 17 = 15 := by norm_num
  have h2 : 5008 % 17 = 16 := by norm_num
  have h3 : 5009 % 17 = 0 := by norm_num
  have h4 : 5010 % 17 = 1 := by norm_num
  have h5 : 5011 % 17 = 2 := by norm_num

  -- Use the modulus values to simplify the product
  have prod : (5007 * 5008 * 5009 * 5010 * 5011) % 17 =
              (15 * 16 * 0 * 1 * 2) % 17 := by
    congr,
    any_goals { assumption }

  -- Calculate the simplified product
  have prod_mod : (15 * 16 * 0 * 1 * 2) % 17 = 0 := by norm_num

  -- Conclude the proof
  rw [prod_mod]
  exact prod_mod

end remainder_of_product_mod_17_l627_627724


namespace polynomial_evaluation_l627_627842

theorem polynomial_evaluation (x : ℝ) (h₁ : 0 < x) (h₂ : x^2 - 2 * x - 15 = 0) :
  x^3 - 2 * x^2 - 8 * x + 16 = 51 :=
sorry

end polynomial_evaluation_l627_627842


namespace student_percentage_first_subject_l627_627392

theorem student_percentage_first_subject
  (P : ℝ)
  (h1 : (P + 60 + 70) / 3 = 60) : P = 50 :=
  sorry

end student_percentage_first_subject_l627_627392


namespace non_congruent_triangles_with_perimeter_18_l627_627987

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627987


namespace projection_of_a_on_b_l627_627505

variables (a : ℝ × ℝ) (b : ℝ × ℝ)
variables (dot_product_ab : a.1 * b.1 + a.2 * b.2 = 10)
variables (b_value : b = (6, -8))

theorem projection_of_a_on_b (h1 : a.1 * b.1 + a.2 * b.2 = 10) (h2 : b = (6, -8)) :
  (a.1 * b.1 + a.2 * b.2) / (b.1 * b.1 + b.2 * b.2) * (b.1, b.2) = (3/5, -4/5) :=
by sorry

end projection_of_a_on_b_l627_627505


namespace first_term_eq_six_l627_627312

-- Define the conditions of the problem
noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) := 
  ∀ n, a (n + 1) = a n * r

variables {a : ℕ → ℝ} {r : ℝ}

-- Condition 1: The sum of the first four terms is 240
def sum_first_four_terms := a 0 * (1 - (r ^ 4)) / (1 - r) = 240

-- Condition 2: The sum of the second and fourth terms is 180
def sum_second_and_fourth_terms := a 1 * r + a 1 * (r ^ 3) = 180

-- Question: Prove that the first term of the sequence is 6 given the conditions
theorem first_term_eq_six (h1 : geometric_sequence a r) 
    (h2 : sum_first_four_terms) 
    (h3 : sum_second_and_fourth_terms) :
  a 0 = 6 :=
sorry

end first_term_eq_six_l627_627312


namespace num_integers_sq_condition_l627_627869

theorem num_integers_sq_condition : 
  {n : ℤ | n < 30 ∧ (∃ k : ℤ, k ^ 2 = n / (30 - n))}.to_finset.card = 3 := 
by
  sorry

end num_integers_sq_condition_l627_627869


namespace probability_divisor_of_8_is_half_l627_627779

theorem probability_divisor_of_8_is_half :
  let outcomes := (1 : ℕ) :: (2 : ℕ) :: (3 : ℕ) :: (4 : ℕ) :: (5 : ℕ) :: (6 : ℕ) :: (7 : ℕ) :: (8 : ℕ) :: []
  let divisors_of_8 := [ 1, 2, 4, 8 ]
  let favorable_outcomes := list.filter (λ x, x ∣ 8) outcomes
  let favorable_probability := (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)
  favorable_probability = (1 / 2 : ℚ) := by
  sorry

end probability_divisor_of_8_is_half_l627_627779


namespace problem_equiv_proof_l627_627526

theorem problem_equiv_proof :
  (∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ), 
  (∀ x : ℝ, (x - 3)^2 * (x + 1)^8 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^10) 
  → log 2 (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10}) = 10) :=
sorry

end problem_equiv_proof_l627_627526


namespace time_to_cross_platform_l627_627765

def train_length : ℝ := 300
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 550.0000000000001

theorem time_to_cross_platform :
  let v := train_length / time_to_cross_pole in
  let D := train_length + platform_length in
  let T := D / v in
  T = 51 :=
by
  sorry

end time_to_cross_platform_l627_627765


namespace number_of_bijections_l627_627161

theorem number_of_bijections (n : ℕ) (h₀ : 1 ≤ n):
  ∃ (f : fin n → fin n), (∀ i : fin n, 1 ≤ i → ∃ j : fin n, j < i ∧ (f i = f j + 1 ∨ f i + 1 = f j)) ∧
  fintype.card {f : fin n → fin n // (∀ i : fin n, 1 ≤ i → ∃ j : fin n, j < i ∧ (f i = f j + 1 ∨ f i + 1 = f j))} = 2^(n-1) :=
by sorry

end number_of_bijections_l627_627161


namespace imaginary_part_is_correct_l627_627674

def complex := ℂ

def num : complex := 5 + complex.I
def denom : complex := 2 - complex.I
def quotient : complex := num / denom

theorem imaginary_part_is_correct : (quotient).im = 7 / 5 := by sorry

end imaginary_part_is_correct_l627_627674


namespace vacation_cost_division_l627_627697

theorem vacation_cost_division (n : ℕ) (total_cost : ℕ) 
  (cost_difference : ℕ)
  (cost_per_person_5 : ℕ) :
  total_cost = 1000 → 
  cost_difference = 50 → 
  cost_per_person_5 = total_cost / 5 →
  (total_cost / n) = cost_per_person_5 + cost_difference → 
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end vacation_cost_division_l627_627697


namespace alex_should_buy_l627_627399

noncomputable def num_jellybeans (n : ℕ) : ℕ :=
  if n >= 150 ∧ n % 17 = 15 then n else 0

theorem alex_should_buy : ∃ n, num_jellybeans n = 151 :=
by
  use 151
  have h1 : 151 >= 150 := by norm_num
  have h2 : 151 % 17 = 15 := by norm_num
  simp [num_jellybeans, h1, h2]
  sorry

end alex_should_buy_l627_627399


namespace trigonometric_identity_l627_627828

theorem trigonometric_identity :
  (Real.cos (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (48 * Real.pi / 180) * Real.sin(18 * Real.pi / 180)) = 1 / 2 :=
by
  -- proof goes here
  sorry

end trigonometric_identity_l627_627828


namespace dave_apps_left_l627_627413

theorem dave_apps_left (initial_apps deleted_apps remaining_apps : ℕ)
  (h_initial : initial_apps = 23)
  (h_deleted : deleted_apps = 18)
  (h_calculation : remaining_apps = initial_apps - deleted_apps) :
  remaining_apps = 5 := 
by 
  sorry

end dave_apps_left_l627_627413


namespace hyperbola_distance_focus_asymptote_l627_627498

theorem hyperbola_distance_focus_asymptote (a b c : ℝ) (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_eccentricity : c / a = 2) (h_real_axis : 2 * a = 2) : 
  let distance_focus_to_asymptote := (|b * c| / (Real.sqrt ((a ^ 2) + (b ^ 2)))) in
  distance_focus_to_asymptote = Real.sqrt 3 :=
by
  sorry

end hyperbola_distance_focus_asymptote_l627_627498


namespace part1_part2_l627_627115

-- Conditions: Definitions of A and B
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem statements
theorem part1 (a b : ℝ) :  2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := sorry

theorem part2 (a : ℝ) (h : ∀ a, 2 * A a 2 - B a 2 = - 4 * a * 2 + 6 * 2 + 8 * a) : 2 = 2 := sorry

end part1_part2_l627_627115


namespace probability_of_rolling_divisor_of_8_l627_627777

open_locale classical

-- Predicate: a number n is a divisor of 8
def is_divisor_of_8 (n : ℕ) : Prop := n ∣ 8

-- The total number of outcomes when rolling an 8-sided die
def total_outcomes : ℕ := 8

-- The probability of rolling a divisor of 8 on a fair 8-sided die
theorem probability_of_rolling_divisor_of_8 (is_fair_die : true) :
  (| {n | is_divisor_of_8 n} ∩ {1, 2, 3, 4, 5, 6, 7, 8} | : ℕ) / total_outcomes = 1 / 2 :=
by
  sorry

end probability_of_rolling_divisor_of_8_l627_627777


namespace irreducible_fraction_l627_627229

theorem irreducible_fraction (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := 
sorry

end irreducible_fraction_l627_627229


namespace find_other_endpoint_l627_627679

theorem find_other_endpoint (mx my x₁ y₁ x₂ y₂ : ℤ) 
  (h1 : mx = (x₁ + x₂) / 2) 
  (h2 : my = (y₁ + y₂) / 2) 
  (h3 : mx = 3) 
  (h4 : my = 4) 
  (h5 : x₁ = -2) 
  (h6 : y₁ = -5) : 
  x₂ = 8 ∧ y₂ = 13 := 
by
  sorry

end find_other_endpoint_l627_627679


namespace train_cross_pole_time_l627_627394

noncomputable def time_to_cross_pole : ℝ :=
  let speed_km_hr := 60
  let speed_m_s := speed_km_hr * 1000 / 3600
  let length_of_train := 50
  length_of_train / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole = 3 := 
by
  sorry

end train_cross_pole_time_l627_627394


namespace number_of_students_solving_only_B_l627_627403

-- Define the different groups of students solving specific problems
variables {I II III IV V : ℕ}

-- Total number of students
def student_total : ℕ := 25

-- Number of students solving only B (desired result)
def students_solving_only_B : ℕ := 6

-- Define the conditions
def conditions :=
  ∃ (I II III IV V : ℕ),
  I + II + III + IV + V = student_total ∧
  (III + IV + V) = 2 * (IV + V) ∧
  I = II + 1 ∧
  I = III + IV

-- The theorem to prove
theorem number_of_students_solving_only_B : 
  conditions → III = students_solving_only_B :=
begin
  sorry
end

end number_of_students_solving_only_B_l627_627403


namespace monotonic_intervals_and_extreme_values_l627_627916

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem monotonic_intervals_and_extreme_values :
  (∀ x ∈ Ioi 1, deriv f x > 0) ∧
  (∀ x ∈ Ioo 0 1, deriv f x < 0) ∧
  (f 1 = -1) :=
by
  sorry

end monotonic_intervals_and_extreme_values_l627_627916


namespace find_expression_roots_l627_627477

-- Define the roots of the given quadratic equation
def is_root (α : ℝ) : Prop := α ^ 2 - 2 * α - 1 = 0

-- Define the main statement to be proven
theorem find_expression_roots (α β : ℝ) (hα : is_root α) (hβ : is_root β) :
  5 * α ^ 4 + 12 * β ^ 3 = 169 := sorry

end find_expression_roots_l627_627477


namespace correct_proposition_3_l627_627810

noncomputable def even_function_symmetric_about_y_axis : Prop :=
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f x = f (-x)) → (∀ x : ℝ, f x = f (-x))

theorem correct_proposition_3 :
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f x = f (-x)) ↔ even_function_symmetric_about_y_axis) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f x = f (-x)) ∧ ¬(∃ y : ℝ, f y = 0)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f (-x) = -f(x)) ∧ ¬(f 0 = 0)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f x = 0 → (x ∈ ℝ → f x = f (-x) ∧ f x = -f x))) → 
  ③ :=
by sorry

end correct_proposition_3_l627_627810


namespace inverse_of_42_mod_43_inverse_of_42_mod_59_l627_627851

theorem inverse_of_42_mod_43 : ∃ x : ℤ, (42 * x) % 43 = 1 :=
by {
  use 42,
  have : 42 * 42 = 42^2,
  ring,
  simp [this],
  exact Mod.natMod_coe_eq_self_of_lt (42*42) 43 (by norm_num),
  norm_num
}

theorem inverse_of_42_mod_59 : ∃ x : ℤ, (42 * x) % 59 = 1 :=
by {
  use 52,
  exact Mod.natMod_coe_eq_self_of_lt (42 * 52) 59 (by norm_num),
  norm_num
}

end inverse_of_42_mod_43_inverse_of_42_mod_59_l627_627851


namespace least_n_for_cubic_sum_l627_627438

theorem least_n_for_cubic_sum (n : ℕ) :
  (∃ (x : Fin n → ℤ), (∑ i in Finset.univ, (x i)^3 = 2002^2002)) → n = 4 :=
by
  sorry

end least_n_for_cubic_sum_l627_627438


namespace num_int_solutions_l627_627837

theorem num_int_solutions (x : ℤ) : 
  (x^4 - 39 * x^2 + 140 < 0) ↔ (x = 3 ∨ x = -3 ∨ x = 4 ∨ x = -4 ∨ x = 5 ∨ x = -5) := 
sorry

end num_int_solutions_l627_627837


namespace sum_of_remainders_eq_11_mod_13_l627_627732

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l627_627732


namespace range_of_d_l627_627104

-- Definitions from conditions
def line_eq (λ : ℝ) (x y : ℝ) : Prop :=
  (2 + λ) * x - (1 + λ) * y - 2 * (3 + 2 * λ) = 0

def point_P : ℝ × ℝ := (-2, 2)

-- The theorem stating the range of the distance 'd'
theorem range_of_d (λ : ℝ) : 0 < dist {p : ℝ × ℝ | line_eq λ p.1 p.2} point_P ∧
                              dist {p : ℝ × ℝ | line_eq λ p.1 p.2} point_P < 4 :=
sorry

end range_of_d_l627_627104


namespace rainfall_wednesday_l627_627404

theorem rainfall_wednesday (m t R w : ℝ) (hm : m = 0.17) (ht : t = 0.42) (hR : R = 0.67) (hw : w = R - (m + t)) :
  w = 0.08 :=
by 
  rw [hm, ht, hR, hw]
  sorry

end rainfall_wednesday_l627_627404


namespace abs_neg_seventeen_l627_627820

theorem abs_neg_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end abs_neg_seventeen_l627_627820


namespace sum_of_remainders_eq_11_mod_13_l627_627734

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l627_627734


namespace fraction_conversion_integer_l627_627411

theorem fraction_conversion_integer (x : ℝ) :
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1 →
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 :=
by sorry

end fraction_conversion_integer_l627_627411


namespace count_non_congruent_triangles_with_perimeter_18_l627_627938

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627938


namespace cone_volume_is_3pi_l627_627121

-- Define the cone with given properties
structure Cone where
  (radius_circumscribed : ℝ)  -- Radius of the circumscribed sphere
  (volume : ℝ)  -- Volume of the cone

-- Define the properties of the specific cone in the problem
def cone_problem := Cone.mk 2 (3 * Real.pi)

-- The theorem stating the volume of cone with given properties is 3π
theorem cone_volume_is_3pi (c : Cone) (h1 : c.radius_circumscribed = 2) : c.volume = 3 * Real.pi :=
  sorry

end cone_volume_is_3pi_l627_627121


namespace incenter_lies_on_MN_l627_627327

noncomputable theory

-- Definitions of triangle ABC and its geometrical properties
variables {A B C K M N : Point}

-- Conditions given in the problem
def are_tangent_circles (circle1 circle2 : Circle) (point : Point) : Prop := sorry

def is_incenter (point : Point) (triangle : Triangle) : Prop := sorry

def lies_on_line (point : Point) (line : Line) : Prop := sorry

axiom circle_cond_1 (circle1 circle2 : Circle) : are_tangent_circles circle1 (circumcircle A B C) K ∧
    are_tangent_circles circle2 (circumcircle A B C) K

axiom circle_cond_2 (circle1 : Circle) : are_tangent_circles circle1 AB M

axiom circle_cond_2 (circle2 : Circle) : are_tangent_circles circle2 AC N

theorem incenter_lies_on_MN (triangle : Triangle) :
  (circle_cond_1 c1 c2) →
  (circle_cond_2 c1) →
  (circle_cond_3 c2) →
  is_incenter K triangle → 
  lies_on_line K (line_through M N) :=
sorry

end incenter_lies_on_MN_l627_627327


namespace marvelous_class_student_count_l627_627620

theorem marvelous_class_student_count (g : ℕ) (jb : ℕ) (jg : ℕ) (j_total : ℕ) (jl : ℕ) (init_jb : ℕ) : 
  jb = g + 3 →  -- Number of boys
  jg = 2 * g + 1 →  -- Jelly beans received by each girl
  init_jb = 726 →  -- Initial jelly beans
  jl = 4 →  -- Leftover jelly beans
  j_total = init_jb - jl →  -- Jelly beans distributed
  (jb * jb + g * jg = j_total) → -- Total jelly beans distributed equation
  2 * g + 1 + g + jb = 31 := -- Total number of students
by
  sorry

end marvelous_class_student_count_l627_627620


namespace shoe_trick_l627_627623

variable (s : ℕ) (y : ℕ)

theorem shoe_trick (h1 : 10 ≤ s ∧ s < 100) : 
  let n := 100 * s + (1990 - y) in (n / 100 = s ∧ n % 100 = 1990 - y) :=
by sorry

end shoe_trick_l627_627623


namespace tony_removed_no_10p_coins_l627_627705

theorem tony_removed_no_10p_coins :
  ∃ (n X : ℕ) (p_1 p_5 p_10 p_20 : ℕ),
    X = 13 * n ∧                                   
    p_1 + p_5 + p_10 + p_20 = n ∧              
    (X - 1) = 14 * (n - 1) ∧                     
    1 * p_1 + 5 * p_5 + 10 * p_10 + 20 * p_20 = X ∧
    p_10 = 0 :=
begin
  sorry
end

end tony_removed_no_10p_coins_l627_627705


namespace smallest_num_of_subsets_l627_627591

def f (n : ℕ) : ℕ :=
let k := Nat.find (λ k, 2^(k-1) ≤ n ∧ n < 2^k) in k

theorem smallest_num_of_subsets (n : ℕ) :
  ∃ X : Finset (Fin n), 
    (∀ a b : Fin n, a ≠ b → ∃ S ∈ X.powerset, (a ∈ S ∧ b ∉ S) ∨ (a ∉ S ∧ b ∈ S)) ∧
      X.card = Nat.find (λ k, 2^(k-1) ≤ n ∧ n < 2^k) :=
sorry

end smallest_num_of_subsets_l627_627591


namespace count_zeros_between_decimal_point_and_first_nonzero_l627_627792

def fraction : ℚ := 7 / 8000

theorem count_zeros_between_decimal_point_and_first_nonzero :
  let decimal_repr := fraction.toReal in
  let zeros := by sorry in -- Here, we calculate the number of zeros through decimal representation and conversion
  zeros = 3 :=
by
  sorry

end count_zeros_between_decimal_point_and_first_nonzero_l627_627792


namespace range_of_a_l627_627533

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ set.Iic (1 : ℝ), f x = 2^x - a^2 - a ∧ f x = 0) : a ∈ set.Ioo 0 1 :=
sorry

end range_of_a_l627_627533


namespace n_pow_m_eq_4_l627_627174

theorem n_pow_m_eq_4 (m n : ℤ) (h1 : ∀ x : ℤ, x^2 - 3 * x + m = (x - 1) * (x + n)) : n^m = 4 :=
by
  sorry

end n_pow_m_eq_4_l627_627174


namespace minimum_distance_is_sqrt2_l627_627886

def vec_sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
( a.1 - b.1, a.2 - b.2, a.3 - b.3 )

def norm_sq (v : ℝ × ℝ × ℝ) : ℝ :=
v.1^2 + v.2^2 + v.3^2

noncomputable def minimum_value_distance (t : ℝ) : ℝ :=
let a := (1 - t, 2 * t - 1, 0)
let b := (2, t, t)
let diff := vec_sub b a
in real.sqrt (norm_sq diff)

theorem minimum_distance_is_sqrt2 : (∃ t : ℝ, minimum_value_distance t = real.sqrt 2) :=
sorry

end minimum_distance_is_sqrt2_l627_627886


namespace probability_divisible_by_Q_l627_627220

noncomputable def lim_X_n : ℝ := 10015 / 20736

theorem probability_divisible_by_Q (n : ℕ) (S : finset ℕ) (P : ℕ → ℝ) : 
  (|S| = 6 ) → 
  (∀ i ∈ S, i ≤ n) → 
  (P(S) = ∑ i in S, x^i) → 
  (∀ Q : polynomial ℝ, degree Q ≤ 3 ∧ Q.eval 0 ≠ 0 → ∃ x : ℝ, Q.eval x = 0) → 
  filter.tendsto (λ n, probability (P(S) ∣ Q) (bernoulli (S ∈ finset.range (n + 1) \choose 6))) filter.at_top (nhds lim_X_n) :=
begin
  sorry
end

end probability_divisible_by_Q_l627_627220


namespace fencing_rate_3_rs_per_meter_l627_627660

noncomputable def rate_per_meter (A_hectares : ℝ) (total_cost : ℝ) : ℝ := 
  let A_m2 := A_hectares * 10000
  let r := Real.sqrt (A_m2 / Real.pi)
  let C := 2 * Real.pi * r
  total_cost / C

theorem fencing_rate_3_rs_per_meter : rate_per_meter 17.56 4456.44 = 3.00 :=
by 
  sorry

end fencing_rate_3_rs_per_meter_l627_627660


namespace tadpoles_percentage_let_go_l627_627326

-- Define the conditions
def total_tadpoles : ℕ := 180
def kept_tadpoles : ℕ := 45

-- Define the number let go
def let_go_tadpoles : ℕ := total_tadpoles - kept_tadpoles

-- Define the goal in terms of percentage
def percentage(let_go total : ℕ) : ℝ := (let_go.to_real / total.to_real) * 100

theorem tadpoles_percentage_let_go : percentage let_go_tadpoles total_tadpoles = 75 := 
by
    sorry

end tadpoles_percentage_let_go_l627_627326


namespace fx_plus_f1x_eq_one_sum_fractions_eq_1011_l627_627913

def f (x : ℝ) := (4^x) / (2 + 4^x)

theorem fx_plus_f1x_eq_one (x : ℝ) : f(x) + f(1 - x) = 1 := 
by sorry

theorem sum_fractions_eq_1011 : 
  ( ∑ k in finset.range(2022), f((k + 1) / 2023) ) = 1011 :=
by sorry

end fx_plus_f1x_eq_one_sum_fractions_eq_1011_l627_627913


namespace donation_problem_l627_627542

theorem donation_problem
  (A B C D : Prop)
  (h1 : ¬A ↔ (B ∨ C ∨ D))
  (h2 : B ↔ D)
  (h3 : C ↔ ¬B) 
  (h4 : D ↔ ¬B): A := 
by
  sorry

end donation_problem_l627_627542


namespace complex_unit_circle_sum_l627_627816

theorem complex_unit_circle_sum :
  let z1 := (1 + Complex.I * Real.sqrt 3) / 2
  let z2 := (1 - Complex.I * Real.sqrt 3) / 2
  (z1 ^ 8 + z2 ^ 8 = -1) :=
by
  sorry

end complex_unit_circle_sum_l627_627816


namespace probability_divisor_of_8_on_8_sided_die_l627_627783

def divisor_probability : ℚ :=
  let sample_space := {1, 2, 3, 4, 5, 6, 7, 8}
  let divisors_of_8 := {1, 2, 4, 8}
  let favorable_outcomes := divisors_of_8 ∩ sample_space
  favorable_outcomes.card / sample_space.card

theorem probability_divisor_of_8_on_8_sided_die :
  divisor_probability = 1 / 2 :=
sorry

end probability_divisor_of_8_on_8_sided_die_l627_627783


namespace max_distance_in_grid_l627_627551

noncomputable def maxS := 50 * Real.sqrt 2

theorem max_distance_in_grid : 
  ∀ (grid : Array (Array ℕ)) (h : grid.size = 100) (h' : ∀ i, grid[i].size = 100)
  (H1 : ∀ i j, (1 ≤ i ∧ i ≤ 99 ∧ 1 ≤ j ∧ j ≤ 100 → grid[i][j] + 1 = grid[i + 1][j])
   ∧ (1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 99 → grid[i][j] + 1 = grid[i][j + 1]))
  (H5000 : ∀ (x y i j : ℕ), (grid[x][y] = grid[i][j] + 5000) →
   Real.sqrt ((i - x)^2 + (j - y)^2) ≥ maxS),
  S = maxS := sorry

end max_distance_in_grid_l627_627551


namespace function_satisfies_conditions_l627_627480

def f (x : ℝ) : ℝ := cos (π / 2 * x)

theorem function_satisfies_conditions :
  (∀ x : ℝ, f (x) = f (x + 4)) ∧ 
  (∀ x : ℝ, x = 2 ↔ f (x) = f (4 - x)) :=
sorry

end function_satisfies_conditions_l627_627480


namespace pencils_bought_l627_627426

theorem pencils_bought (cindi_spent : ℕ) (cost_per_pencil : ℕ) 
  (cindi_pencils : ℕ) 
  (marcia_pencils : ℕ) 
  (donna_pencils : ℕ) :
  cindi_spent = 30 → 
  cost_per_pencil = 1/2 → 
  cindi_pencils = cindi_spent / cost_per_pencil → 
  marcia_pencils = 2 * cindi_pencils → 
  donna_pencils = 3 * marcia_pencils → 
  donna_pencils + marcia_pencils = 480 := 
by
  sorry

end pencils_bought_l627_627426


namespace integer_fraction_condition_l627_627525

theorem integer_fraction_condition (p : ℕ) (h_pos : 0 < p) :
  (∃ k : ℤ, k > 0 ∧ (5 * p + 15) = k * (3 * p - 9)) ↔ (4 ≤ p ∧ p ≤ 19) :=
by
  sorry

end integer_fraction_condition_l627_627525


namespace complementary_event_is_all_red_l627_627455

/--
Given:
1. A bag containing 3 red balls and 2 white balls.
2. Drawing 3 balls at random.
3. Event A defined as "at least one of the three drawn balls is white."

Prove:
The complementary event to A, A', is the event "all three drawn balls are red."
-/
theorem complementary_event_is_all_red :
  ∀ (draw : ℕ → Fin 5),
  (at_least_one_white draw) → (complementary_event draw) = all_red draw :=
sorry

def bag := [red_ball, red_ball, red_ball, white_ball, white_ball]

def draw (x : Fin 3) : Fin 5 := sorry -- Represents drawing a ball randomly

def at_least_one_white (draw : ℕ → Fin 5) : Prop :=
  ∃ i < 3, draw i ∈ {white_ball}

def complementary_event (draw : ℕ → Fin 5) : Prop :=
  ¬ at_least_one_white draw

def all_red (draw : ℕ → Fin 5) : Prop :=
  ∀ i < 3, draw i ∈ {red_ball}

end complementary_event_is_all_red_l627_627455


namespace max_profit_l627_627398

noncomputable def initial_cost : ℝ := 10
noncomputable def cost_per_pot : ℝ := 0.0027
noncomputable def total_cost (x : ℝ) : ℝ := initial_cost + cost_per_pot * x

noncomputable def P (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 5.7 * x + 19
else 108 - 1000 / (3 * x)

noncomputable def r (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 3 * x + 9
else 98 - 1000 / (3 * x) - 27 * x / 10

theorem max_profit (x : ℝ) : r 10 = 39 :=
sorry

end max_profit_l627_627398


namespace find_a1_l627_627471

theorem find_a1 (a b : ℕ → ℝ) (h1 : ∀ n ≥ 1, a (n + 1) + b (n + 1) = (a n + b n) / 2) 
  (h2 : ∀ n ≥ 1, a (n + 1) * b (n + 1) = (a n * b n) ^ (1/2)) 
  (hb2016 : b 2016 = 1) (ha1_pos : a 1 > 0) :
  a 1 = 2^2015 :=
sorry

end find_a1_l627_627471


namespace symmetric_function_l627_627470

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def symmetric_about_axis (f : ℤ → ℤ) (axis : ℤ) : Prop :=
  ∀ x : ℤ, f (axis - x) = f (axis + x)

theorem symmetric_function (a : ℕ → ℤ) (d : ℤ) (f : ℤ → ℤ) (a1 a2 : ℤ) (axis : ℤ) :
  (∀ x, f x = |x - a1| + |x - a2|) →
  arithmetic_sequence a d →
  d ≠ 0 →
  axis = (a1 + a2) / 2 →
  symmetric_about_axis f axis :=
by
  -- Proof goes here
  sorry

end symmetric_function_l627_627470


namespace find_coordinates_of_B_l627_627880

noncomputable theory

def coordinates_of_A : ℝ × ℝ := (1, -2)
def length_of_AB : ℝ := 8

def is_parallel_to_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1

theorem find_coordinates_of_B
  (B : ℝ × ℝ)
  (h1 : is_parallel_to_y_axis coordinates_of_A B)
  (h2 : Real.dist coordinates_of_A B = length_of_AB) 
  : B = (1, 6) ∨ B = (1, -10) :=
sorry

end find_coordinates_of_B_l627_627880


namespace meals_neither_kosher_nor_vegan_l627_627614

theorem meals_neither_kosher_nor_vegan : (total_clients vegan kosher both : ℕ)
    (total_clients = 30) (vegan = 7) (kosher = 8) (both = 3) :
    total_clients - (vegan + kosher - both) = 18 :=
by
  sorry

end meals_neither_kosher_nor_vegan_l627_627614


namespace walter_coins_value_l627_627331

theorem walter_coins_value :
  let pennies : ℕ := 2
  let nickels : ℕ := 2
  let dimes : ℕ := 1
  let quarters : ℕ := 1
  let half_dollars : ℕ := 1
  let penny_value : ℕ := 1
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let half_dollar_value : ℕ := 50
  (pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value) = 97 := 
sorry

end walter_coins_value_l627_627331


namespace gcd_180_450_l627_627091

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l627_627091


namespace function_root_c_range_l627_627177

theorem function_root_c_range (c : ℝ) (f : ℝ → ℝ) (h : ∃ x : ℝ, f x = 0) :
  (∀ x : ℝ, f x = 2^(-|x|) + c) → c ∈ set.Ico (-1 : ℝ) 0 :=
by
  intro h1
  cases h with x hx
  have h2 : 2^(-|x|) ∈ set.Ioc 0 1 := sorry
  have h3 : f x = 2^(-|x|) + c := h1 x
  rw [hx] at h3
  have h4 : c = -2^(-|x|) := sorry
  have h5 : c > -1 := sorry
  have h6 : c < 0 := sorry
  exact ⟨h5, h6⟩

end function_root_c_range_l627_627177


namespace main_problem_l627_627493

-- Defining the function f(x) given a
def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

-- Tangent condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, (1 / x₀ - a = 1) ∧ (x₀ - 1 - log 2 = log x₀ - a * x₀)

-- Inequality condition
def holds_inequality (a : ℝ) : Prop :=
  ∀ x > 0, (x + 1) * (log x - a * x) ≤ log x - (x / exp 1)

-- Main theorem combining both parts
theorem main_problem :
  (∃ a : ℝ, is_tangent a ∧ holds_inequality a) ↔
  ∃ a : ℝ, a = 1 ∧ a ∈ (set.Ici (1 / exp 1)) :=
by sorry

end main_problem_l627_627493


namespace tangent_points_on_curve_l627_627692

theorem tangent_points_on_curve
    (P : ℝ × ℝ)
    (curve_eq : ∀ x : ℝ, P.2 = x^3 + x - 2)
    (tangent_parallel_to : ∀ m : ℝ, m = 4 → tangent_line_slope (x^3 + x - 2) P.1 = m) :
    (P = (1, 0) ∨ P = (-1, -4)) :=
by
    sorry

end tangent_points_on_curve_l627_627692


namespace fill_pipe_fraction_l627_627375

theorem fill_pipe_fraction (t : ℕ) (frac : ℚ) (ht : t = 25) (hfrac : frac = 1 / 25) : frac = 1 / t := 
by {
  rw ht,
  exact hfrac,
}

end fill_pipe_fraction_l627_627375


namespace power_first_digits_l627_627641

theorem power_first_digits (n : ℕ) (h1 : ∀ k : ℕ, n ≠ 10^k) : ∃ j k : ℕ, 1973 ≤ n^j / 10^k ∧ n^j / 10^k < 1974 := by
  sorry

end power_first_digits_l627_627641


namespace inequality_problem_l627_627233

open Real

theorem inequality_problem
  (a b c x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h_condition : 1 / x + 1 / y + 1 / z = 1) :
  a^x + b^y + c^z ≥ 4 * a * b * c * x * y * z / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_problem_l627_627233


namespace christina_total_drive_time_l627_627824

-- Define the conditions as given in the problem.
variables (D : ℕ) (Df : ℕ) (sf : ℕ) (t_f : ℕ) (s1 : ℕ) (s3 : ℕ)

-- Assume the given conditions.
-- Total journey is 210 miles.
axiom total_distance : D = 210

-- Christina's friend drives 120 miles at 40 mph for 3 hours.
axiom friend_drive : Df = 120 ∧ sf = 40 ∧ t_f = 3

-- Speed limits for the first segment and third segment.
axiom speed_limits : s1 = 30 ∧ s3 = 50

-- The remaining distance is 90 miles split equally into the first and third segments.
axiom remaining_distance : ∃ d_rest, D - Df = d_rest ∧ d_rest = 90

-- Prove the total driving time for Christina is 144 minutes.
theorem christina_total_drive_time : s1 = 30 → s3 = 50 → D = 210 → Df = 120 → (D - Df = 90 ∧ s1 = 30 ∧ s3 = 50) → 
  90 / 30 * 60 + 90 / 50 * 60 = 144 :=
begin
  intros hs1 hs3 hD hDf hRest,
  rw [← hRest.left],
  have h1 : (45 / 30) * 60 = 90, by norm_num,
  have h2 : (45 / 50) * 60 = 54, by norm_num,
  calc 90 + 54 = 144 : by norm_num,
end

end christina_total_drive_time_l627_627824


namespace lcm_factor_l627_627291

-- Define the variables and conditions
variables (A B H L x : ℕ)
variable (hcf_23 : Nat.gcd A B = 23)
variable (larger_number_391 : A = 391)
variable (lcm_hcf_mult_factors : L = Nat.lcm A B)
variable (lcm_factors : L = 23 * x * 17)

-- The proof statement
theorem lcm_factor (hcf_23 : Nat.gcd A B = 23) (larger_number_391 : A = 391) (lcm_hcf_mult_factors : L = Nat.lcm A B) (lcm_factors : L = 23 * x * 17) :
  x = 17 :=
sorry

end lcm_factor_l627_627291


namespace store_total_profit_l627_627028

theorem store_total_profit :
  let turtlenecks_initial_cost := 30
  let crewnecks_initial_cost := 25
  let vnecks_initial_cost := 20
  let turtlenecks_quantity := 100
  let crewnecks_quantity := 150
  let vnecks_quantity := 200

  let turtlenecks_initial_markup := turtlenecks_initial_cost * 1.2
  let crewnecks_initial_markup := crewnecks_initial_cost * 1.35
  let vnecks_initial_markup := vnecks_initial_cost * 1.25

  let turtlenecks_new_year_markup := turtlenecks_initial_markup + 0.25 * turtlenecks_initial_cost
  let crewnecks_new_year_markup := crewnecks_initial_markup + 0.15 * crewnecks_initial_cost
  let vnecks_new_year_markup := vnecks_initial_markup + 0.20 * vnecks_initial_cost

  let turtlenecks_final_price := turtlenecks_new_year_markup * 0.91
  let crewnecks_final_price := crewnecks_new_year_markup * 0.88
  let vnecks_final_price := vnecks_new_year_markup * 0.85

  let turtlenecks_profit_per_item := turtlenecks_final_price - turtlenecks_initial_cost
  let crewnecks_profit_per_item := crewnecks_final_price - crewnecks_initial_cost
  let vnecks_profit_per_item := vnecks_final_price - vnecks_initial_cost

  let turtlenecks_total_profit := turtlenecks_profit_per_item * turtlenecks_quantity
  let crewnecks_total_profit := crewnecks_profit_per_item * crewnecks_quantity
  let vnecks_total_profit := vnecks_profit_per_item * vnecks_quantity

  let total_profit := turtlenecks_total_profit + crewnecks_total_profit + vnecks_total_profit
  total_profit = 3088.50 :=

begin
  sorry
end

end store_total_profit_l627_627028


namespace rearrangement_inequality_example_l627_627597

theorem rearrangement_inequality_example (n : ℕ) (a : Fin n → ℕ) 
  (h : set.range a = {1, 2, ..., n}) : 
  ∑ i in Finset.range (n - 1), (i + 1) / (i + 2) ≤ 
  ∑ i in Finset.range (n - 1), a i / a (i + 1) := 
by
  sorry

end rearrangement_inequality_example_l627_627597


namespace sin_of_alpha_plus_pi_over_3_l627_627116

theorem sin_of_alpha_plus_pi_over_3
  (α : ℝ)
  (h1 : cos (α - π / 6) + sin α = (4 * sqrt 3) / 5)
  (h2 : α ∈ Ioo (π / 2) π) :
  sin (α + π / 3) = (4 * sqrt 3 - 3) / 10 :=
sorry

end sin_of_alpha_plus_pi_over_3_l627_627116


namespace meals_neither_kosher_vegan_l627_627612

theorem meals_neither_kosher_vegan : 
  ∀ (total vegan kosher both : ℕ), 
  total = 30 → vegan = 7 → kosher = 8 → both = 3 → 
  (total - (vegan + kosher - both)) = 18 :=
by
  intros total vegan kosher both h_total h_vegan h_kosher h_both
  rw [h_total, h_vegan, h_kosher, h_both]
  rfl

end meals_neither_kosher_vegan_l627_627612


namespace Lavinia_daughter_age_difference_l627_627219

-- Define the ages of the individuals involved
variables (Ld Ls Kd : ℕ)

-- Conditions given in the problem
variables (H1 : Kd = 12)
variables (H2 : Ls = 2 * Kd)
variables (H3 : Ls = Ld + 22)

-- Statement we need to prove
theorem Lavinia_daughter_age_difference(Ld Ls Kd : ℕ) (H1 : Kd = 12) (H2 : Ls = 2 * Kd) (H3 : Ls = Ld + 22) : 
  Kd - Ld = 10 :=
sorry

end Lavinia_daughter_age_difference_l627_627219


namespace f_is_monotonic_l627_627888

variable (f : ℝ → ℝ)

theorem f_is_monotonic (h : ∀ a b x : ℝ, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  (∀ x y : ℝ, x ≤ y → f x <= f y) ∨ (∀ x y : ℝ, x ≤ y → f x >= f y) :=
sorry

end f_is_monotonic_l627_627888


namespace iron_needed_for_hydrogen_l627_627443

-- Conditions of the problem
def reaction (Fe H₂SO₄ FeSO₄ H₂ : ℕ) : Prop :=
  Fe + H₂SO₄ = FeSO₄ + H₂

-- Given data
def balanced_equation : Prop :=
  reaction 1 1 1 1
 
def produced_hydrogen : ℕ := 2
def produced_from_sulfuric_acid : ℕ := 2
def needed_iron : ℕ := 2

-- Problem statement to be proved
theorem iron_needed_for_hydrogen (H₂SO₄ H₂ : ℕ) (h1 : produced_hydrogen = H₂) (h2 : produced_from_sulfuric_acid = H₂SO₄) (balanced_eq : balanced_equation) :
  needed_iron = 2 := by
sorry

end iron_needed_for_hydrogen_l627_627443


namespace triangle_area_l627_627713

def line1 (x : ℝ) : ℝ := (1 / 3) * x + 2
def line2 (x : ℝ) : ℝ := 3 * x - 6
def line3 (x y : ℝ) : Prop := x + y = 12

theorem triangle_area : 
  let A := (3, 3) in
  let B := (4.5, 7.5) in
  let C := (7.5, 4.5) in
  let area := (1 / 2 : ℝ) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  area = 8.625 :=
by
  sorry

end triangle_area_l627_627713


namespace log_b_2023_is_11_l627_627415

noncomputable def clubsuit (a b : ℝ) : ℝ := a^(Real.logBase 5 b)
noncomputable def spadesuit (a b : ℝ) : ℝ := a^(1 / (Real.logBase 5 b))

noncomputable def b : ℕ → ℝ
| 4           := spadesuit 4 2
| (n + 1) := clubsuit (spadesuit n (n - 2)) (b n)

theorem log_b_2023_is_11 : Real.logBase 5 (b 2023) = 11 :=
  sorry

end log_b_2023_is_11_l627_627415


namespace non_congruent_triangles_with_perimeter_18_l627_627999

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l627_627999


namespace non_congruent_triangles_with_perimeter_18_l627_627993

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627993


namespace circles_are_tangent_l627_627417

theorem circles_are_tangent :
  ∀ (x y : ℝ), (x + y^2 - 4 = 0 ∨ x^2 + y^2 + 2x = 0) →
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    (c₁ = (0, 2) ∧ r₁ = 2 ∧ c₂ = (-1, 0) ∧ r₂ = 1) ∧
    ∥c₁ - c₂∥ = r₁ + r₂ :=
begin
  sorry
end

end circles_are_tangent_l627_627417


namespace identify_true_propositions_l627_627487

-- Define the four propositions
def P1 : Prop := ∀ (L1 L2 L3 L4 : Line), intersect(L1, L3) → intersect(L2, L4) → skew(L3, L4) → skew(L1, L2)
def P2 : Prop := ∀ (Π1 Π2 : Plane) (L : Line), perp(L, Π2) → passes_through(Π1, L) → perp(Π1, Π2)
def P3 : Prop := ∀ (L1 L2 L3 : Line), perp(L1, L3) → perp(L2, L3) → parallel(L1, L2)
def P4 : Prop := ∀ (Π1 Π2 : Plane) (L : Line), perp(Π1, Π2) → in_plane(L, Π1) → ¬perp(L, intersection_line(Π1, Π2)) → ¬perp(L, Π2)

-- The equivalent proof problem statement
theorem identify_true_propositions : (P2 ∧ P4) :=
by sorry

end identify_true_propositions_l627_627487


namespace flower_bed_fraction_is_correct_l627_627802

noncomputable def flower_bed_fraction (yard_length yard_height : ℕ) (parallel_side1 parallel_side2 triangle_leg : ℕ) : ℚ :=
  let trapezoid_height := yard_height
  let triangle_area := 0.5 * (triangle_leg ^ 2)
  let total_triangle_area := 2 * triangle_area
  let yard_area := yard_length * yard_height
  total_triangle_area / yard_area

theorem flower_bed_fraction_is_correct :
  flower_bed_fraction 30 6 20 35 7.5 = 5 / 16 :=
by
  sorry

end flower_bed_fraction_is_correct_l627_627802


namespace count_non_congruent_triangles_with_perimeter_18_l627_627935

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627935


namespace right_triangle_area_l627_627535

theorem right_triangle_area (a b c : ℝ) (h : c = 5) (h1 : a = 3) (h2 : c^2 = a^2 + b^2) : 
  1 / 2 * a * b = 6 :=
by
  sorry

end right_triangle_area_l627_627535


namespace edge_length_RS_l627_627688

theorem edge_length_RS {a b c d e f : ℕ} (h : {a, b, c, d, e, f} = {9, 15, 22, 28, 34, 39}) (hPQ : f = 39) : 
  ∃ x ∈ {a, b, c, d, e}, x = 9 := sorry

end edge_length_RS_l627_627688


namespace axis_of_symmetry_and_monotonic_increase_range_of_m_l627_627887

def f (x : ℝ) : ℝ := - (Real.sqrt 2 / 2) * Real.sin (2 * x + Real.pi / 4) + 2

theorem axis_of_symmetry_and_monotonic_increase :
  ∃ k : ℤ, ∀ x : ℝ, (f x = f (x + π / 2)) ∧ (∀ x : ℝ, (π / 8 + k * π ≤ x) ∧ (x ≤ 5 * π / 8 + k * π) → f x ≥ f (π / 8 + k * π)) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) (π / 2), abs (f x - m) ≤ 1) ↔ m ∈ set.Icc (3 / 2) (3 - Real.sqrt 2 / 2) :=
sorry

end axis_of_symmetry_and_monotonic_increase_range_of_m_l627_627887


namespace weight_of_one_bowling_ball_l627_627648

-- Definitions from the problem conditions
variables (b c : ℝ)

-- Conditions as hypotheses
def seven_balls_eq_three_canoes : Prop := 7 * b = 3 * c
def two_canoes_eq_56 : Prop := 2 * c = 56

-- Theorem stating the problem and its solution
theorem weight_of_one_bowling_ball (h1 : seven_balls_eq_three_canoes) (h2 : two_canoes_eq_56) : b = 12 :=
sorry

end weight_of_one_bowling_ball_l627_627648


namespace non_congruent_triangles_with_perimeter_18_l627_627981

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627981


namespace part_A_part_B_part_D_l627_627709

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Part A: single transmission probability
theorem part_A (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 :=
by sorry

-- Part B: triple transmission probability
theorem part_B (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  β * (1 - β)^2 = β * (1 - β)^2 :=
by sorry

-- Part D: comparing single and triple transmission
theorem part_D (α β : ℝ) (hα : 0 < α ∧ α < 0.5) (hβ : 0 < β ∧ β < 1) :
  (1 - α) < (1 - α)^3 + 3 * α * (1 - α)^2 :=
by sorry

end part_A_part_B_part_D_l627_627709


namespace number_of_true_statements_is_one_l627_627488

def statements : List Prop := [
  (π ∈ Set.univ), -- ① \pi ∈ R (Any real number belongs to the universal set of real numbers)
  (↑(3:ℚ) = 0), -- ② \sqrt{3} ∈ Q (Any rational number is equal to its representation as a rational)
  (-3 ∉ Set.Icc (Int.ofNat (-3)) 3), -- ③ -3 ∉ Z (Int.ofNat is the equivalent integer cast in Lean)
  (abs (-3) ∉ Set.univ), -- ④ abs(-3) ∉ N (The absolute value of any integer -3 < 0 < 3)
  (0 ∉ Set.univ) -- ⑤ 0 ∉ Q (any rational number belongs to the universal set)
]

theorem number_of_true_statements_is_one : list.filter id statements = [true] :=
by
  sorry

end number_of_true_statements_is_one_l627_627488


namespace min_value_l627_627363

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2*y = 2) : 
  ∃ c : ℝ, c = 2 ∧ ∀ z, (z = (x^2 / (2*y) + 4*(y^2) / x)) → z ≥ c :=
by
  sorry

end min_value_l627_627363


namespace m1_m2_intersecting_or_skew_l627_627218

-- Definitions of the conditions
def is_skew (L1 L2 : Line) : Prop := 
  ¬∃ p : Point, p ∈ L1 ∧ p ∈ L2

def intersects (m L : Line) : Prop :=
  ∃ p : Point, p ∈ m ∧ p ∈ L
   
-- Our proof objective statement
theorem m1_m2_intersecting_or_skew (L1 L2 m1 m2 : Line) 
  (skew_L1_L2 : is_skew L1 L2)
  (intersect_m1_L1 : intersects m1 L1)
  (intersect_m1_L2 : intersects m1 L2)
  (intersect_m2_L1 : intersects m2 L1)
  (intersect_m2_L2 : intersects m2 L2)
: (∃ p : Point, p ∈ m1 ∧ p ∈ m2) ∨ is_skew m1 m2 :=
sorry

end m1_m2_intersecting_or_skew_l627_627218


namespace candy_problem_l627_627711

theorem candy_problem
  (x y m : ℤ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hxy : x + y = 176)
  (hcond : x - m * (y - 16) = 47)
  (hm : m > 1) :
  x ≥ 131 := 
sorry

end candy_problem_l627_627711


namespace product_divisible_by_perfect_square_l627_627273

theorem product_divisible_by_perfect_square 
  (S : Finset ℕ)
  (h₁ : ∀ x ∈ S, x ∈ Finset.range 11)
  (h₂ : S.card = 6) 
  : ∃ p : ℕ, p > 1 ∧ ∃ k : ℕ, k > 1 ∧ p = k * k ∧ ∃ product : ℕ, ∏ i in S, i % p = 0 :=
by
  sorry

end product_divisible_by_perfect_square_l627_627273


namespace sum_of_number_and_reverse_l627_627285

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l627_627285


namespace count_remarkable_two_digit_numbers_l627_627241

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def remarkable (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ p1 ≠ p2 ∧ n = p1 * p2

def two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

theorem count_remarkable_two_digit_numbers : 
  { n : ℕ | two_digit n ∧ remarkable n }.to_finset.card = 30 := 
by
  sorry

end count_remarkable_two_digit_numbers_l627_627241


namespace correct_option_C_l627_627345

theorem correct_option_C (m n : ℤ) : 
  (4 * m + 1) * 2 * m = 8 * m^2 + 2 * m :=
by
  sorry

end correct_option_C_l627_627345


namespace transformed_parabola_l627_627681

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ :=
  (1 / 4) * x^2

-- Define the transformation functions
def shift_left (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  f (x + a)

def shift_down (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  f x - b

-- Theorem statement
theorem transformed_parabola:
  ∀ x, (shift_down (shift_left original_parabola 2) 3) x = (1 / 4) * (x + 2)^2 - 3 :=
by
  intros x
  unfold original_parabola shift_left shift_down
  rw [add_comm]
  sorry

end transformed_parabola_l627_627681


namespace temp_on_Monday_l627_627281

variable (M T W Th F : ℤ)

-- Given conditions
axiom sum_MTWT : M + T + W + Th = 192
axiom sum_TWTF : T + W + Th + F = 184
axiom temp_F : F = 34
axiom exists_day_temp_42 : ∃ (day : String), 
  (day = "Monday" ∨ day = "Tuesday" ∨ day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday") ∧
  (if day = "Monday" then M else if day = "Tuesday" then T else if day = "Wednesday" then W else if day = "Thursday" then Th else F) = 42

-- Prove temperature of Monday is 42
theorem temp_on_Monday : M = 42 := 
by
  sorry

end temp_on_Monday_l627_627281


namespace tan_sum_identity_l627_627134

theorem tan_sum_identity (α : ℝ) (h₁ : cos α = -4 / 5) (h₂ : α ∈ Ioc (π / 2) π) :
  tan (α + π / 4) = 1 / 7 := 
  sorry

end tan_sum_identity_l627_627134


namespace math_proof_l627_627070

-- Define the problem conditions and given data
noncomputable def problem (A B C D E F G : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G] : Prop :=
  let triangle_ABC := EquilateralTriangle A B C
  let circle_s := Circle A 3
  let line_l1 := ParallelLineThrough D E A
  let line_l2 := ParallelLineThrough E D A
  let F_intersection := Intersection line_l1 line_l2
  let G_on_circle := OnCircle G circle_s A F
  let area_DBG := ∃ (p q r : ℕ), p.gcd r = 1 ∧ q = 3 ∧ area (triangle D B G) = (p * Real.sqrt q) / r in
  -- Given properties and required to show
  triangle_ABC.onCircumcircle circle_s ∧
  ∃ B D E F G, dist A D = 17 ∧ dist A E = 15 ∧ Line B D ∧ Line C E ∧
  Parallel line_l1 line_l2 ∧ line_l1 ≠ line_l2  ∧
  F.intersection line_l1 line_l2 F_intersection ∧
  G_on_circle G_on_circle ∧
  area_DBG := 854 

-- Statement part
theorem math_proof : problem :=
begin
  -- Proof omitted
  sorry
end

end math_proof_l627_627070


namespace perfect_square_trinomial_k_l627_627184

theorem perfect_square_trinomial_k (k : ℤ) : 
  (∀ x : ℝ, x^2 - k*x + 64 = (x + 8)^2 ∨ x^2 - k*x + 64 = (x - 8)^2) → 
  (k = 16 ∨ k = -16) :=
by
  sorry

end perfect_square_trinomial_k_l627_627184


namespace sequence_sum_l627_627924

-- Definition of the sequence satisfying 2a_{n+1} + a_n = 0
def seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, 2 * a (n + 1) + a n = 0

-- Given initial condition
def init_cond (a : ℕ → ℝ) : Prop := a 2 = 1

-- Sum of the first 10 terms of the sequence
def S10 (a : ℕ → ℝ) : ℝ := ∑ i in Finset.range 10, a i

theorem sequence_sum :
  ∀ (a : ℕ → ℝ), seq a → init_cond a → S10 a = 4 / 3 * (2^(-10) - 1) :=
by
  intros
  sorry

end sequence_sum_l627_627924


namespace mark_initial_money_l627_627603

theorem mark_initial_money (X : ℝ) 
  (h1 : X = (1/2) * X + 14 + (1/3) * X + 16) : X = 180 := 
  by
  sorry

end mark_initial_money_l627_627603


namespace non_congruent_triangles_count_l627_627961

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627961


namespace root_of_polynomial_l627_627306

theorem root_of_polynomial :
  ∀ x : ℝ, (x^2 - 3 * x + 2) * x * (x - 4) = 0 ↔ (x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 4) :=
by 
  sorry

end root_of_polynomial_l627_627306


namespace remainder_9_plus_y_mod_31_l627_627235

theorem remainder_9_plus_y_mod_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (9 + y) % 31 = 18 :=
sorry

end remainder_9_plus_y_mod_31_l627_627235


namespace domain_of_function_l627_627668

theorem domain_of_function :
  ∀ x : ℝ, ((2 - x ≥ 0) ∧ (x - 1 > 0)) ↔ (1 < x ∧ x ≤ 2) :=
by
  intros x
  split
  { intro h,
    cases h with h1 h2,
    split,
    { linarith },
    { linarith } },
  { intro h,
    cases h with h1 h2,
    split,
    { linarith },
    { linarith } }
  sorry

end domain_of_function_l627_627668


namespace max_k_exists_l627_627169

noncomputable def max_possible_k (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) : ℝ :=
sorry

theorem max_k_exists (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) :
  ∃ k_max : ℝ, k_max = max_possible_k x y k h_pos h_eq :=
sorry

end max_k_exists_l627_627169


namespace groups_of_three_from_five_l627_627561

theorem groups_of_three_from_five : 
  let number_of_people := 5
  let group_size := 3
  choose number_of_people group_size = 10 :=
by
  sorry

end groups_of_three_from_five_l627_627561


namespace parabola_focus_through_point_l627_627666

theorem parabola_focus_through_point (a : ℝ) (x y : ℝ) (h_eq : y = a * x^2) (h_point : (x, y) = (1, 1)) : 
    focus (parabola_eq := h_eq) = (0, 1 / 4) :=
by
  sorry


end parabola_focus_through_point_l627_627666


namespace func_equiv_l627_627894

noncomputable def f (x : ℝ) : ℝ := if x = 0 then 0 else x + 1 / x

theorem func_equiv {a b : ℝ} (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) :
  (∀ x, f (2 * x) = a * f x + b * x) ∧ (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y)) :=
sorry

end func_equiv_l627_627894


namespace hyperbola_eccentricity_correct_l627_627919

noncomputable def hyperbola_asymptotic_line (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  (b / a = 4 / 3)

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := real.sqrt (a^2 + b^2)
  in c / a

theorem hyperbola_eccentricity_correct
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_asymptote: hyperbola_asymptotic_line a b ha hb) :
  hyperbola_eccentricity a b ha hb = 5 / 3 :=
sorry

end hyperbola_eccentricity_correct_l627_627919


namespace area_of_three_layers_l627_627757

theorem area_of_three_layers
  (total_area_runners : ℕ)
  (table_area : ℕ)
  (total_covered_percentage : ℕ)
  (two_layers_area : ℕ)
  (h1 : total_area_runners = 224)
  (h2 : table_area = 175)
  (h3 : total_covered_percentage = 80)
  (h4 : two_layers_area = 24) :
  let covered_area := (total_covered_percentage * table_area) / 100 in
  let one_layer_area := covered_area - 2 * two_layers_area in
  ∃ (three_layers_area : ℕ), total_area_runners = one_layer_area + 2 * two_layers_area + 3 * three_layers_area ∧ three_layers_area = 12 :=
by sorry

end area_of_three_layers_l627_627757


namespace gcd_180_450_l627_627093

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l627_627093


namespace proof_problem_l627_627286

-- Definitions of the conditions
def domain_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, true

def symmetric_graph_pt (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = 2 * b - f (a + x)

def symmetric (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = -f (x)

def symmetric_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (2*a - x) = f (x)

-- Definitions of the statements to prove
def statement_1 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y = f (x - 1) → y = f (1 - x) → x = 1)

def statement_2 (f : ℝ → ℝ) : Prop :=
  symmetric_line f (3 / 2)

def statement_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -f (x)

-- Main proof problem
theorem proof_problem (f : ℝ → ℝ) 
  (h_domain : domain_R f)
  (h_symmetric_pt : symmetric_graph_pt f (-3 / 4) 0)
  (h_symmetric : ∀ x : ℝ, f (x + 3 / 2) = -f (x))
  (h_property : ∀ x : ℝ, f (x + 2) = -f (-x + 4)) :
  statement_1 f ∧ statement_2 f ∧ statement_3 f :=
sorry

end proof_problem_l627_627286


namespace range_of_k_l627_627178

variable (k x : ℝ)

def f (k x : ℝ) : ℝ := k * x - Real.log x

def f' (k x : ℝ) : ℝ := k - 1/x

theorem range_of_k :
  (∀ x : ℝ, 1 < x → f' k x ≥ 0) ↔ k ∈ Set.Ici 1 := by
  sorry

end range_of_k_l627_627178


namespace number_of_messages_at_most_two_common_digits_l627_627194

theorem number_of_messages_at_most_two_common_digits (message : ℕ → ℕ)
  (h1 : ∀ n, (message n = 0 ∨ message n = 1))
  (h2 : message 0 = 0 ∧ message 1 = 1 ∧ message 2 = 1 ∧ message 3 = 0) :
  ∃ messages : Fin 16 → (ℕ → ℕ),
    ∀ i, (∑ n in Finset.range 4, if messages i n = message n then 1 else 0) ≤ 2 ∧
         Finset.card ((Finset.univ : Finset (Fin 16)).filter (λ i, (∑ n in Finset.range 4, if messages i n = message n then 1 else 0) ≤ 2)) = 11 :=
by
  sorry

end number_of_messages_at_most_two_common_digits_l627_627194


namespace common_elements_count_l627_627303

-- Definitions for the sequences
def a_n (n : ℕ) : ℕ := 3 * n + 2
def b_n (n : ℕ) : ℕ := 5 * n + 3

-- Definition of the set M
def M : set ℕ := {n | n ≤ 2018}

-- We want to count the number of common elements of the sequences within the set M
theorem common_elements_count : (finset.filter (λ x, ∃ m n, a_n m = x ∧ b_n n = x) (finset.range 2019)).card = 135 :=
by
  sorry

end common_elements_count_l627_627303


namespace unique_triangles_count_l627_627954

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627954


namespace non_congruent_triangles_with_perimeter_18_l627_627975

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627975


namespace value_of_y_when_x_plus_y_is_281_l627_627187

-- Definitions for the problem conditions
def sum_integers_from_to (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers_from_to (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

-- The Lean 4 statement for the problem
theorem value_of_y_when_x_plus_y_is_281 :
  let x := sum_integers_from_to 20 30 in
  let y := count_even_integers_from_to 20 30 in
  x + y = 281 → y = 6 :=
by
  -- Placeholders to ensure the statement builds correctly
  intros x y hyp
  exact sorry

end value_of_y_when_x_plus_y_is_281_l627_627187


namespace count_square_of_integer_fraction_l627_627872

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end count_square_of_integer_fraction_l627_627872


namespace range_of_a_l627_627920

-- Define the inequality condition
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  (ax + 1) * (1 + x) < 0

-- Define the sufficient condition for x
def sufficient_condition (x : ℝ) : Prop :=
  -2 < x ∧ x < -1

-- Main theorem to prove the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, sufficient_condition(x) → inequality_condition(a, x)) → 
  0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l627_627920


namespace gcd_of_180_and_450_l627_627085

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l627_627085


namespace simple_expression_for_f_l627_627570

noncomputable def find_simple_expression_for_f (a : ℕ → ℝ) (b : ℕ → ℕ) (n : ℕ) : ℝ :=
  1 / nat.factorial n * (finset.range n).prod (λ i, (b i))

theorem simple_expression_for_f (a : ℕ → ℝ) (b : ℕ → ℕ) (n : ℕ)
  (h_distinct : function.injective b) : find_simple_expression_for_f a b n = 
  1 / nat.factorial n * (finset.range n).prod (λ i, (b i)) :=
sorry

end simple_expression_for_f_l627_627570


namespace class_average_l627_627357

theorem class_average
  (total_students : ℕ) (marks_95_count : ℕ) (marks_0_count : ℕ) (rest_avg : ℕ) :
  total_students = 28 →
  marks_95_count = 4 →
  marks_0_count = 3 →
  rest_avg = 45 →
  (4 * 95 + 3 * 0 + (total_students - 4 - 3) * rest_avg) / total_students = 47.32 := 
by
  intros h_total_students h_marks_95_count h_marks_0_count h_rest_avg
  rw [h_total_students, h_marks_95_count, h_marks_0_count, h_rest_avg]
  norm_num
  sorry

end class_average_l627_627357


namespace six_prime_pairs_sum_to_sixty_l627_627163

open Nat

-- Define the condition that p1 and p2 are prime
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Define the condition that p1 and p2 sum to 60
def sumToSixty (p1 p2 : ℕ) : Prop := p1 + p2 = 60

-- List of prime numbers less than 30
def primesLessThanThirty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition for pairs of primes summing to 60 and p1 < p2 to avoid duplicates
def validPairs : (ℕ × ℕ) → Prop
  | (p1, p2) => isPrime p1 ∧ isPrime p2 ∧ sumToSixty p1 p2 ∧ p1 < p2

-- The main statement to prove the number of valid pairs
theorem six_prime_pairs_sum_to_sixty :
  (primesLessThanThirty.product primesLessThanThirty).filter validPairs = [(7, 53), (13, 47), (17, 43), (19, 41), (23, 37), (29, 31)].length := 6
  := sorry

end six_prime_pairs_sum_to_sixty_l627_627163


namespace smallest_percent_increase_l627_627546

noncomputable def prize_values : ℕ → ℕ
| 1 := 100
| 2 := 250
| 3 := 400
| 4 := 600
| 5 := 1000
| 6 := 2000
| 7 := 4000
| 8 := 7000
| 9 := 12000
| 10 := 20000
| 11 := 35000
| 12 := 60000
| 13 := 100000
| 14 := 160000
| 15 := 250000
| _ := 0

theorem smallest_percent_increase : 
  let percent_increase (q1 q2 : ℕ) := ((prize_values q2 - prize_values q1) * 100) / prize_values q1
  in percent_increase 3 4 = 50 ∧ 
     ∀ (q1 q2 : ℕ), (q1 = 1 ∧ q2 = 2 → percent_increase q1 q2 ≥ 50) ∧ 
                    (q1 = 2 ∧ q2 = 3 → percent_increase q1 q2 ≥ 50) ∧ 
                    (q1 = 12 ∧ q2 = 13 → percent_increase q1 q2 ≥ 50) ∧ 
                    (q1 = 14 ∧ q2 = 15 → percent_increase q1 q2 ≥ 50) := 
begin
  sorry
end

end smallest_percent_increase_l627_627546


namespace dot_product_range_l627_627476

variable (P : ℝ × ℝ) 
variable (E F : ℝ × ℝ) 

def ellipse (x y : ℝ) := x^2 / 16 + y^2 / 15 = 1
def circle (x y : ℝ) := (x - 1)^2 + y^2 = 4

-- Condition that P lies on the ellipse
noncomputable def P_on_ellipse : Prop := ellipse P.1 P.2

-- Condition that E and F define a diameter of the given circle
noncomputable def diameter_condition : Prop := circle E.1 E.2 ∧ circle F.1 F.2 ∧ ∃ N : ℝ × ℝ, 
  N = (1,0) ∧ E.1 + F.1 = 2 * N.1 ∧ E.2 + F.2 = 2 * N.2

-- The problem statement
theorem dot_product_range : 
  P_on_ellipse P →
  diameter_condition E F → 
  (∃ V : ℝ, V ∈ set.Icc 5 21 ∧ V = (P.1 - E.1) * (P.1 - F.1) + (P.2 - E.2) * (P.2 - F.2)) :=
sorry

end dot_product_range_l627_627476


namespace mod11_residue_l627_627723

theorem mod11_residue :
  (305 % 11 = 8) →
  (44 % 11 = 0) →
  (176 % 11 = 0) →
  (18 % 11 = 7) →
  (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end mod11_residue_l627_627723


namespace ellipse_and_fixed_points_l627_627675

-- Definitions using conditions from a):
def is_ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (ecc : ℝ) : Prop :=
  ecc = 1 / 2 ∧ (a = 2 * sqrt(3) / 2) ∧ (b = sqrt(3) * sqrt(3) / 2)

def passes_through_fixed_points (a b x_PQ y_P y_Q : ℝ) (cond1 : x_PQ = 4) (cond2 : y_P ≠ y_Q) : Prop :=
  y_P = 0 ∨ y_P = 7

-- The final theorem that needs to be proved
theorem ellipse_and_fixed_points :
  ∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b), is_ellipse a b a_pos b_pos a_gt_b (1 / 2) ∧
  (∀ (x_PQ y_P y_Q : ℝ), passes_through_fixed_points a b x_PQ y_P y_Q (x_PQ = 4) (y_P ≠ y_Q) → 
  (y_P = 0 ∨ y_P = 7)) :=
by
  sorry

end ellipse_and_fixed_points_l627_627675


namespace max_value_expression_l627_627592

theorem max_value_expression {x : ℝ} (h : 0 < x) : 
  ∃ y, y = (2 * Real.sqrt 2 - 2) ∧ ( ∀ x, 0 < x → (x^2 + 2 - Real.sqrt (x^4 + 4)) / x ≤ y ) :=
begin
  sorry
end

end max_value_expression_l627_627592


namespace non_congruent_triangles_with_perimeter_18_l627_627997

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l627_627997


namespace square_diagonal_perpendicular_bisector_l627_627628

open EuclideanGeometry

theorem square_diagonal_perpendicular_bisector 
  {A B C D M H : Point} 
  (h_square : square A B C D)
  (h_M_on_diagonal : M ∈ line_through A C)
  (h_AM_AB : dist A M = dist A B)
  (h_perpendicular : is_perpendicular (line_through M H) (line_through A C))
  (h_H_on_BC : H ∈ line_through B C) :
  (dist B H = dist H M) ∧ (dist H M = dist M C) :=
by
  sorry

end square_diagonal_perpendicular_bisector_l627_627628


namespace total_payment_difference_l627_627246

structure PaymentPlan where
  principal : ℚ
  rate : ℚ
  n : ℚ
  period : ℚ
  payments : ℚ → ℚ

def plan1 (principal : ℚ) (rate : ℚ) (n : ℚ) (period : ℚ) : ℚ :=
  let balance5 = principal * (1 + rate / n) ^ (n * (period / 2))
  let halfPayment = balance5 / 2
  let remainingBalance = halfPayment * (1 + rate / n) ^ (n * (period / 2))
  halfPayment + remainingBalance

def plan2 (principal : ℚ) (rate : ℚ) (n : ℚ) (period : ℚ) (midPayment : ℚ) : ℚ :=
  let balance5 = (principal - midPayment) * (1 + rate / n) ^ (n * (period / 2))
  midPayment + balance5

theorem total_payment_difference :
  let P := 10000
  let r := 0.08
  let n1 := 2
  let n2 := 1
  let T := 10
  let midPaymentPlan2 := 2000
  let totalPlan1 := plan1 P r n1 T
  let totalPlan2 := plan2 P r n2 T midPaymentPlan2
  abs (totalPlan1 - totalPlan2) = 4484 :=
by
  sorry

end total_payment_difference_l627_627246


namespace combined_area_correct_l627_627387

-- Define the conditions as given in a)
variables (d : ℝ) (w : ℝ) (h1 : d > 0) (h2 : w > 0) -- preconditions

-- Define the length of the rectangle as three times its width
def length_of_rectangle (w : ℝ) : ℝ :=
  3 * w

-- Define the dimensions of the rectangle and the attachment of the equilateral triangle
-- Here calculate the width 'w' from the diagonal using the equation derived from Pythagorean theorem
def width_from_diagonal (d : ℝ) : ℝ :=
  (d / (real.sqrt 10 : ℝ))

def area_of_combined_figure (d w : ℝ) : ℝ :=
  let l := 3 * w in
  let A_rect := w * l in
  let A_triangle := (real.sqrt 3 / 4) * w * w in
  A_rect + A_triangle

-- The theorem to prove
theorem combined_area_correct : 
  area_of_combined_figure d (width_from_diagonal d) = (d * d * (12 + real.sqrt 3) / 40) :=
sorry

end combined_area_correct_l627_627387


namespace bicycle_distance_l627_627358

theorem bicycle_distance (P_b P_f : ℝ) (h1 : P_b = 9) (h2 : P_f = 7) (h3 : ∀ D : ℝ, D / P_f = D / P_b + 10) :
  315 = 315 :=
by
  sorry

end bicycle_distance_l627_627358


namespace maximum_pressure_l627_627633

variables {R V0 T0 a b c : ℝ}
variables {P_max : ℝ}

def cyclic_process (V T : ℝ) : Prop :=
  ((V / V0 - a) ^ 2 + (T / T0 - b) ^ 2 = c ^ 2)

noncomputable def ideal_gas_pressure (R T V : ℝ) : ℝ :=
  R * T / V

theorem maximum_pressure 
  (h_c2_lt : c^2 < a^2 + b^2) :
  ∃ P_max : ℝ, P_max = (R * T0 / V0) * (a * real.sqrt(a^2 + b^2 - c^2) + b * c) / (b * real.sqrt(a^2 + b^2 - c^2) - a * c) :=
sorry

end maximum_pressure_l627_627633


namespace find_c_values_l627_627294

noncomputable def chordLengthOnCircle (c : ℝ) : ℝ :=
  let d := (abs (5 * 1 - 12 * (-2) + c)) / (Real.sqrt (5 ^ 2 + (-12) ^ 2))
  let r := 5
  let l := 8
  r^2 = d^2 + (l / 2) ^ 2

theorem find_c_values (c : ℝ) :
  chordLengthOnCircle c = true ↔ c = 10 ∨ c = -68 := by
  sorry

end find_c_values_l627_627294


namespace triangle_interior_angle_l627_627575

open EuclideanGeometry

noncomputable def problem_statement (A B C D M : Point) : Prop :=
  ∃ (A B C D M : Point),
  (triangle A B C) ∧
  (angle B A C = 2 * angle C A B) ∧
  (angle B A C > 90°) ∧
  (line_contains A B D) ∧
  (perpendicular (line C D) (line A C)) ∧
  (midpoint M B C) ∧
  (angle A M B = angle D M C)

-- The theorem stating the problem
theorem triangle_interior_angle :
  ∀ (A B C D M : Point), problem_statement A B C D M :=
begin
  -- Given conditions (stated in problem_statement)
  sorry
end

end triangle_interior_angle_l627_627575


namespace sqrt_x_minus_5_meaningful_iff_x_ge_5_l627_627185

theorem sqrt_x_minus_5_meaningful_iff_x_ge_5 (x : ℝ) : (∃ y : ℝ, y^2 = x - 5) ↔ (x ≥ 5) :=
sorry

end sqrt_x_minus_5_meaningful_iff_x_ge_5_l627_627185


namespace polar_to_rectangular_coordinates_l627_627830

noncomputable def rectangular_coordinates_from_polar (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_coordinates :
  rectangular_coordinates_from_polar 12 (5 * Real.pi / 4) = (-6 * Real.sqrt 2, -6 * Real.sqrt 2) :=
  sorry

end polar_to_rectangular_coordinates_l627_627830


namespace trapezoid_diagonals_perpendicular_iff_geometric_mean_l627_627642

structure Trapezoid :=
(a b c d e f : ℝ) -- lengths of sides a, b, c, d, and diagonals e, f.
(right_angle : d^2 = a^2 + c^2) -- Condition that makes it a right-angled trapezoid.

theorem trapezoid_diagonals_perpendicular_iff_geometric_mean (T : Trapezoid) :
  (T.e * T.e + T.f * T.f = T.a * T.a + T.b * T.b + T.c * T.c + T.d * T.d) ↔ 
  (T.d * T.d = T.a * T.c) := 
sorry

end trapezoid_diagonals_perpendicular_iff_geometric_mean_l627_627642


namespace fido_leash_yard_reach_area_product_l627_627843

noncomputable def fido_leash_yard_fraction : ℝ :=
  let a := 2 + Real.sqrt 2
  let b := 8
  a * b

theorem fido_leash_yard_reach_area_product :
  ∃ (a b : ℝ), 
  (fido_leash_yard_fraction = (a * b)) ∧ 
  (1 > a) ∧ -- Regular Octagon computation constraints
  (b = 8) ∧ 
  a = 2 + Real.sqrt 2 :=
sorry

end fido_leash_yard_reach_area_product_l627_627843


namespace ryan_learning_hours_l627_627431

theorem ryan_learning_hours (H_E : ℕ) (H_C : ℕ) (h1 : H_E = 6) (h2 : H_C = 2) : H_E - H_C = 4 := by
  sorry

end ryan_learning_hours_l627_627431


namespace angle_OMN_30_l627_627547

theorem angle_OMN_30 {O A B C M N : Point}
  [NonagonInscribedCircle O]
  (AB_side : Side O A B) (BC_side : Side O B C) (M_mid : Midpoint M A B) (N_mid : Midpoint N (Radius O B) (Radius O C))
  (perpendicular : Perpendicular (Radius O B) (LineSegment O N)) :
  ∠ O M N = 30 :=
by
  -- Definitions based on problem conditions, proof omitted
  sorry

end angle_OMN_30_l627_627547


namespace max_students_l627_627186

theorem max_students (A B C : ℕ) (A_left B_left C_left : ℕ)
  (hA : A = 38) (hB : B = 78) (hC : C = 128)
  (hA_left : A_left = 2) (hB_left : B_left = 6) (hC_left : C_left = 20) :
  gcd (A - A_left) (gcd (B - B_left) (C - C_left)) = 36 :=
by {
  sorry
}

end max_students_l627_627186


namespace solve_z_six_eq_neg_sixteen_l627_627859

theorem solve_z_six_eq_neg_sixteen (x y : Real) (z : Complex) (h : z = x + y * Complex.I) :
  z ^ 6 = -16 → z = Complex.I * (2:Real)^(2/3) ∨ z = -Complex.I * (2:Real)^(2/3) ∨
  z = (16/15)^(1/4 : Real) + Complex.I * (16/15)^(1/4 : Real) ∨
  z = -(16/15)^(1/4 : Real) - Complex.I * (16/15)^(1/4 : Real) ∨
  z = -(16/15)^(1/4 : Real) + Complex.I * (16/15)^(1/4 : Real) ∨
  z = (16/15)^(1/4 : Real) - Complex.I * (16/15)^(1/4 : Real) :=
sorry

end solve_z_six_eq_neg_sixteen_l627_627859


namespace problem_solution_l627_627481

variables {f : ℝ → ℝ}

-- f is monotonically decreasing on [1, 3]
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

-- f(x+3) is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = f (3 - x)

-- Given conditions
axiom mono_dec : monotone_decreasing_on f 1 3
axiom even_f : even_function f

-- To prove: f(π) < f(2) < f(5)
theorem problem_solution : f π < f 2 ∧ f 2 < f 5 :=
by
  sorry

end problem_solution_l627_627481


namespace angle_DCB_45_degrees_l627_627054

theorem angle_DCB_45_degrees 
  (O : Type) [MetricSpace O] [NormedAddCommGroup O] [NormedSpace ℝ O]
  (A B C D P : O)
  (hAB : dist A B = 2) -- diameter AB
  (hCP : dist C P = 7 / 5)
  (hPD : dist P D = 5)
  (hAP : dist A P = 1)
  (hCircle : ∀ X : O, dist O X = 1) : -- Circle with radius 1
  angle C D B = π / 4 :=
by sorry

end angle_DCB_45_degrees_l627_627054


namespace angle_BAC_60_l627_627548

theorem angle_BAC_60
  (ABC : Triangle)
  (is_acute : acute_triangle ABC)
  (bisector_AL : angle_bisector ABC A L)
  (altitude_BH : altitude ABC B H)
  (perpendicular_bisector_AB : perpendicular_bisector ABC A B perp_pt)
  (intersect_at_one_point : concurrency {bisector_AL, altitude_BH, perpendicular_bisector_AB}) :
  angle ABC A C = 60 := sorry

end angle_BAC_60_l627_627548


namespace number_of_valid_menus_l627_627374

def Dessert : Type := 
| cake
| pie
| ice_cream
| pudding
| cookies

-- function to count valid dessert menus for a week given the constraints
def count_valid_menus : Nat := 4096

theorem number_of_valid_menus :
  ∃ (menus : List (Fin 7 → Dessert)), 
    (∀ i, menus i ≠ menus (i + 1)) ∧ 
    menus 3 = Dessert.pie ∧ 
    menus.length = 4096 :=
sorry

end number_of_valid_menus_l627_627374


namespace sufficient_but_not_necessary_condition_for_negativity_l627_627141

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b*x + c

theorem sufficient_but_not_necessary_condition_for_negativity (b c : ℝ) :
  (c < 0 → ∃ x : ℝ, f b c x < 0) ∧ (∃ b c : ℝ, ∃ x : ℝ, c ≥ 0 ∧ f b c x < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_negativity_l627_627141


namespace find_original_price_l627_627217

-- Definitions given in the conditions
def original_price : ℝ := 30.22
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.25
def final_price_after_discounts : ℝ := 17
def discounted_price (price : ℝ) (discount_rate : ℝ) : ℝ := price * (1 - discount_rate)

-- The proof problem
theorem find_original_price 
  (P : ℝ)
  (h1 : P = discounted_price (discounted_price original_price first_discount_rate) second_discount_rate)
  (h2 : h1 = final_price_after_discounts) :
  original_price = 30.22 :=
sorry

end find_original_price_l627_627217


namespace greatest_possible_individual_award_l627_627386

variable (prize : ℕ)
variable (total_winners : ℕ)
variable (min_award : ℕ)
variable (fraction_prize : ℚ)
variable (fraction_winners : ℚ)

theorem greatest_possible_individual_award 
  (h1 : prize = 2500)
  (h2 : total_winners = 25)
  (h3 : min_award = 50)
  (h4 : fraction_prize = 3/5)
  (h5 : fraction_winners = 2/5) :
  ∃ award, award = 1300 := by
  sorry

end greatest_possible_individual_award_l627_627386


namespace solution_set_of_inequality_l627_627311

theorem solution_set_of_inequality : {x : ℝ // |x - 2| > x - 2} = {x : ℝ // x < 2} :=
sorry

end solution_set_of_inequality_l627_627311


namespace product_of_six_distinct_divisible_by_perfect_square_l627_627272

theorem product_of_six_distinct_divisible_by_perfect_square :
  ∀ (s : Finset ℕ), (s ⊆ (Finset.range 10).map (Nat.succ)) ∧ s.card = 6 →
  ∃ k : ℕ, 1 < k ∧ k * k ∣ s.prod id :=
by
  sorry

end product_of_six_distinct_divisible_by_perfect_square_l627_627272


namespace nat_set_satisfy_inequality_l627_627309

theorem nat_set_satisfy_inequality :
  {x : ℕ | -3 < 2 * x - 1 ∧ 2 * x - 1 ≤ 3} = {0, 1, 2} :=
by
  sorry

end nat_set_satisfy_inequality_l627_627309


namespace count_valid_leap_years_l627_627385

-- Define the condition for a year to be a leap year under the new rule
def is_new_leap_year (y : ℕ) : Prop :=
  y % 800 = 300 ∨ y % 800 = 500

-- Define the range condition
def in_leap_year_range (y : ℕ) : Prop :=
  1996 ≤ y ∧ y ≤ 4096

-- Combine the conditions for the leap years in the range
def valid_leap_years : set ℕ :=
  {y | is_new_leap_year y ∧ in_leap_year_range y}

-- State the proof goal as a theorem
theorem count_valid_leap_years : ∃ n, n = 4 ∧ ∀ s : finset ℕ, (∀ y ∈ s, y ∈ valid_leap_years) → s.card = n :=
by
  sorry

end count_valid_leap_years_l627_627385


namespace problem_1645800_l627_627365

def numDigits (n: ℕ) : ℕ := Nat.log10' n + 1

def highestDigitPlace (n: ℕ) : String :=
  let s := n.toString
  match s.length with
  | 7 => "million"
  | _ => "unknown"

theorem problem_1645800 :
  (numDigits 1645800 = 7) ∧ (highestDigitPlace 1645800 = "million") :=
by
  sorry

end problem_1645800_l627_627365


namespace number_of_unique_combinations_l627_627640

-- Define the inputs and the expected output.
def n := 8
def r := 3
def expected_combinations := 56

-- We state our theorem indicating that the combination of 8 toppings chosen 3 at a time
-- equals 56.
theorem number_of_unique_combinations :
  (Nat.choose n r = expected_combinations) :=
by
  sorry

end number_of_unique_combinations_l627_627640


namespace prob_A_or_B_l627_627137

open ProbabilityTheory

theorem prob_A_or_B (A B: Event) [Independence A B] (h1: P A = 0.6) (h2: P (A ∩ B) = 0.42) :
  P (A ∪ B) = 0.88 :=
by
  sorry

end prob_A_or_B_l627_627137


namespace max_value_of_a_l627_627226

theorem max_value_of_a
  (a : ℤ)
  (h : ∀ x : ℝ, 0 < x → (e^x + 3) / x ≥ real.exp a) :
  a ≤ 1 := sorry

end max_value_of_a_l627_627226


namespace min_T_tiles_needed_l627_627046

variable {a b c d : Nat}
variable (total_blocks : Nat := a + b + c + d)
variable (board_size : Nat := 8 * 10)
variable (block_size : Nat := 4)
variable (tile_types := ["T_horizontal", "T_vertical", "S_horizontal", "S_vertical"])
variable (conditions : Prop := total_blocks = 20 ∧ a + c ≥ 5)

theorem min_T_tiles_needed
    (h : conditions)
    (covering : total_blocks * block_size = board_size)
    (T_tiles : a ≥ 6) :
    a = 6 := sorry

end min_T_tiles_needed_l627_627046


namespace age_of_youngest_child_l627_627756

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) :
  x = 4 :=
sorry

end age_of_youngest_child_l627_627756


namespace center_is_five_l627_627808

-- Definitions corresponding to the conditions
def numbers := {0, 1, 2, 3, 4, 5, 6, 7, 8}
def consecutive (a b : Nat) : Prop := a ∈ numbers ∧ b ∈ numbers ∧ has_adjacent_square a b
def corners_sum (arr : Matrix Nat Nat Nat) : Prop := arr 0 0 + arr 0 2 + arr 2 0 + arr 2 2 = 20
def center_value (arr : Matrix Nat Nat Nat) : Nat := arr 1 1

-- Proof statement
theorem center_is_five (arr : Matrix Nat Nat Nat)
  (h1 : ∀ a b, consecutive a b → ∃ i j, arr i j = a ∧ arr (has_adjacent_square_pos i j) = b)
  (h2 : corners_sum arr) :
  center_value arr = 5 := 
sorry

end center_is_five_l627_627808


namespace non_congruent_triangles_with_perimeter_18_l627_627970

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627970


namespace non_congruent_triangles_with_perimeter_18_l627_627974

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627974


namespace normal_dist_properties_l627_627238

noncomputable def normal_distribution := sorry

variable (xi : ℝ → ℝ) (µ : ℝ) (σ : ℝ) 

axiom normal_dist : xi ~ normal_distribution

theorem normal_dist_properties (h : normal_dist xi):
  (prob (xi < 2) = prob (xi > 4)) → (µ = 3) ∧ (σ = 7) := by
  sorry

end normal_dist_properties_l627_627238


namespace race_positions_l627_627190

theorem race_positions :
  ∀ (M J T R H D : ℕ),
    (M = J + 3) →
    (J = T + 1) →
    (T = R + 3) →
    (H = R + 5) →
    (D = H + 4) →
    (M = 9) →
    H = 7 :=
by sorry

end race_positions_l627_627190


namespace domain_of_f_sqrt_l627_627062

noncomputable def domain_of_sqrt_function := { x : ℝ | (x ≤ 1 ∨ x ≥ 2) }

theorem domain_of_f_sqrt (x : ℝ) :
  f x = sqrt (7 - sqrt (x ^ 2 - 3 * x + 2)) ↔ x ∈ domain_of_sqrt_function :=
by
  -- We should prove equivalence with the domain conditions
  sorry

end domain_of_f_sqrt_l627_627062


namespace color_linearity_l627_627626

structure Point :=
(x : ℝ)
(y : ℝ)

def is_equilateral (A B C : Point) : Prop :=
  let d1 := (A.x - B.x)^2 + (A.y - B.y)^2 in
  let d2 := (B.x - C.x)^2 + (B.y - C.y)^2 in
  let d3 := (C.x - A.x)^2 + (C.y - A.y)^2 in
  d1 = d2 ∧ d2 = d3

def rotate_60 (P Q : Point) : Point :=
  -- Function to calculate the new point after rotating 60 degrees clockwise
  sorry

theorem color_linearity (K C G : Point)
  (H1: ∀ K C G, is_equilateral K C G)
  (H2: ∀ K G, is_equilateral K G (rotate_60 K G))
  (H3: ∀ C G, is_equilateral C G (rotate_60 C G))
  (H4: ∀ A B, A = B -> A = A) : -- Ensuring paint is allowed on an already painted point
  (∀ moves. ∃ R B Y, all_red_points R → are_collinear R ∧ all_blue_points B → are_collinear B ∧ all_yellow_points Y → are_collinear Y) :=
sorry

end color_linearity_l627_627626


namespace parrot_arrangement_l627_627541

theorem parrot_arrangement : 
  ∃ arrangements : Finset (Perm (Fin 8)), 
  (∀ σ ∈ arrangements, (σ 0 = 0 ∨ σ 0 = 1) ∧ (σ 7 = 0 ∨ σ 7 = 1) ∧ σ 3 = 7) ∧ 
  arrangements.card = 240 :=
sorry

end parrot_arrangement_l627_627541


namespace problem1_problem2_l627_627821

theorem problem1 : (1 * (-5) - (-6) + (-7)) = -6 :=
by
  sorry

theorem problem2 : (-1)^2021 + (-18) * abs (-2 / 9) - 4 / (-2) = -3 :=
by
  sorry

end problem1_problem2_l627_627821


namespace sum_of_absolute_values_l627_627879

theorem sum_of_absolute_values :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ),
    (2 * (Polynomial.X : Polynomial ℤ) - 1)^5 = a_0 * Polynomial.X^5 + a_1 * Polynomial.X^4 + a_2 * Polynomial.X^3 + a_3 * Polynomial.X^2 + a_4 * Polynomial.X + a_5 →
    |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 1 :=
by
  intro a_0 a_1 a_2 a_3 a_4 a_5 h
  sorry

end sum_of_absolute_values_l627_627879


namespace maximum_books_borrowed_l627_627196

theorem maximum_books_borrowed (total_students books_each avg_books_per_student :
  nat) (students_0 students_1 students_2 : nat) :
  total_students = 20 →
  students_0 = 3 →
  students_1 = 9 →
  students_2 = 4 →
  avg_books_per_student = 2 →
  ∀ (remaining_books remaining_students students_with_books : nat),
    remaining_students = total_students - students_0 - students_1 - students_2 →
    students_with_books = remaining_students →
    remaining_books = total_students * avg_books_per_student -
      (students_0 * 0 + students_1 * 1 + students_2 * 2) →
      (∀ (i : ℕ), i < students_with_books → 3 ≤ (books_each)) →
      books_each ≤ remaining_books →
      ∃ (max_books : nat), max_books = remaining_books - (students_2 * 3) ∧
      max_books = 14 :=
begin
  sorry
end

end maximum_books_borrowed_l627_627196


namespace purely_imaginary_iff_l627_627075

noncomputable def roots : set ℝ :=
{x ∈ ℝ | (x - complex.I) * (x + 2 - complex.I) * (x + 4 - complex.I) = 
        (0 + (-(3*x^2 + 10*x + 3) * complex.I))}

theorem purely_imaginary_iff (x : ℝ) 
  (h : ((x - complex.I) * (x + 2 - complex.I) * (x + 4 - complex.I)).re = 0) : 
  x = -3 ∨ x = (-3 + real.sqrt 13) / 2 ∨ x = (-3 - real.sqrt 13) / 2 :=
sorry

end purely_imaginary_iff_l627_627075


namespace probability_region_9_l627_627223

-- Define the unit square and random point Q 
def unit_square := set.Icc (0 : ℝ) 1 × set.Icc (0 : ℝ) 1

-- Define the fixed point (1/4, 3/4)
def fixed_point : ℝ × ℝ := (1/4, 3/4)

-- Define the line inequality
def line_ineq (Q : ℝ × ℝ) : Prop :=
  Q.2 ≥ Q.1 + 1/2

-- Define the region above the line within the unit square
def region := { Q : ℝ × ℝ | Q ∈ unit_square ∧ line_ineq Q }

-- Problem statement
theorem probability_region_9 :
  let p := 1
  let q := 8
  p + q = 9 ∧ MeasureTheory.measure space unit_square = 1 ∧ MeasureTheory.measure space region = 1 / 8 :=
sorry

end probability_region_9_l627_627223


namespace product_distances_l627_627656

noncomputable def parametric_line (α : ℝ) (P : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (P.1 + Real.cos α * t, P.2 + Real.sin α * t)

def circle (r : ℝ) (Q : ℝ × ℝ) : Prop := Q.1^2 + Q.2^2 = r^2

theorem product_distances 
  (P A B : ℝ × ℝ)
  (α : ℝ)
  (hα : α = Real.pi / 6)
  (hp : P = (1, 1))
  (hl : ∃ t, A = parametric_line α P t ∧ circle 2 A)
  (hk : ∃ t, B = parametric_line α P t ∧ circle 2 B) :
  dist P A * dist P B = 2 :=
sorry

end product_distances_l627_627656


namespace non_congruent_triangles_with_perimeter_18_l627_627989

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627989


namespace distance_between_parallel_lines_l627_627585

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 1
def l2 (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem distance_between_parallel_lines :
  let l1_tangent_point_x := 0 in  -- derived from f'(x) = 2
  let l1_tangent_point_y := 1 in  -- since f(l1_tangent_point_x) = 1
  Real.sqrt ((2 * l1_tangent_point_x + (-1) * l1_tangent_point_y + 3)^2 / (2^2 + (-1)^2)) = (2 * Real.sqrt 5) / 5 :=
by 
  sorry

end distance_between_parallel_lines_l627_627585


namespace conjugate_of_z_l627_627532

variable (z : ℂ)

-- Condition: z * (1 + complex.I) = 1 - complex.I
def satisfies_condition (z : ℂ) : Prop :=
  z * (1 + complex.I) = 1 - complex.I

-- Goal: prove that the conjugate of z is i
theorem conjugate_of_z (h : satisfies_condition z) : complex.conj z = complex.I :=
sorry

end conjugate_of_z_l627_627532


namespace area_triangle_PMN_l627_627651

-- Definitions and conditions
variable (P Q R S M N : Type) [Square PQRS] [Midpoint M P Q] [Midpoint N P S]
variable (area_PQRS : Real)
variable (midpoint_condition_1 : M = (P + Q) / 2)
variable (midpoint_condition_2 : N = (P + S) / 2)

-- Given the area of the square PQRS
axiom h_area_PQRS : area PQRS = 900

-- Problem statement: Prove the area of triangle PMN is 112.5
theorem area_triangle_PMN :
  area P M N = 112.5 :=
sorry

end area_triangle_PMN_l627_627651


namespace evaluate_infinite_sum_l627_627071

noncomputable def infinite_sum := ∑' n, (n : ℝ) / (n^4 + 16)

theorem evaluate_infinite_sum : infinite_sum = 5 / 8 := by
  sorry

end evaluate_infinite_sum_l627_627071


namespace cost_of_4_bags_of_ice_l627_627401

theorem cost_of_4_bags_of_ice (
  cost_per_2_bags : ℝ := 1.46
) 
  (h : cost_per_2_bags / 2 = 0.73)
  :
  4 * (cost_per_2_bags / 2) = 2.92 :=
by 
  sorry

end cost_of_4_bags_of_ice_l627_627401


namespace find_c_plus_d_l627_627516

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l627_627516


namespace article_production_l627_627167

-- Conditions
variables (x z : ℕ) (hx : 0 < x) (hz : 0 < z)
-- The given condition: x men working x hours a day for x days produce 2x^2 articles.
def articles_produced_x (x : ℕ) : ℕ := 2 * x^2

-- The question: the number of articles produced by z men working z hours a day for z days
def articles_produced_z (x z : ℕ) : ℕ := 2 * z^3 / x

-- Prove that the number of articles produced by z men working z hours a day for z days is 2 * (z^3) / x
theorem article_production (hx : 0 < x) (hz : 0 < z) :
  articles_produced_z x z = 2 * z^3 / x :=
sorry

end article_production_l627_627167


namespace exists_real_numbers_l627_627571

theorem exists_real_numbers (f : ℝ → ℝ) :
  ∃ (x y : ℝ), f (x - f y) > y * f x + x :=
begin
  sorry
end

end exists_real_numbers_l627_627571


namespace tan_half_angle_l627_627586

theorem tan_half_angle (p q : ℝ) (h_cos : Real.cos p + Real.cos q = 3 / 5) (h_sin : Real.sin p + Real.sin q = 1 / 5) : Real.tan ((p + q) / 2) = 1 / 3 :=
sorry

end tan_half_angle_l627_627586


namespace opposite_of_neg2016_l627_627298

theorem opposite_of_neg2016 : -(-2016) = 2016 := 
by 
  sorry

end opposite_of_neg2016_l627_627298


namespace michael_max_notebooks_l627_627615

theorem michael_max_notebooks (money notebook_price : ℝ) (h_money: money = 12) (h_price: notebook_price = 1.25) : 
    ∃ n : ℕ, n ≤ 12 / 1.25 ∧ n = 9 := 
by 
  sorry

end michael_max_notebooks_l627_627615


namespace find_m_value_find_min_value_l627_627914

noncomputable def f (x m : Real) : Real := sqrt 3 * sin (2 * x) + 2 * cos x ^ 2 + m

theorem find_m_value (m : Real) (hmax : ∀ x ∈ Icc (0 : Real) (Real.pi / 4), f x m ≤ 1) :
  m = -2 :=
sorry

theorem find_min_value (x : Real) (h : m = -2) :
  ∃ (y : Real), f y m = -3 ∧ ∃ k : ℤ, x = 2 * π / 3 + k * π :=
sorry

end find_m_value_find_min_value_l627_627914


namespace num_consecutive_zeros_2006_factorial_l627_627441

/-- The number of consecutive zeros at the end of 2006! in base 10 representation is 500. -/
theorem num_consecutive_zeros_2006_factorial : 
  let f (p n : ℕ) : ℕ := 
    (∑ k in (range (nat.log p n).succ), n / (p ^ (k + 1)))
  in
    f 5 2006 = 500 :=
by
  let f (p n : ℕ) : ℕ := 
    (∑ k in (range (nat.log p n).succ), n / (p ^ (k + 1)))
  have f5_2006_eq_500 : f 5 2006 = 401 + 80 + 16 + 3 := by sorry
  have sum_eq_500 : 401 + 80 + 16 + 3 = 500 := by norm_num
  exact Eq.trans f5_2006_eq_500 sum_eq_500

end num_consecutive_zeros_2006_factorial_l627_627441


namespace exists_triangle_in_divided_square_l627_627391

theorem exists_triangle_in_divided_square (n : ℕ) (h : n > 1) 
  (polygons : Finset (Finset Point)) 
  (h_polygons : ∀ p ∈ polygons, convex p) 
  (distinct_sides : pairwise (≠) (Finset.card ∘ polygon_sides)) : 
  ∃ t ∈ polygons, polygon_sides t = 3 :=
sorry

end exists_triangle_in_divided_square_l627_627391


namespace num_integers_sq_condition_l627_627867

theorem num_integers_sq_condition : 
  {n : ℤ | n < 30 ∧ (∃ k : ℤ, k ^ 2 = n / (30 - n))}.to_finset.card = 3 := 
by
  sorry

end num_integers_sq_condition_l627_627867


namespace integer_range_2014_l627_627292

theorem integer_range_2014 : 1000 < 2014 ∧ 2014 < 10000 := by
  sorry

end integer_range_2014_l627_627292


namespace arithmetic_sequence_201_is_61_l627_627209

def is_arithmetic_sequence_term (a_5 a_45 : ℤ) (n : ℤ) (a_n : ℤ) : Prop :=
  ∃ d a_1, a_1 + 4 * d = a_5 ∧ a_1 + 44 * d = a_45 ∧ a_1 + (n - 1) * d = a_n

theorem arithmetic_sequence_201_is_61 : is_arithmetic_sequence_term 33 153 61 201 :=
sorry

end arithmetic_sequence_201_is_61_l627_627209


namespace part_I_part_II_l627_627686

variable {a : ℕ → ℕ}
variable {S : ℕ → ℚ}
variable {f : ℕ → ℕ}
variable {C : ℕ → ℕ}
variable {T : ℕ → ℕ}

/-- 
Given: 
1. Sequence \{a\} is positive and the sum of the first \(n\) terms \(S(n)\) such that \(S(n) = \frac{1}{4}a(n)^2 + \frac{1}{2}a(n) + \frac{1}{4}\) for \(n \in \mathbb{N}^+\).
2. Function \(f(n)\) defined piecewise:
\[
f(n) = \begin{cases} 
    a(n), & \text{if } n \text{ is odd} \\ 
    f\left(\frac{n}{2}\right), & \text{if } n \text{ is even} 
\end{cases}
\]
3. Sequence \{C\} defined by \(C(n) = f(2^n + 4)\) for \(n \in \mathbb{N}^+\).
Prove:
1. \(a(n) = 2n - 1\).
2. The sum \(T(n)\) of the first \(n\) terms of the sequence \{C\} is:
\[
T(n) = \begin{cases} 
    5, & \text{if } n = 1 \\ 
    2^n + n, & \text{if } n \geq 2 
\end{cases} 
\]
-/
def a_n (n : ℕ) (S : ℕ → ℚ) : ℕ :=
  match n with
  | 0     => 0
  | n+1   => 2*(n+1) - 1

def T (n : ℕ) (f : ℕ → ℕ) (C : ℕ → ℕ) : ℕ :=
  if n = 1 then 5 else 2^n + n

-- Proof statement: a_n is correctly defined and T is correctly computed
theorem part_I (n : ℕ) (S : ℕ → ℚ) : a n = 2 * n - 1 := by
  sorry

theorem part_II (n : ℕ) (C : ℕ → ℕ) (f : ℕ → ℕ) : 
  T n = if n = 1 then 5 else 2 ^ n + n := by
  sorry

end part_I_part_II_l627_627686


namespace count_square_of_integer_fraction_l627_627871

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end count_square_of_integer_fraction_l627_627871


namespace count_lines_with_intersections_l627_627863

theorem count_lines_with_intersections (n : ℕ) (h₁ : ∀ l, l ∈ {1, 2, ..., n} → intersect_count l = 2004) :
    {n : ℕ | ∃ f : set(ℕ), (∀ l ∈ f, l ≠ 0) ∧ ∀ l ∈ f, (intersect_count l = 2004)}.card = 12 :=
begin
  sorry
end

end count_lines_with_intersections_l627_627863


namespace sampling_method_is_systematic_l627_627316

-- Define the conditions
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  required_student_num : Nat

-- Define our specific problem's conditions
def problem_conditions : Grade :=
  { num_classes := 12, students_per_class := 50, required_student_num := 14 }

-- State the theorem
theorem sampling_method_is_systematic (G : Grade) (h1 : G.num_classes = 12) (h2 : G.students_per_class = 50) (h3 : G.required_student_num = 14) : 
  "Systematic sampling" = "Systematic sampling" :=
by
  sorry

end sampling_method_is_systematic_l627_627316


namespace median_mode_hits_l627_627769

theorem median_mode_hits : 
  let data := [7, 7, 8, 8, 8, 9, 9, 9, 9, 10] in 
  list.median data = 8.5 ∧ list.mode data = 9 := 
sorry

end median_mode_hits_l627_627769


namespace sin_difference_identity_example_l627_627899

theorem sin_difference_identity_example
  (h1 : sin α = 12 / 13)
  (h2 : cos β = 4 / 5)
  (h3 : π / 2 < α ∧ α < π) -- Second quadrant condition
  (h4 : -π / 2 < β ∧ β < 0) -- Fourth quadrant condition
  : sin (α - β) = 33 / 65 :=
by
  sorry

end sin_difference_identity_example_l627_627899


namespace non_congruent_triangles_with_perimeter_18_l627_627990

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627990


namespace sum_of_digits_palindrome_l627_627381

theorem sum_of_digits_palindrome 
  (r : ℕ) 
  (h1 : r ≤ 36) 
  (x p q : ℕ) 
  (h2 : 2 * q = 5 * p) 
  (h3 : x = p * r^3 + p * r^2 + q * r + q) 
  (h4 : ∃ (a b c : ℕ), (x * x = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a)) : 
  (2 * (a + b + c) = 36) := 
sorry

end sum_of_digits_palindrome_l627_627381


namespace flower_shop_options_l627_627619

theorem flower_shop_options :
  {n : ℕ // n = {xy : ℕ × ℕ // 2 * xy.1 + 3 * xy.2 = 20}.card} = 3 :=
by
  sorry

end flower_shop_options_l627_627619


namespace problem1_problem2_problem3_l627_627910

open Real

variable {f : ℝ → ℝ} (h_f' : Differentiable ℝ f) (h_f'' : Differentiable ℝ (deriv f))
variable {g : ℝ → ℝ} (h_g : ∀ x, g x = deriv f x) (h_g' : ∀ x, deriv g x < 0)

-- Problem 1:
theorem problem1 {x x0 : ℝ} : f x ≤ f x0 + deriv f x0 * (x - x0) :=
sorry

-- Problem 2:
variable {n : ℕ} {λ : Fin n → ℝ} {x : Fin n → ℝ} (h_λ : ∀ i, λ i ≥ 0) (h_λsum : (∑ i, λ i) = 1)
theorem problem2 : (∑ i, λ i * f (x i)) ≤ f (∑ i, λ i * x i) :=
sorry

-- Problem 3:
variable {a : ℝ} (h_geom : ∃ q > 0, (λ n, f^[n] a) = (λ n, a * q^n))
theorem problem3 : f a = a :=
sorry

end problem1_problem2_problem3_l627_627910


namespace proof_l627_627502

open Set

variable {U : Set ℝ}
variable {P Q : Set ℝ}

def U := univ
def P := { x | x^2 ≥ 9 }
def Q := { x | x > 2 }

theorem proof (h1 : U = univ) (h2 : P = { x : ℝ | x^2 ≥ 9 }) (h3 : Q = { x : ℝ | x > 2 }) :
  Q ∩ (U \ P) = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  -- The proof will go here
  sorry

end proof_l627_627502


namespace positive_integer_solutions_count_l627_627160

theorem positive_integer_solutions_count :
  let positive_solutions := {x : ℕ | 12 < -2 * (x : ℤ) + 18 ∧ x > 0}
  finset.card positive_solutions = 2 :=
by
  sorry

end positive_integer_solutions_count_l627_627160


namespace average_age_of_cricket_team_l627_627663

theorem average_age_of_cricket_team :
  let captain_age := 28
  let ages_sum := 28 + (28 + 4) + (28 - 2) + (28 + 6)
  let remaining_players := 15 - 4
  let total_sum := ages_sum + remaining_players * (A - 1)
  let total_players := 15
  total_sum / total_players = 27.25 := 
by 
  sorry

end average_age_of_cricket_team_l627_627663


namespace triangle_ABC_proof_l627_627584

variables {A B C D E F : Type*}
variables {a b c : Real} -- Side lengths
variables {angle_BAC : Real} -- Angle at A
variables {AD BE CF : Real} -- Angle bisectors

-- Given conditions
variables (hab : a > 0) (hbc : b > 0) (hca : c > 0)
variables (h_triangle : AD * BE * CF = rI) (rI : Real) (h_rI : rI = IE ∧ rI = IF ∧ rI = AD)

-- Proving the results
theorem triangle_ABC_proof (hangle_BAC : angle_BAC = 90):
  (2 * a + 2 * b + c - b) ∧ (angle_BAC > 90) :=
begin
    sorry
end

end triangle_ABC_proof_l627_627584


namespace sum_of_common_divisors_of_140_and_35_is_48_l627_627100

def divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n + 1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ divisors m) (divisors n)

def sum_divisors (divisors : List ℕ) : ℕ :=
  List.foldr (λ x acc, x + acc) 0 divisors

theorem sum_of_common_divisors_of_140_and_35_is_48 :
  sum_divisors (common_divisors 140 35) = 48 :=
by
  -- Proof would go here, using sorry to indicate that the proof is omitted
  sorry

end sum_of_common_divisors_of_140_and_35_is_48_l627_627100


namespace partition_no_infinite_arith_prog_l627_627558

theorem partition_no_infinite_arith_prog :
  ∃ (A B : Set ℕ), 
  (∀ n ∈ A, n ∈ B → False) ∧ 
  (∀ (a b : ℕ) (d : ℕ), (a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b) % d = 0) → False) ∧
  (∀ (a b : ℕ) (d : ℕ), (a ∈ B ∧ b ∈ B ∧ a ≠ b ∧ (a - b) % d = 0) → False) :=
sorry

end partition_no_infinite_arith_prog_l627_627558


namespace solve_trig_equation_l627_627745

theorem solve_trig_equation 
  (t : ℝ) 
  (k : ℤ) 
  (sin cos : ℝ -> ℝ) 
  (h1 : ∀ t, sin 2 * t = 2 * sin t * cos t) 
  (h2 : ∀ t, cos^2 t + sin^2 t = 1)
  (h3 : 4 * (sin t * cos^5 t + cos t * sin^5 t) + (sin (2 * t))^3 = 1) :
  ∃ k : ℤ, t = (-1)^k * (π / 12) + k * (π / 2) := 
begin
  sorry
end

end solve_trig_equation_l627_627745


namespace f_nonneg_for_all_x_ge_0_iff_a_ge_1_l627_627491

def f (a x : ℝ) : ℝ := (x - 2) * Real.exp x + a * x + 2

theorem f_nonneg_for_all_x_ge_0_iff_a_ge_1 (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ 0) ↔ (1 ≤ a) :=
by 
  sorry

end f_nonneg_for_all_x_ge_0_iff_a_ge_1_l627_627491


namespace maximize_tetrahedron_OB_length_l627_627052

theorem maximize_tetrahedron_OB_length
  (cone : Type) [is_cone cone]
  (P A O B H C : cone)
  (isosceles_right_triangle : axial_cross_section cone P)
  (circumference : on_circumference base_circle A)
  (inside_circle : inside_base_circle base_circle B)
  (center : is_center base_circle O)
  (perpendicular_AB_OB : AB ⟂ OB)
  (perpendicular_OH_PB : OH ⟂ PB)
  (PA_equals_4 : PA = 4)
  (midpoint_C : midpoint C PA) :
  OB = sqrt(6) / 3 := 
sorry

end maximize_tetrahedron_OB_length_l627_627052


namespace tangent_line_eq_l627_627903

open Real

noncomputable def f (x a : ℝ) : ℝ := (x + a) * log x

noncomputable def y (x a : ℝ) : ℝ := a * x^3

theorem tangent_line_eq :
  (∃ a : ℝ, (∀ x : ℝ, fderiv ℝ (λ x, f x a) 1 = 0) ∧ a = -1) →
  tangent_line_eq_at (y (1 : ℝ) (λ a, f (1 : ℝ) a)) (1, -1) 1 = 3 * x + y - 2 :=
by
  sorry

end tangent_line_eq_l627_627903


namespace largest_four_digit_number_divisible_5_6_2_l627_627336

noncomputable def largest_four_digit_number_divisible_by (a b c : ℕ) (h₁: a = 5) (h₂: b = 6) (h₃: c = 2) : ℕ :=
  9990

theorem largest_four_digit_number_divisible_5_6_2 : 
  largest_four_digit_number_divisible_by 5 6 2 5.refl 6.refl 2.refl = 9990 :=
by
  -- The proof goes here.
  sorry

end largest_four_digit_number_divisible_5_6_2_l627_627336


namespace probability_sum_greater_than_8_given_even_product_l627_627644

open Finset

def possible_combinations := 
  {{1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 3, 4}, {1, 4, 5}, {2, 3, 4}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5}}

def even_combinations := 
  {{1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 3, 4}, {1, 4, 5}, {2, 3, 4}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5}}

def favorable_combinations := 
  {{1, 4, 5}, {2, 3, 4}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5}}

theorem probability_sum_greater_than_8_given_even_product :
  ∑ (s : Finset ℕ) in possible_combinations.filter (λ s, s.sum > 8), 1 / ∑ (s : Finset ℕ) in even_combinations, 1 = 5 / 9 :=
by
  have : even_combinations = {{1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 3, 4}, {1, 4, 5}, {2, 3, 4}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5}},
  simp [possible_combinations, even_combinations, favorable_combinations]
  sorry

end probability_sum_greater_than_8_given_even_product_l627_627644


namespace max_value_of_linear_expression_l627_627118

theorem max_value_of_linear_expression 
  (x y z : ℝ)
  (h : x^2 + y^2 + z^2 = 2) : 
  3 * x + 4 * y + 5 * z ≤ 10 :=
begin
  sorry
end

end max_value_of_linear_expression_l627_627118


namespace gcd_180_450_l627_627088

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l627_627088


namespace termite_cannot_finish_in_central_cube_l627_627188

open Nat

-- Definitions of the conditions
noncomputable def large_cube : ℕ := 27
noncomputable def steps_needed_for_all_outer_cubes : ℕ := 26

-- Property stating a termite cannot visit all outer cubes and finish at the central cube
theorem termite_cannot_finish_in_central_cube
  (termite_moves_parallel : ℕ → ℕ → Prop)
  (termite_starts : ℕ → Prop)
  (termite_never_returns : ℕ → ℕ → Prop)
  (termite_moves : ℕ → ℕ → ℕ → Prop)
  (termite_starts_at_outer_cube : Prop) :
  (∃ path, termite_starts (center_of_face path) ∧
    (∀ step < length path, termite_moves (path step) (path (step + 1)) length path) ∧
    (path length path = central_cube large_cube) ∧
    (no_repeats_in_path path)) → False := sorry


end termite_cannot_finish_in_central_cube_l627_627188


namespace eggs_cost_as_rice_l627_627197

def cost_of_eggs_as_rice (cost_rice cost_half_liter cost_4_eggs : ℝ) : ℕ :=
  let cost_one_egg := cost_half_liter / 4 in
  (cost_rice / cost_one_egg).to_nat

theorem eggs_cost_as_rice : cost_of_eggs_as_rice 0.33 0.11 4 = 12 := by
  sorry

end eggs_cost_as_rice_l627_627197


namespace min_elements_good_set_l627_627332

theorem min_elements_good_set (X : Finset ℝ):
  (∀ x ∈ X, ∃ a b ∈ X, a ≠ b ∧ a + b = x) → X.card ≥ 6 :=
by 
  sorry

end min_elements_good_set_l627_627332


namespace quadrant_of_complex_l627_627301

def complex_number_z := (2 - Complex.i) / (1 + Complex.i)
def simplified_z := Complex.mk (1/2) (-3/2)
def quadrant := "Fourth"

theorem quadrant_of_complex (z : Complex) : z = simplified_z → quadrant = "Fourth" :=
by
  intro h
  rw [h]
  sorry

end quadrant_of_complex_l627_627301


namespace product_sign_pos_l627_627168

variable (α : ℝ)

-- Assumption based on the condition provided in the problem.
axiom condition : sec α * sqrt (1 + (tan α)^2) + tan α * sqrt ((csc α)^2 - 1) = (tan α)^2

-- Main statement
theorem product_sign_pos : sin (cos α) * cos (sin α) > 0 :=
by
  -- Use the given condition to conclude the proof.
  sorry

end product_sign_pos_l627_627168


namespace two_digit_numbers_tens_greater_ones_odd_l627_627511

theorem two_digit_numbers_tens_greater_ones_odd :
  ∃ (n : ℕ), n = 20 ∧ ∀ (x : ℕ), 10 ≤ x ∧ x < 100 →
    let tens := x / 10 in
    let ones := x % 10 in
    ones ∈ {1, 3, 5, 7, 9} →
    tens > ones :=
begin
  sorry
end

end two_digit_numbers_tens_greater_ones_odd_l627_627511


namespace minimum_n_for_3_zeros_l627_627016

theorem minimum_n_for_3_zeros :
  ∃ n : ℕ, (∀ m : ℕ, (m < n → ∀ k < 10, m + k ≠ 5 * m ∧ m + k ≠ 5 * m + 25)) ∧
  (∀ k < 10, n + k = 16 ∨ n + k = 16 + 9) ∧
  n = 16 :=
sorry

end minimum_n_for_3_zeros_l627_627016


namespace count_non_congruent_triangles_with_perimeter_18_l627_627934

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627934


namespace monochromatic_triangle_l627_627809

theorem monochromatic_triangle (coloring : ℝ × ℝ → fin 3) :
  ∃ (A B C : ℝ × ℝ), (coloring A = coloring B ∧ coloring B = coloring C) ∧ 
  (isosceles A B C ∨ angles_in_geometric_progression A B C) := 
sorry

end monochromatic_triangle_l627_627809


namespace eval_expression_l627_627429

def floor (x : ℝ) : ℤ := ⌊x⌋
def ceil (x : ℝ) : ℤ := ⌈x⌉

def abs_floor_sub_ceil : ℤ :=
  (Int.natAbs (floor (-5.67))) - (ceil 42.1)

theorem eval_expression : abs_floor_sub_ceil = -37 := by
  sorry

end eval_expression_l627_627429


namespace steak_entree_cost_l627_627562

theorem steak_entree_cost
  (total_guests : ℕ)
  (steak_factor : ℕ)
  (chicken_entree_cost : ℕ)
  (total_budget : ℕ)
  (H1 : total_guests = 80)
  (H2 : steak_factor = 3)
  (H3 : chicken_entree_cost = 18)
  (H4 : total_budget = 1860) :
  ∃ S : ℕ, S = 25 := by
  -- Proof steps omitted
  sorry

end steak_entree_cost_l627_627562


namespace crossing_time_l627_627025

-- Define the conditions
def walking_speed_kmh : Float := 10
def bridge_length_m : Float := 1666.6666666666665

-- Convert the man's walking speed to meters per minute
def walking_speed_mpm : Float := walking_speed_kmh * (1000 / 60)

-- State the theorem we want to prove
theorem crossing_time 
  (ws_kmh : Float := walking_speed_kmh)
  (bl_m : Float := bridge_length_m)
  (ws_mpm : Float := walking_speed_mpm) :
  bl_m / ws_mpm = 10 :=
by
  sorry

end crossing_time_l627_627025


namespace remainder_of_sums_modulo_l627_627445

theorem remainder_of_sums_modulo :
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 :=
by
  sorry

end remainder_of_sums_modulo_l627_627445


namespace unique_triangles_count_l627_627958

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627958


namespace profit_percentage_of_cp_is_75_percent_of_sp_l627_627355

/-- If the cost price (CP) is 75% of the selling price (SP), then the profit percentage is 33.33% -/
theorem profit_percentage_of_cp_is_75_percent_of_sp (SP : ℝ) (h : SP > 0) (CP : ℝ) (hCP : CP = 0.75 * SP) :
  (SP - CP) / CP * 100 = 33.33 :=
by
  sorry

end profit_percentage_of_cp_is_75_percent_of_sp_l627_627355


namespace area_triangle_possible_values_l627_627189

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem area_triangle_possible_values (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = Real.pi / 6) :
  ∃ S, S = 2 * Real.sqrt 3 ∨ S = Real.sqrt 3 :=
by
  -- Define the area using the given values
  sorry

end area_triangle_possible_values_l627_627189


namespace total_seats_180_l627_627047

-- define the conditions
variables (s : ℝ)

def first_class_seats := 36
def business_class_seats := 0.2 * s
def economy_class_seats := (3 / 5) * s

-- prove that the total number of seats is exactly 180
theorem total_seats_180 (h : first_class_seats + business_class_seats + economy_class_seats = s) : s = 180 :=
by sorry

end total_seats_180_l627_627047


namespace isosceles_triangle_l627_627460

variable {A B C P : Type} -- Representing points
variable [TriangleABC : Triangle A B C] -- Assume we're working within the context of a triangle

open Triangle

-- Angles associated with triangle ABC and internal point P
variables (angle_PAB : ℕ) (angle_PBA : ℕ) (angle_PCA : ℕ) (angle_PAC : ℕ)
variables (h_PAB : angle_PAB = 10) (h_PBA : angle_PBA = 20)
variables (h_PCA : angle_PCA = 30) (h_PAC : angle_PAC = 40)

theorem isosceles_triangle (ABC : Triangle A B C) :
  ∃ (α : ℕ), ∠ A P B = angle_PAB ∧ ∠ P B A = angle_PBA ∧ ∠ P C A = angle_PCA ∧ 
             ∠ P A C = angle_PAC → Triangle.is_isosceles ABC :=
by
  sorry

end isosceles_triangle_l627_627460


namespace find_m_find_angle_l627_627931
noncomputable def m : ℝ := 2 / 5

theorem find_m (a : ℝ × ℝ) (c : ℝ × ℝ) (h1 : a = (-1, 2)) (h2 : c = (m - 1, 3 * m)) (h3 : ∀ k : ℝ, c = k • a) : m = 2 / 5 := 
sorry

theorem find_angle (a : ℝ × ℝ) (b : ℝ × ℝ) (h1 : a = (-1, 2)) (h2 : ∥b∥ = real.sqrt(5) / 2) (h3 : dot_product (a + (2 : ℝ) • b) (2 • a - b) = 0) : 
  let θ := real.angle a b in θ = real.pi := 
sorry

end find_m_find_angle_l627_627931


namespace arithmetic_sequence_ratio_l627_627685

/-- 
  Given the ratio of the sum of the first n terms of two arithmetic sequences,
  prove the ratio of the 11th terms of these sequences.
-/
theorem arithmetic_sequence_ratio (S T : ℕ → ℚ) 
  (h : ∀ n, S n / T n = (7 * n + 1 : ℚ) / (4 * n + 2)) : 
  S 21 / T 21 = 74 / 43 :=
sorry

end arithmetic_sequence_ratio_l627_627685


namespace fruit_basket_count_l627_627512

theorem fruit_basket_count :
  ∃ n, n = 62 ∧
       ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ 6 → 0 ≤ b ∧ b ≤ 8 →
       (a > 0 ∨ b > 0) :=
begin
  have total_baskets := 7 * 9,
  have empty_basket := 1,
  use (total_baskets - empty_basket),
  split,
  { norm_num },
  { intros a b ha hb hnonempty,
    sorry
  }
end

end fruit_basket_count_l627_627512


namespace derivative_at_2_l627_627884

/-- Given f(x) = x^2 + 2x f'(0), show that f'(2) = 4. -/
theorem derivative_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*f' 0) : f' 2 = 4 :=
by
  sorry

end derivative_at_2_l627_627884


namespace find_smallest_number_ge_0_l627_627318

theorem find_smallest_number_ge_0.6 :
  let numbers := [0.8, 0.5, 0.9],
      filtered := numbers.filter (λ x, x ≥ 0.6),
      smallest := filtered.min (by decide)
  in smallest = 0.8 :=
by
  sorry

end find_smallest_number_ge_0_l627_627318


namespace cookies_left_l627_627606

-- Define the conditions
def pounds_of_flour_used_per_batch : ℕ := 2
def batches_per_bakery_bag_of_flour : ℕ := 5
def total_bags_used : ℕ := 4
def cookies_per_batch : ℕ := 12
def cookies_eaten_by_jim : ℕ := 15

-- Calculate the total pounds of flour used
def total_pounds_of_flour := total_bags_used * batches_per_bakery_bag_of_flour

-- Calculate the total number of batches
def total_batches := total_pounds_of_flour / pounds_of_flour_used_per_batch

-- Calculate the total number of cookies cooked
def total_cookies := total_batches * cookies_per_batch

-- Calculate the number of cookies left
theorem cookies_left :
  let total_cookies := total_batches * cookies_per_batch in 
  total_cookies - cookies_eaten_by_jim = 105 :=
by
  sorry

end cookies_left_l627_627606


namespace find_k_value_l627_627874

theorem find_k_value (k : ℚ) :
  (∀ x y : ℚ, (x = 1/3 ∧ y = -8 → -3/4 - 3 * k * x = 7 * y)) → k = 55.25 :=
by
  sorry

end find_k_value_l627_627874


namespace digit_A_in_comb_60_15_correct_l627_627545

-- Define the combination function
def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main theorem we want to prove
theorem digit_A_in_comb_60_15_correct : 
  ∃ (A : ℕ), (660 * 10^9 + A * 10^8 + B * 10^7 + 5 * 10^6 + A * 10^4 + 640 * 10^1 + A) = comb 60 15 ∧ A = 6 :=
by
  sorry

end digit_A_in_comb_60_15_correct_l627_627545


namespace linear_function_quadrant_l627_627182

theorem linear_function_quadrant :
  ∀ {k b : ℝ}, (∀ x : ℝ, y = k * x + b) →
  (k ≠ 0) →
  (y_increases_as_x_increases : ∀ x₁ x₂, x₁ < x₂ → k > 0) →
  (passes_through_A : ∃ (b : ℝ), y = 0 ∧ k * -2 = -b) →
  ¬(fourth_quadrant : ∀ x > 0, y < 0) :=
by
  sorry

end linear_function_quadrant_l627_627182


namespace total_pencils_correct_l627_627421

variable (donna_pencils marcia_pencils cindi_pencils : ℕ)

-- Given conditions translated into Lean
def condition1 : Prop := donna_pencils = 3 * marcia_pencils
def condition2 : Prop := marcia_pencils = 2 * cindi_pencils
def condition3 : Prop := cindi_pencils = 30 / 0.5

-- The proof statement
theorem total_pencils_correct : 
  condition1 ∧ condition2 ∧ condition3 → donna_pencils + marcia_pencils = 480 :=
begin
  -- Placeholder for the actual proof
  sorry
end

end total_pencils_correct_l627_627421


namespace probability_divisor_of_8_is_half_l627_627787

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l627_627787


namespace find_q_l627_627682

-- Given polynomial Q(x) with coefficients p, q, d
variables {p q d : ℝ}

-- Define the polynomial Q(x)
def Q (x : ℝ) := x^3 + p * x^2 + q * x + d

-- Assume the conditions of the problem
theorem find_q (h1 : d = 5)                   -- y-intercept is 5
    (h2 : (-p / 3) = -d)                    -- mean of zeros = product of zeros
    (h3 : (-p / 3) = 1 + p + q + d)          -- mean of zeros = sum of coefficients
    : q = -26 := 
    sorry

end find_q_l627_627682


namespace remainder_constant_iff_b_eq_neg_four_thirds_l627_627844

theorem remainder_constant_iff_b_eq_neg_four_thirds 
  (b : ℚ) : 
  let f : ℚ[X] := 12 * X^3 - 9 * X^2 + C b * X + 8
  let g : ℚ[X] := 3 * X^2 - 4 * X + 2
  (∀ r : ℚ[X], r.degree < g.degree → ∃ q : ℚ[X], f = q * g + r → r.degree < g.degree ∧ r = C (r.coeff 0)) 
  → b = -4 / 3 :=
begin
  sorry
end

end remainder_constant_iff_b_eq_neg_four_thirds_l627_627844


namespace factor_quadratic_l627_627073

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 16 * x^2 - 56 * x + 49

-- The goal is to prove that the quadratic expression is equal to (4x - 7)^2
theorem factor_quadratic (x : ℝ) : quadratic_expr x = (4 * x - 7)^2 :=
by
  sorry

end factor_quadratic_l627_627073


namespace net_percentage_change_in_income_l627_627566

-- Definitions and conditions
def initial_income : ℝ := 150
def first_raise : ℝ := 0.15
def wage_cut : ℝ := 0.05
def second_raise : ℝ := 0.20
def initial_car_expense : ℝ := 50
def increased_car_expense : ℝ := 55

-- Calculations based on conditions
def new_income_after_first_raise : ℝ := initial_income + (initial_income * first_raise)
def new_income_after_wage_cut : ℝ := new_income_after_first_raise - (new_income_after_first_raise * wage_cut)
def new_income_after_second_raise : ℝ := new_income_after_wage_cut + (new_income_after_wage_cut * second_raise)
def increased_expense : ℝ := increased_car_expense - initial_car_expense
def net_weekly_income : ℝ := new_income_after_second_raise - increased_expense

-- Initial net weekly income after initial expenses
def initial_net_weekly_income : ℝ := initial_income - initial_car_expense

-- Net percentage change in weekly income
def net_percentage_change : ℝ := ((net_weekly_income - initial_net_weekly_income) / initial_net_weekly_income) * 100

-- Theorem stating the net percentage change in John's weekly income is 91.65%
theorem net_percentage_change_in_income : net_percentage_change = 91.65 := by
  sorry

end net_percentage_change_in_income_l627_627566


namespace intersection_complement_l627_627598

open Set

variable (R : Type) [LinearOrderedField R]

def A : Set R := {x | 0 < x ∧ x < 2}
def B : Set R := {x | x ≤ 1}

theorem intersection_complement :
  A ∩ (compl B) = {x : R | 1 < x ∧ x < 2} := 
by
  sorry

end intersection_complement_l627_627598


namespace count_square_of_integer_fraction_l627_627870

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end count_square_of_integer_fraction_l627_627870


namespace part_I_part_II_l627_627215

-- Definitions for the conditions
variables (a b c : ℝ)
variables (A B C : ℝ) -- Angles in the triangle
variables (triangle_abc : Triangle ℝ) -- Representation of triangle ABC

-- Part I: Prove that given the conditions, we can compute sin A
def sin_A (a c : ℝ) (cos_C : ℝ) (h : 4 * a = sqrt 5 * c) (h_cos : cos_C = 3/5) : Prop :=
  sin A = sqrt(5) / 5

-- Part II: Prove that given b = 11 and the previous result, compute the area of the triangle
def area_of_triangle (a b c : ℝ) (cos_C : ℝ) (sin_A : ℝ) (h_cos : cos_C = 3 / 5) (h_sin_A : sin_A = sqrt(5)/5) (h_b : b = 11) : Prop :=
  (1 / 2) * a * b * (sqrt (1 - cos_C^2)) = 22

-- Theorems to be proven
theorem part_I (a c : ℝ) (cos_C : ℝ) (h : 4 * a = sqrt 5 * c) (h_cos : cos_C = 3/5) : sin_A a c cos_C h h_cos :=
  sorry

theorem part_II (a b c : ℝ) (cos_C : ℝ) (sin_A : ℝ) (h_cos : cos_C = 3 / 5) (h_sin_A : sin_A = sqrt(5)/5) (h_b : b = 11) : area_of_triangle a b c cos_C sin_A h_cos h_sin_A h_b :=
  sorry

end part_I_part_II_l627_627215


namespace fourth_child_pays_13_l627_627875

-- Definitions of the conditions
variables (w x y z : ℝ)

-- The conditions from the problem
def condition1 := w + x + y + z = 60
def condition2 := w = 1/2 * (x + y + z)
def condition3 := x = 1/3 * (w + y + z)
def condition4 := y = 1/4 * (w + x + z)

-- The statement we want to prove
theorem fourth_child_pays_13 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  z = 13 :=
sorry

end fourth_child_pays_13_l627_627875


namespace maximum_profit_is_achieved_at_14_yuan_l627_627027

-- Define the initial conditions
def cost_per_unit : ℕ := 8
def initial_selling_price : ℕ := 10
def initial_selling_quantity : ℕ := 100

-- Define the sales volume decrease per price increase
def decrease_per_yuan_increase : ℕ := 10

-- Define the profit function
def profit (price_increase : ℕ) : ℕ :=
  let new_selling_price := initial_selling_price + price_increase
  let new_selling_quantity := initial_selling_quantity - (decrease_per_yuan_increase * price_increase)
  (new_selling_price - cost_per_unit) * new_selling_quantity

-- Define the statement to be proved
theorem maximum_profit_is_achieved_at_14_yuan :
  ∃ price_increase : ℕ, price_increase = 4 ∧ profit price_increase = profit 4 := by
  sorry

end maximum_profit_is_achieved_at_14_yuan_l627_627027


namespace arithmetic_sequence_ratio_l627_627466

variable {a_n : ℕ → ℤ} {S_n : ℕ → ℤ}
variable (d : ℤ)
variable (a1 a3 a4 : ℤ)
variable (h_geom : a3^2 = a1 * a4)
variable (h_seq : ∀ n, a_n (n+1) = a_n n + d)
variable (h_sum : ∀ n, S_n n = (n * (2 * a1 + (n - 1) * d)) / 2)

theorem arithmetic_sequence_ratio :
  (S_n 3 - S_n 2) / (S_n 5 - S_n 3) = 2 :=
by 
  sorry

end arithmetic_sequence_ratio_l627_627466


namespace find_triplets_l627_627076

theorem find_triplets (x y z : ℕ) (h1 : x ≤ y) (h2 : x^2 + y^2 = 3 * 2016^z + 77) :
  (x, y, z) = (4, 8, 0) ∨ (x, y, z) = (14, 77, 1) ∨ (x, y, z) = (35, 70, 1) :=
  sorry

end find_triplets_l627_627076


namespace question_1_question_2_l627_627472

-- Condition: The coordinates of point P are given by the equations x = -3a - 4, y = 2 + a

-- Question 1: Prove coordinates when P lies on the x-axis
theorem question_1 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hy0 : y = 0) :
  a = -2 ∧ x = 2 ∧ y = 0 :=
sorry

-- Question 2: Prove coordinates when PQ is parallel to the y-axis
theorem question_2 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hx5 : x = 5) :
  a = -3 ∧ x = 5 ∧ y = -1 :=
sorry

end question_1_question_2_l627_627472


namespace asymptote_of_log_shifted_l627_627715

theorem asymptote_of_log_shifted :
  let f := fun x : ℝ => log (x+1) / log 2 + 2
  in ∀ y : ℝ, ∃ ε > 0, ∀ x < -1 + ε, f x < y :=
by
  -- sorry, the rest of the proof goes here.
  sorry

end asymptote_of_log_shifted_l627_627715


namespace correct_calculation_result_l627_627347

theorem correct_calculation_result : 
  ∀ (x : ℤ), (x - 63 = 8) → (x * 8 = 568) :=
by
  intro x h
  have h1 : x = 71 := by linarith
  rw h1
  norm_num

end correct_calculation_result_l627_627347


namespace shopkeeper_discount_l627_627033

theorem shopkeeper_discount :
  let CP := 100
  let SP_with_discount := 119.7
  let SP_without_discount := 126
  let discount := SP_without_discount - SP_with_discount
  let discount_percentage := (discount / SP_without_discount) * 100
  discount_percentage = 5 := sorry

end shopkeeper_discount_l627_627033


namespace treadmill_time_saved_l627_627278

-- Define the problem conditions
def treadmill_usage := 
  ∀ distance_per_day : ℕ, 
  ∀ days : ℕ, 
  ∀ [distance_per_day_per_day = 2], 
  ∀ [days = 3], 
  ∀ speed_mon : ℕ, 
  ∀ speed_wed : ℕ, 
  ∀ speed_fri : ℕ,
  [speed_mon = 5, speed_wed = 3, speed_fri = 4] → 
  let total_time := 
    (2 / 5 : ℚ) + (2 / 3 : ℚ) + (2 / 4 : ℚ) in
  let time_if_walked_4mph :=
    (2 * 3 / 4 : ℚ) in
  let time_saved :=
   total_time - time_if_walked_4mph in
  (time_saved * 60 : ℚ) = 4

-- Assert the theorem
theorem treadmill_time_saved : 
  treadmill_usage :=
sorry

end treadmill_time_saved_l627_627278


namespace transform_triple_to_zero_l627_627125

theorem transform_triple_to_zero 
  (x y z : ℕ) 
  (h : 0 < x ∧ x ≤ y ∧ y ≤ z) : 
  ∃ (n : ℕ) (x' y' z' : ℕ), 
  (∀ k ≤ n, by_cases 
    (λ h1 : x' ≤ y, (2 * x', y - x, z)) 
    (λ h2 : x' ≤ z, (2 * x', y, z - x)) 
    (λ h3 : y' ≤ z, (x, 2 * y, z - y)) && 
    (x' ≤ y' ∧ y' ≤ z') && 
    x' = 0)
:=
  sorry

end transform_triple_to_zero_l627_627125


namespace find_sum_of_smallest_multiples_l627_627520

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l627_627520


namespace even_digits_in_base7_of_315_l627_627442

def is_even_in_base7 (d : ℕ) : Prop :=
  d ∈ {0, 2, 4, 6}

def base10_to_base7 (n : ℕ) : ℕ := 
  if n = 315 then 630 else 0  -- specific to this problem

def count_even_digits_in_base7 (n : ℕ) : ℕ :=
  let digits := [6, 3, 0]  -- list of digits in 630 base-7
  digits.count (λ d, is_even_in_base7 d)

theorem even_digits_in_base7_of_315 : count_even_digits_in_base7 (base10_to_base7 315) = 2 :=
sorry

end even_digits_in_base7_of_315_l627_627442


namespace max_divisions_in_rectangle_l627_627801

theorem max_divisions_in_rectangle (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  a + b - Nat.gcd a b = 72 :=
by
  rw [h1, h2]
  have gcd_ab : Nat.gcd 24 60 = 12 := by sorry
  rw [gcd_ab]
  norm_num
  sorry

end max_divisions_in_rectangle_l627_627801


namespace number_of_non_congruent_triangles_perimeter_18_l627_627948

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627948


namespace sums_to_thirteen_l627_627269

open Finset

theorem sums_to_thirteen 
  (chosen_numbers : Finset ℕ) 
  (h_subset : chosen_numbers ⊆ (Finset.range 13).erase 0) 
  (h_card : chosen_numbers.card = 7) : 
  ∃ x y ∈ chosen_numbers, x ≠ y ∧ x + y = 13 := 
by
  sorry

end sums_to_thirteen_l627_627269


namespace cube_height_when_face_on_table_l627_627774

-- Define the context and conditions
def unitCube (a : ℝ) : Prop :=
  ∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a ∧ 0 ≤ z ∧ z ≤ a

-- The side length of the modified cube
def side_length : ℝ := 2

-- The modified cube with the equilateral face on the table
def modified_cube_height (a : ℝ) : ℝ :=
  a - (Real.sqrt 3 / 3)

theorem cube_height_when_face_on_table :
  unitCube side_length →
  modified_cube_height side_length = 2 - Real.sqrt 3 / 3 :=
by
  sorry

end cube_height_when_face_on_table_l627_627774


namespace sum_sqrt_series_is_71_l627_627583

noncomputable def sum_sqrt_series : ℚ :=
  ∑ n in finset.range(4900 + 1), (1 / real.sqrt (n + real.sqrt (n^2 - 1)))

theorem sum_sqrt_series_is_71 :
  ∃ (a b c : ℕ), 
    a + b * real.sqrt c = sum_sqrt_series ∧
    c ∉ { p^2 | p : ℕ } ∧ -- c is not divisible by the square of any prime
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 71 :=
by
  sorry

end sum_sqrt_series_is_71_l627_627583


namespace janet_savings_l627_627560

def wall1_area := 5 * 8 -- wall 1 area
def wall2_area := 7 * 8 -- wall 2 area
def wall3_area := 6 * 9 -- wall 3 area
def total_area := wall1_area + wall2_area + wall3_area
def tiles_per_square_foot := 4
def total_tiles := total_area * tiles_per_square_foot

def turquoise_tile_cost := 13
def turquoise_labor_cost := 6
def total_cost_turquoise := (total_tiles * turquoise_tile_cost) + (total_area * turquoise_labor_cost)

def purple_tile_cost := 11
def purple_labor_cost := 8
def total_cost_purple := (total_tiles * purple_tile_cost) + (total_area * purple_labor_cost)

def orange_tile_cost := 15
def orange_labor_cost := 5
def total_cost_orange := (total_tiles * orange_tile_cost) + (total_area * orange_labor_cost)

def least_expensive_option := total_cost_purple
def most_expensive_option := total_cost_orange

def savings := most_expensive_option - least_expensive_option

theorem janet_savings : savings = 1950 := by
  sorry

end janet_savings_l627_627560


namespace spiders_in_room_l627_627061

theorem spiders_in_room (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) : total_legs / legs_per_spider = 4 :=
by
  rw [h1, h2]
  norm_num

end spiders_in_room_l627_627061


namespace sin_593_l627_627882

theorem sin_593 (h : Real.sin (37 * Real.pi / 180) = 3/5) : 
  Real.sin (593 * Real.pi / 180) = -3/5 :=
by
sorry

end sin_593_l627_627882


namespace triangle_AME_area_l627_627261

/-- Rectangle ABCD has AB = 10 and BC = 8. Point M is the midpoint of diagonal AC,
and E is on AB such that ME is perpendicular to AC. Prove that the area of
triangle AME is 10. -/
theorem triangle_AME_area
  (A B C D M E : EuclideanGeometry.Point ℝ)
  (h1 : A.distance B = 10)
  (h2 : B.distance C = 8)
  (h3 : M = A.midpoint C)
  (h4 : E ≠ A)
  (h5 : E ≠ B)
  (h6 : ∃ ME : EuclideanGeometry.Line ℝ, ME.orthogonal_to (EuclideanGeometry.line A C) ∧ E ∈ ME ∧ ME ∈ EuclideanGeometry.line_set AB)
  : euclidean_geometry.area_triangle A M E = 10 := sorry

#print axioms triangle_AME_area

end triangle_AME_area_l627_627261


namespace sum_of_digits_of_x_squared_eq_36_l627_627383

noncomputable def base_r_representation_sum (r : ℕ) (x : ℕ) := ∃ (p q : ℕ), 
  r <= 36 ∧
  x = p * (r^3 + r^2) + q * (r + 1) ∧
  2 * q = 5 * p ∧
  ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a ∧
  b = 9 ∧
  a + b + c = 18

theorem sum_of_digits_of_x_squared_eq_36 (r x : ℕ) :
  base_r_representation_sum r x → ∑ d in (digits r (x^2)), d = 36 :=
sorry

end sum_of_digits_of_x_squared_eq_36_l627_627383


namespace inequality_le_one_equality_case_l627_627013

open Real

theorem inequality_le_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) :=
sorry

theorem equality_case (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_le_one_equality_case_l627_627013


namespace ticket_price_reduction_l627_627695

theorem ticket_price_reduction
    (original_price : ℝ := 50)
    (increase_in_tickets : ℝ := 1 / 3)
    (increase_in_revenue : ℝ := 1 / 4)
    (x : ℝ)
    (reduced_price : ℝ)
    (new_tickets : ℝ := x * (1 + increase_in_tickets))
    (original_revenue : ℝ := x * original_price)
    (new_revenue : ℝ := new_tickets * reduced_price) :
    new_revenue = (1 + increase_in_revenue) * original_revenue →
    reduced_price = original_price - (original_price / 2) :=
    sorry

end ticket_price_reduction_l627_627695


namespace concert_attendance_l627_627621

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := first_concert + 119
noncomputable def third_concert : ℕ := first_concert - Nat.round (0.05 * first_concert)
noncomputable def fourth_concert : ℕ := 2 * third_concert

theorem concert_attendance :
  first_concert = 65899 ∧
  second_concert = 66018 ∧
  third_concert = 62604 ∧
  fourth_concert = 125208 :=
by
  unfold first_concert second_concert third_concert fourth_concert
  norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  norm_num

#check concert_attendance  -- To verify that the theorem is correctly stated

end concert_attendance_l627_627621


namespace sum_of_coordinates_of_center_of_circle_l627_627302

theorem sum_of_coordinates_of_center_of_circle 
  {A B: Type} [ordered_field A] [ordered_field B] 
  (x1 y1 x2 y2 : A) 
  (h1 : x1 = 4) (h2 : y1 = -7) (h3 : x2 = -8) (h4 : y2 = 5) 
  : (x1 + x2 + y1 + y2) / 2 = -3 := by
  sorry

end sum_of_coordinates_of_center_of_circle_l627_627302


namespace area_triangle_ABF_proof_area_triangle_AFD_proof_l627_627279

variable (A B C D M F : Type)
variable (area_square : Real) (midpoint_D_CM : Prop) (lies_on_line_BC : Prop)

-- Given conditions
axiom area_ABCD_300 : area_square = 300
axiom M_midpoint_DC : midpoint_D_CM
axiom F_on_line_BC : lies_on_line_BC

-- Define areas for the triangles
def area_triangle_ABF : Real := 300
def area_triangle_AFD : Real := 150

-- Prove that given the conditions, the area of triangle ABF is 300 cm²
theorem area_triangle_ABF_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_ABF = 300 :=
by
  intro h
  sorry

-- Prove that given the conditions, the area of triangle AFD is 150 cm²
theorem area_triangle_AFD_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_AFD = 150 :=
by
  intro h
  sorry

end area_triangle_ABF_proof_area_triangle_AFD_proof_l627_627279


namespace tanya_number_75_less_l627_627277

def rotate180 (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => 0 -- invalid assumption for digits outside the defined scope

def two_digit_upside_down (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * rotate180 units + rotate180 tens

theorem tanya_number_75_less (n : ℕ) : 
  ∀ n, (∃ a b, n = 10 * a + b ∧ (a = 0 ∨ a = 1 ∨ a = 6 ∨ a = 8 ∨ a = 9) ∧ 
      (b = 0 ∨ b = 1 ∨ b = 6 ∨ b = 8 ∨ b = 9) ∧  
      n - two_digit_upside_down n = 75) :=
by {
  sorry
}

end tanya_number_75_less_l627_627277


namespace nth_inequality_l627_627248

theorem nth_inequality (n : ℕ) (h_pos : n > 0):
  ∑ k in Finset.range (n + 1), real.sqrt ((k+1) * (k+2)) < (n * (n + 2)) / 2 := sorry

end nth_inequality_l627_627248


namespace russia_is_one_third_bigger_l627_627823

theorem russia_is_one_third_bigger (U : ℝ) (Canada Russia : ℝ) 
  (h1 : Canada = 1.5 * U) (h2 : Russia = 2 * U) : 
  (Russia - Canada) / Canada = 1 / 3 :=
by
  sorry

end russia_is_one_third_bigger_l627_627823


namespace find_ns_l627_627434

def intSqrt (n : ℕ) : ℕ := nat.sqrt n

def divisible (d m : ℕ) : Prop := ∃ k : ℕ, m = k * d

theorem find_ns 
  (n : ℕ) 
  (h1 : divisible ((intSqrt n) - 2) (n - 4))
  (h2 : divisible ((intSqrt n) + 2) (n + 4)) : 
  n = 2 ∨ 
  n = 4 ∨ 
  n = 11 ∨ 
  n = 20 ∨ 
  n = 31 ∨ 
  n = 36 ∨ 
  n = 44 ∨ 
  (∃ a : ℕ, a > 2 ∧ n = a^2 + 2*a - 4) := 
sorry

end find_ns_l627_627434


namespace triangle_with_ap_angles_l627_627068

theorem triangle_with_ap_angles :
  ∃! (a b c : ℕ), ∃ (d : ℕ), 
  (a + b + c = 180) ∧ 
  (a = b - d) ∧ 
  (b = c - d) ∧ 
  (a < b ∧ b < c) ∧ 
  (a = 0 ∨ a > 0) :=
begin
  sorry
end

end triangle_with_ap_angles_l627_627068


namespace friend_p_distance_when_meet_l627_627007

-- Definitions:
variables {v d : ℝ}

-- Condition 1: Two friends start walking at opposite ends of a 43 km trail at the same time.
-- Condition 2: Friend P's rate is 15% faster than friend Q's rate.
def friend_q_rate := v
def friend_p_rate := 1.15 * v
def distance_walked_by_q_when_they_meet := d
def distance_walked_by_p_when_they_meet := 43 - d

theorem friend_p_distance_when_meet (h1 : v > 0) (h2 : d / v = (43 - d) / (1.15 * v)) :
  distance_walked_by_p_when_they_meet = 23 :=
by
  sorry

end friend_p_distance_when_meet_l627_627007


namespace non_congruent_triangles_count_l627_627965

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627965


namespace non_congruent_triangles_with_perimeter_18_l627_627988

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627988


namespace lcm_36_65_l627_627094

-- Definitions based on conditions
def number1 : ℕ := 36
def number2 : ℕ := 65

-- The prime factorization conditions can be implied through deriving LCM hence added as comments to clarify the conditions.
-- 36 = 2^2 * 3^2
-- 65 = 5 * 13

-- Theorem statement that the LCM of number1 and number2 is 2340
theorem lcm_36_65 : Nat.lcm number1 number2 = 2340 := 
by 
  sorry

end lcm_36_65_l627_627094


namespace operational_cost_is_34_l627_627348

-- Define the given conditions as constants
constant lemonade_revenue : ℕ := 47
constant babysitting_revenue : ℕ := 31
constant total_profit : ℕ := 44

-- Define the total gross revenue and operational cost calculation
def total_gross_revenue : ℕ := lemonade_revenue + babysitting_revenue
def operational_cost : ℕ := total_gross_revenue - total_profit

-- State the theorem to prove
theorem operational_cost_is_34 : operational_cost = 34 := by
  -- proof goes here
  sorry

end operational_cost_is_34_l627_627348


namespace non_congruent_triangles_count_l627_627966

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627966


namespace number_of_teams_l627_627319

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 * 10 = 1050) : n = 15 :=
by 
  sorry

end number_of_teams_l627_627319


namespace fourier_series_expansion_correct_l627_627436

def f (x : ℝ) := (x - 4) ^ 2

noncomputable def fourier_series_expansion : ℝ → ℝ :=
  λ x, 16/3 + 64/Real.pi^2 * ∑' m : ℕ in {m // m > 0}, (1 / m^2) * Real.cos (m * Real.pi * x / 4)

theorem fourier_series_expansion_correct :
  ∀ x ∈ Set.Ico 0 4, 
    has_sum (λ m : ℕ, (if m = 0 then (32/3) else (64 / (m^2 * π^2)) * cos (Real.pi * (m:int) * x / 4))) (f x) := 
sorry

end fourier_series_expansion_correct_l627_627436


namespace find_x_l627_627419

theorem find_x (x : ℝ) : (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) → x = -5 :=
by
  sorry

end find_x_l627_627419


namespace median_first_twenty_integers_l627_627721

noncomputable def median (xs : List ℕ) : ℚ :=
if h : xs.length % 2 = 1 then
  xs.xs.nth_le (xs.length / 2) sorry
else
  (xs.xs.nth_le (xs.length / 2 - 1) sorry + xs.xs.nth_le (xs.length / 2) sorry) / 2

theorem median_first_twenty_integers :
  median ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] : List ℕ) = 10.5 :=
sorry

end median_first_twenty_integers_l627_627721


namespace diameter_of_cylinder_l627_627667

noncomputable def diameter_of_tin (h: ℝ) (V: ℝ) : ℝ :=
  2 * real.sqrt (V / (real.pi * h))

theorem diameter_of_cylinder (h : ℝ) (V : ℝ) (d : ℝ) 
  (h_eq : h = 5) (V_eq : V = 245) (d_eq : d = diameter_of_tin h V) : 
  d ≈ 7.894 := 
begin
  -- Placeholder for proof
  sorry
end

end diameter_of_cylinder_l627_627667


namespace average_of_remaining_two_numbers_l627_627280

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 2.5)
  (h2 : (a + b) / 2 = 1.1)
  (h3 : (c + d) / 2 = 1.4) : 
  (e + f) / 2 = 5 :=
by
  sorry

end average_of_remaining_two_numbers_l627_627280


namespace monotonicity_and_extrema_l627_627912

open Function

def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem monotonicity_and_extrema :
  (∀ (x₁ x₂ : ℝ), 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ < f x₂) ∧
  (∃ x_min x_max,
     (3 ≤ x_min ∧ x_min ≤ 5 ∧ f x_min = 5 / 4) ∧
     (3 ≤ x_max ∧ x_max ≤ 5 ∧ f x_max = 3 / 2)) :=
by
  sorry

end monotonicity_and_extrema_l627_627912


namespace median_first_twenty_integers_l627_627722

noncomputable def median (xs : List ℕ) : ℚ :=
if h : xs.length % 2 = 1 then
  xs.xs.nth_le (xs.length / 2) sorry
else
  (xs.xs.nth_le (xs.length / 2 - 1) sorry + xs.xs.nth_le (xs.length / 2) sorry) / 2

theorem median_first_twenty_integers :
  median ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] : List ℕ) = 10.5 :=
sorry

end median_first_twenty_integers_l627_627722


namespace sin_cos_eq_neg_cos_sin_solution_l627_627635

theorem sin_cos_eq_neg_cos_sin_solution (x : Real) (hx : sin (4 * x) * cos (5 * x) = -cos (4 * x) * sin (5 * x)) : x = 20 * Real.pi / 180 :=
by
  sorry

end sin_cos_eq_neg_cos_sin_solution_l627_627635


namespace largest_c_in_range_of_f_l627_627437

theorem largest_c_in_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 - 6 * x + c = 2) -> c ≤ 11 :=
by
  sorry

end largest_c_in_range_of_f_l627_627437


namespace angle_B_acute_l627_627259

theorem angle_B_acute (A B C : Type) [triangle A B C] (angle_C : ∠ABC = 90) : ∠BAC < 90 := by
  sorry

end angle_B_acute_l627_627259


namespace non_congruent_triangles_count_l627_627962

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627962


namespace solve_for_A_l627_627227

def f (A B x : ℝ) : ℝ := A * x ^ 2 - 3 * B ^ 3
def g (B x : ℝ) : ℝ := 2 * B * x + B ^ 2

theorem solve_for_A (B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) :
  A = 3 / (16 / B + 8 + B ^ 3) :=
by
  sorry

end solve_for_A_l627_627227


namespace meet_nearer_to_R_40_l627_627714

noncomputable def distance_nearer_to_R (R S: ℝ) (d_RS: ℝ) (v_R: ℝ) (v_S_init: ℝ) (v_S_doubling: ℕ → ℝ): ℝ :=
  let t := 4 
  let distance_R := v_R * t
  let distance_S := v_S_init * (2^t - 1)
  (distance_R + distance_S) - (distance_S - distance_R)

theorem meet_nearer_to_R_40 :
  let R := 0
  let S := 100
  let d_RS := 100
  let v_R := 5
  let v_S_init := 4
  let v_S_doubling := (λ (n : ℕ), 4 * 2 ^ n)
  distance_nearer_to_R R S d_RS v_R v_S_init v_S_doubling = 40 :=
by
  sorry

end meet_nearer_to_R_40_l627_627714


namespace triangle_right_angle_l627_627214

theorem triangle_right_angle (A B C : Type) (α : ℝ) 
  (h1 : ∠BAC = α) 
  (h2 : ∠ACB = 2 * α) 
  (h3 : AC = 2 * BC) :
  ∠ABC = 90 :=
by
  sorry

end triangle_right_angle_l627_627214


namespace directional_derivative_of_u_l627_627848

variables (x y z : ℝ) (r : ℝ)
def u : ℝ := 1 / r^2

theorem directional_derivative_of_u :
  r^2 = x^2 + y^2 + z^2 → ∥∇ (λ v, u (v 0) (v 1) (v 2))∥ = 4 / r^2 :=
by
  sorry

end directional_derivative_of_u_l627_627848


namespace exercise_l627_627673

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1
axiom h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → f x + f y = f (f x + y)

theorem exercise : ∀ x, 0 ≤ x → x ≤ 1 → f (f x) = f x := 
by 
  sorry

end exercise_l627_627673


namespace equal_probability_of_selection_l627_627704

theorem equal_probability_of_selection :
  ∀ (faculty_size selected_num eliminated_num remaining_num : ℕ),
    faculty_size = 118 →
    selected_num = 16 →
    eliminated_num = 6 →
    remaining_num = faculty_size - eliminated_num →
    remaining_num = 112 →
    (∀ teacher : ℕ, teacher < remaining_num → 
      (prob_of_selection : ℚ) = (selected_num : ℚ) / (remaining_num : ℚ) → 
      prob_of_selection = 1 / 7) := 
begin
  -- Proof goes here
  sorry
end

end equal_probability_of_selection_l627_627704


namespace product_of_solutions_abs_eq_l627_627554

theorem product_of_solutions_abs_eq (x : ℝ) : 
  (|x - 4| - 5 = -3) → (∃ x₁ x₂, |x₁ - 4| = 2 ∧ |x₂ - 4| = 2 ∧ x₁ * x₂ = 12) :=
by
  sorry

end product_of_solutions_abs_eq_l627_627554


namespace hours_of_overtime_worked_l627_627352

theorem hours_of_overtime_worked (regular_pay : ℝ) (hours_regular : ℝ) (total_pay : ℝ) (overtime_rate_multiplier : ℝ) (overtime_hours : ℝ) :
  regular_pay = 3 → 
  hours_regular = 40 →
  total_pay = 168 →
  overtime_rate_multiplier = 2 →
  overtime_hours = (total_pay - regular_pay * hours_regular) / (regular_pay * overtime_rate_multiplier) →
  overtime_hours = 8 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5.symm.trans 
    (by norm_num)

end hours_of_overtime_worked_l627_627352


namespace monthly_cost_per_person_is_1000_l627_627565

noncomputable def john_pays : ℝ := 32000
noncomputable def initial_fee_per_person : ℝ := 4000
noncomputable def total_people : ℝ := 4
noncomputable def john_pays_half : Prop := true

theorem monthly_cost_per_person_is_1000 :
  john_pays_half →
  (john_pays * 2 - (initial_fee_per_person * total_people)) / (total_people * 12) = 1000 :=
by
  intro h
  sorry

end monthly_cost_per_person_is_1000_l627_627565


namespace count_non_congruent_triangles_with_perimeter_18_l627_627936

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627936


namespace partA_partB_partC_partD_l627_627708

variable (α β : ℝ)
variable (hα : 0 < α) (hα1 : α < 1)
variable (hβ : 0 < β) (hβ1 : β < 1)

theorem partA : 
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 := by
  sorry

theorem partB :
  β * (1 - β)^2 = β * (1 - β)^2 := by
  sorry

theorem partC :
  β * (1 - β)^2 + (1 - β)^3 = β * (1 - β)^2 + (1 - β)^3 := by
  sorry

theorem partD (hα0 : α < 0.5) :
  (1 - α) * (α - α^2) < (1 - α) := by
  sorry

end partA_partB_partC_partD_l627_627708


namespace probability_of_shaded_l627_627024

-- Defining the regions as a finite set of four regions
inductive Region
| r1 | r2 | r3 | r4

-- Defining the shape being a square divided into four regions by its diagonals
def square_divided_into_regions : Type :=
  { regions : Finset Region // regions.card = 4 }

-- Defining the shaded regions
def shaded_regions (s : square_divided_into_regions) : Finset Region :=
  {r1, r2, r3}

-- Defining the probability measure on the regions
def probability (s : square_divided_into_regions) (shaded : Finset Region) : ℚ :=
  shaded.card / s.regions.card

-- The main theorem that needs to be proven
theorem probability_of_shaded (s : square_divided_into_regions)
  (h_shaded : shaded_regions s.card = 3) :
  probability s (shaded_regions s) = 3/4 := by
  sorry

end probability_of_shaded_l627_627024


namespace max_pressure_theorem_l627_627630

variables (R V₀ T₀ a b c : ℝ)
variable (c2_lt_a2_b2 : c^2 < a^2 + b^2)

def max_pressure :=
  R * T₀ / V₀ * ((a * (a^2 + b^2 - c^2).sqrt + b * c) / (b * (a^2 + b^2 - c^2).sqrt - a * c))

theorem max_pressure_theorem :
  ∃ P_max, P_max = max_pressure R V₀ T₀ a b c ∧
    ∀ (V T : ℝ), 
    (V / V₀ - a)^2 + (T / T₀ - b)^2 = c^2 → 
    ∀ (P : ℝ), 
    P = R * T / V → 
    P ≤ max_pressure R V₀ T₀ a b c :=
sorry

end max_pressure_theorem_l627_627630


namespace sufficient_but_not_necessary_for_q_l627_627503

variables {a b : Line} {α β : Plane}

-- Conditions given in the problem
axiom lines_are_different : a ≠ b
axiom planes_are_parallel : α ∥ β
axiom line_perpendicular_to_plane : a ⊥ α

-- Propositions
def p := b ∥ β
def q := a ⊥ b

-- The proof statement
theorem sufficient_but_not_necessary_for_q (a b : Line) (α β : Plane) 
  (lines_are_different : a ≠ b)
  (planes_are_parallel : α ∥ β)
  (line_perpendicular_to_plane : a ⊥ α)
  (p : b ∥ β := by sorry)
  (q : a ⊥ b := by sorry) : 
  ¬(p ∧ q) :=
-- The actual detailed proof would go here
sorry

end sufficient_but_not_necessary_for_q_l627_627503


namespace inspector_rejects_8_on_tuesday_l627_627813

open Real

theorem inspector_rejects_8_on_tuesday :
  (defective_rate : ℝ) (monday_rejected : ℕ) (monday_meters : ℝ) (tuesday_meters : ℝ) (tuesday_rejected : ℕ) :
  defective_rate = 0.0007 →
  monday_rejected = 7 →
  monday_meters = 7 / defective_rate →
  tuesday_meters = 1.25 * monday_meters →
  tuesday_rejected = floor (defective_rate * tuesday_meters) →
  tuesday_rejected = 8 :=
begin
  intros,
  sorry
end

end inspector_rejects_8_on_tuesday_l627_627813


namespace triangle_pyramid_cone_l627_627314

noncomputable def angle_between_planes (α β γ x : ℝ) : ℝ :=
  (π/2) - x

theorem triangle_pyramid_cone
  (S A B C O : Type)
  (α β γ x : ℝ)
  (h1 : S = O)
  (h2 : A ∈ circular_base O)
  (h3 : B ∈ circular_base O)
  (h4 : C ∈ circular_base O)
  (h5 : dihedral_angle S A = α)
  (h6 : dihedral_angle S B = β)
  (h7 : dihedral_angle S C = γ)
  (h8 : α = y + z)
  (h9 : β = z + x)
  (h10 : γ = x + y)
  (h11 : plane_perpendicular (plane SCO) (tangent_plane SC)) :
  angle_between_planes α β γ x = (π/2) - x :=
sorry

end triangle_pyramid_cone_l627_627314


namespace min_area_of_triangle_l627_627901

def ellipse : Type := ℝ × ℝ

def ellipse_eq (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

noncomputable def focal_length (c : ℝ) : Prop :=
  2 * c = 2

noncomputable def point_on_ellipse (a b : ℝ) : ellipse := (1, real.sqrt 2 / 2)

noncomputable def tangent_line (a b x2 y2 x y : ℝ) : Prop :=
  (x2 / a ^ 2) * x + (y2 / b ^ 2) * y = 1

noncomputable def triangle_area (y2 x2 : ℝ) : ℝ :=
  1 / 2 * (1 / y2) * (2 / x2)

theorem min_area_of_triangle
  (a b c x2 y2 : ℝ) 
  (h1 : ellipse_eq a b 1 (real.sqrt 2 / 2))
  (h2 : focal_length c)
  (h3 : b = 1)
  (h4 : a ^ 2 = 2)
  (h5 : tangent_line a b x2 y2 0 (1 / y2))
  (h6 : tangent_line a b x2 y2 (2 / x2) 0)
  (h7 : x2 > 0)
  (h8 : y2 > 0)
  (h9 : (x2 ^ 2) / 2 + y2 ^ 2 = 1)
  : triangle_area y2 x2 >= real.sqrt 2 :=
sorry

end min_area_of_triangle_l627_627901


namespace unique_triangles_count_l627_627953

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627953


namespace non_congruent_triangles_with_perimeter_18_l627_627973

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627973


namespace sequence_sum_geq_n_l627_627925

theorem sequence_sum_geq_n (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_condition : ∀ k : ℕ, 0 < k → a (k + 1) ≥ (k * a k) / (a k ^ 2 + k - 1)) :
  ∀ n : ℕ, 2 ≤ n → ∑ i in Finset.range n, a (i + 1) ≥ n := 
by
  intros n hn
  sorry

end sequence_sum_geq_n_l627_627925


namespace non_congruent_triangles_with_perimeter_18_l627_627980

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627980


namespace intersection_P_Q_l627_627222

open Set

def P := { x : ℝ | -2 ≤ x ∧ x ≤ 2 }
def Q := {1, 2, 3, 4} : Set ℝ

theorem intersection_P_Q :
  P ∩ Q = {1, 2} := by
  sorry

end intersection_P_Q_l627_627222


namespace discount_is_25_percent_l627_627284

-- Define the cost of one photocopy
def cost_per_copy := 0.02

-- Define the number of copies made by each person
def copies_per_person := 80

-- Define the total number of copies in the single order
def total_copies := copies_per_person * 2

-- Define the savings for each person
def savings_per_person := 0.40

-- Define the total savings
def total_savings := savings_per_person * 2

-- Define the total cost without discount
def total_cost_without_discount := total_copies * cost_per_copy

-- Define the total cost with discount
def total_cost_with_discount := total_cost_without_discount - total_savings

-- Define the discount percentage calculation
def discount_percentage := ((total_cost_without_discount - total_cost_with_discount) / total_cost_without_discount) * 100

-- Proof statement
theorem discount_is_25_percent : discount_percentage = 25 := by
  -- Proof left as an exercise; apply all the given conditions
  sorry

end discount_is_25_percent_l627_627284


namespace non_congruent_triangles_with_perimeter_18_l627_627969

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627969


namespace probability_of_pq_6p_3q_eq_3_l627_627276

theorem probability_of_pq_6p_3q_eq_3 :
  let S := { p : ℤ // 1 ≤ p ∧ p ≤ 20 ∧ ∃ q : ℤ, p * q - 6 * p - 3 * q = 3} in
  (↑(Set.card S) : ℚ) / 20 = 1 / 5 :=
by
  sorry

end probability_of_pq_6p_3q_eq_3_l627_627276


namespace same_gender_probability_l627_627267

-- Define the conditions as types and constants
def SchoolA : Type := {a : String // a = "A" ∨ a = "B" ∨ a = "1"}
def SchoolB : Type := {b : String // b = "C" ∨ b = "2" ∨ b = "3"}

def MasculineA (teacher : SchoolA) : Prop := teacher.val = "A" ∨ teacher.val = "B"
def FeminineA (teacher : SchoolA) : Prop := teacher.val = "1"

def MasculineB (teacher : SchoolB) : Prop := teacher.val = "C"
def FeminineB (teacher : SchoolB) : Prop := teacher.val = "2" ∨ teacher.val = "3"

-- Define a function to count the favorable pairs
def count_same_gender_pairs : ℕ := 
  if h: (MasculineA ⊓ MasculineB).exists ∧ (FeminineA ⊓ FeminineB).exists then 4 else 0

-- Definition of total pairs
def total_pairs : ℕ := 9

-- Definition of probability as a fraction
def probability : ℚ := (count_same_gender_pairs : ℚ) / (total_pairs : ℚ)

-- The theorem to state
theorem same_gender_probability : probability = 4 / 9 := sorry

end same_gender_probability_l627_627267


namespace closed_convex_contains_two_points_distance_two_l627_627257

theorem closed_convex_contains_two_points_distance_two 
  (C : set ℝ²) 
  [convex C] 
  [is_closed C] 
  (hC : ∀ x ∈ C, ∃ y ∈ C, ∥x - y∥ = 2) 
  : ∃ A B : ℝ², A ≠ B ∧ A ∈ C ∧ B ∈ C ∧ dist A B = 2 := 
sorry

end closed_convex_contains_two_points_distance_two_l627_627257


namespace average_growth_rate_in_may_and_june_l627_627768

-- Define the conditions
def production_in_april : ℕ := 500
def total_production_q2 : ℕ := 1820
def growth_rate := ℝ

-- Define the target equation derived from the conditions
def target_equation (x : growth_rate) : Prop :=
  production_in_april 
  + production_in_april * (1 + x)
  + production_in_april * (1 + x) ^ 2 
  = total_production_q2

-- Define the proof problem statement
theorem average_growth_rate_in_may_and_june : 
  ∃ (x : growth_rate), target_equation x ∧ x = 0.2 := 
begin
  sorry
end

end average_growth_rate_in_may_and_june_l627_627768


namespace domain_of_f_solve_inequality_f_gt_0_l627_627148

open Real

noncomputable def f (x : ℝ) : ℝ := logb 2 (1 + x) - logb 2 (1 - x)

theorem domain_of_f : ∀ x : ℝ, (f x = logb 2 (1 + x) - logb 2 (1 - x)) → (-1 < x < 1) :=
by sorry

theorem solve_inequality_f_gt_0 : ∀ x : ℝ, (f x = logb 2 (1 + x) - logb 2 (1 - x)) → f x > 0 → (0 < x < 1) :=
by sorry

end domain_of_f_solve_inequality_f_gt_0_l627_627148


namespace age_sum_l627_627014

theorem age_sum (P Q : ℕ) (h1 : P - 12 = (1 / 2 : ℚ) * (Q - 12)) (h2 : (P : ℚ) / Q = (3 / 4 : ℚ)) : P + Q = 42 :=
sorry

end age_sum_l627_627014


namespace probability_divisor_of_8_is_half_l627_627789

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l627_627789


namespace decreasing_power_function_has_specific_m_l627_627107

theorem decreasing_power_function_has_specific_m (m : ℝ) (x : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → 
  m = 2 :=
by
  sorry

end decreasing_power_function_has_specific_m_l627_627107


namespace brad_reads_more_pages_l627_627506

-- Definitions based on conditions
def greg_pages_per_day : ℕ := 18
def brad_pages_per_day : ℕ := 26

-- Statement to prove
theorem brad_reads_more_pages : brad_pages_per_day - greg_pages_per_day = 8 :=
by
  -- sorry is used here to indicate the absence of a proof
  sorry

end brad_reads_more_pages_l627_627506


namespace no_integer_solution_n_squared_eq_7955_3_l627_627377

theorem no_integer_solution_n_squared_eq_7955_3 : ¬∃ (n : ℤ), n ^ 2 = 7955.3 := 
  sorry

end no_integer_solution_n_squared_eq_7955_3_l627_627377


namespace number_of_square_integers_l627_627865

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end number_of_square_integers_l627_627865


namespace find_b_l627_627142

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if -1 < x ∧ x < 2 then x^2
  else 2 * x

theorem find_b (b : ℝ) :
  piecewise_function b = 1 / 2 ↔ 
  b = -3 / 2 ∨ b = sqrt 2 / 2 ∨ b = -sqrt 2 / 2 :=
by
  sorry

end find_b_l627_627142


namespace six_applications_of_f_l627_627524

def f (x : ℝ) : ℝ := -1 / x

theorem six_applications_of_f (x : ℝ) : f (f (f (f (f (f x))))) = x :=
by sorry

example : f (f (f (f (f (f 4))))) = 4 :=
by exact six_applications_of_f 4

end six_applications_of_f_l627_627524


namespace lcm_3_15_is_15_l627_627114

theorem lcm_3_15_is_15 : Nat.lcm 3 15 = 15 :=
sorry

end lcm_3_15_is_15_l627_627114


namespace remainder_of_9876543210_mod_140_l627_627446

theorem remainder_of_9876543210_mod_140 :
  let N := 9876543210 in
  (N % 4 = 2) ∧ (N % 5 = 0) ∧ (N % 7 = 3) → N % 140 = 70 :=
by
  sorry

end remainder_of_9876543210_mod_140_l627_627446


namespace median_room_number_l627_627056

open Finset

theorem median_room_number :
  let rooms : Finset ℕ := (range 25).erase 15 |> erase 20
  ∃ median : ℕ, median = 12 ∧ median ∈ rooms :=
by 
  let rooms : Finset ℕ := (range 25).erase 15 |> erase 20
  -- given that rooms now contain 1 to 14, 16 to 19, 21 to 25
  have : rooms.card = 23,
  { 
    -- ([0, 1, 2, ..., 24] \ {15, 20}) = 25 - 2 = 23
    rw [card_erase_of_mem, card_erase_of_mem, card_range 25], 
    { exact nat.lt_succ_self 25, },
    { exact mem_range.mpr (nat.le_succ 24), },
    { exact nat.lt_succ_self 25, },
  },
  -- find the 12th element of this ordered set, which should have 23-1 elements before it 
  -- and 23-12 elements after it
  use ((rooms.to_list).nth_le 11 sorry),
  split,
  { refl, },
  {
    suffices H : (rooms.to_list).sorted (<),
    { exact nth_le_mem _ H _ sorry, },
    -- sort the list of room numbers: [1, 2, ..., 14, 16, ..., 19, 21, ..., 25]
    sorry, 
  }

end median_room_number_l627_627056


namespace square_area_less_than_circle_area_l627_627324

theorem square_area_less_than_circle_area (a : ℝ) (ha : 0 < a) :
    let S1 := (a / 4) ^ 2
    let r := a / (2 * Real.pi)
    let S2 := Real.pi * r^2
    (S1 < S2) := by
sorry

end square_area_less_than_circle_area_l627_627324


namespace exponent_calculation_l627_627817

-- Define the exponents and their sum
def exponent_sum : ℝ := 1.3 + 0.2 + 0.5 - 0.2 + 0.7

-- Define the left-hand side of the equation using the sum of exponents
def lhs : ℝ := 2 ^ exponent_sum

-- Define the right-hand side of the equation
def rhs : ℝ := 4 * Real.sqrt 2

-- The theorem stating the equality to be proved
theorem exponent_calculation : lhs = rhs :=
by sorry -- Proof will be provided later

end exponent_calculation_l627_627817


namespace ellipse_foci_distance_l627_627408

noncomputable def distance_between_foci : ℝ :=
  let F1 := (4, -3) in
  let F2 := (-6, 9) in
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)

theorem ellipse_foci_distance :
  (distance_between_foci = 2 * Real.sqrt 61) :=
sorry

end ellipse_foci_distance_l627_627408


namespace mark_initial_money_l627_627602

theorem mark_initial_money (X : ℝ) 
  (h1 : X = (1/2) * X + 14 + (1/3) * X + 16) : X = 180 := 
  by
  sorry

end mark_initial_money_l627_627602


namespace tan_sum_trig_identity_l627_627201

variable {A B C : ℝ} -- Angles
variable {a b c : ℝ} -- Sides opposite to angles A, B and C

-- Acute triangle implies A, B, C are all less than π/2 and greater than 0
variable (hAcute : 0 < A ∧ A < pi / 2 ∧ 0 < B ∧ B < pi / 2 ∧ 0 < C ∧ C < pi / 2)

-- Given condition in the problem
variable (hCondition : b / a + a / b = 6 * Real.cos C)

theorem tan_sum_trig_identity : 
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 :=
sorry

end tan_sum_trig_identity_l627_627201


namespace melinda_physics_textbooks_probability_melinda_physics_textbooks_m_n_sum_l627_627247

noncomputable def probability_all_physics_same_box (total_books physics_books : ℕ) (box1_capacity box2_capacity box3_capacity : ℕ) : ℚ :=
  let numerator := 
    (nat.choose (total_books - physics_books) (box1_capacity - physics_books)) +
    (nat.choose (total_books - physics_books) (box2_capacity - physics_books + 1) * (nat.choose (total_books - box1_capacity) (box1_capacity))) +
    (nat.choose (total_books - physics_books) (box3_capacity - physics_books + 2) * (nat.choose (total_books - box2_capacity) (box2_capacity)))
  let denominator := 
    (nat.choose total_books box1_capacity) * 
    (nat.choose (total_books - box1_capacity) box2_capacity)
  numerator / denominator

theorem melinda_physics_textbooks_probability :
  probability_all_physics_same_box 15 4 4 5 6 = 1 / 65 :=
by
  sorry

theorem melinda_physics_textbooks_m_n_sum :
  let m := 1
  let n := 65
  m + n = 66 :=
by
  trivial

end melinda_physics_textbooks_probability_melinda_physics_textbooks_m_n_sum_l627_627247


namespace simson_line_rotates_half_arc_angle_l627_627255

theorem simson_line_rotates_half_arc_angle {A B C P : Point}
  (h_circumcircle : P ∈ circumcircle A B C) :
  ∃ (θ : ℝ), (θ = arc_angle_traveled_by P / 2) → 
  rotates_by θ (Simson_line P A B C) :=
sorry

end simson_line_rotates_half_arc_angle_l627_627255


namespace find_c_plus_d_l627_627518

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l627_627518


namespace f_B_monotonic_f_D_monotonic_l627_627534

-- Define function f_B(x) = e^x / x
def f_B (x : ℝ) : ℝ := (Real.exp x) / x

-- Define function f_D(x) = x * (ln x - 1)
def f_D (x : ℝ) : ℝ := x * (Real.log x - 1)

-- Prove that f_B(x) is monotonically increasing on the interval (1, +∞)
theorem f_B_monotonic (x : ℝ) (h : x > 1) : MonotoneOn f_B (Set.Ioi 1) :=
sorry

-- Prove that f_D(x) is monotonically increasing on the interval (1, +∞)
theorem f_D_monotonic (x : ℝ) (h : x > 1) : MonotoneOn f_D (Set.Ioi 1) :=
sorry

end f_B_monotonic_f_D_monotonic_l627_627534


namespace max_intersections_three_circles_two_lines_l627_627335

noncomputable def max_intersections_3_circles_2_lines : ℕ :=
  3 * 2 * 1 + 2 * 3 * 2 + 1

theorem max_intersections_three_circles_two_lines :
  max_intersections_3_circles_2_lines = 19 :=
by
  sorry

end max_intersections_three_circles_two_lines_l627_627335


namespace points_on_circle_d_8cm_away_from_q_l627_627256

-- Define the data for the problem
structure Circle (α : Type) :=
(center : α)
(radius : ℝ)

def point : Type := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the circles (D and the circle centered at Q with radius 8 cm)
def CircleD : Circle point := ⟨(0, 0), 5⟩ -- Assume center of D at (0, 0) for simplicity

def Q : point := (10, 0) -- Assume point Q is at (10, 0)

-- Define the theorem to prove
theorem points_on_circle_d_8cm_away_from_q :
  ∃! (P : point), dist P CircleD.center = CircleD.radius ∧ dist P Q = 8 :=
by
  sorry

end points_on_circle_d_8cm_away_from_q_l627_627256


namespace larry_correct_evaluation_l627_627245

theorem larry_correct_evaluation (a b c d e : ℝ) 
(Ha : a = 5) (Hb : b = 3) (Hc : c = 6) (Hd : d = 4) :
a - b + c + d - e = a - (b - (c + (d - e))) → e = 0 :=
by
  -- Not providing the actual proof
  sorry

end larry_correct_evaluation_l627_627245


namespace recurrent_sequence_solution_l627_627153

theorem recurrent_sequence_solution (a : ℕ → ℕ) : 
  (a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + 2^n) →
  (∀ n, n ≥ 1 → a n = (2 * n - 1) * 2^(n - 1)) :=
by
  sorry

end recurrent_sequence_solution_l627_627153


namespace num_integers_sq_condition_l627_627868

theorem num_integers_sq_condition : 
  {n : ℤ | n < 30 ∧ (∃ k : ℤ, k ^ 2 = n / (30 - n))}.to_finset.card = 3 := 
by
  sorry

end num_integers_sq_condition_l627_627868


namespace number_of_square_integers_l627_627866

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end number_of_square_integers_l627_627866


namespace number_of_square_integers_l627_627864

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end number_of_square_integers_l627_627864


namespace general_term_a_sum_b_l627_627890

-- Given Conditions
def seq_a (λ : ℝ) : ℕ → ℝ
| 0     := 0
| (n+1) := 2^(n)

def partial_sum_a (λ : ℝ) : ℕ → ℝ
| 0     := 0
| (n+1) := seq_a λ (n + 1) + partial_sum_a λ n

axiom condition_Sn (λ : ℝ) : ∀ n, 2 * partial_sum_a λ n = 2^(n+1) + λ

def seq_b (λ : ℝ) (n : ℕ) : ℝ :=
  1 / ((2 * n + 1) * (Real.log 4 (seq_a λ n * seq_a λ (n + 1))))

-- Question I
theorem general_term_a (λ : ℝ) : ∀ n, seq_a λ n = 2^(n-1) := 
sorry

-- Question II
theorem sum_b (λ : ℝ) (n : ℕ) : ∑ k in Finset.range n, seq_b λ k = 2 * n / (2 * n + 1) :=
sorry

end general_term_a_sum_b_l627_627890


namespace non_congruent_triangles_with_perimeter_18_l627_627971

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627971


namespace remarkable_two_digit_numbers_count_l627_627242

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def has_four_distinct_divisors (n : ℕ) : Prop :=
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ a ≠ b ∧ n = a * b

theorem remarkable_two_digit_numbers_count : 
  (finset.filter (λ n, has_four_distinct_divisors n) (finset.filter two_digit (finset.range 100))).card = 30 :=
by
  sorry

end remarkable_two_digit_numbers_count_l627_627242


namespace f_2017_at_0_l627_627462

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.sin x
| (n+1) := λ x, (f n)' x

theorem f_2017_at_0 :
  (f 2017) 0 = 1 :=
sorry

end f_2017_at_0_l627_627462


namespace new_train_distance_l627_627796

-- Define the given conditions
def distance_old : ℝ := 300
def percentage_increase : ℝ := 0.3

-- Define the target distance to prove
def distance_new : ℝ := distance_old + (percentage_increase * distance_old)

-- State the theorem
theorem new_train_distance : distance_new = 390 := by
  sorry

end new_train_distance_l627_627796


namespace find_m_l627_627213

-- Define the points M and N and the normal vector n
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M (m : ℝ) : Point3D := { x := m, y := -2, z := 1 }
def N (m : ℝ) : Point3D := { x := 0, y := m, z := 3 }
def n : Point3D := { x := 3, y := 1, z := 2 }

-- Define the dot product
def dot_product (v1 v2 : Point3D) : ℝ :=
  (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

-- Define the vector MN
def MN (m : ℝ) : Point3D := { x := -(m), y := m + 2, z := 2 }

-- Prove the dot product condition is zero implies m = 3
theorem find_m (m : ℝ) (h : dot_product n (MN m) = 0) : m = 3 :=
by
  sorry

end find_m_l627_627213


namespace length_AD_l627_627207

-- Definitions for conditions
def AB : ℝ := 5
def BC : ℝ := 6
def CD : ℝ := 25
def cosC : ℝ := 4 / 5
def sinB : ℝ := -4 / 5
-- Assuming angles B and C are obtuse.

-- Proposition to prove
theorem length_AD (AB BC CD : ℝ) (cosC sinB: ℝ) (hB_obtuse: true) (hC_obtuse: true) : 
  AD = 18.75 :=
by
  sorry

end length_AD_l627_627207


namespace distance_at_2_point_5_l627_627691

def distance_data : List (ℝ × ℝ) :=
  [(0, 0), (1, 10), (2, 40), (3, 90), (4, 160), (5, 250)]

def quadratic_relation (t s k : ℝ) : Prop :=
  s = k * t^2

theorem distance_at_2_point_5 :
  ∃ k : ℝ, (∀ (t s : ℝ), (t, s) ∈ distance_data → quadratic_relation t s k) ∧ quadratic_relation 2.5 62.5 k :=
by
  sorry

end distance_at_2_point_5_l627_627691


namespace painting_time_l627_627861

theorem painting_time (n1 n2 : ℕ) (t1 t2 : ℝ) (h1 : n1 = 5) (h2 : t1 = 10) (h3 : n2 = 4) :
    t2 = 12.5 :=
by
  let k := n1 * t1
  have hk : k = 5 * 10 := by rw [h1, h2]; norm_num
  have h_const : n2 * t2 = k := by sorry
  have h4 : n2 * t2 = 50 := by rw [h_const, hk]; norm_num
  have h5 : t2 = 50 / 4 := by sorry
  rw [h5]; norm_num

end painting_time_l627_627861


namespace ellipse_property_l627_627221

-- Define the points F1 and F2
def F1 : ℝ × ℝ := (0, 1)
def F2 : ℝ × ℝ := (6, 1)

-- Define the property of the set of points P
def isEllipse (P : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
  let d2 := Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  d1 + d2 = 8

-- Statement to be proven
theorem ellipse_property : let h = 3
                           let k = 1
                           let a = 4
                           let b = Real.sqrt 7
                           h + k + a + b = 8 + Real.sqrt 7 :=
by
  sorry

end ellipse_property_l627_627221


namespace quadratic_transformation_l627_627807

theorem quadratic_transformation :
  ∀ (x : ℝ), (x^2 + 6*x - 2 = 0) → ((x + 3)^2 = 11) :=
by
  intros x h
  sorry

end quadratic_transformation_l627_627807


namespace reflection_over_vect_4_3_l627_627852

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

theorem reflection_over_vect_4_3 (v : Vector ℚ 2) : 
  let reflected_v := reflection_matrix.mulVec v 
  ∃ p : Vector ℚ 2, p = proj (Vector.ofFn (![4, 3])) v ∧ reflected_v = 2 • p - v :=
sorry

end reflection_over_vect_4_3_l627_627852


namespace locus_of_P_is_ellipse_l627_627486

noncomputable def circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 36

def on_circle (A : ℝ × ℝ) : Prop :=
  circle A.1 A.2

def N : ℝ × ℝ := (2, 0)

def M : ℝ × ℝ := (-2, 0)

theorem locus_of_P_is_ellipse (A P : ℝ × ℝ) (hA : on_circle A)
  (hP : ∃ N, ((N = (2, 0)) ∧ (|P.1 - A.1| = |P.1 - N.1|) ∧ ((P.1 - (-2)) * M.2 = (M.2 - P.2) * (P.2 - A.2)) ∧ ((P.1 - A.1) * (P.1 - N.1) + (A.2 - P.2) * (N.2 - P.2) = 0))) :
  ∃ e : set (ℝ × ℝ), is_ellipse e P := 
sorry

end locus_of_P_is_ellipse_l627_627486


namespace negation_of_exists_leq_l627_627680

theorem negation_of_exists_leq {x : ℝ} : 
  (¬ (∃ x ∈ set.Ioi 0, x^2 + 1 ≤ 2 * x)) ↔ (∀ x ∈ set.Ioi 0, x^2 + 1 > 2 * x) :=
by sorry

end negation_of_exists_leq_l627_627680


namespace upper_bound_not_in_T_no_real_lower_bound_in_T_neither_bound_in_T_l627_627231

noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x - 1)

-- Set T definition
def T : set ℝ := {y | ∃ x : ℝ, x ≠ 1 ∧ y = g x}

-- Theorems to prove the upper and lower bounds and their non-membership
theorem upper_bound_not_in_T : ∀ y ∈ upper_bounds T, y = 3 → ¬(y ∈ T) :=
by
  sorry

theorem no_real_lower_bound_in_T : ∀ m : ℝ, m ∈ lower_bounds T → false :=
by
  sorry

-- Final conclusion (for the problem statement D)
theorem neither_bound_in_T : ∀ (M m : ℝ), (M = 3 → ¬(M ∈ T)) ∧ 
  (∀ m, m ∈ lower_bounds T → false) :=
by
  exact ⟨upper_bound_not_in_T 3 infer_instance, no_real_lower_bound_in_T⟩

end upper_bound_not_in_T_no_real_lower_bound_in_T_neither_bound_in_T_l627_627231


namespace magazines_in_fourth_pile_l627_627743

theorem magazines_in_fourth_pile : 
  ∀ (a1 a2 a3 a5 : ℕ), a1 = 3 → a2 = 4 → a3 = 6 → a5 = 13 → 
  ∃ a4 : ℕ, a4 = 9 :=
by {
  intros a1 a2 a3 a5 h1 h2 h3 h5,
  use 9, 
  sorry
}

end magazines_in_fourth_pile_l627_627743


namespace number_of_valid_numbers_l627_627508

theorem number_of_valid_numbers : 
  let valid_middle_pairs := { (a, b) | (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (a * b > 5) } in
  let first_digit_choices := { d | 3 ≤ d ∧ d ≤ 9 } in
  let last_digit_choices := { d | 0 ≤ d ∧ d ≤ 9 } in
  (|first_digit_choices| * |valid_middle_pairs| * |last_digit_choices| = 4970) :=
by
  sorry

end number_of_valid_numbers_l627_627508


namespace problem_area_of_triangle_l627_627232

noncomputable def inverse_f (x a : ℝ) : ℝ := Real.log (x - a) / Real.log 2

def is_arithmetic_sequence (y1 y2 y3 : ℝ) : Prop :=
  y1 + y3 = 2 * y2

def distance_from_origin (x y : ℝ) : ℝ :=
  (x^2 + y^2).sqrt

def distance_point_R (a : ℝ) : ℝ :=
  distance_from_origin (2 + a) (Real.log (2 + a) / Real.log 2)

def points (x a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let P : ℝ × ℝ := (x + a, inverse_f (x + a) a)
  let Q : ℝ × ℝ := (x, inverse_f x a)
  let R : ℝ × ℝ := (2 + a, inverse_f (2 + a) a)
  (P, Q, R)

def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xR, yR) := R
  (1 / 2) * (xP * (yQ - yR) + xQ * (yR - yP) + xR * (yP - yQ)).abs

theorem problem_area_of_triangle 
  (x a : ℝ) 
  (ha : a = -(1 / 2) ∨ a ≥ 0)
  (hseq : is_arithmetic_sequence 
           (inverse_f (x + a) a) 
           (inverse_f x a) 
           (inverse_f (2 + a) a))
  (hmin : distance_point_R a = (13 / 4).sqrt):
  area_triangle (points x a).1 (points x a).2.1 (points x a).2.2 = 1 / 4 :=
by
  sorry

end problem_area_of_triangle_l627_627232


namespace max_value_and_increasing_interval_l627_627853

theorem max_value_and_increasing_interval :
  let f : ℝ → ℝ := λ x, -x^2 - 4 * x + 1
  ∃ (max_val : ℝ) (incr_interval : set.Ioo (-∞) (-2) ), 
    (∀ x : ℝ, f x ≤ max_val) ∧ 
    (∀ x y : ℝ, x ∈ incr_interval → y ∈ incr_interval → (x ≤ y) → (f x ≤ f y)) :=
by
  let f := λ x : ℝ, -x ^ 2 - 4 * x + 1
  use 5, set.Ioo (-∞ : ℝ) (-2)
  sorry

end max_value_and_increasing_interval_l627_627853


namespace abs_neg_2023_l627_627659

theorem abs_neg_2023 : |(-2023)| = 2023 :=
by
  sorry

end abs_neg_2023_l627_627659


namespace proof_part_a_proof_part_b_l627_627354

variable {A B C D P : Point}
variable {dist : Point → Point → ℝ}

def points_on_segment (A B C D : Point) : Prop :=
B ∈ segment A D ∧ C ∈ segment A D

def eq_segments (A B C D : Point) : Prop :=
dist A B = dist C D

theorem proof_part_a (h1 : points_on_segment A B C D) (h2 : eq_segments A B C D) :
  ∀ P : Point, dist P A + dist P D ≥ dist P B + dist P C :=
sorry

theorem proof_part_b (h1 : ∀ P : Point, dist P A + dist P D ≥ dist P B + dist P C) :
  points_on_segment A B C D ∧ eq_segments A B C D :=
sorry

end proof_part_a_proof_part_b_l627_627354


namespace fraction_division_result_l627_627716

theorem fraction_division_result :
  (5/6) / (-9/10) = -25/27 := 
by
  sorry

end fraction_division_result_l627_627716


namespace no_square_in_sequence_l627_627224

-- Define the sequence (a_n) with initial condition a_0 = 2016
def sequence (a : ℕ → ℚ) : Prop :=
  a 0 = 2016 ∧ ∀ n, a (n + 1) = a n + 2 / a n

-- Define the hypothesis that (a_n) is the sequence with the given properties
variable (a : ℕ → ℚ) (h : sequence a)

-- The main theorem to show that no term of the sequence is a square of a rational number
theorem no_square_in_sequence : ∀ n, ∀ q : ℚ, a n ≠ q^2 :=
sorry

end no_square_in_sequence_l627_627224


namespace jonathan_daily_phone_time_l627_627432

-- Define the conditions
def weekly_social_media_hours : ℕ := 28
def daily_hours (x : ℕ) : Prop := weekly_social_media_hours * 2 = x * 7

-- State the problem as a theorem
theorem jonathan_daily_phone_time : ∃ x : ℕ, daily_hours x ∧ x = 8 := by
  existsi 8
  have h1 : weekly_social_media_hours * 2 = 56 := rfl
  have h2 : 56 = 8 * 7 := rfl
  exact ⟨rfl, rfl⟩
  sorry

end jonathan_daily_phone_time_l627_627432


namespace probability_even_sum_of_three_dice_l627_627703

theorem probability_even_sum_of_three_dice :
  (∃ dice : list (fin 8 → ℕ), dice.length = 3 ∧ (∀ d ∈ dice, ∀ i : fin 8, d i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) ∧
  (∃ P : ℚ, P = 1 / 2)) :=
by
  -- Define the faces of the dice
  let faces := [1, 2, 3, 4, 5, 6, 7, 8] 
  -- Define dice outcomes
  let dice := list.replicate 3 (λ _ : fin 8, faces)
  exists dice
  s :
  { dice.length = 3 ∧ (∀ d ∈ dice, ∀ i : fin 8, d i ∈ faces) } 
  {
    let P : ℚ := 1 / 2
    exists P
cl
     
end


end probability_even_sum_of_three_dice_l627_627703


namespace line_equations_through_point_origin_distance_l627_627077

theorem line_equations_through_point_origin_distance
  (A : ℝ × ℝ)
  (hA : A = (1, 2))
  (d : ℝ)
  (hd : d = 1) :
  (∀ (l : ℝ × ℝ → Prop),
    (l A) ∧ (∀ (P : ℝ × ℝ), l P → (∃ (a b c : ℝ), P.1 * a + P.2 * b + c = 0 ∧ abs c / sqrt (a^2 + b^2) = d))
    → (l = (λ P, P.1 = 1) ∨ l = (λ P, 3 * P.1 - 4 * P.2 + 5 = 0))) :=
by
  intros l h
  sorry

end line_equations_through_point_origin_distance_l627_627077


namespace constant_function_of_equation_l627_627572

noncomputable def continuous_function := {f : ℝ → ℝ // continuous f}

theorem constant_function_of_equation (f : continuous_function) 
    (h : ∀ x t, x ∈ ℝ → t ≥ 0 → f.val x = f.val (real.exp t * x)) :
    ∀ x, f.val x = f.val 0 :=
by
  sorry

end constant_function_of_equation_l627_627572


namespace number_of_ten_yuan_bills_l627_627567

theorem number_of_ten_yuan_bills 
(h_total_change : 100 - 5 = 100 - 5)
(h_notes_count : ∃ x y : ℕ, x + y = 16 ∧ 10 * x + 5 * y = 95) :
∃ x : ℕ, x = 3 :=
by 
  obtain ⟨x, y, h1, h2⟩ := h_notes_count,
  have h_eq := @eq_of_add_eq_of_sub_eq ℕ _ 16 5 x y _ _ h1,
  use 3,
  sorry

end number_of_ten_yuan_bills_l627_627567


namespace count_non_congruent_triangles_with_perimeter_18_l627_627933

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627933


namespace number_of_non_congruent_triangles_perimeter_18_l627_627946

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627946


namespace tangent_parallel_range_a_l627_627172

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x

theorem tangent_parallel_range_a :
  (∃ x > 0, deriv (λ x, 2 * Real.log x - a * x) x = -3) → a ∈ Ioi 3 :=
begin
  -- If a tangent line to f(x) is parallel to 3x + y + 1 = 0, the slope of f(x) must be -3.
  intros h,
  rw [deriv_log_eq_inv, deriv_sub_const, deriv_mul_const, deriv_id] at h,
  sorry
end

end tangent_parallel_range_a_l627_627172


namespace max_distance_from_circle_to_line_range_of_possible_slopes_for_reflected_light_ray_l627_627152

-- Definitions
def line (x y : ℝ) : Prop := x - y + 5 = 0
def circle (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 4
def point_B (x y : ℝ) : Prop := x = -2 ∧ y = -3

-- Proof problem for question 1
theorem max_distance_from_circle_to_line : 
  (∀ x y : ℝ, circle x y → ∃ d : ℝ, d = 3 * sqrt 2 + 2) := 
sorry

-- Proof problem for question 2
theorem range_of_possible_slopes_for_reflected_light_ray :
  (∀ x y : ℝ, point_B x y → ∃ k : ℝ, k ∈ set.Icc (-sqrt 35 / 35) (sqrt 35 / 35)) :=
sorry

end max_distance_from_circle_to_line_range_of_possible_slopes_for_reflected_light_ray_l627_627152


namespace line_through_P_perpendicular_l627_627080

theorem line_through_P_perpendicular 
  (P : ℝ × ℝ) (a b c : ℝ) (hP : P = (-1, 3)) (hline : a = 1 ∧ b = -2 ∧ c = 3) :
  ∃ (a' b' c' : ℝ), (a' * P.1 + b' * P.2 + c' = 0) ∧ (a = b' ∧ b = -a') ∧ (a' = 2 ∧ b' = 1 ∧ c' = -1) := 
by
  use 2, 1, -1
  sorry

end line_through_P_perpendicular_l627_627080


namespace tangent_difference_identity_l627_627485

open Real

theorem tangent_difference_identity (θ : ℝ) (h1 : ∀ x, y = 3 * x) : tan (θ - π / 4) = 1 / 2 := by
  -- Given the terminal side of θ lies on the line y = 3x, we have tan θ = 3.
  have hθ : tan θ = 3 := 
    by sorry  -- You need to deduce that tan θ = 3.
  -- Use the trigonometric identity for tangent of a difference.
  have result : tan (θ - π / 4) = (tan θ - tan (π / 4)) / (1 + tan θ * tan (π / 4)) :=
    by sorry  -- You apply the identity here.
  -- Substitute tan θ = 3 and tan (π / 4) = 1.
  rw [tan_pi_div_four] at result
  rw [hθ] at result
  -- Calculate the result.
  have final_result : (3 - 1) / (1 + 3 * 1) = 1 / 2 :=
    by norm_num
  -- Therefore, tan (θ - π / 4) is indeed 1/2.
  exact final_result

end tangent_difference_identity_l627_627485


namespace general_term_formula_sum_of_first_n_terms_l627_627468

-- Define the arithmetic sequence and its sum as given conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Given conditions
axiom h1 : a 2 = 6
axiom h2 : S 3 = 26
axiom h3 : ∀ n : ℕ, S n = ∑ i in finset.range (n+1), a i
axiom h4 : ∀ n : ℕ, a (n+1) > a n -- Monotonically increasing condition

-- Find the general term formula for a_n
theorem general_term_formula : a n = 2 * 3^(n-1) :=
sorry

-- Define b_n and find the sum of the first n terms T_n
def b (n : ℕ) : ℝ := a n - 2 * n
theorem sum_of_first_n_terms : (∑ i in finset.range n, b i) = 3^n - 1 - n^2 - n :=
sorry

end general_term_formula_sum_of_first_n_terms_l627_627468


namespace probability_of_rolling_divisor_of_8_l627_627778

open_locale classical

-- Predicate: a number n is a divisor of 8
def is_divisor_of_8 (n : ℕ) : Prop := n ∣ 8

-- The total number of outcomes when rolling an 8-sided die
def total_outcomes : ℕ := 8

-- The probability of rolling a divisor of 8 on a fair 8-sided die
theorem probability_of_rolling_divisor_of_8 (is_fair_die : true) :
  (| {n | is_divisor_of_8 n} ∩ {1, 2, 3, 4, 5, 6, 7, 8} | : ℕ) / total_outcomes = 1 / 2 :=
by
  sorry

end probability_of_rolling_divisor_of_8_l627_627778


namespace max_omega_l627_627706

-- Define the function f(x) = 2 * sin(ω * x + π / 3)
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3)

-- Translate f to the right by π / (3 * ω) to obtain g
def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x - Real.pi / (3 * ω))

theorem max_omega (ω : ℝ) (hω : 0 < ω) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), g ω x ≤ g ω (x + Real.pi / (4 * ω))) →
  ω ≤ 2 := 
by
  sorry

end max_omega_l627_627706


namespace opening_price_stock_l627_627057

theorem opening_price_stock (closing_price : ℝ) (percent_increase : ℝ) (h₁ : closing_price = 29) (h₂ : percent_increase = 3.571428571428581) : 
  let opening_price := 29 / (1 + percent_increase / 100) in 
  opening_price ≈ 28 :=
by
  sorry

end opening_price_stock_l627_627057


namespace hyperbola_problem_l627_627918

-- Definitions and conditions
def hyperbola_eq (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def eccentricity (c a : ℝ) : Prop := c / a = Real.sqrt 3
def passes_through (x0 y0 x y : ℝ) : Prop := ((x, y) = (x0, y0))

-- Main statement
theorem hyperbola_problem
  (a b c x y : ℝ)
  (h1 : hyperbola_eq a b x y)
  (h2 : eccentricity c a)
  (h3 : passes_through 2 2 x y)
  (h4 : (0, 4))
  (h5 : ∃ A B : ℝ × ℝ, A ≠ B)
  (h6 : ∃ k : ℝ, (k ≠ 2 ∧ passes_through ((Real.sqrt 3) * a) k x y)) :
  (hyperbola_eq (Real.sqrt 2) (Real.sqrt 4) x y ∧ ∃ d : ℝ, d = 2 * Real.sqrt 5 / 5) :=
by
  sorry

end hyperbola_problem_l627_627918


namespace third_gen_dd_prob_third_gen_dominant_2_of_3_l627_627540

/- Define first generation genotype -/
def first_genotype := "Dd"

/- Define the possible genotypes with their probabilities in the second generation -/
def second_genotype_prob : List (String × ℚ) :=
  [("DD", 1/4), ("Dd", 1/2), ("dd", 1/4)]

/- Define the possible genotypes with their probabilities in the third generation -/
def third_genotype_prob (parent_genotype : String) : List (String × ℚ) :=
  match parent_genotype with
  | "DD" => [("DD", 1), ("Dd", 0), ("dd", 0)]
  | "Dd" => [("DD", 1/4), ("Dd", 1/2), ("dd", 1/4)]
  | "dd" => [("DD", 0), ("Dd", 0), ("dd", 1)]
  | _    => []

/- The calculation of the probability of a genotype "dd" in the third generation -/
theorem third_gen_dd_prob :
  (1/4 * 1/4) + (1/2 * 1/4 * 1/2) + (1/2 * 1/4 * 1/2) + (1/4 * 1) = 1/4 :=
by simp; norm_num

/- Calculate the probability of the condition "exactly 2 out of 3 peas exhibit the dominant trait in the third generation -/
def binomial (n k : Nat) : ℚ :=
  Nat.choose n k / (2^n : ℚ)

theorem third_gen_dominant_2_of_3 :
  binomial 3 2 * (3/4) ^ 2 * (1/4) = 27 / 64 :=
by 
  have h: binomial 3 2 = 3 := by simp [binomial, Nat.choose]
  rw [h]
  norm_num

-- Additional sorry to skip proofs that aren't required.

end third_gen_dd_prob_third_gen_dominant_2_of_3_l627_627540


namespace slope_of_line_OM_l627_627923

noncomputable def ellipse_coordinates (t : ℝ) : ℝ × ℝ :=
  (2 * Real.cos t, 4 * Real.sin t)

def point_M : ℝ × ℝ := ellipse_coordinates (Real.pi / 3)
def point_O : ℝ × ℝ := (0, 0)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_line_OM :
  slope point_O point_M = 2 * Real.sqrt 3 :=
by
  sorry

end slope_of_line_OM_l627_627923


namespace pauls_plumbing_hourly_charge_l627_627637

theorem pauls_plumbing_hourly_charge :
  ∀ P : ℕ,
  (55 + 4 * P = 75 + 4 * 30) → 
  P = 35 :=
by
  intros P h
  sorry

end pauls_plumbing_hourly_charge_l627_627637


namespace focus_of_parabola_l627_627665

theorem focus_of_parabola (focus : ℝ × ℝ) : 
  (∃ p : ℝ, y = p * x^2 / 2 → focus = (0, 1 / 2)) :=
by
  sorry

end focus_of_parabola_l627_627665


namespace work_problem_l627_627350

theorem work_problem (W : ℝ) (A_rate : ℝ) (AB_rate : ℝ) : A_rate = W / 14 ∧ AB_rate = W / 10 → 1 / (AB_rate - A_rate) = 35 :=
by
  sorry

end work_problem_l627_627350


namespace book_pages_product_l627_627342

theorem book_pages_product : ∃ n : ℕ, (n + (n + 1) = 217) ∧ ((n * (n + 1)) = 11772) :=
by {
  existsi 108,
  split,
  { calc
      108 + 109 = 217 : by norm_num
  },
  { calc
      108 * 109 = 11772 : by norm_num
  }
}

end book_pages_product_l627_627342


namespace candy_count_correct_l627_627450

-- Define initial count of candy
def initial_candy : ℕ := 47

-- Define number of pieces of candy eaten
def eaten_candy : ℕ := 25

-- Define number of pieces of candy received
def received_candy : ℕ := 40

-- The final count of candy is what we are proving
theorem candy_count_correct : initial_candy - eaten_candy + received_candy = 62 :=
by
  sorry

end candy_count_correct_l627_627450


namespace min_value_and_period_f_l627_627297

def f (x : ℝ) := sqrt 3 * sin (2 * x - π / 6) - 1

theorem min_value_and_period_f :
  (∀ x, f(x) ≥ - sqrt 3 - 1) ∧ (∃ T > 0, ∀ x, f(x + T) = f(x)) ∧ T = π :=
by
  sorry

end min_value_and_period_f_l627_627297


namespace intersection_complement_empty_l627_627239

def A : Set ℝ := { x : ℝ | Real.sqrt (x - 2) ≤ 0 }
def B : Set ℝ := { x : ℝ | 10 ^ (x^2 - 2) = 10 ^ x }

theorem intersection_complement_empty :
  A ∩ (Set.univ \ B) = ∅ := 
by
  sorry

end intersection_complement_empty_l627_627239


namespace correct_propositions_l627_627549

-- Define the different lines l, m, n and different planes α, β.
variables {α β : Plane} {l m n : Line}

-- Define the conditions
def condition1 : Prop := l ∥ α
def condition2 : Prop := m ∥ α
def condition3 : Prop := l ∥ m
def condition4 : Prop := l ⊥ α
def condition5 : Prop := m ⊥ β
def condition6 : Prop := l ⊥ m
def condition7 : Prop := m ⊥ α
def condition8 : Prop := n ⊥ β
def condition9 : Prop := m ∥ n
def condition10 : Prop := α ∥ β
def condition11 : Prop := m ⊆ α
def condition12 : Prop := n ⊆ β

-- Proposition 1: α ∥ β
def proposition1 : Prop := α ∥ β

-- Proposition 2: α ⊥ β
def proposition2 : Prop := α ⊥ β

-- Proposition 3: α ∥ β
def proposition3 : Prop := α ∥ β

-- Proposition 4: m ∥ n
def proposition4 : Prop := m ∥ n

-- Proof problem to show which propositions are correct given the conditions.
theorem correct_propositions 
  (h1 : condition4)
  (h2 : condition5)
  (h3 : condition6)
  (h4 : condition7)
  (h5 : condition8)
  (h6 : condition9)
  (h7 : condition10)
  (h8 : condition11)
  (h9 : condition12) :
  (proposition2 ∧ proposition3) ∧ ¬proposition1 ∧ ¬proposition4 :=
by 
  sorry

end correct_propositions_l627_627549


namespace sum_of_digits_of_x_squared_eq_36_l627_627384

noncomputable def base_r_representation_sum (r : ℕ) (x : ℕ) := ∃ (p q : ℕ), 
  r <= 36 ∧
  x = p * (r^3 + r^2) + q * (r + 1) ∧
  2 * q = 5 * p ∧
  ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a ∧
  b = 9 ∧
  a + b + c = 18

theorem sum_of_digits_of_x_squared_eq_36 (r x : ℕ) :
  base_r_representation_sum r x → ∑ d in (digits r (x^2)), d = 36 :=
sorry

end sum_of_digits_of_x_squared_eq_36_l627_627384


namespace part_1_part_2_part_3_part_4_l627_627109

open Complex

def Z (m : ℝ) : Complex :=
  (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * I

theorem part_1 (m : ℝ) : (Z m).im = 0 ↔ m = 1 ∨ m = 2 :=
by sorry

theorem part_2 (m : ℝ) : (Z m).re = 0 ↔ m ≠ 1 ∧ m ≠ 2 :=
by sorry

theorem part_3 (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = -1/2 :=
by sorry

theorem part_4 (m : ℝ) : (Z m).im > 0 ↔ m ∈ Set.Ioo 0 1 ∨ m ∈ Set.Ioo 2 ⊤ :=
by sorry


end part_1_part_2_part_3_part_4_l627_627109


namespace range_of_independent_variable_l627_627304

theorem range_of_independent_variable (x : ℝ) : (√(x - 3) / x).is_defined → x ≥ 3 :=
by
  intros h
  sorry

end range_of_independent_variable_l627_627304


namespace ticket_price_reduction_l627_627694

-- Definitions of the problem constants and variables
def original_price : ℝ := 50
def increase_fraction : ℝ := 1 / 3
def revenue_increase_fraction : ℝ := 1 / 4

-- New number of tickets sold after price reduction
def new_number_of_tickets_sold (x : ℝ) : ℝ := x * (1 + increase_fraction)

-- New price per ticket after reduction
def new_price_per_ticket (reduction : ℝ) : ℝ := original_price - reduction

-- Original revenue
def original_revenue (x : ℝ) : ℝ := x * original_price

-- New revenue after price reduction
def new_revenue (x reduction : ℝ) : ℝ := new_number_of_tickets_sold x * new_price_per_ticket reduction

-- The equation relating new revenue to the original revenue with the given increase
def revenue_relation (x reduction : ℝ) : Prop :=
  new_revenue x reduction = (1 + revenue_increase_fraction) * original_revenue x

-- The goal is to find the reduction in price per ticket (reduction) such that the revenue_relation holds
theorem ticket_price_reduction :
  ∃ y : ℝ, ∀ x > 0, revenue_relation x y ∧ y = 25 / 2 :=
begin
  sorry -- Proof goes here
end

end ticket_price_reduction_l627_627694


namespace maximum_pressure_l627_627632

variables {R V0 T0 a b c : ℝ}
variables {P_max : ℝ}

def cyclic_process (V T : ℝ) : Prop :=
  ((V / V0 - a) ^ 2 + (T / T0 - b) ^ 2 = c ^ 2)

noncomputable def ideal_gas_pressure (R T V : ℝ) : ℝ :=
  R * T / V

theorem maximum_pressure 
  (h_c2_lt : c^2 < a^2 + b^2) :
  ∃ P_max : ℝ, P_max = (R * T0 / V0) * (a * real.sqrt(a^2 + b^2 - c^2) + b * c) / (b * real.sqrt(a^2 + b^2 - c^2) - a * c) :=
sorry

end maximum_pressure_l627_627632


namespace part_I_part_II_l627_627149

-- Define the function f(x) = |x - a|
def f (x a : ℝ) : ℝ := |x - a|

-- Part (I)
theorem part_I (x : ℝ) : (x - 2) := 
begin
  sorry
end

-- Part (II)
theorem part_II (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 + 2 * n^2 = 1) : 
  ∀ (x : ℝ), 0 ≤ x → x ≤ 2 → |x - 1| ≤ 1 → m + 4 * n ≤ 3 :=
begin
  sorry
end

end part_I_part_II_l627_627149


namespace derivative_at_zero_l627_627759

noncomputable def f : ℝ → ℝ :=
λ x : ℝ, 
if x ≠ 0 then log (1 - sin (x^3 * sin (1 / x))) else 0

theorem derivative_at_zero :
  deriv f 0 = 0 :=
by
  sorry

end derivative_at_zero_l627_627759


namespace polygon_length_l627_627661

noncomputable def DE : ℝ := 3
noncomputable def EF : ℝ := 6
noncomputable def DE_plus_EF : ℝ := DE + EF

theorem polygon_length 
  (area_ABCDEF : ℝ)
  (AB BC FA : ℝ)
  (A B C D E F : ℝ × ℝ) :
  area_ABCDEF = 60 →
  AB = 10 →
  BC = 7 →
  FA = 6 →
  A = (0, 10) →
  B = (10, 10) →
  C = (10, 0) →
  D = (6, 0) →
  E = (6, 3) →
  F = (0, 3) →
  DE_plus_EF = 9 :=
by
  intros
  sorry

end polygon_length_l627_627661


namespace incorrect_statement_D_l627_627907

theorem incorrect_statement_D (a b r : ℝ) (hr : r > 0) :
  ¬ ∀ b < r, ∃ x, (x - a)^2 + (0 - b)^2 = r^2 :=
by 
  sorry

end incorrect_statement_D_l627_627907


namespace tangent_perpendicular_to_OA_l627_627840

-- Define the setup and conditions
variables {A B C O : Point} -- Points are vertices and circumcenter
variable (tangent : Line) -- A common tangent of excircles

-- Assumptions
axiom excircle_common_tangent : ∀ (P Q : Point), is_excircle P Q A B C ∧ common_tangent P Q ∧ ¬(on_side P Q A B C)
axiom circumcenter : circumcenter O A B C

-- Theorem: Show the tangent is perpendicular to OA
theorem tangent_perpendicular_to_OA : perpendicular tangent (line_through O A) :=
sorry

end tangent_perpendicular_to_OA_l627_627840


namespace number_of_non_congruent_triangles_perimeter_18_l627_627942

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627942


namespace convex_pentagon_area_l627_627283

noncomputable def pentagon_area (A B C D E : Point) : Real := 7 * sqrt 3

theorem convex_pentagon_area
  (A B C D E : Type)
  [Geometry.Point A] [Geometry.Point B] [Geometry.Point C] [Geometry.Point D] [Geometry.Point E]
  (h_convex : Geometry.ConvexPentagon A B C D E)
  (h_angleA : Geometry.Angle A = 120)
  (h_angleB : Geometry.Angle B = 120)
  (h_EA : Geometry.Distance E A = 2)
  (h_AB : Geometry.Distance A B = 2)
  (h_BC : Geometry.Distance B C = 2)
  (h_CD : Geometry.Distance C D = 4)
  (h_DE : Geometry.Distance D E = 4)
  : Geometry.Area A B C D E = pentagon_area A B C D E :=
sorry

end convex_pentagon_area_l627_627283


namespace selling_price_of_cycle_l627_627373

theorem selling_price_of_cycle (CP GP : ℝ) (hCP : CP = 675) (hGP : GP = 0.6) : CP + (GP * CP) = 1080 :=
by
  rw [hCP, hGP]
  norm_num
  exact sorry

end selling_price_of_cycle_l627_627373


namespace number_of_non_congruent_triangles_perimeter_18_l627_627950

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627950


namespace part1_part2_l627_627140

-- Part (1): Given m = 4, show that x = 2
theorem part1 (x : ℝ) : (3 * x / (x - 1) = 4 / (x - 1) + 2) → (x = 2) :=
by
  intros h
  have h1 : 3 * x = 4 + 2 * (x - 1),
  { linarith [h] },
  rw [sub_self, add_zero, add_sub_assoc, add_comm, add_sub_assoc, add_sub_assoc, zero_add] at h1,
  rw [eq_add_of_sub_eq h1] at *,
  linarith,
  sorry

-- Part (2): Show that when the equation has no solution, m = 3
theorem part2 : (∀ x, ¬ (3 * x / (x - 1) = m / (x - 1) + 2)) → m = 3 :=
by
  intros h
  have h1 : ∃ x, (x = m - 2), from sorry,
  cases h1 with x hx,
  have h2 : x = 1, from sorry,
  rw [hx, h2] at *,
  linarith,
  sorry

end part1_part2_l627_627140


namespace number_of_incorrect_propositions_l627_627409

-- Definitions matching the conditions in the problem
def proposition1 (a b : Type) [LinearSpace a] [LinearSpace b] : Prop :=
  ¬(∃ p : point, p ∈ a ∧ p ∈ b) → ¬(∃ q r : point, q ∈ a ∧ r ∈ b ∧ q ≠ r)

def proposition2 (a : Type) (β : Type) [LinearSpace α] [Plane β] : Prop :=
  (∀ l : Line, l ∈ β → a ⊥ l) → a ⊥ β

def proposition3 (a b : Type) (β : Type) [LinearSpace a] [LinearSpace b] [Plane β] : Prop :=
  (∀ proj_b : Line, proj_b = projection β b → a ⊥ proj_b) → a ⊥ b

def proposition4 (a : Type) (β : Type) [LinearSpace a] [Plane β] : Prop :=
  (∃ l : Line, l ∈ β ∧ a ∥ l) → a ∥ β

-- Main theorem to prove the number of incorrect propositions
theorem number_of_incorrect_propositions
  (a b : Type) (β : Type) [LinearSpace a] [LinearSpace b] [Plane β]
  (p1 : ¬(proposition1 a b))
  (p2 : ¬(proposition2 a β))
  (p3 : ¬(proposition3 a b β))
  (p4 : ¬(proposition4 a β)) :
  (4 = 4) :=
by
  sorry

end number_of_incorrect_propositions_l627_627409


namespace sum_of_digits_palindrome_l627_627382

theorem sum_of_digits_palindrome 
  (r : ℕ) 
  (h1 : r ≤ 36) 
  (x p q : ℕ) 
  (h2 : 2 * q = 5 * p) 
  (h3 : x = p * r^3 + p * r^2 + q * r + q) 
  (h4 : ∃ (a b c : ℕ), (x * x = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a)) : 
  (2 * (a + b + c) = 36) := 
sorry

end sum_of_digits_palindrome_l627_627382


namespace find_c_plus_d_l627_627519

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l627_627519


namespace ratio_tangent_circles_constant_l627_627569

variables (C1 C2 : Type) (O1 O2 A B C T : Type)
variables (R r : ℝ)
variables [MetricSpace C1] [MetricSpace C2]
variables [Circle C1] [Circle C2]
variables (BC : ℝ)

-- Defining the conditions
def tangent_internally (C1 : Type) (C2 : Type) (A : Type) : Prop :=
  -- C1 and C2 are tangent internally at point A
  True

def is_chord (C1 : Type) (B C : Type) : Prop :=
  -- B and C are points on the circle C1
  True

def is_tangent (C2 : Type) (T : Type) : Prop :=
  -- BC is tangent to the circle C2 at point T
  True

def length_chord (BC : ℝ) : ℝ :=
  -- length of the chord BC
  2 * sqrt (r^2 - 2 * R * r)

def perimeter_triangle (R r : ℝ) : ℝ :=
  -- perimeter of the triangle ABC
  2 * R + 2 * sqrt (r^2 - 2 * R * r)

def ratio_constant (BC : ℝ) (R r : ℝ) : Prop :=
  -- ratio between BC and the perimeter of the triangle ABC is constant
  (length_chord BC) / (perimeter_triangle R r) = (sqrt (r^2 - 2 * R * r)) / (R + sqrt (r^2 - 2 * R * r))

-- Main theorem statement
theorem ratio_tangent_circles_constant :
  tangent_internally C1 C2 A →
  is_chord C1 B C →
  is_tangent C2 T →
  ratio_constant BC R r :=
by
  intros
  sorry

end ratio_tangent_circles_constant_l627_627569


namespace simplify_expression_l627_627270

theorem simplify_expression : (-5 : ℝ)^2 - real.sqrt 3 = 25 - real.sqrt 3 := by
  -- The theorem statement reflects the translation of the proof problem
  sorry

end simplify_expression_l627_627270


namespace partA_partB_partC_partD_l627_627707

variable (α β : ℝ)
variable (hα : 0 < α) (hα1 : α < 1)
variable (hβ : 0 < β) (hβ1 : β < 1)

theorem partA : 
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 := by
  sorry

theorem partB :
  β * (1 - β)^2 = β * (1 - β)^2 := by
  sorry

theorem partC :
  β * (1 - β)^2 + (1 - β)^3 = β * (1 - β)^2 + (1 - β)^3 := by
  sorry

theorem partD (hα0 : α < 0.5) :
  (1 - α) * (α - α^2) < (1 - α) := by
  sorry

end partA_partB_partC_partD_l627_627707


namespace min_value_4x_plus_3y_l627_627474

theorem min_value_4x_plus_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_plus_3y_l627_627474


namespace find_profit_rate_first_year_l627_627023

noncomputable def profit_rate_first_year (initial_investment : ℝ) (profit_first_year_rate : ℝ) (profit_second_year : ℝ) : ℝ :=
  let capital_second_year := initial_investment * (1 + profit_first_year_rate)
  let profit_second_year_rate := profit_first_year_rate + 0.08
  solve_by_elim only [equations_of_the_day]

theorem find_profit_rate_first_year (initial_investment : ℝ) (profit_second_year : ℝ) :
  (initial_investment = 5) →
  (profit_second_year = 1.12) →
  profit_rate_first_year initial_investment 0.12 profit_second_year = 0.12 :=
by
  intros h1 h2
  have h3 : (500 + 500 * 0.12) * (0.12 + 0.08) = 1.12 := sorry
  sorry

end find_profit_rate_first_year_l627_627023


namespace gcd_of_180_and_450_l627_627084

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l627_627084


namespace photos_per_album_correct_l627_627812

-- Define the conditions
def total_photos : ℕ := 4500
def first_batch_photos : ℕ := 1500
def first_batch_albums : ℕ := 30
def second_batch_albums : ℕ := 60
def remaining_photos : ℕ := total_photos - first_batch_photos

-- Define the number of photos per album for the first batch (should be 50)
def photos_per_album_first_batch : ℕ := first_batch_photos / first_batch_albums

-- Define the number of photos per album for the second batch (should be 50)
def photos_per_album_second_batch : ℕ := remaining_photos / second_batch_albums

-- Statement to prove
theorem photos_per_album_correct :
  photos_per_album_first_batch = 50 ∧ photos_per_album_second_batch = 50 :=
by
  simp [photos_per_album_first_batch, photos_per_album_second_batch, remaining_photos]
  sorry

end photos_per_album_correct_l627_627812


namespace train_speed_kmph_l627_627037

/-- Define the lengths of the train and bridge, as well as the time taken to cross the bridge. --/
def train_length : ℝ := 150
def bridge_length : ℝ := 150
def crossing_time_seconds : ℝ := 29.997600191984642

/-- Calculate the speed of the train in km/h. --/
theorem train_speed_kmph : 
  let total_distance := train_length + bridge_length
  let time_in_hours := crossing_time_seconds / 3600
  let speed_mph := total_distance / time_in_hours
  let speed_kmph := speed_mph / 1000
  speed_kmph = 36 := by
  /- Proof omitted -/
  sorry

end train_speed_kmph_l627_627037


namespace frog_jump_probability_l627_627376

theorem frog_jump_probability :
  let P : ℕ × ℕ → ℝ := λ (pos : ℕ × ℕ),
    if pos.2 = 0 ∨ pos.2 = 5 then 1
    else if pos.1 = 0 ∨ pos.1 = 5 then 0
    else
      match pos with
      | (2, 3) => (1/2) * P (1, 3) + (1/2) * P (3, 3)
      | (1, 3) => (1/4) * P (0, 3) + (1/4) * P (2, 3) + (1/2) * P (1, 2)
      | (3, 3) => (1/4) * P (4, 3) + (1/4) * P (2, 3) + (1/2) * P (3, 2)
      | (1, 2) => (1/4) * P (0, 2) + (1/4) * P (2, 2) + (1/4) * P (1, 1) + (1/4) * P (1, 3)
      | (3, 2) => (1/4) * P (4, 2) + (1/4) * P (2, 2) + (1/4) * P (3, 1) + (1/4) * P (3, 3)
      | _ => sorry -- All other cases are beyond the initial question conditions
  in P (2, 3) = 0.6 :=
sorry

end frog_jump_probability_l627_627376


namespace total_lunch_bill_l627_627647

theorem total_lunch_bill (hotdog salad : ℝ) (h1 : hotdog = 5.36) (h2 : salad = 5.10) : hotdog + salad = 10.46 := 
by
  rw [h1, h2]
  norm_num
  

end total_lunch_bill_l627_627647


namespace sum_log_ceiling_minus_floor_l627_627406

theorem sum_log_ceiling_minus_floor :
  ∑ k in Finset.range 501, k * ((⌈Real.log k / Real.log 3⌉ - ⌊Real.log k / Real.log 3⌋) : ℝ) = 124886 := 
sorry

end sum_log_ceiling_minus_floor_l627_627406


namespace third_part_of_division_l627_627748

noncomputable def divide_amount (total_amount : ℝ) : (ℝ × ℝ × ℝ) :=
  let part1 := (1/2)/(1/2 + 2/3 + 3/4) * total_amount
  let part2 := (2/3)/(1/2 + 2/3 + 3/4) * total_amount
  let part3 := (3/4)/(1/2 + 2/3 + 3/4) * total_amount
  (part1, part2, part3)

theorem third_part_of_division :
  divide_amount 782 = (261.0, 214.66666666666666, 306.0) :=
by
  sorry

end third_part_of_division_l627_627748


namespace root_interval_l627_627738

noncomputable def f : ℝ → ℝ := λ x, (1 / 2^x) - x^(1 / 2)

theorem root_interval :
  (∃ x ∈ set.Icc (0:ℝ) (1:ℝ), f(x) = 0) :=
begin
  -- we use the conditions given in the problem
  have f0 : f 0 = 1, by simp [f],
  have f1 : f 1 = - (1 / 2), by dec_trivial, -- manually checked and simplified
  -- results from the decreasing property not explicitly proved here
  exact sorry,
end

end root_interval_l627_627738


namespace problem_1_sol_problem_2_l627_627012

noncomputable def problem_1 (P : ℕ → ℕ) (x : ℕ) : Prop :=
  3 * (P x)^3 ≤ 2 * (P (x + 1))^2 + 6 * (P x)^2 ∧ x ≥ 3

theorem problem_1_sol (P : ℕ → ℕ) (x : ℕ) (h : 3 * (P x)^3 ≤ 2 * (P (x + 1))^2 + 6 * (P x)^2) : 
  x ∈ {3, 4, 5} := 
sorry

noncomputable def comb (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem problem_2 (m : ℕ) (h : 1 / comb 5 m - 1 / comb 6 m = 7 / (10 * comb 7 m)) :
  comb 8 m = 28 := 
sorry

end problem_1_sol_problem_2_l627_627012


namespace exists_convex_polyhedra_arrangement_l627_627258

theorem exists_convex_polyhedra_arrangement :
  ∃ (P : Fin 2001 → Set (Set Point)), 
    (∀ (i j k : Fin 2001), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ (P i ∩ P j ∩ P k).NonEmpty) ∧ 
    (∀ (i j : Fin 2001), i ≠ j → (P i ∩ P j).NonEmpty ∧ (Interior (P i) ∩ Interior (P j) = ∅)) :=
sorry

end exists_convex_polyhedra_arrangement_l627_627258


namespace which_is_linear_l627_627739

-- Define what it means to be a linear equation in two variables
def is_linear_equation_in_two_vars (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y = (a * x + b * y = c)

-- Define each of the given equations
def equation_A (x y : ℝ) : Prop := x / 2 + 3 * y = 2
def equation_B (x y : ℝ) : Prop := x / 2 + 1 = 3 * x * y
def equation_C (x y : ℝ) : Prop := 2 * x + 1 = 3 * x
def equation_D (x y : ℝ) : Prop := 3 * x + 2 * y^2 = 1

-- Theorem stating which equation is linear in two variables
theorem which_is_linear : 
  is_linear_equation_in_two_vars equation_A ∧ 
  ¬ is_linear_equation_in_two_vars equation_B ∧ 
  ¬ is_linear_equation_in_two_vars equation_C ∧ 
  ¬ is_linear_equation_in_two_vars equation_D := 
by 
  sorry

end which_is_linear_l627_627739


namespace apples_to_eat_raw_l627_627244

/-- Proof of the number of apples left to eat raw given the conditions -/
theorem apples_to_eat_raw 
  (total_apples : ℕ)
  (pct_wormy : ℕ)
  (pct_moldy : ℕ)
  (wormy_apples_offset : ℕ)
  (wormy_apples bruised_apples moldy_apples apples_left : ℕ) 
  (h1 : total_apples = 120)
  (h2 : pct_wormy = 20)
  (h3 : pct_moldy = 30)
  (h4 : wormy_apples = pct_wormy * total_apples / 100)
  (h5 : moldy_apples = pct_moldy * total_apples / 100)
  (h6 : bruised_apples = wormy_apples + wormy_apples_offset)
  (h7 : wormy_apples_offset = 9)
  (h8 : apples_left = total_apples - (wormy_apples + moldy_apples + bruised_apples))
  : apples_left = 27 :=
sorry

end apples_to_eat_raw_l627_627244


namespace Zhukov_birth_year_l627_627290

-- Define the conditions
def years_lived_total : ℕ := 78
def years_lived_20th_more_than_19th : ℕ := 70

-- Define the proof problem
theorem Zhukov_birth_year :
  ∃ y19 y20 : ℕ, y19 + y20 = years_lived_total ∧ y20 = y19 + years_lived_20th_more_than_19th ∧ (1900 - y19) = 1896 :=
by
  sorry

end Zhukov_birth_year_l627_627290


namespace locus_of_centers_of_tangent_circles_l627_627095

theorem locus_of_centers_of_tangent_circles (L : set (ℝ × ℝ)) (a : ℝ) (hL : ∃ x, L = {p : ℝ × ℝ | p.2 = x}) :
  {O : ℝ × ℝ | ∃ r, r = a ∧ ∃ C : ℝ × ℝ, (C.2 - O.2)^2 + (C.1 - O.1)^2 = a^2 ∧ O.2 = x + a ∨ O.2 = x - a} =
  {p : ℝ × ℝ | p ∈ (λ x, {p : ℝ × ℝ | p.2 = x}) '' {h : ℝ | h = (classical.some hL).2 + a ∨ h = (classical.some hL).2 - a}} :=
by
  sorry

end locus_of_centers_of_tangent_circles_l627_627095


namespace water_to_milk_ratio_l627_627751

theorem water_to_milk_ratio 
  (V : ℝ) 
  (hV : V > 0) 
  (milk_volume1 : ℝ := (3 / 5) * V) 
  (water_volume1 : ℝ := (2 / 5) * V) 
  (milk_volume2 : ℝ := (4 / 5) * V) 
  (water_volume2 : ℝ := (1 / 5) * V)
  (total_milk_volume : ℝ := milk_volume1 + milk_volume2)
  (total_water_volume : ℝ := water_volume1 + water_volume2) :
  total_water_volume / total_milk_volume = (3 / 7) := 
  sorry

end water_to_milk_ratio_l627_627751


namespace complex_sum_equals_one_l627_627588

noncomputable def main (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1))

theorem complex_sum_equals_one (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : main x h1 h2 = 1 := by
  sorry

end complex_sum_equals_one_l627_627588


namespace cases_2_and_4_stratified_possible_l627_627378

-- Define the total number of students and their distribution.
def total_students : ℕ := 270
def first_grade_students : ℕ := 108
def second_grade_students : ℕ := 81
def third_grade_students : ℕ := 81

-- Define the individual cases of drawn numbers.
def case_1 : list ℕ := [7, 9, 100, 107, 111, 121, 180, 197, 200, 265]
def case_2 : list ℕ := [6, 33, 60, 87, 114, 141, 168, 195, 222, 249]
def case_3 : list ℕ := [30, 57, 84, 111, 138, 165, 192, 219, 246, 270]
def case_4 : list ℕ := [12, 39, 66, 93, 120, 147, 174, 201, 228, 255]

-- Define the conditions for stratified sampling.
def is_stratified_sampling (drawn_numbers : list ℕ) : Prop :=
  let first_grade_count := drawn_numbers.countp (λ n, n ≤ 108) in
  let second_grade_count := drawn_numbers.countp (λ n, n > 108 ∧ n ≤ 189) in
  let third_grade_count := drawn_numbers.countp (λ n, n > 189 ∧ n ≤ 270) in
  first_grade_count = 4 ∧ second_grade_count = 3 ∧ third_grade_count = 3

-- Statement that cases 2 and 4 could be from stratified sampling.
theorem cases_2_and_4_stratified_possible : 
  is_stratified_sampling case_2 ∧ is_stratified_sampling case_4 :=
sorry

end cases_2_and_4_stratified_possible_l627_627378


namespace Roberta_spent_on_shoes_l627_627263

-- Define the conditions as per the problem statement
variables (S B L : ℝ) (h1 : B = S - 17) (h2 : L = B / 4) (h3 : 158 - (S + B + L) = 78)

-- State the theorem to be proved
theorem Roberta_spent_on_shoes : S = 45 :=
by
  -- use variables and conditions
  have := h1
  have := h2
  have := h3
  sorry -- Proof steps can be filled later

end Roberta_spent_on_shoes_l627_627263


namespace max_value_quadratic_max_value_quadratic_attained_l627_627338

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_value_quadratic : ∀ (x : ℝ), quadratic (-8) 32 (-1) x ≤ 31 :=
by
  sorry

theorem max_value_quadratic_attained : 
  quadratic (-8) 32 (-1) 2 = 31 :=
by
  sorry

end max_value_quadratic_max_value_quadratic_attained_l627_627338


namespace gymnastics_performance_participation_l627_627307

def total_people_in_gym_performance (grades : ℕ) (classes_per_grade : ℕ) (students_per_class : ℕ) : ℕ :=
  grades * classes_per_grade * students_per_class

theorem gymnastics_performance_participation :
  total_people_in_gym_performance 3 4 15 = 180 :=
by
  -- This is where the proof would go
  sorry

end gymnastics_performance_participation_l627_627307


namespace probability_identical_cubes_same_size_painted_l627_627328

theorem probability_identical_cubes_same_size_painted :
  let total_paint_ways := 531441 in
  ∃ (matching_ways : ℕ), matching_ways = 1178 ∧ 
  (matching_ways / total_paint_ways = 1178 / 531441) :=
sorry

end probability_identical_cubes_same_size_painted_l627_627328


namespace largest_number_is_B_l627_627740

noncomputable def number_A := 9.12445
noncomputable def number_B := 9.124555555555...  -- repeating 5
noncomputable def number_C := 9.124545454545...  -- repeating 45
noncomputable def number_D := 9.1245245245245... -- repeating 245
noncomputable def number_E := 9.124512451245...  -- repeating 1245

theorem largest_number_is_B :
  number_B > number_A ∧
  number_B > number_C ∧
  number_B > number_D ∧
  number_B > number_E :=
by
  /-
    Proof omitted.
    This is where we would normally provide the proof steps showing that
    number_B (i.e., 9.12455555...) is indeed greater than the other numbers.
  -/
  sorry


end largest_number_is_B_l627_627740


namespace probability_divisor_of_8_is_half_l627_627782

theorem probability_divisor_of_8_is_half :
  let outcomes := (1 : ℕ) :: (2 : ℕ) :: (3 : ℕ) :: (4 : ℕ) :: (5 : ℕ) :: (6 : ℕ) :: (7 : ℕ) :: (8 : ℕ) :: []
  let divisors_of_8 := [ 1, 2, 4, 8 ]
  let favorable_outcomes := list.filter (λ x, x ∣ 8) outcomes
  let favorable_probability := (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)
  favorable_probability = (1 / 2 : ℚ) := by
  sorry

end probability_divisor_of_8_is_half_l627_627782


namespace ab_value_l627_627475

theorem ab_value (a b : ℝ) (h1 : a = Real.exp (2 - a)) (h2 : 1 + Real.log b = Real.exp (1 - Real.log b)) : 
  a * b = Real.exp 1 :=
sorry

end ab_value_l627_627475


namespace max_k_value_l627_627132

theorem max_k_value : ∃ (k : ℕ), (1 ≤ k) ∧ (∀ m : ℕ, (1 ≤ m → m > k → ¬(is_integer (1001 * 1002 * ... * 2005 * 2006 / 11^m))) ∧ k = 101 :=
sorry

end max_k_value_l627_627132


namespace log_graph_intersects_x_axis_l627_627289

theorem log_graph_intersects_x_axis :
  ∃ x : ℝ, x > 0 ∧ log x = 0 :=
begin
  use 1,
  split,
  { -- x > 0
    norm_num, },
  { -- log x = 0
    norm_num, },
end

end log_graph_intersects_x_axis_l627_627289


namespace car_overtakes_truck_l627_627029

theorem car_overtakes_truck 
  (car_speed : ℝ)
  (truck_speed : ℝ)
  (car_arrival_time : ℝ)
  (truck_arrival_time : ℝ)
  (route_same : Prop)
  (time_difference : ℝ)
  (car_speed_km_min : car_speed = 66 / 60)
  (truck_speed_km_min : truck_speed = 42 / 60)
  (arrival_time_difference : truck_arrival_time - car_arrival_time = 18 / 60) :
  ∃ d : ℝ, d = 34.65 := 
by {
  sorry
}

end car_overtakes_truck_l627_627029


namespace domain_of_f_l627_627669

-- Define the function
def f (x : ℝ) : ℝ := sqrt (x + 1) + 1/x

-- Define the conditions
def condition1 (x : ℝ) : Prop := x + 1 ≥ 0
def condition2 (x : ℝ) : Prop := x ≠ 0

-- State the main theorem
theorem domain_of_f : {x : ℝ | condition1 x ∧ condition2 x} = {x : ℝ | x ∈ (-1, 0) ∪ (0, 1)} :=
by
  sorry

end domain_of_f_l627_627669


namespace nat_know_albums_l627_627072

/-- Define the number of novels, comics, documentaries and crates properties --/
def novels := 145
def comics := 271
def documentaries := 419
def crates := 116
def items_per_crate := 9

/-- Define the total capacity of crates --/
def total_capacity := crates * items_per_crate

/-- Define the total number of other items --/
def other_items := novels + comics + documentaries

/-- Define the number of albums --/
def albums := total_capacity - other_items

/-- Theorem: Prove that the number of albums is equal to 209 --/
theorem nat_know_albums : albums = 209 := by
  sorry

end nat_know_albums_l627_627072


namespace counterexample_to_statement_composite_not_prime_l627_627010

theorem counterexample_to_statement_composite_not_prime
  (n : ℕ)
  (composite_22 : n = 22 → n % 2 = 0 ∧ ¬is_prime n)
  (composite_26 : n = 26 → n % 2 = 0 ∧ ¬is_prime n)
  (composite_30 : n = 30 → n % 2 = 0 ∧ ¬is_prime n)
  (composite_34 : n = 34 → n % 2 = 0 ∧ ¬is_prime n)
  (composite_35 : n = 35 → n % 5 = 0 ∧ ¬is_prime n) :
  ¬(∀ (n : ℕ), (¬is_prime n ∧ n > 1 ∧ (∃ (d : ℕ), d ≥ 2 ∧ d < n ∧ n % d = 0)) → (¬is_prime (n + 2))) → 
  is_prime 37 := by
  sorry

end counterexample_to_statement_composite_not_prime_l627_627010


namespace cyclists_meet_time_l627_627712

theorem cyclists_meet_time (v1 v2 : ℝ) (C : ℝ) (h1: v1 = 7) (h2: v2 = 8) (hC: C = 600) : 
  (C / (v1 + v2)) = 40 := 
by
  simp [h1, h2, hC]
  norm_num
  sorry

end cyclists_meet_time_l627_627712


namespace pear_percentage_increase_l627_627043

theorem pear_percentage_increase (total_pears sold_pears poached_pears canned_pears : ℕ)
  (h_total : total_pears = 42)
  (h_sold : sold_pears = 20)
  (h_poached : poached_pears = 0.5 * sold_pears)
  (h_sum : sold_pears + poached_pears + canned_pears = total_pears) :
  (canned_pears - poached_pears) / poached_pears * 100 = 20 :=
by 
  sorry

end pear_percentage_increase_l627_627043


namespace mark_money_l627_627605

theorem mark_money (M : ℝ) (h1 : M / 2 + 14 ≤ M) (h2 : M / 3 + 16 ≤ M) :
  M - (M / 2 + 14) - (M / 3 + 16) = 0 → M = 180 := by
  sorry

end mark_money_l627_627605


namespace mass_percentage_Cl_in_NaClO_is_47_61_l627_627096

def atomic_mass_Na : ℝ := 22.99
def atomic_mass_Cl : ℝ := 35.45
def atomic_mass_O : ℝ := 16.00
def molar_mass_NaClO : ℝ := atomic_mass_Na + atomic_mass_Cl + atomic_mass_O

def mass_percentage_Cl : ℝ := (atomic_mass_Cl / molar_mass_NaClO) * 100

theorem mass_percentage_Cl_in_NaClO_is_47_61 :
  mass_percentage_Cl = 47.61 :=
by
  -- Here would be the proof steps
  sorry

end mass_percentage_Cl_in_NaClO_is_47_61_l627_627096


namespace ticket_price_reduction_l627_627693

-- Definitions of the problem constants and variables
def original_price : ℝ := 50
def increase_fraction : ℝ := 1 / 3
def revenue_increase_fraction : ℝ := 1 / 4

-- New number of tickets sold after price reduction
def new_number_of_tickets_sold (x : ℝ) : ℝ := x * (1 + increase_fraction)

-- New price per ticket after reduction
def new_price_per_ticket (reduction : ℝ) : ℝ := original_price - reduction

-- Original revenue
def original_revenue (x : ℝ) : ℝ := x * original_price

-- New revenue after price reduction
def new_revenue (x reduction : ℝ) : ℝ := new_number_of_tickets_sold x * new_price_per_ticket reduction

-- The equation relating new revenue to the original revenue with the given increase
def revenue_relation (x reduction : ℝ) : Prop :=
  new_revenue x reduction = (1 + revenue_increase_fraction) * original_revenue x

-- The goal is to find the reduction in price per ticket (reduction) such that the revenue_relation holds
theorem ticket_price_reduction :
  ∃ y : ℝ, ∀ x > 0, revenue_relation x y ∧ y = 25 / 2 :=
begin
  sorry -- Proof goes here
end

end ticket_price_reduction_l627_627693


namespace natives_per_tribe_l627_627050

structure Tribe where
  members : Finset ℕ

def is_from_same_tribe (tribes : Finset Tribe) (n m : ℕ) : Prop :=
  ∃ t ∈ tribes, n ∈ t.members ∧ m ∈ t.members

variables (natives : Finset ℕ)
          (tribes : Finset Tribe)
          (tribes_count : tribes.card = 4)
          (natives_count : natives.card = 8)
          (statement : ℕ → ℕ → Prop)
          (truth : ℕ → ℕ → Prop)

-- Condition: 8 natives are sitting in a circle
-- Note: Assuming n1 neighbors n2, n2 neighbors n3, ..., n8 neighbors n1 in natives Finset

-- Condition 1: Each native makes a statement to their left neighbor: "If you don't count me, there is no one else here from my tribe."
-- natives lie to outsiders and tell the truth to their own tribe members
variables (neighbor_left : natives → ℕ)

def lying_to_neighbor_left (n : ℕ) : Prop :=
  statement n (neighbor_left n) → ¬truth n (neighbor_left n)

def telling_truth_to_neighbor_left (n : ℕ) : Prop :=
  statement n (neighbor_left n) → truth n (neighbor_left n)

def unique_member_lying (n : ℕ) : Prop :=
  ∀ m ∈ natives, ¬is_from_same_tribe tribes n m → lying_to_neighbor_left n

def unique_member_truth (n : ℕ) : Prop :=
  ∀ m ∈ natives, is_from_same_tribe tribes n m → telling_truth_to_neighbor_left n

-- Question: How many natives can there be from each tribe?
-- The correct answer we need to prove: 
theorem natives_per_tribe (tribes : Finset Tribe) (tribes_count : tribes.card = 4) 
  (natives : Finset ℕ) (natives_card : natives.card = 8) 
  (H : ∀ n ∈ natives, unique_member_truth n ∧ unique_member_lying n):
  ∀ t ∈ tribes, t.members.card = 2 := 
sorry

end natives_per_tribe_l627_627050


namespace ticket_price_reduction_l627_627696

theorem ticket_price_reduction
    (original_price : ℝ := 50)
    (increase_in_tickets : ℝ := 1 / 3)
    (increase_in_revenue : ℝ := 1 / 4)
    (x : ℝ)
    (reduced_price : ℝ)
    (new_tickets : ℝ := x * (1 + increase_in_tickets))
    (original_revenue : ℝ := x * original_price)
    (new_revenue : ℝ := new_tickets * reduced_price) :
    new_revenue = (1 + increase_in_revenue) * original_revenue →
    reduced_price = original_price - (original_price / 2) :=
    sorry

end ticket_price_reduction_l627_627696


namespace unique_triangles_count_l627_627957

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627957


namespace cookies_left_correct_l627_627609

def cookies_left (cookies_per_dozen : ℕ) (flour_per_dozen_lb : ℕ) (bag_count : ℕ) (flour_per_bag_lb : ℕ) (cookies_eaten : ℕ) : ℕ :=
  let total_flour_lb := bag_count * flour_per_bag_lb
  let total_cookies := (total_flour_lb / flour_per_dozen_lb) * cookies_per_dozen
  total_cookies - cookies_eaten

theorem cookies_left_correct :
  cookies_left 12 2 4 5 15 = 105 :=
by sorry

end cookies_left_correct_l627_627609


namespace first_group_persons_l627_627649

def Work (persons : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  persons * days * hours_per_day

theorem first_group_persons
  (P : ℕ) -- number of persons in the first group
  (Work1_is_correct : Work P 12 5 = Work 30 23 6) : P = 69 :=
by
  -- provided fact
  have h₁ : Work P 12 5 = P * 12 * 5 := by rfl
  -- total work done by the second group
  have h₂ : Work 30 23 6 = 30 * 23 * 6 := by rfl
  -- equating the two works
  have h₃ : P * 12 * 5 = 30 * 23 * 6 := by
    rw [←h₁, ←h₂]
    exact Work1_is_correct
  -- solving for P
  have h₄ : P = (30 * 23 * 6) / (12 * 5) := mul_left_inj' (by norm_num : (12 * 5) ≠ 0) h₃
  -- simplify to find P = 69
  norm_num at h₄
  exact h₄

end first_group_persons_l627_627649


namespace max_constant_C_l627_627105

theorem max_constant_C :
  ∃ C : ℝ, (∀ a b : ℝ, max (|a + b|) (max (|a - b|) (|2006 - b|)) ≥ C) ∧ C = 1003 :=
by
  use 1003
  intro a b
  have h1 : |a + b| ≤ max (|a + b|) (max (|a - b|) (|2006 - b|)) := le_max_of_le_left (le_max_left _ _)
  have h2 : |a - b| ≤ max (|a + b|) (max (|a - b|) (|2006 - b|)) := le_max_of_le_left (le_max_right _ _)
  have h3 : |2006 - b| ≤ max (|a + b|) (max (|a - b|) (|2006 - b|)) := le_max_right _ _
  exact max_le (max_le h1 h2) h3
  sorry

end max_constant_C_l627_627105


namespace gender_matching_probability_l627_627265

def SchoolA : Type := {a // a = "m" ∨ a = "m" ∨ a = "f"}
def SchoolB : Type := {b // b = "m" ∨ b = "f" ∨ b = "f"}

noncomputable def matching_gender_probability : ℚ :=
  -- Defining sets of males and females in each school
  let males_in_a := {"m", "m"} in
  let females_in_a := {"f"} in
  let males_in_b := {"m"} in
  let females_in_b := {"f", "f"} in

  -- Counting the gender matches
  let male_pairs := (males_in_a.card : ℕ) * males_in_b.card in
  let female_pairs := (females_in_a.card : ℕ) * females_in_b.card in

  -- Total possible pairs and favorable pairs
  let total_pairs := (males_in_a.card + females_in_a.card) * (males_in_b.card + females_in_b.card) in
  let favorable_pairs := male_pairs + female_pairs in

  -- Probability calculation
  (favorable_pairs : ℚ) / (total_pairs : ℚ)

theorem gender_matching_probability : matching_gender_probability = 4/9 := by sorry

end gender_matching_probability_l627_627265


namespace time_to_empty_basket_total_people_entered_l627_627380

-- Given conditions
variable X : ℝ

-- Definitions
def T (X : ℝ) : ℝ := 4634 / (2 * X)
def people_entered (X : ℝ) : ℝ := 4634 / 2

-- Problem statements
theorem time_to_empty_basket (X : ℝ) : 
  T X = 4634 / (2 * X) := by
  sorry

theorem total_people_entered (X : ℝ) : 
  people_entered X = 2317 := by
  sorry

end time_to_empty_basket_total_people_entered_l627_627380


namespace median_first_twenty_positive_integers_l627_627719

theorem median_first_twenty_positive_integers : 
  let s := {i | 1 ≤ i ∧ i ≤ 20} in
  let sorted_s := List.range' 1 20 in
  (sorted_s[9] + sorted_s[10]) / 2 = 10.5 :=
by {
let s := {i | 1 ≤ i ∧ i ≤ 20},
let sorted_s := List.range' 1 20, -- range' starts at 1 and goes up to 20 (exclusive), thus takes numbers 1 to 20
have h1 : sorted_s[9] = 10 := by rfl,
have h2 : sorted_s[10] = 11 := by rfl,
have h3 : (sorted_s[9] + sorted_s[10]) = 21 := by rw [h1, h2]; rfl,
show (sorted_s[9] + sorted_s[10]) / 2 = 10.5, by rw [h3]; norm_num
}

end median_first_twenty_positive_integers_l627_627719


namespace number_of_subsets_of_union_l627_627131

open Finset

variable (A B : Finset ℕ)
variable hypA : A = {1, 2}
variable hypB : B = {0, 1}

theorem number_of_subsets_of_union : (A ∪ B).card = 3 → (A ∪ B).powerset.card = 8 :=
by
  intros h
  rw [←powerset_card]
  rw [h]
  norm_num
  sorry

end number_of_subsets_of_union_l627_627131


namespace monthly_charge_l627_627798

theorem monthly_charge (weekly_charge monthly_savings : ℤ) (weeks_in_year months_in_year : ℕ) (M : ℤ) :
  weekly_charge = 10 →
  monthly_savings = 40 →
  weeks_in_year = 52 →
  months_in_year = 12 →
  (weeks_in_year * weekly_charge) - (months_in_year * M) = monthly_savings →
  M = 40 :=
by
  intros h1 h2 h3 h4 h5
  calc
    M = (weeks_in_year * weekly_charge - monthly_savings) / months_in_year : by sorry
    ... = 40 : by sorry

end monthly_charge_l627_627798


namespace interval_of_increase_correct_l627_627069

-- Define the function f
def f (x : ℝ) : ℝ := logBase (1/2) (3 - 2 * x - x ^ 2)

-- Define the domain condition
def domain_condition (x : ℝ) : Prop := 3 - 2 * x - x ^ 2 > 0

-- Define the interval of increase
def interval_of_increase : set ℝ := {x | -1 < x ∧ x < 1}

-- The statement to prove
theorem interval_of_increase_correct : ∀ x, domain_condition x → x ∈ interval_of_increase :=
sorry

end interval_of_increase_correct_l627_627069


namespace tan_ratio_l627_627459

-- Definitions of alpha and beta within the given interval
variables {α β : ℝ}
axiom h1 : α ∈ Ioo 0 (Real.pi / 2)
axiom h2 : β ∈ Ioo 0 (Real.pi / 2)

-- Given trigonometric condition
axiom h3 : Real.sin (α + β) = 3 * Real.sin (Real.pi - α + β)

-- Target proof statement
theorem tan_ratio (h1 : α ∈ Ioo 0 (Real.pi / 2)) (h2 : β ∈ Ioo 0 (Real.pi / 2)) (h3 : Real.sin (α + β) = 3 * Real.sin (Real.pi - α + β)) :
  Real.tan α / Real.tan β = 2 :=
sorry

end tan_ratio_l627_627459


namespace probability_divisor_of_8_on_8_sided_die_l627_627786

def divisor_probability : ℚ :=
  let sample_space := {1, 2, 3, 4, 5, 6, 7, 8}
  let divisors_of_8 := {1, 2, 4, 8}
  let favorable_outcomes := divisors_of_8 ∩ sample_space
  favorable_outcomes.card / sample_space.card

theorem probability_divisor_of_8_on_8_sided_die :
  divisor_probability = 1 / 2 :=
sorry

end probability_divisor_of_8_on_8_sided_die_l627_627786


namespace phi_consists_of_one_part_l627_627833

-- Definitions for the conditions
def condition1 (x y : ℝ) : Prop := (sqrt(y^2 - 8*x^2 - 6*y + 9)) ≤ (3*y - 1)
def condition2 (x y : ℝ) : Prop := (x^2 + y^2) ≤ 9

-- The main theorem to be proven
theorem phi_consists_of_one_part :
  (∀ (x y : ℝ), condition1 x y ∧ condition2 x y) → true :=
by 
  -- This is where the proof would go, but it is assumed by "sorry"
  sorry

end phi_consists_of_one_part_l627_627833


namespace find_f_parity_and_monotonicity_range_of_k_l627_627117

variable {a x t k : ℝ}

noncomputable def f (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

-- Conditions for a
variable (h_pos : a > 0) (h_neq_one : a ≠ 1)

-- Proof that f(x) is correctly defined as given in problem (1).
theorem find_f (x : ℝ) : f x = (a / (a^2 - 1)) * (a^x - a^(-x)) := sorry

-- Proof that f(x) is odd and increasing as in (2).
theorem parity_and_monotonicity : f (-x) = -f x ∧ ∀ x1 x2, x1 < x2 → f x1 < f x2 := sorry

-- Proof for the range of k as in (3).
theorem range_of_k (f_increasing : ∀ x1 x2, x1 < x2 → f x1 < f x2) 
                    (h_inequality : ∀ t, f(t^2 - 2 * t) + f(2 * t^2 - k) > 0) : k < -1/3 := sorry

end find_f_parity_and_monotonicity_range_of_k_l627_627117


namespace min_degree_of_g_l627_627574

open Polynomial

noncomputable def min_deg_g (f k g : Polynomial ℝ) : ℕ :=
  if 5 * f + 6 * g = k ∧ degree f = 10 ∧ degree k = 12 then
    degree g
  else
    0

theorem min_degree_of_g (f g k : Polynomial ℝ) 
  (h : 5 * f + 6 * g = k) (hf : degree f = 10) (hk : degree k = 12) :
  degree g ≥ 12 :=
sorry

end min_degree_of_g_l627_627574


namespace frog_return_prob_A_after_2022_l627_627815

def initial_prob_A : ℚ := 1
def transition_prob_A_to_adj : ℚ := 1/3
def transition_prob_adj_to_A : ℚ := 1/3
def transition_prob_adj_to_adj : ℚ := 2/3

noncomputable def prob_A_return (n : ℕ) : ℚ :=
if (n % 2 = 0) then
  (2/9) * (1/2^(n/2)) + (1/9)
else
  0

theorem frog_return_prob_A_after_2022 : prob_A_return 2022 = (2/9) * (1/2^1010) + (1/9) :=
by
  sorry

end frog_return_prob_A_after_2022_l627_627815


namespace count_non_congruent_triangles_with_perimeter_18_l627_627941

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627941


namespace part1_part2_case1_part2_case2_part2_case3_l627_627461

namespace InequalityProof

variable {a x : ℝ}

def f (a x : ℝ) := a * x^2 + x - a

theorem part1 (h : a = 1) : (x > 1 ∨ x < -2) → f a x > 1 :=
by sorry

theorem part2_case1 (h1 : a < 0) (h2 : a < -1/2) : (- (a + 1) / a) < x ∧ x < 1 → f a x > 1 :=
by sorry

theorem part2_case2 (h1 : a < 0) (h2 : a = -1/2) : x ≠ 1 → f a x > 1 :=
by sorry

theorem part2_case3 (h1 : a < 0) (h2 : 0 > a) (h3 : a > -1/2) : 1 < x ∧ x < - (a + 1) / a → f a x > 1 :=
by sorry

end InequalityProof

end part1_part2_case1_part2_case2_part2_case3_l627_627461


namespace smallest_n_for_convex_100gon_l627_627857

noncomputable def smallest_n_intersection (sides : ℕ) (triangles : ℕ) : ℕ :=
  if sides = 100 then
    if triangles = 50 then
      50
    else
      0
else 
  0

theorem smallest_n_for_convex_100gon (n : ℕ) :
  (∀ (P : Type) [poly : convex_polygon P] (sides : ℕ), sides = 100 → P s = sides
  → ∃ (t : Type → Type) [triangles t] (n : ℕ), t n = triangles ∧ n = 50) :=
sorry

end smallest_n_for_convex_100gon_l627_627857


namespace percentage_gain_proof_l627_627767

-- Let C be the cost price of one book
variable (C : ℝ)
-- Let S be the selling price of one book
variable (S : ℝ)

-- Condition: The cost price of 50 books equals the selling price of 35 books
axiom h : 50 * C = 35 * S

-- Define the percentage gain
def percentage_gain (C S : ℝ) : ℝ := ((S - C) / C) * 100

-- Prove that the percentage gain is 42.86%
theorem percentage_gain_proof : percentage_gain C S = 42.86 :=
by
  -- From the condition, solve for S in terms of C
  have h1 : S = (50 * C) / 35 := by sorry
  -- Substitute back into the formula for percentage_gain
  sorry

end percentage_gain_proof_l627_627767


namespace problem_proof_l627_627883

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 4 - 3 * x
def g (x : ℝ) : ℝ := x^2 + x + 1

-- The theorem stating the equivalence
theorem problem_proof : f(g(real.sqrt 3)) = -8 - 3 * real.sqrt 3 := by
  sorry

end problem_proof_l627_627883


namespace unique_plants_in_all_beds_l627_627112

theorem unique_plants_in_all_beds:
  let A := 600
  let B := 500
  let C := 400
  let D := 300
  let AB := 80
  let AC := 70
  let ABD := 40
  let BC := 0
  let AD := 0
  let BD := 0
  let CD := 0
  let ABC := 0
  let ACD := 0
  let BCD := 0
  let ABCD := 0
  A + B + C + D - AB - AC - BC - AD - BD - CD + ABC + ABD + ACD + BCD - ABCD = 1690 :=
by
  sorry

end unique_plants_in_all_beds_l627_627112


namespace ratio_of_squares_l627_627538

theorem ratio_of_squares (r : ℝ) :
  let new_radius := 2 * r,
      new_diameter := 2 * new_radius,
      new_circumference := 2 * Real.pi * new_radius,
      square_of_new_circumference := new_circumference^2,
      square_of_new_diameter := new_diameter^2
  in (square_of_new_circumference / square_of_new_diameter) = Real.pi^2 :=
by
  intros
  sorry

end ratio_of_squares_l627_627538


namespace min_odd_is_1_l627_627329

def min_odd_integers (a b c d e f : ℤ) : ℤ :=
  if (a + b) % 2 = 0 ∧ 
     (a + b + c + d) % 2 = 1 ∧ 
     (a + b + c + d + e + f) % 2 = 0 then
    1
  else
    sorry -- This should be replaced by a calculation of the true minimum based on conditions.

def satisfies_conditions (a b c d e f : ℤ) :=
  a + b = 30 ∧ 
  a + b + c + d = 47 ∧ 
  a + b + c + d + e + f = 65

theorem min_odd_is_1 (a b c d e f : ℤ) (h : satisfies_conditions a b c d e f) : 
  min_odd_integers a b c d e f = 1 := 
sorry

end min_odd_is_1_l627_627329


namespace max_real_roots_of_polynomial_l627_627439

theorem max_real_roots_of_polynomial (n : ℕ) (c : ℝ) (h_pos : 0 < n) (h_c : c ≠ 1) :
  (∃ x : ℝ, x^n + x^(n-1) + ... + x^2 + x + c = 0) → ∃ x' : ℝ, ∀ x : ℝ, 
  x^n + x^(n-1) + ... + x^2 + x + c = 0 → x = x' :=
sorry

end max_real_roots_of_polynomial_l627_627439


namespace min_value_trig_expression_l627_627463

theorem min_value_trig_expression (x : ℝ) : 
  ∃ y : ℝ, y = (4 * sin x * cos x + 3) / (cos x ^ 2) ∧ 
           (∀ z : ℝ, ∃ t : ℝ, t = (4 * sin z * cos z + 3) / (cos z ^ 2) → y ≤ t) := 
begin
  use 5 / 3,
  split,
  { sorry }, -- Proof that y equals (4 * sin x * cos x + 3) / (cos x ^ 2)
  { intros z,
    use (4 * sin z * cos z + 3) / (cos z ^ 2),
    intro h,
    sorry } -- Proof that 5/3 is the minimum value
end

end min_value_trig_expression_l627_627463


namespace probability_multiple_choice_and_essay_correct_l627_627794

noncomputable def probability_multiple_choice_and_essay (C : ℕ → ℕ → ℕ) : ℚ :=
    (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3)

theorem probability_multiple_choice_and_essay_correct (C : ℕ → ℕ → ℕ) :
    probability_multiple_choice_and_essay C = (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3) :=
by
  sorry

end probability_multiple_choice_and_essay_correct_l627_627794


namespace max_sqrt_expression_l627_627097

theorem max_sqrt_expression : 
  ∃ x ∈ set.Icc (0 : ℝ) 25, 
  ∀ y ∈ set.Icc (0 : ℝ) 25, 
  sqrt (y + 64) + 2 * sqrt (25 - y) + sqrt y ≤ sqrt 328 ∧ 
  sqrt (x + 64) + 2 * sqrt (25 - x) + sqrt x = sqrt 328 :=
by 
  use 20
  split
  sorry

end max_sqrt_expression_l627_627097


namespace TriangleIsEquilateral_l627_627051

theorem TriangleIsEquilateral 
  (A B C E F : Point) 
  (CE_median : is_median CE A B C) 
  (AF_median : is_median AF A B C) 
  (angle_EAF_30 : ∠ A E F = 30°) 
  (angle_ECF_30 : ∠ C E F = 30°)
  : is_equilateral_triangle A B C :=
sorry

end TriangleIsEquilateral_l627_627051


namespace average_squares_of_first_10_multiples_of_7_correct_l627_627448

def first_10_multiples_of_7 : List ℕ := List.map (fun n => 7 * n) (List.range 10)

def squares (l : List ℕ) : List ℕ := List.map (fun n => n * n) l

def sum (l : List ℕ) : ℕ := List.foldr (· + ·) 0 l

theorem average_squares_of_first_10_multiples_of_7_correct :
  (sum (squares first_10_multiples_of_7) / 10 : ℚ) = 1686.5 :=
by
  sorry

end average_squares_of_first_10_multiples_of_7_correct_l627_627448


namespace range_of_a_not_monotonic_l627_627684

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x ^ 2 + a * x - 5

def derivative_of_f (a : ℝ) : ℝ → ℝ := 
  λ x, x^2 - 2 * x + a

theorem range_of_a_not_monotonic :
  {a : ℝ | ∃ x ∈ set.Icc (-1 : ℝ) 2, derivative_of_f a x = 0} = set.Ioo (-3 : ℝ) 1 :=
sorry

end range_of_a_not_monotonic_l627_627684


namespace sum_of_obtuse_angles_l627_627897

-- Define the given conditions
variable (A B : Real)
variable (hA : π / 2 < A ∧ A < π) -- A is obtuse
variable (hB : π / 2 < B ∧ B < π) -- B is obtuse
variable (h_sinA : Real.sin A = √5 / 5)
variable (h_sinB : Real.sin B = √10 / 10)

-- The statement we need to prove
theorem sum_of_obtuse_angles (hA : π / 2 < A ∧ A < π) (hB : π / 2 < B ∧ B < π) 
(h_sinA : Real.sin A = √5 / 5) (h_sinB : Real.sin B = √10 / 10) : 
  A + B = 7 * π / 4 := 
  sorry

end sum_of_obtuse_angles_l627_627897


namespace sum_of_converted_2016_is_correct_l627_627638

theorem sum_of_converted_2016_is_correct :
  (20.16 + 20.16 + 20.16 + 201.6 + 201.6 + 201.6 = 463.68 ∨
   2.016 + 2.016 + 2.016 + 20.16 + 20.16 + 20.16 = 46.368) :=
by
  sorry

end sum_of_converted_2016_is_correct_l627_627638


namespace find_length_of_train_l627_627038

noncomputable def length_of_train (v : ℝ) (L_p : ℝ) (t : ℝ) : ℝ :=
  let v_m_s := v * 1000 / 3600
  let total_distance := v_m_s * t
  total_distance - L_p

theorem find_length_of_train :
  length_of_train 55 520 57.59539236861051 = 360 :=
  by
    delta length_of_train
    simp [*, show_real]
    norm_num
    sorry

end find_length_of_train_l627_627038


namespace unique_triangles_count_l627_627952

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627952


namespace strawberry_pancakes_l627_627066

theorem strawberry_pancakes (total blueberry banana chocolate : ℕ) (h_total : total = 150) (h_blueberry : blueberry = 45) (h_banana : banana = 60) (h_chocolate : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 :=
by
  sorry

end strawberry_pancakes_l627_627066


namespace sum_mod_13_l627_627737

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l627_627737


namespace max_true_statements_l627_627225

theorem max_true_statements
  (a b : ℝ)
  (h₁ : a < 0)
  (h₂ : b < 0)
  (h₃ : a < b) :
  ∃ S : finset ℕ, S.card = 4 ∧
    ((1 ∈ S → ¬ (1 / a < 1 / b)) ∧
     (2 ∈ S → a^3 < b^3) ∧
     (3 ∈ S → a < b) ∧
     (4 ∈ S → a < 0) ∧
     (5 ∈ S → b < 0)) :=
by
  sorry

end max_true_statements_l627_627225


namespace part_A_part_B_part_D_l627_627710

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Part A: single transmission probability
theorem part_A (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 :=
by sorry

-- Part B: triple transmission probability
theorem part_B (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  β * (1 - β)^2 = β * (1 - β)^2 :=
by sorry

-- Part D: comparing single and triple transmission
theorem part_D (α β : ℝ) (hα : 0 < α ∧ α < 0.5) (hβ : 0 < β ∧ β < 1) :
  (1 - α) < (1 - α)^3 + 3 * α * (1 - α)^2 :=
by sorry

end part_A_part_B_part_D_l627_627710


namespace max_lambda_correct_l627_627103

noncomputable def max_lambda (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : ℝ :=
  2 * Real.sqrt 3

theorem max_lambda_correct :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a + b + c = 1 →
    ∀ λ, (a^2 + b^2 + c^2 + λ * Real.sqrt (a * b * c) ≤ 1) ↔ λ ≤ max_lambda a b c (by assumption) (by assumption) :=
by
  intro a b c h_pos h_sum λ
  sorry

end max_lambda_correct_l627_627103


namespace gcd_180_450_l627_627087

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l627_627087


namespace concentration_of_mixture_l627_627395

theorem concentration_of_mixture (c1 c2 : ℝ) (v1 v2 : ℝ) (total_volume : ℝ) :
  c1 = 0.30 → v1 = 2 → c2 = 0.40 → v2 = 6 → total_volume = 10 →
  ((c1 * v1 + c2 * v2) / total_volume) * 100 = 30 := by
  intros hc1 hv1 hc2 hv2 ht
  have h_alcohol_amount := hc1 * hv1 + hc2 * hv2
  have h_total_volume := ht
  have h_concentration := (h_alcohol_amount / h_total_volume) * 100
  sorry

end concentration_of_mixture_l627_627395


namespace inverse_proportion_m_range_l627_627106

theorem inverse_proportion_m_range (m : ℝ) :
  (∀ x : ℝ, x < 0 → ∀ y1 y2 : ℝ, y1 = (1 - 2 * m) / x → y2 = (1 - 2 * m) / (x + 1) → y1 < y2) 
  ↔ (m > 1 / 2) :=
by sorry

end inverse_proportion_m_range_l627_627106


namespace positive_difference_between_sums_is_zero_l627_627017

def original_matrix : Matrix (Fin 4) (Fin 4) ℕ := ![
  ![5, 6, 7, 8],
  ![9, 10, 11, 12],
  ![13, 14, 15, 16],
  ![17, 18, 19, 20]
]

def modified_matrix : Matrix (Fin 4) (Fin 4) ℕ := ![
  ![5, 6, 7, 8],
  ![12, 11, 10, 9],
  ![16, 17, 18, 19],
  ![20, 23, 22, 21]
]

def main_diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2 + m 3 3

def anti_diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ :=
  m 0 3 + m 1 2 + m 2 1 + m 3 0

theorem positive_difference_between_sums_is_zero :
  (main_diagonal_sum modified_matrix - anti_diagonal_sum modified_matrix).natAbs = 0 :=
by
  sorry

end positive_difference_between_sums_is_zero_l627_627017


namespace main_problem_l627_627760

variable {n : ℕ} {x y : ℤ}

theorem main_problem (h : x ≠ y) :
  (∃ n, x + x^2 + x^4 + ... + x^(2^n) = y + y^2 + y^4 + ... + y^(2^n))
  → (n = 1 ∧ ∃ k : ℤ, x = k ∧ y = -k - 1) ∨ (n ≥ 2 ∧ ∀ x y : ℤ, ¬ (x + x^2 + x^4 + ... + x^(2^n) = y + y^2 + y^4 + ... + y^(2^n))) :=
by
  sorry

end main_problem_l627_627760


namespace base7_to_base10_l627_627334

theorem base7_to_base10 : 
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  digit0 * base^0 + digit1 * base^1 + digit2 * base^2 + digit3 * base^3 = 1934 :=
by
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  sorry

end base7_to_base10_l627_627334


namespace circle_theorem_l627_627236

noncomputable def circle_problem (A B C D O : Point) (AB : LineSegment A B) (CD : Diameter C D) : Prop :=
  let S_ΔCAB := area (triangle C A B)
  let S_ΔDAB := area (triangle D A B)
  let S_ΔOAB := area (triangle O A B)
  let M := |S_ΔCAB - S_ΔDAB|
  let N := 2 * S_ΔOAB
  M = N

-- A statement that these conditions hold true
theorem circle_theorem (A B C D O : Point) (AB : LineSegment A B) (CD : Diameter C D) :
  intersects (Chord AB) (Diameter CD) → circle_problem A B C D O AB CD :=
by
  sorry

end circle_theorem_l627_627236


namespace probability_divisor_of_8_is_half_l627_627790

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l627_627790


namespace non_congruent_triangles_with_perimeter_18_l627_627983

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627983


namespace find_c_plus_d_l627_627517

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l627_627517


namespace four_digit_even_numbers_five_digit_multiples_of_5_l627_627330

-- Four-digit even numbers with no repeated digits using 0, 1, 2, 3, 4, 5
theorem four_digit_even_numbers (l : List ℕ) (h : l = [0, 1, 2, 3, 4, 5]) :
  (∃ (n : ℕ), four_digit_even l n ∧ no_repeated_digits n) → 156 :=
sorry

-- Five-digit numbers that are multiples of 5 with no repeated digits using 0, 1, 2, 3, 4, 5
theorem five_digit_multiples_of_5 (l : List ℕ) (h : l = [0, 1, 2, 3, 4, 5]) :
  (∃ (n : ℕ), five_digit_multiple_of_5 l n ∧ no_repeated_digits n) → 216 :=
sorry

-- Definitions used in theorems
def four_digit_even (l : List ℕ) (n : ℕ) : Prop :=
  is_four_digit n ∧ is_even n ∧ ∀ d ∈ (digits n), d ∈ l

def five_digit_multiple_of_5 (l : List ℕ) (n : ℕ) : Prop :=
  is_five_digit n ∧ is_multiple_of_5 n ∧ ∀ d ∈ (digits n), d ∈ l

def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def no_repeated_digits (n : ℕ) : Prop :=
  let ds := digits n in
  list.nodup ds

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else aux n []
where aux : ℕ → List ℕ → List ℕ
  | 0, ds => ds
  | n, ds => aux (n / 10) (Nat.mod n 10 :: ds)

end four_digit_even_numbers_five_digit_multiples_of_5_l627_627330


namespace non_congruent_triangles_with_perimeter_18_l627_627998

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l627_627998


namespace repeating_prime_exists_l627_627893

open Nat

theorem repeating_prime_exists (p : Fin 2021 → ℕ) 
  (prime_seq : ∀ i : Fin 2021, Nat.Prime (p i))
  (diff_condition : ∀ i : Fin 2019, (p (i + 1) - p i = 6 ∨ p (i + 1) - p i = 12) ∧ (p (i + 2) - p (i + 1) = 6 ∨ p (i + 2) - p (i + 1) = 12)) : 
  ∃ i j : Fin 2021, i ≠ j ∧ p i = p j := by
  sorry

end repeating_prime_exists_l627_627893


namespace part1_a3_a4_a5_a6_part1_arithmetic_sequence_part2_T_n_l627_627501

noncomputable def a : ℕ → ℝ
| 1 := 1
| 2 := 2
| (n + 2) := (1 + (Real.cos ((n:ℝ) * π / 2)) ^ 2) * a n + (Real.sin ((n:ℝ) * π / 2)) ^ 2

theorem part1_a3_a4_a5_a6 :
  a 3 = 2 ∧ a 4 = 4 ∧ a 5 = 3 ∧ a 6 = 8 := 
sorry

theorem part1_arithmetic_sequence (k : ℕ) (hk : k > 0) :
  a (2 * k - 1) = k :=
sorry

noncomputable def b (n : ℕ) : ℝ :=
1 / (a (2 * n - 1) * Real.sqrt (a (2 * n + 1)) + a (2 * n + 1) * Real.sqrt (a (2 * n - 1)))

noncomputable def T (n : ℕ) : ℝ :=
∑ i in Finset.range (n + 1), b i

theorem part2_T_n (n : ℕ) :
  T n = 1 - 1 / (Real.sqrt (n + 1)) :=
sorry

end part1_a3_a4_a5_a6_part1_arithmetic_sequence_part2_T_n_l627_627501


namespace roger_total_experience_l627_627101

-- Define the necessary variables: R, P, T, Rb, M
variables (R P T Rb M : ℕ)

-- Define the conditions provided in the problem
def condition1 : Prop := R = P + T + Rb + M
def condition2 : Prop := P = 19 - 7
def condition3 : Prop := T = 2 * Rb
def condition4a : Prop := Rb = P - 4
def condition4b : Prop := Rb = M + 2
def condition5 : Prop := true  -- Roger has to work 8 more years before he retires

-- Prove the target value
def target_value : Prop := R + 8 = 50

-- Main theorem stating that if all conditions hold, then target_value holds
theorem roger_total_experience : condition1 → condition2 → condition3 → condition4a → condition4b → condition5 → target_value :=
by
  -- Introduce all variables (R, P, T, Rb, M) and conditions to the context
  intro h1 h2 h3 h4a h4b h5,
  rw [condition2, condition4a, condition4b] at h1,  -- Replace P, Rb, and M with their calculations
  sorry  -- The proof step is skipped, but this is where the remaining calculations would go

end roger_total_experience_l627_627101


namespace unique_triangles_count_l627_627959

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627959


namespace arnold_and_danny_age_l627_627049

theorem arnold_and_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 9) : x = 4 :=
sorry

end arnold_and_danny_age_l627_627049


namespace chord_segment_lengths_l627_627216

theorem chord_segment_lengths (R : ℝ) (OM : ℝ) (AB : ℝ) (M : ℝ) :
  R = 15 → OM = 13 → AB = 18 → 
  let CB := 9 in
  let OC := real.sqrt (R^2 - CB^2) in
  let MC := real.sqrt (OM^2 - OC^2) in
  (CB + MC = 14) ∧ (CB - MC = 4) :=
by
  intros hR hOM hAB
  let CB := 9
  let OC := real.sqrt (R^2 - CB^2)
  let MC := real.sqrt (OM^2 - OC^2)
  have hCB : CB = 9, from rfl
  have hOC : OC = real.sqrt (15^2 - 9^2), from sorry
  have hMC : MC = real.sqrt (13^2 - 12^2), from sorry
  split
  case left =>
    rw [←hCB, ←hMC]
    exact sorry
  case right =>
    rw [←hCB, ←hMC]
    exact sorry

end chord_segment_lengths_l627_627216


namespace same_gender_probability_l627_627268

-- Define the conditions as types and constants
def SchoolA : Type := {a : String // a = "A" ∨ a = "B" ∨ a = "1"}
def SchoolB : Type := {b : String // b = "C" ∨ b = "2" ∨ b = "3"}

def MasculineA (teacher : SchoolA) : Prop := teacher.val = "A" ∨ teacher.val = "B"
def FeminineA (teacher : SchoolA) : Prop := teacher.val = "1"

def MasculineB (teacher : SchoolB) : Prop := teacher.val = "C"
def FeminineB (teacher : SchoolB) : Prop := teacher.val = "2" ∨ teacher.val = "3"

-- Define a function to count the favorable pairs
def count_same_gender_pairs : ℕ := 
  if h: (MasculineA ⊓ MasculineB).exists ∧ (FeminineA ⊓ FeminineB).exists then 4 else 0

-- Definition of total pairs
def total_pairs : ℕ := 9

-- Definition of probability as a fraction
def probability : ℚ := (count_same_gender_pairs : ℚ) / (total_pairs : ℚ)

-- The theorem to state
theorem same_gender_probability : probability = 4 / 9 := sorry

end same_gender_probability_l627_627268


namespace number_of_combinations_l627_627369

noncomputable def jack_roll_combinations : Nat :=
  let kinds : Nat := 4
  let total_rolls : Nat := 10
  -- Calculate total combinations according to the solution:
  -- Case 1: 4 combinations
  let case1 : Nat := 4
  -- Case 2: 4 combinations
  let case2 : Nat := 4
  -- Case 3: 12 combinations
  let case3 : Nat := 12
  -- Case 4: 24 combinations
  let case4 : Nat := 24
  case1 + case2 + case3 + case4

theorem number_of_combinations (kinds : Nat) (total_rolls : Nat) (at_least_one_each : Nat) : jack_roll_combinations = 44 :=
by
  have h1 : kinds = 4 := rfl
  have h2 : total_rolls = 10 := rfl
  have h3 : at_least_one_each = 1 := rfl
  -- Combine results from different cases
  show jack_roll_combinations = 44
  sorry

end number_of_combinations_l627_627369


namespace incorrect_conclusion_l627_627877

def Event (Ω : Type) := set Ω

variable {Ω : Type}
variable (A B C : Event Ω)

-- Conditions
def eventA : Prop := ∀ ω ∈ A, ω ∉ B ∧ ω ∉ C
def eventB : Prop := ∀ ω ∈ B, ω ∉ A ∧ ω ∉ C
def eventC : Prop := ∀ ω ∈ C, ω ∉ A ∧ ω ∉ B

-- Definitions of mutual exclusivity and complementarity
def mutually_exclusive (X Y : Event Ω) : Prop := ∀ ω, ω ∈ X → ω ∉ Y
def complementary (X Y : Event Ω) : Prop := 
  (∀ ω, ω ∈ X ↔ ω ∉ Y) ∧ ∀ ω, ω ∈ X ∨ ω ∈ Y

-- Proof Problem
theorem incorrect_conclusion :
  ¬ complementary A B :=
by sorry

end incorrect_conclusion_l627_627877


namespace concurrency_condition_l627_627556

-- Defining the necessary geometric entities and conditions
variables (A B C D E F : Type)
variables [InnerProductSpace ℝ A B C D E F]
variables (angle : A → A → A → ℝ)
variables h1 : ∀ (a b c : A), angle a b c ∈ set.Ico 0 (2 * π) -- constraint on angle range
variables (α β x y : ℝ)

-- Assumptions based on the problem statement
def problem_conditions : Prop :=
  angle B A E = angle C A F ∧
  angle A B D = angle C B F

-- Statement of the theorem
theorem concurrency_condition (h : problem_conditions A B C D E F angle) :
  (/* AD, BE, and CF are concurrent*/) ↔ angle A C D = angle B C E :=
sorry

end concurrency_condition_l627_627556


namespace least_x1_divides_x2006_2006_l627_627452

theorem least_x1_divides_x2006_2006 (x_1 : ℕ) (h_pos : x_1 > 0)
    (x : ℕ → ℕ)
    (h_seq : ∀ n ≥ 1, x (n+1) = (Finset.range n).sum (λ i, x_1^2)) :
    (2006 ∣ x 2006) ↔ x_1 = 531 :=
begin
  sorry
end

end least_x1_divides_x2006_2006_l627_627452


namespace range_of_k_l627_627179

variable (k x : ℝ)

def f (k x : ℝ) : ℝ := k * x - Real.log x

def f' (k x : ℝ) : ℝ := k - 1/x

theorem range_of_k :
  (∀ x : ℝ, 1 < x → f' k x ≥ 0) ↔ k ∈ Set.Ici 1 := by
  sorry

end range_of_k_l627_627179


namespace acute_triangle_exists_l627_627803

theorem acute_triangle_exists
  (red_tri blue_tri green_tri : Triangle)
  (h_red : red_tri.acute)
  (h_blue : blue_tri.acute)
  (h_green : green_tri.acute)
  (circumscribed : ∀ (t : Triangle), t ⊆ circle) :
  ∃ (P : red_tri.vertex) (Q : blue_tri.vertex) (R : green_tri.vertex), 
    ¬ obtuse_triangle P Q R :=
by
  sorry

end acute_triangle_exists_l627_627803


namespace find_parabola_equation_l627_627379

noncomputable def parabola_equation (a : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A : ℝ × ℝ), 
    F.1 = a / 4 ∧ F.2 = 0 ∧
    A.1 = 0 ∧ A.2 = a / 2 ∧
    (abs (F.1 * A.2) / 2) = 4

theorem find_parabola_equation :
  ∀ (a : ℝ), parabola_equation a → a = 8 ∨ a = -8 :=
by
  sorry

end find_parabola_equation_l627_627379


namespace Henry_trays_per_trip_l627_627932

theorem Henry_trays_per_trip (trays1 trays2 trips : ℕ) (h1 : trays1 = 29) (h2 : trays2 = 52) (h3 : trips = 9) :
  (trays1 + trays2) / trips = 9 :=
by
  sorry

end Henry_trays_per_trip_l627_627932


namespace sum_grouped_sequence_l627_627430

theorem sum_grouped_sequence : 
  let groups := list.map (λi : ℕ, i * 3 + 1 + i * 3 + 2 - (i * 3 + 3)) (list.range 70) in
  list.sum groups = 7245 := 
by 
  sorry

end sum_grouped_sequence_l627_627430


namespace parabola_functions_eq_l627_627252

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (c : ℝ) (b : ℝ) : ℝ := x^2 + c * x + b

theorem parabola_functions_eq : ∀ (x₁ x₂ : ℝ), 
  (∃ t : ℝ, (f t b c = g t c b) ∧ (t = 1)) → 
    (f x₁ 2 (-3) = x₁^2 + 2 * x₁ - 3) ∧ (g x₂ (-3) 2 = x₂^2 - 3 * x₂ + 2) :=
sorry

end parabola_functions_eq_l627_627252


namespace perpendicular_vectors_dot_product_zero_l627_627930

theorem perpendicular_vectors_dot_product_zero
  (m : ℝ)
  (a : ℝ × ℝ × ℝ := (2, -1, 2))
  (b : ℝ × ℝ × ℝ := (-4, 2, m))
  (h_perp : a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) :
  m = 5 :=
sorry

end perpendicular_vectors_dot_product_zero_l627_627930


namespace range_of_k_if_f_monotonically_increasing_l627_627180

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem range_of_k_if_f_monotonically_increasing :
  (∀ (x : ℝ), 1 < x → 0 ≤ (k - 1 / x)) → k ∈ Set.Ici (1: ℝ) :=
by
  intro hyp
  have : ∀ (x : ℝ), 1 < x → 0 ≤ k - 1 / x := hyp
  sorry

end range_of_k_if_f_monotonically_increasing_l627_627180


namespace probability_divisible_by_7_l627_627653

theorem probability_divisible_by_7 (n : ℕ) (h1 : 10000 ≤ n ∧ n < 100000)
  (h2 : (nat.digits 10 n).sum = 40)
  (h3 : (nat.digits 10 n).head ≠ 0) :
  (∃ k, n = k * 7) ↔ false :=
sorry

end probability_divisible_by_7_l627_627653


namespace minimum_value_inverse_sum_l627_627500

variables {m n : ℝ}

theorem minimum_value_inverse_sum 
  (hm : m > 0) 
  (hn : n > 0) 
  (hline : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1)
  (hchord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ m * x2 + n * y2 + 2 = 0 → 
    (x1 - x2)^2 + (y1 - y2)^2 = 4) : 
  ∃ m n : ℝ, 3 * m + n = 2 ∧ m > 0 ∧ n > 0 ∧ 
    (∀ m' n' : ℝ, 3 * m' + n' = 2 → m' > 0 → n' > 0 → 
      (1 / m' + 3 / n' ≥ 6)) :=
sorry

end minimum_value_inverse_sum_l627_627500


namespace music_track_duration_l627_627795

theorem music_track_duration (minutes : ℝ) (seconds_per_minute : ℝ) (duration_in_minutes : minutes = 12.5) (seconds_per_minute_is_60 : seconds_per_minute = 60) : minutes * seconds_per_minute = 750 := by
  sorry

end music_track_duration_l627_627795


namespace non_congruent_triangles_with_perimeter_18_l627_627995

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627995


namespace loss_is_selling_price_of_16_pencils_l627_627636

theorem loss_is_selling_price_of_16_pencils
  (S : ℝ) -- Assume the selling price of one pencil is S
  (C : ℝ) -- Assume the cost price of one pencil is C
  (h₁ : 80 * C = 1.2 * 80 * S) -- The cost of 80 pencils is 1.2 times the selling price of 80 pencils
  : (80 * C - 80 * S) = 16 * S := -- The loss for selling 80 pencils equals the selling price of 16 pencils
  sorry

end loss_is_selling_price_of_16_pencils_l627_627636


namespace convert_m3_to_dm3_convert_cm2_to_m2_convert_L_to_mL_l627_627410

-- Definition of conversions as conditions
def m3_to_dm3 (x : ℝ) : ℝ := x * 1000
def cm2_to_m2 (y : ℝ) : ℝ := y / 10000
def L_to_mL (z : ℝ) : ℝ := z * 1000

-- Proof statements
theorem convert_m3_to_dm3 : m3_to_dm3 4.75 = 4750 :=
by sorry

theorem convert_cm2_to_m2 : cm2_to_m2 6500 = 0.65 :=
by sorry

theorem convert_L_to_mL : L_to_mL 3.05 = 3050 :=
by sorry

end convert_m3_to_dm3_convert_cm2_to_m2_convert_L_to_mL_l627_627410


namespace impossible_to_place_1995_points_l627_627559

theorem impossible_to_place_1995_points :
  ¬∃ (points : Fin 1995 → ℝ × ℝ),
    ∀ (i j : Fin 1995) (hij : i ≠ j), ∃ (k : Fin 1995), k ≠ i ∧ k ≠ j ∧
    ∃ (λ : ℝ), points k = (1 - λ) • points i + λ • points j := sorry

end impossible_to_place_1995_points_l627_627559


namespace probability_divisor_of_8_on_8_sided_die_l627_627784

def divisor_probability : ℚ :=
  let sample_space := {1, 2, 3, 4, 5, 6, 7, 8}
  let divisors_of_8 := {1, 2, 4, 8}
  let favorable_outcomes := divisors_of_8 ∩ sample_space
  favorable_outcomes.card / sample_space.card

theorem probability_divisor_of_8_on_8_sided_die :
  divisor_probability = 1 / 2 :=
sorry

end probability_divisor_of_8_on_8_sided_die_l627_627784


namespace radius_of_arch_bridge_l627_627055

theorem radius_of_arch_bridge (span height : ℝ) (span_eq : span = 12) (height_eq : height = 4) : 
  let radius := 6.5 in 
  True :=
by
  sorry

end radius_of_arch_bridge_l627_627055


namespace hypotenuse_length_l627_627296

theorem hypotenuse_length (x : ℝ) (h1 : 3 * x - 1 > 0) (h2 : 0 < x) (h_area : 1 / 2 * x * (3 * x - 1) = 90) :
  let y := 3 * x - 1 in
  let hypotenuse := Real.sqrt (x^2 + y^2) in
  hypotenuse = Real.sqrt 593 := 
by 
  sorry

end hypotenuse_length_l627_627296


namespace count_non_congruent_triangles_with_perimeter_18_l627_627937

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627937


namespace dice_sum_not_22_l627_627159

theorem dice_sum_not_22 (a b c d e : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 6) (h₁ : 1 ≤ b ∧ b ≤ 6)
  (h₂ : 1 ≤ c ∧ c ≤ 6) (h₃ : 1 ≤ d ∧ d ≤ 6) (h₄ : 1 ≤ e ∧ e ≤ 6) 
  (h₅ : a * b * c * d * e = 432) : a + b + c + d + e ≠ 22 :=
sorry

end dice_sum_not_22_l627_627159


namespace log_base_range_l627_627165

theorem log_base_range (a : ℝ) (h1 : log a (2/3) < 1) (h2 : 0 < a) (h3 : a ≠ 1) : 
  a ∈ (Set.Ioo 0 (2/3)) ∪ (Set.Ioi 1) :=
sorry

end log_base_range_l627_627165


namespace shaded_area_of_triangle_l627_627048

theorem shaded_area_of_triangle (legs : ℝ) (n_smaller_triangles : ℕ) (n_shaded_triangles : ℕ) : 
  legs = 10 → 
  n_smaller_triangles = 25 → 
  n_shaded_triangles = 15 → 
  (legs * legs / 2 / n_smaller_triangles * n_shaded_triangles) = 30 := 
by {
  intro h1 h2 h3,
  sorry
}

example : shaded_area_of_triangle 10 25 15 := by sorry

end shaded_area_of_triangle_l627_627048


namespace non_congruent_triangles_with_perimeter_18_l627_627996

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l627_627996


namespace find_value_of_X_l627_627578

theorem find_value_of_X : 
  let M := 3009 / 3 in
  let N := M / 4 in
  let X := M + 2 * N in
  X = 1504.5 :=
by
  let M := 3009 / 3
  let N := M / 4
  let X := M + 2 * N
  have hM : M = 1003 := by ring_nf
  have hN : N = 250.75 := by norm_num1
  have hX : X = 1003 + 2 * 250.75 := by ring
  have hX_val : X = 1504.5 := by norm_num1
  exact hX_val

end find_value_of_X_l627_627578


namespace flower_options_l627_627616

theorem flower_options (x y : ℕ) : 2 * x + 3 * y = 20 → ∃ x1 y1 x2 y2 x3 y3, 
  (2 * x1 + 3 * y1 = 20) ∧ (2 * x2 + 3 * y2 = 20) ∧ (2 * x3 + 3 * y3 = 20) ∧ 
  (((x1, y1) ≠ (x2, y2)) ∧ ((x2, y2) ≠ (x3, y3)) ∧ ((x1, y1) ≠ (x3, y3))) ∧ 
  ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :=
sorry

end flower_options_l627_627616


namespace problem_statement_l627_627515

theorem problem_statement
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023 * (c * d) - (a + b) / 20 = 2024 := 
by 
  sorry

end problem_statement_l627_627515


namespace find_continuously_differentiable_functions_l627_627846

-- Let f be a continuously differentiable function from ℝ to ℝ
variable (f : ℝ → ℝ)
variable (f_diff : ∀ x, differentiable ℝ f)

-- Condition: For every rational number q, f(q) is rational
def preserves_rationals := ∀ (q : ℚ), ∃ r : ℚ, f q = r

-- Condition: f(q) has the same denominator as q
def same_denominator (q r : ℚ) : Prop := q.denom = r.denom

-- Theorem declaration stating the equivalence to the correct answer
theorem find_continuously_differentiable_functions 
  (h1 : ∀ (q : ℚ), ∃ r : ℚ, f q = r ∧ same_denominator q r)
  : ∃ (k n : ℤ), ∀ x, f x = k * x + n :=
by
  sorry

end find_continuously_differentiable_functions_l627_627846


namespace orthocenter_is_incenter_of_pedal_triangle_l627_627764

variables {α : Type*} [EuclideanGeometry α]

def is_acute (T : Triangle α) : Prop := -- Definition stating the triangle is acute.
  ∀ (a b c : Point α), T.has_vertex a → T.has_vertex b → T.has_vertex c →
  ∠ a b c < pi / 2

def orthocenter (T : Triangle α) (H : Point α) : Prop := -- Definition stating H is the orthocenter of T
  ∀ (a b c : Point α), T.has_vertex a → T.has_vertex b → T.has_vertex c →
  Altitude T a H c ∧ Altitude T b H a ∧ Altitude T c H b

def feet_of_altitudes (T : Triangle α) (A' B' C' : Point α) : Prop := -- Defining the feet of the altitudes.
  ∀ (a b c : Point α), T.has_vertex a → T.has_vertex b → T.has_vertex c →
  Foot_of_altitude T a A' ∧ Foot_of_altitude T b B' ∧ Foot_of_altitude T c C'

def incenter (T : Triangle α) (I : Point α) : Prop := -- Definition of the incenter.
  ∀ (a b c : Point α), T.has_vertex a → T.has_vertex b → T.has_vertex c →
  Angle_bisector T a I b ∧ Angle_bisector T b I c ∧ Angle_bisector T c I a

theorem orthocenter_is_incenter_of_pedal_triangle (T : Triangle α) (H A' B' C' : Point α)
  (h_acute : is_acute T) (h_orthocenter : orthocenter T H) (h_feet : feet_of_altitudes T A' B' C') :
  incenter (Triangle.mk A' B' C') H :=
by
  sorry

end orthocenter_is_incenter_of_pedal_triangle_l627_627764


namespace count_non_congruent_triangles_with_perimeter_18_l627_627940

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627940


namespace common_chord_line_perpendicular_bisector_l627_627139

open Real

noncomputable def Circle1 := { x : ℝ × ℝ | x.1^2 + x.2^2 - 2 * x.1 = 0 }
noncomputable def Circle2 := { x : ℝ × ℝ | x.1^2 + x.2^2 + 2 * x.1 - 4 * x.2 = 0 }

def ChordLine := { x : ℝ × ℝ | x.1 - x.2 = 0 }
def PerpendicularBisector := { x : ℝ × ℝ | x.1 + x.2 - 1 = 0 }

theorem common_chord_line (A B : ℝ × ℝ) (hA : A ∈ Circle1 ∧ A ∈ Circle2) (hB : B ∈ Circle1 ∧ B ∈ Circle2) :
  ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → P ∈ ChordLine := 
sorry

theorem perpendicular_bisector (A B : ℝ × ℝ) (hA : A ∈ Circle1 ∧ A ∈ Circle2) (hB : B ∈ Circle1 ∧ B ∈ Circle2) :
  ∀ (M : ℝ × ℝ), M ∈ ((λ x, (x.1 + x.2 - 1) = 0) : real_univ → Prop) := 
sorry

end common_chord_line_perpendicular_bisector_l627_627139


namespace range_of_k_if_f_monotonically_increasing_l627_627181

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem range_of_k_if_f_monotonically_increasing :
  (∀ (x : ℝ), 1 < x → 0 ≤ (k - 1 / x)) → k ∈ Set.Ici (1: ℝ) :=
by
  intro hyp
  have : ∀ (x : ℝ), 1 < x → 0 ≤ k - 1 / x := hyp
  sorry

end range_of_k_if_f_monotonically_increasing_l627_627181


namespace ram_actual_distance_l627_627752

-- Define the given data
def map_distance_1 : ℝ := 312 -- inches
def actual_distance_1 : ℝ := 136 -- km
def ram_map_distance : ℝ := 34 -- inches

-- Define the problem in Lean 4
theorem ram_actual_distance : 
  let scale := actual_distance_1 / map_distance_1 in
  let ram_actual_dist := scale * ram_map_distance in
  abs (ram_actual_dist - 14.82) < 0.01 :=
by {
  let scale := actual_distance_1 / map_distance_1,
  let ram_actual_dist := scale * ram_map_distance,
  have : abs (ram_actual_dist - 14.82) < 0.01, {
    -- Calculation part skipped
    sorry
  },
  exact this
}

end ram_actual_distance_l627_627752


namespace rectangular_field_area_in_square_yards_l627_627136

theorem rectangular_field_area_in_square_yards :
  (3 : ℤ) = (1 : ℤ) →  
  let length_ft := (12 : ℤ)
  let width_ft := (9 : ℤ)
  let length_yd := length_ft / 3
  let width_yd := width_ft / 3
  let area_sq_yd := length_yd * width_yd
  area_sq_yd = (12 : ℤ) :=
by 
  intro h
  unfold length_ft width_ft length_yd width_yd area_sq_yd
  -- Converting units from feet to yards
  have l_yd : length_yd = 4, by sorry
  have w_yd : width_yd = 3, by sorry
  -- Calculating the area in square yards
  have area : area_sq_yd = l_yd * w_yd, by rw [l_yd, w_yd]; exact rfl
  exact area 

end rectangular_field_area_in_square_yards_l627_627136


namespace part1_parallel_vectors_part2_perpendicular_vectors_l627_627157

def vect_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vect_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem part1_parallel_vectors 
  (m : ℝ)
  (OA OB OC : ℝ × ℝ)
  (HOA : OA = (3, -4))
  (HOB : OB = (6, -3))
  (HOC : OC = (5 - m, -3 - m))
  (HAB_parallel_BC : vect_sub OB OA = (3, 1) ∧ vect_sub OC OB = (-1 - m, -m)) :
  m = 1 / 2 :=
sorry

theorem part2_perpendicular_vectors 
  (m : ℝ)
  (OA OB OC : ℝ × ℝ)
  (HOA : OA = (3, -4))
  (HOB : OB = (6, -3))
  (HOC : OC = (5 - m, -3 - m))
  (HAB_perp_AC : vect_dot (vect_sub OB OA) (vect_sub OC OA) = 0) :
  m = 7 / 4 :=
sorry

end part1_parallel_vectors_part2_perpendicular_vectors_l627_627157


namespace total_sum_of_digits_of_valid_five_digit_numbers_l627_627599

-- Define the condition of a five-digit number that Lidia likes; none of the digits are divisible by 3.
def valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 8

def valid_five_digit_num (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  valid_digit (n / 10000) ∧
  valid_digit ((n / 1000) % 10) ∧
  valid_digit ((n / 100) % 10) ∧
  valid_digit ((n / 10) % 10) ∧
  valid_digit (n % 10)

theorem total_sum_of_digits_of_valid_five_digit_numbers : 
  (∑ n in Finset.filter valid_five_digit_num (Finset.range 100000), 
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)) = 174960 :=
by
  sorry

end total_sum_of_digits_of_valid_five_digit_numbers_l627_627599


namespace value_of_f2010_l627_627414

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x) = f(-x)
def special_property (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(2 + x) = -f(2 - x)

theorem value_of_f2010 (f : ℝ → ℝ) (h_even : even_function f) (h_special : special_property f) : 
  f 2010 = 0 :=
sorry

end value_of_f2010_l627_627414


namespace cookies_available_last_night_l627_627507

-- Define the initial quantities
def initial_cookies_three_days_ago := 31
def initial_cookies_two_days_ago := 270
def initial_cookies_yesterday := 419

-- Define the quantities lost each day
def cookies_eaten_by_beaky := 5
def crumble_percentage := 15
def cookies_given_away := 30

-- Define the gift quantity
def gift_from_lucy := 45

-- Calculate cookies remaining after each day considering losses
def remaining_cookies_three_days_ago := initial_cookies_three_days_ago - cookies_eaten_by_beaky
def remaining_cookies_two_days_ago := initial_cookies_two_days_ago - (initial_cookies_two_days_ago * crumble_percentage / 100).floor
def remaining_cookies_yesterday := initial_cookies_yesterday - cookies_given_away

-- Calculate the total cookies available as of last night
def total_cookies_available := 
  remaining_cookies_three_days_ago + remaining_cookies_two_days_ago + remaining_cookies_yesterday + gift_from_lucy

-- Statement to prove
theorem cookies_available_last_night : total_cookies_available = 690 := 
by 
  -- Skipping proof steps
  sorry

end cookies_available_last_night_l627_627507


namespace monotonically_increasing_intervals_max_min_values_interval_l627_627496

def f (x : ℝ) : ℝ := (sin x + cos x) ^ 2 + 2 * (cos x) ^ 2 - 2

theorem monotonically_increasing_intervals (k : ℤ) :
  ∀ x, (k * π - 3 * π / 8 ≤ x ∧ x ≤ k * π + π / 8) → monotonic_increasing (f x) :=
by sorry

theorem max_min_values_interval : 
  (∀ x, (π / 4 ≤ x ∧ x ≤ 3 * π / 4) → -√2 ≤ f(x) ∧ f(x) ≤ 1) :=
by sorry

end monotonically_increasing_intervals_max_min_values_interval_l627_627496


namespace income_of_wealthiest_individuals_l627_627672

theorem income_of_wealthiest_individuals :
  ∀ (x : ℝ), 
    (2 * 10^9 * x^(-2) = 500) → 
    x = 10^4 :=
by sorry

end income_of_wealthiest_individuals_l627_627672


namespace part_one_part_two_l627_627915

def f (x : ℝ) : ℝ := x - (1 / x)

theorem part_one (x : ℝ) (h : x ≠ 0) : f (-x) = - f x :=
by 
  unfold f
  field_simp
  ring

theorem part_two (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f x1 < f x2 :=
by
  unfold f
  have h1 : (x1 - x2) * (x1 * x2 + 1) < 0, from
    mul_neg_of_neg_of_pos (sub_neg_of_lt h) (add_pos_of_pos_of_nonneg (mul_pos hx1 hx2) zero_le_one)
  simpa using div_neg_of_neg_of_pos h1 (mul_pos hx1 hx2)

end part_one_part_two_l627_627915


namespace unique_triangles_count_l627_627951

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627951


namespace find_length_of_PC_l627_627364

theorem find_length_of_PC (P A B C D : ℝ × ℝ) (h1 : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 25)
                            (h2 : (P.1 - D.1)^2 + (P.2 - D.2)^2 = 36)
                            (h3 : (P.1 - B.1)^2 + (P.2 - B.2)^2 = 49)
                            (square_ABCD : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2) :
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = 38 :=
by
  sorry

end find_length_of_PC_l627_627364


namespace Donna_and_Marcia_total_pencils_l627_627424

def DonnaPencils (CindiPencils MarciaPencils DonnaPencils : ℕ) : Prop :=
  DonnaPencils = 3 * MarciaPencils

def MarciaPencils (CindiPencils MarciaPencils : ℕ) : Prop :=
  MarciaPencils = 2 * CindiPencils

def CindiPencils (CindiSpent CindiPencilCost CindiPencils : ℕ) : Prop :=
  CindiPencils = CindiSpent / CindiPencilCost

theorem Donna_and_Marcia_total_pencils (CindiSpent CindiPencilCost : ℕ) (DonnaPencils MarciaPencils CindiPencils : ℕ)
  (hCindi : CindiPencils CindiSpent CindiPencilCost CindiPencils)
  (hMarcia : MarciaPencils CindiPencils MarciaPencils)
  (hDonna : DonnaPencils CindiPencils MarciaPencils DonnaPencils) :
  DonnaPencils + MarciaPencils = 480 := 
sorry

end Donna_and_Marcia_total_pencils_l627_627424


namespace find_AN_l627_627313

variable (Point : Type) [MetricSpace Point]

structure Config :=
  (A B C D M N : Point)
  (circle : Circle Point)
  (trapezoid : Trapezoid Point)
  (A_on_circle : A ∈ circle)
  (B_on_circle : B ∈ circle)
  (C_on_circle : C ∈ circle)
  (D_on_circle : D ∈ circle)
  (AD_eq_6 : dist A D = 6)
  (tangent_at_A : TangentToCircle Point A circle)
  (M_on_BD : M ∈ LineThrough Point B D)
  (N_on_CD : N ∈ LineThrough Point C D)
  (AB_perp_MD : IsPerpendicular Point (LineThrough Point A B) (LineThrough Point M D))
  (AM_eq_3 : dist A M = 3)

theorem find_AN (c : Config Point) : dist c.A c.N = 12 := by
  sorry

end find_AN_l627_627313


namespace first_quartile_example_l627_627537

noncomputable def median (l : List ℚ) : ℚ :=
  let sorted := l.qsort (· ≤ ·)
  if h : (sorted.length % 2 = 1) then
    sorted[nat.pred ((sorted.length + 1) / 2)]
  else
    (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2

noncomputable def first_quartile (l : List ℚ) : ℚ :=
  let m := median l
  let less_than_m := l.filter (· < m)
  median less_than_m

theorem first_quartile_example :
  first_quartile [42, 24, 30, 22, 26, 27, 33, 35] = 25 := by
  sorry

end first_quartile_example_l627_627537


namespace rachel_picture_books_shelves_l627_627260

theorem rachel_picture_books_shelves (mystery_shelves : ℕ) (books_per_shelf : ℕ) (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : books_per_shelf = 9) 
  (h3 : total_books = 72) : 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 :=
by sorry

end rachel_picture_books_shelves_l627_627260


namespace correct_statement_l627_627045

-- Let’s define what it means for angles to be complementary.
def complementary (θ ψ : ℝ) : Prop := θ + ψ = 90

-- Define the incorrect statement (A)
def statement_A : Prop := ∀ θ ψ, complementary θ ψ → (θ < 90 ∧ ψ > 90) ∨ (θ > 90 ∧ ψ < 90)

-- Define the incorrect statement (B)
def statement_B : Prop := ∀ (A B : ℝ × ℝ), ∃ shortest_distance, ∀ (P Q : ℝ × ℝ), 
  (P = A ∧ Q = B) → shortest_distance (P, Q) = line_segment_length (P, Q)

-- Define the correct statement (C)
def statement_C : Prop := ∀ θ ψ,  complementary θ ψ → θ = ψ

-- Define the incorrect statement (D)
def statement_D : Prop := ∀ (A : ℝ × ℝ) (line : ℝ × ℝ → ℝ), 
    ∃ shortest_distance, shortest_distance A line = perpendicular_distance A line

-- The main theorem: Statement C is correct.
theorem correct_statement : statement_C :=
by {
  -- Proof of correctness
  sorry
}

end correct_statement_l627_627045


namespace intern_teacher_arrangement_l627_627031

theorem intern_teacher_arrangement :
  ∃ n : ℕ, n = 48 ∧
  (let teachers := {A, B, C, D, E, F} in
  let classes := {ClassA, ClassB, ClassC} in
  let arrangements := {arr : teachers → classes // 
      arr 'A ≠ ClassA ∧ arr 'B ≠ arr 'C} in
  finset.card arrangements = n) :=
begin
  use 48,
  sorry,
end

end intern_teacher_arrangement_l627_627031


namespace find_inverse_sum_l627_627652

variable {R : Type*} [OrderedRing R]

-- Define the function f and its inverse
variable (f : R → R)
variable (f_inv : R → R)

-- Conditions
axiom f_inverse : ∀ y, f (f_inv y) = y
axiom f_prop : ∀ x, f x + f (1 - x) = 2

-- The theorem we need to prove
theorem find_inverse_sum (x : R) : f_inv (x - 2) + f_inv (4 - x) = 1 :=
by
  sorry

end find_inverse_sum_l627_627652


namespace find_a8_l627_627126

-- Define an arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a d : ℕ → ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a d : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n / 2) * (a + arithmetic_sequence a d n)

-- Conditions
axiom cond1 {a d : ℕ → ℚ} : sum_arithmetic_sequence a d 15 = 15

-- Question
theorem find_a8 (a d : ℕ → ℚ) : 
  (arithmetic_sequence a d 8) = 1 :=
  sorry

end find_a8_l627_627126


namespace non_congruent_triangles_with_perimeter_18_l627_627991

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627991


namespace remainder_when_b_divided_by_11_l627_627228

theorem remainder_when_b_divided_by_11 (n : ℕ) (hn : n > 0) 
  (hb : ∃ b : ℕ, b ≡ (5 ^ (3 * n) + 3)⁻¹ [MOD 11]) :
  ∃ b : ℕ, b % 11 = 8 :=
by
  sorry

end remainder_when_b_divided_by_11_l627_627228


namespace value_of_a_l627_627926

theorem value_of_a {a : ℝ} (A : Set ℝ) (B : Set ℝ) (hA : A = {-1, 0, 2}) (hB : B = {2^a}) (hSub : B ⊆ A) : a = 1 := 
sorry

end value_of_a_l627_627926


namespace sqrt_inequality_l627_627900

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) :=
sorry

end sqrt_inequality_l627_627900


namespace gcd_of_180_and_450_l627_627083

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l627_627083


namespace magnitude_AB_eq_one_l627_627458

def vectorOA : ℝ × ℝ := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))
def vectorOB : ℝ × ℝ := (Real.cos (75 * Real.pi / 180), Real.sin (75 * Real.pi / 180))
def vectorAB : ℝ × ℝ := (vectorOB.1 - vectorOA.1, vectorOB.2 - vectorOA.2)

theorem magnitude_AB_eq_one : (vectorAB.1^2 + vectorAB.2^2).sqrt = 1 := by
  sorry

end magnitude_AB_eq_one_l627_627458


namespace min_max_solution_A_l627_627701

theorem min_max_solution_A (x y z : ℕ) (h₁ : x + y + z = 100) (h₂ : 5 * x + 8 * y + 9 * z = 700) 
                           (h₃ : 0 ≤ x ∧ x ≤ 60) (h₄ : 0 ≤ y ∧ y ≤ 60) (h₅ : 0 ≤ z ∧ z ≤ 47) :
    35 ≤ x ∧ x ≤ 49 :=
by
  sorry

end min_max_solution_A_l627_627701


namespace is_isosceles_right_triangle_l627_627040

theorem is_isosceles_right_triangle (α β γ : ℝ) (h_sum : α + β + γ = 180) (h_ratio : α / 45 = β / 90 ∧ β / 90 = γ / 45) : 
  α = 45 ∧ β = 90 ∧ γ = 45 ∧ Isosceles ∧ RightAngle β :=
by 
  sorry

end is_isosceles_right_triangle_l627_627040


namespace part_a_part_b_l627_627416

-- Problem definition
def S := {p : ℕ | ∃ r : ℕ, r > 0 ∧ (p ≥ 2) ∧ (∀ k : ℕ, k ≥ 1 → (∀ n, (10^n % p) != 1) → (10^(3*r) % p = 10^(3*r+k) % p))}

def a (k r : ℕ) : ℕ := (10^(k-1) % (10^(3*r) - 1))

def f(k p : ℕ) : ℕ := a(k, r p) + a(k + r p, r p) + a(k + 2*r p, r p)

-- Part (a): Proving that S is infinite
theorem part_a : set.infinite S := 
sorry

-- Part (b): Finding the highest value of f(k, p) for k ≥ 1 and p ∈ S
theorem part_b : ∀ k ≥ 1, ∀ p ∈ S, f(k, p) ≤ 19 ∧ (∃ k p, f(k, p) = 19) := 
sorry

end part_a_part_b_l627_627416


namespace five_digit_number_with_integer_cube_root_l627_627170

theorem five_digit_number_with_integer_cube_root (n : ℕ) 
  (h1 : n ≥ 10000 ∧ n < 100000) 
  (h2 : n % 10 = 3) 
  (h3 : ∃ k : ℕ, k^3 = n) : 
  n = 19683 ∨ n = 50653 :=
sorry

end five_digit_number_with_integer_cube_root_l627_627170


namespace range_of_m_value_of_x_l627_627494

noncomputable def a : ℝ := 3 / 2

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log a

-- Statement for the range of m
theorem range_of_m :
  ∀ m : ℝ, f (3 * m - 2) < f (2 * m + 5) ↔ (2 / 3) < m ∧ m < 7 :=
by
  intro m
  sorry

-- Value of x
theorem value_of_x :
  ∃ x : ℝ, f (x - 2 / x) = Real.log (7 / 2) / Real.log (3 / 2) ∧ x > 0 ∧ x = 4 :=
by
  use 4
  sorry

end range_of_m_value_of_x_l627_627494


namespace unique_triangles_count_l627_627955

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l627_627955


namespace smallest_positive_period_l627_627492

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (x - π / 6) * Real.sin (x + π / 3)

theorem smallest_positive_period (x : ℝ) :
  Function.periodic f π := 
sorry

end smallest_positive_period_l627_627492


namespace pythagorean_triple_check_l627_627400

theorem pythagorean_triple_check :
  (∃ (a₁ b₁ c₁ : ℝ), a₁ = 0.3 ∧ b₁ = 0.4 ∧ c₁ = 0.5 ∧ a₁^2 + b₁^2 = c₁^2) ∨
  (∃ (a₂ b₂ c₂ : ℝ), a₂ = 1 ∧ b₂ = 1 ∧ c₂ = real.sqrt 2 ∧ a₂^2 + b₂^2 = c₂^2) ∨
  (∃ (a₃ b₃ c₃ : ℝ), a₃ = 5 ∧ b₃ = 12 ∧ c₃ = 13 ∧ a₃^2 + b₃^2 = c₃^2) ∨
  (∃ (a₄ b₄ c₄ : ℝ), a₄ = 1 ∧ b₄ = real.sqrt 3 ∧ c₄ = 2 ∧ a₄^2 + b₄^2 = c₄^2) ↔
  (∃ (a b c : ℕ), a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2) :=
sorry

end pythagorean_triple_check_l627_627400


namespace highest_temperature_day_l627_627058

theorem highest_temperature_day (days : List String) (temperatures : List ℕ) (h : days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]) (t : temperatures = [m, t, w, th, f, sa, su]) : 
  (max temperatures) = f → 
  days[temperatures.indexOf (max temperatures)] = "Friday" :=
by
  sorry

end highest_temperature_day_l627_627058


namespace exists_n_iterations_in_interval_l627_627008

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x < sqrt 2 / 2 then x + (2 - sqrt 2) / 2
else x - sqrt 2 / 2

theorem exists_n_iterations_in_interval (a b : ℝ) (h_ab : 0 < a ∧ a < b ∧ b < 1)
    : ∃ x ∈ set.Ioo a b, ∃ n : ℕ, f^[n] x ∈ set.Ioo a b := sorry

end exists_n_iterations_in_interval_l627_627008


namespace mark_money_l627_627604

theorem mark_money (M : ℝ) (h1 : M / 2 + 14 ≤ M) (h2 : M / 3 + 16 ≤ M) :
  M - (M / 2 + 14) - (M / 3 + 16) = 0 → M = 180 := by
  sorry

end mark_money_l627_627604


namespace triangle_area_l627_627171

theorem triangle_area (CE : ℝ) (hCE : CE = 2 * Real.sqrt 2)
  (h1 : ∀ A B C, angle A C E = 45) -- AEC is a 45-45-90 triangle
  (h2 : ∀ A B C, angle B A C = 45) -- ABC is a 45-45-90 triangle
  : ∃ area : ℝ, area = 8 :=
by
  sorry

end triangle_area_l627_627171


namespace sum_digits_of_six_digit_palindromes_sum_l627_627829

theorem sum_digits_of_six_digit_palindromes_sum : 
  (∑ a in finset.range 9, 
    ∑ b in finset.range 10, 
    ∑ c in finset.range 10, 
    ∑ d in finset.range 10, 
    nat.digits 10 ((1000001 * (a + 1) + 100010 * b + 10010 * c + 1000 * d)).sum = 36) :=
sorry

end sum_digits_of_six_digit_palindromes_sum_l627_627829


namespace parallelepiped_ratio_l627_627253

-- Definitions of vectors and centroids
variables {u v w : ℝ^3}
def P := 0
def Q := v
def R := w
def S := u
def G := (v + w + u) / 3

-- Distances squared
noncomputable def dist_squared (a b : ℝ^3) := ∥a - b∥ ^ 2
noncomputable def PQ_squared := dist_squared P Q
noncomputable def PR_squared := dist_squared P R
noncomputable def PS_squared := dist_squared P S
noncomputable def QR_squared := dist_squared Q R
noncomputable def RS_squared := dist_squared R S
noncomputable def SP_squared := dist_squared S P
noncomputable def QG_squared := dist_squared Q G
noncomputable def RG_squared := dist_squared R G
noncomputable def SG_squared := dist_squared S G

-- Proof statement
theorem parallelepiped_ratio :
  (PQ_squared + PR_squared + PS_squared + QR_squared + RS_squared + SP_squared) /
  (QG_squared + RG_squared + SG_squared) =
  4.5 := sorry

end parallelepiped_ratio_l627_627253


namespace symmetric_point_origin_l627_627208

theorem symmetric_point_origin :
  ∀ (P : ℝ × ℝ × ℝ), P = (3, 1, 5) → 
  let P_sym := (-P.1, -P.2, -P.3) in 
  P_sym = (-3, -1, -5) :=
by
  sorry

end symmetric_point_origin_l627_627208


namespace comparison_of_square_roots_l627_627573

theorem comparison_of_square_roots (P Q : ℝ) (hP : P = Real.sqrt 2) (hQ : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q :=
by
  sorry

end comparison_of_square_roots_l627_627573


namespace n_pow_m_eq_4_l627_627173

theorem n_pow_m_eq_4 (m n : ℤ) (h1 : ∀ x : ℤ, x^2 - 3 * x + m = (x - 1) * (x + n)) : n^m = 4 :=
by
  sorry

end n_pow_m_eq_4_l627_627173


namespace total_members_is_15_l627_627021

variable (M : ℕ)   -- M is the total number of members in the club

-- Conditions
def lemonJuice := 2 * M / 5
def remainingMembers := M - lemonJuice
def mangoJuice := remainingMembers / 3
def orangeJuice := remainingMembers - mangoJuice

-- Given: 6 members ordered orange juice
axiom orangeJuiceCondition : orangeJuice = 6

-- Goal: Prove the total number of members M is 15
theorem total_members_is_15 (h : orangeJuiceCondition) : M = 15 := by
  sorry

end total_members_is_15_l627_627021


namespace acute_triangle_AQ_MP_FR_concurrent_l627_627203

noncomputable def point : Type := sorry

variables (A B C D E F M P Q R : point)
variables (triangle_ABC : ∀ (A B C : point), Prop)
variables (is_acute_angled: triangle_ABC A B C)
variables (altitude_foot_D : ∀ (A B C D : point), Prop)
variables (altitude_foot_E : ∀ (B A C E : point), Prop)
variables (altitude_foot_F : ∀ (C A B F : point), Prop)
variables (orthocenter_M : ∀ (A B C M : point), Prop)
variables (circle_k1 : ∀ (A B : point), Prop)
variables (circumcircle_k2 : ∀ (D E M : point), Prop)
variables (on_arc_P : ∀ (E M D P : point), Prop)
variables (intersects_k1_at_Q : ∀ (D P Q : point), Prop)
variables (is_midpoint_R : ∀ (P Q R : point), Prop)
variables (line_AQ : ∀ (A Q : point), Prop)
variables (line_MP : ∀ (M P : point), Prop)
variables (line_FR : ∀ (F R : point), Prop)
variables (concurrent : ∀ (line_AQ line_MP line_FR : Prop), Prop)

theorem acute_triangle_AQ_MP_FR_concurrent
  (h_triangle : triangle_ABC A B C)
  (h_acute : is_acute_angled)
  (h_alt_D : altitude_foot_D A B C D)
  (h_alt_E : altitude_foot_E B A C E)
  (h_alt_F : altitude_foot_F C A B F)
  (h_orth_M : orthocenter_M A B C M)
  (h_circle_k1 : circle_k1 A B)
  (h_circ_k2 : circumcircle_k2 D E M)
  (h_on_arc_P : on_arc_P E M D P)
  (h_intersect_Q : intersects_k1_at_Q D P Q)
  (h_midpoint_R : is_midpoint_R P Q R) :
  concurrent (line_AQ A Q) (line_MP M P) (line_FR F R)
:= sorry

end acute_triangle_AQ_MP_FR_concurrent_l627_627203


namespace find_AD_l627_627211

-- Given conditions as definitions
def AB := 5 -- given length in meters
def angle_ABC := 85 -- given angle in degrees
def angle_BCA := 45 -- given angle in degrees
def angle_DBC := 20 -- given angle in degrees

-- Lean theorem statement to prove the result
theorem find_AD : AD = AB := by
  -- The proof will be filled in afterwards; currently, we leave it as sorry.
  sorry

end find_AD_l627_627211


namespace f_1000000_is_25_l627_627234

noncomputable def f (n : ℕ) : ℕ :=
  Inf { N : ℕ | n ∣ Nat.factorial N }

theorem f_1000000_is_25 : f 1000000 = 25 :=
by
  sorry

end f_1000000_is_25_l627_627234


namespace train_crosses_pole_in_15_seconds_l627_627004

noncomputable def train_crossing_time (length : ℕ) (speed_km_hr : ℕ) : ℕ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  in length / speed_m_s

theorem train_crosses_pole_in_15_seconds :
  train_crossing_time 600 144 = 15 :=
by
  sorry

end train_crosses_pole_in_15_seconds_l627_627004


namespace sum_of_remainders_eq_11_mod_13_l627_627733

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l627_627733


namespace range_of_a_l627_627457

noncomputable def satisfies_condition (a : ℝ) : Prop :=
∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs ((1 / 2) * x^3 - a * x) ≤ 1

theorem range_of_a :
  {a : ℝ | satisfies_condition a} = {a : ℝ | - (1 / 2) ≤ a ∧ a ≤ (3 / 2)} :=
by
  sorry

end range_of_a_l627_627457


namespace eccentricity_of_ellipse_l627_627908

def a_squared : ℝ := 4
def b_squared : ℝ := 3
def a : ℝ := Real.sqrt a_squared
def b : ℝ := Real.sqrt b_squared
def c : ℝ := Real.sqrt (a_squared - b_squared)
def e : ℝ := c / a

theorem eccentricity_of_ellipse : e = 1 / 2 := by
  -- skipped proof
  sorry

end eccentricity_of_ellipse_l627_627908


namespace radius_of_sphere_eq_cuberoot_six_l627_627022

theorem radius_of_sphere_eq_cuberoot_six (r h : ℝ) (R : ℝ) (π : ℝ) :
  r = 2 ∧ h = 6 ∧ (1 / 3) * π * r^2 * h = (4 / 3) * π * R^3 → R = real.cbrt 6 :=
by
  sorry

end radius_of_sphere_eq_cuberoot_six_l627_627022


namespace baking_powder_difference_l627_627744

-- Define the known quantities
def baking_powder_yesterday : ℝ := 0.4
def baking_powder_now : ℝ := 0.3

-- Define the statement to prove, i.e., the difference in baking powder
theorem baking_powder_difference : baking_powder_yesterday - baking_powder_now = 0.1 :=
by
  -- Proof omitted
  sorry

end baking_powder_difference_l627_627744


namespace find_varphi_and_sin2alpha_l627_627456

-- Definitions based on conditions
variable (ϕ α : ℝ)
variable (h0 : 0 < ϕ)
variable (h1 : ϕ < π)
variable (h2 : sin (ϕ + π / 4) = sin (ϕ - π / 4))
variable (h3 : π / 4 < α)
variable (h4 : α < π / 2)
variable (h5 : sin (2 * α + π / 4) = -5 / 13)

-- Prove the values of ϕ and sin 2α
theorem find_varphi_and_sin2alpha : 
  ϕ = π / 2 ∧ sin (2 * α) = (7 * sqrt 2) / 26 :=
by
  -- Proof steps skipped
  sorry

end find_varphi_and_sin2alpha_l627_627456


namespace incorrect_statement_l627_627742

theorem incorrect_statement :
  ¬(∀ (d c : ℝ) (h : d = 2 * c), (∃ (O : set.point), O.perpendicular c ∧ O.bisects c ∧ O.diameter = d)) :=
by
  sorry

end incorrect_statement_l627_627742


namespace smallest_integral_value_k_l627_627340

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x * (k * x - 5) - x^2 + 4

-- Define the condition for the quadratic equation having no real roots
def no_real_roots (k : ℝ) : Prop :=
  let a := 3 * k - 1
  let b := -15
  let c := 4
  discriminant a b c < 0

-- The Lean 4 statement to find the smallest integral value of k such that the quadratic has no real roots
theorem smallest_integral_value_k : ∃ (k : ℤ), no_real_roots k ∧ (∀ (m : ℤ), no_real_roots m → k ≤ m) :=
  sorry

end smallest_integral_value_k_l627_627340


namespace equation_of_line_l_volume_of_solid_l627_627902

-- Line passing through point (3,1) and parallel to x + y - 1 = 0 
def line_passing_through_and_parallel (x y : ℝ) (c : ℝ) : Prop :=
  x + y + c = 0

-- Given conditions
def point_on_line (x y : ℝ) : Prop :=
  x = 3 ∧ y = 1

def parallel_line (a b c : ℝ) : Prop :=
  a = 1 ∧ b = 1 ∧ c = -1

def find_line_equation (c : ℝ) : Prop :=
  c = -4

-- Equation of the line l is x + y - 4 = 0
theorem equation_of_line_l : ∃ c : ℝ, line_passing_through_and_parallel 3 1 c ∧ parallel_line 1 1 c ∧ find_line_equation c :=
begin
  use -4,
  split,
  { simp [line_passing_through_and_parallel, point_on_line],
    norm_num },
  { split,
    { simp [parallel_line],
      norm_num },
    { simp [find_line_equation],
      norm_num } }
end

-- Volume of the solid
def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

def cone_with_radius_and_height : Prop :=
  volume_cone 4 4 = (64 / 3) * real.pi

-- The plane figure forms a cone with radius 4 and height 4, and its volume is 64/3 * pi
theorem volume_of_solid : cone_with_radius_and_height :=
begin
  simp [volume_cone],
  norm_num,
end

end equation_of_line_l_volume_of_solid_l627_627902


namespace proof_expectation_red_balls_drawn_l627_627191

noncomputable def expectation_red_balls_drawn : Prop :=
  let total_ways := Nat.choose 5 2
  let ways_2_red := Nat.choose 3 2
  let ways_1_red_1_yellow := Nat.choose 3 1 * Nat.choose 2 1
  let p_X_eq_2 := (ways_2_red : ℝ) / total_ways
  let p_X_eq_1 := (ways_1_red_1_yellow : ℝ) / total_ways
  let expectation := 2 * p_X_eq_2 + 1 * p_X_eq_1
  expectation = 1.2

theorem proof_expectation_red_balls_drawn :
  expectation_red_balls_drawn :=
by
  sorry

end proof_expectation_red_balls_drawn_l627_627191


namespace find_angle_ZYP_l627_627407

-- Definitions of angles and properties to set up the problem
variables (W X Y Z P : Type)
variables (angle : W → W → ℝ)
variables [QuasilinearOrderedField ℝ] -- Assuming a setting where geometry and trigonometry make sense

-- Conditions given in the problem
def cyclic_quadrilateral (W X Y Z : Type) : Prop := sorry
def extended_beyond (WY WP : Type) : Prop := sorry
def angle_WZX := 110
def angle_WXY := 74

-- Statement of the theorem (proof obligation)
theorem find_angle_ZYP 
  (h_cyclic : cyclic_quadrilateral W X Y Z)
  (h_extended : extended_beyond (angle W Y) (angle W P))
  (h_WZX : angle W Z = 110)
  (h_WXY : angle W X = 74) :
  angle Z Y = 74 :=
by
  sorry

end find_angle_ZYP_l627_627407


namespace real_part_fraction_l627_627589

theorem real_part_fraction (x y : ℝ) (h : x^2 + y^2 = 4) :
    let z := x + y*complex.I in
    (complex.re (1 / (1 - z))) = (1 - x) / (5 - 2 * x) :=
by
  let z := x + y*complex.I
  have hz : |z| = complex.norm (x + y*complex.I),
    { rw complex.norm_eq_abs },
  sorry

end real_part_fraction_l627_627589


namespace fraction_of_people_under_21_correct_l627_627019

variable (P : ℕ) (frac_over_65 : ℚ) (num_under_21 : ℕ) (frac_under_21 : ℚ)

def total_people_in_range (P : ℕ) : Prop := 50 < P ∧ P < 100

def fraction_of_people_over_65 (frac_over_65 : ℚ) : Prop := frac_over_65 = 5/12

def number_of_people_under_21 (num_under_21 : ℕ) : Prop := num_under_21 = 36

def fraction_of_people_under_21 (frac_under_21 : ℚ) : Prop := frac_under_21 = 3/7

theorem fraction_of_people_under_21_correct :
  ∀ (P : ℕ),
  total_people_in_range P →
  fraction_of_people_over_65 (5 / 12) →
  number_of_people_under_21 36 →
  P = 84 →
  fraction_of_people_under_21 (36 / P) :=
by
  intros P h_range h_over_65 h_under_21 h_P
  sorry

end fraction_of_people_under_21_correct_l627_627019


namespace probability_more_than_60000_l627_627627

def boxes : List ℕ := [8, 800, 8000, 40000, 80000]

def probability_keys (keys : ℕ) : ℚ :=
  1 / keys

def probability_winning (n : ℕ) : ℚ :=
  if n = 4 then probability_keys 5 + probability_keys 5 * probability_keys 4 else 0

theorem probability_more_than_60000 : 
  probability_winning 4 = 1/4 := sorry

end probability_more_than_60000_l627_627627


namespace magnitude_order_l627_627120

theorem magnitude_order (a : ℝ) (h : (π / 4) < a ∧ a < (π / 2)) :
  (cos a) ^ (sin a) < (cos a) ^ (cos a) ∧ (cos a) ^ (cos a) < (sin a) ^ (cos a) := by
  sorry

end magnitude_order_l627_627120


namespace find_f_2009_l627_627479

-- Defining the function f and specifying the conditions
variable (f : ℝ → ℝ)
axiom h1 : f 3 = -Real.sqrt 3
axiom h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x

-- Proving the desired statement
theorem find_f_2009 : f 2009 = 2 + Real.sqrt 3 :=
sorry

end find_f_2009_l627_627479


namespace decipher_rebus_l627_627212

theorem decipher_rebus (a b c d : ℕ) :
  (a = 10 ∧ b = 14 ∧ c = 12 ∧ d = 13) ↔
  (∀ (x y z w: ℕ), 
    (x = 10 → 5 + 5 * 7 = 49) ∧
    (y = 14 → 2 - 4 * 3 = 9) ∧
    (z = 12 → 12 - 1 - 1 * 2 = 20) ∧
    (w = 13 → 13 - 1 + 10 - 5 = 17) ∧
    (49 + 9 + 20 + 17 = 95)) :=
by sorry

end decipher_rebus_l627_627212


namespace first_pipe_fills_tank_in_36_minutes_l627_627702

-- Definitions
def fills_in_time (first_pipe_time : ℕ) (first_rate := 1 / first_pipe_time) : Prop :=
  ∃ (both_time : ℕ), ∃ (second_time: ℕ),
    both_time = 180 ∧ second_time = 45 ∧
    (1 / first_pipe_time - 1 / second_time) = 1 / both_time

-- Statement
theorem first_pipe_fills_tank_in_36_minutes : fills_in_time 36 :=
by {
  exists 180; exists 45,
  split; swap,
  { norm_num,
    sorry, -- the actual calculation steps
  },
  split; norm_num,
}

end first_pipe_fills_tank_in_36_minutes_l627_627702


namespace true_proposition_l627_627130

-- Definitions of propositions p and q
def prop_p (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 → ∀ (x : ℝ), (x = -1) → (a ^ (x + 1) + 1 = 2)

def prop_q : Prop := 
  ∀ (m α β : Type), α // (α parallel β)  -> 
  ¬((m // parallel alpha) ↔ (m // parallel beta)) 

-- Proposition: p ∧ ¬q is true
theorem true_proposition (a : ℝ) : prop_p a ∧ ¬prop_q :=
by
  sorry

end true_proposition_l627_627130


namespace probability_divisor_of_8_is_half_l627_627780

theorem probability_divisor_of_8_is_half :
  let outcomes := (1 : ℕ) :: (2 : ℕ) :: (3 : ℕ) :: (4 : ℕ) :: (5 : ℕ) :: (6 : ℕ) :: (7 : ℕ) :: (8 : ℕ) :: []
  let divisors_of_8 := [ 1, 2, 4, 8 ]
  let favorable_outcomes := list.filter (λ x, x ∣ 8) outcomes
  let favorable_probability := (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)
  favorable_probability = (1 / 2 : ℚ) := by
  sorry

end probability_divisor_of_8_is_half_l627_627780


namespace problem1_problem2_l627_627011

-- Problem 1: Prove the calculation result
theorem problem1 : abs (-1) + (real.sqrt 2 / 2) ^ (-2:ℤ) - (3 + real.sqrt 5) ^ 0 - real.sqrt 8 = 2 - 2 * real.sqrt 2 :=
by
  -- The proof steps will go here
  sorry

-- Problem 2: Prove the solution of the system of equations
theorem problem2 (x y : ℝ) (h1 : 2 * x - y = 4) (h2 : x + y = 2) : x = 2 ∧ y = 0 :=
by
  -- The proof steps will go here
  sorry

end problem1_problem2_l627_627011


namespace cookies_left_l627_627607

-- Define the conditions
def pounds_of_flour_used_per_batch : ℕ := 2
def batches_per_bakery_bag_of_flour : ℕ := 5
def total_bags_used : ℕ := 4
def cookies_per_batch : ℕ := 12
def cookies_eaten_by_jim : ℕ := 15

-- Calculate the total pounds of flour used
def total_pounds_of_flour := total_bags_used * batches_per_bakery_bag_of_flour

-- Calculate the total number of batches
def total_batches := total_pounds_of_flour / pounds_of_flour_used_per_batch

-- Calculate the total number of cookies cooked
def total_cookies := total_batches * cookies_per_batch

-- Calculate the number of cookies left
theorem cookies_left :
  let total_cookies := total_batches * cookies_per_batch in 
  total_cookies - cookies_eaten_by_jim = 105 :=
by
  sorry

end cookies_left_l627_627607


namespace count_non_congruent_triangles_with_perimeter_18_l627_627939

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l627_627939


namespace non_congruent_triangles_count_l627_627967

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627967


namespace real_number_x_equal_2_l627_627513

theorem real_number_x_equal_2 (x : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2 * i) * (x + i) = 4 - 3 * i → x = 2 :=
by
  sorry

end real_number_x_equal_2_l627_627513


namespace all_black_after_finite_adjustments_l627_627361

def transition_rule (pieces : List Int) : List Int :=
  pieces.zipWith (λ x y => if x = y then x else 1) (pieces.tail ++ [pieces.head])

def adjustment (pieces : List Int) : List Int :=
  transition_rule pieces

theorem all_black_after_finite_adjustments 
  (n : ℕ) 
  (pieces : List Int) 
  (h : pieces.length = 2^n) 
  (initial_pieces : ∀ i, pieces.nth i ∈ [1, -1]) :
  ∃ (k : ℕ), (∀ (m: ℕ), m ≥ k → List.foldr (λ x acc => x * acc) 1 (adjustment^[m] pieces) = 1) :=
sorry

end all_black_after_finite_adjustments_l627_627361


namespace gcd_180_450_l627_627092

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l627_627092


namespace robin_packages_l627_627264

theorem robin_packages (p t n : ℕ) (h1 : p = 18) (h2 : t = 486) : t / p = n ↔ n = 27 :=
by
  rw [h1, h2]
  norm_num
  sorry

end robin_packages_l627_627264


namespace positive_sqrt_729_l627_627166

theorem positive_sqrt_729 (x : ℝ) (h_pos : 0 < x) (h_eq : x^2 = 729) : x = 27 :=
by
  sorry

end positive_sqrt_729_l627_627166


namespace inequality_solution_l627_627847

theorem inequality_solution :
  {x : Real | (2 * x - 5) * (x - 3) / x ≥ 0} = {x : Real | (x ∈ Set.Ioc 0 (5 / 2)) ∨ (x ∈ Set.Ici 3)} := 
sorry

end inequality_solution_l627_627847


namespace initial_mean_corrected_l627_627678

theorem initial_mean_corrected (M : ℝ) (H : 30 * M + 30 = 30 * 151) : M = 150 :=
sorry

end initial_mean_corrected_l627_627678


namespace minimum_product_abc_l627_627594

noncomputable def is_positive_real (x : ℝ) := x > 0

theorem minimum_product_abc 
  (a b c : ℝ) 
  (h_pos_a : is_positive_real a)
  (h_pos_b : is_positive_real b)
  (h_pos_c : is_positive_real c)
  (h_sum : a + b + c = 2)
  (h_cond : c ≤ 3 * a ∧ b ≤ 3 * a ∧ a ≤ 3 * b ∧ c ≤ 3 * b ∧ b ≤ 3 * c ∧ a ≤ 3 * c) :
  ∃ (abc_min : ℝ), abc_min = a * b * c ∧ abc_min = 2/3 := 
begin
  use a * b * c,
  split,
  { refl },
  { sorry }
end

end minimum_product_abc_l627_627594


namespace find_perpendicular_line_l627_627078

-- Define the point P
structure Point where
  x : ℤ
  y : ℤ

-- Define the line
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- The given problem conditions
def P : Point := { x := -1, y := 3 }

def given_line : Line := { a := 1, b := -2, c := 3 }

def perpendicular_line (line : Line) (point : Point) : Line :=
  ⟨ -line.b, line.a, -(line.a * point.y - line.b * point.x) ⟩

-- Theorem statement to prove
theorem find_perpendicular_line :
  perpendicular_line given_line P = { a := 2, b := 1, c := -1 } :=
by
  sorry

end find_perpendicular_line_l627_627078


namespace dot_product_ae_l627_627593

section
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c e : V)

-- Conditions:
-- a, b, c, e are distinct unit vectors
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1) (he : ∥e∥ = 1)
variables (habc : a ≠ b) (habc' : a ≠ c) (habc'' : a ≠ e) (hbc : b ≠ c) (hbe : b ≠ e) (hce : c ≠ e)

-- Inner product conditions
variables (ha_b : ⟪a, b⟫ = -1/8) (ha_c : ⟪a, c⟫ = -1/8) (hb_c : ⟪b, c⟫ = -1/8)
variables (hb_e : ⟪b, e⟫ = -1/8) (hc_e : ⟪c, e⟫ = -1/8)

-- Question: Find ⟪a, e⟫
theorem dot_product_ae : ⟪a, e⟫ = -35/34 := sorry

end

end dot_product_ae_l627_627593


namespace number_of_non_congruent_triangles_perimeter_18_l627_627949

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627949


namespace time_for_b_and_d_together_l627_627746

theorem time_for_b_and_d_together :
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  (∃ B_rate C_rate : ℚ,
    B_rate + C_rate = 1 / 3 ∧
    A_rate + C_rate = 1 / 2 ∧
    1 / (B_rate + D_rate) = 2.4) :=
  
by
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  use 1 / 6, 1 / 6
  sorry

end time_for_b_and_d_together_l627_627746


namespace probability_of_rolling_divisor_of_8_l627_627776

open_locale classical

-- Predicate: a number n is a divisor of 8
def is_divisor_of_8 (n : ℕ) : Prop := n ∣ 8

-- The total number of outcomes when rolling an 8-sided die
def total_outcomes : ℕ := 8

-- The probability of rolling a divisor of 8 on a fair 8-sided die
theorem probability_of_rolling_divisor_of_8 (is_fair_die : true) :
  (| {n | is_divisor_of_8 n} ∩ {1, 2, 3, 4, 5, 6, 7, 8} | : ℕ) / total_outcomes = 1 / 2 :=
by
  sorry

end probability_of_rolling_divisor_of_8_l627_627776


namespace round_robin_tournament_participant_can_mention_all_l627_627838

theorem round_robin_tournament_participant_can_mention_all :
  ∀ (n : ℕ) (participants : Fin n → Fin n → Prop),
  (∀ i j : Fin n, i ≠ j → (participants i j ∨ participants j i)) →
  (∃ A : Fin n, ∀ (B : Fin n), B ≠ A → (participants A B ∨ ∃ C : Fin n, participants A C ∧ participants C B)) := by
  sorry

end round_robin_tournament_participant_can_mention_all_l627_627838


namespace rooms_already_painted_l627_627797

-- Define the conditions as variables and hypotheses
variables (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
variables (h1 : total_rooms = 10)
variables (h2 : hours_per_room = 8)
variables (h3 : remaining_hours = 16)

-- Define the theorem stating the number of rooms already painted
theorem rooms_already_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 10) (h2 : hours_per_room = 8) (h3 : remaining_hours = 16) :
  (total_rooms - (remaining_hours / hours_per_room) = 8) :=
sorry

end rooms_already_painted_l627_627797


namespace meals_neither_kosher_nor_vegan_l627_627613

theorem meals_neither_kosher_nor_vegan : (total_clients vegan kosher both : ℕ)
    (total_clients = 30) (vegan = 7) (kosher = 8) (both = 3) :
    total_clients - (vegan + kosher - both) = 18 :=
by
  sorry

end meals_neither_kosher_nor_vegan_l627_627613


namespace find_circle_radius_l627_627771

noncomputable def circle_radius_is_sqrt_2
  (O A B C D : ℝ×ℝ)
  (R : ℝ)
  (dist_AB : ℝ)
  (dist_CD : ℝ)
  (tangent : Prop)
  (intersects : Prop)
  (angle_bisector : Prop) : Prop :=
  (dist_AB = real.sqrt 6) →
  (dist_CD = real.sqrt 7) →
  tangent →
  intersects →
  angle_bisector →
  (R = real.sqrt 2)

theorem find_circle_radius (O A B C D : ℝ×ℝ)
  (R : ℝ)
  (dist_AB : ℝ)
  (dist_CD : ℝ)
  (tangent : Prop)
  (intersects : Prop)
  (angle_bisector : Prop) :
  circle_radius_is_sqrt_2 O A B C D R dist_AB dist_CD tangent intersects angle_bisector :=
by
  sorry

end find_circle_radius_l627_627771


namespace value_of_a5_l627_627467

theorem value_of_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = 2 * n * (n + 1)) (ha : ∀ n, a n = S n - S (n - 1)) :
  a 5 = 20 :=
by
  sorry

end value_of_a5_l627_627467


namespace sum_of_transformed_numbers_l627_627690

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  a'' + b'' = 3 * S + 24 := 
by
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  sorry

end sum_of_transformed_numbers_l627_627690


namespace solve_for_m_l627_627898

variables (e1 e2 : Vector) (m : ℝ) (A B C : Point)

-- Define collinearity condition
def collinear (A B C : Point) : Prop := 
  ∃ (λ : ℝ), vec B A = λ ∙ vec C B

axiom non_collinear_vectors : e1 ≠ 0 ∧ e2 ≠ 0 ∧ ¬ collinear_vectors e1 e2

axiom AB_eq : vec B A = 2 • e1 + m • e2
axiom BC_eq : vec C B = e1 + 3 • e2
axiom collinear_points : collinear A B C

theorem solve_for_m : m = 6 :=
by
  sorry

end solve_for_m_l627_627898


namespace geometric_sequence_m_solution_l627_627514

theorem geometric_sequence_m_solution (m : ℝ) (h : ∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 4 ∧ a * c = b^2) :
  m = 2 ∨ m = -2 :=
by
  sorry

end geometric_sequence_m_solution_l627_627514


namespace flower_shop_options_l627_627618

theorem flower_shop_options :
  {n : ℕ // n = {xy : ℕ × ℕ // 2 * xy.1 + 3 * xy.2 = 20}.card} = 3 :=
by
  sorry

end flower_shop_options_l627_627618


namespace smallest_n_for_three_nested_rectangles_l627_627099

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  x : ℕ
  y : ℕ
  h1 : 1 ≤ x
  h2 : x ≤ y
  h3 : y ≤ 100

/-- Define the nesting relation between rectangles -/
def nested (R1 R2 : Rectangle) : Prop :=
  R1.x < R2.x ∧ R1.y < R2.y

/-- Prove the smallest n such that there exist 3 nested rectangles out of n rectangles where n = 101 -/
theorem smallest_n_for_three_nested_rectangles (n : ℕ) (h : n ≥ 101) :
  ∀ (rectangles : Fin n → Rectangle), 
    ∃ (R1 R2 R3 : Fin n), nested (rectangles R1) (rectangles R2) ∧ nested (rectangles R2) (rectangles R3) :=
  sorry

end smallest_n_for_three_nested_rectangles_l627_627099


namespace pencils_bought_l627_627428

theorem pencils_bought (cindi_spent : ℕ) (cost_per_pencil : ℕ) 
  (cindi_pencils : ℕ) 
  (marcia_pencils : ℕ) 
  (donna_pencils : ℕ) :
  cindi_spent = 30 → 
  cost_per_pencil = 1/2 → 
  cindi_pencils = cindi_spent / cost_per_pencil → 
  marcia_pencils = 2 * cindi_pencils → 
  donna_pencils = 3 * marcia_pencils → 
  donna_pencils + marcia_pencils = 480 := 
by
  sorry

end pencils_bought_l627_627428


namespace square_side_length_l627_627390

theorem square_side_length (s : ℝ) (h : s^2 = 1/9) : s = 1/3 :=
sorry

end square_side_length_l627_627390


namespace length_MN_l627_627891

-- Defining the constants and equation of the ellipse
def a : ℝ := 4
def b : ℝ := 2 * Real.sqrt 2
def c : ℝ := b

def eq_ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 8) = 1

-- Condition for the midpoint (1,1) of segment MN
def midpoint_MN (M N : ℝ × ℝ) : Prop :=
  (fst M + fst N)/2 = 1 ∧ (snd M + snd N)/2 = 1

-- Statement asserting the length of |MN| given the conditions
theorem length_MN (M N : ℝ × ℝ) (hM : eq M.fst M.snd) (hN : eq N.fst N.snd) (h_midpoint : midpoint_MN M N) :
  |M - N| = sqrt(390) / 3 :=
begin
  sorry
end

end length_MN_l627_627891


namespace num_frisbees_more_than_deck_cards_l627_627059

variables (M F D x : ℕ)
variable (bought_fraction : ℝ)

theorem num_frisbees_more_than_deck_cards :
  M = 60 ∧ M = 2 * F ∧ F = D + x ∧
  M + bought_fraction * M + F + bought_fraction * F + D + bought_fraction * D = 140 ∧ bought_fraction = 2/5 →
  x = 20 :=
by
  sorry

end num_frisbees_more_than_deck_cards_l627_627059


namespace non_congruent_triangles_with_perimeter_18_l627_627972

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627972


namespace fibonacci_mod_127_l627_627657

def fibonacci : ℕ → ℕ
| 0        := 0
| 1        := 1
| (n + 2)  := fibonacci (n + 1) + fibonacci n

theorem fibonacci_mod_127 (m : ℕ) :
  (∃ m, m > 0 ∧ fibonacci m % 127 = 0 ∧ fibonacci (m + 1) % 127 = 1) → m = 256 := 
sorry

end fibonacci_mod_127_l627_627657


namespace deposit_ratio_l627_627601

-- Definitions based on conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining_amount : ℕ) : ℕ := remaining_amount / 2
def cash_left_after_deposit : ℕ := 24

-- Theorem to prove the ratio of the money deposited to the money left after buying the t-shirt
theorem deposit_ratio :
  let remaining_after_ice_cream := initial_amount - spent_on_ice_cream in
  let remaining_after_tshirt := remaining_after_ice_cream - spent_on_tshirt remaining_after_ice_cream in
  let deposited := remaining_after_tshirt - cash_left_after_deposit in
  deposited * 5 = remaining_after_tshirt :=
by
  sorry

end deposit_ratio_l627_627601


namespace gcd_5670_9800_l627_627717

-- Define the two given numbers
def a := 5670
def b := 9800

-- State that the GCD of a and b is 70
theorem gcd_5670_9800 : Int.gcd a b = 70 := by
  sorry

end gcd_5670_9800_l627_627717


namespace real_part_w3_l627_627664

namespace ComplexProof

-- Define complex number 'w' with positive imaginary part and magnitude 5
def w : ℂ := sorry

-- Conditions about 'w'
axiom w_positive_imaginary : w.im > 0
axiom w_magnitude : complex.abs w = 5

-- Define the triangle vertices
def w1 := w
def w2 := w * w
def w3 := w * w * w

-- Right angle condition at 'w' in the triangle
axiom right_angle : complex.dot (w2 - w1) (w3 - w1) = 0

-- The goal is to prove that real part of 'w^3' is -73
theorem real_part_w3 : w.re = -73 := sorry

end ComplexProof

end real_part_w3_l627_627664


namespace midpoint_sum_l627_627237

theorem midpoint_sum (x y : ℝ) (h1 : (x + 0) / 2 = 2) (h2 : (y + 9) / 2 = 4) : x + y = 3 := by
  sorry

end midpoint_sum_l627_627237


namespace non_congruent_triangles_with_perimeter_18_l627_627994

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627994


namespace sum_sqrt_series_is_71_l627_627582

noncomputable def sum_sqrt_series : ℚ :=
  ∑ n in finset.range(4900 + 1), (1 / real.sqrt (n + real.sqrt (n^2 - 1)))

theorem sum_sqrt_series_is_71 :
  ∃ (a b c : ℕ), 
    a + b * real.sqrt c = sum_sqrt_series ∧
    c ∉ { p^2 | p : ℕ } ∧ -- c is not divisible by the square of any prime
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 71 :=
by
  sorry

end sum_sqrt_series_is_71_l627_627582


namespace fifth_closest_is_park_l627_627662

-- Defining the buildings and their order in terms of distance
def bank : ℕ := 1
def school : ℕ := 2
def stationery_store : ℕ := 3
def convenience_store : ℕ := 4
def park : ℕ := 5

-- The list of buildings in their order of distance
def buildings : list ℕ := [bank, school, stationery_store, convenience_store, park]

-- Theorem to prove the 5th closest building is the park
theorem fifth_closest_is_park : buildings.nth 4 = some park :=
by 
  -- This is where the proof would go
  sorry

end fifth_closest_is_park_l627_627662


namespace amount_rick_same_outcome_probability_l627_627811

variable {p q : ℕ}

/- Given conditions -/
def fair_six_sided_die_distribution := (1/6 : ℝ)
def fair_four_sided_die_distribution := (1/4 : ℝ)
def coin_head_probability := (5/8 : ℝ)

/- Definition of the game outcomes -/
def GameOutcome : Type := (ℕ × ℕ × ℕ) -- (sum of 6-sided die, sum of 4-sided die, coin number of heads)

/- Statement of the theorem to prove -/
theorem amount_rick_same_outcome_probability (p q : ℕ) :
  ( fair_six_sided_die_distribution * fair_four_sided_die_distribution * coin_head_probability ) ^ 2 = p / q :=
sorry

end amount_rick_same_outcome_probability_l627_627811


namespace min_value_expression_l627_627536

theorem min_value_expression (a b: ℝ) (h : 2 * a + b = 1) : (a - 1) ^ 2 + (b - 1) ^ 2 = 4 / 5 :=
sorry

end min_value_expression_l627_627536


namespace maximum_point_of_f_l627_627917

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x - 2) * Real.exp x

theorem maximum_point_of_f : ∃ x : ℝ, x = -2 ∧
  ∀ y : ℝ, f y ≤ f x :=
sorry

end maximum_point_of_f_l627_627917


namespace flower_options_l627_627617

theorem flower_options (x y : ℕ) : 2 * x + 3 * y = 20 → ∃ x1 y1 x2 y2 x3 y3, 
  (2 * x1 + 3 * y1 = 20) ∧ (2 * x2 + 3 * y2 = 20) ∧ (2 * x3 + 3 * y3 = 20) ∧ 
  (((x1, y1) ≠ (x2, y2)) ∧ ((x2, y2) ≠ (x3, y3)) ∧ ((x1, y1) ≠ (x3, y3))) ∧ 
  ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :=
sorry

end flower_options_l627_627617


namespace remaining_fruits_l627_627368

theorem remaining_fruits (initial_apples initial_oranges initial_mangoes taken_apples twice_taken_apples taken_mangoes) : 
  initial_apples = 7 → 
  initial_oranges = 8 → 
  initial_mangoes = 15 → 
  taken_apples = 2 → 
  twice_taken_apples = 2 * taken_apples → 
  taken_mangoes = 2 * initial_mangoes / 3 → 
  initial_apples - taken_apples + initial_oranges - twice_taken_apples + initial_mangoes - taken_mangoes = 14 :=
by
  sorry

end remaining_fruits_l627_627368


namespace find_f_2019_l627_627143

def f : ℕ → ℕ
| x := if x ≤ 2015 then x + 2 else f (x - 5)

theorem find_f_2019 : f 2019 = 2016 := 
by
  sorry

end find_f_2019_l627_627143


namespace Donna_and_Marcia_total_pencils_l627_627425

def DonnaPencils (CindiPencils MarciaPencils DonnaPencils : ℕ) : Prop :=
  DonnaPencils = 3 * MarciaPencils

def MarciaPencils (CindiPencils MarciaPencils : ℕ) : Prop :=
  MarciaPencils = 2 * CindiPencils

def CindiPencils (CindiSpent CindiPencilCost CindiPencils : ℕ) : Prop :=
  CindiPencils = CindiSpent / CindiPencilCost

theorem Donna_and_Marcia_total_pencils (CindiSpent CindiPencilCost : ℕ) (DonnaPencils MarciaPencils CindiPencils : ℕ)
  (hCindi : CindiPencils CindiSpent CindiPencilCost CindiPencils)
  (hMarcia : MarciaPencils CindiPencils MarciaPencils)
  (hDonna : DonnaPencils CindiPencils MarciaPencils DonnaPencils) :
  DonnaPencils + MarciaPencils = 480 := 
sorry

end Donna_and_Marcia_total_pencils_l627_627425


namespace probability_divisor_of_8_on_8_sided_die_l627_627785

def divisor_probability : ℚ :=
  let sample_space := {1, 2, 3, 4, 5, 6, 7, 8}
  let divisors_of_8 := {1, 2, 4, 8}
  let favorable_outcomes := divisors_of_8 ∩ sample_space
  favorable_outcomes.card / sample_space.card

theorem probability_divisor_of_8_on_8_sided_die :
  divisor_probability = 1 / 2 :=
sorry

end probability_divisor_of_8_on_8_sided_die_l627_627785


namespace sum_of_numbers_l627_627753

theorem sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) : a + b + c = 22 :=
by
  sorry

end sum_of_numbers_l627_627753


namespace gcd_180_450_l627_627086

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l627_627086


namespace range_of_a_l627_627555

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 0)

-- Define the curve C
def C (a : ℝ) : ℝ × ℝ → Prop :=
  λ P, (P.1 ^ 2 + P.2 ^ 2 - 2 * a * P.1 - 4 * a * P.2 + 5 * a ^ 2 - 9 = 0)

-- The predicate that verifies if |PB| = 2 * |PA|
def condition (P : ℝ × ℝ) (a : ℝ) : Prop :=
  real.sqrt ((P.1 - B.1) ^ 2 + P.2 ^ 2) = 2 * real.sqrt ((P.1 - A.1) ^ 2 + P.2 ^ 2)

-- The proof statement
theorem range_of_a (a : ℝ) :
  (∃ P, C a P ∧ condition P a)
  ↔ (a ≥ -real.sqrt 5 ∧ a ≤ -real.sqrt 5 / 5) ∨ (a ≥ real.sqrt 5 / 5 ∧ a ≤ real.sqrt 5) :=
sorry

end range_of_a_l627_627555


namespace gcd_180_450_l627_627089

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l627_627089


namespace hyperbola_equation_l627_627497

theorem hyperbola_equation (a b : ℝ) (h1 : 2 * sqrt 5 = 2 * a) 
  (h2 : sqrt 5 = (b * sqrt (5) / sqrt (a^2 + b^2))) : 
  (a = sqrt 5) ∧ (b = sqrt 5) → 
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 5 - y^2 / 5 = 1)) :=
by 
  intro h
  cases h with ha hb
  rw [ha, hb]
  sorry

end hyperbola_equation_l627_627497


namespace area_of_new_circle_l627_627639

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

noncomputable def radius (A B : ℝ × ℝ) : ℝ :=
  distance A B / 2

noncomputable def area_of_circle (A B : ℝ × ℝ) : ℝ :=
  real.pi * (radius A B)^2

theorem area_of_new_circle : 
  let A := (-1, 3)
  let B_prime := (13, 12)
  area_of_circle A B_prime = 277 * real.pi / 4 :=
by
  let A := (-1, 3)
  let B_prime := (13, 12)
  calc
    area_of_circle A B_prime = 277 * real.pi / 4 := sorry

end area_of_new_circle_l627_627639


namespace intersection_AB_l627_627155

/-- Define the set A based on the given condition -/
def setA : Set ℝ := {x | 2 * x ^ 2 + x > 0}

/-- Define the set B based on the given condition -/
def setB : Set ℝ := {x | 2 * x + 1 > 0}

/-- Prove that A ∩ B = {x | x > 0} -/
theorem intersection_AB : (setA ∩ setB) = {x | x > 0} :=
sorry

end intersection_AB_l627_627155


namespace find_principal_amount_l627_627855

theorem find_principal_amount :
  ∃ P : ℝ, let r_1 := 0.05,
               r_2 := 0.06,
               r_3 := 0.07,
               t_1 := 1,
               t_2 := 1,
               t_3 := (2/5 : ℝ),
               final_amount := 1008 in
           P * (1 + r_1)^(t_1) * (1 + r_2)^(t_2) * (1 + r_3)^(t_3) = final_amount ∧ 
           |P - 905.08| < 0.01 :=
begin
  sorry
end

end find_principal_amount_l627_627855


namespace smallest_prime_dividing_sum_l627_627727

theorem smallest_prime_dividing_sum (h1 : 2 ^ 14 % 2 = 0) (h2 : 7 ^ 9 % 2 = 1) (h3 : (2 ^ 14 + 7 ^ 9) % 2 = 1) : ∃ p : ℕ, p.prime ∧ p ∣ (2 ^ 14 + 7 ^ 9) ∧ ∀ q : ℕ, q.prime ∧ q ∣ (2 ^ 14 + 7 ^ 9) → p ≤ q :=
by
  sorry

end smallest_prime_dividing_sum_l627_627727


namespace sum_of_superb_rectangles_areas_l627_627388

theorem sum_of_superb_rectangles_areas :
  ∀ (a b : ℕ), (a * b = 6 * (a + b)) → (∃ areas : Finset ℕ, 
  (∀ (a b : ℕ), (a * b = 6 * (a + b)) → areas ∈ a * b → a ≠ b → a > 0 → b > 0) ∧ 
  areas.sum = 942) := sorry

end sum_of_superb_rectangles_areas_l627_627388


namespace decreasing_function_range_l627_627489

variable {a : ℝ}
def f (x : ℝ) : ℝ := if x < 1 then (a - 3) * x + 3 * a else Real.log x / Real.log a

theorem decreasing_function_range (h : ∀ x y : ℝ, x < y → f x ≥ f y) : 3 / 4 ≤ a ∧ a < 1 :=
sorry

end decreasing_function_range_l627_627489


namespace sum_mistake_l627_627634

variables {a b : ℤ → ℤ} -- assumably functions that can be positive or negative transformations

def sum1 := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, a i
def sum2 := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, b i

theorem sum_mistake (h1 : sum1 = 21) (h2 : sum2 = 20) : 
  ∃ i, a i ≠ b i :=
sorry

end sum_mistake_l627_627634


namespace min_distance_midpoint_to_origin_eq_3sqrt2_l627_627929

theorem min_distance_midpoint_to_origin_eq_3sqrt2 (x1 y1 x2 y2 : ℝ) :
  (x1 + y1 = 7) → (x2 + y2 = 5) →
  let Mx := (x1 + x2) / 2;
      My := (y1 + y2) / 2 in
  (abs (Mx + My - 0) / Real.sqrt (1^2 + 1^2)) = 3 * Real.sqrt 2 :=
by
  intros h1 h2
  let Mx := (x1 + x2) / 2
  let My := (y1 + y2) / 2
  calc
    abs (Mx + My - 0) / Real.sqrt (1^2 + 1^2) = sorry

end min_distance_midpoint_to_origin_eq_3sqrt2_l627_627929


namespace x_plus_q_eq_2q_minus_3_l627_627528

theorem x_plus_q_eq_2q_minus_3 (x q : ℝ) (h1: |x + 3| = q) (h2: x > -3) :
  x + q = 2q - 3 :=
sorry

end x_plus_q_eq_2q_minus_3_l627_627528


namespace curve_C2_equation_and_length_AB_l627_627299

open Real 

noncomputable def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (2 * cos α, 2 + 2 * sin α)

def midpoint (O P : ℝ × ℝ) : ℝ × ℝ :=
  ((O.1 + P.1) / 2, (O.2 + P.2) / 2)

def trajectory_C2 (P : ℝ × ℝ) : Prop :=
  (P.1)^2 + (P.2 - 4)^2 = 16

def polar_line (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 4) = sqrt 2

def line_cartesian (x y : ℝ) : Prop :=
  x + y - 2 = 0

def distance_from_center_to_line (center : ℝ × ℝ) : ℝ :=
  abs (center.2 - 2) / sqrt 2

def segment_length_AB (r d : ℝ) : ℝ :=
  2 * sqrt (r^2 - d^2)

theorem curve_C2_equation_and_length_AB :
  (∀ α : ℝ, M = midpoint ⟨0, 0⟩ ⟨x, y⟩ ∧ parametric_curve_C1 α = M → trajectory_C2 (2 * M.1, 2 * M.2)) ∧
  (let center := (0, 4)
   let radius := 4
   let d := distance_from_center_to_line center
   ∀ A B : ℝ × ℝ, polar_line (sqrt (A.1^2 + A.2^2)) (arctan (A.2 / A.1)) =
     polar_line (sqrt (B.1^2 + B.2^2))  (arctan (B.2 / B.1)) →
     line_cartesian A.1 A.2 ∧ line_cartesian B.1 B.2 →
     A ≠ B → segment_length_AB radius d = 2 * sqrt 14) := sorry

end curve_C2_equation_and_length_AB_l627_627299


namespace mass_percentage_B_in_H3BO3_l627_627718

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_B : ℝ := 10.81
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_H3BO3 : ℝ := 3 * atomic_mass_H + atomic_mass_B + 3 * atomic_mass_O

theorem mass_percentage_B_in_H3BO3 : (atomic_mass_B / molar_mass_H3BO3) * 100 = 17.48 :=
by
  sorry

end mass_percentage_B_in_H3BO3_l627_627718


namespace chess_board_cut_l627_627770

theorem chess_board_cut :
  ∃ line : ℕ × ℕ → Prop, ∀ d : ℕ × ℕ × ℕ × ℕ, is_domino d → 
  (∃ line_segment : (ℕ × ℕ) × (ℕ × ℕ), on_segment line_segment d ∧ does_not_cut line_segment d) :=
by
  sorry

-- Definitions required (skipped actual implementations for brevity)
def is_domino : (ℕ × ℕ × ℕ × ℕ) → Prop :=
sorry

def on_segment : ((ℕ × ℕ) × (ℕ × ℕ)) → (ℕ × ℕ × ℕ × ℕ) → Prop :=
sorry

def does_not_cut : ((ℕ × ℕ) × (ℕ × ℕ)) → (ℕ × ℕ × ℕ × ℕ) → Prop :=
sorry

end chess_board_cut_l627_627770


namespace non_congruent_triangles_with_perimeter_18_l627_627979

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627979


namespace find_p_l627_627113

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_valid_configuration (p q s r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime s ∧ is_prime r ∧ 
  1 < p ∧ p < q ∧ q < s ∧ p + q + s = r

-- The theorem statement
theorem find_p (p q s r : ℕ) (h : is_valid_configuration p q s r) : p = 2 :=
by
  sorry

end find_p_l627_627113


namespace train_crossing_time_approx_l627_627353

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmph : ℝ := 70

noncomputable def convert_speed_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def crossing_time (length : ℝ) (speed_mps : ℝ) : ℝ :=
  length / speed_mps

theorem train_crossing_time_approx :
  (crossing_time train_length (convert_speed_to_mps train_speed_kmph)) ≈ 12.86 :=
by 
  sorry

end train_crossing_time_approx_l627_627353


namespace problem_proof_l627_627164

variable (P Q M N : ℝ)

axiom hp1 : M = 0.40 * Q
axiom hp2 : Q = 0.30 * P
axiom hp3 : N = 1.20 * P

theorem problem_proof : (M / N) = (1 / 10) := by
  sorry

end problem_proof_l627_627164


namespace isosceles_triangle_of_given_condition_l627_627530

variables {V : Type*} [inner_product_space ℝ V]

-- Axioms for defining points and vectors
variables {A B C M : V}

-- Definition representing M, B, and C lie in the same plane
-- This is implied by inner product space over reals in this context

-- Given condition
axiom eq_condition : (B - C) • (B + C - 2 • A) = 0

-- Proof statement
theorem isosceles_triangle_of_given_condition (M A B C : V) :
  (B - C) • (B + C - 2 • A) = 0 → dist A B = dist A C :=
by sorry

end isosceles_triangle_of_given_condition_l627_627530


namespace data_set_is_1133_l627_627889

variables (x1 x2 x3 x4 : ℕ)

-- Defining the conditions
def is_positive_set (x1 x2 x3 x4 : ℕ): Prop := x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0

def mean_is_two (x1 x2 x3 x4 : ℕ): Prop := (x1 + x2 + x3 + x4) / 4 = 2

def median_is_two (x1 x2 x3 x4 : ℕ): Prop := 
  ∃ (l1 l2 l3 l4 : ℕ), list.sort (≤) [x1, x2, x3, x4] = [l1, l2, l3, l4] ∧ (l2 + l3) / 2 = 2

def std_dev_is_one (x1 x2 x3 x4 : ℕ): Prop := 
  let μ := (x1 + x2 + x3 + x4) / 4 in
  let variance := ((x1 - μ)^2 + (x2 - μ)^2 + (x3 - μ)^2 + (x4 - μ)^2) / 4 in
  variance = 1

-- Assertion that the sorted set is {1, 1, 3, 3}
theorem data_set_is_1133 :
  is_positive_set x1 x2 x3 x4 →
  mean_is_two x1 x2 x3 x4 →
  median_is_two x1 x2 x3 x4 →
  std_dev_is_one x1 x2 x3 x4 →
  list.sort (≤) [x1, x2, x3, x4] = [1, 1, 3, 3] :=
sorry

end data_set_is_1133_l627_627889


namespace vector_subtraction_l627_627119

def a : ℝ^3 := ⟨3, -2, 5⟩
def b : ℝ^3 := ⟨-1, 4, 0⟩

theorem vector_subtraction :
  a - 4 • b = ⟨7, -18, 5⟩ := by
  sorry

end vector_subtraction_l627_627119


namespace length_of_train_l627_627754

noncomputable def speed_m_s := (120 * 1000) / 3600
def time_s : ℝ := 75
def total_distance := speed_m_s * time_s

theorem length_of_train : ∃ L : ℝ, 2 * L = total_distance ∧ L = 1249.875 :=
by
  let L := total_distance / 2
  use L
  split
  . exact calc
      2 * L = 2 * (total_distance / 2) : by rw mul_div_cancel' _ two_ne_zero
      ...   = total_distance          : by rw mul_comm
  . sorry

end length_of_train_l627_627754


namespace pentagon_area_eq_half_l627_627552

variables {A B C D E : Type*} -- Assume A, B, C, D, E are some points in a plane

-- Assume the given conditions in the problem
variables (angle_A angle_C : ℝ)
variables (AB AE BC CD AC : ℝ)
variables (pentagon_area : ℝ)

-- Assume the constraints from the problem statement
axiom angle_A_eq_90 : angle_A = 90
axiom angle_C_eq_90 : angle_C = 90
axiom AB_eq_AE : AB = AE
axiom BC_eq_CD : BC = CD
axiom AC_eq_1 : AC = 1

theorem pentagon_area_eq_half : pentagon_area = 1 / 2 :=
sorry

end pentagon_area_eq_half_l627_627552


namespace x4_plus_1_exact_division_l627_627288

noncomputable def exact_division_pairs : set (ℚ × ℚ) :=
  { ⟨0, Complex.i⟩, ⟨0, -Complex.i⟩, ⟨Real.sqrt 2, 1⟩, ⟨-Real.sqrt 2, 1⟩ }

theorem x4_plus_1_exact_division (p q : ℚ) :
  (∀ x : ℂ, x^4 + 1 = (x^2 + p * x + q) * (x^2 + -p * x + if p = 0 then -q else if q = 1 then 0 else q)) ↔ (p, q) ∈ exact_division_pairs := 
by sorry

end x4_plus_1_exact_division_l627_627288


namespace hyperbola_equation_hyperbola_slope_product_l627_627138

noncomputable def hyperbola_asymptote (C : Type) := 
∀ (x y : ℝ), x^2 - (y^2 / 3) = 1

theorem hyperbola_equation (h1 : P (2, 3)) (h2 : asymptotes (λ x : ℝ, sqrt 3 * x, -sqrt 3 * x)) : 
  hyperbola_asymptote C :=
sorry

theorem hyperbola_slope_product (line_eq : ∀ x m k : ℝ, kx + m) (intersects : ∀ A B : point, lies_on_hyperbola A B) :
  ∃ k : ℝ = -3/2, ∀ k₁ k₂ : slope, k₁ * k₂ = -3 :=
sorry

end hyperbola_equation_hyperbola_slope_product_l627_627138


namespace translation_proof_l627_627039

def complex_translation (a b c d : ℂ) : Prop :=
  ∃ w : ℂ, b = a + w ∧ c = 6 - 4 * complex.I + w ∧ d = 9 + complex.I

theorem translation_proof :
  complex_translation (1 - 3 * complex.I) (4 + 2 * complex.I) (c := 9 + complex.I) (d := 9 + complex.I) :=
begin
  use 3 + 5 * complex.I,
  split,
  { simp, ring, },
  { split,
    { simp, ring, },
    { simp, ring, } }
end

end translation_proof_l627_627039


namespace smallest_positive_n_common_factor_l627_627725

theorem smallest_positive_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ (∃ d : ℕ, d > 1 ∧ d ∣ (8 * n - 3) ∧ d ∣ (6 * n + 4)) ∧ n = 1 :=
by
  sorry

end smallest_positive_n_common_factor_l627_627725


namespace a_1000_value_l627_627195

noncomputable def a : ℕ → ℤ
| 0     := 2010  -- Lean uses zero-indexing, adjust accordingly
| 1     := 2011
| (n+2) := 2 * (n + 1) - (a n + a (n + 1))

theorem a_1000_value : a 999 = 2676 :=  -- Adjusted indexing for Lean
by
  sorry

end a_1000_value_l627_627195


namespace pie_eating_contest_l627_627323

theorem pie_eating_contest:
  let pie1 := 4/5
  let pie2 := 5/6
  let pie3 := 3/4
  max pie1 (max pie2 pie3) - min pie1 (min pie2 pie3) = 1/12 := by
    sorry

end pie_eating_contest_l627_627323


namespace focal_length_is_valid_l627_627150

noncomputable def hyperbola_focal_length : Prop :=
  ∀ (a b d : ℝ), a > 0 ∧ b > 0 ∧ d = 1 ∧ (sin (4/5) = 4/5) → 
  (b = 2 * a ∨ b = a / 2) → 
  (let c1 := b * (sqrt(a^2 + b^2)) / b, c2 := sqrt(b^2) in c1 = sqrt(5) ∨ c1 = 2 * sqrt(5))

theorem focal_length_is_valid : hyperbola_focal_length :=
  sorry

end focal_length_is_valid_l627_627150


namespace number_of_non_congruent_triangles_perimeter_18_l627_627943

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l627_627943


namespace hyperbola_eccentricity_l627_627577

def is_hyperbola (x y a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def is_right_branch (x y a b : ℝ) : Prop := ∃ P : ℝ × ℝ, is_hyperbola P.1 P.2 a b ∧ P.1 > 0

def perpendicular (F1 F2 P : ℝ × ℝ) : Prop := 
  let v1 := ((P.1 - F1.1), (P.2 - F1.2))
  let v2 := ((P.1 - F2.1), (P.2 - F2.2))
  (v1.1 * v2.1 + v1.2 * v2.2) = 0

def arithmetic_sequence (d1 d2 d3 : ℝ) : Prop := 2 * d2 = d1 + d3

def focal_distance (a b : ℝ) : ℝ := √(a^2 + b^2)

noncomputable def eccentricity (a b : ℝ) : ℝ := (focal_distance a b) / a

theorem hyperbola_eccentricity (a b : ℝ) (F1 F2 P : ℝ × ℝ)
  (h_hyperbola : is_hyperbola F1.1 F1.2 a b)
  (h_right_branch : is_right_branch P.1 P.2 a b)
  (h_perpendicular : perpendicular F1 F2 P)
  (h_arithmetic : arithmetic_sequence (dist F1 F2) (dist F1 P) (dist P F2)) :
  eccentricity a b = 5 := by
  sorry

end hyperbola_eccentricity_l627_627577


namespace hanoi_moves_minimal_l627_627397

theorem hanoi_moves_minimal (n : ℕ) : ∃ m, 
  (∀ move : ℕ, move = 2^n - 1 → move = m) := 
by
  sorry

end hanoi_moves_minimal_l627_627397


namespace find_a_l627_627282

theorem find_a (a : ℝ) (h : (x^2 + 1) * (x + a)^8 ≡ 113 * x^8 + C_α(x, 8)) : 
  (a = 2 ∨ a = -2) := 
by sorry

end find_a_l627_627282


namespace total_pencils_correct_l627_627420

variable (donna_pencils marcia_pencils cindi_pencils : ℕ)

-- Given conditions translated into Lean
def condition1 : Prop := donna_pencils = 3 * marcia_pencils
def condition2 : Prop := marcia_pencils = 2 * cindi_pencils
def condition3 : Prop := cindi_pencils = 30 / 0.5

-- The proof statement
theorem total_pencils_correct : 
  condition1 ∧ condition2 ∧ condition3 → donna_pencils + marcia_pencils = 480 :=
begin
  -- Placeholder for the actual proof
  sorry
end

end total_pencils_correct_l627_627420


namespace rate_is_correct_l627_627295

/-- Define the conditions of the problem -/
def roomLength : ℝ := 5.5
def roomWidth : ℝ := 3.75
def totalCost : ℝ := 16500

/-- Define the computation for the area -/
def area : ℝ := roomLength * roomWidth

/-- Define the computation for the rate -/
def ratePerSquareMeter : ℝ := totalCost / area

/-- The statement of the problem -/
theorem rate_is_correct : ratePerSquareMeter = 800 := by
  sorry

end rate_is_correct_l627_627295


namespace total_profit_or_loss_is_negative_175_l627_627032

theorem total_profit_or_loss_is_negative_175
    (price_A price_B selling_price : ℝ)
    (profit_A loss_B : ℝ)
    (h1 : selling_price = 2100)
    (h2 : profit_A = 0.2)
    (h3 : loss_B = 0.2)
    (hA : price_A * (1 + profit_A) = selling_price)
    (hB : price_B * (1 - loss_B) = selling_price) :
    (selling_price + selling_price) - (price_A + price_B) = -175 := 
by 
  -- The proof is omitted
  sorry

end total_profit_or_loss_is_negative_175_l627_627032


namespace jina_mascots_l627_627563

variables (x y z x_new Total : ℕ)

def mascots_problem :=
  (y = 3 * x) ∧
  (x_new = x + 2 * y) ∧
  (z = 2 * y) ∧
  (Total = x_new + y + z) →
  Total = 16 * x

-- The statement only, no proof is required
theorem jina_mascots : mascots_problem x y z x_new Total := sorry

end jina_mascots_l627_627563


namespace tangent_line_x_squared_l627_627850

theorem tangent_line_x_squared (P : ℝ × ℝ) (hP : P = (1, -1)) :
  ∃ (a : ℝ), a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 ∧
    ((∀ x : ℝ, (2 * (1 + Real.sqrt 2) * x - (3 + 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 + Real.sqrt 2) * P.1 - (3 + 2 * Real.sqrt 2))) ∨
    (∀ x : ℝ, (2 * (1 - Real.sqrt 2) * x - (3 - 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 - Real.sqrt 2) * P.1 - (3 - 2 * Real.sqrt 2)))) := by
  sorry

end tangent_line_x_squared_l627_627850


namespace find_x_l627_627478

-- Define the vectors and the condition of them being parallel
def vector_a : (ℝ × ℝ) := (3, 1)
def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -1)
def parallel (a b : (ℝ × ℝ)) := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The theorem to prove
theorem find_x (x : ℝ) (h : parallel (3, 1) (x, -1)) : x = -3 :=
by
  sorry

end find_x_l627_627478


namespace interval_of_monotonic_decrease_minimum_value_in_interval_l627_627146

noncomputable def f (x a : ℝ) : ℝ := 1 / x + a * Real.log x

-- Define the derivative of f
noncomputable def f_prime (x a : ℝ) : ℝ := (a * x - 1) / x^2

-- Prove that the interval of monotonic decrease is as specified
theorem interval_of_monotonic_decrease (a : ℝ) :
  if a ≤ 0 then ∀ x ∈ Set.Ioi (0 : ℝ), f_prime x a < 0
  else ∀ x ∈ Set.Ioo 0 (1/a), f_prime x a < 0 := sorry

-- Prove that, given x in [1/2, 1], the minimum value of f(x) is 0 when a = 2 / log 2
theorem minimum_value_in_interval :
  ∃ a : ℝ, (a = 2 / Real.log 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f x a ≥ 0 ∧ (∃ y ∈ Set.Icc (1/2 : ℝ) 1, f y a = 0) := sorry

end interval_of_monotonic_decrease_minimum_value_in_interval_l627_627146


namespace range_f_pos_l627_627127

noncomputable def f : ℝ → ℝ := sorry
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom increasing_f : ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x ≤ f y
axiom f_at_neg_one : f (-1) = 0

theorem range_f_pos : {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := 
by
  sorry

end range_f_pos_l627_627127


namespace helicopter_rental_cost_l627_627325

theorem helicopter_rental_cost :
  let hours_per_day := 2
  let days := 3
  let rate_first_day := 85
  let rate_second_day := 75
  let rate_third_day := 65
  let total_cost_before_discount := hours_per_day * rate_first_day + hours_per_day * rate_second_day + hours_per_day * rate_third_day
  let discount := 0.05
  let discounted_amount := total_cost_before_discount * discount
  let total_cost_after_discount := total_cost_before_discount - discounted_amount
  total_cost_after_discount = 427.50 :=
by
  sorry

end helicopter_rental_cost_l627_627325


namespace non_congruent_triangles_with_perimeter_18_l627_627978

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l627_627978


namespace first_common_digit_three_digit_powers_l627_627333

theorem first_common_digit_three_digit_powers (m n: ℕ) (hm: 100 ≤ 2^m ∧ 2^m < 1000) (hn: 100 ≤ 3^n ∧ 3^n < 1000) :
  (∃ d, (2^m).div 100 = d ∧ (3^n).div 100 = d ∧ d = 2) :=
sorry

end first_common_digit_three_digit_powers_l627_627333


namespace diameter_of_circle_A_l627_627826

theorem diameter_of_circle_A (r_B r_C : ℝ) (h1 : r_B = 12) (h2 : r_C = 3)
  (area_relation : ∀ (r_A : ℝ), π * (r_B^2 - r_A^2) = 4 * (π * r_C^2)) :
  ∃ r_A : ℝ, 2 * r_A = 12 * Real.sqrt 3 := by
  -- We will club the given conditions and logical sequence here
  sorry

end diameter_of_circle_A_l627_627826


namespace part_a_l627_627389

theorem part_a (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, a (n + 2) = a (n + 1) * a n + 1) :
  ∀ n, ¬ (4 ∣ a n) :=
by
  sorry

end part_a_l627_627389


namespace area_of_transformed_triangle_l627_627654

-- The function g(x) defined on the domain {x_4, x_5, x_6}
variable (x4 x5 x6 : ℝ)
variable (g : ℝ → ℝ)

-- Condition: The area of the original triangle is 50
variable (area_original : ℝ)
hypothesis h_area_original : area_original = 50

-- Defining the transformation function
def transform (x : ℝ) : ℝ × ℝ := (x / 3, 3 * g x)

-- The points after transformation
def points_transformed := {(x4 / 3, 3 * g x4), (x5 / 3, 3 * g x5), (x6 / 3, 3 * g x6)}

-- Required proof statement
theorem area_of_transformed_triangle : 
  let area_original := 50 in
  (area_original = 50) → 
  (area_points (points_transformed x4 x5 x6 g) = 50) :=
by
  sorry

end area_of_transformed_triangle_l627_627654


namespace series_converges_l627_627762

noncomputable def a_n_seq : ℕ → ℕ :=
sorry -- sequence definition (a_n) based on natural numbers without digit 1

theorem series_converges :
  (∀ n, a_n_seq n ∈ ℕ ∧
        (∀ k, (10^k ≤ a_n_seq n ∨ a_n_seq n ∉ {m | m.mod 1 = 1}))) →
  ∑' n, 1 / (a_n_seq n : ℝ) < ∞ :=
begin
  sorry -- proof goes here
end

end series_converges_l627_627762


namespace sum_of_roots_l627_627860

-- Define polynomials
def P1 (x : ℝ) : ℝ := 3 * x^3 - 6 * x^2 - 9 * x + 27
def P2 (x : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + 16

-- Using Vieta's formulas, the sum of the roots of each polynomial is:
def sum_roots_P1 : ℝ := -((-6)/3)  -- i.e., 2
def sum_roots_P2 : ℝ := -((-8)/4)  -- i.e., 2

-- Main theorem to prove
theorem sum_of_roots :
  sum_roots_P1 + sum_roots_P2 = 4 := 
by
  rw [sum_roots_P1, sum_roots_P2]
  norm_num -- This simplifies the expression.
  sorry -- Proof will be filled here.

end sum_of_roots_l627_627860


namespace moles_HCl_formed_l627_627854

-- Definitions for the conditions of the problem
def NaCl : Type := ℝ
def HNO3 : Type := ℝ
def HCl : Type := ℝ
def Reaction : NaCl → HNO3 → HCl → Prop :=
λ nacl hno3 hcl, nacl = 1 ∧ hno3 = 1 ∧ hcl = 1

-- Theorem statement proving the production of 1 mole of HCl
theorem moles_HCl_formed (nacl : NaCl) (hno3 : HNO3) : ∃ hcl : HCl, Reaction nacl hno3 hcl :=
by {
  existsi (1 : HCl),
  simp,
  tauto,
}

end moles_HCl_formed_l627_627854


namespace poly_has_two_distinct_negative_real_roots_l627_627834

-- Definition of the polynomial equation
def poly_eq (p x : ℝ) : Prop :=
  x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1 = 0

-- Theorem statement that needs to be proved
theorem poly_has_two_distinct_negative_real_roots (p : ℝ) :
  p > 1 → ∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly_eq p x1 ∧ poly_eq p x2 :=
by
  sorry

end poly_has_two_distinct_negative_real_roots_l627_627834


namespace cone_base_circumference_l627_627772

-- We establish the given problem conditions
def radius : ℝ := 6
def angle_sector : ℝ := 180

-- Definition to compute the circumference of a circle
def circumference_of_circle (r : ℝ) : ℝ := 2 * Real.pi * r

-- Definition to compute the angle fraction of a sector
def angle_fraction (angle_sector : ℝ) : ℝ := angle_sector / 360

-- The theorem to prove the circumference of the base of the cone
theorem cone_base_circumference : 
  circumference_of_circle radius * angle_fraction angle_sector = 6 * Real.pi :=
by
  sorry

end cone_base_circumference_l627_627772


namespace prop_1_prop_3_prop_4_l627_627909

variables {m n : Line} {α β : Plane} {A : Point}

-- Proposition 1: m ⊥ α, n ∥ α ⇒ m ⊥ n
theorem prop_1 (h1 : m ⊥ α) (h2 : n ∥ α) : m ⊥ n := sorry

-- Proposition 3: m ∥ n, n ⊥ β, m ∥ α ⇒ α ⊥ β
theorem prop_3 (h1 : m ∥ n) (h2 : n ⊥ β) (h3 : m ∥ α) : α ⊥ β := sorry

-- Proposition 4: m ∩ n = A, m ∥ α, m ∥ β, n ∥ α, n ∥ β ⇒ α ∥ β
theorem prop_4 (h1 : m ∩ n = A) (h2 : m ∥ α) (h3 : m ∥ β) (h4 : n ∥ α) (h5 : n ∥ β) : α ∥ β := sorry

end prop_1_prop_3_prop_4_l627_627909


namespace max_k_guarded_l627_627761

-- Define the size of the board
def board_size : ℕ := 8

-- Define the directions a guard can look
inductive Direction
| up | down | left | right

-- Define a guard's position on the board as a pair of Fin 8
def Position := Fin board_size × Fin board_size

-- Guard record that contains its position and direction
structure Guard where
  pos : Position
  dir : Direction

-- Function to determine if guard A is guarding guard B
def is_guarding (a b : Guard) : Bool :=
  match a.dir with
  | Direction.up    => a.pos.1 < b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.down  => a.pos.1 > b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.left  => a.pos.1 = b.pos.1 ∧ a.pos.2 > b.pos.2
  | Direction.right => a.pos.1 = b.pos.1 ∧ a.pos.2 < b.pos.2

-- The main theorem states that the maximum k is 5
theorem max_k_guarded : ∃ k : ℕ, (∀ g : Guard, ∃ S : Finset Guard, (S.card ≥ k) ∧ (∀ s ∈ S, is_guarding s g)) ∧ k = 5 :=
by
  sorry

end max_k_guarded_l627_627761


namespace find_sum_of_smallest_multiples_l627_627521

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l627_627521


namespace coins_difference_l627_627509

theorem coins_difference (cents_in_dollar : ℕ) (cents_in_nickel : ℕ) (cents_in_dime : ℕ) :
  (cents_in_dollar = 100) → (cents_in_nickel = 5) → (cents_in_dime = 10) →
  (cents_in_dollar / cents_in_nickel) - (cents_in_dollar / cents_in_dime) = 10 :=
by
  intro h1 h2 h3
  have hn : cents_in_dollar / cents_in_nickel = 20 := by
    rw [h1, h2]
    exact Nat.div_eq_of_eq_mul_right (by norm_num : 0 < 5) rfl
  have hd : cents_in_dollar / cents_in_dime = 10 := by
    rw [h1, h3]
    exact Nat.div_eq_of_eq_mul_right (by norm_num : 0 < 10) (rfl)
  rw [hn, hd]
  norm_num


end coins_difference_l627_627509


namespace find_x_values_l627_627856

theorem find_x_values (x : ℝ) :
  (3 ≤ |x - 2| ∧ |x - 2| ≤ 7) ∧ x^2 ≤ 36 ↔ (x ∈ set.Icc (-5) (-1) ∨ x ∈ set.Icc 5 6) :=
by
  sorry

end find_x_values_l627_627856


namespace num_palindromic_years_l627_627102

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string.to_list
  s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

def is_palindrome_prime (n : ℕ) : Prop :=
  is_palindrome n ∧ is_prime n

def satisfies_properties (year : ℕ) : Prop :=
  is_palindrome year ∧ (∃ (p1 p2 : ℕ), p1 * p2 = year ∧ is_palindrome p1 ∧ is_prime p1 ∧ is_palindrome p2 ∧ is_prime p2)

theorem num_palindromic_years : (finset.filter satisfies_properties (finset.range 1001 2001)).card = 4 :=
by
  sorry

end num_palindromic_years_l627_627102


namespace weird_numbers_count_2_to_100_l627_627822

def is_weird (n : ℕ) : Prop :=
  ¬(n ∣ nat.factorial (n - 2))

def weird_numbers_between (a b : ℕ) : List ℕ :=
  List.filter is_weird (List.range' a (b - a + 1))

theorem weird_numbers_count_2_to_100 : (weird_numbers_between 2 100).length = 26 := by
  sorry

end weird_numbers_count_2_to_100_l627_627822


namespace length_squared_t_graph_interval_l627_627275

noncomputable def p (x : ℝ) : ℝ := -x + 2
noncomputable def q (x : ℝ) : ℝ := x + 2
noncomputable def r (x : ℝ) : ℝ := 2
noncomputable def t (x : ℝ) : ℝ :=
  if x ≤ -2 then p x
  else if x ≤ 2 then r x
  else q x

theorem length_squared_t_graph_interval :
  let segment_length (f : ℝ → ℝ) (a b : ℝ) : ℝ := Real.sqrt ((f b - f a)^2 + (b - a)^2)
  segment_length t (-4) (-2) + segment_length t (-2) 2 + segment_length t 2 4 = 4 + 2 * Real.sqrt 32 →
  (4 + 2 * Real.sqrt 32)^2 = 80 :=
sorry

end length_squared_t_graph_interval_l627_627275


namespace total_pencils_correct_l627_627422

variable (donna_pencils marcia_pencils cindi_pencils : ℕ)

-- Given conditions translated into Lean
def condition1 : Prop := donna_pencils = 3 * marcia_pencils
def condition2 : Prop := marcia_pencils = 2 * cindi_pencils
def condition3 : Prop := cindi_pencils = 30 / 0.5

-- The proof statement
theorem total_pencils_correct : 
  condition1 ∧ condition2 ∧ condition3 → donna_pencils + marcia_pencils = 480 :=
begin
  -- Placeholder for the actual proof
  sorry
end

end total_pencils_correct_l627_627422


namespace rational_pairs_natural_conditions_l627_627074

theorem rational_pairs_natural_conditions :
  ∀ x y : ℚ, 
  (∃ a b : ℕ, x = a / b ∧ (a.gcd b = 1) ∧ (x + 1/y).isNat ∧ (y + 1/x).isNat) ->
  (x, y) = (1,1) ∨ (x, y) = (2,1) ∨ (x, y) = (1,2) ∨ (x, y) = (2,2) := 
by 
  sorry

end rational_pairs_natural_conditions_l627_627074


namespace bakery_sold_boxes_l627_627766

theorem bakery_sold_boxes
  (boxes_capacity : ℕ)
  (total_doughnuts : ℕ)
  (given_away_doughnuts : ℕ)
  (boxes_capacity_eq : boxes_capacity = 10)
  (total_doughnuts_eq : total_doughnuts = 300)
  (given_away_doughnuts_eq : given_away_doughnuts = 30) :
  total_doughnuts / boxes_capacity - given_away_doughnuts / boxes_capacity = 27 :=
by
  rw [boxes_capacity_eq, total_doughnuts_eq, given_away_doughnuts_eq]
  norm_num
  sorry

end bakery_sold_boxes_l627_627766


namespace strawberry_candies_count_l627_627699

theorem strawberry_candies_count (S G : ℕ) (h1 : S + G = 240) (h2 : G = S - 2) : S = 121 :=
by
  sorry

end strawberry_candies_count_l627_627699


namespace sum_b_eq_l627_627124

namespace SequenceSum

-- Definitions of sequences a_n and b_n.
def a : ℕ → ℕ
| 0 => 0
| 1 => 2
| (n + 2) => a (n + 1) ^ 2 - (n + 1) * a (n + 1) + 1

def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define the sum S_n
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k, b k)

-- The theorem to prove
theorem sum_b_eq {n : ℕ} : S n = 1/2 - 1/(n+2) :=
sorry

end SequenceSum

end sum_b_eq_l627_627124


namespace quadratic_inequality_solution_l627_627433

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 18 < 0} = {x : ℝ | -3 < x ∧ x < 6} :=
by
  sorry

end quadratic_inequality_solution_l627_627433


namespace runner_overtake_time_l627_627322

theorem runner_overtake_time
  (L : ℝ)
  (v1 v2 v3 : ℝ)
  (h1 : v1 = v2 + L / 6)
  (h2 : v1 = v3 + L / 10) :
  L / (v3 - v2) = 15 := by
  sorry

end runner_overtake_time_l627_627322


namespace angle_ADB_eq_three_angle_BAC_l627_627200

theorem angle_ADB_eq_three_angle_BAC
  (ABC : Triangle)
  (D E : Point)
  (h1 : is_acute_triangle ABC)
  (h2 : D ∈ AC)
  (h3 : segment_length AD = segment_length BC)
  (h4 : segment_length AC ^ 2 - segment_length AD ^ 2 = segment_length AC * segment_length AD)
  (h5 : ∃ l, is_parallel_line l (angle_bisector ∠ACB) ∧ D ∈ l ∧ ∃ E, E ∈ AB ∧ l ∩ AB = {E})
  (h6 : segment_length AE = segment_length CD) :
  angle ADB = 3 * angle BAC :=
sorry

end angle_ADB_eq_three_angle_BAC_l627_627200


namespace p_minus_q_eq_16_sqrt_2_l627_627587

theorem p_minus_q_eq_16_sqrt_2 (p q : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 28 * x - 84 → x = p ∨ x = q)
  (h_distinct : p ≠ q) (h_p_gt_q : p > q) : p - q = 16 * Real.sqrt 2 :=
sorry

end p_minus_q_eq_16_sqrt_2_l627_627587


namespace center_of_circle_l627_627835

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

-- Define the condition for the center of the circle
def is_center_of_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = 4

-- The main theorem to be proved
theorem center_of_circle : is_center_of_circle 1 (-1) :=
by
  sorry

end center_of_circle_l627_627835


namespace product_divisible_by_perfect_square_l627_627274

theorem product_divisible_by_perfect_square 
  (S : Finset ℕ)
  (h₁ : ∀ x ∈ S, x ∈ Finset.range 11)
  (h₂ : S.card = 6) 
  : ∃ p : ℕ, p > 1 ∧ ∃ k : ℕ, k > 1 ∧ p = k * k ∧ ∃ product : ℕ, ∏ i in S, i % p = 0 :=
by
  sorry

end product_divisible_by_perfect_square_l627_627274


namespace concert_revenue_l627_627758

theorem concert_revenue :
  let base_price := 20
  let discount_40 := 0.40
  let discount_15 := 0.15
  let first_group_people := 10
  let second_group_people := 20
  let total_people := 45
  let first_group_revenue := first_group_people * (base_price - discount_40 * base_price)
  let second_group_revenue := second_group_people * (base_price - discount_15 * base_price)
  let remaining_group_revenue := (total_people - first_group_people - second_group_people) * base_price
  in first_group_revenue + second_group_revenue + remaining_group_revenue = 760 :=
  by
  sorry

end concert_revenue_l627_627758


namespace hall_length_width_difference_l627_627006

theorem hall_length_width_difference :
  ∃ (L W : ℝ), 
  (W = (1 / 2) * L) ∧
  (L * W = 288) ∧
  (L - W = 12) :=
by
  -- The mathematical proof follows from the conditions given
  sorry

end hall_length_width_difference_l627_627006


namespace locus_of_circumcenter_of_triangle_ABE_is_circular_arc_l627_627396

-- We define the necessary geometric entities and conditions
variables {O A B C D E O1 O2 : Type*}
variables {circleO : Set O}
variables [geometry.circle O A C]
variables {BD : Set D}
variables (BD_perp_OC : ∀ {BD OC}, BD ⊥ OC)
variables (D_on_minor_arc_AC : ∀ {D A C}, D ∈ arc_minor A C)
variables (E : Set E)

-- The hypothesis that BD intersects AC at E
variables (BD_intersects_AC_E : ∀ {BD AC E}, BD ∩ AC = E)

-- The definition of the problem: locus of the circumcenter of triangle ABE is a circular arc
theorem locus_of_circumcenter_of_triangle_ABE_is_circular_arc 
    (H_locus : ∀ (triangle ABE), ∃ (circumcenter O1 : O), O1 ∈ arc_with_center_and_angle O A C (1/2) (arc_angle A C)) :
    statement :=
begin
    sorry
end

end locus_of_circumcenter_of_triangle_ABE_is_circular_arc_l627_627396


namespace speed_of_other_person_l627_627650

-- Definitions related to the problem conditions
def pooja_speed : ℝ := 3  -- Pooja's speed in km/hr
def time : ℝ := 4  -- Time in hours
def distance : ℝ := 20  -- Distance between them after 4 hours in km

-- Define the unknown speed S as a parameter to be solved
variable (S : ℝ)

-- Define the relative speed when moving in opposite directions
def relative_speed (S : ℝ) : ℝ := S + pooja_speed

-- Create a theorem to encapsulate the problem and to be proved
theorem speed_of_other_person 
  (h : distance = relative_speed S * time) : S = 2 := 
  sorry

end speed_of_other_person_l627_627650


namespace find_values_of_M_l627_627041

theorem find_values_of_M :
  ∃ M : ℕ, 
    (M = 81 ∨ M = 92) ∧ 
    (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ M = 10 * a + b ∧
     (∃ k : ℕ, k ^ 3 = 9 * (a - b) ∧ k > 0)) :=
sorry

end find_values_of_M_l627_627041


namespace n_pow_m_l627_627176

theorem n_pow_m (
  m n : ℤ 
  (h : ∀ x : ℂ, x^2 - 3 * x + m = (x - 1) * (x + n))
) : n^m = 4 := by
  sorry

end n_pow_m_l627_627176


namespace school_uniforms_for_height_range_l627_627123

noncomputable def school_uniforms_needed : ℕ :=
  let mu := 173
  let sigma := 5
  let total_students := 10000
  let probability := 99.73 / 100
  total_students * probability

theorem school_uniforms_for_height_range :
  school_uniforms_needed = 9973 :=
by
  -- The correct response would be to prove the theorem, but we leave it as sorry for now.
  sorry

end school_uniforms_for_height_range_l627_627123


namespace probability_sum_is_eight_l627_627343

theorem probability_sum_is_eight (n : ℕ) (hn : n = 6 * 6) (f : Fin 6 × Fin 6 → ℕ) 
  (hf : ∀ p : Fin 6 × Fin 6, f p = p.1 + 1 + (p.2 + 1)) : 
  (∑ p in (Finset.univ : Finset (Fin 6 × Fin 6)), if f p = 8 then 1 else 0) / n = 5 / 36 :=
sorry

end probability_sum_is_eight_l627_627343


namespace students_scored_no_less_than_90_l627_627544

noncomputable def normal_distribution (mean variance : ℝ) : Type := sorry

variable (X : ℝ → ℝ)
variable (σ² : ℝ)
variable (school_students : ℝ := 400)

axiom normal_X : normal_distribution 84 σ²
axiom prob_78_to_84 : P (78 < X ≤ 84) = 0.3

theorem students_scored_no_less_than_90 : 
  (estimated_students_scored_no_less_than_90 school_students 90 84 σ²) = 80 :=
by 
  sorry

end students_scored_no_less_than_90_l627_627544


namespace period_is_seven_l627_627862

-- Define the conditions
def apples_per_sandwich (a : ℕ) := a = 4
def sandwiches_per_day (s : ℕ) := s = 10
def total_apples (t : ℕ) := t = 280

-- Define the question to prove the period
theorem period_is_seven (a s t d : ℕ) 
  (h1 : apples_per_sandwich a)
  (h2 : sandwiches_per_day s)
  (h3 : total_apples t)
  (h4 : d = t / (a * s)) 
  : d = 7 := 
sorry

end period_is_seven_l627_627862


namespace math_problem_l627_627133

theorem math_problem 
  (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := 
by
  sorry

end math_problem_l627_627133


namespace range_of_a_l627_627499

variable (x a : ℝ)

theorem range_of_a (h1 : ∀ x, x ≤ a → x < 2) (h2 : ∀ x, x < 2) : a ≥ 2 :=
sorry

end range_of_a_l627_627499


namespace probability_divisor_of_8_is_half_l627_627788

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l627_627788


namespace repeating_decimal_sum_l627_627412

theorem repeating_decimal_sum {x : ℚ} (h₀ : x = 0.457) : (x.num + x.denom) = 1456 :=
by sorry

end repeating_decimal_sum_l627_627412


namespace smallest_6digit_palindrome_base2_is_4digit_palindrome_base5_l627_627447

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

def smallest_6digit_palindrome_base2 : ℕ :=
  33 -- This is the decimal representation of 100001_2

theorem smallest_6digit_palindrome_base2_is_4digit_palindrome_base5 :
  is_palindrome smallest_6digit_palindrome_base2 2 ∧
  (∃ b, b ≠ 2 ∧ is_palindrome smallest_6digit_palindrome_base2 b ∧ Nat.digits b smallest_6digit_palindrome_base2).length = 4 :=
by
  sorry

end smallest_6digit_palindrome_base2_is_4digit_palindrome_base5_l627_627447


namespace total_students_l627_627622

-- Define the conditions
def chocolates_distributed (y z : ℕ) : ℕ :=
  y * y + z * z

-- Define the main theorem to be proved
theorem total_students (y z : ℕ) (h : z = y + 3) (chocolates_left: ℕ) (initial_chocolates: ℕ)
  (h_chocolates: chocolates_distributed y z = initial_chocolates - chocolates_left) : 
  y + z = 33 :=
by
  sorry

end total_students_l627_627622


namespace false_divisibility_statement_l627_627624

theorem false_divisibility_statement (a : ℕ) (h2 : a % 2 = 0) (h4 : a % 4 = 0) (h12 : a % 12 = 0) (div2 : ∃ k : ℕ, a = 2 * k) (div4 : ∃ m : ℕ, a = 4 * m) (div12 : ∃ n : ℕ, a = 12 * n) : ¬ (a % 24 = 0 ∧ ∃ p : ℕ, a = 24 * p) → (∏ S ∈ {2, 4, 12, 24}, (a % S = 0)) = 3 :=
by
  sorry

end false_divisibility_statement_l627_627624


namespace minimum_area_of_triangle_l627_627550

noncomputable def area_of_triangle (b : ℝ) (k : ℝ) (h : b ≠ 0) (hk : k = b / (b - 1)) : ℝ :=
  abs ((b^2 - b) / 2)

theorem minimum_area_of_triangle (b : ℝ) (h : b ≠ 0) (hb : b ≥ 2) :
  area_of_triangle b (b / (b - 1)) h (by { rw [mul_div_cancel' b (sub_ne_zero_of_ne (ne_of_gt (lt_of_lt_of_le zero_lt_one hb))), mul_one] }) = 1 :=
sorry

end minimum_area_of_triangle_l627_627550


namespace tangent_line_eq_min_a_satisfies_ineq_l627_627490

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_eq (t : ℝ) (h : f t = 0) : 
  ∃ k b, (λ x, k * x + b) = (λ x, derivative f t * (x - t) + f t) ∧ k = 1 ∧ b = -1 := by
  sorry

theorem min_a_satisfies_ineq (a : ℝ) : 
  (∀ x : ℝ, 0 < x → f x ≥ a * x^2 + 2 / a) ↔ a ≥ -Real.exp 3 := by
  sorry

end tangent_line_eq_min_a_satisfies_ineq_l627_627490


namespace ratio_of_ducks_l627_627645

theorem ratio_of_ducks (lily_ducks lily_geese rayden_geese rayden_ducks : ℕ) 
  (h1 : lily_ducks = 20) 
  (h2 : lily_geese = 10) 
  (h3 : rayden_geese = 4 * lily_geese) 
  (h4 : rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70) : 
  rayden_ducks / lily_ducks = 3 :=
by
  sorry

end ratio_of_ducks_l627_627645


namespace subset_implies_inequality_l627_627927

noncomputable def set_A : set ℝ := { x | x^2 - 3*x + 2 ≤ 0 }
noncomputable def set_B (a : ℝ) : set ℝ := { x | x^2 - (a + 1)*x + a ≤ 0 }

theorem subset_implies_inequality (a : ℝ) : set_A ⊆ set_B a → a ≥ 2 := by
  sorry

end subset_implies_inequality_l627_627927


namespace frosting_cupcakes_total_l627_627060

noncomputable def rate_cagney := 1 / 24  -- Rate of Cagney frosting cupcakes
noncomputable def rate_lacey := 1 / 30  -- Rate of Lacey frosting cupcakes
noncomputable def rate_casey := 1 / 40  -- Rate of Casey frosting cupcakes

theorem frosting_cupcakes_total : 
  let combined_rate := rate_cagney + rate_lacey + rate_casey,
      total_time := 6 * 60 in
  combined_rate * total_time = 36 := 
by
  sorry

end frosting_cupcakes_total_l627_627060


namespace highest_success_ratio_beta_achievable_l627_627044

theorem highest_success_ratio_beta_achievable :
  ∀ (a b c d : ℕ),
    a = 220 ∧ b = 300 ∧ c = 200 ∧ d = 200 → 
    0 < a ∧ 0 < c ∧ 
    a < b * 2 / 3 ∧ c < d / 2 ∧
    200 < b ∧ 100 < d → 
    ((a + c) / 500 : ℝ) ≤ 62 / 125 :=
begin
  sorry
end

end highest_success_ratio_beta_achievable_l627_627044


namespace calculate_mirror_area_l627_627625

def outer_frame_width : ℝ := 65
def outer_frame_height : ℝ := 85
def frame_width : ℝ := 15

def mirror_width : ℝ := outer_frame_width - 2 * frame_width
def mirror_height : ℝ := outer_frame_height - 2 * frame_width
def mirror_area : ℝ := mirror_width * mirror_height

theorem calculate_mirror_area : mirror_area = 1925 := by
  sorry

end calculate_mirror_area_l627_627625


namespace point_coordinates_l627_627315

def point : Type := ℝ × ℝ

def x_coordinate (P : point) : ℝ := P.1

def y_coordinate (P : point) : ℝ := P.2

theorem point_coordinates (P : point) (h1 : x_coordinate P = -3) (h2 : abs (y_coordinate P) = 5) :
  P = (-3, 5) ∨ P = (-3, -5) :=
by
  sorry

end point_coordinates_l627_627315


namespace triangle_area_is_100_l627_627610

noncomputable def triangle_area (AF BE : ℝ) (h1 : AF = 10) (h2 : BE = 15) (h3 : ⟪AF, BE⟫ = 0) : ℝ :=
if h1 ∧ h2 ∧ h3 then 100 else 0

theorem triangle_area_is_100 (h1 : AF = 10) (h2 : BE = 15) (h3 : ⟪AF, BE⟫ = 0) :
  triangle_area 10 15 h1 h2 h3 = 100 := by sorry

end triangle_area_is_100_l627_627610


namespace intersection_A_B_l627_627896

def set_A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def set_B : Set ℝ := {x | 2^(x-1) > 1}

theorem intersection_A_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l627_627896


namespace smallest_non_negative_k_l627_627858

theorem smallest_non_negative_k (x : Fin 50 → ℝ)
  (h_sum : ∑ i, x i = 0)
  (h_order : ∀ i j, i ≤ j → x i ≤ x j):
  (∑ i, (x i)^2) ≥ 0 * (x 24)^2 := 
by
  sorry

end smallest_non_negative_k_l627_627858


namespace polynomial_value_sum_l627_627568

-- Define the polynomials and their properties
def p (x : ℤ) : ℤ := x^6 - x^3 - x^2 - 1

theorem polynomial_value_sum :
  ∃ (q : ℕ → polynomial ℤ) (m : ℕ), 
  (∀ i, 1 ≤ i ∧ i ≤ m → is_monic (q i) ∧ is_irreducible (q i) ∧ degree (q i) > 0 ∧ p = ∏ i in finset.range m, q i) ∧ 
  m = 2 ∧ 
  (q 0).eval 3 + (q 1).eval 3 = 81 :=
begin
  sorry
end

end polynomial_value_sum_l627_627568


namespace pucks_cannot_return_to_initial_position_after_25_hits_l627_627251

noncomputable theory
open_locale classical

def puck_problem (hit_count : ℕ) : Prop :=
  ∀ (A B C : ℝ × ℝ), 
  (hit_count % 2 = 1 → ¬∃ (A' B' C' : ℝ × ℝ), (A' = A ∧ B' = B ∧ C' = C))

theorem pucks_cannot_return_to_initial_position_after_25_hits :
  puck_problem 25 :=
by
  sorry

end pucks_cannot_return_to_initial_position_after_25_hits_l627_627251


namespace train_passing_time_l627_627750

theorem train_passing_time
  (length_of_train : ℝ)
  (speed_in_kmph : ℝ)
  (conversion_factor : ℝ)
  (speed_in_mps : ℝ)
  (time : ℝ)
  (H1 : length_of_train = 65)
  (H2 : speed_in_kmph = 36)
  (H3 : conversion_factor = 5 / 18)
  (H4 : speed_in_mps = speed_in_kmph * conversion_factor)
  (H5 : time = length_of_train / speed_in_mps) :
  time = 6.5 :=
by
  sorry

end train_passing_time_l627_627750


namespace non_congruent_triangles_with_perimeter_18_l627_627992

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627992


namespace festive_numbers_count_l627_627367

def is_festive_num (n : ℕ) : Prop := 
  ∃ a b c d : ℕ,
    n = 1000 * a + 100 * b + 10 * c + d ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct digits
    ({a, b, c, d} = {0, 1, 2, 4}) ∧ -- digits must be 0, 1, 2, 4
    a ≠ 0 -- first digit cannot be 0

theorem festive_numbers_count : 
  {n : ℕ | n ≥ 1000 ∧ n < 10000 ∧ is_festive_num n}.toFinset.card = 18 := 
by
  sorry

end festive_numbers_count_l627_627367


namespace probability_adjacent_chinese_athletes_l627_627839

theorem probability_adjacent_chinese_athletes :
  let total_athletes := 8
  let chinese_athletes := 2
  let foreign_athletes := 6
  let total_arrangements := fact total_athletes -- 8!
  let favorable_arrangements := 7 * fact (chinese_athletes - 1) * fact foreign_athletes -- 7 * 2! * 6!
in (favorable_arrangements / total_arrangements = 1 / 4) := 
sorry

end probability_adjacent_chinese_athletes_l627_627839


namespace find_z_l627_627035

noncomputable def longer_parallel_side (side_length : ℝ) (area : ℝ) (shorter_base : ℝ) : ℝ :=
  let z := (2 * area / side_length) - shorter_base
  in z

theorem find_z :
  let s := 2 -- side length of the square
  let A := 4 / 3 -- area of each trapezoid and quadrilateral
  let b := 1 -- length of the shorter base of the trapezoid
  longer_parallel_side s A b = 5 / 3 := sorry

end find_z_l627_627035


namespace non_congruent_triangles_with_perimeter_18_l627_627977

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l627_627977


namespace decreasing_function_l627_627210

theorem decreasing_function (x_1 x_2 : ℝ) (h : x_1 < x_2) : 
  let y := λ x : ℝ, 7 - x in y x_1 > y x_2 := 
by
  sorry

end decreasing_function_l627_627210


namespace find_sum_of_smallest_multiples_l627_627522

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l627_627522


namespace Pascal_High_School_students_l627_627254

def total_students (x a b c : ℕ) :=
  0.5 * x = a + b + 160 ∧
  0.8 * x = a + c + 160 ∧
  0.9 * x = b + c + 160 ∧
  x = a + b + c + 160

theorem Pascal_High_School_students (x a b c : ℕ) :
  total_students x a b c → x = 800 :=
by
  sorry

end Pascal_High_School_students_l627_627254


namespace fibonacci_coprime_l627_627643

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_coprime (n : ℕ) (hn : n ≥ 1) :
  Nat.gcd (fibonacci n) (fibonacci (n - 1)) = 1 := by
  sorry

end fibonacci_coprime_l627_627643


namespace sum_of_digits_square_1222222221_l627_627730

theorem sum_of_digits_square_1222222221 :
  let n := 1222222221
  (∑ d in digits 10 (n^2), d) = 80 :=
by
  sorry

end sum_of_digits_square_1222222221_l627_627730


namespace part1_min_value_part2_find_a_l627_627122

noncomputable def quadratic_function (a x : ℝ) : ℝ := a * x^2 - 4 * a * x + 3 * a

theorem part1_min_value :
  quadratic_function 1 2 = -1 :=
by {
  sorry
}

theorem part2_find_a :
  (∃ a : ℝ, (∀ x ∈ (set.Icc 1 4), quadratic_function a x ≤ 4) ∧ (∀ x ∈ (set.Icc 1 4), quadratic_function a x = 4 ↔ x = 4) ∧ (a = 4 / 3)) :=
by {
  sorry
}

end part1_min_value_part2_find_a_l627_627122


namespace non_congruent_triangles_count_l627_627968

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l627_627968


namespace area_of_rhombus_with_given_vertices_l627_627749

def point := ℝ × ℝ

def rhombus_vertices : set point :=
{ (0, 4.5), (8, 0), (0, -4.5), (-8, 0) }

def diagonal_length (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def area_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem area_of_rhombus_with_given_vertices : 
  let p1 := (0, 4.5)
  let p2 := (8, 0)
  let p3 := (0, -4.5)
  let p4 := (-8, 0)
  let d1 := diagonal_length p1 p3
  let d2 := diagonal_length p2 p4
  area_rhombus d1 d2 = 72 :=
by {
  let p1 := (0, 4.5),
  let p2 := (8, 0),
  let p3 := (0, -4.5),
  let p4 := (-8, 0),
  let d1 := diagonal_length p1 p3,
  let d2 := diagonal_length p2 p4,
  have hd1 : d1 = 9, from sorry, -- calculation of d1
  have hd2 : d2 = 16, from sorry, -- calculation of d2
  have ha : area_rhombus d1 d2 = 72, from sorry, -- final area calculation
  exact ha,
}

end area_of_rhombus_with_given_vertices_l627_627749


namespace second_quadrant_coordinates_l627_627895

theorem second_quadrant_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : y^2 = 1) :
    (x, y) = (-2, 1) :=
  sorry

end second_quadrant_coordinates_l627_627895


namespace sum_expression_l627_627580

noncomputable def T : ℝ := ∑ n in Finset.range 4900, (1 / (Real.sqrt (n+1 + Real.sqrt ((n+1)^2 - 1))))

theorem sum_expression (a b c : ℕ) (ha : a = 98) (hb : b = 70) (hc : c = 2) (hT : T = a + b * Real.sqrt c) : a + b + c = 170 :=
by
  sorry

end sum_expression_l627_627580


namespace vector_projection_l627_627504

namespace VectorProjection

def vec_a : ℝ × ℝ × ℝ := (4, -2, -4)
def vec_b : ℝ × ℝ × ℝ := (6, -3, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def projection (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / (magnitude v2)

theorem vector_projection :
  projection vec_a vec_b = 22 / 7 :=
by
  sorry

end VectorProjection

end vector_projection_l627_627504


namespace range_of_a_l627_627911

open Real

noncomputable def f (a : ℝ) : Piecewise (ℝ → ℝ) :=
  Piecewise (fun x => (a-1) * x + 4 - 2 * a) (fun x => 1 + log x) (fun x => x < 1)

theorem range_of_a (R : Set ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, (if x < 1 then (a-1) * x + 4 - 2 * a else 1 + log x) ∈ R) ↔ a ∈ Ioo 1 2 :=
sorry

end range_of_a_l627_627911


namespace probability_at_least_25_cents_l627_627655

-- Define the coins and their values
inductive Coin
| penny
| nickel
| dime
| quarter₁
| quarter₂

-- Function to get the value of each coin when it is heads
def coin_value (c : Coin) : ℕ :=
  match c with
  | Coin.penny     => 1
  | Coin.nickel    => 5
  | Coin.dime      => 10
  | Coin.quarter₁  => 25
  | Coin.quarter₂  => 25

-- Function to calculate the total value of a list of coins coming up heads
def total_value (coins : List Coin) (heads : Coin → Bool) : ℕ :=
  ((coins.filter heads).map coin_value).sum

-- Set of all possible outcomes (2^5 outcomes)
def all_possible_outcomes : List (Coin → Bool) :=
  List.replicate 32 (λ c, false)   -- Placeholder for all possible flips

-- Condition to check if the total heads value is at least 25 cents
def at_least_25_cents (heads : Coin → Bool) : Prop :=
  total_value [Coin.penny, Coin.nickel, Coin.dime, Coin.quarter₁, Coin.quarter₂] heads ≥ 25

-- Calculate the successful outcomes where at least 25 cents come up heads
def successful_outcomes : ℕ :=
  (all_possible_outcomes.filter at_least_25_cents).length

-- The probability of successful outcomes
def probability_successful : ℚ :=
  successful_outcomes / 32

theorem probability_at_least_25_cents :
  probability_successful = 3 / 8 := by
  sorry

end probability_at_least_25_cents_l627_627655


namespace arithmetic_sequence_difference_l627_627339

theorem arithmetic_sequence_difference :
  let a := -8
  let d := 7
  let a_n (n : ℕ) := a + (n - 1) * d
  a_n 110 - a_n 100 = 70 :=
by
  let a := -8
  let d := 7
  let a_n (n : ℕ) := a + (n - 1) * d
  have h₁ : a_n 100 = -8 + (100 - 1) * 7 := by sorry
  have h₂ : a_n 110 = -8 + (110 - 1) * 7 := by sorry
  have h₃ : a_n 110 - a_n 100 = 70 := by 
    calc
      a_n 110 - a_n 100 = (-8 + (110 - 1) * 7) - (-8 + (100 - 1) * 7)   : by rw [h₁, h₂]
                       ... = (110 - 1) * 7 - (100 - 1) * 7               : by ring
                       ... = 109 * 7 - 99 * 7                           : by norm_num
                       ... = 70                                        : by norm_num
  exact h₃

end arithmetic_sequence_difference_l627_627339


namespace chef_potatoes_l627_627370

theorem chef_potatoes (already_cooked : ℕ) (time_per_potato : ℕ) (time_remaining : ℕ) :
  already_cooked = 8 → time_per_potato = 9 → time_remaining = 63 →
  (already_cooked + time_remaining / time_per_potato) = 15 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end chef_potatoes_l627_627370


namespace balls_in_bag_l627_627018

theorem balls_in_bag (n : ℕ) 
  (h1 : ∀ i, 0 ≤ i ∧ i ≤ 9 → ∀ P, P (λ i, i = 0) → P (λ i, i = 9))
  (h2 : (1 : ℝ) / (6 * n^3 : ℝ) = 0.12) : 
  n = 10 :=
sorry

end balls_in_bag_l627_627018


namespace number_of_females_l627_627806

theorem number_of_females (total_population : ℕ) (bars : ℕ) (population_per_bar : ℕ)
    (is_equal_parts : bars * population_per_bar = total_population) (female_bars : ℕ) :
    female_bars * population_per_bar = 360 :=
by
  have h1 : bars = 4 := by sorry  -- The condition implies bars is 4
  have h2 : total_population = 720 := by sorry  -- Given total population is 720
  have h3 : population_per_bar = total_population / bars := by sorry  -- Each bar represents equal population
  have h4 : female_bars = 2 := by sorry  -- The next two bars represent females
  rw [h2, h3, h4, h1]
  norm_num

end number_of_females_l627_627806


namespace find_minor_premise_l627_627836

def rect_is_parallelogram : Prop := ∀ (r : Type), r = "rectangle" → r = "parallelogram"
def square_is_rectangle : Prop := ∀ (s : Type), s = "square" → s = "rectangle"
def square_is_parallelogram : Prop := ∀ (s : Type), s = "square" → s = "parallelogram"

def minor_premise (s1 s2 s3 : Prop) : Prop := s2

theorem find_minor_premise : minor_premise square_is_rectangle rect_is_parallelogram square_is_parallelogram = square_is_rectangle :=
by sorry

end find_minor_premise_l627_627836


namespace couriers_meet_l627_627372

-- Define the sequence and sum function for first courier
def first_courier_distance (x : ℕ) : ℝ :=
  (x * (79 + x)) / 8
  
-- Define the sequence and sum function for second courier
def second_courier_distance (x : ℕ) : ℝ :=
  ((17 + x) * (x - 3)) / 3
  
-- Define the meeting point condition
theorem couriers_meet : ∃ x : ℕ, x = 6 ∨ x = 19 :=
  by
    -- Define the meeting point equation
    have h1 : ∀ x : ℕ, first_courier_distance x = second_courier_distance x + 40 := sorry
    -- Solving the equation yields x = 6 or x = 19
    use 6
    use 19
    split
    . left
      sorry
    . right
      sorry

end couriers_meet_l627_627372


namespace number_added_multiplied_l627_627366

theorem number_added_multiplied (x : ℕ) (h : (7/8 : ℚ) * x = 28) : ((x + 16) * (5/16 : ℚ)) = 15 :=
by
  sorry

end number_added_multiplied_l627_627366


namespace length_of_train_l627_627677

noncomputable def train_length : ℕ := 1200

theorem length_of_train 
  (L : ℝ) 
  (speed_km_per_hr : ℝ) 
  (time_min : ℕ) 
  (speed_m_per_s : ℝ) 
  (time_sec : ℕ) 
  (distance : ℝ) 
  (cond1 : L = L)
  (cond2 : speed_km_per_hr = 144) 
  (cond3 : time_min = 1)
  (cond4 : speed_m_per_s = speed_km_per_hr * 1000 / 3600)
  (cond5 : time_sec = time_min * 60)
  (cond6 : distance = speed_m_per_s * time_sec)
  (cond7 : 2 * L = distance)
  : L = train_length := 
sorry

end length_of_train_l627_627677


namespace repaved_today_l627_627773

theorem repaved_today (total before : ℕ) (h_total : total = 4938) (h_before : before = 4133) : total - before = 805 := by
  sorry

end repaved_today_l627_627773


namespace tourist_purchased_total_checks_l627_627393

-- Definitions based on the conditions
variables (F H R : ℕ)
variable total_worth : ℕ := 1800
variable spent_fifty_checks : ℕ := 18
noncomputable def remaining_worth := total_worth - (spent_fifty_checks * 50)
noncomputable def average_remaining := 75

-- Theorem statement
theorem tourist_purchased_total_checks (h1 : 50 * F + 100 * H = total_worth)
                                       (h2 : average_remaining * R = remaining_worth) :
  F + H = 30 :=
begin
  sorry
end

end tourist_purchased_total_checks_l627_627393


namespace find_second_discount_l627_627755

noncomputable def list_price : ℝ := 67
noncomputable def final_price : ℝ := 56.16
noncomputable def first_discount : ℝ := 0.1
noncomputable def price_after_first_discount : ℝ := list_price * (1 - first_discount)
noncomputable def second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100

theorem find_second_discount : second_discount ≈ 6.86 := by
  sorry

end find_second_discount_l627_627755


namespace minimum_cubes_l627_627791

def front_view := [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)]
def side_view := [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)]

def cube (x y z : ℤ) := (x, y, z)

def figure := [
  cube 0 0 0, cube 0 1 0, cube 0 2 0,
  cube 1 0 0, cube 2 0 0]

theorem minimum_cubes : ∃ figure, (∀ (x y : ℤ), (x, y) ∈ front_view → ∃ z, cube x y z ∈ figure) ∧ 
  (∀ (y z : ℤ), (y, z) ∈ side_view → ∃ x, cube x y z ∈ figure) ∧
  (∀ c1 c2 ∈ figure, ∃ (d : ℤ × ℤ × ℤ), c1 = (d.1, d.2, d.3) ∧ c2 = (d.1 + 1, d.2, d.3) ∨
    c1 = (d.1, d.2, d.3) ∧ c2 = (d.1 - 1, d.2, d.3) ∨
    c1 = (d.1, d.2, d.3) ∧ c2 = (d.1, d.2 + 1, d.3) ∨
    c1 = (d.1, d.2, d.3) ∧ c2 = (d.1, d.2 - 1, d.3) ∨
    c1 = (d.1, d.2, d.3) ∧ c2 = (d.1, d.2, d.3 + 1) ∨
    c1 = (d.1, d.2, d.3) ∧ c2 = (d.1, d.2, d.3 - 1)) ∧
  figure.card = 5 := 
begin
  sorry
end

end minimum_cubes_l627_627791


namespace count_non_factorial_tails_lt_3000_l627_627832

def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125

theorem count_non_factorial_tails_lt_3000 : 
  let total_count := 2999 in 
  let factorial_tail_count := 2405 in -- inferred from the math solution
  total_count - factorial_tail_count = 594 :=
by
  sorry

end count_non_factorial_tails_lt_3000_l627_627832


namespace triangle_side_split_l627_627687

theorem triangle_side_split
  (PQ QR PR : ℝ)  -- Triangle sides
  (PS SR : ℝ)     -- Segments of PR divided by angle bisector
  (h_ratio : PQ / QR = 3 / 4)
  (h_sum : PR = 15)
  (h_PS_SR : PS / SR = 3 / 4)
  (h_PR_split : PS + SR = PR) :
  SR = 60 / 7 :=
by
  sorry

end triangle_side_split_l627_627687


namespace remaining_task_orders_l627_627205

theorem remaining_task_orders :
  (∑ k in finset.range 9, (nat.choose 8 k) * (k + 2)) = 1440 :=
by
  sorry

end remaining_task_orders_l627_627205


namespace sequence_properties_l627_627451

def seq (n : ℕ) : ℕ → ℝ := λ i, 2 * i

def G (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in range n, (2 : ℝ) ^ i * a (i + 1)) / n

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, a (i + 1)

theorem sequence_properties:
  let a := seq n in
  (∀ n, G a n = 2^n) →
  (a (n+1) = n+1) ∧
  (S a 2023 / 2023 = 1013) ∧
  let b := λ n, (9 / 10) ^ n * a (n + 1) in
  ∃ n, ∀ m, b n ≥ b (n + m) := sorry

end sequence_properties_l627_627451


namespace marina_cannot_prevent_katya_win_l627_627317

/-- Given 8 white cubes, each with 6 faces, we are to paint 24 faces blue and 24 faces red.
    When these cubes are assembled into a 2x2x2 larger cube, Katya wins if the surface has equal
    numbers of blue and red faces, otherwise, Marina wins. We need to prove that Marina cannot 
    paint the cubes in such a way that Katya cannot achieve her goal. -/
theorem marina_cannot_prevent_katya_win :
  let total_faces := 8 * 6,
      painted_blue := 24,
      painted_red := 24,
      surface_faces := 24 in
  (painted_blue + painted_red = total_faces) ∧ 
  (surface_faces = 6 * 4) →
  ∃ a : ℕ, (a = surface_faces / 2 ∧ surface_faces - a = surface_faces / 2) :=
by
  sorry

end marina_cannot_prevent_katya_win_l627_627317


namespace cards_of_D_l627_627876

variable {A B C D : ℤ}

-- Given Conditions
def condition_A := A = C + 16
def condition_B := D = C + 6
def condition_C := A = D + 9
def condition_D := D + 2 = 3 * C
def condition_fewest_is_lying (a b c d : ℤ) := min a (min b (min c d)) = c

theorem cards_of_D
  (hA : condition_A)
  (hB : condition_B)
  (hC : ¬condition_C)
  (hD : condition_D)
  (hFewest : condition_fewest_is_lying A B C D) : D = 10 :=
sorry

end cards_of_D_l627_627876
