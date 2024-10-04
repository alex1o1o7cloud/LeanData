import Mathlib

namespace lucky_numbers_count_l554_554656

def isLucky (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≠ 0 ∧ 221 = n * k + (221 % n) ∧ (221 % n) % k = 0

def countLuckyNumbers : ℕ :=
  Nat.card (Finset.filter isLucky (Finset.range 222))

theorem lucky_numbers_count : countLuckyNumbers = 115 :=
  sorry

end lucky_numbers_count_l554_554656


namespace correct_statements_proof_l554_554253

theorem correct_statements_proof :
  (∀ (a b : ℤ), a - 3 = b - 3 → a = b) ∧
  ¬ (∀ (a b c : ℤ), a = b → a + c = b - c) ∧
  (∀ (a b m : ℤ), m ≠ 0 → (a / m) = (b / m) → a = b) ∧
  ¬ (∀ (a : ℤ), a^2 = 2 * a → a = 2) :=
by
  -- Here we would prove the statements individually:
  -- sorry is a placeholder suggesting that the proofs need to be filled in.
  sorry

end correct_statements_proof_l554_554253


namespace standard_equation_of_ellipse_max_S_ΔAOB_l554_554789

-- Define the initial conditions
def a : ℝ := sqrt 3
def b : ℝ := 1
def e : ℝ := sqrt 6 / 3
def point_on_ellipse : Prop := 
  let x := 2 * sqrt 2 / 3 in
  let y := sqrt 3 / 3 in
  y^2 / a^2 + x^2 / b^2 = 1

-- Problem statements

-- Part 1: Find the standard equation of the ellipse
theorem standard_equation_of_ellipse (a b : ℝ) (point_on_ellipse : Prop) :
  e = sqrt 6 / 3 → a^2 = 3 → b^2 = 1 → (∀ x y, point_on_ellipse → y^2 / a^2 + x^2 / b^2 = 1) :=
by
  sorry

-- Part 2: Find the maximum value of S_ΔAOB
theorem max_S_ΔAOB (N : ℝ × ℝ) (A B : ℝ × ℝ) :
  N = (2, 0) → e = sqrt 6 / 3 → a^2 = 3 → b^2 = 1 → 
  (S_ΔAOB) = (sqrt 3 / 2) :=
by 
  sorry

end standard_equation_of_ellipse_max_S_ΔAOB_l554_554789


namespace jasmine_dinner_time_l554_554878

-- Define the conditions as variables and constants.
constant work_end_time : ℕ := 16  -- Representing 4:00 pm in 24-hour format (16:00)
constant commute_time : ℕ := 30   -- in minutes
constant grocery_time : ℕ := 30   -- in minutes
constant dry_cleaning_time : ℕ := 10  -- in minutes
constant dog_grooming_time : ℕ := 20  -- in minutes
constant cooking_time : ℕ := 90   -- in minutes

-- Define a function to sum up the times
def total_time_after_work : ℕ := commute_time + grocery_time + dry_cleaning_time + dog_grooming_time + cooking_time

def time_to_hour_minutes (total_minutes : ℕ) : (ℕ × ℕ) := 
  (total_minutes / 60, total_minutes % 60)

-- Prove that Jasmine will eat dinner at 7:00 pm (19:00 in 24-hour format)
theorem jasmine_dinner_time : total_time_after_work / 60 + work_end_time = 19 := by
  -- Leave the proof part as sorry since we don't need to provide proof steps
  sorry

end jasmine_dinner_time_l554_554878


namespace solution_set_l554_554216

theorem solution_set (x : ℝ) : (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l554_554216


namespace g_g_x_eq_2_count_l554_554200

-- Define the function g(x)
def g (x : ℝ) : ℝ :=
  if x = -2 then 2 else if x = 0 then 2 else if x = 4 then 2 else if x = -1 then 0 else if x = 3 then 4 else 0

-- The main theorem stating the number of x values satisfying g(g(x)) = 2 is 2
theorem g_g_x_eq_2_count : (finite { x : ℝ | g (g x) = 2 }).toFinset.card = 2 :=
by
  sorry

end g_g_x_eq_2_count_l554_554200


namespace integer_part_of_decimal_shift_l554_554913

theorem integer_part_of_decimal_shift :
  let a := 141 in
  (a - 1.41 : ℝ).floor = 139 :=
by
  sorry

end integer_part_of_decimal_shift_l554_554913


namespace practice_time_for_second_recital_l554_554324

-- Definitions based on given conditions
def first_recital_hours : ℝ := 5
def first_recital_score : ℝ := 60
def average_score : ℝ := 75
def second_recital_score : ℝ := 90   -- Derived from solution step 1
def total_time_constant : ℝ := first_recital_hours * first_recital_score

-- Proof statement
theorem practice_time_for_second_recital (h : ℝ) : 
    (total_time_constant = second_recital_score * h) → (h ≈ 10/3) :=
by
  sorry

end practice_time_for_second_recital_l554_554324


namespace intersection_M_N_l554_554815

open Set

noncomputable def M : Set ℝ := {x | x ≥ 2}

noncomputable def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} :=
by
  sorry

end intersection_M_N_l554_554815


namespace sum_of_solutions_l554_554518

def g (x : ℝ) : ℝ := 3 * x - 2

noncomputable def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem sum_of_solutions : 
  (∀ x : ℝ, g_inv x = g (x⁻¹) → x = 1 ∨ x = -9) → 
  ∑ x in {-9, 1}, x = -8 :=
by {
  intros h,
  have h1 : true := sorry,  -- a place holder as the exact implementation of the proofs are omitted.
  sorry
}

end sum_of_solutions_l554_554518


namespace compute_expression_l554_554732

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l554_554732


namespace man_wage_l554_554279

variable (m w b : ℝ) -- wages of man, woman, boy respectively
variable (W : ℝ) -- number of women equivalent to 5 men and 8 boys

-- Conditions given in the problem
axiom condition1 : 5 * m = W * w
axiom condition2 : W * w = 8 * b
axiom condition3 : 5 * m + 8 * b + 8 * b = 90

-- Prove the wage of one man
theorem man_wage : m = 6 := 
by
  -- proof steps would be here, but skipped as per instructions
  sorry

end man_wage_l554_554279


namespace kona_distance_proof_l554_554774

-- Defining the distances as constants
def distance_to_bakery : ℕ := 9
def distance_from_grandmother_to_home : ℕ := 27
def additional_trip_distance : ℕ := 6

-- Defining the variable for the distance from bakery to grandmother's house
def x : ℕ := 30

-- Main theorem to prove the distance
theorem kona_distance_proof :
  distance_to_bakery + x + distance_from_grandmother_to_home = 2 * x + additional_trip_distance :=
by
  sorry

end kona_distance_proof_l554_554774


namespace least_cost_grass_seed_l554_554258

variable (cost_5_pound_bag : ℕ) [Fact (cost_5_pound_bag = 1380)]
variable (cost_10_pound_bag : ℕ) [Fact (cost_10_pound_bag = 2043)]
variable (cost_25_pound_bag : ℕ) [Fact (cost_25_pound_bag = 3225)]
variable (min_weight : ℕ) [Fact (min_weight = 65)]
variable (max_weight : ℕ) [Fact (max_weight = 80)]

theorem least_cost_grass_seed :
  ∃ (n5 n10 n25 : ℕ),
    n5 * 5 + n10 * 10 + n25 * 25 ≥ min_weight ∧
    n5 * 5 + n10 * 10 + n25 * 25 ≤ max_weight ∧
    n5 * cost_5_pound_bag + n10 * cost_10_pound_bag + n25 * cost_25_pound_bag = 9675 :=
  sorry

end least_cost_grass_seed_l554_554258


namespace log_eq_neg_two_l554_554747

theorem log_eq_neg_two : ∀ (x : ℝ), (1 / 5) ^ x = 25 → x = -2 :=
by
  intros x h
  sorry

end log_eq_neg_two_l554_554747


namespace circleEquation_and_pointOnCircle_l554_554484

-- Definition of the Cartesian coordinate system and the circle conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def inSecondQuadrant (p : ℝ × ℝ) := p.1 < 0 ∧ p.2 > 0

def tangentToLine (C : Circle) (line : ℝ → ℝ) (tangentPoint : ℝ × ℝ) :=
  let centerToLineDistance := (abs (C.center.1 - C.center.2)) / Real.sqrt 2
  C.radius = centerToLineDistance ∧ tangentPoint = (0, 0)

-- Main statements to prove
theorem circleEquation_and_pointOnCircle :
  ∃ C : Circle, ∃ Q : ℝ × ℝ,
    inSecondQuadrant C.center ∧
    C.radius = 2 * Real.sqrt 2 ∧
    tangentToLine C (fun x => x) (0, 0) ∧
    ((∃ p : ℝ × ℝ, p = (-2, 2) ∧ C = Circle.mk p (2 * Real.sqrt 2) ∧
      (∀ x y : ℝ, ((x + 2)^2 + (y - 2)^2 = 8))) ∧
    (Q = (4/5, 12/5) ∧
      ((Q.1 + 2)^2 + (Q.2 - 2)^2 = 8) ∧
      Real.sqrt ((Q.1 - 4)^2 + Q.2^2) = 4))
    := sorry

end circleEquation_and_pointOnCircle_l554_554484


namespace four_digit_number_count_l554_554449

noncomputable def digit_avg_count : ℕ :=
  let digits := {d : ℕ | d ≥ 0 ∧ d ≤ 9}
  let four_digit_numbers :=
    {(A, B, C, D) :
      ∃ A B C D,
        A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits ∧
        B = (A + C) / 2 ∧ C = (B + D) / 2 ∧
        A ≠ 0} -- A must be non-zero to ensure it's a four-digit number
  in four_digit_numbers.card

theorem four_digit_number_count : digit_avg_count = 225 :=
  sorry

end four_digit_number_count_l554_554449


namespace Rachel_total_books_l554_554554

theorem Rachel_total_books :
  (8 * 15) + (4 * 15) + (3 * 15) + (5 * 15) = 300 :=
by {
  sorry
}

end Rachel_total_books_l554_554554


namespace ellipse_equation_lamda_range_l554_554808

-- Conditions
variables (a b x y e : ℝ)

-- Definitions based on conditions
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def eccentricity (a b e : ℝ) : Prop := e = sqrt (1 - (b / a)^2)
def line (x y : ℝ) : Prop := y = 2 * x - 1
def distanceAB (A B : ℝ × ℝ) : ℝ := real.dist A B
def intersect_points (C : Prop) (line : Prop) : Prop := true -- Formalize as needed

-- Given data
axiom Ha_gt_b : a > b
axiom Hb_gt_0 : b > 0
axiom H_e : e = sqrt(2) / 2

noncomputable def C_eq : Prop := ∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = sqrt(2) / 2 ∧ ellipse a b x y ∧ 
line x y ∧ distanceAB (x₁, y₁) (x₂, y₂) = 8 / 9 * sqrt 5

-- Lean statements for the problems
theorem ellipse_equation : C_eq a b x y e → ellipse a b x y := sorry

-- For question 2: Find the range of lambda
variables (M E F : ℝ × ℝ)
variable (lambda : ℝ)
def triangle_area (A B C : ℝ × ℝ) : ℝ := (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def lambda_def : Prop := lambda = triangle_area (0, 0) M E / triangle_area (0, 0) M F

theorem lamda_range : lambda_def lambda → 0 < lambda ∧ lambda ≠ 1 ∧ lambda < 3 + 2 * sqrt 2 := sorry

end ellipse_equation_lamda_range_l554_554808


namespace probability_contemporaries_l554_554239

theorem probability_contemporaries (total_years : ℕ) (life_span : ℕ) (born_range : ℕ)
  (h1 : total_years = 300)
  (h2 : life_span = 80)
  (h3 : born_range = 300) :
  (∃λ p : ℚ, p = 104 / 225 ∧ 
   let lines_intersect := (0, 80) :: (80, 0) :: (220, 300) :: (300, 220) :: []
   in lines_intersect ≠ [] ∧ lines_intersect.length = 4 
   ∧ region_area 0 300 300 (λ x y, (y ≥ x - life_span) ∧ (y ≤ x + life_span)) = 41600
   ∧ total_area 0 300 300 = 90000
   ∧ prob := region_area / total_area,
     prob = p) :=
by sorry

end probability_contemporaries_l554_554239


namespace complex_number_property_exists_l554_554166

theorem complex_number_property_exists :
  ∃ (c : ℂ) (d : ℝ), c ≠ 0 ∧ (∀ (z : ℂ), abs z = 1 → 1 + z + z^2 ≠ 0 → abs ((1 / (1 + z + z^2)) - c) = d) :=
sorry

end complex_number_property_exists_l554_554166


namespace lucky_numbers_l554_554687

def is_lucky (n : ℕ) : Prop :=
  ∃ q r : ℕ, 221 = n * q + r ∧ r % q = 0

def lucky_numbers_count (a b : ℕ) : ℕ :=
  Nat.card {n // a ≤ n ∧ n ≤ b ∧ is_lucky n}

theorem lucky_numbers (a b : ℕ) (h1 : a = 1) (h2 : b = 221) :
  lucky_numbers_count a b = 115 := 
sorry

end lucky_numbers_l554_554687


namespace root_relationship_l554_554254

theorem root_relationship (a x₁ x₂ : ℝ) 
  (h_eqn : x₁^2 - (2*a + 1)*x₁ + a^2 + 2 = 0)
  (h_roots : x₂ = 2*x₁)
  (h_vieta1 : x₁ + x₂ = 2*a + 1)
  (h_vieta2 : x₁ * x₂ = a^2 + 2) : 
  a = 4 := 
sorry

end root_relationship_l554_554254


namespace proof_1_proof_2_l554_554032

variable {a b k : ℝ}

/-- Given conditions: f(x) = ax^2 + bx + 1, a > 0,
    f(-1) = 0 and for any real number x, f(x) ≥ 0 -/
def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- The derived function and conditions
def F (x : ℝ) : ℝ := x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := f x - k * x

theorem proof_1 (h1 : a > 0) (h2 : f (-1) = 0) (h3 : ∀ x : ℝ, f x ≥ 0) :
    (∀ x : ℝ, f x = F x) :=
by
    sorry

theorem proof_2 (h1 : a > 0) (h2 : f (-1) = 0) (h3 : ∀ x : ℝ, f x ≥ 0) :
    (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), (∀ y z ∈ set.Icc (-2 : ℝ) (2 : ℝ), g y = g z → y = z) →
    (k ≤ -2 ∨ k ≥ 6)) :=
by
    sorry

end proof_1_proof_2_l554_554032


namespace math_proof_problem_l554_554632

variable (a b c : ℝ)

def condition1 := (10 = 0.06 * a)
def condition2 := (6 = 0.10 * b)
def definition_of_c := (c = b / a)
def value_of_c := (c = 0.36)

theorem math_proof_problem (h1 : condition1) (h2 : condition2) (h3 : definition_of_c) : value_of_c :=
by 
  -- proof using h1, h2 and h3 would go here
  -- omitted for brevity
  sorry

end math_proof_problem_l554_554632


namespace Tom_completes_wall_l554_554875

theorem Tom_completes_wall :
  let avery_rate_per_hour := (1:ℝ)/3
  let tom_rate_per_hour := (1:ℝ)/2
  let combined_rate_per_hour := avery_rate_per_hour + tom_rate_per_hour
  let portion_completed_together := combined_rate_per_hour * 1 
  let remaining_wall := 1 - portion_completed_together
  let time_for_tom := remaining_wall / tom_rate_per_hour
  time_for_tom = (1:ℝ)/3 := 
by 
  sorry

end Tom_completes_wall_l554_554875


namespace simplest_sqrt_l554_554972

noncomputable def sqrt_simplest (x : ℝ) : ℝ :=
  if x = 0.1 then sqrt (1/10)
  else if x = 8 then 2 * sqrt 2
  else if x = 1/2 then sqrt 2 / 2
  else if x = 3 then sqrt 3
  else x

theorem simplest_sqrt (x : ℝ) : sqrt_simplest 3 = sqrt 3 :=
by
  show sqrt_simplest 3 = sqrt 3
    sorry

end simplest_sqrt_l554_554972


namespace countLuckyNumbers_l554_554667

def isLuckyNumber (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧
  (let q := 221 / n in (221 % n) % q = 0)

theorem countLuckyNumbers : 
  { n : ℕ | isLuckyNumber n }.toFinset.card = 115 :=
by
  sorry

end countLuckyNumbers_l554_554667


namespace binomial_symmetry_binomial_1512_eq_2730_l554_554351

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_symmetry (n k : ℕ) : binomial n k = binomial n (n - k) := nat.choose_symm

theorem binomial_1512_eq_2730 : 3 * binomial 15 12 = 2730 := by
  sorry

end binomial_symmetry_binomial_1512_eq_2730_l554_554351


namespace students_remaining_after_fourth_stop_l554_554071

theorem students_remaining_after_fourth_stop : 
  ∀ (initial number of students : ℕ) (get_off_fraction : ℚ) (stops : ℕ),
    initial = 60 →
    get_off_fraction = 1 / 3 →
    stops = 4 →
    let remaining_fraction := 1 - get_off_fraction in
    let final_students := initial * (remaining_fraction ^ stops) in
  final_students = 11 :=
begin
  intros initial_students get_off_fraction stops h1 h2 h3,
  let remaining_fraction := 1 - get_off_fraction,
  let final_students := initial_students * (remaining_fraction ^ stops),
  have h_initial : initial_students = 60, by exact h1,
  have h_get_off : get_off_fraction = 1 / 3, by exact h2,
  have h_stops : stops = 4, by exact h3,
  calc
  final_students = 60 * (2 / 3) ^ 4 : by { rw [h_initial, h_get_off, h_stops], sorry }
               ... = 11 : sorry
end

end students_remaining_after_fourth_stop_l554_554071


namespace find_b_l554_554837

-- Definition: Factor Condition
constant cx3_bx2_1 : (c : ℤ) → (b : ℤ) → ℤ[X]
def polynomial_factor (x : ℤ) : Prop :=
  ∃ (c b : ℤ), (λ x, x^2 - 2*x - 1) ∣ (λ x, c*x^3 + b*x^2 + 1)

-- Theorem: Value of b
theorem find_b (c : ℤ) (b : ℤ) (h1 : polynomial_factor c b) : b = 3 :=
sorry

end find_b_l554_554837


namespace cross_section_area_ratio_squared_l554_554096

theorem cross_section_area_ratio_squared (s : ℝ) (h : s > 0) :
  let A := (0, 0, 0)
  let B := (s, 0, 0)
  let E := (0, s, s)
  let F := (s, s, s)
  let K := (s/2, 0, 0)
  let L := (γ, γ, s) -- L would be F in the ideal position as midpoint of EF
  in (K = (s/2, 0, 0)) ∧ (L = F) ∧ (R := (area_of_triangle (A, K, L)) / (s^2)) ∧ (R^2 = 1/8) := sorry

-- Provide necessary definitions for area_of_triangle
def area_of_triangle (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ := sorry


end cross_section_area_ratio_squared_l554_554096


namespace sum_of_remaining_two_scores_l554_554927

open Nat

theorem sum_of_remaining_two_scores :
  ∃ x y : ℕ, x + y = 160 ∧ (65 + 75 + 85 + 95 + x + y) / 6 = 80 :=
by
  sorry

end sum_of_remaining_two_scores_l554_554927


namespace Janet_has_fewer_siblings_l554_554495

-- Defining the number of siblings for each person
def Masud_siblings : ℕ := 45

def Janet_siblings : ℕ := 4 * Masud_siblings - 60 -- Condition 2, Janet's siblings
def Carlos_siblings_initial : ℚ := (3/4) * Masud_siblings -- Condition 3, Initial value of Carlos's siblings
def Stella_siblings_initial := (5/2) * Carlos_siblings_initial - 8 -- Condition 4, Initial Stella siblings

def Stella_siblings : ℕ := 77
def Carlos_siblings : ℕ := Stella_siblings + 20 -- Condition 6, Correct Carlos's siblings
def Lila_siblings : ℚ := (Carlos_siblings + Stella_siblings) + (1/3) * (Carlos_siblings + Stella_siblings) -- Condition 5, Lila's siblings
def Total_siblings_combined : ℚ := Carlos_siblings + Stella_siblings + Lila_siblings

-- Proving that Janet has fewer siblings than the total number combined by 286
theorem Janet_has_fewer_siblings : Janet_siblings - Total_siblings_combined = -286 :=
by
  have M := 45
  have J := 4 * M - 60
  have C := 97 -- Corrected Carlos's siblings
  have S := 77 -- Corrected Stella's siblings
  have L := (C + S) + (1/3) * (C + S)
  have T := C + S + L
  show J - T = -286 from sorry

end Janet_has_fewer_siblings_l554_554495


namespace trig_inequality_l554_554889

def sin_deg (deg: ℝ) : ℝ := Real.sin (deg * Real.pi / 180)
def cos_deg (deg: ℝ) : ℝ := Real.cos (deg * Real.pi / 180)
def tan_deg (deg: ℝ) : ℝ := Real.tan (deg * Real.pi / 180)

theorem trig_inequality :
  let a := sin_deg 46,
      b := cos_deg 46,
      c := tan_deg 46
  in c > a ∧ a > b :=
by
  sorry

end trig_inequality_l554_554889


namespace largest_int_starting_with_8_l554_554380

theorem largest_int_starting_with_8 (n : ℕ) : 
  (n / 100 = 8) ∧ (n >= 800) ∧ (n < 900) ∧ ∀ (d : ℕ), (d ∣ n ∧ d ≠ 0 ∧ d ≠ 7) → d ∣ 864 → (n ≤ 864) :=
sorry

end largest_int_starting_with_8_l554_554380


namespace polynomial_divisible_iff_l554_554389

theorem polynomial_divisible_iff (a b : ℚ) : 
  ((a + b) * 1^5 + (a * b) * 1^2 + 1 = 0) ∧ 
  ((a + b) * 2^5 + (a * b) * 2^2 + 1 = 0) ↔ 
  (a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1) := 
by 
  sorry

end polynomial_divisible_iff_l554_554389


namespace arithmetic_sequence_bn_general_formula_an_sum_of_first_n_l554_554873

noncomputable def a : (ℕ → ℕ)
| 0     := 1
| (n+1) := 2 * a n + 2^n

noncomputable def b (n : ℕ) : ℕ := a n / 2^(n-1)

theorem arithmetic_sequence_bn (n : ℕ) : b (n+1) - b n = 1 := 
sorry

theorem general_formula_an (n : ℕ) : a n = n * 2^(n-1) :=
sorry

theorem sum_of_first_n (n : ℕ) : 
  let S : ℕ → ℕ := λ n, (Finset.range n).sum (λ i, a i)
  S n = (n-1) * 2^n + 1 :=
sorry

end arithmetic_sequence_bn_general_formula_an_sum_of_first_n_l554_554873


namespace total_suitcases_correct_l554_554536

-- Conditions as definitions
def num_siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def num_parents : Nat := 2
def suitcases_per_parent : Nat := 3

-- Total suitcases calculation
def total_suitcases :=
  (num_siblings * suitcases_per_sibling) + (num_parents * suitcases_per_parent)

-- Statement to prove
theorem total_suitcases_correct : total_suitcases = 14 :=
by
  sorry

end total_suitcases_correct_l554_554536


namespace volume_of_one_pizza_slice_l554_554305

-- Condition definitions
def pizza_thickness := (1 : ℝ) / 4
def pizza_diameter := 16
def pizza_radius := pizza_diameter / 2
def full_pizza_volume := Real.pi * pizza_radius^2 * pizza_thickness
def number_of_slices := 8

-- Problem statement
theorem volume_of_one_pizza_slice :=
  let volume_per_slice := full_pizza_volume / number_of_slices
  volume_per_slice = 2 * Real.pi :=
by
  -- By proving the equivalence
  sorry

end volume_of_one_pizza_slice_l554_554305


namespace market_value_of_stock_l554_554998

noncomputable def market_value_stock (D : ℝ) (Y : ℝ) (F : ℝ) : ℝ :=
  (D / Y) * F

theorem market_value_of_stock : 
  let D := 0.07 * 100 in
  let Y := 0.10 in
  let F := 100 in
  market_value_stock D Y F = 70 :=
by
  sorry

end market_value_of_stock_l554_554998


namespace find_w_l554_554181

def first_polynomial_roots : Type := { p q r : ℝ // 
  p + q + r = 5 ∧ 
  p * q + q * r + r * p = 9 ∧ 
  p * q * r = 7 }

def w_value (roots : first_polynomial_roots) : ℝ :=
  let ⟨p, q, r, h₁, h₂, h₃⟩ := roots in
  -(5 - r) * (5 - p) * (5 - q)

theorem find_w : ∀ roots : first_polynomial_roots, w_value roots = -13 :=
  by
    intros
    cases roots with p q r
    sorry

end find_w_l554_554181


namespace centroid_of_equal_area_triangles_l554_554712

theorem centroid_of_equal_area_triangles 
  (A B C P : Point)
  (h_in_triangle : P ∈ triangle A B C)
  (h_equal_areas : area (triangle P A B) = area (triangle P B C) ∧
                   area (triangle P B C) = area (triangle P C A)) :
  P = centroid A B C := 
sorry

end centroid_of_equal_area_triangles_l554_554712


namespace length_of_yellow_line_l554_554298

theorem length_of_yellow_line
  (w1 w2 w3 w4 : ℝ) (path_width : ℝ) (middle_line_dist : ℝ)
  (h1 : w1 = 40) (h2 : w2 = 10) (h3 : w3 = 20) (h4 : w4 = 30) (h5 : path_width = 5) (h6 : middle_line_dist = 2.5) :
  w1 - path_width * middle_line_dist/2 + w2 + w3 + w4 - path_width * middle_line_dist/2 = 95 :=
by sorry

end length_of_yellow_line_l554_554298


namespace count_lucky_numbers_l554_554655

def is_lucky (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 221) ∧ (221 % n % (221 / n) = 0)

theorem count_lucky_numbers : (finset.filter is_lucky (finset.range 222)).card = 115 :=
by
  sorry

end count_lucky_numbers_l554_554655


namespace gus_eggs_l554_554821

theorem gus_eggs : 
  let eggs_breakfast := 2 in
  let eggs_lunch := 3 in
  let eggs_dinner := 1 in
  let total_eggs := eggs_breakfast + eggs_lunch + eggs_dinner in
  total_eggs = 6 :=
by
  sorry

end gus_eggs_l554_554821


namespace cryptarithm_solution_l554_554491

theorem cryptarithm_solution (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_adjacent : A = C + 1 ∨ A = C - 1)
  (h_diff : B = D + 2 ∨ B = D - 2) :
  1000 * A + 100 * B + 10 * C + D = 5240 :=
sorry

end cryptarithm_solution_l554_554491


namespace collinear_proj_l554_554909

noncomputable def a : ℝ × ℝ × ℝ := (1, -1, 2)
noncomputable def b : ℝ × ℝ × ℝ := (2, 3, 0)
noncomputable def p : ℝ × ℝ × ℝ := (4/3, 1/3, 4/3)

theorem collinear_proj (v : ℝ × ℝ × ℝ) :
  (a.1 + b.1 - 1) * v.1 + (-1 + 4 * v.2) * v.2 + (2 - 2 * v.3) * v.3 = 0 → 
  (∃ k : ℝ, p = (a.1 + k * (b.1 - a.1), a.2 + k * (b.2 - a.2), a.3 + k * (b.3 - a.3))) :=
sorry

end collinear_proj_l554_554909


namespace pages_printed_l554_554876

theorem pages_printed (P : ℕ) 
  (H1 : P % 7 = 0)
  (H2 : P % 3 = 0)
  (H3 : P - (P / 7 + P / 3 - P / 21) = 24) : 
  P = 42 :=
sorry

end pages_printed_l554_554876


namespace sandy_sums_attempted_l554_554170

theorem sandy_sums_attempted 
  (marks_correct : ℕ)
  (marks_incorrect : ℕ)
  (total_marks : ℤ)
  (correct_sums : ℕ)
  (incorrect_sums : ℕ) 
  (total_sums : ℕ) :  

  marks_correct = 3 →
  marks_incorrect = -2 →
  total_marks = 55 →
  correct_sums = 23 →
  total_marks = marks_correct * correct_sums + marks_incorrect * incorrect_sums →
  total_sums = correct_sums + incorrect_sums →
  total_sums = 30 :=
  
by 
  intros h_marks_correct h_marks_incorrect h_total_marks h_correct_sums h_eq_marks h_total_sums
  sorry

end sandy_sums_attempted_l554_554170


namespace cos_angle_sum_difference_l554_554410

variable (α : ℝ)

-- Given condition:
def given_condition : Prop := cos (2 * α) = 7 / 9

-- Statement to prove:
theorem cos_angle_sum_difference (h : given_condition α) :
  cos (π / 4 + α) * cos (π / 4 - α) = 7 / 18 :=
sorry

end cos_angle_sum_difference_l554_554410


namespace range_of_k_l554_554075

theorem range_of_k (k : ℝ) :
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l554_554075


namespace possible_value_of_x1_x2_l554_554426

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - 2 * cos x * cos x + 1
noncomputable def g (x : ℝ) : ℝ := 2 * sin (4 * x - π / 6) + 1

theorem possible_value_of_x1_x2 (x1 x2 : ℝ) (h : g x1 * g x2 = 9) :
  |x1 - x2| = π / 2 :=
sorry

end possible_value_of_x1_x2_l554_554426


namespace probability_distance_at_least_one_correct_l554_554896

noncomputable def probability_distance_at_least_one := (12 - 0 * Real.pi) / 4

theorem probability_distance_at_least_one_correct :
  let T := { side_length := 2, shape := "square" }
  ∃ (p q r : ℤ), p = 12 ∧ q = 0 ∧ r = 4 ∧ 
  Real.gcd p (Real.gcd q r) = 1 ∧
  probability_distance_at_least_one = (p - q * Real.pi) / r ∧
  p + q + r = 16 :=
by
  let p := 12
  let q := 0
  let r := 4
  use (p, q, r)
  split
  . rfl
  split
  . rfl
  split
  . rfl
  split
  . exact sorry
  split
  . exact sorry

end probability_distance_at_least_one_correct_l554_554896


namespace digit_appears_in_3n_l554_554201

-- Define a function to check if a digit is in a number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n / 10^k % 10 = d

-- Define the statement that n does not contain the digits 1, 2, or 9
def does_not_contain_1_2_9 (n : ℕ) : Prop :=
  ¬ (contains_digit n 1 ∨ contains_digit n 2 ∨ contains_digit n 9)

theorem digit_appears_in_3n (n : ℕ) (hn : 1 ≤ n) (h : does_not_contain_1_2_9 n) :
  contains_digit (3 * n) 1 ∨ contains_digit (3 * n) 2 ∨ contains_digit (3 * n) 9 :=
by
  sorry

end digit_appears_in_3n_l554_554201


namespace value_2_std_devs_less_than_mean_l554_554984

-- Define the arithmetic mean
def mean : ℝ := 15.5

-- Define the standard deviation
def standard_deviation : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value_2_std_less_than_mean : ℝ := mean - 2 * standard_deviation

-- The theorem we want to prove
theorem value_2_std_devs_less_than_mean : value_2_std_less_than_mean = 12.5 := by
  sorry

end value_2_std_devs_less_than_mean_l554_554984


namespace event_not_equally_likely_l554_554601

-- Definitions based on the conditions
def coin_outcomes : set (string × string) :=
  {("head", "head"), ("head", "tail"), ("tail", "head"), ("tail", "tail")}

def event_two_heads : set (string × string) :=
  {("head", "head")}

def event_one_head_one_tail : set (string × string) :=
  {("head", "tail"), ("tail", "head")}

def event_two_tails : set (string × string) :=
  {("tail", "tail")}

-- Equally Likelihood checking
def is_equally_likely (E F G : set (string × string)) : Prop :=
  (E.card = F.card) ∧ (F.card = G.card)

-- The proof statement
theorem event_not_equally_likely :
  ¬ is_equally_likely event_two_heads event_one_head_one_tail event_two_tails :=
by
  sorry

end event_not_equally_likely_l554_554601


namespace large_cross_area_is_60_cm_squared_l554_554213

noncomputable def small_square_area (s : ℝ) := s * s
noncomputable def large_square_area (s : ℝ) := 4 * small_square_area s
noncomputable def small_cross_area (s : ℝ) := 5 * small_square_area s
noncomputable def large_cross_area (s : ℝ) := 5 * large_square_area s
noncomputable def remaining_area (s : ℝ) := large_cross_area s - small_cross_area s

theorem large_cross_area_is_60_cm_squared :
  ∃ (s : ℝ), remaining_area s = 45 → large_cross_area s = 60 :=
by
  sorry

end large_cross_area_is_60_cm_squared_l554_554213


namespace probability_even_toys_l554_554623

theorem probability_even_toys:
  let total_toys := 21
  let even_toys := 10
  let probability_first_even := (even_toys : ℚ) / total_toys
  let probability_second_even := (even_toys - 1 : ℚ) / (total_toys - 1)
  let probability_both_even := probability_first_even * probability_second_even
  probability_both_even = 3 / 14 :=
by
  sorry

end probability_even_toys_l554_554623


namespace lori_marbles_l554_554910

theorem lori_marbles (friends marbles_per_friend : ℕ) (h_friends : friends = 5) (h_marbles_per_friend : marbles_per_friend = 6) : friends * marbles_per_friend = 30 := sorry

end lori_marbles_l554_554910


namespace hyperbola_foci_coordinates_l554_554192

theorem hyperbola_foci_coordinates :
  let a : ℝ := Real.sqrt 7
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  (c = Real.sqrt 10 ∧
  ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0))) :=
by
  let a := Real.sqrt 7
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  have hc : c = Real.sqrt 10 := sorry
  have h_foci : ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0)) := sorry
  exact ⟨hc, h_foci⟩

end hyperbola_foci_coordinates_l554_554192


namespace positive_difference_l554_554614

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end positive_difference_l554_554614


namespace tetrahedron_face_areas_less_sum_l554_554086

theorem tetrahedron_face_areas_less_sum 
  (A B C D E F G : Type) 
  [linear_ordered_field A] 
  [metric_space B] 
  (AB BC CA DE DF DG : A)
  (theta : B) -- the angle 
  (area_DAB area_DBC area_DCA : A)
  (h_midpoint_1 : E = midpoint AB)
  (h_midpoint_2 : F = midpoint BC)
  (h_midpoint_3 : G = midpoint CA)
  (h_equal_angles : ∀ (theta1 theta2 theta3 : B), theta1 = theta2 ∨ theta2 = theta3 ∨ theta3 = theta1 = true) -- angles formed are equal
  (area_DAB_eq : area_DAB = (1 / 2) * DE * AB * sin theta)
  (area_DBC_eq : area_DBC = (1 / 2) * DF * BC * sin theta)
  (area_DCA_eq : area_DCA = (1 / 2) * DG * CA * sin theta) :
  ∀ (x : A), 
  area_DAB < area_DBC + area_DCA :=
by 
  sorry

end tetrahedron_face_areas_less_sum_l554_554086


namespace sum_of_x_eq_240_l554_554766

theorem sum_of_x_eq_240 
    (x : ℝ) 
    (hx1 : 80 < x) 
    (hx2 : x < 160) 
    (hx3 : (Real.cos (2*x))^3 + (Real.cos (6*x))^3 = 8 * (Real.cos (4*x) * Real.cos x)^3) : 
    ∑ (x_val : ℝ) in ({x | 80 < x ∧ x < 160 ∧ (Real.cos (2*x))^3 + (Real.cos (6*x))^3 = 8 * (Real.cos (4*x) * Real.cos x)^3}.to_finset), x_val = 240 := 
  sorry

end sum_of_x_eq_240_l554_554766


namespace count_pairs_l554_554576

theorem count_pairs (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2012)
  (cond : ∀ n : ℕ, 5^n < 2^m ∧ 2^m < 2^(m + 2) ∧ 2^(m + 2) < 5^(n + 1)) :
  ∃ k : ℕ, k = 279 :=
begin
  sorry
end

end count_pairs_l554_554576


namespace largest_square_area_l554_554105

theorem largest_square_area (a b c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (square_area_sum : a^2 + b^2 + c^2 = 450)
  (area_a : a^2 = 100) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l554_554105


namespace x_finishes_remaining_work_in_7_days_l554_554627

-- Definitions based on the conditions.
def x_work_rate := 1 / 21
def y_work_rate := 1 / 15
def y_days_worked := 10

-- These definitions are derived from the conditions, but are necessary steps
-- for formulating the remaining work.
def work_completed_by_y := y_days_worked * y_work_rate
def work_remaining := 1 - work_completed_by_y

-- Question/Proof: x alone needs 7 days to finish the remaining work
theorem x_finishes_remaining_work_in_7_days :
  let remaining_work := work_remaining in
  let x_work_days := remaining_work / x_work_rate in
  x_work_days = 7 := by
  sorry

end x_finishes_remaining_work_in_7_days_l554_554627


namespace probability_convex_quadrilateral_l554_554405

theorem probability_convex_quadrilateral (n : ℕ) (h : n = 6) :
  let total_chords := Nat.choose n 2,
      ways_to_choose_four_chords := Nat.choose total_chords 4,
      ways_to_choose_four_points := Nat.choose n 4,
      probability := ways_to_choose_four_points / ways_to_choose_four_chords
  in probability = 1 / 91 := sorry

end probability_convex_quadrilateral_l554_554405


namespace greatest_radius_l554_554841

theorem greatest_radius (r : ℤ) (h : π * r^2 < 100 * π) : r < 10 :=
sorry

example : ∃ r : ℤ, π * r^2 < 100 * π ∧ ∀ r' : ℤ, (π * r'^2 < 100 * π) → r' ≤ r :=
begin
  use 9,
  split,
  { linarith },
  { intros r' h',
    have hr' : r' < 10,
    { linarith },
    exact int.lt_of_le_of_lt (int.le_of_lt_add_one hr') (by linarith) }
end

end greatest_radius_l554_554841


namespace simplify_sqrt7_to_the_six_l554_554924

theorem simplify_sqrt7_to_the_six : (sqrt 7)^6 = 343 :=
by 
  sorry

end simplify_sqrt7_to_the_six_l554_554924


namespace cone_to_sphere_surface_area_ratio_l554_554308

noncomputable def sphere_radius (r : ℝ) := r
noncomputable def cone_height (r : ℝ) := 3 * r
noncomputable def side_length_of_triangle (r : ℝ) := 2 * Real.sqrt 3 * r
noncomputable def surface_area_of_sphere (r : ℝ) := 4 * Real.pi * r^2
noncomputable def surface_area_of_cone (r : ℝ) := 9 * Real.pi * r^2
noncomputable def ratio_of_areas (cone_surface : ℝ) (sphere_surface : ℝ) := cone_surface / sphere_surface

theorem cone_to_sphere_surface_area_ratio (r : ℝ) :
    ratio_of_areas (surface_area_of_cone r) (surface_area_of_sphere r) = 9 / 4 := sorry

end cone_to_sphere_surface_area_ratio_l554_554308


namespace quadratic_expression_evaluation_l554_554406

theorem quadratic_expression_evaluation (x y : ℝ) (h1 : 3 * x + y = 10) (h2 : x + 3 * y = 14) :
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 :=
by
  -- Proof goes here
  sorry

end quadratic_expression_evaluation_l554_554406


namespace fifth_term_of_geometric_sequence_l554_554641

theorem fifth_term_of_geometric_sequence (a r : ℕ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h_a : a = 5) (h_fourth_term : a * r^3 = 405) :
  a * r^4 = 405 := by
  sorry

end fifth_term_of_geometric_sequence_l554_554641


namespace limit_problem_l554_554342

open Real

theorem limit_problem :
  tendsto (fun x => (root 3 (1 + arctan (4 * x)) - root 3 (1 - arctan (4 * x))) /
                   (sqrt (1 - asin (3 * x)) - sqrt (1 + arctan (3 * x)))) (𝓝 0) (𝓝 (-8 / 9)) :=
begin
  -- sorry is a placeholder for the proof
  sorry
end

end limit_problem_l554_554342


namespace positive_difference_l554_554615

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end positive_difference_l554_554615


namespace ratio_triangle_DEF_to_rectangle_ABCD_l554_554861

theorem ratio_triangle_DEF_to_rectangle_ABCD
  (A B C D E F : Type)
  [rect : rectangle A B C D]
  (h1 : D.C = 2 * C.B)
  (h2 : lies_on_segment E A B)
  (h3 : lies_on_segment F A B)
  (trisect_EDF : trisect_angle E D F (angle A D C) (by norm_num)) :
  (area_triangle D E F) / (area_rectangle A B C D) = 3 * sqrt 3 / 16 :=
sorry

end ratio_triangle_DEF_to_rectangle_ABCD_l554_554861


namespace limit_of_sequence_l554_554403

noncomputable def arithmetic_sequence (a₁ : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a₁ (n + 1) = a₁ n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1) + (n * (n - 1) / 2) * 2

theorem limit_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : arithmetic_sequence a 2) (h2 : sum_of_first_n_terms a S) :
  ∀ a₁ : ℤ, (∀ n : ℕ, a n = 2 * n + a₁ - 2) → 
  ∀ n : ℕ, S n = n * (n + a₁ - 1) →
  -- Here we use the real number limits so we have to cast ints into reals
  (real.abs ((r := ℝ) ((λ x:ℝ, x) (lim (λ n, ((S n : ℝ) / (a n * a (n + 1) : ℤ))) - 1 / 4))) = 0) :=
begin
  sorry
end

end limit_of_sequence_l554_554403


namespace points_earned_l554_554549

-- Define the number of pounds required to earn one point
def pounds_per_point : ℕ := 4

-- Define the number of pounds Paige recycled
def paige_recycled : ℕ := 14

-- Define the number of pounds Paige's friends recycled
def friends_recycled : ℕ := 2

-- Define the total number of pounds recycled
def total_recycled : ℕ := paige_recycled + friends_recycled

-- Define the total number of points earned
def total_points : ℕ := total_recycled / pounds_per_point

-- Theorem to prove the total points earned
theorem points_earned : total_points = 4 := by
  sorry

end points_earned_l554_554549


namespace largest_subset_size_with_property_l554_554707

def no_four_times_property (S : Finset ℕ) : Prop := 
  ∀ {x y}, x ∈ S → y ∈ S → x = 4 * y → False

noncomputable def max_subset_size : ℕ := 145

theorem largest_subset_size_with_property :
  ∃ (S : Finset ℕ), (∀ x ∈ S, x ≤ 150) ∧ no_four_times_property S ∧ S.card = max_subset_size :=
sorry

end largest_subset_size_with_property_l554_554707


namespace angle_between_line_and_plane_range_l554_554934

def angle_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ ≤ 90

theorem angle_between_line_and_plane_range (θ : ℝ) :
  (∃ l p, θ = acute_angle_between_line_and_its_projection l p) → angle_range θ :=
by 
  sorry

end angle_between_line_and_plane_range_l554_554934


namespace total_chocolates_l554_554591

-- Definitions based on conditions
def chocolates_per_bag := 156
def number_of_bags := 20

-- Statement to prove
theorem total_chocolates : chocolates_per_bag * number_of_bags = 3120 :=
by
  -- skip the proof
  sorry

end total_chocolates_l554_554591


namespace f_range_g_min_value_t_range_l554_554051

noncomputable def f (x : ℝ) : ℝ := x - 4 / x

theorem f_range : Set.Icc (-3 : ℝ) 0 = Set.range (f : Set.Icc 1 2 → ℝ) :=
sorry

noncomputable def F (x a : ℝ) : ℝ := x^2 + 16 / x^2 - 2 * a * (x - 4 / x)

def g (a : ℝ) : ℝ :=
if a ≤ -3 then 6 * a + 17
else if a ≥ 0 then 8
else 8 - a^2

theorem g_min_value (a : ℝ) : ∃ x ∈ Set.Icc 1 2, F x a = g a :=
sorry

def ineq (a t : ℝ) : Prop := 8 - a^2 > -2 * a^2 + a * t + 4

theorem t_range (t : ℝ) : (∀ a ∈ Set.Ioo (-3 : ℝ) 0, ineq a t) → t > -4 :=
sorry

end f_range_g_min_value_t_range_l554_554051


namespace lucky_numbers_count_l554_554661

def isLucky (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≠ 0 ∧ 221 = n * k + (221 % n) ∧ (221 % n) % k = 0

def countLuckyNumbers : ℕ :=
  Nat.card (Finset.filter isLucky (Finset.range 222))

theorem lucky_numbers_count : countLuckyNumbers = 115 :=
  sorry

end lucky_numbers_count_l554_554661


namespace speed_of_water_current_l554_554316

theorem speed_of_water_current (v : ℝ) :
  (∀ (swim_still_speed : ℝ) (time_against_current : ℝ) (distance_against_current : ℝ), 
    swim_still_speed = 4 ∧ time_against_current = 2.5 ∧ distance_against_current = 5 →
    time_against_current = distance_against_current / (swim_still_speed - v)) →
  v = 2 :=
by
  intros h
  have h1 := h 4 2.5 5
  specialize h1 (by { simp })
  simp at h1
  linarith

end speed_of_water_current_l554_554316


namespace find_angle_A_l554_554851

theorem find_angle_A 
  (a b c A B C : ℝ)
  (h₀ : a = Real.sqrt 2)
  (h₁ : b = 2)
  (h₂ : Real.sin B - Real.cos B = Real.sqrt 2)
  (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  : A = Real.pi / 6 := 
  sorry

end find_angle_A_l554_554851


namespace problem_a_problem_b_l554_554211

open Real

-- Define the function d(t)
noncomputable def d (x : List ℝ) (t : ℝ) : ℝ :=
  (List.minimum (x.map (λ xi => abs (xi - t))) + List.maximum (x.map (λ xi => abs (xi - t)))) / 2

-- Define the midpoint c
noncomputable def c (x : List ℝ) : ℝ :=
  (List.minimum x + List.maximum x) / 2

-- Median function (for completeness)
noncomputable def median (x : List ℝ) : ℝ :=
  let sorted := List.sort (· ≤ ·) x
  if List.length x % 2 = 1 then
    sorted.get (List.length x / 2)
  else
    (sorted.get (List.length x / 2 - 1) + sorted.get (List.length x / 2)) / 2

-- Part (a) uniqueness of minimum value not always guaranteed
theorem problem_a (x : List ℝ) : ∃ t1 t2, t1 ≠ t2 ∧ d x t1 = d x t2 :=
  sorry

-- Part (b) comparison of d(c) and d(m)
theorem problem_b (x : List ℝ) : d x (c x) ≤ d x (median x) :=
  sorry

end problem_a_problem_b_l554_554211


namespace find_theta_l554_554140

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry
noncomputable def complex_conjugate (z : ℂ) : ℂ := sorry
noncomputable def abs (z : ℂ) : ℝ := sorry
noncomputable def arg (z : ℂ) : ℝ := sorry
noncomputable def omega (θ : ℝ) : ℂ := 
  let z := (cos θ : ℂ) + (sin θ : ℂ) * complex.I in
  let z_conj := complex_conjugate z in
  (1 - z_conj^4) / (1 + z^4)

theorem find_theta (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : abs (omega θ) = (Real.sqrt 3 / 3 : ℝ)) 
  (h4 : arg (omega θ) < π / 2) : 
  θ = π / 12 := 
sorry

end find_theta_l554_554140


namespace inequality_proof_l554_554411

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ 1 / 2 * (a + b + c) := 
by
  sorry

end inequality_proof_l554_554411


namespace remaining_amount_is_correct_l554_554838

-- Define the original price based on the deposit paid
def original_price : ℝ := 1500

-- Define the discount percentage
def discount_percentage : ℝ := 0.05

-- Define the sales tax percentage
def tax_percentage : ℝ := 0.075

-- Define the deposit already paid
def deposit_paid : ℝ := 150

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_percentage)

-- Define the sales tax amount
def sales_tax : ℝ := discounted_price * tax_percentage

-- Define the final cost after adding sales tax
def final_cost : ℝ := discounted_price + sales_tax

-- Define the remaining amount to be paid
def remaining_amount : ℝ := final_cost - deposit_paid

-- The statement we need to prove
theorem remaining_amount_is_correct : remaining_amount = 1381.875 :=
by
  -- We'd normally write the proof here, but that's not required for this task.
  sorry

end remaining_amount_is_correct_l554_554838


namespace min_value_f_l554_554384

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / real.sqrt (x^2 + 5)

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f(y) ≥ f(x) ∧ f(x) = 9 / real.sqrt 5 :=
by sorry

end min_value_f_l554_554384


namespace distinct_remainders_of_prime_squares_l554_554333

open Nat

theorem distinct_remainders_of_prime_squares (p : ℕ) (hp_prime: Prime p) (hp_gt_five : p > 5) :
    ∃ s : Finset ℕ, (∀ q : fin el 240, q ∈ s ↔ ∃ (n : ℕ), n < 240 ∧ q = p^2 % 240) ∧ s.card = 2 := 
sorry

end distinct_remainders_of_prime_squares_l554_554333


namespace Poe_speed_is_40_l554_554185

variable (P : ℝ) -- Poe's speed
variable (Teena_speed : ℝ := 55) -- Teena's speed in mph
variable (Teena_gap : ℝ := 7.5) -- Initial gap between Teena and Poe in miles
variable (Teena_ahead : ℝ := 15) -- Distance Teena will be ahead of Poe in 90 minutes
variable (time : ℝ := 1.5) -- Time period in hours

-- Total distance gained by Teena over Poe in 1.5 hours is 22.5 miles
def distance_condition (P : ℝ) : Prop :=
  Teena_speed * time - P * time = Teena_gap + Teena_ahead

axiom Poe_speed : P = 40

theorem Poe_speed_is_40 : distance_condition P → P = 40 := by
  intro h
  sorry

end Poe_speed_is_40_l554_554185


namespace common_fraction_difference_l554_554341

def repeating_decimal := 23 / 99
def non_repeating_decimal := 23 / 100
def fraction_difference := 23 / 9900

theorem common_fraction_difference : repeating_decimal - non_repeating_decimal = fraction_difference := 
by
  sorry

end common_fraction_difference_l554_554341


namespace log_base_one_fifth_twenty_five_l554_554756

theorem log_base_one_fifth_twenty_five : log (1/5) 25 = -2 :=
by
  sorry

end log_base_one_fifth_twenty_five_l554_554756


namespace arc_invariance_l554_554985

variable {n : ℕ}
variables {R B : Fin n → ℝ × ℝ}
 -- R and B represent functions from indices to points on the unit circle except (1, 0)

theorem arc_invariance (h_1 : ∀ i, R i ≠ (1, 0)) (h_2 : ∀ i, B i ≠ (1, 0))
    (h_R : ∀ i j, i ≠ j → R i ≠ R j) (h_B : ∀ i j, i ≠ j → B i ≠ B j)
    (h_circle_R : ∀ i, (R i).fst * (R i).fst + (R i).snd * (R i).snd = 1)
    (h_circle_B : ∀ i, (B i).fst * (B i).fst + (B i).snd * (B i).snd = 1)
    (count_arcs : Fin n → Fin n)
    (ha : ∀ i, ∃ j, count_arcs j = i)
    (hb : ∀ i, B (count_arcs i) = ith_nearest_blue(R i, B, (i))) :
    ∀ (σ : Perm (Fin n)),
    (∑ i in Finset.range n, arc_contains_point (R (σ i)) (B (σ i)) (1,0)) =
    (∑ i in Finset.range n, arc_contains_point (R i) (B i) (1,0)) :=
sorry

/-- Helper function to find the ith nearest blue point counterclockwise around the circle from the ith red point --/
noncomputable def ith_nearest_blue(R : Fin n → (ℝ × ℝ), B : Fin n → (ℝ × ℝ), i : Fin n) : ℝ × ℝ := sorry

/-- Helper function to check if an arc contains the point (1, 0) --/
noncomputable def arc_contains_point (p1 p2 point : ℝ × ℝ) : Bool := sorry

end arc_invariance_l554_554985


namespace lucky_numbers_l554_554690

def is_lucky (n : ℕ) : Prop :=
  ∃ q r : ℕ, 221 = n * q + r ∧ r % q = 0

def lucky_numbers_count (a b : ℕ) : ℕ :=
  Nat.card {n // a ≤ n ∧ n ≤ b ∧ is_lucky n}

theorem lucky_numbers (a b : ℕ) (h1 : a = 1) (h2 : b = 221) :
  lucky_numbers_count a b = 115 := 
sorry

end lucky_numbers_l554_554690


namespace log_base_one_fifth_twenty_five_l554_554757

theorem log_base_one_fifth_twenty_five : log (1/5) 25 = -2 :=
by
  sorry

end log_base_one_fifth_twenty_five_l554_554757


namespace acute_triangle_cevians_angle_l554_554563

-- Define the problem in Lean 4 as stated, with the necessary conditions.
theorem acute_triangle_cevians_angle {ABC : Type} [triangle ABC] (A B C D M H O : Point)
  (AngleBisector : ∀ (A B C D : Point), Cevian A B C D)
  (Median : ∀ (A B C M : Point), M = midpoint A C)
  (Altitude : ∀ (A B C H : Point), is_perpendicular (line C H) (line A B))
  (Intersection : ∀ (A D B M C H O : Point), concurrent (line A D) (line B M) (line C H))
  (Acute : ∀ (α : ℝ), 0 < α ∧ α < pi / 2 ∧ ∠ B A C = α) :
  ∠ B A C > pi / 4 :=
by 
  sorry

end acute_triangle_cevians_angle_l554_554563


namespace apples_total_l554_554118

-- We start by defining the variables and conditions in Lean
variables (x : ℕ) -- number of apples Kylie picked
variables (kayla_apples : ℕ) (total_apples : ℕ)

-- Define the condition given in the problem
def condition1 : Prop := kayla_apples = 4 * x + 10
def condition2 : Prop := kayla_apples = 274

-- State the theorem to be proved
theorem apples_total (h_condition1 : condition1) (h_condition2 : condition2) : total_apples = 340 :=
by
  -- proof omitted
  sorry

end apples_total_l554_554118


namespace problem_solution_l554_554922

theorem problem_solution :
  ∀ (m n : ℕ), 
  (∀ (x y z : ℝ), x + y + z = 0 → 
  (x^(m + n) + y^(m + n) + z^(m + n)) / (m + n) = 
  ((x^m + y^m + z^m) / m) * ((x^n + y^n + z^n) / n)) →
  (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 5 ∧ n = 2) :=
begin
  sorry
end

end problem_solution_l554_554922


namespace triangle_side_not_cut_l554_554533

-- Definitions representing the triangle and the line not passing through any vertices of the triangle
structure Triangle :=
  (A B C : Point)

def is_line {P : Type} (ℓ : P → Prop) : Prop :=
  ∃ P Q: P, P ≠ Q ∧ ∀ R, ℓ R ↔ collinear P Q R

-- Non-intersection condition: the line does not pass through any vertex of the triangle
def line_not_through_vertices (T : Triangle) (ℓ : Point → Prop) : Prop := 
  ¬ (ℓ T.A ∨ ℓ T.B ∨ ℓ T.C)

-- The main theorem to prove: some side of the triangle is not cut by the line
theorem triangle_side_not_cut (T : Triangle) (ℓ : Point → Prop) 
  (hℓ : is_line ℓ) (h_not_through : line_not_through_vertices T ℓ) : 
    ∃ (P Q : Point), (P = T.A ∧ Q = T.B ∨ P = T.B ∧ Q = T.C ∨ P = T.C ∧ Q = T.A) ∧ ¬ ∃ R, ℓ R ∧ collinear P Q R :=
sorry

end triangle_side_not_cut_l554_554533


namespace sample_mean_and_variance_l554_554430

noncomputable def xy_product : ℝ  :=
  let x := 10 + Float.sqrt 5
  let y := 10 - Float.sqrt 5
  x * y

theorem sample_mean_and_variance (x y : ℝ)
  (avg_cond : (9 + 10 + 11 + x + y) / 5 = 10)
  (var_cond : ((9-10)^2 + (10-10)^2 + (11-10)^2 + (x-10)^2 + (y-10)^2) / 5 = 2) :
  x * y = 96 :=
by
  sorry

end sample_mean_and_variance_l554_554430


namespace count_pairs_l554_554575

theorem count_pairs (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2012)
  (cond : ∀ n : ℕ, 5^n < 2^m ∧ 2^m < 2^(m + 2) ∧ 2^(m + 2) < 5^(n + 1)) :
  ∃ k : ℕ, k = 279 :=
begin
  sorry
end

end count_pairs_l554_554575


namespace count_lucky_numbers_l554_554682

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 221 ∧ let q := 221 / n in let r := 221 % n in r % q = 0

theorem count_lucky_numbers : ∃ (N : ℕ), N = 115 ∧ (∀ n, is_lucky n → n ≤ 221) := 
by
  apply Exists.intro 115
  constructor
  · rfl
  · intros n hn
    cases hn with hpos hrange
    exact hrange.left
  · sorry

end count_lucky_numbers_l554_554682


namespace general_formula_min_value_Tn_l554_554787

-- Given conditions
def an_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 + a 5 = 14 ∧ (a 1 + a 2 + a 3 + a 4 + a 5) = 25

-- General formula for a_n
theorem general_formula (a : ℕ → ℤ) (ha : an_arithmetic_sequence a) :
  ∀ n, a n = 2 * n - 1 :=
sorry

-- Minimum value of sum of first n terms of sequence b_n
theorem min_value_Tn (a : ℕ → ℤ) (ha : an_arithmetic_sequence a) :
  let bn n := (2 : ℚ) / (a n * a (n+1))
  in ∀ Tn : ℕ → ℚ, (Tn n = ∑ i in finset.range n, bn i) →
      ∃ N, ( ∀ n, Tn n ≥ Tn N ) ∧ Tn N = (2 : ℚ) / 3 :=
sorry

end general_formula_min_value_Tn_l554_554787


namespace n_leq_1972_l554_554452

theorem n_leq_1972 (n : ℕ) (h1 : 4 ^ 27 + 4 ^ 1000 + 4 ^ n = k ^ 2) : n ≤ 1972 :=
by
  sorry

end n_leq_1972_l554_554452


namespace log_one_fifth_25_eq_neg2_l554_554753

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end log_one_fifth_25_eq_neg2_l554_554753


namespace min_value_proof_l554_554901

noncomputable def min_value_problem := 
∀ (a b c d e f : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
(a + b + c + d + e + f = 9) → 
(\frac{1}{a} + \frac{2}{b} + \frac{9}{c} + \frac{8}{d} + \frac{18}{e} + \frac{32}{f} ≥ 24)

theorem min_value_proof : min_value_problem :=
by
  sorry

end min_value_proof_l554_554901


namespace no_real_roots_prob_l554_554146

noncomputable def xi_distribution : string := "Normal Distribution N(2, σ^2)"

def discriminant (ξ : ℝ) : ℝ :=
  16 - 8 * ξ

theorem no_real_roots_prob (ξ : ℝ) (hξ : xi_distribution = "Normal Distribution N(2, σ^2)") :
  (∀ f : ℝ → ℝ, f = (λ x : ℝ, 2 * x^2 - 4 * x + ξ) → (P (discriminant ξ < 0) = 1 / 2)) := sorry

end no_real_roots_prob_l554_554146


namespace tan_cot_solutions_count_l554_554763

/-- Prove that the number of solutions to the equation 
\tan (7 \pi \cos \theta) = \cot (7 \pi \sin \theta) where \theta \in (0, 2 \pi) is 36. -/
theorem tan_cot_solutions_count :
  ∃ (n : ℕ), n = 36 ∧ (∀ θ : ℝ, θ ∈ Ioo 0 (2 * Real.pi) → tan (7 * Real.pi * cos θ) = 1 / tan (7 * Real.pi * sin θ) → true) :=
sorry

end tan_cot_solutions_count_l554_554763


namespace only_a_7_has_integer_solution_l554_554359

theorem only_a_7_has_integer_solution :
  ∀ (a : ℕ), (∃ (x : ℤ), 
    (∏ i in finset.range (a+1), (1 + 1 / (x + i))) = a - x) ↔ (a = 7) :=
sorry

end only_a_7_has_integer_solution_l554_554359


namespace inclination_angle_of_line_l554_554942

theorem inclination_angle_of_line (c : ℝ) : 
  ∃ θ ∈ Icc (0 : ℝ) real.pi, 
    tan θ = -1 / sqrt 3 ∧ θ = 5 * real.pi / 6 :=
by
suffices hθ : tan (5 / 6 * real.pi) = -1 / sqrt 3, from
⟨5 / 6 * real.pi, ⟨by norm_num, by norm_num⟩, hθ, by norm_num⟩,
-- further proof steps would be
sorry

end inclination_angle_of_line_l554_554942


namespace rides_with_remaining_tickets_l554_554989

theorem rides_with_remaining_tickets (T_total : ℕ) (T_spent : ℕ) (C_ride : ℕ)
  (h1 : T_total = 40) (h2 : T_spent = 28) (h3 : C_ride = 4) :
  (T_total - T_spent) / C_ride = 3 := by
  sorry

end rides_with_remaining_tickets_l554_554989


namespace parallel_vectors_implies_lambda_value_l554_554819

theorem parallel_vectors_implies_lambda_value :
  ∀ (λ : ℝ), let a := (1, 1) 
             let b := (2, -1) in 
             (λ • a + b) ∥ (a - 2 • b) → λ = -1 / 2 :=
by
  intros λ a b h
  sorry

end parallel_vectors_implies_lambda_value_l554_554819


namespace triangle_intersection_area_sum_l554_554957

-- Lean statement for the problem
theorem triangle_intersection_area_sum (C D A B : Type)
[DecidableEq C] [DecidableEq D] [DecidableEq A] [DecidableEq B]
(triangle_ABC_congruent_BAD : congruent_triangle (Triangle A B C) (Triangle B A D))
(AB : length (Segment A B) = 9)
(BC : length (Segment B C) = 10)
(AD : length (Segment A D) = 10)
(CA : length (Segment C A) = 17)
(DB : length (Segment D B) = 17) :
  let I := intersection (Triangle A B C) (Triangle B A D) in
  let area_I := area I in
  ∃ m n : ℕ, 
    area_I = (m : ℝ) / (n : ℝ) ∧ 
    nat.coprime m n ∧ 
    m + n = 59 := 
sorry

end triangle_intersection_area_sum_l554_554957


namespace simplify_sqrt7_pow6_l554_554926

theorem simplify_sqrt7_pow6 : (Real.sqrt 7)^6 = (343 : Real) :=
by 
  -- we'll fill in the proof later
  sorry

end simplify_sqrt7_pow6_l554_554926


namespace ellipse_more_circular_l554_554971

noncomputable def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 - (b*a) / (b*a))

theorem ellipse_more_circular :
  let A := eccentricity (2 * real.sqrt 2 / 3) (2 * real.sqrt 2 / 3)
  let B := eccentricity (1 / 2) (1 / 2)
  let C := eccentricity (2 * real.sqrt 2 / 3) (2 * real.sqrt 2 / 3)
  let D := eccentricity (real.sqrt 10 / 5) (real.sqrt 10 / 5)
  in B < A ∧ B < C ∧ B < D :=
by {
  have e_A := (2 : ℝ) * real.sqrt 2 / 3,
  have e_B := 1 / 2,
  have e_C := (2 : ℝ) * real.sqrt 2 / 3,
  have e_D := real.sqrt 10 / 5,
  sorry
}

end ellipse_more_circular_l554_554971


namespace number_of_distinct_terms_l554_554741

-- Define the given expression
def initial_expr (x y : ℝ) : ℝ := ((x + 5 * y) ^ 3 * (x - 5 * y) ^ 3) ^ 3

-- Define the expanded expression
def expanded_expr (x y : ℝ) : ℝ := (x ^ 2 - 25 * y ^ 2) ^ 9

-- State the problem as a theorem
theorem number_of_distinct_terms {x y : ℝ} :
  (∃ n : ℕ, n = 10 ∧ (initial_expr x y = expanded_expr x y ∧ ∀ k : ℕ, k < 10 → (x ^ (2 * (9 - k)) * y ^ (2 * k)).isDistinctTerm)) := sorry

end number_of_distinct_terms_l554_554741


namespace find_a_l554_554891

def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x) / (x + a)

theorem find_a :
  (∀ x, f x a = (Real.exp x) / (x + a)) →
  (∂ x, ∀ x, f x a) 1 = Real.exp 1 / 4 →
  a = 1 :=
by
  intros H₁ H₂
  sorry

end find_a_l554_554891


namespace part1_part2_l554_554400

-- Define the sequence {a_n} given the conditions a1 = -1 and a_n = 2a_{n-1} - 1 for n >= 2
noncomputable def seq_a : ℕ → ℝ
| 0     := 0  -- we will skip using a_0 since it is irrelevant for this problem
| 1     := -1
| (n+2) := 2 * seq_a (n+1) - 1

-- Define the general formula for {a_n}
def general_a (n : ℕ) : ℝ := 1 - 2^n

-- (1) Prove that the general formula for the sequence {a_n} is correct
theorem part1 (n : ℕ) : seq_a (n+1) = general_a (n+1) := sorry

-- For part 2, we need to use the sequence {b_n} and {T_n}
def b (n : ℕ) : ℝ := real.log2 (1 - seq_a (n+1))

noncomputable def sum_T (n : ℕ) : ℝ :=
  ∑ k in finset.range n, 1 / (b (k+2) * b (k+1))

-- (2) Prove that T_n < 1
theorem part2 (n : ℕ) : sum_T n < 1 := sorry

end part1_part2_l554_554400


namespace probability_n2_mod_2310_l554_554884

open BigOperators

noncomputable def euler_totient (n : ℕ) : ℕ :=
∑ d in (Nat.divisors n), if Nat.coprime d n then 1 else 0

noncomputable def first_25_primes : List ℕ :=
[take 25 $ List.filter Nat.prime (List.range 1000)]

noncomputable def n0 : ℕ :=
first_25_primes.foldr (*) 1

theorem probability_n2_mod_2310 :
  let n1 := some_divisor n0
  let n2 := some_divisor n1
  ∀ (n1 n2 : ℕ), (P n1 = euler_totient(n1) / euler_totient(n0)) →
               (P n2 = euler_totient(n2) / euler_totient(n1)) →
               (P n2 ≡ 0 [MOD 2310]) =
               (256 / 5929) := sorry

end probability_n2_mod_2310_l554_554884


namespace find_QR_length_l554_554699

noncomputable def O := sorry
noncomputable def A := sorry
noncomputable def B := sorry
noncomputable def P := sorry
noncomputable def Q := sorry
noncomputable def R := sorry

axiom OP_eq_15 : dist O P = 15
axiom OA_eq_32 : dist O A = 32
axiom OB_eq_64 : dist O B = 64

noncomputable def PR_length : ℝ := 30

theorem find_QR_length :
  dist Q R = PR_length :=
sorry

end find_QR_length_l554_554699


namespace proof_example_l554_554137

variable {α : Type*}

def distinct_positive (a : α) [PartialOrder α] := ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 8 → a i ≠ a j

def gcd_three_one (a : α → ℕ) := ∀ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 8 → (Nat.gcd (Nat.gcd (a i) (a j)) (a k) = 1)

theorem proof_example (a : ℕ → ℕ) (h_distinct : distinct_positive a) (h_gcd : gcd_three_one a) :
  ∃ n ≥ 8, ∃ (m : ℕ → ℕ), (distinct_positive m ∧ ∀ p q r, 1 ≤ p ∧ p < q ∧ q < r ∧ r ≤ n →
  ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 8 ∧ (a i * a j ∣ (m p + m q + m r))) :=
sorry

end proof_example_l554_554137


namespace river_flow_rate_l554_554301

variables (d w : ℝ) (V : ℝ)

theorem river_flow_rate (h₁ : d = 4) (h₂ : w = 40) (h₃ : V = 10666.666666666666) :
  ((V / 60) / (d * w) * 3.6) = 4 :=
by sorry

end river_flow_rate_l554_554301


namespace find_a_l554_554793

theorem find_a (a : ℝ) (h : (2:ℝ)^2 + 2 * a - 3 * a = 0) : a = 4 :=
sorry

end find_a_l554_554793


namespace exists_k_stabilize_special_case_a_k_l554_554196

def f (A : ℕ) : ℕ :=
  let digits := A.digits 10 -- Digits in base 10
  digits.mapWithIndex (λ i ai => 2^i * ai).sum

theorem exists_k_stabilize (A : ℕ) : ∃ k : ℕ, let Ak := (nat.iterate f k A) in nat.iterate f (k+1) A = Ak :=
  sorry

theorem special_case_a_k (A : ℕ) (hA : A = 19^86) : ∃ k : ℕ, let Ak := (nat.iterate f k A) in Ak = 19 :=
  sorry

end exists_k_stabilize_special_case_a_k_l554_554196


namespace pentagon_leftmost_vertex_x_coordinate_l554_554950

noncomputable def area_pentagon_log_consecutive (n : ℕ) : ℝ :=
  ∑ i in finset.range 5, (↑n + (i : ℕ)) * (Real.log (↑n + (i : ℕ) + 1)) - 
  ∑ i in finset.range 5, (Real.log (↑n + (i : ℕ))) * (↑n + (i : ℕ) + 1)

theorem pentagon_leftmost_vertex_x_coordinate :
  ∃ n : ℕ, 
  n > 0 ∧ (Real.log ((n + 1) * (n + 2) * (n + 3)) - Real.log (n * (n + 4))) = Real.log (23 / 21) 
  ∧ n = 9 :=
sorry

end pentagon_leftmost_vertex_x_coordinate_l554_554950


namespace hyperbola_foci_coordinates_l554_554194

theorem hyperbola_foci_coordinates :
  (a^2 = 7) → (b^2 = 3) → (c^2 = a^2 + b^2) → (c = Real.sqrt c^2) →
  ∃ (x y : ℝ), (x = Real.sqrt 10 ∧ y = 0) ∨ (x = -Real.sqrt 10 ∧ y = 0) :=
by
  intros a2_eq b2_eq c2_eq c_eq
  have h1 : c2 = 10 := by rw [a2_eq, b2_eq, add_comm]
  have h2 : c = Real.sqrt 10 := by rw [h1, Real.sq_sqrt (show 0 ≤ 10 by norm_num)]
  use (Real.sqrt 10)
  use 0
  use (-Real.sqrt 10)
  use 0
  sorry

end hyperbola_foci_coordinates_l554_554194


namespace count_mappings_l554_554902

open Finset

def M : Finset (Fin 3) := {0, 1, 2}
def N : Finset ℤ := {-1, 0, 1}

noncomputable def satisfies_condition (f : Fin 3 → ℤ) : Prop :=
  f 0 + f 1 + f 2 = 0

theorem count_mappings : (card {f : (Fin 3 → ℤ) // ∀ x ∈ M, f x ∈ N ∧ satisfies_condition f} = 7) :=
sorry

end count_mappings_l554_554902


namespace convex_hexagon_equal_angles_equal_sides_l554_554325

theorem convex_hexagon_equal_angles_equal_sides (A B C D E F : Point)
    (hABCDEF : ConvexHexagon A B C D E F)
    (h_ang_eq : ∀ i, (internal_angle_deg (vertices_of_hexagon A B C D E F) i) = 120) :
    abs (BC - EF) = abs (DE - AB) ∧ abs (DE - AB) = abs (AF - CD) :=
by
  sorry

end convex_hexagon_equal_angles_equal_sides_l554_554325


namespace faster_train_length_l554_554959

theorem faster_train_length (speed_faster speed_slower : ℕ) (time_seconds : ℕ) (kmph_to_mps : ℚ)
  (h_speed_faster : speed_faster = 90) (h_speed_slower : speed_slower = 36) (h_time_seconds : time_seconds = 29) 
  (h_conversion_factor : kmph_to_mps = 5 / 18) :
  let relative_speed_kmph := speed_faster - speed_slower in
  let relative_speed_mps := relative_speed_kmph * kmph_to_mps in
  let length_meters := relative_speed_mps * time_seconds in
  length_meters = 435 := by
  sorry

end faster_train_length_l554_554959


namespace sum_ineq_l554_554920

theorem sum_ineq (x y z t : ℝ) (h₁ : x + y + z + t = 0) (h₂ : x^2 + y^2 + z^2 + t^2 = 1) :
  -1 ≤ x * y + y * z + z * t + t * x ∧ x * y + y * z + z * t + t * x ≤ 0 :=
by
  sorry

end sum_ineq_l554_554920


namespace angle_in_interval_l554_554077

theorem angle_in_interval (alpha : ℝ) (k : ℤ) :
    α = 2 * k * real.pi + 8 * real.pi / 5 →
    (∃ θ ∈ [0, 2 * real.pi], θ = α / 4) :=
by
    assume h : α = 2 * k * real.pi + 8 * real.pi / 5
    use (k : ℝ) * real.pi / 2 + 2 * real.pi / 5
    split
    -- Show that θ is in [0, 2 * real.pi]
    . sorry
    -- Show that θ equals α / 4 given the definition of α
    . sorry

end angle_in_interval_l554_554077


namespace root_avg_one_l554_554721

noncomputable def avg_roots {a : ℝ} (u : ℕ → ℝ) (v : ℕ → ℝ) (k l : ℕ) : ℝ :=
  (∑ i in finset.range k, u i + ∑ i in finset.range l, v i) / (k + l)

theorem root_avg_one (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (u : ℕ → ℝ) 
  (v : ℕ → ℝ) 
  (k l : ℕ) 
  (hu : ∀ i < k, a ^ (u i) + 2 * u i - 4 = 0) 
  (hv : ∀ i < l, log a (2 * v i) + v i - 2 = 0) 
  (hk : k > 0) 
  (hl : l > 0) : 
  avg_roots u v k l = 1 := 
sorry

end root_avg_one_l554_554721


namespace price_increase_percentage_l554_554296

theorem price_increase_percentage (c : ℝ) (r : ℝ) (p : ℝ) 
  (h1 : r = 1.4 * c) 
  (h2 : p = 1.15 * r) : 
  (p - c) / c * 100 = 61 := 
sorry

end price_increase_percentage_l554_554296


namespace triangle_perimeter_l554_554710

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 28) (h2 : b = 16) (h3 : c = 18) : a + b + c = 62 := 
by
  rw [h1, h2, h3]
  sorry

end triangle_perimeter_l554_554710


namespace soccer_basketball_prices_cost_effective_plan_l554_554287

theorem soccer_basketball_prices (x y : ℕ) (hx : x + 3 * y = 275) (hy : 3 * x + 2 * y = 300) :
  x = 50 ∧ y = 75 := by
  sorry

theorem cost_effective_plan (x y m n : ℕ) 
  (hx : x = 50) (hy : y = 75) 
  (h1 : m ≤ 3 * (80 - m)) 
  (h2 : m + n = 80) 
  (W : ℕ → ℕ → ℕ := λ m n, 50 * m + 75 * n) 
  : m = 60 ∧ n = 20 := by
  sorry

end soccer_basketball_prices_cost_effective_plan_l554_554287


namespace lcm_18_35_l554_554003

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l554_554003


namespace greatest_possible_radius_of_circle_l554_554840

theorem greatest_possible_radius_of_circle
  (π : Real)
  (r : Real)
  (h : π * r^2 < 100 * π) :
  ∃ (n : ℕ), n = 9 ∧ (r : ℝ) ≤ 10 ∧ (r : ℝ) ≥ 9 :=
by
  sorry

end greatest_possible_radius_of_circle_l554_554840


namespace positive_integer_sequence_large_divisor_count_l554_554505

noncomputable def x : ℝ := sorry  -- we assume x exists such that x + x⁻¹ = 3
noncomputable def a (n : ℕ) : ℝ := x^n + x^(-n)

lemma x_plus_inv_eq_three : x + x^-1 = 3 := sorry

theorem positive_integer_sequence :
  ∀ n ≥ 0, ∃ k : ℤ, a n = k ∧ 0 < k := sorry

theorem large_divisor_count :
  ∃ (d : ℕ), d ≥ 1439 * 2^1437 ∧ d ∣ (x ^ (3^1437) + x ^ (-3^1437)) := sorry

end positive_integer_sequence_large_divisor_count_l554_554505


namespace log_eq_neg_two_l554_554748

theorem log_eq_neg_two : ∀ (x : ℝ), (1 / 5) ^ x = 25 → x = -2 :=
by
  intros x h
  sorry

end log_eq_neg_two_l554_554748


namespace area_ratio_S_to_T_l554_554501

-- Definitions based on the conditions
def T := { t : ℝ × ℝ × ℝ // t.1 + t.2 + t.3 = 1 ∧ t.1 ≥ 0 ∧ t.2 ≥ 0 ∧ t.3 ≥ 0 }
def S := { t : T // ((t.val.1 ≥ 1/2 ∧ t.val.2 ≥ 1/3) ∧ t.val.3 < 1/6) ∨
                    ((t.val.1 ≥ 1/2 ∧ t.val.3 ≥ 1/6) ∧ t.val.2 < 1/3) ∨
                    ((t.val.2 ≥ 1/3 ∧ t.val.3 ≥ 1/6) ∧ t.val.1 < 1/2) }

-- Statement to prove the question
theorem area_ratio_S_to_T : 
  let m : ℕ := 7,
      n : ℕ := 18 in
  m + n = 25 :=
  by
  sorry

end area_ratio_S_to_T_l554_554501


namespace classroom_boys_count_l554_554230

theorem classroom_boys_count (initial_g: ℕ) (initial_b: ℕ) (new_b: ℕ) :
  (initial_g + 10 = 22) → (initial_b = initial_g + 5) → (new_b = initial_b + 3) → new_b = 20 :=
by
  -- giving condition a name for clarity
  intro h1
  -- initial number of girls from the first condition
  have g_eq := Nat.eq_of_add_eq_add_right h1
  -- using the second condition b = initial_g + 5
  intro h2
  have b_eq := congr_arg (λ g, g + 5) g_eq
  -- concluding b = 17
  have b_val := Eq.trans b_eq (add_comm (5 : Nat) initial_g)
  have init_b_value := Nat.add_comm initial_g 5
  -- proving new number of boys after additional boys entered
  intro h3
  exact h3.subst (Nat.add_comm 17 3).symm

end classroom_boys_count_l554_554230


namespace lucky_numbers_count_l554_554694

def is_lucky (n : ℕ) : Prop := 
  ∃ (k q r : ℕ), 
    1 ≤ n ∧ n ≤ 221 ∧
    221 = n * q + r ∧ 
    r = k * q ∧ 
    r < n

def count_lucky_numbers : ℕ := 
  ∑ n in (finset.range 222).filter is_lucky, 1

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
  by 
    sorry

end lucky_numbers_count_l554_554694


namespace inequality_one_solution_range_of_a_l554_554145

def f (x : ℝ) : ℝ := |2 * x + 3| + |x - 1|

theorem inequality_one_solution :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 0} :=
sorry

theorem range_of_a (a : ℝ) :
  ∃ x ∈ Icc (-3 / 2) 1, a + 1 > f x → a > 3 / 2 :=
sorry

end inequality_one_solution_range_of_a_l554_554145


namespace jasmine_dinner_time_l554_554877

-- Define the conditions as variables and constants.
constant work_end_time : ℕ := 16  -- Representing 4:00 pm in 24-hour format (16:00)
constant commute_time : ℕ := 30   -- in minutes
constant grocery_time : ℕ := 30   -- in minutes
constant dry_cleaning_time : ℕ := 10  -- in minutes
constant dog_grooming_time : ℕ := 20  -- in minutes
constant cooking_time : ℕ := 90   -- in minutes

-- Define a function to sum up the times
def total_time_after_work : ℕ := commute_time + grocery_time + dry_cleaning_time + dog_grooming_time + cooking_time

def time_to_hour_minutes (total_minutes : ℕ) : (ℕ × ℕ) := 
  (total_minutes / 60, total_minutes % 60)

-- Prove that Jasmine will eat dinner at 7:00 pm (19:00 in 24-hour format)
theorem jasmine_dinner_time : total_time_after_work / 60 + work_end_time = 19 := by
  -- Leave the proof part as sorry since we don't need to provide proof steps
  sorry

end jasmine_dinner_time_l554_554877


namespace sum_reciprocals_seven_l554_554587

variable (x y : ℝ)

theorem sum_reciprocals_seven (h : x + y = 7 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / x) + (1 / y) = 7 := 
sorry

end sum_reciprocals_seven_l554_554587


namespace minimum_expression_l554_554511

theorem minimum_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 8) :
  \(\frac{a}{2b} + \frac{2b}{c} + \frac{c}{a}\) ≥ 3 :=
sorry

end minimum_expression_l554_554511


namespace measure_XIZ_l554_554493

def triangle (X Y Z I : Type) : Prop :=
  -- Define the properties of the triangle
  sorry

def angle_bisector (X Y Z P Q R I : Type) : Prop :=
  -- Define the properties of the angle bisectors intersecting at I
  sorry

constant XYZ : Type  -- Represents triangle XYZ

constant X IQ P : Type  -- Represents point types for X and IQ

constant Y IQ Q : Type  -- Represents point types for Y and IQ

constant Z IQ R : Type  -- Represents point types for Z and IQ

constant X I Z : Type  -- Represents point types for X and Z with incenter I

constant angle : XYZ → ℕ

-- Given conditions
axiom angle_XYZ : angle XYZ = 42

-- Question: Find the measure of angle XIZ
theorem measure_XIZ (X Y Z P Q R I : XYZ) : ∀ angle X I Z, angle X I Z = 69 := sorry

end measure_XIZ_l554_554493


namespace dot_product_a_b_l554_554795

variables (i j k : ℝ → ℝ → ℝ)
variables (a b : ℝ × ℝ × ℝ)

-- Assuming that i, j, and k are the standard unit vectors in R^3
def i := (1, 0, 0)
def j := (0, 1, 0)
def k := (0, 0, 1)

-- Definitions of vectors a and b based on the problem
def vector_a := (2, -1, 1)
def vector_b := (1, 1, -3)

-- Dot product function for 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the theorem
theorem dot_product_a_b : dot_product vector_a vector_b = -2 :=
begin
  sorry
end

end dot_product_a_b_l554_554795


namespace count_lucky_numbers_l554_554650

def is_lucky (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 221) ∧ (221 % n % (221 / n) = 0)

theorem count_lucky_numbers : (finset.filter is_lucky (finset.range 222)).card = 115 :=
by
  sorry

end count_lucky_numbers_l554_554650


namespace box_occupancy_percentage_l554_554306

-- Defining the problem
def box_dimensions : ℕ × ℕ × ℕ := (8, 5, 14)
def cube_side_length : ℕ := 4

-- Volume calculations
def box_volume : ℕ :=
  let (length, width, height) := box_dimensions
  length * width * height

def cube_volume : ℕ :=
  cube_side_length * cube_side_length * cube_side_length

def max_cubes : ℕ :=
  let (length, width, height) := box_dimensions
  (length / cube_side_length) * (width / cube_side_length) * (height / cube_side_length)

def occupied_volume : ℕ :=
  max_cubes * cube_volume

def percentage_occupied : ℚ :=
  ((occupied_volume : ℚ) / (box_volume : ℚ)) * 100

-- Statement to be proven
theorem box_occupancy_percentage : percentage_occupied ≈ 68.57 :=
by
  sorry

end box_occupancy_percentage_l554_554306


namespace chord_length_eq_two_l554_554042

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := Real.sqrt (a^2 - b^2)
noncomputable def focus_left : ℝ × ℝ := (-c, 0)
noncomputable def angle : ℝ := Real.pi / 6

-- Ellipse equation
def ellipse (x y : ℝ) : Prop :=
  (x^2) / 9 + y^2 = 1

-- Equation of the line passing through the left focus of the ellipse at the given angle
def line (x y : ℝ) : Prop :=
  y = Real.sqrt(3) / 3 * (x + 2 * Real.sqrt(2))

-- Length of the chord AB where the line intersects the ellipse
def length_of_chord : ℝ :=
  let x1 := -3 * Real.sqrt(2) - Real.sqrt(2)
  let x2 := -3 * Real.sqrt(2) + Real.sqrt(2)
  let x1_x2 := x1 * x2
  let x1_plus_x2 := x1 + x2
  Real.sqrt((1 + 1 / 3) * (x1_plus_x2^2 - 4 * x1_x2))

theorem chord_length_eq_two : length_of_chord = 2 := by
  sorry

end chord_length_eq_two_l554_554042


namespace hemisphere_surface_area_with_base_l554_554588

noncomputable def total_surface_area_hemisphere (r : ℝ) : ℝ := 
  let base_area := π * r^2
  let curved_area := 2 * π * r^2
  base_area + curved_area

theorem hemisphere_surface_area_with_base (r : ℝ) (h : r = 8) : 
  total_surface_area_hemisphere r = 192 * π := by
  rw [h]
  unfold total_surface_area_hemisphere
  simp
  sorry

end hemisphere_surface_area_with_base_l554_554588


namespace f_2008_eq_neg1_over_2007_l554_554531

def f (x : ℝ) : ℝ := 1 / (1 - x)

def f_iter (k : ℕ) (x : ℝ) : ℝ :=
  Nat.recOn k (f x) (λ n acc, f acc)

theorem f_2008_eq_neg1_over_2007 : f_iter 2008 2008 = -1 / 2007 := 
  sorry

end f_2008_eq_neg1_over_2007_l554_554531


namespace max_sumo_wrestlers_l554_554337

/-- 
At a sumo wrestling tournament, 20 sumo wrestlers participated.
The average weight of the wrestlers is 125 kg.
Individuals weighing less than 90 kg cannot participate.
Prove that the maximum possible number of sumo wrestlers weighing more than 131 kg is 17.
-/
theorem max_sumo_wrestlers : 
  ∀ (weights : Fin 20 → ℝ), 
    (∀ i, weights i ≥ 90) → 
    (∑ i, weights i = 2500) → 
    (∃ n : ℕ,  n ≤ 20 ∧ 
      (n = 17 → (∑ i in Finset.filter (λ i, weights i > 131) Finset.univ).card = n) ∧ 
      ∀ m, m > 17 → (∀ j ∈ Finset.filter (λ i, weights i > 131) Finset.univ, m = (Finset.card (Finset.filter (λ i, weights i > 131) Finset.univ) + j) → False))
:= sorry

end max_sumo_wrestlers_l554_554337


namespace friendship_configurations_l554_554715

-- Defining the individual entities
inductive Individual
| Adam | Benin | Chiang | Deshawn | Esther | Fiona
deriving DecidableEq

open Individual

-- Defining friendship as a relation
def is_friend (a b : Individual) : Prop := sorry -- Friendship relation

-- There are 6 individuals
def individuals := [Adam, Benin, Chiang, Deshawn, Esther, Fiona]

-- Check if the list is a valid set of individuals
def valid_individuals (l : List Individual) : Prop :=
  ∀ x, x ∈ l

-- Each individual has same number of friends
def same_number_of_friends (n : ℕ) : Prop :=
  ∀ x : Individual, (l.countp (is_friend x)) = n

-- Determine the number of valid configurations
def number_of_valid_configurations : ℕ :=
  sorry

-- The theorem we need to prove
theorem friendship_configurations : number_of_valid_configurations = 170 :=
  sorry

end friendship_configurations_l554_554715


namespace lucky_numbers_l554_554688

def is_lucky (n : ℕ) : Prop :=
  ∃ q r : ℕ, 221 = n * q + r ∧ r % q = 0

def lucky_numbers_count (a b : ℕ) : ℕ :=
  Nat.card {n // a ≤ n ∧ n ≤ b ∧ is_lucky n}

theorem lucky_numbers (a b : ℕ) (h1 : a = 1) (h2 : b = 221) :
  lucky_numbers_count a b = 115 := 
sorry

end lucky_numbers_l554_554688


namespace probability_product_odd_l554_554183

theorem probability_product_odd :
  let range := setOf (λ x, 3 ≤ x ∧ x ≤ 20),
      n := set.card range,
      odd_elements := setOf (λ x, x ∈ range ∧ x % 2 = 1),
      n_odd := set.card odd_elements,
      total_combinations := nat.choose n 2,
      odd_combinations := nat.choose n_odd 2
  in total_combinations ≠ 0 → 
     (odd_combinations : ℚ) / total_combinations = 4 / 17 :=
by
  assume range n odd_elements n_odd total_combinations odd_combinations _,
  sorry

end probability_product_odd_l554_554183


namespace meetings_percentage_l554_554148

def workday_hours := 10
def first_meeting_minutes := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_workday_minutes := workday_hours * 60
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

theorem meetings_percentage :
    (total_meeting_minutes / total_workday_minutes) * 100 = 40 :=
by
  sorry

end meetings_percentage_l554_554148


namespace number_after_one_minute_is_not_54_l554_554152

theorem number_after_one_minute_is_not_54 :
  let n0 := 12 in
  let nt := 54 in
  let operations := 60 in
  (∀ (op : ℤ → ℤ), (∀ (x : ℤ), x = x * 2 ∨ x = x * 3 ∨ x = x / 2 ∨ x = x / 3)) →
  (n0 = 2^2 * 3^1) →
  (nt = 2^1 * 3^3) →
  nt ≠ ((λ n, (op^[operations] n)) n0) := by
  sorry

end number_after_one_minute_is_not_54_l554_554152


namespace pentagon_external_angle_l554_554585

theorem pentagon_external_angle (n : ℕ) (h1 : n = 5) :
  let exterior_angle := (360 : ℝ) / n in
  let intersection_angle := 360 - exterior_angle in
  intersection_angle = 288 :=
by
  sorry

end pentagon_external_angle_l554_554585


namespace find_s_value_l554_554365

noncomputable def value_of_s (s : ℝ) : Prop :=
  let v := (λ s => (1 + 5 * s, -2 + 3 * s, 4 - 2 * s))
  let a := (-3, 6, 7)
  -- Orthogonality condition
  let d := (λ s => (1 + 5 * s + 3, -2 + 3 * s - 6, 4 - 2 * s - 7))
  let direction := (5, 3, -2)
  (d s).fst * direction.fst + (d s).snd.fst * direction.snd.fst + (d s).snd.snd * direction.snd.snd = 0

theorem find_s_value : ∃ s, value_of_s s ∧ s = -1 / 19 := 
by
  use -1/19
  sorry

end find_s_value_l554_554365


namespace num_divisors_of_2002_l554_554062

theorem num_divisors_of_2002 : 
  (Nat.divisors 2002).length = 16 := 
sorry

end num_divisors_of_2002_l554_554062


namespace each_charity_gets_45_l554_554114

-- Define the conditions
def dozen := 12
def total_cookies := 6 * dozen
def price_per_cookie := 1.5
def cost_per_cookie := 0.25
def total_revenue := total_cookies * price_per_cookie
def total_cost := total_cookies * cost_per_cookie
def total_profit := total_revenue - total_cost

-- Define the expected outcome
def expected_each_charity_gets := 45

-- The theorem to prove
theorem each_charity_gets_45 :
  total_profit / 2 = expected_each_charity_gets :=
by
  sorry

end each_charity_gets_45_l554_554114


namespace angle_between_vectors_45_deg_vector_difference_magnitude_1_l554_554028

variables {a b : ℝ^2}

theorem angle_between_vectors_45_deg (ha : ‖a‖ = sqrt 2) (hb : ‖b‖ = 1) (dot_ab : a • b = 1) : 
  let θ := real.arccos ((a • b) / (‖a‖ * ‖b‖))
  in θ = real.pi / 4 :=
sorry

theorem vector_difference_magnitude_1 (ha : ‖a‖ = sqrt 2) (hb : ‖b‖ = 1) (θ : ℝ) (hθ : θ = real.pi / 4) : 
  ‖a - b‖ = 1 :=
sorry

end angle_between_vectors_45_deg_vector_difference_magnitude_1_l554_554028


namespace range_of_A_l554_554514

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f(x)

def strictly_increasing_on_pos (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 < x → x < y → f(x) < f(y)

theorem range_of_A (f : ℝ → ℝ) (A : ℝ) :
  odd_function f →
  strictly_increasing_on_pos f →
  f (1 / 2) = 0 →
  (0 <= A ∧ A <= π) →
  f (Real.cos A) < 0 ↔
  ((π / 3 < A ∧ A < π / 2) ∨ (2 * π / 3 < A ∧ A < π)) :=
by
  sorry

end range_of_A_l554_554514


namespace a_equals_1_or_2_l554_554056

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x : ℤ | x^2 - 3 * x < 0}
def non_empty_intersection (a : ℤ) : Prop := (M a ∩ N).Nonempty

theorem a_equals_1_or_2 (a : ℤ) (h : non_empty_intersection a) : a = 1 ∨ a = 2 := by
  sorry

end a_equals_1_or_2_l554_554056


namespace magnitude_angle_XYZ_l554_554992

theorem magnitude_angle_XYZ
  (X Y Z W : Type)
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z] [MetricSpace W]
  (h_right_ang : right_triangle_at X Y Z)
  (h_on_line : W ∈ line X Z)
  (h_XW_2WZ : dist X W = 2 * dist W Z)
  (h_XY_2YW : dist X Y = 2 * dist Y W) :
  angle_magnitude_at X Y Z = 10 := 
sorry

end magnitude_angle_XYZ_l554_554992


namespace volume_of_regular_tetrahedron_l554_554402

theorem volume_of_regular_tetrahedron (a : ℝ) (h : a = sqrt 2) :
  let S := (sqrt 3) / 2 in
  let H := (2 * sqrt 3) / 3 in
  let V := (1 / 3) * S * H in
  V / 4 = 1 / 12 :=
by
  intro S H V
  sorry

end volume_of_regular_tetrahedron_l554_554402


namespace measure_of_smallest_angle_l554_554331

-- Define basic properties and constants
def right_angle := 90
def large_angle := (1.4 : ℝ) * right_angle -- Angle that is 40% larger than a right angle
def isosceles_sum_of_angles := 180
def remaining_angles_sum := isosceles_sum_of_angles - large_angle -- Sum of the other two angles
def smallest_angle := remaining_angles_sum / 2 -- Measure of each of the equal smallest angles

-- Theorem stating the measure of one of the two smallest angles
theorem measure_of_smallest_angle (a b c : ℝ) (h1 : a = smallest_angle) :
  a = 27.0 :=
  by
    sorry

end measure_of_smallest_angle_l554_554331


namespace range_of_mn_squared_l554_554132

noncomputable def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f(x) < f(y)

theorem range_of_mn_squared (f : ℝ → ℝ)
  (h_inc : is_increasing f)
  (h_symm : ∀ x, f (1 - x) + f (1 + x) = 0)
  (m n x : ℝ)
  (h_fm_pos : f m > 0)
  (h_fn_nonpos : f n ≤ 0)
  (h_mn_le : m^2 + n^2 ≤ x^2) :
  13 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
sorry

end range_of_mn_squared_l554_554132


namespace congruent_triangle_exists_l554_554786

-- Definition of a triangle
structure Triangle (A B C : Type u) :=
  (length_AB : ℝ)
  (length_BC : ℝ)
  (length_CA : ℝ)

-- Declaration of the problem in Lean

theorem congruent_triangle_exists (A B C A' B' C' : Type u) (t1 : Triangle A B C)
  (seg1 : Triangle A' B' C') :
  (seg1.length_AB = t1.length_AB) ∧
  (seg1.length_BC = t1.length_BC) ∧
  (seg1.length_CA = t1.length_CA) ↔
  ((t1.length_AB = t1.length_AB) ∧
  (t1.length_BC = t1.length_BC) ∧
  (t1.length_CA = t1.length_CA) → 
  Triangle A' B' C'
  ) :=
by
  sorry

end congruent_triangle_exists_l554_554786


namespace simplify_f_of_alpha_value_of_f_given_cos_l554_554409

variable (α : Real) (f : Real → Real)

def third_quadrant (α : Real) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

noncomputable def f_def : Real → Real := 
  λ α => (Real.sin (α - Real.pi / 2) * 
           Real.cos (3 * Real.pi / 2 + α) * 
           Real.tan (Real.pi - α)) / 
           (Real.tan (-α - Real.pi) * 
           Real.sin (-Real.pi - α))

theorem simplify_f_of_alpha (h : third_quadrant α) :
  f α = -Real.cos α := sorry

theorem value_of_f_given_cos 
  (h : third_quadrant α) 
  (cos_h : Real.cos (α - 3 / 2 * Real.pi) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := sorry

end simplify_f_of_alpha_value_of_f_given_cos_l554_554409


namespace number_of_x_intercepts_l554_554442

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end number_of_x_intercepts_l554_554442


namespace five_digit_numbers_without_repeating_digits_with_one_even_between_odds_l554_554390

theorem five_digit_numbers_without_repeating_digits_with_one_even_between_odds :
  ∃ n : ℕ, n = 28 ∧
    ∃ (digits : Finset ℕ), 
    digits = {0, 1, 2, 3, 4} ∧
    ∀ (number : list ℕ), number.nodup ∧
    (∀ d ∈ number, d ∈ digits) ∧
    (∃ e : ℕ, e ∈ {0, 2, 4} ∧
      (∃ o1 o2 : ℕ, o1 ∈ {1, 3} ∧ o2 ∈ {1, 3} ∧
        (number = o1 :: e :: o2 :: (number.tail.tail).to_list ∨
         number = (number.init).to_list ++ [o1, e, o2]))) :=
begin
  existsi (28 : ℕ),
  split,
  { refl },
  existsi ({0, 1, 2, 3, 4} : Finset ℕ),
  split,
  { refl },
  sorry
end

end five_digit_numbers_without_repeating_digits_with_one_even_between_odds_l554_554390


namespace projection_area_relationship_l554_554863

noncomputable def S1 (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def S2 (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.2 - A.2) * (C.3 - A.3) - (C.2 - A.2) * (B.3 - A.3))

noncomputable def S3 (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.3 - A.3) * (C.1 - A.1) - (C.3 - A.3) * (B.1 - A.1))

theorem projection_area_relationship :
  let A := (2, 0, 0)
  let B := (2, 2, 0)
  let C := (0, 2, 0)
  let D := (1, 1, real.sqrt 2)
  S3 A B C D = S2 A B C D ∧ S3 A B C D ≠ S1 A B C D :=
by
  sorry

end projection_area_relationship_l554_554863


namespace scrap_cookie_radius_l554_554932

theorem scrap_cookie_radius :
  (∀ R : ℝ, 
    R = 4 → 
    (∀ r : ℝ, 
      r = 1 → 
      (∀ total_small_cookies : ℕ, 
        total_small_cookies = 10 →
        (∀ arrangement_optimal : Prop, 
          arrangement_optimal → 
          (∀ thickness_maintained : Prop, 
            thickness_maintained → 
            (∃ r_scrap : ℝ, 
              r_scrap = real.sqrt 6))))))) :=
by
  sorry

end scrap_cookie_radius_l554_554932


namespace calculate_expression_l554_554728

theorem calculate_expression :
  (-3 : ℝ) ^ 2 - sqrt 4 + (1/2) ^ (-1 : ℝ) = 9 :=
by 
  sorry

end calculate_expression_l554_554728


namespace actual_distance_between_towns_l554_554566

theorem actual_distance_between_towns
  (d_map : ℕ) (scale1 : ℕ) (scale2 : ℕ) (distance1 : ℕ) (distance2 : ℕ) (remaining_distance : ℕ) :
  d_map = 9 →
  scale1 = 10 →
  distance1 = 5 →
  scale2 = 8 →
  remaining_distance = d_map - distance1 →
  d_map = distance1 + remaining_distance →
  (distance1 * scale1 + remaining_distance * scale2 = 82) := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end actual_distance_between_towns_l554_554566


namespace B_pow_2023_eq_l554_554882

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [Real.cos (Real.pi / 4), 0, -Real.sin (Real.pi / 4)],
    [0, 1, 0],
    [Real.sin (Real.pi / 4), 0, Real.cos (Real.pi / 4)]
  ]

noncomputable def B_2023 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [Real.sqrt 2 / 2, 0, Real.sqrt 2 / 2],
    [0, 1, 0],
    [-Real.sqrt 2 / 2, 0, Real.sqrt 2 / 2]
  ]

theorem B_pow_2023_eq : B^2023 = B_2023 := by
  -- the proof would go here
  sorry

end B_pow_2023_eq_l554_554882


namespace firstTenNiceNumbersSumEq90_l554_554724

def isProperDivisor (d n : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def properDivisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => isProperDivisor d n)

def sumProperDivisors (n : ℕ) : ℕ :=
  (properDivisors n).sum id

def isNice (n : ℕ) : Prop :=
  sumProperDivisors n = n

def firstTenNiceNumbers : List ℕ :=
  [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

def sumFirstTenNiceNumbers : ℕ :=
  firstTenNiceNumbers.sum

theorem firstTenNiceNumbersSumEq90 : sumFirstTenNiceNumbers = 90 := by
  sorry

end firstTenNiceNumbersSumEq90_l554_554724


namespace lucky_numbers_count_l554_554657

def isLucky (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≠ 0 ∧ 221 = n * k + (221 % n) ∧ (221 % n) % k = 0

def countLuckyNumbers : ℕ :=
  Nat.card (Finset.filter isLucky (Finset.range 222))

theorem lucky_numbers_count : countLuckyNumbers = 115 :=
  sorry

end lucky_numbers_count_l554_554657


namespace count_lucky_numbers_l554_554652

def is_lucky (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 221) ∧ (221 % n % (221 / n) = 0)

theorem count_lucky_numbers : (finset.filter is_lucky (finset.range 222)).card = 115 :=
by
  sorry

end count_lucky_numbers_l554_554652


namespace log_base_one_fifth_twenty_five_l554_554755

theorem log_base_one_fifth_twenty_five : log (1/5) 25 = -2 :=
by
  sorry

end log_base_one_fifth_twenty_five_l554_554755


namespace lucky_numbers_count_l554_554674

def is_lucky (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧ (221 % n) % (221 / n) = 0

def count_lucky_numbers : ℕ :=
  (Finset.range 222).filter is_lucky |>.card

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
by
  sorry

end lucky_numbers_count_l554_554674


namespace number_of_babies_in_nest_l554_554539

theorem number_of_babies_in_nest
    (worms_per_baby_per_day : ℕ)
    (papa_bird_worms : ℕ)
    (mama_bird_worms : ℕ)
    (stolen_worms : ℕ)
    (extra_worms_needed : ℕ)
    (total_days : ℕ)
    (total_worms_per_baby : ℕ)
    : ∃ (babies : ℕ), 
        worms_per_baby_per_day = 3 ∧
        papa_bird_worms = 9 ∧
        mama_bird_worms = 13 ∧
        stolen_worms = 2 ∧
        extra_worms_needed = 34 ∧
        total_days = 3 ∧
        total_worms_per_baby = 9 ∧
        babies * total_worms_per_baby = (papa_bird_worms + (mama_bird_worms - stolen_worms) + extra_worms_needed) :=
begin
    sorry
end

end number_of_babies_in_nest_l554_554539


namespace lucky_numbers_count_l554_554691

def is_lucky (n : ℕ) : Prop := 
  ∃ (k q r : ℕ), 
    1 ≤ n ∧ n ≤ 221 ∧
    221 = n * q + r ∧ 
    r = k * q ∧ 
    r < n

def count_lucky_numbers : ℕ := 
  ∑ n in (finset.range 222).filter is_lucky, 1

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
  by 
    sorry

end lucky_numbers_count_l554_554691


namespace infinite_non_n_good_polynomials_l554_554387

theorem infinite_non_n_good_polynomials : 
  ∀ (n : ℕ), ∃ (F : ℤ[X]), (∀ (c : ℕ), c > 0 → F.eval c > 0) ∧ (F.eval 0 = 1) ∧ (¬ (∃ (A : ℤ), A ≠ 0 ∧ ∃ (c_set : Finset ℕ), c_set.card = n ∧ ∀ (c ∈ c_set), Nat.Prime (F.eval c))) := 
by
  sorry

end infinite_non_n_good_polynomials_l554_554387


namespace circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l554_554036

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 10

theorem circle_through_points (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (h_center : ∃ C: ℝ × ℝ, C.1 - C.2 - 4 = 0 ∧ (circle_eq C.1 C.2)) : ∀ x y, circle_eq x y ↔ (x - 3) ^ 2 + (y + 1) ^ 2 = 10 := 
by sorry

theorem line_perpendicular_and_tangent_to_circle (line_slope : ℝ) (tangent : ∀ x y, circle_eq x y → (x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0)) : ∀ x, x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0 :=
by sorry

theorem max_area_triangle (A B P : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (hP : circle_eq P.1 P.2) : ∃ area : ℝ, area = 5 + 5 * Real.sqrt 2
:= 
by sorry

end circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l554_554036


namespace surface_area_of_sphere_of_tetrahedron_M_ABC_l554_554466

-- Definitions of the given problem

structure Triangle :=
  (A B C : Type)
  (angle_C : ℝ)
  (angle_B : ℝ)
  (side_AC : ℝ)

def midpoint {X : Type} (A B : X) : X := A -- Midpoint should be more precisely defined in practice

variables (A B C M : Type) (t : Triangle)

-- Given conditions
def conditions : Prop :=
  t.angle_C = π / 2 ∧
  t.angle_B = π / 6 ∧
  t.side_AC = 2 ∧
  M = midpoint A B ∧
  dist A B = 2 * sqrt 2

-- The goal is to prove the surface area of the circumscribed sphere of tetrahedron M-ABC
theorem surface_area_of_sphere_of_tetrahedron_M_ABC (A B C M : Type) (t : Triangle) (h: conditions A B C M t) : 
  surface_area_of_sphere t M A B C = 16 * π := by sorry

end surface_area_of_sphere_of_tetrahedron_M_ABC_l554_554466


namespace cubic_polynomial_given_conditions_l554_554886

theorem cubic_polynomial_given_conditions (Q : ℚ[X])
  (h0 : Q.eval 0 = m)
  (h1 : Q.eval 1 = 3 * m)
  (hm1 : Q.eval (-1) = 4 * m) : 
  Q.eval 2 + Q.eval (-2) = 22 * m := 
sorry

end cubic_polynomial_given_conditions_l554_554886


namespace prob_of_interval_l554_554417

noncomputable def X : Type := ℝ

def normal_dist (μ σ : ℝ) : ProbabilityDistribution X :=
sorry -- The exact details of the normal distribution are abstracted.

def X_distribution := normal_dist 3 σ

axiom P_lt (a : ℝ) : ℝ :=
sorry -- Assume this axiom gives the probability that X < a.

axiom P_interval (a b : ℝ) : ℝ :=
sorry -- Assume this axiom gives the probability that a < X < b.

axiom condition1 : X_distribution = normal_dist 3 σ
axiom condition2 : P_lt 5 = 0.8

theorem prob_of_interval : P_interval 1 3 = 0.3 :=
by
  -- Using the given conditions and properties of the normal distribution,
  -- we can derive the following result:
  sorry

end prob_of_interval_l554_554417


namespace log4_statements_incorrect_l554_554892

theorem log4_statements_incorrect (x : ℝ) (h : true) : 
  (x = 1 → ∃ y : ℝ, y = 0 ∧ y = log 4 1) ∧ 
  (x = 4 → ∃ y : ℝ, y = 1 ∧ y = log 4 4) ∧ 
  (x = -4 → ∃ y : ℂ, y.im ≠ 0 ∧ y = log 4 (-4)) ∧ 
  ((0 < x ∧ x < 1) → ∃ y : ℝ, y < 0 ∧ ∀ ε > 0, ∃ δ > 0, x < δ → y < -ε) → 
  ¬ (∃ y : ℝ, y = 0 ∧ (x = 1 → y = log 4 1) ∧ (x = 4 → y = log 4 4) ∧ 
     (∀ y : ℂ, y.im ≠ 0 ∧ (x = -4 → y = log 4 (-4))) ∧ 
     ((0 < x ∧ x < 1) → y < 0 ∧ ∀ ε > 0, ∃ δ > 0, x < δ → y < -ε)) :=
begin
  sorry
end

end log4_statements_incorrect_l554_554892


namespace sequence_general_term_l554_554212

theorem sequence_general_term (a : ℕ → ℕ) (h : ∀ n : ℕ, a 1 + 3 * a 2 + 5 * a 3 + ... + (2 * n - 1) * a n = (n - 1) * 3^(n + 1) + 3) :
  ∀ n : ℕ, a n = 3^n :=
sorry

end sequence_general_term_l554_554212


namespace angle_sum_ratios_l554_554903

theorem angle_sum_ratios (A B C D E : Type) [IsConvexQuadrilateral A B C D E] :
  let S  := degree_sum (∠ CDE) (∠ DCE)
  let S' := degree_sum (∠ BAD) (∠ ABC)
  r = S / S' := 1 :=
by sorry

end angle_sum_ratios_l554_554903


namespace centroid_of_weighted_triangle_l554_554860

theorem centroid_of_weighted_triangle (A B C : Point) (q1 q2 q3 : ℝ) (h1 : q1 = 1) (h2 : q2 = 2) (h3 : q3 = 6) : 
  ∃ S : Point, is_centroid S A B C :=
by sorry

end centroid_of_weighted_triangle_l554_554860


namespace center_temperature_l554_554826

-- Define the conditions as a structure
structure SquareSheet (f : ℝ × ℝ → ℝ) :=
  (temp_0: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 0) = 0 ∧ f (0, x) = 0 ∧ f (1, x) = 0)
  (temp_100: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 1) = 100)
  (no_radiation_loss: True) -- Just a placeholder since this condition is theoretical in nature

-- Define the claim as a theorem
theorem center_temperature (f : ℝ × ℝ → ℝ) (h : SquareSheet f) : f (0.5, 0.5) = 25 :=
by
  sorry -- Proof is not required and skipped

end center_temperature_l554_554826


namespace perimeter_of_bisected_squares_l554_554179

theorem perimeter_of_bisected_squares :
  let side := 2 in
  let numSquares := 4 in
  let newPerimeter :=
    let horizontalSegments := 3 * side in
    let verticalSegments := 2 * (2 * side) in
    let diagonalContributions := numSquares * (side * Real.sqrt 2) / 2 in
    horizontalSegments + verticalSegments + diagonalContributions
  in
  newPerimeter = 14 + 4 * Real.sqrt 2 :=
sorry

end perimeter_of_bisected_squares_l554_554179


namespace find_x_set_l554_554144

noncomputable def f (x : ℝ) : ℝ :=
if x < -1 then 2 * x - 3
else if x <= 1 then 1
else 2 * x - 3

theorem find_x_set :
  { x : ℝ | f(f(x)) = 1 } = { x | -1 ≤ x ∧ x ≤ 2 } ∪ { x | x = 5/2 } :=
by sorry

end find_x_set_l554_554144


namespace zero_points_ordering_l554_554052

noncomputable def f (x : ℝ) : ℝ := x + 2^x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) : ℝ := x^3 + x - 2

theorem zero_points_ordering :
  ∃ x1 x2 x3 : ℝ,
    f x1 = 0 ∧ x1 < 0 ∧ 
    g x2 = 0 ∧ 0 < x2 ∧ x2 < 1 ∧
    h x3 = 0 ∧ 1 < x3 ∧ x3 < 2 ∧
    x1 < x2 ∧ x2 < x3 := sorry

end zero_points_ordering_l554_554052


namespace second_person_days_l554_554633

theorem second_person_days (P1 P2 : ℝ) (h1 : P1 = 1 / 24) (h2 : P1 + P2 = 1 / 8) : 1 / P2 = 12 :=
by
  sorry

end second_person_days_l554_554633


namespace x_intercepts_count_l554_554445

def parabola_x (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem x_intercepts_count : 
  (∃ y : ℝ, parabola_x y = 0) → 1 := sorry

end x_intercepts_count_l554_554445


namespace cos_sequence_l554_554401

noncomputable def sequence (a : Nat → ℝ) := 
∀ (n : ℕ), a (n + 2) = 2 * a (n + 1) - a n

theorem cos_sequence (a : ℕ → ℝ) 
  (h_seq : sequence a) 
  (h_sum : a 3 + a 8 + a 13 = Real.pi) :
  Real.cos (a 7 + a 9 + Real.pi) = 1 / 2 := 
sorry

end cos_sequence_l554_554401


namespace part1_part2_part3_l554_554508

variable (M : ℕ → ℤ)
axiom M_def : ∀ n, M n = (-2)^n

theorem part1 : M 5 + M 6 = 32 := by
  rw [M_def, M_def]
  change (-2)^5 + (-2)^6 = 32
  norm_num
  rfl

theorem part2 : 2 * M 2015 + M 2016 = 0 := by
  rw [M_def, M_def, M_def]
  change 2 * (-2)^2015 + (-2)^2016 = 0
  norm_num
  rfl

theorem part3 (n : ℕ) : 2 * M n + M (n + 1) = 0 := by
  rw [M_def, M_def, M_def]
  change 2 * (-2)^n + (-2)^(n + 1) = 0
  norm_num
  rfl

#print axioms part1
#print axioms part2
#print axioms part3

end part1_part2_part3_l554_554508


namespace clock_angle_at_3_22_l554_554968

noncomputable def minute_hand_angle (minutes : ℕ) : ℝ :=
  (minutes / 60.0) * 360.0

noncomputable def hour_hand_angle (hours minutes : ℕ) : ℝ :=
  (hours / 12.0) * 360.0 + (minutes / 60.0) * 30.0

lemma abs_sub (a b : ℝ) : ℝ :=
  abs (a - b)

theorem clock_angle_at_3_22 : abs_sub (hour_hand_angle 3 22) (minute_hand_angle 22) = 35 :=
by
  sorry

end clock_angle_at_3_22_l554_554968


namespace probability_of_solution_l554_554529

noncomputable def S : set ℝ :=
  {-12, -10, -9.5, -6, -5, -4, -2.5, -1, -0.5, 0, 2.5, 4, 6, 7, 10, 12, 14, 3/2}

theorem probability_of_solution :
  (S.filter (λ x, (x + 5) * (x + 10) * (2 * x - 5) * (x^2 - 2 * x + 3) = 0)).card / S.card = 1 / 6 :=
by
  sorry

end probability_of_solution_l554_554529


namespace midpoint_statement_trisection_statement_l554_554228

variables {A B C D E F : Type}
variables [LinearOrderedField A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup F]

-- Definitions based on conditions
def line_segment_eq (p q : A) : A := q - p
def is_midpoint (p q r : A) : Prop := line_segment_eq p q = line_segment_eq q r

def is_trisection_point (p q r : A) : Prop :=
  line_segment_eq p q = (line_segment_eq p r) / 3

-- Theorem statements based on questions and correct answers
theorem midpoint_statement (A B C : A) (h : line_segment_eq A B = line_segment_eq B C) :
  is_midpoint A B C := sorry

theorem trisection_statement (D E F : A) (x : A) (h : line_segment_eq E D = x) (h' : is_trisection_point D E F) :
  line_segment_eq E F = 3 * x := sorry

end midpoint_statement_trisection_statement_l554_554228


namespace solution_set_of_inequality_l554_554215

theorem solution_set_of_inequality (x : ℝ) : 
  (|x+1| - |x-4| > 3) ↔ x > 3 :=
sorry

end solution_set_of_inequality_l554_554215


namespace lucky_numbers_l554_554686

def is_lucky (n : ℕ) : Prop :=
  ∃ q r : ℕ, 221 = n * q + r ∧ r % q = 0

def lucky_numbers_count (a b : ℕ) : ℕ :=
  Nat.card {n // a ≤ n ∧ n ≤ b ∧ is_lucky n}

theorem lucky_numbers (a b : ℕ) (h1 : a = 1) (h2 : b = 221) :
  lucky_numbers_count a b = 115 := 
sorry

end lucky_numbers_l554_554686


namespace marigolds_total_l554_554547

theorem marigolds_total (a : ℕ)
  (h1 : a = 14)
  (h2 : ∀ n, 0 < n → marigolds_sold (n + 1) = 2 * marigolds_sold n) :
  ∑ i in finset.range 7, marigolds_sold i = 1778 := 
by
  sorry

noncomputable def marigolds_sold : ℕ → ℕ 
| 0       := 14
| (n + 1) := 2 * marigolds_sold n

end marigolds_total_l554_554547


namespace simplest_quadratic_radical_l554_554252

theorem simplest_quadratic_radical (a x : ℝ) :
  ∀ (radical : ℝ), (radical = sqrt (a^2 + 4) ∨ radical = sqrt (1 / 2) ∨ radical = sqrt (3x^2) ∨ radical = sqrt 0.3) ->
  (radical = sqrt (a^2 + 4)) = (radical = sqrt (a^2 + 4)) :=
by
  intro radical h
  cases h
  case inl h_1 {
    exact rfl
  }
  case inr h_1 {
    cases h_1
    case inl h_2 {
      sorry
    }
    case inr h_2 {
      cases h_2
      case inl h_3 {
        sorry
      }
      case inr h_3 {
        sorry
      }
    }
  }

end simplest_quadratic_radical_l554_554252


namespace each_charity_gets_45_dollars_l554_554115

def dozens : ℤ := 6
def cookies_per_dozen : ℤ := 12
def total_cookies : ℤ := dozens * cookies_per_dozen
def selling_price_per_cookie : ℚ := 1.5
def cost_per_cookie : ℚ := 0.25
def profit_per_cookie : ℚ := selling_price_per_cookie - cost_per_cookie
def total_profit : ℚ := profit_per_cookie * total_cookies
def charities : ℤ := 2
def amount_per_charity : ℚ := total_profit / charities

theorem each_charity_gets_45_dollars : amount_per_charity = 45 := 
by
  sorry

end each_charity_gets_45_dollars_l554_554115


namespace find_possible_k_values_l554_554867

noncomputable theory
open_locale classical

def line_through_point (k b : ℤ) : Prop :=
  ∃ (x y : ℤ), (x = -1 ∧ y = 2020 ∧ y = k * x + b)

def parabola_properties (a c k: ℤ) : Prop :=
  ∃ (x y : ℤ), (a ≠ 0 ∧ c = -1 - 2020 / k ∧ y = a * (x - c)^2)

def valid_k_values (k: ℤ) : Prop := 
  k = -404 ∨ k = -1010

theorem find_possible_k_values : 
  (∃ b : ℤ, line_through_point k b) ∧
  (∀ a c, parabola_properties a c k) → valid_k_values k :=
sorry

end find_possible_k_values_l554_554867


namespace lucky_numbers_count_l554_554692

def is_lucky (n : ℕ) : Prop := 
  ∃ (k q r : ℕ), 
    1 ≤ n ∧ n ≤ 221 ∧
    221 = n * q + r ∧ 
    r = k * q ∧ 
    r < n

def count_lucky_numbers : ℕ := 
  ∑ n in (finset.range 222).filter is_lucky, 1

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
  by 
    sorry

end lucky_numbers_count_l554_554692


namespace pair_of_operations_equal_l554_554251

theorem pair_of_operations_equal :
  (-3) ^ 3 = -(3 ^ 3) ∧
  (¬((-2) ^ 4 = -(2 ^ 4))) ∧
  (¬((3 / 2) ^ 2 = (2 / 3) ^ 2)) ∧
  (¬(2 ^ 3 = 3 ^ 2)) :=
by 
  sorry

end pair_of_operations_equal_l554_554251


namespace quadrilateral_pyramid_volume_l554_554767

theorem quadrilateral_pyramid_volume (h Q : ℝ) : 
  ∃ V : ℝ, V = (2 / 3 : ℝ) * h * (Real.sqrt (h^2 + 4 * Q^2) - h^2) :=
by
  sorry

end quadrilateral_pyramid_volume_l554_554767


namespace count_distinct_products_of_divisors_36000_l554_554125

-- Define the set of divisors of 36000
def is_divisor (n : ℕ) : Prop := 36000 % n = 0

-- Define the set T of all positive integer divisors of 36000
def T : Finset ℕ := 
  (Finset.range (36000 + 1)).filter is_divisor

-- Count the number of products of two distinct elements of T
def count_distinct_products (T : Finset ℕ) : ℕ :=
  T.bUnion (λ x, T.filter (λ y, x < y)).card

-- Statement of the problem
theorem count_distinct_products_of_divisors_36000 :
  count_distinct_products T = 382 :=
sorry

end count_distinct_products_of_divisors_36000_l554_554125


namespace surface_area_of_sphere_l554_554418

-- Given conditions
variables {O A B C : Point}
variables (h₁ : cos A = (2 * real.sqrt 2) / 3)
variables (h₂ : dist B C = 1)
variables (h₃ : dist A C = 3)
variables (h₄ : volume (tetrahedron_points O A B C) = real.sqrt 14 / 6)

-- Statement to prove
theorem surface_area_of_sphere
:
  surface_area (sphere_radius (dist O A)) = 16 * π :=
sorry

end surface_area_of_sphere_l554_554418


namespace inequality_proof_l554_554266

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 :=
by
  sorry

end inequality_proof_l554_554266


namespace joe_first_lift_weight_l554_554982

variable (x y : ℕ)

def conditions : Prop :=
  (x + y = 1800) ∧ (2 * x = y + 300)

theorem joe_first_lift_weight (h : conditions x y) : x = 700 := by
  sorry

end joe_first_lift_weight_l554_554982


namespace time_to_finish_furniture_l554_554269

-- Define the problem's conditions
def chairs : ℕ := 7
def tables : ℕ := 3
def minutes_per_piece : ℕ := 4

-- Define total furniture
def total_furniture : ℕ := chairs + tables

-- Define the function to calculate total time
def total_time (pieces : ℕ) (time_per_piece: ℕ) : ℕ :=
  pieces * time_per_piece

-- Theorem statement to be proven
theorem time_to_finish_furniture : total_time total_furniture minutes_per_piece = 40 := 
by
  -- Provide a placeholder for the proof
  sorry

end time_to_finish_furniture_l554_554269


namespace range_of_x0_l554_554034

noncomputable def circle (x y : ℝ) := x^2 + y^2 = 3
noncomputable def line (x y : ℝ) := x + 3 * y - 6 = 0

theorem range_of_x0 :
  (∃ P : ℝ × ℝ, P ∈ line ∧ (∃ Q : ℝ × ℝ, Q ∈ circle ∧ ∠ O P Q = 60)) →
  ∀ x0 y0 : ℝ, line x0 y0 → 
  (0 ≤ x0 ∧ x0 ≤ 6/5) :=
sorry

end range_of_x0_l554_554034


namespace probability_contemporaries_l554_554238

theorem probability_contemporaries (total_years : ℕ) (life_span : ℕ) (born_range : ℕ)
  (h1 : total_years = 300)
  (h2 : life_span = 80)
  (h3 : born_range = 300) :
  (∃λ p : ℚ, p = 104 / 225 ∧ 
   let lines_intersect := (0, 80) :: (80, 0) :: (220, 300) :: (300, 220) :: []
   in lines_intersect ≠ [] ∧ lines_intersect.length = 4 
   ∧ region_area 0 300 300 (λ x y, (y ≥ x - life_span) ∧ (y ≤ x + life_span)) = 41600
   ∧ total_area 0 300 300 = 90000
   ∧ prob := region_area / total_area,
     prob = p) :=
by sorry

end probability_contemporaries_l554_554238


namespace center_of_symmetry_max_min_values_l554_554439

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos (Real.pi - x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + 1

theorem center_of_symmetry (k : ℤ) : 
  ∃ x : ℝ, x = k * (Real.pi / 2) + (Real.pi / 8) ∧ f x = 0 := sorry

theorem max_min_values : 
  ∃ x_max x_min : ℝ, 
    x_max ∈ set.Icc (0 : ℝ) (Real.pi / 2) ∧ 
    x_min ∈ set.Icc (0 : ℝ) (Real.pi / 2) ∧ 
    f x_max = Real.sqrt 2 ∧ 
    f x_min = -1 ∧ 
    x_max = 3 * (Real.pi / 8) ∧ 
    x_min = 0 := sorry

end center_of_symmetry_max_min_values_l554_554439


namespace certain_number_value_l554_554572

theorem certain_number_value
  (x : ℝ)
  (y : ℝ)
  (h1 : (28 + x + 42 + 78 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  y = 104 :=
by
  -- Proof goes here
  sorry

end certain_number_value_l554_554572


namespace find_wall_width_l554_554233

-- Define the volume of one brick
def volume_of_one_brick : ℚ := 100 * 11.25 * 6

-- Define the total number of bricks
def number_of_bricks : ℕ := 1600

-- Define the volume of all bricks combined
def total_volume_of_bricks : ℚ := volume_of_one_brick * number_of_bricks

-- Define dimensions of the wall
def wall_height : ℚ := 800 -- in cm (since 8 meters = 800 cm)
def wall_depth : ℚ := 22.5 -- in cm

-- Theorem to prove the width of the wall
theorem find_wall_width : ∃ width : ℚ, total_volume_of_bricks = wall_height * width * wall_depth ∧ width = 600 :=
by
  -- skipping the actual proof
  sorry

end find_wall_width_l554_554233


namespace range_of_a_l554_554791

-- Define the equation and the interval
def equation (a x : ℝ) : Prop := 2 * x^2 + a * x - a^2 = 0

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a (a : ℝ) :
  (¬∃ x ∈ Icc (-1 : ℝ) (1 : ℝ), equation a x) ∧
  (∀ x, ¬inequality a x ∨ ∃ y, y ≠ x ∧ inequality a y) →
  a > 2 ∨ a < -2 := by
  sorry

end range_of_a_l554_554791


namespace num_children_with_dogs_only_l554_554470

-- Defining the given values and constants
def total_children : ℕ := 30
def children_with_cats : ℕ := 12
def children_with_dogs_and_cats : ℕ := 6

-- Define the required proof statement
theorem num_children_with_dogs_only : 
  ∃ (D : ℕ), D + children_with_dogs_and_cats + (children_with_cats - children_with_dogs_and_cats) = total_children ∧ D = 18 :=
by
  sorry

end num_children_with_dogs_only_l554_554470


namespace students_per_table_correct_l554_554286

-- Define the number of tables and students
def num_tables := 34
def num_students := 204

-- Define x as the number of students per table
def students_per_table := 6

-- State the theorem
theorem students_per_table_correct : num_students / num_tables = students_per_table :=
by
  sorry

end students_per_table_correct_l554_554286


namespace trajectory_of_A_l554_554874

theorem trajectory_of_A (A B C : (ℝ × ℝ)) (x y : ℝ) : 
  B = (-2, 0) ∧ C = (2, 0) ∧ (dist A (0, 0) = 3) → 
  (x, y) = A → 
  x^2 + y^2 = 9 ∧ y ≠ 0 := 
sorry

end trajectory_of_A_l554_554874


namespace find_x_l554_554455

theorem find_x (x y : ℝ) (h1 : x ≠ 0)
  (h2 : x / 3 = y^2 + 1)
  (h3 : x / 5 = 5 * y) :
  x = (625 + 25 * Real.sqrt 589) / 6 :=
by
  sorry

end find_x_l554_554455


namespace part1_part2_l554_554792

noncomputable def A : set ℝ := {x | x^2 - 2*x - 3 < 0}
noncomputable def B (m : ℝ) : set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

theorem part1 : A ∩ B 0 = {x | 1 ≤ x ∧ x < 3} :=
sorry

theorem part2 : (∀ x, x ∈ A → x ∈ B m) ∧ ¬(∀ x, x ∈ B m → x ∈ A) ↔ m ∈ (-∞, -2] ∪ [4, +∞) :=
sorry

end part1_part2_l554_554792


namespace trapezoid_diagonals_l554_554917

theorem trapezoid_diagonals (a c b d e f : ℝ) (h1 : a ≠ c):
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧ 
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := 
by
  sorry

end trapezoid_diagonals_l554_554917


namespace number_of_squares_in_grid_l554_554399

theorem number_of_squares_in_grid :
  ∀ (rows cols : ℕ), 
  rows = 5 → cols = 6 →
  ∑ i in range (min rows cols), (rows - i) * (cols - i) = 40 :=
by
  intros rows cols h_rows h_cols
  rw [h_rows, h_cols]
  have h_min: (min rows cols) = 5 := by simp [min_eq_left]
  rw [h_min]
  repeat { rw ← add_comm }
  sorry

end number_of_squares_in_grid_l554_554399


namespace total_suitcases_l554_554538

-- Definitions based on the conditions in a)
def siblings : ℕ := 4
def suitcases_per_sibling : ℕ := 2
def parents : ℕ := 2
def suitcases_per_parent : ℕ := 3
def suitcases_per_Lily : ℕ := 0

-- The statement to be proved
theorem total_suitcases : (siblings * suitcases_per_sibling) + (parents * suitcases_per_parent) + suitcases_per_Lily = 14 :=
by
  sorry

end total_suitcases_l554_554538


namespace unique_set_of_numbers_l554_554164

-- Definition of the sequence x_i
def x_i (n : ℕ) (i : ℕ) : ℝ := 1 - i / (n + 1)

-- The statement we need to prove
theorem unique_set_of_numbers (n : ℕ) (h : n > 0) :
  (finset.range n).sum (λ i, let xi := x_i n (i + 1) in
    if i = 0 then (1 - xi) ^ 2 else (x_i n i - xi) ^ 2) + (x_i n n) ^ 2 = 1 / (n + 1) :=
by
  sorry

end unique_set_of_numbers_l554_554164


namespace combinatorial_even_sum_l554_554849

theorem combinatorial_even_sum :
  (number_of_ways_to_select_even_sum {1, 2, 3, 4, 5, 6, 7, 8, 9} = 66) :=
sorry

end combinatorial_even_sum_l554_554849


namespace minimum_difference_of_composites_sum_100_l554_554607

-- Definition of composite numbers
def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

def min_positive_diff_composite : ℕ :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = 100 ∧ ∀ c d : ℕ, is_composite c ∧ is_composite d ∧ c + d = 100 → (abs (c - d) ≥ 4)

theorem minimum_difference_of_composites_sum_100 : min_positive_diff_composite :=
by
  existsi 48
  existsi 52
  split
  -- condition 1: 48 is composite
  use [2, 24]
  split
  { exact dec_trivial }
  split
  { exact dec_trivial }
  exact dec_trivial
  split
  -- condition 2: 52 is composite
  use [4, 13]
  split
  { exact dec_trivial }
  split
  { exact dec_trivial }
  exact dec_trivial
  split
  -- condition 3: sum is 100
  exact dec_trivial
  -- proving minimum difference 4
  intros c d h_composite_c h_composite_d h_sum_cd
  have h : abs (c - d) ≥ 4 := sorry
  exact h

end minimum_difference_of_composites_sum_100_l554_554607


namespace log_identity_proof_l554_554995

theorem log_identity_proof : 2 * log 5 10 + log 5 0.25 = 2 := 
by 
  sorry

end log_identity_proof_l554_554995


namespace find_ab_l554_554796

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 100) : a * b = -3 :=
by
sorry

end find_ab_l554_554796


namespace quadrilateral_area_proof_l554_554726

open EuclideanGeometry

noncomputable def area_of_quadrilateral (A B C D : Point) : ℝ :=
  (1 / 2) * | (A.x * B.y + B.x * C.y + C.x * D.y + D.x * A.y) -
             (A.y * B.x + B.y * C.x + C.y * D.x + D.y * A.x) |

theorem quadrilateral_area_proof :
  let A := (0, 0)
  let B := (4, 0)
  let C := (2, 4)
  let D := (4, 6)
  area_of_quadrilateral A B C D = 6 := by
    sorry

end quadrilateral_area_proof_l554_554726


namespace max_value_of_f_range_of_a_unique_value_of_m_l554_554905

-- Define the function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - b * x

-- Part 1: When a = b = 1/2, prove that the maximum value of f(x) is -3/4
theorem max_value_of_f : 
  ∀ x > 0, f x (1/2) (1/2) = Real.log x - (1/4) * x^2 - (1/2) * x → f 1 (1/2) (1/2) = -3/4 :=
sorry

-- Define the function F
def F (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := f x a b + (1/2) * x^2 + b * x + (a / x)

-- Part 2: If the slope of the tangent line k at any point P(x₀, y₀) on the graph is always ≤ 1/2, find the range of a
theorem range_of_a :
  ∀ x₀ > 0, x₀ ≤ 3 → (x₀ - a) / x₀^2 ≤ (1/2) → a >= (1/2) :=
sorry

-- Part 3: When a = 0 and b = -1, prove the unique value of m is 1/2 for the given equation x^2 = 2m f(x)
theorem unique_value_of_m :
  ∃ x > 0, x^2 = 2 * (1/2) * (Real.log x + x) → x^2 = 2 * m * (Real.log x + x) ∧ m > 0 ∧ ∃! x, x² = 2m (f x 0 (-1)) → m = (1/2) :=
sorry

end max_value_of_f_range_of_a_unique_value_of_m_l554_554905


namespace hyperbola_foci_coordinates_l554_554193

theorem hyperbola_foci_coordinates :
  (a^2 = 7) → (b^2 = 3) → (c^2 = a^2 + b^2) → (c = Real.sqrt c^2) →
  ∃ (x y : ℝ), (x = Real.sqrt 10 ∧ y = 0) ∨ (x = -Real.sqrt 10 ∧ y = 0) :=
by
  intros a2_eq b2_eq c2_eq c_eq
  have h1 : c2 = 10 := by rw [a2_eq, b2_eq, add_comm]
  have h2 : c = Real.sqrt 10 := by rw [h1, Real.sq_sqrt (show 0 ≤ 10 by norm_num)]
  use (Real.sqrt 10)
  use 0
  use (-Real.sqrt 10)
  use 0
  sorry

end hyperbola_foci_coordinates_l554_554193


namespace work_speed_ratio_l554_554999

open Real

theorem work_speed_ratio (A B : Type) 
  (A_work_speed B_work_speed : ℝ) 
  (combined_work_time : ℝ) 
  (B_work_time : ℝ)
  (h_combined : combined_work_time = 12)
  (h_B : B_work_time = 36)
  (combined_speed : A_work_speed + B_work_speed = 1 / combined_work_time)
  (B_speed : B_work_speed = 1 / B_work_time) :
  A_work_speed / B_work_speed = 2 :=
by sorry

end work_speed_ratio_l554_554999


namespace asymptotes_of_hyperbola_l554_554803

variable {a : ℝ}

/-- Given that the length of the real axis of the hyperbola x^2/a^2 - y^2 = 1 (a > 0) is 1,
    we want to prove that the equation of its asymptotes is y = ± 2x. -/
theorem asymptotes_of_hyperbola (ha : a > 0) (h_len : 2 * a = 1) :
  ∀ x y : ℝ, (y = 2 * x) ∨ (y = -2 * x) :=
by {
  sorry
}

end asymptotes_of_hyperbola_l554_554803


namespace curve_intersects_triangle_one_point_not_parallel_midline_l554_554894

variables {a b c : ℝ}
def z0 := complex.I * a
def z1 := (1 / 2 : ℝ) + complex.I * b
def z2 := 1 + complex.I * c

def curve (t : ℝ) : ℂ := z0 * (real.cos t)^4 + 2 * z1 * (real.cos t)^2 * (real.sin t)^2 + z2 * (real.sin t)^4

-- Hypothesis: Points A, B, C are non-collinear
def non_collinear (a b c : ℝ) : Prop := (a + c - 2 * b) ≠ 0

theorem curve_intersects_triangle_one_point_not_parallel_midline (h : non_collinear a b c) :
  ∃ (t : ℝ), curve t = (1 / 2 : ℝ) + complex.I * ((a + c + 2 * b) / 4) :=
sorry

end curve_intersects_triangle_one_point_not_parallel_midline_l554_554894


namespace simple_interest_calculation_l554_554978

-- Define the principal (P), rate (R), and time (T)
def principal : ℝ := 10000
def rate : ℝ := 0.08
def time : ℝ := 1

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- The theorem to be proved
theorem simple_interest_calculation : simple_interest principal rate time = 800 :=
by
  -- Proof steps would go here, but this is left as an exercise
  sorry

end simple_interest_calculation_l554_554978


namespace fraction_sum_equals_4_l554_554121

noncomputable def a_n (n : ℕ) : ℝ := ∑ d in Finset.filter (λ d, d ∣ n) (Finset.range (n+1)), (1 : ℝ) / (2 : ℝ)^(d + n/d)

theorem fraction_sum_equals_4 : 
  (∑ k in Finset.range 1; ∞, k * a_n k) / (∑ k in Finset.range 1; ∞, a_n k) = 4 := 
sorry

end fraction_sum_equals_4_l554_554121


namespace eggs_total_l554_554823

-- Definitions based on the conditions
def breakfast_eggs : Nat := 2
def lunch_eggs : Nat := 3
def dinner_eggs : Nat := 1

-- Theorem statement
theorem eggs_total : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  -- This part will be filled in with proof steps, but it's omitted here
  sorry

end eggs_total_l554_554823


namespace measure_GDA_l554_554098

def isosceles_right_triangle (A B C : Type) [MetricSpace A] (a b c : A) :=
  dist a b = dist a c ∧ ∠ a b c = π / 2 ∧ ∠ b a c = π / 4

def square (A B : Type) [MetricSpace A] (a b c d : A) :=
  ∡ a b c = π / 2 ∧ ∡ b c d = π / 2 ∧ ∡ c d a = π / 2 ∧ ∡ d a b = π / 2

theorem measure_GDA (A B : Type) [MetricSpace A] [MetricSpace B] 
  (D E F G : A) (A B C : B) :
  isosceles_right_triangle B A C D →
  ∠(A B C) = π / 2 ∧ ∠(D E F) = π / 2 →
  ∡(G D A) = 3 * π / 4 :=
by
  sorry

end measure_GDA_l554_554098


namespace parabola_x_intercepts_l554_554448

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end parabola_x_intercepts_l554_554448


namespace friend_spent_13_50_l554_554622

noncomputable def amount_you_spent : ℝ := 
  let x := (22 - 5) / 2
  x

noncomputable def amount_friend_spent (x : ℝ) : ℝ := 
  x + 5

theorem friend_spent_13_50 :
  ∃ x : ℝ, (x + (x + 5) = 22) ∧ (x + 5 = 13.5) :=
by
  sorry

end friend_spent_13_50_l554_554622


namespace angle_45_deg_is_75_venerts_l554_554153

-- There are 600 venerts in a full circle.
def venus_full_circle : ℕ := 600

-- A full circle on Earth is 360 degrees.
def earth_full_circle : ℕ := 360

-- Conversion factor from degrees to venerts.
def degrees_to_venerts (deg : ℕ) : ℕ :=
  deg * (venus_full_circle / earth_full_circle)

-- Angle of 45 degrees in venerts.
def angle_45_deg_in_venerts : ℕ := 45 * (venus_full_circle / earth_full_circle)

theorem angle_45_deg_is_75_venerts :
  angle_45_deg_in_venerts = 75 :=
by
  -- Proof will be inserted here.
  sorry

end angle_45_deg_is_75_venerts_l554_554153


namespace length_of_CE_l554_554172

theorem length_of_CE {AB DE CE : ℝ} (h1 : AB = 8) (h2 : DE = 5) (h3 : DEF ∥ AB) (h4 : DE > 0) 
  (D_on_AC : ∃ D : ℝ, D ∈ segment AC) (E_on_BC : ∃ E : ℝ, E ∈ segment BC) (bisect_angle : bisects (line AE) (angle FEC)) : 
  CE = 40 / 3 := 
by 
  -- Given conditions are provided in hypotheses
  -- The calculations follow the provided solution steps
  sorry

end length_of_CE_l554_554172


namespace power_function_point_l554_554414

theorem power_function_point (m n: ℝ) (h: (m - 1) * m^n = 8) : n^(-m) = 1/9 := 
  sorry

end power_function_point_l554_554414


namespace all_ones_l554_554503

theorem all_ones (k : ℕ) (h₁ : k ≥ 2) (n : ℕ → ℕ) (h₂ : ∀ i, 1 ≤ i → i < k → n (i + 1) ∣ (2 ^ n i - 1))
(h₃ : n 1 ∣ (2 ^ n k - 1)) : (∀ i, 1 ≤ i → i ≤ k → n i = 1) :=
by
  sorry

end all_ones_l554_554503


namespace find_p_l554_554486

theorem find_p (p : ℝ) (h1 : (1/2) * 15 * (3 + 15) - ((1/2) * 3 * (15 - p) + (1/2) * 15 * p) = 40) : 
  p = 12.0833 :=
by sorry

end find_p_l554_554486


namespace part1_part2_l554_554908

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def vector_b : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)

theorem part1 (θ : ℝ) (h_parallel : (vector_a θ).fst * vector_b.snd = (vector_a θ).snd * vector_b.fst) (h_range : 0 < θ ∧ θ < Real.pi) :
  θ = 2 * Real.pi / 3 :=
sorry

theorem part2 (θ : ℝ) (h_magnitude : (3 * vector_a θ.fst + vector_b.fst) ^ 2 + (3 * vector_a θ.snd + vector_b.snd) ^ 2 = (vector_a θ.fst + 3 * vector_b.fst) ^ 2 + (vector_a θ.snd - 3 * vector_b.snd) ^ 2) :
  Real.sqrt (((vector_a θ).fst - 1/2) ^ 2 + ((vector_a θ).snd + Real.sqrt 3 / 2) ^2) = Real.sqrt 2 :=
sorry

end part1_part2_l554_554908


namespace new_books_to_clothes_ratio_l554_554210

def ratio (a b : ℕ) : ℚ := (a : ℚ) / b

variables (B C E : ℕ)

-- given conditions
axiom initial_ratio : ratio B C = 7 / 4
axiom electronics_weight : E = 9
axiom initial_total_weight : B + C + E = 42
axiom remove_clothes_weight : 6

-- statement to prove
theorem new_books_to_clothes_ratio (B C : ℕ) : 
  B = 21 → C = 12 → 
  ratio B (C - remove_clothes_weight) = 7 / 2 :=
by
  intros HB HC
  rw [HB, HC, remove_clothes_weight]
  norm_num
  sorry

end new_books_to_clothes_ratio_l554_554210


namespace molly_swam_28_meters_on_sunday_l554_554149

def meters_swam_on_saturday : ℕ := 45
def total_meters_swum : ℕ := 73
def meters_swam_on_sunday := total_meters_swum - meters_swam_on_saturday

theorem molly_swam_28_meters_on_sunday : meters_swam_on_sunday = 28 :=
by
  -- sorry to skip the proof
  sorry

end molly_swam_28_meters_on_sunday_l554_554149


namespace area_ratio_mn_l554_554506

-- Establishing the given conditions with Lean definitions
section
  variables {P Q R S T U V W X Y Z: Type*} [metric_space P]

  variable (A B C D E F: P)
  variable [metric_space P]
  
  variables (AB: ℝ) (BC: ℝ) (EF: ℝ)
  variables [parallel: set.parallel A B D E] 
            [parallel1: set.parallel B C E F]
            [parallel2: set.parallel C D A F]
  
  -- Setting angles in radians
  variable (∠ABC : real) (∠DEF : real)
  variable (angle_ABC : 150 * (π/180) = ∠ABC)

  -- Distances
  variable (dist_AB: dist A B = 4)
  variable (dist_BC: dist B C = 7)
  variable (dist_EF: dist E F = 21)

  -- Appropriate statements for similarity of triangles (dummy for placeholders)
  axiom (triangle_similarity: similar (triangle A B C) (triangle D E F))

  -- Statement for the ratio of the areas
  theorem area_ratio_mn : (∃ (m n : ℕ), ∃ h : nat.gcd m n = 1, 
    (area (triangle A B C) / area (triangle D E F) = (m : ℝ) / (n : ℝ)) ∧ m + n = 457) := 
  sorry
end

end area_ratio_mn_l554_554506


namespace num_ways_choose_edges_no_common_vertices_l554_554451

-- Define the cube structure
structure Cube (V E : Type) (vertices : Finset V) (edges : Finset E) :=
  (vertex_count : vertices.card = 8)
  (edge_count : edges.card = 12)
  (incident : E → V × V)
  (vertex_incidence : ∀ v : V, (edges.filter (λ e, v ∈ (incident e).fst ∨ v ∈ (incident e).snd)).card = 3)

-- Define the condition that four edges have no common vertices
def no_common_vertices {V E : Type} (c : Cube V E) (e1 e2 e3 e4 : E) : Prop :=
  (∃ v1 v2 v3 v4 : V, v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧
                      v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
                      c.incident e1 = (v1, v2) ∧ c.incident e2 = (v3, v4) ∧
                      c.incident e3 = (v1, v3) ∧ c.incident e4 = (v2, v4))

-- Main theorem statement
theorem num_ways_choose_edges_no_common_vertices : ∃ (V E : Type) (c : Cube V E), 
  (∃ es : Finset E, es.card = 4 ∧ ∀ e1 e2 ∈ es, e1 ≠ e2 → no_common_vertices c e1 e2 es.univ.erase ⟨e1, H⟩ ⟨e2, H⟩.toFinset → card = 15) :=
sorry

end num_ways_choose_edges_no_common_vertices_l554_554451


namespace leo_peeled_potatoes_l554_554911

noncomputable def lucy_rate : ℝ := 4
noncomputable def leo_rate : ℝ := 6
noncomputable def total_potatoes : ℝ := 60
noncomputable def lucy_time_alone : ℝ := 6
noncomputable def total_potatoes_left : ℝ := total_potatoes - lucy_rate * lucy_time_alone
noncomputable def combined_rate : ℝ := lucy_rate + leo_rate
noncomputable def combined_time : ℝ := total_potatoes_left / combined_rate
noncomputable def leo_potatoes : ℝ := combined_time * leo_rate

theorem leo_peeled_potatoes :
  leo_potatoes = 22 :=
by
  sorry

end leo_peeled_potatoes_l554_554911


namespace countLuckyNumbers_l554_554664

def isLuckyNumber (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧
  (let q := 221 / n in (221 % n) % q = 0)

theorem countLuckyNumbers : 
  { n : ℕ | isLuckyNumber n }.toFinset.card = 115 :=
by
  sorry

end countLuckyNumbers_l554_554664


namespace distinct_cube_paintings_correct_l554_554294

noncomputable def distinct_cube_paintings : ℕ := 4

theorem distinct_cube_paintings_correct :
  ∃ (distinct_cube_paintings : ℕ), distinct_cube_paintings = 4 ∧
  (∀ (C : (cube : Type) → (face_color : cube → string)),
    (∃ (yellow purple : cube) (orange_faces : Fin 4 → cube),
      face_color yellow = "yellow" ∧
      face_color purple = "purple" ∧
      (∀ i, face_color (orange_faces i) = "orange") ∧
      distinct_cube_paintings = 4)) :=
by
  use distinct_cube_paintings
  simp
  use [4]
  intros cube face_color
  use ["yellow", "purple"]
  use (λ i, "orange")
  simp
  split
  sorry

end distinct_cube_paintings_correct_l554_554294


namespace find_a_b_l554_554091

noncomputable def ellipse_condition_1 (a b : ℝ) : Prop := 
  (∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) ∧ (a > b)

noncomputable def eccentricity_condition (a c : ℝ) : Prop :=
  c / a = 1 / 2

noncomputable def distance_condition (a c : ℝ) : Prop :=
  a - c = 2

theorem find_a_b :
  ∃ a b c : ℝ, ellipse_condition_1 a b ∧ a > 0 ∧ b > 0 ∧ a = 4 ∧ b = 2 * sqrt 3 ∧
    eccentricity_condition a c ∧ distance_condition a c :=
sorry

end find_a_b_l554_554091


namespace total_suitcases_l554_554537

-- Definitions based on the conditions in a)
def siblings : ℕ := 4
def suitcases_per_sibling : ℕ := 2
def parents : ℕ := 2
def suitcases_per_parent : ℕ := 3
def suitcases_per_Lily : ℕ := 0

-- The statement to be proved
theorem total_suitcases : (siblings * suitcases_per_sibling) + (parents * suitcases_per_parent) + suitcases_per_Lily = 14 :=
by
  sorry

end total_suitcases_l554_554537


namespace range_of_b_l554_554883

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a x b : ℝ) (ha : -1 ≤ a) (ha' : a < 0) (hx : 0 < x) (hx' : x ≤ 1) 
  (h : f x a < b) : -3 / 2 < b := 
sorry

end range_of_b_l554_554883


namespace f_neg_one_f_monotonic_decreasing_solve_inequality_l554_554031

-- Definitions based on conditions in part a)
variables {f : ℝ → ℝ}
axiom f_add : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂ - 2
axiom f_one : f 1 = 0
axiom f_neg : ∀ x > 1, f x < 0

-- Proof statement for the value of f(-1)
theorem f_neg_one : f (-1) = 4 := by
  sorry

-- Proof statement for the monotonicity of f(x)
theorem f_monotonic_decreasing : ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Proof statement for the inequality solution
theorem solve_inequality (x : ℝ) :
  ∀ t, t = f (x^2 - 2*x) →
  t^2 + 2*t - 8 < 0 → (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 3) := by
  sorry

end f_neg_one_f_monotonic_decreasing_solve_inequality_l554_554031


namespace remaining_lawn_mowing_l554_554912

-- Definitions based on the conditions in the problem.
def Mary_mowing_time : ℝ := 3  -- Mary can mow the lawn in 3 hours
def John_mowing_time : ℝ := 6  -- John can mow the lawn in 6 hours
def John_work_time : ℝ := 3    -- John works for 3 hours

-- Question: How much of the lawn remains to be mowed?
theorem remaining_lawn_mowing : (Mary_mowing_time = 3) ∧ (John_mowing_time = 6) ∧ (John_work_time = 3) →
  (1 - (John_work_time / John_mowing_time) = 1 / 2) :=
by
  sorry

end remaining_lawn_mowing_l554_554912


namespace coeff_of_x7_in_expansion_l554_554246

noncomputable def coeff_x7 : ℤ :=
(∑ k in Finset.range 11, (Nat.choose 10 k) * (3 ^ (10 - k)) * (-4)^k) * (if k = 7 then 1 else 0)

theorem coeff_of_x7_in_expansion :
  coeff_x7 = -16796160 :=
by
  sorry

end coeff_of_x7_in_expansion_l554_554246


namespace total_expenditure_l554_554997

/-- Total money spent by 9 men in a hotel, where 8 spent 3 each and the ninth spent 5 more than the average expenditure of all nine, is 33. -/
theorem total_expenditure (A : ℝ) (T : ℝ) (h₁ : ∀ (i : ℕ), i < 8 → expenditure i = 3)
  (h₂ : expenditure 8 = A + 5)
  (h₃ : A = (8 * 3 + (A + 5)) / 9)
  (h₄ : T = 9 * A) :
  T = 33 :=
sorry

end total_expenditure_l554_554997


namespace sum_solutions_eq_neg_eight_l554_554515

def g (x : ℝ) : ℝ := 3 * x - 2

def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem sum_solutions_eq_neg_eight :
  (∑ x in {x : ℝ | g_inv x = g (x⁻¹)}, x) = -8 :=
by
  sorry

end sum_solutions_eq_neg_eight_l554_554515


namespace volume_of_folded_pyramid_l554_554352

open Real

noncomputable def triangle_vertices: list (ℝ × ℝ) :=
  [(0, 0), (30, 0), (12, 20)]

noncomputable def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def midpoint_triangle (vertices : list (ℝ × ℝ)) : list (ℝ × ℝ) :=
  [midpoint vertices.nth 0 vertices.nth 1, 
   midpoint vertices.nth 1 vertices.nth 2, 
   midpoint vertices.nth 2 vertices.nth 0]

noncomputable def area_of_triangle (a b c : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2))

noncomputable def volume_of_pyramid (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem volume_of_folded_pyramid :
  let vertices := triangle_vertices in
  let midpoints := midpoint_triangle vertices in
  let area := area_of_triangle midpoints.nth 0 midpoints.nth 1 midpoints.nth 2 in
  let height := 14.6 in       -- This height calculation is derived from the problem
  volume_of_pyramid area height = 365.5 :=
by
  let vertices := triangle_vertices
  let midpoints := midpoint_triangle vertices
  let area := area_of_triangle midpoints.nth 0 midpoints.nth 1 midpoints.nth 2
  let height := 14.6
  show volume_of_pyramid area height = 365.5
  sorry

end volume_of_folded_pyramid_l554_554352


namespace value_of_f_a1_a3_a5_l554_554396

-- Definitions
def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Problem statement
theorem value_of_f_a1_a3_a5 (f : ℝ → ℝ) (a : ℕ → ℝ) :
  monotonically_increasing f →
  odd_function f →
  arithmetic_sequence a →
  a 3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  intros h_mono h_odd h_arith h_a3
  sorry

end value_of_f_a1_a3_a5_l554_554396


namespace approx_ratio_of_d_to_square_side_l554_554856

noncomputable def square_side : ℝ := 2020
noncomputable def sector_probability : ℝ := 0.5
noncomputable def circle_area (d : ℝ) : ℝ := 4 * Real.pi * d^2
noncomputable def square_area : ℝ := square_side^2
noncomputable def required_ratio : ℝ := 0.4

theorem approx_ratio_of_d_to_square_side (d : ℝ) :
  (circle_area d) / square_area = sector_probability → 
  d / square_side ≈ required_ratio :=
by
  intro h
  rw [← h]
  sorry

end approx_ratio_of_d_to_square_side_l554_554856


namespace fractional_sum_inequality_l554_554122

-- Define fractional part
def frac (x : ℝ) : ℝ := x - x.floor

theorem fractional_sum_inequality (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ)
  (hx_pos : ∀ i, 0 < x i) (hx_prod : ∏ i, x i = 1) :
  (∑ i, frac (x i)) < (2 * n - 1) / 2 :=
  sorry

end fractional_sum_inequality_l554_554122


namespace range_of_lambda_l554_554802

def f (x : ℝ) : ℝ := x^2 + x

def g (x : ℝ) : ℝ := -x^2 + x

def h (λ : ℝ) (x : ℝ) : ℝ := g(x) - λ * f(x) + 1

theorem range_of_lambda :
  ∀ λ, (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), deriv (h λ) x ≥ 0) ↔ (-3 ≤ λ ∧ λ ≤ -1/3) :=
sorry

end range_of_lambda_l554_554802


namespace measure_angle_P_l554_554089

variable (Ω : Type)
variable [Circle Ω]
variable (P Q R : Ω)
variable (x : ℝ)

-- Conditions
def minor_arc_PQ : ℝ := 2 * x + 60
def minor_arc_QR : ℝ := 3 * x - 15
def minor_arc_RP : ℝ := 4 * x + 10
def tangent_angle : ℝ := 30

-- Questions and Correct Answer
theorem measure_angle_P (h : minor_arc_PQ + minor_arc_QR + minor_arc_RP = 360) 
  : angle_P Q R = 43.33 := by
  sorry

end measure_angle_P_l554_554089


namespace sum_two_digit_d_l554_554130

-- Define the given condition
def valid_d (d : ℕ) : Prop := d > 0 ∧ 143 % d = 3

-- Prove that the sum of all two-digit values of d satisfying the condition is 107
theorem sum_two_digit_d : 
  (∑ d in (Finset.filter (λ d, valid_d d ∧ 10 ≤ d ∧ d < 100)
   (Finset.range 144)), d) = 107 :=
by
  sorry

end sum_two_digit_d_l554_554130


namespace total_suitcases_correct_l554_554535

-- Conditions as definitions
def num_siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def num_parents : Nat := 2
def suitcases_per_parent : Nat := 3

-- Total suitcases calculation
def total_suitcases :=
  (num_siblings * suitcases_per_sibling) + (num_parents * suitcases_per_parent)

-- Statement to prove
theorem total_suitcases_correct : total_suitcases = 14 :=
by
  sorry

end total_suitcases_correct_l554_554535


namespace dave_paid_for_6_candy_bars_l554_554497

-- Given conditions
def number_of_candy_bars : ℕ := 20
def cost_per_candy_bar : ℝ := 1.50
def amount_paid_by_john : ℝ := 21

-- Correct answer
def number_of_candy_bars_paid_by_dave : ℝ := 6

-- The proof problem in Lean statement
theorem dave_paid_for_6_candy_bars (H : number_of_candy_bars * cost_per_candy_bar - amount_paid_by_john = 9) :
  number_of_candy_bars_paid_by_dave = 6 := by
sorry

end dave_paid_for_6_candy_bars_l554_554497


namespace interesting_quadruples_count_l554_554357

-- Defining the conditions for interesting ordered quadruples
def isInterestingQuadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d > b + c

-- The main theorem statement
theorem interesting_quadruples_count : 
  { (a, b, c, d) : ℕ × ℕ × ℕ × ℕ | isInterestingQuadruple a b c d }.toFinset.card = 80 := 
by
sorry

end interesting_quadruples_count_l554_554357


namespace composite_for_infinitely_many_n_l554_554171

theorem composite_for_infinitely_many_n :
  ∃ᶠ n in at_top, (n > 0) ∧ (n % 6 = 4) → ∃ p, p ≠ 1 ∧ p ≠ n^n + (n+1)^(n+1) :=
sorry

end composite_for_infinitely_many_n_l554_554171


namespace joan_sandwiches_l554_554111

theorem joan_sandwiches :
  ∀ (H : ℕ), (∀ (h_slice g_slice total_cheese num_grilled_cheese : ℕ),
  h_slice = 2 →
  g_slice = 3 →
  num_grilled_cheese = 10 →
  total_cheese = 50 →
  total_cheese - num_grilled_cheese * g_slice = H * h_slice →
  H = 10) :=
by
  intros H h_slice g_slice total_cheese num_grilled_cheese h_slice_eq g_slice_eq num_grilled_cheese_eq total_cheese_eq cheese_eq
  sorry

end joan_sandwiches_l554_554111


namespace safe_descent_possible_l554_554648

-- Definition of the parameters provided in the conditions.
structure MountaineerProblem :=
  (height_of_cliff : ℕ) -- 100m high cliff
  (length_of_rope : ℕ) -- 75m long rope
  (height_of_branch : ℕ) -- branch located 50m above the ground

-- The problem statement in Lean 4
theorem safe_descent_possible (p : MountaineerProblem) (h1 : p.height_of_cliff = 100) (h2 : p.length_of_rope = 75) (h3 : p.height_of_branch = 50) :
  ∃ steps : list string, steps ≠ [] ∧ "safe descent" ∈ steps := 
sorry

-- Example instance of the problem
def example_problem : MountaineerProblem := ⟨100, 75, 50⟩

end safe_descent_possible_l554_554648


namespace find_coin_flips_l554_554631

theorem find_coin_flips (n : ℕ) (h : (Nat.choose n 2 : ℚ) / 2^n = 7 / 32) : n = 7 :=
by 
  sorry

end find_coin_flips_l554_554631


namespace mailman_total_delivered_l554_554646

def pieces_of_junk_mail : Nat := 6
def magazines : Nat := 5
def newspapers : Nat := 3
def bills : Nat := 4
def postcards : Nat := 2

def total_pieces_of_mail : Nat := pieces_of_junk_mail + magazines + newspapers + bills + postcards

theorem mailman_total_delivered : total_pieces_of_mail = 20 := by
  sorry

end mailman_total_delivered_l554_554646


namespace possible_collections_l554_554373

-- Define the problem conditions
def collection (x : ℕ → ℤ) : Prop :=
  ∀ i, ∃ j, j ≠ i ∧ ∑ k in Finset.range 63, x k = x i ^ 2 + x j

/-- Given the conditions, show the possible collections -/
theorem possible_collections :
  ∃ x : ℕ → ℤ,
    (collection x) ∧
    (∃ m n k l : ℕ,
      (m = 39 ∧ n = 24 ∧ k = 12 ∧ l = -13 ∧
      (∀ i < 39, x i = k) ∧ (∀ j < 24, x (39 + j) = l)) ∨
      (m = 39 ∧ n = 24 ∧ k = 2 ∧ l = -3 ∧
      (∀ i < 39, x i = k) ∧ (∀ j < 24, x (39 + j) = l))) :=
begin
  sorry
end

end possible_collections_l554_554373


namespace mike_picked_12_pears_l554_554496

theorem mike_picked_12_pears
  (jason_pears : ℕ)
  (keith_pears : ℕ)
  (total_pears : ℕ)
  (H1 : jason_pears = 46)
  (H2 : keith_pears = 47)
  (H3 : total_pears = 105) :
  (total_pears - (jason_pears + keith_pears)) = 12 :=
by
  sorry

end mike_picked_12_pears_l554_554496


namespace minimize_perimeter_quadrilateral_l554_554326

-- Given values for the length of diagonals and angle
variables (a b : ℝ) (α : ℝ)

-- Definitions of points and their relations could be more detailed in a full implementation.
-- This is a simplified version for illustration.
noncomputable def minimal_perimeter_quadrilateral {A B C D K M N : Type} (ABCD : A × B × C × D) 
  (ACKM : A × C × K × M) (diagonalAC : A → C → ℝ) (diagonalBD : B → D → ℝ) 
  (angle : angle_between_diagonals AC BD = α) : Prop :=
  A × C × K × M →  D = N

theorem minimize_perimeter_quadrilateral
  (ABCD : Type)
  (a b α : ℝ)
  (diagonalAC : A → C → ℝ) (diagonalBD : B → D → ℝ)
  (angle : angle_between_diagonals diagonalAC diagonalBD = α) :
  minimal_perimeter_quadrilateral ABCD diagonalAC diagonalBD angle = true :=
sorry

end minimize_perimeter_quadrilateral_l554_554326


namespace diabetes_related_to_drinking_with_confidence_expected_cost_to_identify_drugs_l554_554494

open scoped ProbabilityTheory

noncomputable theory

/-- Part 1: Prove that with a given contingency table and completion conditions,
     the calculated value of \( K^{2} \) confirms that with 99.5% confidence,
     diabetes is related to drinking. -/
theorem diabetes_related_to_drinking_with_confidence :
  ∀ (n : ℕ) (a b c d : ℕ) (K0 : ℝ),
  (n = 30) →
  (a = 8) → (b = 2) → (c = 4) → (d = 16) →
  (K0 = 7.879) →
  let K2 := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  K2 > K0 :=
by sorry

/-- Part 2: Prove that the expected cost to identify the 2 particularly effective drugs
    from the 5 total drugs, given the probability distribution derived, is 700 yuan. -/
theorem expected_cost_to_identify_drugs (n : ℝ) (pX_400 pX_600 pX_800  E_X : ℝ) :
  n = 200 →
  pX_400 = 1 / 10 →
  pX_600 = 3 / 10 →
  pX_800 = 3 / 5 →
  E_X = 400 * pX_400 + 600 * pX_600 + 800 * pX_800 →
  E_X = 700 :=
by sorry

end diabetes_related_to_drinking_with_confidence_expected_cost_to_identify_drugs_l554_554494


namespace simplify_complex_division_l554_554277

theorem simplify_complex_division :
  (∃ (a b : ℝ), a = 5 ∧ b = 7 ∧ (∃ (c d : ℝ), c = 2 ∧ d = 3 ∧ ((a + (b * Complex.I)) / (c + (d * Complex.I)) = (31/13) - ((1/13) * Complex.I)))) :=
by {
  -- Defining the complex numbers.
  let num := 5 + 7 * Complex.I,
  let denom := 2 + 3 * Complex.I,
  
  -- Stating the main claim.
  have h : num / denom = (31/13) - ((1/13) * Complex.I),
  sorry, -- Proof is omitted

  -- Providing the solution variables to complete the statement.
  exact ⟨5, 7, rfl, rfl, ⟨2, 3, rfl, rfl, h⟩⟩
}

end simplify_complex_division_l554_554277


namespace sum_of_five_digits_from_set_l554_554018

def distinct_digits (s : set ℕ) (l : list ℕ) : Prop :=
  ∀ d ∈ l, d ∈ s ∧ list.nodup l ∧ list.length l = 5 

theorem sum_of_five_digits_from_set : 
  ∀ (s : set ℕ) (a b c x y z : ℕ), 
    s = {2, 3, 4, 5, 6, 7, 8} →
    distinct_digits s [a, b, c, x, y] →
    a + b + c = 17 →
    x + y + z = 14 →
    b = y →
    a + b + c + x + z = 26 :=
by
  intros s a b c x y z hs hd h1 h2 h3
  sorry

end sum_of_five_digits_from_set_l554_554018


namespace chord_length_is_six_l554_554189

theorem chord_length_is_six (θ : ℝ) :
  let curve_x := 2 + 5 * Real.cos θ,
      curve_y := 1 + 5 * Real.sin θ,
      line_eq := (3 * curve_x + 4 * curve_y + 10 = 0),
      circle_eq := (curve_x - 2) ^ 2 + (curve_y - 1) ^ 2 = 25,
      center_dist := abs (3 * 2 + 4 * 1 + 10) / Real.sqrt (3 ^ 2 + 4 ^ 2) 
  in center_dist = 4 
  ∧ 2 * Real.sqrt (5 ^ 2 - center_dist ^ 2) = 6 := 
by {
  sorry
}

end chord_length_is_six_l554_554189


namespace negation_proposition_l554_554204

variable {f : ℝ → ℝ}

theorem negation_proposition : ¬ (∀ x : ℝ, f x > 0) ↔ ∃ x : ℝ, f x ≤ 0 := by
  sorry

end negation_proposition_l554_554204


namespace kolya_or_leva_wins_l554_554499

-- Definitions for segment lengths
variables (k l : ℝ)

-- Definition of the condition when Kolya wins
def kolya_wins (k l : ℝ) : Prop :=
  k > l

-- Definition of the condition when Leva wins
def leva_wins (k l : ℝ) : Prop :=
  k ≤ l

-- Theorem statement for the proof problem
theorem kolya_or_leva_wins (k l : ℝ) : kolya_wins k l ∨ leva_wins k l :=
sorry

end kolya_or_leva_wins_l554_554499


namespace range_of_x_l554_554906

noncomputable def f (x : ℝ) : ℝ := real.log (3 + |x|) - (4 / (1 + x^2))

theorem range_of_x :
  { x : ℝ | f x - f (3 * x + 1) < 0 } = 
  { x : ℝ | x < -1/2 } ∪ { x : ℝ | -1/4 < x } :=
begin
  sorry
end

end range_of_x_l554_554906


namespace sum_of_coordinates_l554_554076

-- Define the function g such that g(4) = 8
def g : ℝ → ℝ := sorry  -- The details of the function are not known

-- Define the function h as 2 * (g(x))^3
def h (x : ℝ) : ℝ := 2 * (g x)^3

-- Define the specific condition that g(4) = 8
axiom g_at_4 : g 4 = 8

-- The theorem that states the sum of the coordinates equals 1028
theorem sum_of_coordinates : (4 : ℝ) + h 4 = 1028 :=
by
  rw [h, g_at_4]
  norm_num
  sorry

end sum_of_coordinates_l554_554076


namespace angle_parallel_lines_l554_554453

theorem angle_parallel_lines (AD_parallel_BC : ∀ A D B C : Prop, Prop)
    (angle_BDE : ∀ y : ℝ, y = (4 * (180 / 7)))
    (angle_EDC : ∀ y : ℝ, y = 3 * (180 / 7)) : 
    ( angle_EDC = 540 / 7 ) :=
by
  sorry

end angle_parallel_lines_l554_554453


namespace PQ_equals_sqrt_373_l554_554868

-- Definitions of the conditions
variable (P Q R M Q' R' E : Type)
variable [metric_space P]
variable [metric_space Q]
variable [metric_space R]
variable [metric_space M]
variable [metric_space Q']
variable [metric_space R']
variable [metric_space E]

variable PE ER QD : ℝ

-- Conditions given in the problem
def conditions : Prop :=
  PE = 8 ∧ ER = 16 ∧ QD = 15

-- Proof to show PQ = sqrt(373) with the given conditions
theorem PQ_equals_sqrt_373 (h : conditions) : sqrt (373) = PQ := 
by
  sorry

end PQ_equals_sqrt_373_l554_554868


namespace train_time_l554_554321

-- Condition 1: Time to travel from A to B
def time_A_to_B (p : ℝ) : ℝ := p / 50

-- Condition 2: Time to travel from B to C
def time_B_to_C (p : ℝ) : ℝ := 2 * p / 80

-- Condition 3: Pause time at B
def pause_B : ℝ := 0.5

-- Definition of total time as per the problem
def total_time (p : ℝ) : ℝ := time_A_to_B p + pause_B + time_B_to_C p

-- Statement of the theorem to prove that the total time equals (9p + 100) / 200
theorem train_time (p : ℝ) : total_time p = (9 * p + 100) / 200 := sorry

end train_time_l554_554321


namespace odd_function_f_l554_554462

variable {α : Type*} [LinearOrderedField α]

def f (a m x : α) : α := 1 + m / (a^x - 1)

theorem odd_function_f (a m : α) (h_odd : ∀ x, f a m (-x) = -f a m x) : m = 2 :=
sorry

end odd_function_f_l554_554462


namespace yield_per_acre_minimum_acres_to_convert_l554_554168

-- Part 1: Yield per acre of ordinary and hybrid rice
theorem yield_per_acre (x : ℕ) (y : ℕ) (h1 : 9600 = y * 1200) (h2 : 7200 = x * 600) (h3 : y = (x - 4)) : 
    x = 12 ∧ y = 8 :=
by
  -- solving the equations and proving x = 12 and y = 8
  sorry

-- Part 2: Minimum acres to be converted to meet yield goal
theorem minimum_acres_to_convert (total_yield : ℕ) (acres_a : ℕ) (acres_b : ℕ) (yield_a : ℕ) 
    (yield_b : ℕ) (converted_acres : ℕ) (h1 : total_yield = 17700) (h2 : acres_b - converted_acres ≥ 0) :
    converted_acres ≥ 1.5 :=
by
  -- solving the inequality to prove converted_acres ≥ 1.5
  sorry

end yield_per_acre_minimum_acres_to_convert_l554_554168


namespace train_crossing_time_l554_554319

open Real

noncomputable def time_to_cross_bridge 
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000/3600)
  total_distance / speed_train_ms

theorem train_crossing_time
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ)
  (h_length_train : length_train = 160)
  (h_speed_train_kmh : speed_train_kmh = 45)
  (h_length_bridge : length_bridge = 215) :
  time_to_cross_bridge length_train speed_train_kmh length_bridge = 30 :=
sorry

end train_crossing_time_l554_554319


namespace lucky_numbers_count_l554_554676

def is_lucky (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧ (221 % n) % (221 / n) = 0

def count_lucky_numbers : ℕ :=
  (Finset.range 222).filter is_lucky |>.card

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
by
  sorry

end lucky_numbers_count_l554_554676


namespace jason_speed_l554_554879

theorem jason_speed : 
  (∀ (distance : ℝ) (time : ℝ), distance = 17.48 * 1000 → time = 38 → 
   distance / time ≈ 460) := 
by
  -- Convert distance from km to m
  let distance_m : ℝ := 17.48 * 1000 
  -- Given time in seconds
  let time_s : ℝ := 38 
  -- Speed calculation
  let speed := distance_m / time_s  
  -- Assertion about speed
  have speed_approx : speed ≈ 460 := sorry 
  exact speed_approx

end jason_speed_l554_554879


namespace probability_multiple_of_12_l554_554846

-- Defining the set and necessary combinatorial functions
def my_set : set ℕ := {2, 3, 6, 9}

-- Function to determine if a product is a multiple of 12
def is_multiple_of_12 (a b : ℕ) : Prop := (a * b) % 12 = 0

-- Define the probability (number of favorable outcomes divided by total number of possible outcomes)
def probability_event : ℕ := 
  let total_choices : ℕ := nat.choose 4 2 in
  let favorable_pairs : list (ℕ × ℕ) := [(2, 6)] in
  favorable_pairs.length / total_choices

theorem probability_multiple_of_12 : probability_event = 1 / 6 := by sorry

end probability_multiple_of_12_l554_554846


namespace hyperbola_equation_proof_l554_554041

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  ∃ M : ℝ × ℝ, 
    (M.1^2 / 9 - M.2^2 = 1) ∧
    (let F1 := (-real.sqrt 10, 0 : ℝ),
         F2 := (real.sqrt 10, 0 : ℝ),
         MF1 := (M.1 - F1.1, M.2 - F1.2),
         MF2 := (M.1 - F2.1, M.2 - F2.2) in
     MF1.1 * MF2.1 + MF1.2 * MF2.2 = 0 ∧
     real.sqrt (MF1.1^2 + MF1.2^2) * real.sqrt (MF2.1^2 + MF2.2^2) = 2)

-- Statement of the theorem
theorem hyperbola_equation_proof :
  ∀ (x y : ℝ), hyperbola_equation x y → (x^2 / 9 - y^2 = 1) :=
by sorry

end hyperbola_equation_proof_l554_554041


namespace projection_correct_l554_554010

def projection_onto_plane : ℝ × ℝ × ℝ :=
  let v : ℝ × ℝ × ℝ := ⟨2, 3, 1⟩
  let n : ℝ × ℝ × ℝ := ⟨1, 2, -1⟩
  let dot (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let scalar_mul (k : ℝ) (a : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ⟨k * a.1, k * a.2, k * a.3⟩
  let sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ⟨a.1 - b.1, a.2 - b.2, a.3 - b.3⟩
  let c := dot v n / dot n n
  let v_proj_n := scalar_mul c n
  sub v v_proj_n

theorem projection_correct :
  projection_onto_plane = ⟨5/6, 4/6, 13/6⟩ :=
sorry

end projection_correct_l554_554010


namespace real_a_equals_two_l554_554833

theorem real_a_equals_two (a : ℝ) (h : 1 + a * complex.I = complex.I * (2 - complex.I)) : 
  a = 2 := 
sorry

end real_a_equals_two_l554_554833


namespace game_is_unfair_swap_to_make_fair_l554_554482

-- Part 1: Prove the game is unfair
theorem game_is_unfair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) :
  ¬((b : ℚ) / (y + b + r) = (y : ℚ) / (y + b + r)) :=
by
  -- The proof is omitted as per the instructions.
  sorry

-- Part 2: Prove that swapping 4 black balls with 4 yellow balls makes the game fair.
theorem swap_to_make_fair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) (x: ℕ) :
  x = 4 →
  (b - x : ℚ) / (y + b + r) = (y + x : ℚ) / (y + b + r) :=
by
  -- The proof is omitted as per the instructions.
  sorry

end game_is_unfair_swap_to_make_fair_l554_554482


namespace count_numerators_l554_554134

def rational_count (s : set ℚ) : ℕ :=
  (set.image num (set.filter (λ r : ℚ, 0 < r ∧ r < 1 ∧ ∃ (a b c : ℕ), r = (a*100 + b*10 + c) / 999) s)).toFinset.card

theorem count_numerators (s : set ℚ) : rational_count s = 660 :=
sorry

end count_numerators_l554_554134


namespace proof_problem_l554_554807

open Real

-- Definitions
noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Conditions
def eccentricity (c a : ℝ) : Prop :=
  c / a = (sqrt 2) / 2

def min_distance_to_focus (a c : ℝ) : Prop :=
  a - c = sqrt 2 - 1

-- Proof problem statement
theorem proof_problem (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (b_lt_a : b < a)
  (ecc : eccentricity c a) (min_dist : min_distance_to_focus a c)
  (x y k m : ℝ) (line_condition : y = k * x + m) :
  ellipse_equation x y a b → ellipse_equation x y (sqrt 2) 1 ∧
  (parabola_equation x y → (y = sqrt 2 / 2 * x + sqrt 2 ∨ y = -sqrt 2 / 2 * x - sqrt 2)) :=
sorry

end proof_problem_l554_554807


namespace pizza_volume_per_piece_l554_554303

-- Define the given conditions
def pizza_thickness : ℝ := 1 / 4
def pizza_diameter : ℝ := 16
def pizza_pieces : ℕ := 8

-- Compute the radius from the diameter
def pizza_radius : ℝ := pizza_diameter / 2

-- Compute the volume of the entire pizza
def pizza_volume : ℝ := π * pizza_radius^2 * pizza_thickness

-- Volume of one piece of pizza
theorem pizza_volume_per_piece : pizza_volume / pizza_pieces = 2 * π :=
by
  -- Write the proof here
  sorry

end pizza_volume_per_piece_l554_554303


namespace maximum_f_l554_554895

def is_permutation (S : List ℕ) (n : ℕ) : Prop :=
  S.perm (List.range (n + 1))

def f (S : List ℕ) : ℕ :=
  List.minimum (List.map (λ i => abs (S.nth_le (i + 1) (Nat.lt_of_lt_of_le (i + 1) S.length) - S.nth_le i (Nat.lt_succ_of_lt i))) (List.range (S.length - 1)))

theorem maximum_f (S : List ℕ) (n : ℕ) (hperm : is_permutation S n) :
  f(S) ≤ n / 2 :=
sorry

end maximum_f_l554_554895


namespace tangent_slope_neg_one_third_tangent_through_point_P_l554_554420

-- Curve definition: y = 1/x
def curve (x : ℝ) : ℝ := 1 / x

-- Problem (1): Prove the equation of the tangent line with slope -1/3 is given
theorem tangent_slope_neg_one_third :
  ∃ (a : ℝ), (curve a = 1 / a) ∧ (x_translate : ℝ = a) ∧ (y_translate : ℝ = (curve a)) ∧ 
    ((x : ℝ) (y : ℝ) → 3 * y = - x + 2 * a * sqrt 3 ∨ 3 * y = - x - 2 * a * sqrt 3) := 
sorry

-- Problem (2): Prove the equation of the tangent line passing through the point P(1,0) is given
theorem tangent_through_point_P :
  (curve 1 = 1 / 1) ∧ (x_through : ℝ = 1) ∧ (y_through : ℝ = 0) ∧ 
    (((curve 1) = (1 : ℝ)) ∧ (x = ℝ)) -> 
    ∃ (b : ℝ), (curve b = 1 / b) ∧ (x = 4 * b / 1) ∧ (y = 1 / b) :=
sorry

end tangent_slope_neg_one_third_tangent_through_point_P_l554_554420


namespace polyhedron_points_distance_l554_554178

/-- 
 Given a 20-faced polyhedron circumscribed around a sphere of radius 10, 
 prove that there exist two points on its surface such that the distance
 between them is greater than 21.
-/
theorem polyhedron_points_distance (F : ℕ) (r : ℝ) (p : F = 20 ∧ r = 10)  : 
  ∃ A B : ℝ², dist A B > 21 := 
sorry

end polyhedron_points_distance_l554_554178


namespace prime_q_from_prime_p_l554_554159

theorem prime_q_from_prime_p (p q : ℕ) 
  (hp : p.prime) 
  (hq : q.prime) 
  (h : 13 * p + 1 = q + 1) 
  : q = 41 := 
sorry

end prime_q_from_prime_p_l554_554159


namespace lucky_numbers_l554_554689

def is_lucky (n : ℕ) : Prop :=
  ∃ q r : ℕ, 221 = n * q + r ∧ r % q = 0

def lucky_numbers_count (a b : ℕ) : ℕ :=
  Nat.card {n // a ≤ n ∧ n ≤ b ∧ is_lucky n}

theorem lucky_numbers (a b : ℕ) (h1 : a = 1) (h2 : b = 221) :
  lucky_numbers_count a b = 115 := 
sorry

end lucky_numbers_l554_554689


namespace train_length_is_correct_l554_554320

noncomputable def speed_of_train_kmph : ℝ := 77.993280537557

noncomputable def speed_of_man_kmph : ℝ := 6

noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * conversion_factor

noncomputable def speed_of_man_mps : ℝ := speed_of_man_kmph * conversion_factor

noncomputable def relative_speed : ℝ := speed_of_train_mps + speed_of_man_mps

noncomputable def time_to_pass_man : ℝ := 6

noncomputable def length_of_train : ℝ := relative_speed * time_to_pass_man

theorem train_length_is_correct : length_of_train = 139.99 := by
  sorry

end train_length_is_correct_l554_554320


namespace max_magnitude_add_sub_l554_554794

noncomputable theory
open Real
open EuclideanSpace

/-- Conditions: a, b, and c are unit vectors, 
    a is orthogonal to b, 
    and the dot product of (a - b) with (b - c) is non-negative. --/
variables {a b c : E} [HilbertSpace E]

-- Definitions for the unit vectors and orthogonality condition
def unit_vector (x : E) : Prop := ∥x∥ = 1
def orthogonal (x y : E) : Prop := inner x y = 0

-- Main theorem: proving the maximum value of |a + b - c|
theorem max_magnitude_add_sub (h1 : unit_vector a) (h2 : unit_vector b) (h3 : unit_vector c) 
  (h4 : orthogonal a b) (h5 : inner (a - b) (b - c) ≥ 0) : 
  ∥a + b - c∥ ≤ √(2 + 1)^2 :=
sorry

end max_magnitude_add_sub_l554_554794


namespace problem1_problem2_l554_554376

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (h : ∀ x, f (f x) = 4 * x - 1) :
  (f = λ x, 2 * x - 1 / 3) ∨ (f = λ x, -2 * x + 1) :=
sorry

-- Problem 2
theorem problem2 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = 2 * x^2 - x + 1) :
  f = λ x, 2 * x^2 - 3 * x + 2 :=
sorry

end problem1_problem2_l554_554376


namespace solution_l554_554397

noncomputable def f : ℝ → ℝ 
| x => if x > 0 then logBase (3 : ℝ) x else 2 ^ x

theorem solution : f (f (1/9)) = 1/4 :=
by
  sorry

end solution_l554_554397


namespace feed_total_amount_l554_554232

theorem feed_total_amount {C E : ℝ} 
  (hC : C = 12.2051282051) 
  (cost_eq : 0.11 * C + 0.50 * E = (C + E) * 0.22) : 
  C + E ≈ 17 := 
by 
  sorry

end feed_total_amount_l554_554232


namespace determine_m_l554_554074

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  Real.log (m * x + Real.sqrt (x^2 + 1))

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

theorem determine_m (m : ℝ) :
  (is_odd_function (f m)) → (m = 1 ∨ m = -1) :=
by
  -- Proof is omitted
  sorry

end determine_m_l554_554074


namespace min_value_l554_554799

theorem min_value : ∀ (x y : ℝ), x > -1 → y > 0 → x + y = 1 → 
  (∃ (m : ℝ), m = \frac{1}{x+1} + \frac{4}{y} ∧ m ≥ \frac{9}{2}) :=
by
  intros x y hx hy hxy
  sorry

end min_value_l554_554799


namespace problem_1_problem_2_l554_554275

-- Problem (1)
theorem problem_1 (a c : ℝ) (h1 : ∀ x, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c > 0) :
  ∃ s, s = { x | -2 < x ∧ x < 3 } ∧ (∀ x, x ∈ s → cx^2 - 2*x + a < 0) := 
sorry

-- Problem (2)
theorem problem_2 (m : ℝ) (h : ∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) :
  m < 4 := 
sorry

end problem_1_problem_2_l554_554275


namespace todd_numbers_sum_eq_l554_554084

def sum_of_todd_numbers (n : ℕ) : ℕ :=
  sorry -- This would be the implementation of the sum based on provided problem conditions

theorem todd_numbers_sum_eq :
  sum_of_todd_numbers 5000 = 1250025 :=
sorry

end todd_numbers_sum_eq_l554_554084


namespace simplify_expression_l554_554175

theorem simplify_expression :
  ( ∀ (a b c : ℕ), c > 0 ∧ (∀ p : ℕ, Prime p → ¬ p^2 ∣ c) →
  (a - b * Real.sqrt c = (28 - 16 * Real.sqrt 3) * 2 ^ (-2 - Real.sqrt 5))) :=
sorry

end simplify_expression_l554_554175


namespace valid_sequences_proof_count_valid_sequences_length_6_l554_554776

def seq1 := [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
def seq2 := [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
def seq3 := [1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
def seq4 := [1, 1, 0, 0, 1, 1, 0, 1, 1, 0]

def valid_y_sequence (x : List ℕ) : List ℕ :=
  (List.zipWith (*) x (x.tail.dropLast 1))

-- Prove:
theorem valid_sequences_proof : 
  (valid_y_sequence seq1 = seq1) ∧ 
  (valid_y_sequence seq2 = seq2) ∧ 
  (valid_y_sequence seq3 = seq3) ∧ 
  (valid_y_sequence seq4 = seq4) :=
sorry

theorem count_valid_sequences_length_6 (valid_count : ℕ) :
  valid_count = 37 :=
sorry

end valid_sequences_proof_count_valid_sequences_length_6_l554_554776


namespace compute_expression_l554_554735

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l554_554735


namespace solve_equation_l554_554929

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔ x = -2 ∨ x = 1 / 3 :=
by
  sorry

end solve_equation_l554_554929


namespace find_wall_width_l554_554310

theorem find_wall_width
  (side_length_mirror : ℝ)
  (area_ratio : ℝ)
  (length_wall : ℝ) :
  side_length_mirror = 21 →
  area_ratio = 0.5 →
  length_wall = 31.5 →
  let width_wall := 2 * (side_length_mirror ^ 2) / length_wall in
  width_wall = 28 :=
by
  intros side_length_mirror_eq area_ratio_eq length_wall_eq
  rw [side_length_mirror_eq, area_ratio_eq, length_wall_eq]
  let width_wall := 2 * (21 ^ 2) / 31.5
  show width_wall = 28
  exact sorry

end find_wall_width_l554_554310


namespace count_lucky_numbers_l554_554654

def is_lucky (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 221) ∧ (221 % n % (221 / n) = 0)

theorem count_lucky_numbers : (finset.filter is_lucky (finset.range 222)).card = 115 :=
by
  sorry

end count_lucky_numbers_l554_554654


namespace minimum_draws_to_guarantee_three_of_same_color_l554_554635

theorem minimum_draws_to_guarantee_three_of_same_color :
  ∀ (balls : ℕ) (colors : ℕ) (balls_per_color : ℕ),
  balls = 60 →
  colors = 10 →
  balls_per_color = 6 →
  (∀ (n : ℕ), n = 21 → ∃ (m : ℕ → ℕ), 
    (∀ (i : ℕ), i < colors → m i ≤ balls_per_color) ∧ 
    (∑ i in finset.range colors, m i = n) ∧ 
    (∃ i : ℕ, i < colors ∧ m i ≥ 3)) :=
by sorry

end minimum_draws_to_guarantee_three_of_same_color_l554_554635


namespace tangent_equations_of_exp_function_l554_554772

theorem tangent_equations_of_exp_function :
  ∀ (x : ℝ), 
    (∃ (a : ℝ), a = 0 ∧ (∀ (y : ℝ), f (a) = (y - (a * x + f a))) = ℝ) 
        ∧ (∀ (b : ℝ), b = 1 ∧ (∀ (y : ℝ), f (b) = (y - (b * x + f b))) = ℝ) :=
begin
  let f := λ x : ℝ, Real.exp x,
  sorry
end

end tangent_equations_of_exp_function_l554_554772


namespace cubic_inches_in_four_cubic_yards_l554_554061

-- Define the conversion factors
def yard_to_feet : ℕ := 3
def foot_to_inches : ℕ := 12
def cubic_yard_to_cubic_feet : ℕ := yard_to_feet ^ 3
def cubic_foot_to_cubic_inches : ℕ := foot_to_inches ^ 3

-- Define our specific example
def four_cubic_yards : ℕ := 4

-- Calculate the conversion
theorem cubic_inches_in_four_cubic_yards :
  four_cubic_yards * cubic_yard_to_cubic_feet * cubic_foot_to_cubic_inches = 186624 := 
by
sory

end cubic_inches_in_four_cubic_yards_l554_554061


namespace math_problem_l554_554020

-- Definition of the conditions
variables (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_lt_ab : a < b)
variable (h_sum_ab : a + b = 2)

-- Problem statement
theorem math_problem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_lt_ab : a < b) (h_sum_ab : a + b = 2) : 
  (1 < b ∧ b < 2) ∧ (¬(2^(a - b) > 1)) ∧ (sqrt a + sqrt b < 2) ∧ (¬(1 / a + 2 / b >= 3)) :=
by sorry

end math_problem_l554_554020


namespace cos_alpha_eq_3_div_5_l554_554831

theorem cos_alpha_eq_3_div_5 (α : ℝ) (h1 : cos (2 * α) = -7 / 25) (h2 : 0 < α ∧ α < π / 2) :
  cos α = 3 / 5 :=
sorry

end cos_alpha_eq_3_div_5_l554_554831


namespace power_graph_point_l554_554415

theorem power_graph_point :
  ∀ (m n : ℕ), (m = 2 ∧ n = 3 ∧ 8 = (m - 1) * m^n) → n^(-m) = 1 / 9 :=
by
  intros m n h,
  cases h with hm hn,
  cases hn with hn1 hn2,
  have h1 : 8 = (m - 1) * m ^ n := hn2,
  have h2 : m = 2 := hm,
  have h3 : n = 3 := hn1,
  sorry

end power_graph_point_l554_554415


namespace lateral_surface_area_of_hexagonal_prism_l554_554643

def base_side_length : ℕ := 3
def lateral_edge_length : ℕ := 4
def regular_hexagon_perimeter (n : ℕ) : ℕ := 6 * n

theorem lateral_surface_area_of_hexagonal_prism :
  (regular_hexagon_perimeter base_side_length) * lateral_edge_length = 72 :=
by
  rw [regular_hexagon_perimeter, base_side_length, lateral_edge_length]
  sorry

end lateral_surface_area_of_hexagonal_prism_l554_554643


namespace range_of_a_l554_554513

def f (x : ℝ) : ℝ := if x >= 0 then 2^x + 1 else 2^(-x) + 1

theorem range_of_a (a : ℝ) (h₁ : f a < 3) : -1 < a ∧ a < 1 :=
by
  -- f is even: f(a) = f(-a)
  have h_even : ∀ x : ℝ, f x = f (-x) := by
    intro x
    -- Separate cases based on the value of x
    by_cases h : x >= 0
    · -- Case x >= 0
      simp [h, f] at *
      rw [←neg_neg x, if_pos (le_of_not_ge h)]
      rw [←if_pos h]
    · -- Case x < 0
      simp [h, f] at *
      rw [←neg_neg x, if_neg h, neg_neg, if_pos (le_of_lt (not_le.mp h))]
      rw [←if_pos (le_of_lt (not_le.mp h))]
  -- Simplify using h₁ and known properties of the function
  sorry

end range_of_a_l554_554513


namespace problem_a_l554_554260

theorem problem_a {O : Type*} [MetricSpace O] {A0 A1 A2 M : O} (h_eq_tri : EquilateralTriangle A0 A1 A2)
(h_on_circle : OnCircle M O) (h_A0A1 : OnCircle A0 O) (h_A1A2 : OnCircle A1 O) (h_A2A0 : OnCircle A2 O): 
dist M A0 = dist M A1 + dist M A2 :=
sorry

end problem_a_l554_554260


namespace number_of_x_intercepts_l554_554441

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end number_of_x_intercepts_l554_554441


namespace external_tangent_length_l554_554436

theorem external_tangent_length (R r : ℝ) (hRr : R > r) (hpos_R : 0 < R) (hpos_r : 0 < r)
    (h_right_angle : ∀ A B C D : Point, ⟦A, B⟧ ⟂ ⟦C, D⟧) : 
    (2 * sqrt (R * r)) = L :=
by
  sorry

end external_tangent_length_l554_554436


namespace find_f_prime_0_l554_554904

def f (x : ℝ) : ℝ := x^2 + 2 * x * (derivative f 1)

theorem find_f_prime_0 : (derivative f 0) = -4 :=
by sorry

end find_f_prime_0_l554_554904


namespace quadratic_two_distinct_real_roots_l554_554214

theorem quadratic_two_distinct_real_roots (a : ℝ) : 
  let Δ := a^2 + 4 in Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + a*x1 - 1 = 0) ∧ (x2^2 + a*x2 - 1 = 0)) :=
by
  intro h_discriminant
  sorry

end quadratic_two_distinct_real_roots_l554_554214


namespace complex_conjugate_in_third_quadrant_l554_554190

theorem complex_conjugate_in_third_quadrant (z : Complex) (h : z = 3 * Complex.i / (1 - Complex.i)) :
  Complex.conj z.re < 0 ∧ Complex.conj z.im < 0 :=
by
  sorry

end complex_conjugate_in_third_quadrant_l554_554190


namespace Mint_Coin_Denominations_l554_554941

theorem Mint_Coin_Denominations :
  ∃ denominations : Finset ℕ, 
    denominations.card = 12 ∧ 
    (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6543 → ∃ coins : Multiset ℕ, 
      (∀ c ∈ coins, c ∈ denominations) ∧ 
      coins.sum = n ∧ 
      coins.card ≤ 8) :=
sorry

end Mint_Coin_Denominations_l554_554941


namespace unique_three_digit_numbers_l554_554241

noncomputable def three_digit_numbers_no_repeats : Nat :=
  let total_digits := 10
  let permutations := total_digits * (total_digits - 1) * (total_digits - 2)
  let invalid_start_with_zero := (total_digits - 1) * (total_digits - 2)
  permutations - invalid_start_with_zero

theorem unique_three_digit_numbers : three_digit_numbers_no_repeats = 648 := by
  sorry

end unique_three_digit_numbers_l554_554241


namespace positive_factors_of_96_that_are_multiples_of_12_l554_554830

theorem positive_factors_of_96_that_are_multiples_of_12 : 
  {n : ℕ | n ∣ 96 ∧ 12 ∣ n}.card = 4 := by
  sorry

end positive_factors_of_96_that_are_multiples_of_12_l554_554830


namespace koala_fiber_l554_554117

theorem koala_fiber (absorption_rate : ℝ) (fiber_absorbed : ℝ) (x : ℝ) : 
  absorption_rate = 0.35 → fiber_absorbed = 15.75 → 0.35 * x = 15.75 → x = 45 :=
by
  intros h1 h2 h3
  rw [←h1, ←h2] at h3
  field_simp at h3
  exact h3

end koala_fiber_l554_554117


namespace algebraic_cofactor_a_l554_554385
-- Lean 4 statement

variable (a : ℝ)

def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, a, 5], ![0, -4, 1], ![-2, 1, 3]]

def minor (m : Matrix (Fin 3) (Fin 3) ℝ) (i j : Fin 3) : Matrix (Fin 2) (Fin 2) ℝ :=
  fun i' j' => m (if i'.val < i.val then i'.val else i'.val + 1) 
                 (if j'.val < j.val then j'.val else j'.val + 1)

theorem algebraic_cofactor_a :
  let minor_matrix := minor matrix 0 1 in
  matrix.det minor_matrix = -2 :=
by
  -- The matrix in minor_matrix should be the 2x2 minor of the original matrix after removing the first row and second column
  -- Then we calculate the determinant of this minor
  sorry

end algebraic_cofactor_a_l554_554385


namespace correct_statements_l554_554567

-- Define each of the conditions as boolean statements
def functional_relationship_deterministic : Prop := 
  ∀ (x y : Type) (f : x → y), ∀ a b : x, a = b → f a = f b

def correlation_relationship_nondeterministic : Prop :=
  ∀ (x y : Type) (corr : x → y → Prop), ¬ (∀ a b : x, a = b → ∀ c d : y, corr a c → corr b d)

def regression_analysis_for_functional_relationship : Prop :=
  ∀ (x y : Type) (reg : x → y), (∀ a b : x, a = b → reg a = reg b) → false

def regression_analysis_for_correlation_relationship : Prop :=
  ∀ (x y : Type) (reg : x → y), ∃ (corr : x → y → Prop), (¬ (∀ a b : x, a = b → ∀ c d : y, corr a c → corr b d)) ∧ (∀ a b : x, corr a b → reg a = reg b)

-- Define the final proof problem
theorem correct_statements (h1 : functional_relationship_deterministic) 
                          (h2 : correlation_relationship_nondeterministic)
                          (h3 : regression_analysis_for_functional_relationship)
                          (h4 : regression_analysis_for_correlation_relationship) :
  ({1, 2, 4} : set ℕ) = {1, 2, 4} :=
by 
  sorry

end correct_statements_l554_554567


namespace sum_of_x_values_l554_554617

theorem sum_of_x_values : (∑ x in {x : ℝ | real.sqrt (x - 2)^2 = 9}, x) = 4 :=
sorry

end sum_of_x_values_l554_554617


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l554_554045

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * (sin x) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x := 
by sorry

theorem max_min_values_of_f_on_interval : 
  ∃ (max_val min_val : ℝ), (max_val = 3) ∧ (min_val = 0) ∧ 
  (∀ x ∈ set.Icc 0 (π/2), f x ≤ max_val ∧ f x ≥ min_val) := 
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l554_554045


namespace small_box_dolls_l554_554723

theorem small_box_dolls (x : ℕ) : 
  (5 * 7 + 9 * x = 71) → x = 4 :=
by
  sorry

end small_box_dolls_l554_554723


namespace projection_of_v_on_plane_l554_554014

def projection_of_vector_on_plane 
  (v : ℝ × ℝ × ℝ := (2, 3, 1))
  (plane_normal : ℝ × ℝ × ℝ := (1, 2, -1))
  (p : ℝ × ℝ × ℝ := (5/6, 2/3, 13/6)) : Prop :=
  ∃ (k : ℝ × ℝ × ℝ), 
    k = (plane_normal.1 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2),
        plane_normal.2 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2),
        plane_normal.3 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2)) 
    ∧ (v.1 - k.1, v.2 - k.2, v.3 - k.3) = p

theorem projection_of_v_on_plane :
  projection_of_vector_on_plane :=
sorry

end projection_of_v_on_plane_l554_554014


namespace can_rearrange_figure_into_square_l554_554639

-- Define the initial conditions of the problem
def original_figure_drawn_on_squared_paper : Type := sorry  -- Placeholder for geometric figure

-- Define the ability to cut the figure into 5 triangles
def can_cut_into_5_triangles (figure : original_figure_drawn_on_squared_paper) : Prop :=
  sorry  -- Placeholder for the cutting property

-- Define the rearrangement property into a square
def can_rearrange_to_form_square (triangles : list (Type)) : Prop :=
  sorry  -- Placeholder for rearrangement property

-- Define the 5 triangles
def five_triangles (figure : original_figure_drawn_on_squared_paper) : list (Type) :=
  sorry  -- Placeholder for the list of triangles

-- State the theorem
theorem can_rearrange_figure_into_square (figure : original_figure_drawn_on_squared_paper) :
  (can_cut_into_5_triangles figure) ∧ (can_rearrange_to_form_square (five_triangles figure)) :=
begin
  sorry
end

end can_rearrange_figure_into_square_l554_554639


namespace compute_pqr_l554_554066

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 26) (h_eq : (1 : ℚ) / ↑p + (1 : ℚ) / ↑q + (1 : ℚ) / ↑r + 360 / (p * q * r) = 1) : 
  p * q * r = 576 := 
sorry

end compute_pqr_l554_554066


namespace preston_total_received_l554_554162

-- Conditions
def cost_per_sandwich := 5
def delivery_fee := 20
def number_of_sandwiches := 18
def tip_percentage := 0.10

-- Correct Answer
def total_amount_received := 121

-- Lean Statement
theorem preston_total_received : 
  (cost_per_sandwich * number_of_sandwiches + delivery_fee) * (1 + tip_percentage) = total_amount_received :=
by 
  sorry

end preston_total_received_l554_554162


namespace relationship_between_x_and_y_l554_554779

theorem relationship_between_x_and_y (a b : ℝ) (x y : ℝ)
  (h1 : x = a^2 + b^2 + 20)
  (h2 : y = 4 * (2 * b - a)) :
  x ≥ y :=
by 
-- we need to prove x ≥ y
sorry

end relationship_between_x_and_y_l554_554779


namespace count_lucky_numbers_l554_554677

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 221 ∧ let q := 221 / n in let r := 221 % n in r % q = 0

theorem count_lucky_numbers : ∃ (N : ℕ), N = 115 ∧ (∀ n, is_lucky n → n ≤ 221) := 
by
  apply Exists.intro 115
  constructor
  · rfl
  · intros n hn
    cases hn with hpos hrange
    exact hrange.left
  · sorry

end count_lucky_numbers_l554_554677


namespace number_added_is_10_l554_554597

-- Define the conditions.
def number_thought_of : ℕ := 55
def result : ℕ := 21

-- Define the statement of the problem.
theorem number_added_is_10 : ∃ (y : ℕ), (number_thought_of / 5 + y = result) ∧ (y = 10) := by
  sorry

end number_added_is_10_l554_554597


namespace fewest_number_of_tiles_l554_554702

theorem fewest_number_of_tiles (tile_length tile_width region_length region_width : ℕ)
    (tile_length_eq : tile_length = 2)
    (tile_width_eq : tile_width = 5)
    (region_length_eq : region_length = 3 * 12)
    (region_width_eq : region_width = 8 * 12)
    (tiles_needed : ℕ)
    (needed_eq : tiles_needed = Nat.ceil (region_length * region_width / (tile_length * tile_width))) :
  tiles_needed = 346 :=
by
  rw [tile_length_eq, tile_width_eq, region_length_eq, region_width_eq] at needed_eq
  exact needed_eq
  sorry

end fewest_number_of_tiles_l554_554702


namespace number_of_elements_l554_554126

def greatestInt (x : ℝ) : ℤ := floor x

noncomputable def problemSet : Set ℤ :=
  {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2004 ∧ n = greatestInt ((k^2 : ℝ) / 2005)}

theorem number_of_elements : (problemSet).card = 1503 := 
  sorry

end number_of_elements_l554_554126


namespace equal_reading_time_l554_554082

theorem equal_reading_time (total_pages: ℕ) (alice_time: ℕ) (bob_time: ℕ) (chandra_time: ℕ) (x y z: ℕ):
  total_pages = 900 ->
  alice_time = 18 ->
  bob_time = 36 ->
  chandra_time = 27 ->
  x = 416 ->
  y = 208 ->
  z = 276 ->
  x + y + z = total_pages :=
by
  intros h_total_pages h_alice_time h_bob_time h_chandra_time h_x h_y h_z
  simp [h_total_pages, h_alice_time, h_bob_time, h_chandra_time, h_x, h_y, h_z]
  exact eq.refl 900

end equal_reading_time_l554_554082


namespace gain_percentage_is_66_67_l554_554195

variable (C S : ℝ)
variable (cost_price_eq : 20 * C = 12 * S)

theorem gain_percentage_is_66_67 (h : 20 * C = 12 * S) : (((5 / 3) * C - C) / C) * 100 = 66.67 := by
  sorry

end gain_percentage_is_66_67_l554_554195


namespace car_speed_l554_554257

theorem car_speed (v : ℝ) (h : 2 * (60 : ℝ) * v = 3600) : v = 3600 / (60 + 2) :=
by
  field_simp [h]
  linarith

end car_speed_l554_554257


namespace positive_integer_k_l554_554760

theorem positive_integer_k {k : ℕ} :
  (∀ n ∈ ℕ, (∃ f : Fin k → ℕ, (∀ i : Fin k, f i ∣ n) ∧ (∑ i, (f i) ^ 2 = n)) → 
             ∃ g : Fin k → ℕ, (∀ i : Fin k, g i ∣ n) ∧ (∑ i, g i = n)) ↔ (k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 6) :=
by sorry

end positive_integer_k_l554_554760


namespace pappus_theorem_l554_554550

variables {Point Line : Type} [geometry Line Point]

-- Given conditions
axiom A : Point
axiom B : Point
axiom C : Point
axiom l : Line
axiom A1 : Point
axiom B1 : Point
axiom C1 : Point
axiom l1 : Line
axiom A_on_l : A ∈ l
axiom B_on_l : B ∈ l
axiom C_on_l : C ∈ l
axiom A1_on_l1 : A1 ∈ l1
axiom B1_on_l1 : B1 ∈ l1
axiom C1_on_l1 : C1 ∈ l1

-- Lines formed by intersections
noncomputable def AB1 := line_through A B1
noncomputable def BA1 := line_through B A1
noncomputable def BC1 := line_through B C1
noncomputable def CB1 := line_through C B1
noncomputable def CA1 := line_through C A1
noncomputable def AC1 := line_through A C1

noncomputable def D : Point := intersect AB1 BA1
noncomputable def E : Point := intersect BC1 CB1
noncomputable def F : Point := intersect CA1 AC1

-- Statement of Pappus's Theorem
theorem pappus_theorem :
  collinear {D, E, F} :=
by
  sorry

end pappus_theorem_l554_554550


namespace positive_diff_probability_fair_coin_l554_554610

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end positive_diff_probability_fair_coin_l554_554610


namespace length_of_PX_l554_554869

-- Define the lengths and ratios based on the conditions
constants (CD_parallel_WX : Prop) (CX : ℝ) (DP : ℝ) (PW : ℝ)
-- Assign given values
axiom h1 : CX = 56
axiom h2 : DP = 16
axiom h3 : PW = 32

-- Define the lengths PX and CP
noncomputable def PX : ℝ := 112 / 3
noncomputable def CP : ℝ := PX / 2

-- The proof statement
theorem length_of_PX : CD_parallel_WX → CX = CP + PX ∧ CP = PX / 2 → PX = 112 / 3 :=
by
  intro h_parallel h
  obtain ⟨hCX_eq, hCP_eq⟩ := h
  sorry

end length_of_PX_l554_554869


namespace avg_visitors_per_day_correct_l554_554259

-- Define the given conditions
def avg_sundays : Nat := 540
def avg_other_days : Nat := 240
def num_days : Nat := 30
def sundays_in_month : Nat := 5
def other_days_in_month : Nat := 25

-- Define the total visitors calculation
def total_visitors := (sundays_in_month * avg_sundays) + (other_days_in_month * avg_other_days)

-- Define the average visitors per day calculation
def avg_visitors_per_day := total_visitors / num_days

-- State the proof problem
theorem avg_visitors_per_day_correct : avg_visitors_per_day = 290 :=
by
  sorry

end avg_visitors_per_day_correct_l554_554259


namespace probability_multiple_of_12_l554_554845

-- Defining the set and necessary combinatorial functions
def my_set : set ℕ := {2, 3, 6, 9}

-- Function to determine if a product is a multiple of 12
def is_multiple_of_12 (a b : ℕ) : Prop := (a * b) % 12 = 0

-- Define the probability (number of favorable outcomes divided by total number of possible outcomes)
def probability_event : ℕ := 
  let total_choices : ℕ := nat.choose 4 2 in
  let favorable_pairs : list (ℕ × ℕ) := [(2, 6)] in
  favorable_pairs.length / total_choices

theorem probability_multiple_of_12 : probability_event = 1 / 6 := by sorry

end probability_multiple_of_12_l554_554845


namespace coffee_cream_ratio_l554_554824

theorem coffee_cream_ratio :
  let
    harry_initial_coffee := 20,
    harry_drunk_coffee := 4,
    harry_added_cream := 3,
    sally_initial_coffee := 20,
    sally_added_cream := 3,
    sally_drunk_mix := 4,
    harry_remaining_coffee := harry_initial_coffee - harry_drunk_coffee,
    harry_final_mixture := harry_remaining_coffee + harry_added_cream,
    sally_initial_mixture := sally_initial_coffee + sally_added_cream,
    sally_fraction_cream := sally_added_cream / sally_initial_mixture,
    sally_cream_drunk := sally_fraction_cream * sally_drunk_mix,
    sally_remaining_cream := sally_added_cream - sally_cream_drunk,
    harry_final_cream := harry_added_cream,
    cream_ratio := harry_final_cream / sally_remaining_cream in
  cream_ratio = (23 : ℚ) / 19 :=
by
  sorry

end coffee_cream_ratio_l554_554824


namespace fixed_point_P_l554_554811

variable (a : ℝ) (a_pos : a > 0) (a_ne_one : a ≠ 1)

def f (x : ℝ) : ℝ := a^(2 * x - 1) + 2

theorem fixed_point_P : f a (1/2) = 3 :=
by
  -- sorry as we are not providing the proof
  sorry

end fixed_point_P_l554_554811


namespace part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_l554_554919

noncomputable def angle_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem part_a (A B C : ℝ) (h : angle_in_triangle A B C) :
  0 < sin (A / 2) * sin (B / 2) * sin (C / 2) ∧ sin (A / 2) * sin (B / 2) * sin (C / 2) ≤ 1 / 8 :=
sorry

theorem part_b (A B C : ℝ) (h : angle_in_triangle A B C) :
  -1 < cos A * cos B * cos C ∧ cos A * cos B * cos C ≤ 1 / 8 :=
sorry

theorem part_c (A B C : ℝ) (h : angle_in_triangle A B C) :
  1 < sin (A / 2) + sin (B / 2) + sin (C / 2) ∧ sin (A / 2) + sin (B / 2) + sin (C / 2) ≤ 3 / 2 :=
sorry

theorem part_d (A B C : ℝ) (h : angle_in_triangle A B C) :
  2 < cos (A / 2) + cos (B / 2) + cos (C / 2) ∧ cos (A / 2) + cos (B / 2) + cos (C / 2) ≤ 3 * sqrt 3 / 2 :=
sorry

theorem part_e (A B C : ℝ) (h : angle_in_triangle A B C) :
  0 < sin A + sin B + sin C ∧ sin A + sin B + sin C ≤ 3 * sqrt 3 / 2 :=
sorry

theorem part_f (A B C : ℝ) (h : angle_in_triangle A B C) :
  1 < cos A + cos B + cos C ∧ cos A + cos B + cos C ≤ 3 / 2 :=
sorry

theorem part_g (A B C : ℝ) (h : angle_in_triangle A B C) :
  1 / sin A + 1 / sin B + 1 / sin C ≥ 2 * sqrt 3 :=
sorry

theorem part_h (A B C : ℝ) (h : angle_in_triangle A B C) :
  1 / (cos A ^ 2) + 1 / (cos B ^ 2) + 1 / (cos C ^ 2) ≥ 3 :=
sorry

end part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_l554_554919


namespace intersection_points_l554_554237

noncomputable def parabola1 : ℝ → ℝ := λ x, 4 * x^2 + 3 * x - 7
noncomputable def parabola2 : ℝ → ℝ := λ x, 2 * x^2 + 5

theorem intersection_points:
  (parabola1 (3 / 2) = 9.5) ∧ (parabola2 (3 / 2) = 9.5) ∧
  (parabola1 (-4) = 37) ∧ (parabola2 (-4) = 37) :=
by sorry

end intersection_points_l554_554237


namespace exists_two_digit_singular_number_exists_four_digit_singular_number_exists_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_hundred_digit_singular_numbers_exists_thirty_digit_singular_number_l554_554458

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_singular (n : ℕ) : Prop :=
  let s := n.to_digits in
  let len := s.length in
  let first_half := list.take (len / 2) s in
  let second_half := list.drop (len / 2) s in
  len.mod 2 = 0 ∧
  s ≠ [] ∧
  is_perfect_square n ∧
  ((first_half ≠ [] ∧ ! first_half.head!.is_zero ∧ is_perfect_square first_half.from_digits) ∧
  (second_half ≠ [] ∧ second_half.head!.is_zero ∧ second_half.from_digits ≠ 0 ∧ is_perfect_square second_half.from_digits))

theorem exists_two_digit_singular_number : ∃ n : ℕ, n.digits.length = 2 ∧ is_singular n ∧ n = 49 :=
sorry

theorem exists_four_digit_singular_number : ∃ n : ℕ, n.digits.length = 4 ∧ is_singular n ∧ n = 1681 :=
sorry

theorem exists_six_digit_singular_number : ∃ n : ℕ, n.digits.length = 6 ∧ is_singular n :=
sorry

theorem exists_twenty_digit_singular_number : ∃ n : ℕ, n.digits.length = 20 ∧ is_singular n :=
sorry

theorem at_most_ten_hundred_digit_singular_numbers : ∃! S : finset ℕ, S.card ≤ 10 ∧ ∀ n ∈ S, n.digits.length = 100 ∧ is_singular n :=
sorry

theorem exists_thirty_digit_singular_number : ∃ n : ℕ, n.digits.length = 30 ∧ is_singular n :=
sorry

end exists_two_digit_singular_number_exists_four_digit_singular_number_exists_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_hundred_digit_singular_numbers_exists_thirty_digit_singular_number_l554_554458


namespace max_number_of_substances_produced_l554_554970

-- Definitions for the types of isomers formed
inductive ChlorinatedEthane
  | monochloroethane
  | dichloroethane_1, dichloroethane_2, dichloroethane_3  -- Considering geometric isomer
  | trichloroethane_1, trichloroethane_2
  | tetrachloroethane_1, tetrachloroethane_2
  | pentachloroethane
  | hexachloroethane

def NumberOfProducts : ℕ :=
  ChlorinatedEthane.recOn 1 (fun _ => 1 + 1) (fun _ => 1 + 1 + 1) (fun _ => 2) (fun _ => 2) 
                          (fun _ => 1) (fun _ => 1) + 1 -- Adding HCl

theorem max_number_of_substances_produced :
  NumberOfProducts = 10 :=
by
  sorry

end max_number_of_substances_produced_l554_554970


namespace pens_to_make_desired_profit_l554_554704

-- Define the conditions
def num_pens_bought : ℕ := 2000
def cost_per_pen : ℝ := 0.15
def selling_price_per_pen : ℝ := 0.30
def desired_profit : ℝ := 120.00

-- Define the total_cost
def total_cost (n : ℕ) (c : ℝ) : ℝ := n * c

-- Define the total_revenue to achieve desired profit
def total_revenue (total_cost : ℝ) (desired_profit : ℝ) : ℝ := total_cost + desired_profit

-- Define the number of pens sold to achieve the total_revenue
def num_pens_sold (total_revenue : ℝ) (selling_price : ℝ) : ℝ := total_revenue / selling_price

theorem pens_to_make_desired_profit : 
  num_pens_sold (total_revenue (total_cost num_pens_bought cost_per_pen) desired_profit) selling_price_per_pen = 1400 :=
by 
  sorry

end pens_to_make_desired_profit_l554_554704


namespace garden_area_l554_554197

-- Definitions for the conditions
def perimeter : ℕ := 36
def width : ℕ := 10

-- Define the length using the perimeter and width
def length : ℕ := (perimeter - 2 * width) / 2

-- Define the area using the length and width
def area : ℕ := length * width

-- The theorem to prove the area is 80 square feet given the conditions
theorem garden_area : area = 80 :=
by 
  -- Here we use sorry to skip the proof
  sorry

end garden_area_l554_554197


namespace compute_expression_l554_554731

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l554_554731


namespace problem1_problem2_l554_554887

variable {a : ℕ → ℝ} {q : ℝ}
variable (hq : ∀ n, a n = a 0 * q ^ n) (hpos : 0 < q)

-- Definition of the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := (finset.range (n+1)).sum (λ i, a i)

-- Problem 1: Prove that lg S_n + lg S_{n+2} < 2 lg S_{n+1}
theorem problem1 (n : ℕ) (h : ∀ k, a 0 * q ^ k > 0) : 
  log (S n) + log (S (n + 2)) < 2 * log (S (n + 1)) := sorry

-- Problem 2: Prove that there does not exist a constant C > 0 such that
-- lg(S_n - C) + lg(S_{n + 2} - C) = 2 * lg(S_{n + 1} - C) for all n.
theorem problem2 : ¬ ∃ C : ℝ, 0 < C ∧ ∀ n, 
  log (S n - C) + log (S (n + 2) - C) = 2 * log (S (n + 1) - C) := sorry

end problem1_problem2_l554_554887


namespace find_n_l554_554987

def p (m : ℕ) : ℕ :=
sorry

def f (m : ℕ) : ℕ :=
sorry

theorem find_n (n : ℕ) : 
  (∀ (m : ℕ), m > 1 → p m = numberOfDistinctPrimeDivisors m) →
  f(n^2 + 2) + f(n^2 + 5) = 2n - 4 ↔ n = 5 :=
sorry

end find_n_l554_554987


namespace max_theta_condition_l554_554762

noncomputable def theta : ℝ := 2046 * Real.pi / 2047

theorem max_theta_condition : 
  (theta < Real.pi) ∧ (∀ (k : ℕ), 0 ≤ k ∧ k ≤ 10 → cos (2^k * theta) ≠ 0) ∧ (
  Real.prod (Finset.range 11) (λ k, 1 + 1 / cos (2^k * theta)) = 1) :=
by
  sorry

end max_theta_condition_l554_554762


namespace countLuckyNumbers_l554_554663

def isLuckyNumber (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧
  (let q := 221 / n in (221 % n) % q = 0)

theorem countLuckyNumbers : 
  { n : ℕ | isLuckyNumber n }.toFinset.card = 115 :=
by
  sorry

end countLuckyNumbers_l554_554663


namespace area_of_quadrilateral_l554_554600

theorem area_of_quadrilateral (h1 : ∀ x y : ℝ, x^2 + y^2 = 1 → 
  ∃ t1 t2 : ℝ, tangent_line_through (1,2) (x,y) t1 t2) : 
  let q_area := area_of_quad (tangents_from_point_to_circle (1,2) (x^2 + y^2 = 1)) :=
  13/8 := 
by
  sorry

end area_of_quadrilateral_l554_554600


namespace sin_four_alpha_l554_554778

theorem sin_four_alpha (α : ℝ) (h1 : Real.sin (2 * α) = -4 / 5) (h2 : -Real.pi / 4 < α ∧ α < Real.pi / 4) :
  Real.sin (4 * α) = -24 / 25 :=
sorry

end sin_four_alpha_l554_554778


namespace common_ratio_of_sequence_l554_554485

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (n1 n2 n3 : ℕ) : Prop :=
  a n2 = a n1 * r ∧ a n3 = a n1 * r^2

theorem common_ratio_of_sequence {a : ℕ → ℝ} {d : ℝ}
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence a ((a 2)/(a 1)) 2 3 6) :
  ((a 3) / (a 2)) = 3 ∨ ((a 3) / (a 2)) = 1 :=
sorry

end common_ratio_of_sequence_l554_554485


namespace sum_series_eq_l554_554370

theorem sum_series_eq : 
  (∑' k : ℕ, (k + 1) * (1/4)^(k + 1)) = 4 / 9 :=
by sorry

end sum_series_eq_l554_554370


namespace remainder_polynomial_div_l554_554255

noncomputable def find_k_a (P D : Polynomial ℝ) (x k a : ℝ) : Prop :=
  P = Polynomial.X^4 - 3 * Polynomial.X^3 + 10 * Polynomial.X^2 - 16 * Polynomial.X + 5 ∧
  D = Polynomial.X^2 - Polynomial.X + Polynomial.C k ∧
  Polynomial.modByMonic P (D.monicOfContentEqOne) = 2 * Polynomial.X + Polynomial.C a
  
theorem remainder_polynomial_div :
  find_k_a (Polynomial.X^4 - 3 * Polynomial.X^3 + 10 * Polynomial.X^2 - 16 * Polynomial.X + 5)
            (Polynomial.X^2 - Polynomial.X + Polynomial.C 8.5) 8.5 9.25 :=
begin
  sorry
end

end remainder_polynomial_div_l554_554255


namespace main_theorem_l554_554865

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n, a n = a 0 + n * d

-- Given condition for specific terms
def condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Prove the required equality
theorem main_theorem (a : ℕ → ℝ) (d : ℝ) [arithmetic_sequence a d] [condition a d] :
  2 * a 10 - a 12 = 24 :=
  sorry

end main_theorem_l554_554865


namespace george_total_blocks_l554_554022

/--
George has 2 boxes of large blocks and 3 boxes of small blocks. 
Each large box holds 6 large blocks. Each small box holds 8 small blocks. 
There are 5 boxes in a case, each holding 10 medium blocks.
-/
theorem george_total_blocks :
  let large_blocks := 2 * 6 in
  let small_blocks := 3 * 8 in
  let medium_blocks := 5 * 10 in
  large_blocks + small_blocks + medium_blocks = 86 :=
by
  let large_blocks := 2 * 6
  let small_blocks := 3 * 8
  let medium_blocks := 5 * 10
  show large_blocks + small_blocks + medium_blocks = 86
  calc
    12 + 24 + 50 = 86 : by sorry

end george_total_blocks_l554_554022


namespace triangles_from_10_points_l554_554933

theorem triangles_from_10_points : 
  (finset.univ.choose 3).card = 120 :=
by 
  -- There are 10 distinct points on the circumference of a circle
  let points : finset ℕ := finset.range 10
  -- Number of ways to choose 3 points to form a triangle
  have h : (points.choose 3).card = finset.card ((finset.range 10).choose 3),
  from rfl, -- trivial rewriting
  rw finset.card_choose,
  norm_num,
  exact nat.choose_eq (finset.card points) 3,
  assumption,
-- skipping further detailed proof
sorry

end triangles_from_10_points_l554_554933


namespace lucky_numbers_count_l554_554697

def is_lucky (n : ℕ) : Prop := 
  ∃ (k q r : ℕ), 
    1 ≤ n ∧ n ≤ 221 ∧
    221 = n * q + r ∧ 
    r = k * q ∧ 
    r < n

def count_lucky_numbers : ℕ := 
  ∑ n in (finset.range 222).filter is_lucky, 1

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
  by 
    sorry

end lucky_numbers_count_l554_554697


namespace point_quadrant_l554_554459

theorem point_quadrant (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) : 
  ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  sorry

end point_quadrant_l554_554459


namespace perimeter_bisector_concurrence_l554_554104

theorem perimeter_bisector_concurrence 
  (A B C D E F X : Type) 
  [add_comm_group A] [add_comm_group B] [add_comm_group C]
  (h1 : sorry)  -- Triangle ABC
  (h2 : ∀ {D}, B + D = C + D) -- Perimeter-bisector condition AB + BD = AC + CD
  (h3 : ∀ {E}, sorry) -- Similar condition for vertex B
  (h4 : ∀ {F}, sorry) -- Similar condition for vertex C
  : X ∩ E ∩ F := 
begin
  sorry -- Proof skipped
end

end perimeter_bisector_concurrence_l554_554104


namespace length_BC_eq_13_sqrt_14_l554_554120

-- Define the given properties.
variables {A B C I P : Type}
variables [triangle : Π A B C : Type, Prop] [incenter : Π A B C I : Type, Prop]
variables [circumcircle_tangent_intersection : Π B C P : Type, Prop]
variables [distance : Π (X Y : Type), ℝ]

-- Given conditions.
variable (ABC_triangle : triangle A B C)
variable (I_incenter : incenter A B C I)
variable (P_intersection : circumcircle_tangent_intersection B C P)
variable (AB_eq_10 : distance A B = 10)
variable (AC_eq_16 : distance A C = 16)
variable (PA_eq_8 : distance P A = 8)

-- Target theorem.
theorem length_BC_eq_13_sqrt_14 :
  distance B C = 13 * Real.sqrt 14 :=
sorry

end length_BC_eq_13_sqrt_14_l554_554120


namespace intersect_at_one_point_l554_554954

open EuclideanGeometry

variables {A B C A1 B1 C1 A2 B2 C2 : Point}
variables {ABC_circumcircle : Circle}

-- Assumptions
-- A, B, C are vertices of the triangle ABC.
-- Points A1, B1, C1 are intersections of parallel lines drawn through A, B, C, 
-- intersecting the circumcircle of ABC again at A1, B1, C1 respectively.
-- Points A2, B2, C2 are symmetric to A1, B1, C1 regarding sides BC, CA, AB respectively.

theorem intersect_at_one_point 
  (h1 : IsVertex A ABC)
  (h2 : IsVertex B ABC)
  (h3 : IsVertex C ABC)
  (h4 : IsSecondIntersection A A1 ABC_circumcircle)
  (h5 : IsSecondIntersection B B1 ABC_circumcircle)
  (h6 : IsSecondIntersection C C1 ABC_circumcircle)
  (h7 : SymmetricPoint A1 BC A2)
  (h8 : SymmetricPoint B1 CA B2)
  (h9 : SymmetricPoint C1 AB C2) :
  Concurrent (Line A A2) (Line B B2) (Line C C2) :=
sorry

end intersect_at_one_point_l554_554954


namespace variable_order_l554_554393

noncomputable def a : ℝ := Real.log 8 / Real.log 3
noncomputable def b : ℝ := 2 ^ 1.2
noncomputable def c : ℝ := 0.98 ^ 2.1

theorem variable_order (ha : a = Real.log 8 / Real.log 3)
                       (hb : b = 2 ^ 1.2)
                       (hc : c = 0.98 ^ 2.1) :
    c < a ∧ a < b := by
    sorry

end variable_order_l554_554393


namespace power_function_point_l554_554413

theorem power_function_point (m n: ℝ) (h: (m - 1) * m^n = 8) : n^(-m) = 1/9 := 
  sorry

end power_function_point_l554_554413


namespace number_of_paperbacks_l554_554227

theorem number_of_paperbacks (P H : ℕ) (total_books : ℕ)
  (book_condition : total_books = 8)
  (hb_condition : H = 6)
  (comb_condition : (∃ P H, ∑ (_, _, c) in Finset.range 3, choose P c + choose H c = 36))
  : P = 2  :=
  sorry

end number_of_paperbacks_l554_554227


namespace lucky_numbers_l554_554685

def is_lucky (n : ℕ) : Prop :=
  ∃ q r : ℕ, 221 = n * q + r ∧ r % q = 0

def lucky_numbers_count (a b : ℕ) : ℕ :=
  Nat.card {n // a ≤ n ∧ n ≤ b ∧ is_lucky n}

theorem lucky_numbers (a b : ℕ) (h1 : a = 1) (h2 : b = 221) :
  lucky_numbers_count a b = 115 := 
sorry

end lucky_numbers_l554_554685


namespace solve_A6C6_l554_554994

theorem solve_A6C6 :
  (∑ i in finset.range 2, (nat.factorial 6) / (nat.factorial (6 - i))) +
  ((nat.factorial 6) / ((nat.factorial 2) * (nat.factorial (6 - 2)))) = 45 :=
by
  sorry

end solve_A6C6_l554_554994


namespace intersection_eq_l554_554434

-- Define sets P and Q
def setP := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def setQ := {y : ℝ | ∃ x : ℝ, y = -x + 2}

-- The main theorem statement
theorem intersection_eq: setP ∩ setQ = {y : ℝ | y ≤ 2} :=
by
  sorry

end intersection_eq_l554_554434


namespace geometric_series_sum_l554_554526

noncomputable def first_term : ℝ := 6
noncomputable def common_ratio : ℝ := -2 / 3

theorem geometric_series_sum :
  (|common_ratio| < 1) → (first_term / (1 - common_ratio) = 18 / 5) :=
by
  intros h
  simp [first_term, common_ratio]
  sorry

end geometric_series_sum_l554_554526


namespace abs_z_squared_l554_554457

-- Definitions based on the conditions
def z : ℂ := 5 + 2 * Complex.i
def z_squared : ℂ := z^2

-- Main proof statement
theorem abs_z_squared : Complex.abs z_squared = 29 := by
  -- Proof steps are omitted (using sorry)
  sorry

end abs_z_squared_l554_554457


namespace perfect_essay_implies_pass_l554_554853

-- Define the propositions
variables (P Q : Prop)

-- Condition: If Alice writes a perfect essay, then Alice passes the course.
def if_perf_essay_then_pass : Prop := P → Q

-- Statement to prove: If Alice failed the course, then she did not write a perfect essay.
def failed_then_no_perf_essay : Prop := ¬Q → ¬P

-- The theorem statement
theorem perfect_essay_implies_pass (h : if_perf_essay_then_pass) : failed_then_no_perf_essay :=
sorry

end perfect_essay_implies_pass_l554_554853


namespace distinct_points_equality_l554_554142

theorem distinct_points_equality (n : ℕ) 
    (x : Fin (n + 2) → ℝ) 
    (h : StrictlyMonotone x)
    (p q : ℝ) : 
  let y (i : Fin (n + 2)) := p * x i + q in
  (∑ i in Finset.range (n + 1), (y i + y (Fin.succ i)) * (x (Fin.succ i) - x i)) = 
  (y 0 + y n.succ) * (x n.succ - x 0) := 
by 
  -- Provide definition and start the proof
  sorry

end distinct_points_equality_l554_554142


namespace part_one_part_two_l554_554425

section part_one
variables {x : ℝ}

def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

theorem part_one : ∀ x : ℝ, f x ≥ 3 ↔ (x ≤ 1 ∨ x ≥ 4) := by
  sorry
end part_one

section part_two
variables {a x : ℝ}

def g (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

theorem part_two : (∀ x ∈ (Set.Icc 1 2), g a x ≤ |x - 4|) → (a ∈ Set.Icc (-3) 0) := by
  sorry
end part_two

end part_one_part_two_l554_554425


namespace cost_price_of_one_meter_of_cloth_l554_554709

variable C : ℝ
variable (h1 : 50 * (C + 15) + 35 * (C + 20) = 8925)

theorem cost_price_of_one_meter_of_cloth : C = 88 :=
by
  sorry

end cost_price_of_one_meter_of_cloth_l554_554709


namespace sum_of_reciprocals_of_slopes_l554_554738

noncomputable def parabola_equation : polynomial ℝ := polynomial.X^2 - 4 * polynomial.C 1

def point_P := (4, 4)

def slope_AB := 1

-- Define point A and B and conditions for their slopes k1 and k2
def point_A (k1 : ℝ) := 
  let x := 4 * (1 - k1)^2 / k1^2 in 
  let y := 4 / k1 - 4 in 
  (x, y)

def point_B (k2 : ℝ) := 
  let x := 4 * (1 - k2)^2 / k2^2 in 
  let y := 4 / k2 - 4 in 
  (x, y)

def k_condition (k1 k2 : ℝ) :=
  k1 * k2 ≠ 0 ∧
  let (xA, yA) := point_A k1 in 
  let (xB, yB) := point_B k2 in 
  let slope_AB' := (yB - yA) / (xB - xA) in 
  slope_AB' = slope_AB

theorem sum_of_reciprocals_of_slopes (k1 k2 : ℝ) (h : k_condition k1 k2) : 
  (1 / k1) + (1 / k2) = 3 :=
sorry

end sum_of_reciprocals_of_slopes_l554_554738


namespace book_arrangements_l554_554276

theorem book_arrangements (n : ℕ) (b1 b2 b3 b4 b5 : ℕ) (h_b123 : b1 < b2 ∧ b2 < b3):
  n = 20 := sorry

end book_arrangements_l554_554276


namespace part1_l554_554386

theorem part1 (x : ℝ) (hx : x > 0) : 
  (1 / (2 * Real.sqrt (x + 1))) < (Real.sqrt (x + 1) - Real.sqrt x) ∧ (Real.sqrt (x + 1) - Real.sqrt x) < (1 / (2 * Real.sqrt x)) := 
sorry

end part1_l554_554386


namespace find_n_mod_8_l554_554379

theorem find_n_mod_8 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4702 [MOD 8] ∧ n = 2 :=
begin
  sorry
end

end find_n_mod_8_l554_554379


namespace general_term_arithmetic_seq_l554_554804

def arithmetic_seq (n : ℕ) := {a : ℕ // a = 2 * n}

theorem general_term_arithmetic_seq (a₆ S₃ : ℕ) (h₁ : a₆ = 12) (h₂ : S₃ = 12) : ∀ n : ℕ, arithmetic_seq n = 2 * n :=
by
  sorry

end general_term_arithmetic_seq_l554_554804


namespace equilateral_triangle_l554_554801

variable (R a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC_is_triangle : A + B + C = π)

-- Condition: circumradius and given equation
def circumradius_condition 
  (circumradius : R)
  (equation : (a * Real.cos A + b * Real.cos B + c * Real.cos C) / 
             (a * Real.sin B + b * Real.sin C + c * Real.sin A) = 
             (a + b + c) / (9 * R)) : Prop := 
  equation

-- Conjecture: The interior angles of the triangle are 60 degrees each
theorem equilateral_triangle 
  (h1 : triangle_ABC_is_triangle) 
  (h2 : circumradius_condition R a b c ((a * Real.cos A + b * Real.cos B + c * Real.cos C) / 
                                       (a * Real.sin B + b * Real.sin C + c * Real.sin A) = 
                                       (a + b + c) / (9 * R)))
  : A = π / 3 ∧ B = π / 3 ∧ C = π / 3 := 
sorry

end equilateral_triangle_l554_554801


namespace range_of_a_l554_554464

open Real

theorem range_of_a {a : ℝ} : (∃ x ∈ Icc (exp 1) (exp 2), (x / log x) ≤ (1 / 4 + a * x)) ↔ a ≥ (1/2 - 1/(4 * (exp 2)^2)) :=
by
  sorry

end range_of_a_l554_554464


namespace lucky_numbers_count_l554_554670

def is_lucky (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧ (221 % n) % (221 / n) = 0

def count_lucky_numbers : ℕ :=
  (Finset.range 222).filter is_lucky |>.card

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
by
  sorry

end lucky_numbers_count_l554_554670


namespace volume_ratio_upper_lower_l554_554136

-- Define the regular triangular pyramid and its properties
structure RegularTriangularPyramid :=
(P A B C O M : Type)
(height_PO : Type)
(M_midpoint_PO : ∀ (P O : Type), midpoint P O M)

-- Define the volume ratio calculation
noncomputable def volume_ratio {P A B C O M : Type} [RegularTriangularPyramid P A B C O M height_PO M_midpoint_PO] :=
let upper_volume := 4 in
let lower_volume := 21 in
upper_volume / lower_volume

-- Define the theorem statement
theorem volume_ratio_upper_lower
  {P A B C O M : Type} [pyramid : RegularTriangularPyramid P A B C O M height_PO M_midpoint_PO]:
  volume_ratio P A B C O M height_PO M_midpoint_PO = 4 / 21 :=
begin
  sorry
end

end volume_ratio_upper_lower_l554_554136


namespace percentage_both_languages_l554_554472

open BigOperators

-- Define the data for the conditions
variable (total : ℝ) (english : ℝ) (german : ℝ) (both : ℝ)

-- Set the values corresponding to the problem's conditions
def student_data : Prop := 
  total = 1 ∧
  english = 0.8 ∧
  german = 0.7

-- The correct answer as a definition based on given conditions
def answer : ℝ := 0.5

-- The theorem to prove equality given the conditions
theorem percentage_both_languages (h : student_data total english german both) : 
  both = answer :=
by
  unfold student_data at h
  cases h with ht hrest
  cases hrest with he hg
  have : both = english + german - total, by sorry
  rw [he, hg, ht] at this
  linarith

end percentage_both_languages_l554_554472


namespace geom_seq_sum_a3_a5_l554_554487

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a(n+1) = a(n) * r

theorem geom_seq_sum_a3_a5 (a : ℕ → ℝ) (h1 : a 1 > 0) 
(h2 : geom_seq a)
(h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
a 3 + a 5 = 6 :=
sorry

end geom_seq_sum_a3_a5_l554_554487


namespace lcm_18_35_is_630_l554_554005

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l554_554005


namespace small_poster_ratio_l554_554705

theorem small_poster_ratio (total_posters : ℕ) (medium_posters large_posters small_posters : ℕ)
  (h1 : total_posters = 50)
  (h2 : medium_posters = 50 / 2)
  (h3 : large_posters = 5)
  (h4 : small_posters = total_posters - medium_posters - large_posters)
  (h5 : total_posters ≠ 0) :
  small_posters = 20 ∧ (small_posters : ℚ) / total_posters = 2 / 5 := 
sorry

end small_poster_ratio_l554_554705


namespace project_recommendation_l554_554289

/-- Given the innovation and practicality scores for four projects, and the weights 
    for these scores, determine that Project B has the highest total score. -/
theorem project_recommendation :
  let innovation_A := 90
  let practicality_A := 90
  let innovation_B := 95
  let practicality_B := 90
  let innovation_C := 90
  let practicality_C := 95
  let innovation_D := 90
  let practicality_D := 85
  let weight_innovation := 0.6
  let weight_practicality := 0.4
  let score (innovation practicality : ℕ) := weight_innovation * innovation + weight_practicality * practicality
  score innovation_B practicality_B > score innovation_C practicality_C ∧
  score innovation_C practicality_C > score innovation_A practicality_A ∧
  score innovation_A practicality_A > score innovation_D practicality_D :=
by
  sorry

end project_recommendation_l554_554289


namespace abc_base_10_value_l554_554293

def map_base4_to_set (n : ℕ) : char :=
  match n with
  | 0 => 'A'
  | 1 => 'B'
  | 2 => 'C'
  | 3 => 'D'
  | _ => ' '

def encode_base4 (s : string) : ℕ :=
  let (a, b, c) := (s.get 0, s.get 1, s.get 2)
  let val (ch : char) : ℕ :=
    if ch = 'A' then 0
    else if ch = 'B' then 1
    else if ch = 'C' then 2
    else if ch = 'D' then 3
    else 0
  in val a * 4^2 + val b * 4^1 + val c * 4^0

theorem abc_base_10_value :
  encode_base4 "ABC" = 57 :=
by
  sorry

end abc_base_10_value_l554_554293


namespace solve_problem_l554_554800

noncomputable def problem_statement (α : ℝ) : Prop :=
    (α > π / 2 ∧ α < π) ∧ 
    sin α = 3 / 5 → 
    cos α = - 4 / 5 ∧ 
    tan α = - 3 / 4 ∧ 
    cos (2 * α) + sin (π + α) = - 8 / 25

theorem solve_problem (α : ℝ) : problem_statement α := 
    by
    sorry

end solve_problem_l554_554800


namespace max_wrestlers_more_than_131_l554_554335

theorem max_wrestlers_more_than_131
  (n : ℤ)
  (total_wrestlers : ℤ := 20)
  (average_weight : ℕ := 125)
  (min_weight : ℕ := 90)
  (constraint1 : n ≥ 0)
  (constraint2 : n ≤ total_wrestlers)
  (total_weight := 2500) :
  n ≤ 17 :=
by
  sorry

end max_wrestlers_more_than_131_l554_554335


namespace problem_inequality_minimum_value_maximum_value_minimum_value_conditions_reached_l554_554555

variables {α : Type*} [Field α]

theorem problem_inequality (n : ℕ) (a b : Fin n → α) :
  (Finset.univ.sum (λ i, (a i) ^ 2)) * (Finset.univ.sum (λ i, (b i) ^ 2)) ≥ (Finset.univ.sum (λ i, (a i) * (b i))) ^ 2 :=
sorry

theorem minimum_value (x y z : α) (h : x + 2 * y + 3 * z = 6) :
  x^2 + y^2 + z^2 ≥ 18 / 7 :=
sorry

theorem maximum_value (x y z : α) (h : 2 * x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ Real.sqrt 5 :=
sorry

theorem minimum_value_conditions_reached (x y z : α)
  (h : x = 3 / 7) (h2 : y = 6 / 7) (h3 : z = 9 / 7) :
  x^2 + y^2 + z^2 = 18 / 7 :=
sorry

end problem_inequality_minimum_value_maximum_value_minimum_value_conditions_reached_l554_554555


namespace count_lucky_numbers_l554_554651

def is_lucky (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 221) ∧ (221 % n % (221 / n) = 0)

theorem count_lucky_numbers : (finset.filter is_lucky (finset.range 222)).card = 115 :=
by
  sorry

end count_lucky_numbers_l554_554651


namespace find_angle_B_find_max_value_l554_554467

theorem find_angle_B (a b c A B C : ℝ) (h1 : a^2 + c^2 = b^2 + √2 * a * c)
  (h2 : 0 < B) (h3 : B < π) : B = π / 4 :=
by
  sorry

theorem find_max_value (a b c A B C : ℝ) (h1 : a^2 + c^2 = b^2 + √2 * a * c)
  (h2 : 0 < A) (h3 : A < 3 * π / 4) (h4 : B = π / 4) (h5 : C = 3 * π / 4 - A) :
  √2 * cos A + cos C ≤ 1 :=
by
  sorry

end find_angle_B_find_max_value_l554_554467


namespace problem_proof_l554_554027

def a : ℝ := Real.log 0.05 / Real.log 0.2
def b : ℝ := 0.5 ^ 1.002
def c : ℝ := 4 * Real.cos 1

theorem problem_proof : b < a ∧ a < c :=
by
  -- Conditions for a, b, and c
  let a := Real.log 0.05 / Real.log 0.2
  let b := 0.5 ^ 1.002
  let c := 4 * Real.cos 1
  
  sorry -- Proof to be filled in

end problem_proof_l554_554027


namespace tasks_completed_correctly_l554_554855

theorem tasks_completed_correctly (x y : ℕ) (h1 : 9 * x - 5 * y = 57) (h2 : x + y ≤ 15) : x = 8 := 
by
  sorry

end tasks_completed_correctly_l554_554855


namespace work_done_during_second_isothermal_process_l554_554156

open Real

variables (R : ℝ) (n : ℝ) (one_mole : n = 1) (W_iso : ℝ) (Q_iso : ℝ) (W_iso_equal_Q_iso : W_iso = Q_iso)

def isobaric_work_performed (W : ℝ) := W = 40
def heat_added_during_isobaric_process (Q : ℝ) := Q = 100

theorem work_done_during_second_isothermal_process : 
    ∀ (W_iso : ℝ), 
    (isobaric_work_performed 40) →
    (heat_added_during_isobaric_process 100) →
    (W_iso = 100) :=
by {
    intros,
    sorry
}

end work_done_during_second_isothermal_process_l554_554156


namespace ellipse_equation_line_through_focus_max_area_triangle_l554_554864

noncomputable def ellipse (a b : ℝ) : Prop := 
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def circle_tangent (b : ℝ) : Prop :=
  distance (0, 0) (line 1 (-1) (real.sqrt 2)) = b

theorem ellipse_equation (a b : ℝ) (h₁ : a > b) (h₂ : b > 0)
  (h3 : ellipse_equation a b)
  (h4 : eccentricity a b = (real.sqrt 3) / 2) 
  (h5 : circle_tangent b) 
  : ellipse a b :=
begin
  sorry
end

theorem line_through_focus (l : ℝ → ℝ)  (x y : ℝ)
  (h₁ : ellipse 2 1)
  (h₂ : ∃ m : ℝ, l = \sqrt(2) x + mx - \sqrt(6)) 
  : l (∃ x, x ∈ k) :=
begin
  sorry
end

theorem max_area_triangle (F1 F2 M N: ℝ × ℝ)
  (h₁ : ellipse 2 1)
  (h₂ : line_through_focus)
  (h3 : MF1 = 3 (F2N))
  : triangle_area F1 M N = 2 :=
begin
  sorry
end

end ellipse_equation_line_through_focus_max_area_triangle_l554_554864


namespace circle_area_eq_l554_554862

theorem circle_area_eq (x y : ℝ) (h : 2 * x^2 + 2 * y^2 + 8 * x - 4 * y - 16 = 0) : 
  ∃ (r : ℝ), r = (sqrt 13) ∧ (π * r^2 = 13 * π) := 
begin
  sorry
end

end circle_area_eq_l554_554862


namespace compute_expression_l554_554733

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l554_554733


namespace area_of_triangle_ABC_l554_554102

/-- In triangle ABC, let N and R be the midpoints of sides BC and AB respectively. Let the medians
AN and CR intersect at point O, the centroid. Let P be the midpoint of side AC, and suppose line
PR intersects CN at point Q. If the area of triangle OQR is m, then the area of triangle ABC is 27m. -/
theorem area_of_triangle_ABC (A B C N R P O Q : Point)
  (h_N : midpoint B C N) (h_R : midpoint A B R) (h_P : midpoint A C P)
  (h_AN_median : median A N) (h_CR_median : median C R)
  (h_O_centroid : centroid A B C O) (h_PR_extends : extends PR P R)
  (h_PR_intersect_CN : intersects PR CN Q)
  (h_area_OQR : area (triangle O Q R) = m) :
  area (triangle A B C) = 27 * m :=
sorry

end area_of_triangle_ABC_l554_554102


namespace average_weight_l554_554483

theorem average_weight (w_girls w_boys : ℕ) (avg_girls avg_boys : ℕ) (n : ℕ) : 
  n = 5 → avg_girls = 45 → avg_boys = 55 → 
  w_girls = n * avg_girls → w_boys = n * avg_boys →
  ∀ total_weight, total_weight = w_girls + w_boys →
  ∀ avg_weight, avg_weight = total_weight / (2 * n) →
  avg_weight = 50 :=
by
  intros h_n h_avg_girls h_avg_boys h_w_girls h_w_boys h_total_weight h_avg_weight
  -- here you would start the proof, but it is omitted as per the instructions
  sorry

end average_weight_l554_554483


namespace count_valid_three_digit_numbers_l554_554770

def remove_zeros (n : ℕ) : ℕ :=
  nat.of_digits 10 (List.filter (λ d, d ≠ 0) (nat.digits 10 n))

theorem count_valid_three_digit_numbers : 
  (Finset.card (Finset.filter (λ n, remove_zeros n < n ∧ remove_zeros n ∣ n) (Finset.Icc 100 999))) = 93 :=
by sorry

end count_valid_three_digit_numbers_l554_554770


namespace number_of_geometric_sequences_is_three_l554_554023

theorem number_of_geometric_sequences_is_three :
  ∀ (d : ℝ), (1 < 1 + d) → (1 + d < 1 + 2 * d) → (1 + 2 * d < 1 + 3 * d) →
  (1 * 1 = (1 + d) * (1 + 2 * d) ∨ (1 + d) * (1 + d) = 1 * (1 + 2 * d) ∨ (1 + 2 * d) * (1 + 2 * d) = 1 * (1 + 3 * d) ∨
   (1 * 1 = (1 + d) * (1 + 3 * d) ∨ (1 + d) * (1 + d) = 1 * (1 + 3 * d) ∨ (1 + 3 * d) * (1 + 3 * d) = 1 * (1 + d) ∨
   (1 * 1 = (1 + 2 * d) * (1 + 3 * d) ∨ (1 + 2 * d) * (1 + 2 * d) = 1 * (1 + 3 * d) ∨ (1 + 3 * d) * (1 + 3 * d) = 1 * (1 + d) ∨
   ((1 + d) * (1 + d) = (1 + 2 * d) * (1 + 3 * d)) ∨ ((1 + 2 * d) * (1 + 2 * d) = (1 + d) * (1 + 3 * d) ∨ (1 + 3 * d) * (1 + 3 * d) = (1 + d) * (1 + 2 * d)) →
  ∃ (n : ℕ), n = 3 := sorry

end number_of_geometric_sequences_is_three_l554_554023


namespace tomato_plants_per_row_l554_554339

-- Definitions based on given conditions.
variables (T C P : ℕ)

-- Condition 1: For each row of tomato plants, she is planting 2 rows of cucumbers
def cucumber_rows (T : ℕ) := 2 * T

-- Condition 2: She has enough room for 15 rows of plants in total
def total_rows (T : ℕ) (C : ℕ) := T + C = 15

-- Condition 3: If each plant produces 3 tomatoes, she will have 120 tomatoes in total
def total_tomatoes (P : ℕ) := 5 * P * 3 = 120

-- The task is to prove that P = 8
theorem tomato_plants_per_row : 
  ∀ T C P : ℕ, cucumber_rows T = C → total_rows T C → total_tomatoes P → P = 8 :=
by
  -- The actual proof will go here.
  sorry

end tomato_plants_per_row_l554_554339


namespace negation_of_proposition_l554_554205

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l554_554205


namespace average_monthly_growth_rate_price_reduction_for_profit_l554_554598

-- Given conditions
constants (cost price_in_march price_growth sales_march sales_april_may price_reduction_sales : ℕ)
noncomputable def cost := 25
noncomputable def price_in_march := 40
noncomputable def sales_march := 256
noncomputable def sales_april_may := 400
noncomputable def price_growth := 1
noncomputable def price_reduction_sales := λ reduction : ℕ, reduction * 5

-- Part 1: Prove the average monthly growth rate for April and May
theorem average_monthly_growth_rate (x : ℚ) :
  (1 + x)^2 = (400 : ℚ) / (256 : ℚ) → x = 0.25 := sorry

-- Part 2: Prove the price reduction needed to earn 4250 yuan per month
theorem price_reduction_for_profit (m : ℕ) :
  (40 - m - 25) * (400 + 5 * m) = 4250 → m = 5 := sorry

end average_monthly_growth_rate_price_reduction_for_profit_l554_554598


namespace project_recommendation_l554_554288

/-- Given the innovation and practicality scores for four projects, and the weights 
    for these scores, determine that Project B has the highest total score. -/
theorem project_recommendation :
  let innovation_A := 90
  let practicality_A := 90
  let innovation_B := 95
  let practicality_B := 90
  let innovation_C := 90
  let practicality_C := 95
  let innovation_D := 90
  let practicality_D := 85
  let weight_innovation := 0.6
  let weight_practicality := 0.4
  let score (innovation practicality : ℕ) := weight_innovation * innovation + weight_practicality * practicality
  score innovation_B practicality_B > score innovation_C practicality_C ∧
  score innovation_C practicality_C > score innovation_A practicality_A ∧
  score innovation_A practicality_A > score innovation_D practicality_D :=
by
  sorry

end project_recommendation_l554_554288


namespace part1_part2_l554_554392

-- Part (1)
theorem part1 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (arithmetic_seq : ∀ n, a_n (n+1) = a_n n + d)
  (S1_eq : S_n 1 = 5)
  (S2_eq : S_n 2 = 18) :
  ∀ n, a_n n = 3 * n + 2 := by
  sorry

-- Part (2)
theorem part2 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (geometric_seq : ∃ q, ∀ n, a_n (n+1) = q * a_n n)
  (S1_eq : S_n 1 = 3)
  (S2_eq : S_n 2 = 15) :
  ∀ n, S_n n = (3^(n+2) - 6 * n - 9) / 4 := by
  sorry

end part1_part2_l554_554392


namespace find_solutions_l554_554375

-- Define the condition for a solution
def is_solution (a b n : ℕ) : Prop := a^2 + b^2 = nat.factorial n ∧ a ≤ b ∧ 0 < a ∧ 0 < b ∧ n < 14

-- Theorem statement about all solutions to the given problem
theorem find_solutions (a b n : ℕ) :
  is_solution a b n ↔ (a = 1 ∧ b = 1 ∧ n = 2) ∨ (a = 12 ∧ b = 24 ∧ n = 6) :=
sorry

end find_solutions_l554_554375


namespace gus_eggs_l554_554820

theorem gus_eggs : 
  let eggs_breakfast := 2 in
  let eggs_lunch := 3 in
  let eggs_dinner := 1 in
  let total_eggs := eggs_breakfast + eggs_lunch + eggs_dinner in
  total_eggs = 6 :=
by
  sorry

end gus_eggs_l554_554820


namespace minimize_distances_is_k5_l554_554565

-- Define the coordinates of points A, B, and D
def A : ℝ × ℝ := (4, 3)
def B : ℝ × ℝ := (1, 2)
def D : ℝ × ℝ := (0, 5)

-- Define C as a point vertically below D, implying the x-coordinate is the same as that of D and y = k
def C (k : ℝ) : ℝ × ℝ := (0, k)

-- Prove that the value of k that minimizes the distances over AC and BC is k = 5
theorem minimize_distances_is_k5 : ∃ k : ℝ, (C k = (0, 5)) ∧ k = 5 :=
by {
  sorry
}

end minimize_distances_is_k5_l554_554565


namespace probability_of_A_being_leader_l554_554368

/-- 
  Suppose we have 12 people divided into 2 equal groups of 6 people each. In each group, 
  one person is designated as the leader and another as the deputy leader.
  Prove that the probability of person A being chosen as the leader is 1/6.
-/
theorem probability_of_A_being_leader 
  (A : Type) 
  (people : fin 12 → A) 
  (group1 group2 : fin 6 → A)
  (h_disjoint : ∀ i j, people i ≠ people j → (people i ∈ group1 ∧ people j ∈ group2) ∨ (people i ∈ group2 ∧ people j ≠ group1))
  (h_A_in_group1 : A ∈ group1) 
  (h_leader : A)
  (h_leader_selection : ∀ a ∈ group1, probability (a = h_leader) = 1/6) :
  probability (A = h_leader) = 1/6 := sorry

end probability_of_A_being_leader_l554_554368


namespace correct_statement_l554_554169

def f (x : ℝ) : ℝ := 4 * sin (2 * x + π / 3)

theorem correct_statement :
  (∃ x, f x = 0 ∧ x = π / 12) ∧
  ¬(∃ y, f y = 0 ∧ y = π / 6) ∧
  ¬((4 * sin (2 * (π / 6) + π / 3)) = 0) :=
by
  sorry

end correct_statement_l554_554169


namespace eggs_total_l554_554822

-- Definitions based on the conditions
def breakfast_eggs : Nat := 2
def lunch_eggs : Nat := 3
def dinner_eggs : Nat := 1

-- Theorem statement
theorem eggs_total : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  -- This part will be filled in with proof steps, but it's omitted here
  sorry

end eggs_total_l554_554822


namespace max_wrestlers_more_than_131_l554_554336

theorem max_wrestlers_more_than_131
  (n : ℤ)
  (total_wrestlers : ℤ := 20)
  (average_weight : ℕ := 125)
  (min_weight : ℕ := 90)
  (constraint1 : n ≥ 0)
  (constraint2 : n ≤ total_wrestlers)
  (total_weight := 2500) :
  n ≤ 17 :=
by
  sorry

end max_wrestlers_more_than_131_l554_554336


namespace solve_problem_l554_554986

noncomputable def problem :=
  ∃ (x y : ℝ), (x^2 + x * y + y^2 = 23) ∧ (x^4 + x^2 * y^2 + y^4 = 253) ∧
  -- solutions for x and y
  ((x = real.sqrt 29 ∧ y = real.sqrt 5) ∨
   (x = -real.sqrt 29 ∧ y = real.sqrt 5) ∨
   (x = real.sqrt 29 ∧ y = -real.sqrt 5) ∨
   (x = -real.sqrt 29 ∧ y = -real.sqrt 5))

theorem solve_problem : problem :=
sorry

end solve_problem_l554_554986


namespace find_longer_parallel_side_length_l554_554311

noncomputable def longer_parallel_side_length_of_trapezoid : ℝ :=
  let square_side_length : ℝ := 2
  let center_to_side_length : ℝ := square_side_length / 2
  let midline_length : ℝ := square_side_length / 2
  let equal_area : ℝ := (square_side_length^2) / 3
  let height_of_trapezoid : ℝ := center_to_side_length
  let shorter_parallel_side_length : ℝ := midline_length
  let longer_parallel_side_length := (2 * equal_area / height_of_trapezoid) - shorter_parallel_side_length
  longer_parallel_side_length

theorem find_longer_parallel_side_length : 
  longer_parallel_side_length_of_trapezoid = 5/3 := 
sorry

end find_longer_parallel_side_length_l554_554311


namespace total_hours_uploaded_l554_554717

def hours_June_1_to_10 : ℝ := 5 * 2 * 10
def hours_June_11_to_20 : ℝ := 10 * 1 * 10
def hours_June_21_to_25 : ℝ := 7 * 3 * 5
def hours_June_26_to_30 : ℝ := 15 * 0.5 * 5

def total_video_hours : ℝ :=
  hours_June_1_to_10 + hours_June_11_to_20 + hours_June_21_to_25 + hours_June_26_to_30

theorem total_hours_uploaded :
  total_video_hours = 342.5 :=
by
  sorry

end total_hours_uploaded_l554_554717


namespace sum_of_powers_of_7_divisible_by_400_l554_554177

theorem sum_of_powers_of_7_divisible_by_400 (K : ℕ) : (∑ i in finset.range (4 * K + 1), 7^i) % 400 = 0 :=
by
  sorry

end sum_of_powers_of_7_divisible_by_400_l554_554177


namespace first_player_can_ensure_odd_result_l554_554206

-- Lean statement representing the problem
theorem first_player_can_ensure_odd_result :
  ∃ strategy : (ℕ → ℕ → ℕ) → (ℕ → ℕ → ℕ), ∀ (seq : List ℕ) (operations : List (ℕ → ℕ → ℕ)),
    seq = List.range' 1 100 →
    (∀ op ∈ operations, op = (+) ∨ op = (-) ∨ op = ( * ) ) →
    seq.length = 100 →
    (∃ result, foldr (λ (n: ℕ) (acc: ℕ), (operations.get! (n - 2)) (seq.get! n) acc) 0 seq % 2 = 1) :=
begin
  sorry
end

end first_player_can_ensure_odd_result_l554_554206


namespace solution_set_quadratic_inequality_l554_554586

def quadraticInequalitySolutionSet 
  (x : ℝ) : Prop := 
  3 + 5 * x - 2 * x^2 > 0

theorem solution_set_quadratic_inequality :
  { x : ℝ | quadraticInequalitySolutionSet x } = 
  { x : ℝ | - (1:ℝ) / 2 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_quadratic_inequality_l554_554586


namespace prob_green_marble_l554_554596

-- Definitions
def total_marbles := 90
def prob_white := 1 / 3
def prob_red_or_blue := 42 / total_marbles -- Simplifying 0.4666666666666667 as 42/90
def w := total_marbles * prob_white
def rb := total_marbles * 0.4666666666666667
def g := total_marbles - (w + rb)

-- Theorem
theorem prob_green_marble :
  (g / total_marbles) = 0.2 := by
  sorry

end prob_green_marble_l554_554596


namespace find_pairs_l554_554422

noncomputable def f (k : ℤ) (x y : ℝ) : ℝ :=
  if k = 0 then 0 else (x^k + y^k + (-1)^k * (x + y)^k) / k

theorem find_pairs (x y : ℝ) (hxy : x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0) :
  ∃ (m n : ℤ), m ≠ 0 ∧ n ≠ 0 ∧ m ≤ n ∧ m + n ≠ 0 ∧ 
    (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0 → f m x y * f n x y = f (m + n) x y) :=
  sorry

end find_pairs_l554_554422


namespace coeff_x4_in_deriv_l554_554781

def f (x : ℝ) : ℝ := (x + 1) * (x^2 + 2) * (x^3 + 3)

theorem coeff_x4_in_deriv (x : ℝ) : (f' x).coeff 4 = 5 :=
by
  sorry

end coeff_x4_in_deriv_l554_554781


namespace expression_value_l554_554248

theorem expression_value (y : ℤ) (h : y = 5) : (y^2 - y - 12) / (y - 4) = 8 :=
by
  rw[h]
  sorry

end expression_value_l554_554248


namespace sum_of_solutions_l554_554522

theorem sum_of_solutions (g : ℝ → ℝ) (h : ∀ x, g x = 3 * x - 2) :
  let g_inv x := (x + 2) / 3 in
  ∑ x in {x : ℝ | g_inv x = g (x⁻¹)}, x = -8 :=
by
  -- Definitions based on conditions
  let g_inv (x : ℝ) := (x + 2) / 3
  have inv_eq := (λ x, g_inv x = g (x⁻¹))

  -- Proof setup
  sorry

end sum_of_solutions_l554_554522


namespace general_term_l554_554584

noncomputable def sequence (n : ℕ) : ℚ :=
match n with
| 0     := 1
| (k+1) := (k^2 + k) * (sequence k) / (3 * (sequence k) + k^2 + k)

theorem general_term (n : ℕ) (h : n > 0): (sequence n) = n / (4 * n - 3) :=
by
  sorry

end general_term_l554_554584


namespace probability_multiple_of_6_or_8_or_both_l554_554928

theorem probability_multiple_of_6_or_8_or_both (n : ℕ) (h : 1 ≤ n ∧ n ≤ 60) :
  (∃ k, n = 6 * k ∨ n = 8 * k ∨ n = 24 * k) → 
  (∃ p q r, n/15 = 1/4) :=
sorry

end probability_multiple_of_6_or_8_or_both_l554_554928


namespace minimum_value_expression_l554_554135

def given_expression (x y z : ℝ) : ℝ :=
  ((2 * x^2 + 5 * x + 2) * (2 * y^2 + 5 * y + 2) * (2 * z^2 + 5 * z + 2)) / (x * y * z * (1 + x) * (1 + y) * (1 + z))

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ x y z : ℝ, given_expression x y z = 729 / 8 :=
sorry

end minimum_value_expression_l554_554135


namespace count_lucky_numbers_l554_554678

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 221 ∧ let q := 221 / n in let r := 221 % n in r % q = 0

theorem count_lucky_numbers : ∃ (N : ℕ), N = 115 ∧ (∀ n, is_lucky n → n ≤ 221) := 
by
  apply Exists.intro 115
  constructor
  · rfl
  · intros n hn
    cases hn with hpos hrange
    exact hrange.left
  · sorry

end count_lucky_numbers_l554_554678


namespace probability_product_multiple_of_12_is_1_over_6_l554_554847

variable (s : Finset ℕ) (h : s = {2, 3, 6, 9})

/-- The probability that the product of two randomly chosen numbers from the set {2, 3, 6, 9}
    will be a multiple of 12 is 1/6. -/
theorem probability_product_multiple_of_12_is_1_over_6 :
  (∃ p : ℚ, p = 1/6 ∧
    ∀ (a b ∈ s) (ha : a ≠ b), (a * b) % 12 = 0 ↔ p = 1 / 6) :=
by
  exists sorry

end probability_product_multiple_of_12_is_1_over_6_l554_554847


namespace chord_length_of_intersection_l554_554053

-- Definitions
def line_equation (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem chord_length_of_intersection (A B : ℝ × ℝ) (r : ℝ)
  (hA : line_equation A.1 A.2 ∧ circle_equation A.1 A.2)
  (hB : line_equation B.1 B.2 ∧ circle_equation B.1 B.2)
  (hr : r = 1)
  : dist A B = 2 * sqrt (1 - (2 / sqrt 5)^2) := sorry

end chord_length_of_intersection_l554_554053


namespace profit_divided_equally_l554_554541

noncomputable def Mary_investment : ℝ := 800
noncomputable def Mike_investment : ℝ := 200
noncomputable def total_profit : ℝ := 2999.9999999999995
noncomputable def Mary_extra : ℝ := 1200

theorem profit_divided_equally (E : ℝ) : 
  (E / 2 + 4 / 5 * (total_profit - E)) - (E / 2 + 1 / 5 * (total_profit - E)) = Mary_extra →
  E = 1000 :=
  by sorry

end profit_divided_equally_l554_554541


namespace train_crossing_time_l554_554060

theorem train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 150 ∧ train_speed_kmph = 36 → (train_length / (train_speed_kmph * (1000 / 3600))) = 15 := 
by 
  intros h,
  cases h with h_len h_speed,
  rw [h_len, h_speed],
  simp,
  norm_num,
  sorry

end train_crossing_time_l554_554060


namespace find_fx_l554_554798

theorem find_fx (f : ℝ → ℝ) (h : ∀ x, x ≥ 0 → f (x + 1) = x^2 + 2 * real.sqrt 2) :
  ∀ x, x ≥ 1 → f x = x + 2 * real.sqrt 2 :=
begin
  sorry
end

end find_fx_l554_554798


namespace middle_number_is_five_l554_554599

theorem middle_number_is_five
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 20)
  (h_sorted : a < b ∧ b < c)
  (h_bella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → x = a → y = b ∧ z = c)
  (h_della : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → y = b → x = a ∧ z = c)
  (h_nella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → z = c → x = a ∧ y = b) :
  b = 5 := sorry

end middle_number_is_five_l554_554599


namespace triangle_shape_l554_554961

section
variables (BA BC : ℝ × ℝ)

def is_scalene (a b c : ℝ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_right_triangle (AC BC : ℝ × ℝ) := AC.1 * BC.1 + AC.2 * BC.2 = 0

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def triangle_shape_condition (BA BC : ℝ × ℝ) : Prop :=
  let AB := (-BA.1, -BA.2) in
  let AC := (AB.1 + BC.1, AB.2 + BC.2) in
  is_right_triangle AC BC ∧ is_scalene (vector_magnitude AB) (vector_magnitude BC) (vector_magnitude AC)

theorem triangle_shape (BA BC : ℝ × ℝ) (hBA : BA = (4, -3)) (hBC : BC = (2, -4)) : 
  triangle_shape_condition BA BC :=
by {
  -- Note: This is where the proof would go.
  sorry
}
end

end triangle_shape_l554_554961


namespace closest_approx_of_q_l554_554244

theorem closest_approx_of_q :
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  abs (q - 9.24) < 0.005 := 
by 
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  sorry

end closest_approx_of_q_l554_554244


namespace stratified_sampling_l554_554638

theorem stratified_sampling (total_employees senior_titles intermediate_titles junior_titles sample_size : ℕ) 
  (h_total: total_employees = 150) 
  (h_senior: senior_titles = 45) 
  (h_intermediate: intermediate_titles = 90) 
  (h_junior: junior_titles = 15) 
  (h_sample: sample_size = 30) : 
  let sampled_senior := sample_size * senior_titles / total_employees,
      sampled_intermediate := sample_size * intermediate_titles / total_employees,
      sampled_junior := sample_size * junior_titles / total_employees in
  (sampled_senior = 9) ∧ (sampled_intermediate = 18) ∧ (sampled_junior = 3) :=
by {
  sorry
}

end stratified_sampling_l554_554638


namespace smallest_b_even_l554_554765

theorem smallest_b_even (b : ℕ) (h1 : 0 < b) : (∀ x : ℤ, even (x^4 + b^3 + b^2)) ↔ b = 1 :=
by
  sorry

end smallest_b_even_l554_554765


namespace union_of_A_and_B_l554_554404

-- Definition of the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := ∅

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = {1, 2} := 
by sorry

end union_of_A_and_B_l554_554404


namespace evaluate_i_thirteen_i_eighteen_i_twenty_three_i_twenty_eight_i_thirty_three_l554_554758

-- Define i as a complex number with its powers cycling every 4 steps
noncomputable def i : ℂ := complex.I

-- State the three conditions given in the problem
axiom i_squared : i^2 = -1
axiom i_cubed : i^3 = -i
axiom i_fourth : i^4 = 1

-- The proof problem statement
theorem evaluate_i_thirteen_i_eighteen_i_twenty_three_i_twenty_eight_i_thirty_three :
  i^13 + i^18 + i^23 + i^28 + i^33 = i :=
by
  sorry

end evaluate_i_thirteen_i_eighteen_i_twenty_three_i_twenty_eight_i_thirty_three_l554_554758


namespace more_girls_than_boys_l554_554478

def total_pupils : ℕ := 1455
def girls : ℕ := 868
def boys : ℕ := total_pupils - girls

theorem more_girls_than_boys : girls - boys = 281 :=
by 
  calc 
  girls - boys 
      = 868 - boys : by rfl
  ... = 868 - (total_pupils - girls) : by rfl
  ... = 868 - (1455 - 868) : by rfl
  ... = 281 : by decide

end more_girls_than_boys_l554_554478


namespace find_a_l554_554524

noncomputable def complex_a : ℝ :=
let i := complex.I in
let lhs := (2 + a * i) / (1 + sqrt 2 * i) in
let rhs := - sqrt 2 * i in
a = - sqrt 2

theorem find_a (a : ℝ) : 
  let i := complex.I in 
  (2 + a * i) / (1 + sqrt 2 * i) = - sqrt 2 * i → 
  a = - sqrt 2 := 
by 
  intro h,
  sorry

end find_a_l554_554524


namespace combined_markup_percentage_is_correct_l554_554297

-- Definitions
def cost_apples : ℝ := 30
def cost_oranges : ℝ := 40
def cost_bananas : ℝ := 50

def markup_apples : ℝ := 0.10 * cost_apples
def markup_oranges : ℝ := 0.15 * cost_oranges
def markup_bananas : ℝ := 0.20 * cost_bananas

def total_cost : ℝ := cost_apples + cost_oranges + cost_bananas
def total_markup : ℝ := markup_apples + markup_oranges + markup_bananas
def total_selling_price : ℝ := total_cost + total_markup

def combined_markup_percentage : ℝ := (total_markup / total_selling_price) * 100

-- The theorem to be proven
theorem combined_markup_percentage_is_correct : 
  combined_markup_percentage ≈ 13.67 :=
sorry

end combined_markup_percentage_is_correct_l554_554297


namespace total_pencils_l554_554953

theorem total_pencils (pencils_in_drawer : ℕ) (initial_pencils_on_desk : ℕ) (added_pencils : ℕ) :
  pencils_in_drawer = 43 → initial_pencils_on_desk = 19 → added_pencils = 16 →
  pencils_in_drawer + (initial_pencils_on_desk + added_pencils) = 78 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_pencils_l554_554953


namespace number_of_lines_passing_through_P_l554_554590

-- Define the point P
def P : ℝ × ℝ := (-2, 3)

-- Define the condition for the area of the triangle formed with the coordinate axes
def area_condition (m b : ℝ) : Prop :=
  0.5 * |b * m| = 24

-- State the theorem: The number of such lines passing through P forming an area of 24 with the axes is 4
theorem number_of_lines_passing_through_P (m : ℝ) (b : ℝ) (hP : P = ( -2, 3 )) :
  area_condition m b Σ (number_of_lines : ℕ) (h : number_of_lines = 4),
  ∃ m b, area_condition m b ∧ h := sorry

end number_of_lines_passing_through_P_l554_554590


namespace total_crayons_l554_554217

def original_crayons := 41
def added_crayons := 12

theorem total_crayons : original_crayons + added_crayons = 53 := by
  sorry

end total_crayons_l554_554217


namespace lucky_numbers_count_l554_554662

def isLucky (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≠ 0 ∧ 221 = n * k + (221 % n) ∧ (221 % n) % k = 0

def countLuckyNumbers : ℕ :=
  Nat.card (Finset.filter isLucky (Finset.range 222))

theorem lucky_numbers_count : countLuckyNumbers = 115 :=
  sorry

end lucky_numbers_count_l554_554662


namespace division_result_l554_554525

def m : ℕ := 16 ^ 2024

theorem division_result : m / 8 = 8 * 16 ^ 2020 :=
by
  -- sorry for the actual proof
  sorry

end division_result_l554_554525


namespace cody_paid_amount_l554_554347

/-- Cody buys $40 worth of stuff,
    the tax rate is 5%,
    he receives an $8 discount after taxes,
    and he and his friend split the final price equally.
    Prove that Cody paid $17. -/
theorem cody_paid_amount
  (initial_cost : ℝ)
  (tax_rate : ℝ)
  (discount : ℝ)
  (final_split : ℝ)
  (H1 : initial_cost = 40)
  (H2 : tax_rate = 0.05)
  (H3 : discount = 8)
  (H4 : final_split = 2) :
  (initial_cost * (1 + tax_rate) - discount) / final_split = 17 :=
by
  sorry

end cody_paid_amount_l554_554347


namespace positive_diff_probability_fair_coin_l554_554608

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end positive_diff_probability_fair_coin_l554_554608


namespace perpendicular_vectors_l554_554817

theorem perpendicular_vectors (k : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (0, 2)) 
  (hb : b = (Real.sqrt 3, 1)) 
  (h : (a.1 - k * b.1) * (k * a.1 + b.1) + (a.2 - k * b.2) * (k * a.2 + b.2) = 0) :
  k = -1 ∨ k = 1 :=
sorry

end perpendicular_vectors_l554_554817


namespace each_charity_gets_45_dollars_l554_554116

def dozens : ℤ := 6
def cookies_per_dozen : ℤ := 12
def total_cookies : ℤ := dozens * cookies_per_dozen
def selling_price_per_cookie : ℚ := 1.5
def cost_per_cookie : ℚ := 0.25
def profit_per_cookie : ℚ := selling_price_per_cookie - cost_per_cookie
def total_profit : ℚ := profit_per_cookie * total_cookies
def charities : ℤ := 2
def amount_per_charity : ℚ := total_profit / charities

theorem each_charity_gets_45_dollars : amount_per_charity = 45 := 
by
  sorry

end each_charity_gets_45_dollars_l554_554116


namespace geometric_sequence_S4_l554_554871

noncomputable def geometric_sequence_sum {α : Type*} (q a_1 : α) (n : ℕ) [OrderedRing α] : α := 
  a_1 * (q^n - 1) / (q - 1)

theorem geometric_sequence_S4 {α : Type*} [LinearOrderedField α] :
  ∀ (a_1 a_2 a_3 a_4 : α) (S_3 S_4 : α) (q : α),
  q > 0 →
  a_2 = a_1 * q →
  a_3 = a_1 * q^2 →
  a_4 = a_1 * q^3 →
  S_3 = a_1 + a_2 + a_3 →
  2 * S_3 = 8 * a_1 + 3 * a_2 →
  a_4 = 16 →
  S_4 = geometric_sequence_sum q a_1 4 →
  S_4 = 30 := sorry

end geometric_sequence_S4_l554_554871


namespace lcm_18_35_l554_554002

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l554_554002


namespace closest_point_on_line_l554_554764

theorem closest_point_on_line {x y : ℝ} (h : y = x / 3) :
  ∃ (x y : ℝ), y = x / 3 ∧ (x, y) = (87 / 10, 29 / 10) :=
begin
  use (87 / 10, 29 / 10),
  split,
  { exact rfl },
  { split; refl },
  sorry
end

end closest_point_on_line_l554_554764


namespace units_digit_8th_group_l554_554858

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_8th_group (t k : ℕ) (ht : t = 7) (hk : k = 8) : 
  units_digit (t + k) = 5 := 
by
  -- Proof step will go here.
  sorry

end units_digit_8th_group_l554_554858


namespace sum_of_f_l554_554131

def f : ℕ → ℚ
| 0     := 0
| (n+1) := f n + 1 / (n + 1)

theorem sum_of_f (n : ℕ) (hn : n ≥ 2) : 
  (∑ i in Finset.range (n-1), f (i+1)) = n * (f n - 1) := by
  sorry

end sum_of_f_l554_554131


namespace solve_problem_l554_554843

variable (f : ℝ → ℝ)

axiom f_property : ∀ x : ℝ, f (x + 1) = x^2 - 2 * x

theorem solve_problem : f 2 = -1 :=
by
  sorry

end solve_problem_l554_554843


namespace crayon_count_after_actions_l554_554223

theorem crayon_count_after_actions (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) :
  initial_crayons = 48 → kiley_fraction = 1 / 4 → joe_fraction = 1 / 2 → 
  let crayons_after_kiley := initial_crayons - (kiley_fraction * initial_crayons).to_nat;
      crayons_after_joe := crayons_after_kiley - (joe_fraction * crayons_after_kiley).to_nat
  in crayons_after_joe = 18 :=
by 
  intros h1 h2 h3;
  sorry

end crayon_count_after_actions_l554_554223


namespace remainder_three_n_l554_554981

theorem remainder_three_n (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 :=
by
  sorry

end remainder_three_n_l554_554981


namespace valid_pairings_around_circle_l554_554857

theorem valid_pairings_around_circle : 
  ∃ (pairs : finset (finset (ℕ × ℕ))),
    (∀ (pair ∈ pairs), ∃ i, pair = {(i, i+1), (i, i-1), (i, i+6), (i, i-6)}.filter (λ j, j % 12 ∈ pair ∧ j.snd % 12 ∈ pair)) ∧
    pairs.card = 6 →
    ∃! pairs, 
      (∀ pair ∈ pairs, ∃ i, pair ∈ {(i, (i+6) % 12)} ∨ pair ∈ {(i, (i-6) % 12)}) :=
sorry

end valid_pairings_around_circle_l554_554857


namespace parallel_lines_distance_l554_554058

theorem parallel_lines_distance {c : ℝ} :
    (∃ x y : ℝ, 6 * x + 8 * y + 10 = 0 ∧ 6 * x + 8 * y + c = 0) →
    (∀ A B : ℝ, A A + B B ≠ 0) →
    |10 - c| / Real.sqrt (6^2 + 8^2) = 2 →
    (c = -10 ∨ c = 30) :=
by
  intros
  sorry

end parallel_lines_distance_l554_554058


namespace negation_of_existential_l554_554574

theorem negation_of_existential :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 - 3 > 0) = (∀ x : ℝ, x^2 + 2 * x - 3 ≤ 0) := 
by
  sorry

end negation_of_existential_l554_554574


namespace tetrahedron_circumradius_l554_554480

theorem tetrahedron_circumradius
  (A B C D : ℝ^3)
  (h_AB : dist A B = 2)
  (h_AC : dist A C = 2)
  (h_AD : dist A D = 2)
  (h_BC : dist B C = 3)
  (h_BD : dist B D = 2)
  (h_CD : dist C D = 2) :
  ∃ r : ℝ, r = sqrt (7 / 3) :=
  sorry

end tetrahedron_circumradius_l554_554480


namespace original_triangle_area_l554_554938

-- Define the variables
variable (A_new : ℝ) (r : ℝ)

-- The conditions from the problem
def conditions := r = 5 ∧ A_new = 100

-- Goal: Prove that the original area is 4
theorem original_triangle_area (A_orig : ℝ) (h : conditions r A_new) : A_orig = 4 := by
  sorry

end original_triangle_area_l554_554938


namespace quadratic_eq_sol_l554_554560

theorem quadratic_eq_sol (x : ℂ) : x^2 - 6 * x + 2 = 0 ↔ x = 3 + complex.sqrt 7 ∨ x = 3 - complex.sqrt 7 :=
by sorry

end quadratic_eq_sol_l554_554560


namespace mean_median_difference_l554_554471

def percentage_15 := 0.15
def percentage_40 := 0.40
def percentage_20 := 0.20
def percentage_25 := 1 - (percentage_15 + percentage_40 + percentage_20)

def score_65 := 65
def score_75 := 75
def score_85 := 85
def score_95 := 95

def mean_score := (percentage_15 * score_65) + (percentage_40 * score_75) +
                  (percentage_20 * score_85) + (percentage_25 * score_95)

def median_score := score_75

def score_difference := mean_score - median_score

theorem mean_median_difference : score_difference = 5.5 := by
  unfold mean_score
  unfold median_score
  unfold score_difference
  norm_num
  sorry

end mean_median_difference_l554_554471


namespace sum_fractions_inequality_l554_554836

noncomputable def sum_fractions (xs : List ℕ) : ℝ :=
(List.sum (List.map (λ k, Real.sqrt ((xs.get! (k+1))- (xs.get! k)) / (xs.get! (k+1))) (List.range (xs.length - 1))))

noncomputable def harmonic_series (n : ℕ) : ℝ :=
  (1 + (List.sum (List.map (λ k, 1/(k:ℝ)) (List.range (n^2 - 1)))))

theorem sum_fractions_inequality {n : ℕ} (xs : List ℕ) (h : xs.length = n + 1) (h_sorted : List.sorted (<) xs) :
  sum_fractions xs < harmonic_series n :=
sorry

end sum_fractions_inequality_l554_554836


namespace count_lucky_numbers_l554_554649

def is_lucky (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 221) ∧ (221 % n % (221 / n) = 0)

theorem count_lucky_numbers : (finset.filter is_lucky (finset.range 222)).card = 115 :=
by
  sorry

end count_lucky_numbers_l554_554649


namespace tangent_lines_l554_554395

variables (x y : ℝ)

def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def is_point_on_line (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem tangent_lines (h : x^2 + y^2 = 4) (P : ℝ × ℝ) (Pt : P = (-2, -3)) :
  (is_point_on_line 1 0 (-2) x y → circle x y) ∧
  (is_point_on_line 5 (-12) (-26) x y → circle x y) :=
sorry

end tangent_lines_l554_554395


namespace total_price_of_hats_l554_554962

-- Declare the conditions as Lean definitions
def total_hats : Nat := 85
def green_hats : Nat := 38
def blue_hat_cost : Nat := 6
def green_hat_cost : Nat := 7

-- The question becomes proving the total cost of the hats is $548
theorem total_price_of_hats :
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  total_blue_cost + total_green_cost = 548 := by
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  show total_blue_cost + total_green_cost = 548
  sorry

end total_price_of_hats_l554_554962


namespace impossible_values_of_d_l554_554580

-- Given definitions
variables {d t l w : ℝ}
def triangle_perimeter := 3 * t
def rectangle_perimeter := 2 * l + 2 * w
def rectangle_length := 2 * w
def triangle_side := l + d

-- Conditions
def condition1 : Prop := triangle_perimeter - rectangle_perimeter = 2016
def condition2 : Prop := t - l = d
def condition3 : Prop := l = 2 * w
def condition4 : Prop := 2 * (w + l) > 0

-- Theorem statement
theorem impossible_values_of_d : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → ∃ (d : ℕ), d ≠ 672 :=
begin
  sorry
end

end impossible_values_of_d_l554_554580


namespace hyperbola_foci_coordinates_l554_554191

theorem hyperbola_foci_coordinates :
  let a : ℝ := Real.sqrt 7
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  (c = Real.sqrt 10 ∧
  ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0))) :=
by
  let a := Real.sqrt 7
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  have hc : c = Real.sqrt 10 := sorry
  have h_foci : ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0)) := sorry
  exact ⟨hc, h_foci⟩

end hyperbola_foci_coordinates_l554_554191


namespace tetrahedron_acute_vertex_exists_l554_554951

theorem tetrahedron_acute_vertex_exists (T : geom3d.Tetrahedron) : 
  ∃ v : geom3d.Point, ∀ α ∈ T.vertex_angles v, α < π / 2 :=
sorry

end tetrahedron_acute_vertex_exists_l554_554951


namespace cesaro_sum_extension_l554_554907

theorem cesaro_sum_extension (A : Fin 110 → ℝ) (S : ℝ)
  (hS : S = 1200 * 110)
  (h_cesaro : (∑ i in Finset.range 110, (i + 1) * A ⟨i, Fin.is_lt i 110⟩) / 110 = 1200) :
  (∑ i in Finset.range 111, (∑ j in Finset.range i, A ⟨j, Fin.is_lt j 110⟩)) / 111 = 132000 / 111 :=
by 
  sorry

end cesaro_sum_extension_l554_554907


namespace friendship_configurations_l554_554714

-- Defining the individual entities
inductive Individual
| Adam | Benin | Chiang | Deshawn | Esther | Fiona
deriving DecidableEq

open Individual

-- Defining friendship as a relation
def is_friend (a b : Individual) : Prop := sorry -- Friendship relation

-- There are 6 individuals
def individuals := [Adam, Benin, Chiang, Deshawn, Esther, Fiona]

-- Check if the list is a valid set of individuals
def valid_individuals (l : List Individual) : Prop :=
  ∀ x, x ∈ l

-- Each individual has same number of friends
def same_number_of_friends (n : ℕ) : Prop :=
  ∀ x : Individual, (l.countp (is_friend x)) = n

-- Determine the number of valid configurations
def number_of_valid_configurations : ℕ :=
  sorry

-- The theorem we need to prove
theorem friendship_configurations : number_of_valid_configurations = 170 :=
  sorry

end friendship_configurations_l554_554714


namespace original_houses_count_l554_554229

namespace LincolnCounty

-- Define the constants based on the conditions
def houses_built_during_boom : ℕ := 97741
def houses_now : ℕ := 118558

-- Statement of the theorem
theorem original_houses_count : houses_now - houses_built_during_boom = 20817 := 
by sorry

end LincolnCounty

end original_houses_count_l554_554229


namespace max_n_value_of_arithmetic_sequence_l554_554790

variables {A : Π (x : ℝ) (y : ℝ), x = sqrt 5 ∧ y = 0}
variables {P : ℕ → ℝ × ℝ}
variables {curve : Π (x : ℝ), x^2 / 4 - 1}
variables {d : ℝ}

-- Existential statement for points P_i, and proving the maximum n
theorem max_n_value_of_arithmetic_sequence
  (hA : A = (sqrt 5, 0))
  (hx : ∀ i : ℕ, 2 ≤ (P i).1 ∧ (P i).1 ≤ 2 * sqrt 5)
  (hy : ∀ i : ℕ, (P i).2 = sqrt ((P i).1 ^ 2 / 4 - 1))
  (hd : d ∈ (1/5 : ℝ, 1/sqrt 5))
  (h_seq : ∀ i, i < n → real.dist (P (i+1)) A = real.dist (P i) A + d) :
  n ≤ 14 := 
sorry

end max_n_value_of_arithmetic_sequence_l554_554790


namespace sequence_property_l554_554813

theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = 1 →
  (∀ n, n ≥ 2 → a n + a (n - 1) = (1 / 3) ^ n) →
  (∀ n, S n = ∑ i in finset.range n, a (i + 1) * 3 ^ (i + 1)) →
  ∀ n, 4 * S n - a n * 3 ^ (n + 1) = 
  if n = 1 then -5 else n + 2 :=
by
  intros h₁ h₂ h₃ n
  sorry

end sequence_property_l554_554813


namespace prob_both_correct_l554_554626

def prob_A : ℤ := 70
def prob_B : ℤ := 55
def prob_neither : ℤ := 20

theorem prob_both_correct : (prob_A + prob_B - (100 - prob_neither)) = 45 :=
by
  sorry

end prob_both_correct_l554_554626


namespace main_problem_l554_554327

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f (x) < f (y)
def func_A (x : ℝ) := x + 1
def func_B (x : ℝ) := -x^2
def func_C (x : ℝ) := 1 / x
def func_D (x : ℝ) := x * abs x

theorem main_problem :
  (is_odd func_D ∧ is_increasing func_D) ∧
  ¬ (is_odd func_A ∧ is_increasing func_A) ∧
  ¬ (is_odd func_B ∧ is_increasing func_B) ∧
  ¬ (is_odd func_C ∧ is_increasing func_C) :=
by
  sorry

end main_problem_l554_554327


namespace range_of_values_a_angle_between_vectors_l554_554274

-- Part 1: Range of values for a
theorem range_of_values_a (Z a : ℂ) (H1 : Re (Z + 2 * complex.I) = Z + 2 * complex.I)
    (H2 : Re (Z / (2 - complex.I)) = Z / (2 - complex.I))
    (H3 : Re ((Z + a * complex.I)^2) = (Z + a * complex.I)^2)
    (H4 : Im ((Z + a * complex.I)^2) > 0):
  2 < a ∧ a < 6 :=
sorry

-- Part 2: Angle between vectors
theorem angle_between_vectors (z1 z2 : ℂ): (z1 = 3 ∧ z2 = -5 + 5 * complex.I) →
    angle (vec_of_complex z1) (vec_of_complex z2) = 3 / 4 * real.pi :=
sorry

end range_of_values_a_angle_between_vectors_l554_554274


namespace sum_of_arithmetic_sequence_has_remainder_2_l554_554969

def arithmetic_sequence_remainder : ℕ := 
  let first_term := 1
  let common_difference := 6
  let last_term := 259
  -- Calculate number of terms
  let n := (last_term + 5) / common_difference
  -- Sum of remainders of each term when divided by 6
  let sum_of_remainders := n * 1
  -- The remainder when this sum is divided by 6
  sum_of_remainders % 6 
theorem sum_of_arithmetic_sequence_has_remainder_2 : 
  arithmetic_sequence_remainder = 2 := by 
  sorry

end sum_of_arithmetic_sequence_has_remainder_2_l554_554969


namespace number_of_pairs_l554_554578

theorem number_of_pairs : 
  (∀ n m : ℕ, (1 ≤ m ∧ m ≤ 2012) → (5^n < 2^m ∧ 2^m < 2^(m+2) ∧ 2^(m+2) < 5^(n+1))) → 
  (∃ c, c = 279) :=
by
  sorry

end number_of_pairs_l554_554578


namespace equation_solution_l554_554893

/-- 
  Let / stand for - , * stand for / , + stand for * , - stand for +, and ^ stand for ** (exponential). 
  Given x = 4, prove that the equation 9 - 8 / 7 * 5 ** x + 10 evaluates to approximately 18.99817143.
--/
theorem equation_solution (x : ℝ) (h : x = 4) : 
  9 - 8 / (7 * 5 ** x) + 10 ≈ 18.99817143 :=
by {
  sorry
}

end equation_solution_l554_554893


namespace lucky_numbers_count_l554_554675

def is_lucky (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧ (221 % n) % (221 / n) = 0

def count_lucky_numbers : ℕ :=
  (Finset.range 222).filter is_lucky |>.card

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
by
  sorry

end lucky_numbers_count_l554_554675


namespace difference_between_max_and_min_l554_554592

theorem difference_between_max_and_min : 
  let nums := {10, 11, 12, 13, 14}
  max nums - min nums = 4 :=
by {
  have h_max : max nums = 14 := by sorry,
  have h_min : min nums = 10 := by sorry,
  rw [h_max, h_min],
  norm_num,
}

end difference_between_max_and_min_l554_554592


namespace distance_to_focus_l554_554072

open Real

theorem distance_to_focus {P : ℝ × ℝ} 
  (h₁ : P.2 ^ 2 = 4 * P.1)
  (h₂ : abs (P.1 + 3) = 5) :
  dist P ⟨1, 0⟩ = 3 := 
sorry

end distance_to_focus_l554_554072


namespace solve_inequality_l554_554768

theorem solve_inequality (x : ℝ) : 2 ≤ |x - 5| ∧ |x - 5| ≤ 8 ↔ x ∈ (set.Icc (-3) 3 ∪ set.Icc 7 13) :=
by
  sorry

end solve_inequality_l554_554768


namespace measure_angle_GDA_l554_554095

noncomputable def is_square (s : set point) : Prop :=
  ∃ A B C D : point, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ -- Distinct vertices
  s = {A, B, C, D} ∧
  ∃ side : ℝ, side > 0 ∧
  dist A B = side ∧ dist B C = side ∧ dist C D = side ∧ dist D A = side ∧
  -- Right angles implied
  (A - B) ⊥ (C - B) ∧ (B - C) ⊥ (D - C) ∧ (C - D) ⊥ (A - D) ∧ (D - A) ⊥ (B - A)

def is_triangle_CDE (C D E : point) : Prop :=
  ∃ A B : ℝ, A + B = 180 ∧ -- Sum of angles in a triangle always 180°
  angle D C E = 45 ∧       -- ∠CDE
  angle C D E = 75 ∧       -- ∠CDE
  angle C E D = 60         -- ∠CED

theorem measure_angle_GDA {A B C D E F G : point} :
  is_square ({A, B, C, D}) →
  is_square ({D, E, F, G}) →
  is_triangle_CDE C D E →
  measure (angle G D A) = 105 :=
by
  sorry

end measure_angle_GDA_l554_554095


namespace range_of_k_for_ellipse_l554_554461

theorem range_of_k_for_ellipse (k : ℝ) :
  (k < 9) ∧ (k > 1) ∧ (2 * k > 10) ↔ (5 < k) ∧ (k < 9) := by
sorly

end range_of_k_for_ellipse_l554_554461


namespace max_even_numbers_among_a_l554_554546

theorem max_even_numbers_among_a (a : List ℕ) (h1 : a.length = 100)
  (h2 : ∃ n, n = 99 ∧ ∃ f : ℕ → Boolean,
     (∀ i, i < 99 → (f i = true → ((a.take i).product + (a.drop (i+1)).product) % 2 = 0) ∧
                  (f i = false → ((a.take i).product + (a.drop (i+1)).product) % 2 ≠ 0)) ∧
      (f.count true = 32)) :
  ∃ n, n = 33 ∧ (∃ evens_33 : List ℕ, evens_33.length = 33 ∧ evens_33.forall (λ x, x % 2 = 0)) := sorry

end max_even_numbers_among_a_l554_554546


namespace find_a_l554_554100

open Real

theorem find_a
  (a : ℝ)
  (ha : a > 0)
  (hC1 : ∃ (ρ θ : ℝ), ρ * (√2 * cos θ + sin θ) = 1)
  (hC2 : ∃ (ρ : ℝ), ρ = a)
  (intersection_on_polar_axis : ∃ (x : ℝ), x = √2 / 2 ∧ x^2 + 0^2 = a^2) :
  a = √2 / 2 :=
sorry

end find_a_l554_554100


namespace probability_p_lt_q_lt_r_l554_554313

theorem probability_p_lt_q_lt_r 
  (std_die : Finset ℕ := {1, 2, 3, 4, 5, 6}) 
  (roll1 roll2 roll3 : ℕ) 
  (H1 : roll1 ∈ std_die) 
  (H2 : roll2 ∈ std_die) 
  (H3 : roll3 ∈ std_die) :
  (↑(Fintype.card {x // x ∈ std_die} : ℝ) := 6) → 
  (Fintype.card (finset.powerset_len 3 std_die)) = 20 →
  (∀ p q r, p ∈ std_die ∧ q ∈ std_die ∧ r ∈ std_die ∧ p < q ∧ q < r ↔ ∃ s ∈ Finset.powerset_len 3 std_die, 
    ∃ h : p ∈ s ∧ q ∈ s ∧ r ∈ s, (p < q ∧ q < r)) → 
  (↑(Fintype.card {s | s ∈ Finset.powerset_len 3 std_die ∧ (∀ x y z ∈ s, x < y < z)} : ℝ) / (6^3 : ℝ)) = 5 / 54 := sorry

end probability_p_lt_q_lt_r_l554_554313


namespace max_sumo_wrestlers_l554_554338

/-- 
At a sumo wrestling tournament, 20 sumo wrestlers participated.
The average weight of the wrestlers is 125 kg.
Individuals weighing less than 90 kg cannot participate.
Prove that the maximum possible number of sumo wrestlers weighing more than 131 kg is 17.
-/
theorem max_sumo_wrestlers : 
  ∀ (weights : Fin 20 → ℝ), 
    (∀ i, weights i ≥ 90) → 
    (∑ i, weights i = 2500) → 
    (∃ n : ℕ,  n ≤ 20 ∧ 
      (n = 17 → (∑ i in Finset.filter (λ i, weights i > 131) Finset.univ).card = n) ∧ 
      ∀ m, m > 17 → (∀ j ∈ Finset.filter (λ i, weights i > 131) Finset.univ, m = (Finset.card (Finset.filter (λ i, weights i > 131) Finset.univ) + j) → False))
:= sorry

end max_sumo_wrestlers_l554_554338


namespace largest_possible_square_area_l554_554701

def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 6
def largest_square_side : ℕ := rectangle_width
def largest_square_area : ℕ := largest_square_side * largest_square_side

theorem largest_possible_square_area :
  largest_square_area = 36 := by
    sorry

end largest_possible_square_area_l554_554701


namespace max_tan_B_l554_554080

theorem max_tan_B (A B : ℝ) (h : Real.sin (2 * A + B) = 2 * Real.sin B) : 
  Real.tan B ≤ Real.sqrt 3 / 3 := sorry

end max_tan_B_l554_554080


namespace inscribed_circle_radius_l554_554332

noncomputable def side1 := 13
noncomputable def side2 := 13
noncomputable def side3 := 10
noncomputable def s := (side1 + side2 + side3) / 2
noncomputable def area := Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))
noncomputable def inradius := area / s

theorem inscribed_circle_radius :
  inradius = 10 / 3 :=
by
  sorry

end inscribed_circle_radius_l554_554332


namespace largest_shaded_area_X_l554_554085

def side_length : ℝ := 3
def radius_w : ℝ := 1.5
def radius_y : ℝ := 1
def diagonal_x : ℝ := side_length * Real.sqrt 2
def radius_x : ℝ := diagonal_x / 2

def area_square : ℝ := side_length ^ 2
def area_circle_w : ℝ := Real.pi * radius_w ^ 2
def area_circle_x : ℝ := Real.pi * radius_x ^ 2
def area_quarter_circle : ℝ := Real.pi * radius_y ^ 2 / 4
def area_four_quarter_circles : ℝ := 4 * area_quarter_circle

def shaded_area_w : ℝ := area_square - area_circle_w
def shaded_area_x : ℝ := area_circle_x - area_square
def shaded_area_y : ℝ := area_square - area_four_quarter_circles

theorem largest_shaded_area_X :
  shaded_area_x > shaded_area_w ∧ shaded_area_x > shaded_area_y :=
by
  sorry

end largest_shaded_area_X_l554_554085


namespace white_tshirts_per_pack_l554_554740

def packs_of_white := 3
def packs_of_blue := 2
def blue_in_each_pack := 4
def total_tshirts := 26

theorem white_tshirts_per_pack :
  ∃ W : ℕ, packs_of_white * W + packs_of_blue * blue_in_each_pack = total_tshirts ∧ W = 6 :=
by
  sorry

end white_tshirts_per_pack_l554_554740


namespace decreasing_intervals_l554_554047

noncomputable def f (x : ℝ) : ℝ := 2 * Math.sin (Real.pi / 4 - 2 * x)

theorem decreasing_intervals : 
  ∀ (x k : ℤ), -Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ 3 * Real.pi / 8 + k * Real.pi → 
               ∀ (y : ℝ), (x ≤ y → f x ≥ f y) :=
by
  sorry

end decreasing_intervals_l554_554047


namespace greatest_radius_l554_554842

theorem greatest_radius (r : ℤ) (h : π * r^2 < 100 * π) : r < 10 :=
sorry

example : ∃ r : ℤ, π * r^2 < 100 * π ∧ ∀ r' : ℤ, (π * r'^2 < 100 * π) → r' ≤ r :=
begin
  use 9,
  split,
  { linarith },
  { intros r' h',
    have hr' : r' < 10,
    { linarith },
    exact int.lt_of_le_of_lt (int.le_of_lt_add_one hr') (by linarith) }
end

end greatest_radius_l554_554842


namespace proof_sum_cos_inequality_l554_554139

def sum_cos_inequality (n : ℕ) (x : ℕ → ℝ) : Prop :=
  ∑ i in finset.range(n), ∑ j in finset.range(n), if i < j then cos (x j - x i) else 0 ≥ - (n : ℝ) / 2

theorem proof_sum_cos_inequality (n : ℕ) (x : ℕ → ℝ) : sum_cos_inequality n x :=
  sorry

end proof_sum_cos_inequality_l554_554139


namespace intersection_problem_impossible_l554_554628

theorem intersection_problem_impossible {P : Type} [AffineSpace P ℝ] 
  (L : Fin 7 → AffineSubspace ℝ P) (dist : ∀ i j : Fin 7, i ≠ j → L i ≠ L j)
  (h1 : ∃ S : Finset P, S.card ≥ 6 ∧ (∀ p ∈ S, ∃ (i j k : Fin 7), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ p ∈ L i ∧ p ∈ L j ∧ p ∈ L k))
  (h2 : ∃ T : Finset P, T.card ≥ 4 ∧ (∀ p ∈ T, ∃ (i j : Fin 7), i ≠ j ∧ p ∈ L i ∧ p ∈ L j ∧ (∀ k, k ≠ i ∧ k ≠ j → p ∉ L k)))
  : false :=
sorry

end intersection_problem_impossible_l554_554628


namespace range_of_a_unique_real_root_l554_554073

def f (x : ℝ) : ℝ := -x^2 + 5*x - 3

theorem range_of_a_unique_real_root {a : ℝ} :
  (∃ x : ℝ, lg (x - 1) + lg (3 - x) = lg (a - x) ∧ 1 < x ∧ x < 3 ∧ x < a) ↔ (1 < a ∧ a ≤ 13/4) :=
sorry

end range_of_a_unique_real_root_l554_554073


namespace value_of_p_line_intersection_fixed_point_l554_554429
noncomputable theory

-- Problem 1: Prove that p = 2
theorem value_of_p (p : ℝ) (h₁ : (1 : ℝ)^2 = 2 * p) : p = 2 :=
by
  have h₂ : 1 = 2 * p := by exact h₁
  have h₃ : 2 = 2 * p by simp at h₂; exact h₂
  exact eq_of_mul_eq_mul_left zero_lt_two h₃

-- Problem 2: Prove line passes through (1, 0)
theorem line_intersection_fixed_point (p : ℝ) (y₁ y₂ : ℝ) (m : ℝ) (h₁ : y₁ * y₂ = -4) (h₂ : ∀ (x b : ℝ), x = m * y₁ + b ∧ x = m * y₂ + b → y₁ * y₂ = -4) : ∀ x, x = m * 0 + 1 :=
by
  intro x
  have : y₁ * y₂ = -4 := by exact h₁
  have b : (-4)/(m * (y₁ + y₂)) = 1 := 
    by
      rw [mul_comm, <-div_eq_mul_inv]
      simp only [eq_self_iff_true, h₁, neg_eq_iff_add_eq_zero, add_sub_cancel]
  have : (y₁ + y₂) ≠ 0 := sorry
  use x,
  simp only [mul_comm, <-div_eq_mul_inv] at h₂_eq b_eq,
  exact (right_inverse_eq left_inverse_eq).symm

sorry

end value_of_p_line_intersection_fixed_point_l554_554429


namespace quadrilateral_area_proof_l554_554963

noncomputable def quadrilateral_area : ℕ → ℕ → ℕ → ℕ :=
  λ (n a₁ aₙ : ℕ), (n * (a₁ + aₙ)) / 2

theorem quadrilateral_area_proof :
  ∀ (n a₁ aₙ : ℕ), n = 100 → a₁ = 1 → aₙ = 2 →
  quadrilateral_area n a₁ aₙ = 150 :=
begin
  intros n a₁ aₙ h_n h_a₁ h_aₙ,
  subst h_n,
  subst h_a₁,
  subst h_aₙ,
  simp [quadrilateral_area]
end

end quadrilateral_area_proof_l554_554963


namespace max_tan_A_l554_554391

theorem max_tan_A (A B : ℝ) (hA1 : 0 < A) (hA2 : A < π / 2) (hB1 : 0 < B) (hB2 : B < π / 2) (h : 3 * sin A = cos (A + B) * sin B) :
  tan A ≤ sqrt 3 / 12 :=
  sorry

end max_tan_A_l554_554391


namespace terminating_decimals_count_l554_554771

theorem terminating_decimals_count :
  (∀ m : ℤ, 1 ≤ m ∧ m ≤ 999 → ∃ k : ℕ, (m : ℝ) / 1000 = k / (2 ^ 3 * 5 ^ 3)) :=
by
  sorry

end terminating_decimals_count_l554_554771


namespace positive_difference_l554_554616

-- Define the binomial coefficient
def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Define the probability of heads in a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of exactly k heads out of n flips
def prob_heads (n k : ℕ) : ℚ :=
  binomial n k * (fair_coin_prob ^ k) * (fair_coin_prob ^ (n - k))

-- Define the probabilities for the given problem
def prob_3_heads_out_of_5 : ℚ := prob_heads 5 3
def prob_5_heads_out_of_5 : ℚ := prob_heads 5 5

-- Claim the positive difference
theorem positive_difference :
  prob_3_heads_out_of_5 - prob_5_heads_out_of_5 = 9 / 32 :=
by
  sorry

end positive_difference_l554_554616


namespace crayons_left_l554_554226

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
  (initial_crayons_eq : initial_crayons = 48) 
  (kiley_fraction_eq : kiley_fraction = 1/4) 
  (joe_fraction_eq : joe_fraction = 1/2): 
  (initial_crayons - initial_crayons * kiley_fraction - (initial_crayons - initial_crayons * kiley_fraction) * joe_fraction) = 18 := 
by 
  sorry

end crayons_left_l554_554226


namespace sin_double_angle_l554_554026

theorem sin_double_angle (θ : ℝ) (h : sin (π / 4 + θ) = 1 / 3) : sin (2 * θ) = -7 / 9 :=
  sorry

end sin_double_angle_l554_554026


namespace students_remaining_after_fourth_stop_l554_554070

theorem students_remaining_after_fourth_stop : 
  ∀ (initial number of students : ℕ) (get_off_fraction : ℚ) (stops : ℕ),
    initial = 60 →
    get_off_fraction = 1 / 3 →
    stops = 4 →
    let remaining_fraction := 1 - get_off_fraction in
    let final_students := initial * (remaining_fraction ^ stops) in
  final_students = 11 :=
begin
  intros initial_students get_off_fraction stops h1 h2 h3,
  let remaining_fraction := 1 - get_off_fraction,
  let final_students := initial_students * (remaining_fraction ^ stops),
  have h_initial : initial_students = 60, by exact h1,
  have h_get_off : get_off_fraction = 1 / 3, by exact h2,
  have h_stops : stops = 4, by exact h3,
  calc
  final_students = 60 * (2 / 3) ^ 4 : by { rw [h_initial, h_get_off, h_stops], sorry }
               ... = 11 : sorry
end

end students_remaining_after_fourth_stop_l554_554070


namespace complex_quadrant_fourth_l554_554866

noncomputable def complex_number : ℂ := (1 / ((1 + complex.i)^2 + 1)) + complex.i^4

theorem complex_quadrant_fourth :
  complex_number.re > 0 ∧ complex_number.im < 0 :=
sorry

end complex_quadrant_fourth_l554_554866


namespace molecular_weight_AlOH₃_l554_554343

def Al_wc : Float := 26.98
def O_wc : Float := 16.00
def H_wc : Float := 1.01

theorem molecular_weight_AlOH₃ : (Al_wc + 3 * O_wc + 3 * H_wc) = 78.01 :=
by
  -- Definitions and assumptions
  let Al_weight := Al_wc
  let O_weight := O_wc
  let H_weight := H_wc
  -- Calculation
  have h1 : Al_weight = 26.98 := rfl
  have h2 : O_weight = 16.00 := rfl
  have h3 : H_weight = 1.01 := rfl
  -- Summation
  calc
    Al_wc + 3 * O_wc + 3 * H_wc
    = 26.98 + 3 * 16.00 + 3 * 1.01 : by rw [h1, h2, h3]
    = 26.98 + 48.00 + 3.03         : by norm_num
    = 78.01                        : by norm_num

end molecular_weight_AlOH₃_l554_554343


namespace centroid_distance_to_O_l554_554885

noncomputable def point (x y z : ℝ) := (x, y, z)

def point_distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def plane (α β γ : ℝ) := 
  { p : ℝ × ℝ × ℝ // p.1 / α + p.2 / β + p.3 / γ = 1 }

def centroid (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ( (A.1 + B.1 + C.1) / 3,
    (A.2 + B.2 + C.2) / 3,
    (A.3 + B.3 + C.3) / 3 )

theorem centroid_distance_to_O
  (O : ℝ × ℝ × ℝ)
  (α β γ : ℝ)
  (A := point α 0 0)
  (B := point 0 β 0)
  (C := point 0 0 γ)
  (p q r : ℝ)
  (P := centroid A B C)
  (h2 : point_distance O (α, β, γ) = 2)
  (h_centr : P = (p, q, r)) :
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 9 := sorry

end centroid_distance_to_O_l554_554885


namespace projection_correct_l554_554009

def projection_onto_plane : ℝ × ℝ × ℝ :=
  let v : ℝ × ℝ × ℝ := ⟨2, 3, 1⟩
  let n : ℝ × ℝ × ℝ := ⟨1, 2, -1⟩
  let dot (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let scalar_mul (k : ℝ) (a : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ⟨k * a.1, k * a.2, k * a.3⟩
  let sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ⟨a.1 - b.1, a.2 - b.2, a.3 - b.3⟩
  let c := dot v n / dot n n
  let v_proj_n := scalar_mul c n
  sub v v_proj_n

theorem projection_correct :
  projection_onto_plane = ⟨5/6, 4/6, 13/6⟩ :=
sorry

end projection_correct_l554_554009


namespace exists_statement_l554_554109

-- Define the structure for the Transylvanian's statement
variable (T : Type) (trustworthy : T → Prop) (DraculaAlive : Prop)

-- Main theorem stating the existence of such a statement
theorem exists_statement (t : T) : ∃ S : Prop, (S ↔ DraculaAlive) ∧ (¬determine_truth_value S) := by
  -- This is where we would normally prove the theorem
  sorry

-- Auxiliary definition to indicate indeterminate truth value
def determine_truth_value (S : Prop) : Prop :=
  ¬(S ∧ ¬S) ∧ ¬(¬S ∧ S)

#check @exists_statement -- Check if the theorem statement is well-defined

end exists_statement_l554_554109


namespace count_valid_solutions_l554_554450

noncomputable def numerator (x : ℝ) := list.prod (list.map (λ k, x - k) (list.range 100).map (λ n, n + 1))
noncomputable def denominator (x : ℝ) := list.prod (list.map (λ k, x - k^2) (list.range 10).map (λ n, (n + 1)^2))

def is_valid_solution (x : ℕ) : Prop :=
  numerator x = 0 ∧ denominator x ≠ 0

theorem count_valid_solutions : 
  (finset.filter is_valid_solution (finset.range 101)).card = 90 :=
sorry

end count_valid_solutions_l554_554450


namespace find_a_l554_554065

variable (a : ℝ)

theorem find_a (h1 : (2 + a * Complex.i) / (1 + Complex.i) = 3 + Complex.i) : a = 4 := 
by 
  -- placeholder for the proof
  sorry

end find_a_l554_554065


namespace value_of_1_minus_a_l554_554834

theorem value_of_1_minus_a (a : ℤ) (h : a = -(-6)) : 1 - a = -5 := 
by 
  sorry

end value_of_1_minus_a_l554_554834


namespace prob_rel_prime_50_l554_554362

def relatively_prime_to (n k : ℕ) : Prop := Nat.gcd n k = 1

def count_relatively_prime_to_range (n k : ℕ) : ℕ :=
  (List.range n).filter (λ m => relatively_prime_to m k).length

def probability_relatively_prime (n k : ℕ) : ℚ :=
  (count_relatively_prime_to_range (n+1) k : ℚ) / (n+1 : ℚ)

theorem prob_rel_prime_50 : probability_relatively_prime 50 50 = 2 / 5 := 
  sorry

end prob_rel_prime_50_l554_554362


namespace overlapping_angle_condition_l554_554267

theorem overlapping_angle_condition {α : ℝ} (h1 : 0 < α) (h2 : α < 180) :
  (100 < α.to_degrees) ∧ (α.to_degrees < 120) := by
  sorry

end overlapping_angle_condition_l554_554267


namespace smallest_number_of_coins_to_pay_up_to_2_dollars_l554_554240

def smallest_number_of_coins_to_pay_up_to (max_amount : Nat) : Nat :=
  sorry  -- This function logic needs to be defined separately

theorem smallest_number_of_coins_to_pay_up_to_2_dollars :
  smallest_number_of_coins_to_pay_up_to 199 = 11 :=
sorry

end smallest_number_of_coins_to_pay_up_to_2_dollars_l554_554240


namespace eval_star_l554_554454

-- Define the operation *
def star (A B : ℝ) : ℝ := (A + B) / 2

-- State the theorem we need to prove
theorem eval_star : star (star 7 9) 4 = 6 :=
by
  sorry

end eval_star_l554_554454


namespace MrEithanSavings_l554_554960

theorem MrEithanSavings :
  let stimulus_check := 4000
  let wife_share := 1 / 4 * stimulus_check
  let remaining_after_wife := stimulus_check - wife_share
  let first_son_share := 3 / 8 * remaining_after_wife
  let remaining_after_first_son := remaining_after_wife - first_son_share
  let second_son_share := 25 / 100 * remaining_after_first_son
  let remaining_after_second_son := remaining_after_first_son - second_son_share
  let third_son_share := 500
  let remaining_after_third_son := remaining_after_second_son - third_son_share
  let daughter_share := 15 / 100 * remaining_after_third_son
  let savings := remaining_after_third_son - daughter_share
  in savings = 770.31 :=
by
  let stimulus_check := 4000
  let wife_share := 1 / 4 * stimulus_check
  let remaining_after_wife := stimulus_check - wife_share
  let first_son_share := 3 / 8 * remaining_after_wife
  let remaining_after_first_son := remaining_after_wife - first_son_share
  let second_son_share := 25 / 100 * remaining_after_first_son
  let remaining_after_second_son := remaining_after_first_son - second_son_share
  let third_son_share := 500
  let remaining_after_third_son := remaining_after_second_son - third_son_share
  let daughter_share := 15 / 100 * remaining_after_third_son
  let savings := remaining_after_third_son - daughter_share
  have h : savings = 770.3125 := sorry
  exact sorry

end MrEithanSavings_l554_554960


namespace deposit_amount_correct_l554_554059

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r / 100) ^ t

noncomputable def geometric_series_sum (B : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  B * (1 + r / 100) * ((1 + r / 100) ^ n - 1) / (r / 100)

noncomputable def yearly_deposit (target : ℝ) (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  let total_from_lump_sum := compound_interest P r t
  let needed_yearly_deposit := target / geometric_series_sum 1 r t
  needed_yearly_deposit

theorem deposit_amount_correct:
  yearly_deposit 248832 100000 20 5 = 27864.98 :=
by
  have h1 : compound_interest 100000 20 5 = 248832 := by sorry
  have h2 : geometric_series_sum 1 20 5 ≈ 8.92992 := by sorry
  have h3 : 248832 / 8.92992 ≈ 27864.98 := by sorry
  sorry

end deposit_amount_correct_l554_554059


namespace solve_lambda_l554_554438

variable (λ : ℝ)

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (λ, 1)

def vector_add (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 + y.1, x.2 + y.2)
def vector_sub (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 - y.1, x.2 - y.2)
def vector_mag_sq (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem solve_lambda :
  vector_mag_sq (vector_add a b) = vector_mag_sq (vector_sub a b) :=
  sorry

end solve_lambda_l554_554438


namespace probability_of_endsInZeroAndDistinctNonZeroDigits_l554_554640

def endsInZeroAndDistinctNonZeroDigits (n : ℕ) : Prop :=
  n % 10 = 0 ∧
  (let digits := List.ofFn (fun i => (n / 10^i) % 10) in
   let nonZeroDigits := (digits.dropLast 1).filter (fun x => x ≠ 0) in
   nonZeroDigits.nodup
  )

def totalFiveDigitIntegers : ℕ := 90000

def favorableOutcomes : ℕ :=
  9 * 8 * 7 * 6

def probability : ℚ :=
  favorableOutcomes / totalFiveDigitIntegers

theorem probability_of_endsInZeroAndDistinctNonZeroDigits :
  probability = 21 / 625 :=
by
  sorry

end probability_of_endsInZeroAndDistinctNonZeroDigits_l554_554640


namespace find_area_enclosed_by_curve_l554_554761

noncomputable def area_enclosed_by_curve : ℝ :=
  - ∫ x in -1..0, (- x^3 + x^2 + 2 * x) + ∫ x in 0..2, (- x^3 + x^2 + 2 * x)

theorem find_area_enclosed_by_curve :
  area_enclosed_by_curve = 37 / 12 :=
by
  sorry

end find_area_enclosed_by_curve_l554_554761


namespace cosine_transform_l554_554460

theorem cosine_transform (α : ℝ) (h : sin α = -2 * cos α) : 
  cos (2 * α + (Real.pi / 2)) = 4 / 5 :=
by
  sorry

end cosine_transform_l554_554460


namespace lucky_numbers_count_l554_554693

def is_lucky (n : ℕ) : Prop := 
  ∃ (k q r : ℕ), 
    1 ≤ n ∧ n ≤ 221 ∧
    221 = n * q + r ∧ 
    r = k * q ∧ 
    r < n

def count_lucky_numbers : ℕ := 
  ∑ n in (finset.range 222).filter is_lucky, 1

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
  by 
    sorry

end lucky_numbers_count_l554_554693


namespace concyclic_points_l554_554532

theorem concyclic_points
  (ABC : Type) [triangle ABC]
  (H : point)
  (D E F : point)
  (midpoint_D : midpoint D BC)
  (midpoint_E : midpoint E CA)
  (midpoint_F : midpoint F AB)
  (circle_D : ∃ (Γ₁ : circle), center Γ₁ = D ∧ point_on Γ₁ H ∧ point_on Γ₁ A₁ ∧ point_on Γ₁ A₂)
  (circle_E : ∃ (Γ₂ : circle), center Γ₂ = E ∧ point_on Γ₂ H ∧ point_on Γ₂ B₁ ∧ point_on Γ₂ B₂)
  (circle_F : ∃ (Γ₃ : circle), center Γ₃ = F ∧ point_on Γ₃ H ∧ point_on Γ₃ C₁ ∧ point_on Γ₃ C₂) :
  are_concyclic [A₁, A₂, B₁, B₂, C₁, C₂] :=
sorry

end concyclic_points_l554_554532


namespace pie_piece_cost_l554_554548

theorem pie_piece_cost (pieces_per_pie : ℕ) (pies_per_hour : ℕ) (total_earnings : ℝ) :
  pieces_per_pie = 3 → pies_per_hour = 12 → total_earnings = 138 →
  (total_earnings / (pieces_per_pie * pies_per_hour)) = 3.83 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end pie_piece_cost_l554_554548


namespace prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l554_554782

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4 * x + a) / (x^2 + 1)

-- 1. Prove that a = 0 given that f(x) is an odd function
theorem prove_a_eq_0 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = - f x a) : a = 0 := sorry

-- 2. Prove that f(x) = 4x / (x^2 + 1) is monotonically decreasing on [1, +∞) for x > 0
theorem prove_monotonic_decreasing (x : ℝ) (hx : x > 0) :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (f x1 0) > (f x2 0) := sorry

-- 3. Prove that |f(x1) - f(x2)| ≤ m for all x1, x2 ∈ R implies m ≥ 4
theorem prove_m_ge_4 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, |f x1 0 - f x2 0| ≤ m) : m ≥ 4 := sorry

end prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l554_554782


namespace pyramid_volume_division_l554_554188

open BigOperators

-- Definitions based on the given problem conditions
variables {P A B C D K M : Point}
variables [pyramid P A B C D] [parallelogram A B C D]
variables h₁ : K ∈ segment A B
variables h₂ : M ∈ segment P C
variables h₃ : AK / KB = 1/2
variables h₄ : CM / MP = 1/2
variables h₅ : plane_through_parallel_to BD K M 

theorem pyramid_volume_division (h₁ : ∃ h₆, plane_through_parallel_to h₆ K M) :
  divides_volume P A B C D 11 7 :=
sorry

end pyramid_volume_division_l554_554188


namespace P_inter_Q_eq_l554_554433

def P : Set ℕ := {0, 1, 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 3 ^ x}

theorem P_inter_Q_eq : P ∩ Q = {1, 2} := by
  sorry

end P_inter_Q_eq_l554_554433


namespace expected_cuts_l554_554498

noncomputable def f (x : ℝ) : ℝ :=
1 + log x

theorem expected_cuts : f 2016 = 1 + log 2016 := 
by
  have non_zero : 2016 ≠ 0 := by norm_num
  rw [f, log_mul_base 2016 non_zero]
  sorry

end expected_cuts_l554_554498


namespace three_consecutive_integers_l554_554974

theorem three_consecutive_integers (n : ℕ) (hn_pos : 0 < n) (hn_div4 : n % 4 = 0) :
  ∃ (a b c : ℕ), a = n ∧ b = n + 1 ∧ c = n + 2 ∧
    (a = 4 ∧ b = 5 ∧ c = 6) ∧
    (∀ x y, x ∈ {a, b, c} → y ∈ {a, b, c} → x ≠ y → (min x y) % ((x - y) ^ 2) = 0) := 
by
  sorry

end three_consecutive_integers_l554_554974


namespace students_in_each_group_l554_554988

-- Define the total number of students
def total_students : ℕ := 65

-- Define the number of students who didn't get picked
def not_picked_students : ℕ := 17

-- Define the number of groups
def number_of_groups : ℕ := 8

-- Define the number of students who got picked
def picked_students : ℕ := total_students - not_picked_students

-- Define the number of students per group
def students_per_group : ℕ := picked_students / number_of_groups

theorem students_in_each_group : students_per_group = 6 :=
begin
  -- use the specific theorem proving steps here
  sorry
end

end students_in_each_group_l554_554988


namespace sum_of_solutions_l554_554519

def g (x : ℝ) : ℝ := 3 * x - 2

noncomputable def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem sum_of_solutions : 
  (∀ x : ℝ, g_inv x = g (x⁻¹) → x = 1 ∨ x = -9) → 
  ∑ x in {-9, 1}, x = -8 :=
by {
  intros h,
  have h1 : true := sorry,  -- a place holder as the exact implementation of the proofs are omitted.
  sorry
}

end sum_of_solutions_l554_554519


namespace no_two_perfect_cubes_between_two_perfect_squares_l554_554108

theorem no_two_perfect_cubes_between_two_perfect_squares :
  ∀ n a b : ℤ, n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2 → False :=
by 
  sorry

end no_two_perfect_cubes_between_two_perfect_squares_l554_554108


namespace length_AC_l554_554202

-- Defining the points and distances
structure Point where
  x : ℝ
  y : ℝ

noncomputable def A : Point := {x := 0, y := 0 }
noncomputable def B : Point := {x := 25, y := 0 }
noncomputable def C : Point := {x := 17, y := 8 }
noncomputable def D : Point := {x := 8, y := 8 }

def distance (p q : Point) : ℝ :=
  Real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2)

def isosceles_trapezoid (A B C D : Point) : Prop :=
  distance A B = 25 ∧
  distance B C = 11 ∧
  distance C D = 10 ∧
  distance D A = 11 ∧
  A.y = 0 ∧ B.y = 0 ∧ C.y = 8 ∧ D.y = 8

theorem length_AC (A B C D : Point) (h : isosceles_trapezoid A B C D) : 
  distance A C = 19.262 :=
by
  sorry

end length_AC_l554_554202


namespace inscribed_circle_probability_l554_554271

-- Define the right-angled triangle with legs of 5 and 12
def right_angled_triangle : Type := {a b : ℕ // a = 5 ∧ b = 12}

-- Define the hypotenuse using Pythagorean theorem
def hypotenuse {a b : ℕ} (h : right_angled_triangle) : ℚ :=
  Real.sqrt (a^2 + b^2)

-- Define the radius of the inscribed circle
def inscribed_circle_radius {a b : ℕ} (h : right_angled_triangle) : ℚ :=
  (a + b - hypotenuse h) / 2

-- Define the area of the inscribed circle
def inscribed_circle_area {a b : ℕ} (h : right_angled_triangle) : ℚ :=
  Real.pi * (inscribed_circle_radius h)^2

-- Define the area of the right-angled triangle
def triangle_area {a b : ℕ} (h : right_angled_triangle) : ℚ :=
  (a * b) / 2

-- Define the probability that a point lies within the inscribed circle
def point_within_circle_probability {a b : ℕ} (h : right_angled_triangle) : ℚ :=
  inscribed_circle_area h / triangle_area h

-- The theorem to be proved
theorem inscribed_circle_probability : right_angled_triangle → point_within_circle_probability right_angled_triangle = 2 * Real.pi / 15 :=
by 
  intro h
  sorry

end inscribed_circle_probability_l554_554271


namespace preston_receives_total_amount_l554_554161

theorem preston_receives_total_amount :
  let price_per_sandwich := 5
  let delivery_fee := 20
  let num_sandwiches := 18
  let tip_percent := 0.10
  let sandwich_cost := num_sandwiches * price_per_sandwich
  let initial_total := sandwich_cost + delivery_fee
  let tip := initial_total * tip_percent
  let final_total := initial_total + tip
  final_total = 121 := 
by
  sorry

end preston_receives_total_amount_l554_554161


namespace measure_of_A_max_value_l554_554407

-- Definitions and conditions
variables {a b c A B C : ℝ}
variables {S : ℝ}
variables (h1 : (b + c - a) * (sin A + sin B + sin C) = c * sin B)
variables (h_triangle : a ≠ 0)
variables {abc_triangle : a = 2 * sqrt 3}

-- Goal 1: Prove that given h1, A must be 2π/3
theorem measure_of_A : (b + c - a) * (sin A + sin B + sin C) = c * sin B → A = 2 * π / 3 :=
sorry

-- Goal 2: Prove that given abc_triangle, the maximum value of S + 4√3 cos B cos C is 4√3
theorem max_value : a = 2 * sqrt 3 ∧ ((b + c - a) * (sin A + sin B + sin C) = c * sin B) →
  ∃ B C, B = C ∧ A = 2 * π / 3 ∧ S + 4 * sqrt 3 * cos B * cos C = 4 * sqrt 3 :=
sorry

end measure_of_A_max_value_l554_554407


namespace question_equals_answer_l554_554492

def fill_grid (a b: ℕ) : Prop :=
  ∃ (r1 r2 r3: list ℕ),
    r1 = [2, 4, 3] ∧
    r2 = [4 - a, 3, a] ∧
    r3 = [4 - b, a, 4 - a] ∧
    ∀ r, r ∈ [r1, r2, r3] → (2 ∈ r ∧ 3 ∈ r ∧ 4 ∈ r) 

theorem question_equals_answer : 
  ∀ A B, (fill_grid A B) → A + B = 6 :=
by
  sorry

end question_equals_answer_l554_554492


namespace sin_16_over_3_pi_l554_554993

theorem sin_16_over_3_pi : Real.sin (16 / 3 * Real.pi) = -Real.sqrt 3 / 2 := 
sorry

end sin_16_over_3_pi_l554_554993


namespace count_lucky_numbers_l554_554653

def is_lucky (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 221) ∧ (221 % n % (221 / n) = 0)

theorem count_lucky_numbers : (finset.filter is_lucky (finset.range 222)).card = 115 :=
by
  sorry

end count_lucky_numbers_l554_554653


namespace sum_solutions_eq_neg_eight_l554_554517

def g (x : ℝ) : ℝ := 3 * x - 2

def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem sum_solutions_eq_neg_eight :
  (∑ x in {x : ℝ | g_inv x = g (x⁻¹)}, x) = -8 :=
by
  sorry

end sum_solutions_eq_neg_eight_l554_554517


namespace line_parabola_one_intersection_not_tangent_l554_554069

theorem line_parabola_one_intersection_not_tangent {A B C D : ℝ} (h: ∀ x : ℝ, ((A * x ^ 2 + B * x + C) = D) → False) :
  ¬ ∃ x : ℝ, (A * x ^ 2 + B * x + C) = D ∧ 2 * x * A + B = 0 := sorry

end line_parabola_one_intersection_not_tangent_l554_554069


namespace sum_of_reciprocals_eq_three_l554_554949

theorem sum_of_reciprocals_eq_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1/x + 1/y) = 3 := 
by
  sorry

end sum_of_reciprocals_eq_three_l554_554949


namespace lucky_numbers_l554_554684

def is_lucky (n : ℕ) : Prop :=
  ∃ q r : ℕ, 221 = n * q + r ∧ r % q = 0

def lucky_numbers_count (a b : ℕ) : ℕ :=
  Nat.card {n // a ≤ n ∧ n ≤ b ∧ is_lucky n}

theorem lucky_numbers (a b : ℕ) (h1 : a = 1) (h2 : b = 221) :
  lucky_numbers_count a b = 115 := 
sorry

end lucky_numbers_l554_554684


namespace num_tangent_lines_with_equal_intercepts_l554_554579

theorem num_tangent_lines_with_equal_intercepts :
  let circle_eqn : ∀ x y : ℝ, (x - 3)^2 + (y - 4)^2 = 2 in
  let is_tangent (line_eqn : ℝ → ℝ → Prop) : Prop :=
    ∃ k : ℝ, ∀ x y : ℝ, line_eqn x y = (y = k * x) ∨ (x + y = k) in
  let equal_intercepts (line_eqn : ℝ → ℝ → Prop) : Prop :=
    ∃ a : ℝ, ∀ x y : ℝ, line_eqn x y = (x / y = a) ∨ (x + y = a) in
  ∃ (tangent_lines : list (ℝ → ℝ → Prop)), 
  (∀ l, l ∈ tangent_lines → is_tangent l) ∧
  (∀ l, l ∈ tangent_lines → equal_intercepts l) ∧
  tangent_lines.length = 4 :=
sorry

end num_tangent_lines_with_equal_intercepts_l554_554579


namespace lucky_numbers_count_l554_554672

def is_lucky (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧ (221 % n) % (221 / n) = 0

def count_lucky_numbers : ℕ :=
  (Finset.range 222).filter is_lucky |>.card

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
by
  sorry

end lucky_numbers_count_l554_554672


namespace function_passes_through_point_l554_554900

theorem function_passes_through_point (a b c x : ℝ) :
    (tan x) * (tan (π/4 - x)) = -c / a ∧ 
    tan x + tan (π/4 - x) = -b / a →
    (1, (a * 1^2 + b * 1 - c)) = (1, 0) :=
by
  sorry

end function_passes_through_point_l554_554900


namespace transform_triple_eq_l554_554353

noncomputable def f : ℝ → ℝ := sorry

def h (x : ℝ) : ℝ := 2 * f (2 * x + 3) + 1

theorem transform_triple_eq : ∃ a b c : ℝ, 
  (a = 2 ∧ b = 2 ∧ c = 1 ∧ (h x = a * f (b * x + 3) + c)) := by
  use 2, 2, 1
  sorry

end transform_triple_eq_l554_554353


namespace log_base_one_fifth_of_25_l554_554743

theorem log_base_one_fifth_of_25 : log (1/5) 25 = -2 := by
  sorry

end log_base_one_fifth_of_25_l554_554743


namespace sum_solutions_eq_neg_eight_l554_554516

def g (x : ℝ) : ℝ := 3 * x - 2

def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem sum_solutions_eq_neg_eight :
  (∑ x in {x : ℝ | g_inv x = g (x⁻¹)}, x) = -8 :=
by
  sorry

end sum_solutions_eq_neg_eight_l554_554516


namespace move_line_up_l554_554716

/-- Define the original line equation as y = 3x - 2 -/
def original_line (x : ℝ) : ℝ := 3 * x - 2

/-- Define the resulting line equation as y = 3x + 4 -/
def resulting_line (x : ℝ) : ℝ := 3 * x + 4

theorem move_line_up (x : ℝ) : resulting_line x = original_line x + 6 :=
by
  sorry

end move_line_up_l554_554716


namespace tan_of_perpendicular_vectors_l554_554141

theorem tan_of_perpendicular_vectors (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (ha : ℝ × ℝ := (Real.cos θ, 2)) (hb : ℝ × ℝ := (-1, Real.sin θ))
  (h_perpendicular : ha.1 * hb.1 + ha.2 * hb.2 = 0) :
  Real.tan θ = 1 / 2 := 
sorry

end tan_of_perpendicular_vectors_l554_554141


namespace count_lucky_numbers_l554_554681

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 221 ∧ let q := 221 / n in let r := 221 % n in r % q = 0

theorem count_lucky_numbers : ∃ (N : ℕ), N = 115 ∧ (∀ n, is_lucky n → n ≤ 221) := 
by
  apply Exists.intro 115
  constructor
  · rfl
  · intros n hn
    cases hn with hpos hrange
    exact hrange.left
  · sorry

end count_lucky_numbers_l554_554681


namespace problem_l554_554423

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - x + (Real.exp 3) * a

def g (x x₀ a : ℝ) : ℝ :=
  if x ≤ x₀ then x + a - (x - a) / Real.exp x
  else (1 - x) * Real.log x - a * (x + 1)

theorem problem
  (a : ℝ) (h_a : -6 / 5 ≤ a ∧ a < 3 / Real.exp 3 - 1)
  (x₀ : ℝ) (h_x₀ : f x₀ a = 0 ∧ 0 < x₀)
  (h_interval : 3 < x₀ ∧ x₀ < 4) :
  (∃ x₁ x₂, x₁ < x₂ ∧ g x₁ x₀ a = 0 ∧ g x₂ x₀ a = 0) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ g x₁ x₀ a = 0 ∧ g x₂ x₀ a = 0 →
    (Real.exp x₂ - x₂) / (Real.exp x₁ - x₁) > Real.exp ((x₁ + x₂) / 2) :=
sorry

end problem_l554_554423


namespace crayons_left_l554_554218

/-- Given initially 48 crayons, if Kiley takes 1/4 and Joe takes half of the remaining,
then 18 crayons are left. -/
theorem crayons_left (initial_crayons : ℕ) (kiley_fraction joe_fraction : ℚ)
    (h_initial : initial_crayons = 48) (h_kiley : kiley_fraction = 1 / 4) (h_joe : joe_fraction = 1 / 2) :
  let kiley_takes := kiley_fraction * initial_crayons,
      remaining_after_kiley := initial_crayons - kiley_takes,
      joe_takes := joe_fraction * remaining_after_kiley,
      crayons_left := remaining_after_kiley - joe_takes
  in crayons_left = 18 :=
by
  sorry

end crayons_left_l554_554218


namespace trapezoid_area_l554_554087

theorem trapezoid_area (AC BD : ℝ) (EF : ℝ) (hAC : AC = 3) (hBD : BD = 5) (hEF : EF = 2) : 
  let area := 6 in area = 6 :=
by
  sorry

end trapezoid_area_l554_554087


namespace tan_product_identity_l554_554345

theorem tan_product_identity (n : ℕ) :
  (∏ k in finset.range 30, (1 + real.tan ((k + 1) * real.pi / 180))) = 2 ^ n →
  n = 15 := by
  sorry

end tan_product_identity_l554_554345


namespace rectangular_garden_perimeter_l554_554207

-- Definitions corresponding to conditions.
def length (L : ℕ) : Prop := L = 500
def breadth (B : ℕ) : Prop := B = 400
def perimeter (P : ℕ) (L B : ℕ) : Prop := P = 2 * (L + B)

-- The theorem we want to prove.
theorem rectangular_garden_perimeter : 
  ∀ (L B P : ℕ), length L → breadth B → P = 1800 := 
by
  intros L B P hL hB
  rw [hL, hB]
  sorry

end rectangular_garden_perimeter_l554_554207


namespace altitude_of_triangle_l554_554625

theorem altitude_of_triangle (x : ℝ) (h : ℝ) 
  (h1 : x^2 = (1/2) * x * h) : h = 2 * x :=
by
  sorry

end altitude_of_triangle_l554_554625


namespace countLuckyNumbers_l554_554666

def isLuckyNumber (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧
  (let q := 221 / n in (221 % n) % q = 0)

theorem countLuckyNumbers : 
  { n : ℕ | isLuckyNumber n }.toFinset.card = 115 :=
by
  sorry

end countLuckyNumbers_l554_554666


namespace coin_probability_l554_554292

theorem coin_probability :
  ∃ p : ℝ, p < 1 / 2 ∧ (20 * p^3 * (1 - p)^3 = 1 / 20) ∧
    p = (1 - real.sqrt (1 - 4 / real.cbrt 400)) / 2 :=
begin
  sorry
end

end coin_probability_l554_554292


namespace point_in_second_quadrant_l554_554094

theorem point_in_second_quadrant : ∀ (x y : ℝ), x = -1 → y = 3 → (x < 0 ∧ y > 0) → (x = -1 ∧ y = 3) → "Second Quadrant" :=
by
  intros x y h1 h2 h3 h4
  sorry

end point_in_second_quadrant_l554_554094


namespace triangle_incircle_ratio_p_q_sum_l554_554469

theorem triangle_incircle_ratio_p_q_sum
  (A B C : Type*) [preorder A] [has_add A] [has_le A]
  (AB BC AC : ℝ) (N : A → Prop) (hAB : AB = 6) (hBC : BC = 8) (hAC : AC = 10)
  (incircles_equal : ∀ x y z : ℝ, N x → N y → x * y = z) :
  ∃ p q : ℕ, gcd p q = 1 ∧ (AN : ℝ) (hAN : AN = (classical.some hAC)) ∧
  (NC : ℝ) (hNC : NC = 10 - classical.some hAN) ∧ incircles_equal (AN/NC) (p/q) ∧
  p + q = 42 :=
sorry

end triangle_incircle_ratio_p_q_sum_l554_554469


namespace baseball_team_opponents_total_runs_l554_554281

theorem baseball_team_opponents_total_runs :
  ∃ (opponents_scores : list ℕ), 
    opponents_scores.length = 8 ∧
    [2, 4, 5, 6, 7, 8, 9, 11].sum = opponents_scores.sum ∧
    (countp (λ s, s.1 = s.2 + 1) (zip [2, 4, 5, 6, 7, 8, 9, 11] opponents_scores) = 4) ∧
    (countp (λ s, s.1 = 3 * s.2) (zip [2, 4, 5, 6, 7, 8, 9, 11] opponents_scores) = 4) →
    opponents_scores.sum = 33 := 
by
  sorry

end baseball_team_opponents_total_runs_l554_554281


namespace point_B_between_A_and_C_l554_554367

theorem point_B_between_A_and_C (a b c : ℚ) (h_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end point_B_between_A_and_C_l554_554367


namespace distinct_colorings_eq_two_l554_554964

-- Definitions for chip colors and rules
-- Each chip can be either red or blue.
inductive ChipColor
| red
| blue

open ChipColor

-- A function to model the color of chips
def chip_color : ℕ → ChipColor

-- Given chip 5 is blue
axiom chip_5_is_blue : chip_color 5 = blue

-- Condition rules:
-- Rule (a): If chips x and y have different colors, then |x - y| must be red.
axiom rule_a (x y : ℕ) (hxy : chip_color x ≠ chip_color y) : chip_color (abs (x - y)) = red

-- Rule (b): If chips x and y have different colors and 1 ≤ x * y ≤ 50, then x * y must be blue.
axiom rule_b (x y : ℕ) (hxy : chip_color x ≠ chip_color y) (hc : 1 ≤ x * y ∧ x * y ≤ 50) : chip_color (x * y) = blue

-- Goal: Prove that there are exactly 2 distinct ways to color the chips
theorem distinct_colorings_eq_two : 
  ∃ C : set (ℕ → ChipColor), C = {chip_color | chip_color 1 = blue} ∨ C = {chip_color | chip_color 1 = red ∧ chip_color n mod 5 = 0 → chip_color n = blue} ∧ C.size = 2 :=
sorry

end distinct_colorings_eq_two_l554_554964


namespace domestic_probability_short_haul_probability_long_haul_probability_l554_554278

variable (P_internet_domestic P_snacks_domestic P_entertainment_domestic P_legroom_domestic : ℝ)
variable (P_internet_short_haul P_snacks_short_haul P_entertainment_short_haul P_legroom_short_haul : ℝ)
variable (P_internet_long_haul P_snacks_long_haul P_entertainment_long_haul P_legroom_long_haul : ℝ)

noncomputable def P_domestic :=
  P_internet_domestic * P_snacks_domestic * P_entertainment_domestic * P_legroom_domestic

theorem domestic_probability :
  P_domestic 0.40 0.60 0.70 0.50 = 0.084 := by
  sorry

noncomputable def P_short_haul :=
  P_internet_short_haul * P_snacks_short_haul * P_entertainment_short_haul * P_legroom_short_haul

theorem short_haul_probability :
  P_short_haul 0.50 0.75 0.55 0.60 = 0.12375 := by
  sorry

noncomputable def P_long_haul :=
  P_internet_long_haul * P_snacks_long_haul * P_entertainment_long_haul * P_legroom_long_haul

theorem long_haul_probability :
  P_long_haul 0.65 0.80 0.75 0.70 = 0.273 := by
  sorry

end domestic_probability_short_haul_probability_long_haul_probability_l554_554278


namespace sum_of_solutions_l554_554520

def g (x : ℝ) : ℝ := 3 * x - 2

noncomputable def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem sum_of_solutions : 
  (∀ x : ℝ, g_inv x = g (x⁻¹) → x = 1 ∨ x = -9) → 
  ∑ x in {-9, 1}, x = -8 :=
by {
  intros h,
  have h1 : true := sorry,  -- a place holder as the exact implementation of the proofs are omitted.
  sorry
}

end sum_of_solutions_l554_554520


namespace irrational_count_l554_554328

theorem irrational_count :
  let nums := [3.1415926, Real.cbrt 3, 10 / 3, Real.sqrt 16, Real.sqrt 12, 5 / 6] in
  let irrational := nums.filter (λ x, ¬ ∃ p q : ℤ, x = p / q) in
  irrational.length = 2 :=
by
  sorry

end irrational_count_l554_554328


namespace x_intercepts_count_l554_554443

def parabola_x (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem x_intercepts_count : 
  (∃ y : ℝ, parabola_x y = 0) → 1 := sorry

end x_intercepts_count_l554_554443


namespace ratio_of_supplies_to_remaining_l554_554358

-- Define the amounts
def initial_amount : ℕ := 960
def textbooks_amount : ℕ := initial_amount / 2
def remaining_amount : ℕ := initial_amount - textbooks_amount
def amount_left_after_supplies : ℕ := 360
def supplies_amount : ℕ := remaining_amount - amount_left_after_supplies

-- Define the proof statement
theorem ratio_of_supplies_to_remaining : supplies_amount * 4 = remaining_amount :=
by
  -- correctness of the initial assumption values
  have textbooks_amount_val : textbooks_amount = 480, from rfl,
  have remaining_amount_val : remaining_amount = 480, from rfl,
  have supplies_amount_val : supplies_amount = 120, from rfl,
  have amount_left_after_supplies_val : amount_left_after_supplies = 360, from rfl,

  -- calculation for the ratio
  calc supplies_amount * 4
      = 120 * 4 : by rw supplies_amount_val
  ... = 480 : by norm_num
  ... = remaining_amount : by rw remaining_amount_val
  ... = 960 / 2 - 360 : by rw [remaining_amount_val, textbooks_amount_val, amount_left_after_supplies_val]

end ratio_of_supplies_to_remaining_l554_554358


namespace find_weight_difference_l554_554936

variables (W_A W_B W_C W_D W_E : ℝ)

-- Definitions of the conditions
def average_weight_abc := (W_A + W_B + W_C) / 3 = 84
def average_weight_abcd := (W_A + W_B + W_C + W_D) / 4 = 80
def average_weight_bcde := (W_B + W_C + W_D + W_E) / 4 = 79
def weight_a := W_A = 77

-- The theorem statement
theorem find_weight_difference (h1 : average_weight_abc W_A W_B W_C)
                               (h2 : average_weight_abcd W_A W_B W_C W_D)
                               (h3 : average_weight_bcde W_B W_C W_D W_E)
                               (h4 : weight_a W_A) :
  W_E - W_D = 5 :=
sorry

end find_weight_difference_l554_554936


namespace mittens_in_each_box_l554_554975

theorem mittens_in_each_box (boxes scarves_per_box total_clothing : ℕ) (h1 : boxes = 8) (h2 : scarves_per_box = 4) (h3 : total_clothing = 80) :
  ∃ (mittens_per_box : ℕ), mittens_per_box = 6 :=
by
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  let mittens_per_box := total_mittens / boxes
  use mittens_per_box
  sorry

end mittens_in_each_box_l554_554975


namespace triangle_angle_A_eq_60_l554_554079

theorem triangle_angle_A_eq_60 (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_tan : (Real.tan A) / (Real.tan B) = (2 * c - b) / b) : 
  A = π / 3 :=
by
  sorry

end triangle_angle_A_eq_60_l554_554079


namespace triangle_perimeter_find_side_c_l554_554468

noncomputable def problem1 (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * (sin (A / 2))^2 + a * (sin (B / 2))^2 = c / 2

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : b * (sin (A / 2))^2 + a * (sin (B / 2))^2 = c / 2) (hc : c = 2) : 
  a + b + c = 6 := 
sorry

noncomputable def problem2 (a b c : ℝ) (A B C : ℝ) : Prop :=
  C = π / 3 ∧ 2 * sqrt 3 = (1 / 2) * a * b * (sqrt 3 / 2) ∧ c^2 = (a + b)^2 - 3 * a * b

theorem find_side_c (a b c : ℝ) (A B C : ℝ) (h1 : C = π / 3) (h2 : 2 * sqrt 3 = (1 / 2) * a * b * (sqrt 3 / 2)) (h3 : a + b = 2 * c) (h4 : c^2 = (a + b)^2 - 3 * a * b) : 
  c = 2 * sqrt 2 := 
sorry

end triangle_perimeter_find_side_c_l554_554468


namespace mode_is_89_l554_554583

noncomputable def mode_of_scores (scores : List ℕ) : ℕ :=
  let freq_map := scores.foldl (λ freq_map score, freq_map.insertWith (+) score 1) (Std.Data.HashMap.empty)
  freq_map.toList.maxBy (λ ⟨score, count⟩, count).fst

def stem_and_leaf_scores : List ℕ :=
  [50, 50, 63, 77, 78, 82, 86, 87, 89, 89, 89, 89, 91, 94, 94, 94, 96, 100, 100, 100]

theorem mode_is_89 : mode_of_scores stem_and_leaf_scores = 89 := by
  sorry

end mode_is_89_l554_554583


namespace omega_is_correct_l554_554569

noncomputable
def determine_omega (ω : ℝ) : Prop :=
  ∀ ω > 0, 
  ∀ x ∈ set.Icc 0 (real.pi / 4),
  ∀ x' ∈ set.Icc 0 (real.pi / 4),
  x < x' → 2 * real.sin (ω * x) < 2 * real.sin (ω * x') ∧
  real.sqrt 3 = 2 * real.sin (real.pi * ω / 4)

theorem omega_is_correct : determine_omega (4 / 3) :=
by sorry

end omega_is_correct_l554_554569


namespace unique_zero_in_interval_l554_554553

noncomputable def f (x : ℝ) : ℝ := x * Real.log10 x - 1

theorem unique_zero_in_interval (h_inc : ∀ x y, (2 < x ∧ x < y ∧ y < 3) → f x < f y)
  (h_f2 : f 2 < 0) (h_f3 : f 3 > 0) :
  ∃! (x : ℝ), (2 < x ∧ x < 3) ∧ f x = 0 :=
sorry

end unique_zero_in_interval_l554_554553


namespace count_lucky_numbers_l554_554683

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 221 ∧ let q := 221 / n in let r := 221 % n in r % q = 0

theorem count_lucky_numbers : ∃ (N : ℕ), N = 115 ∧ (∀ n, is_lucky n → n ≤ 221) := 
by
  apply Exists.intro 115
  constructor
  · rfl
  · intros n hn
    cases hn with hpos hrange
    exact hrange.left
  · sorry

end count_lucky_numbers_l554_554683


namespace constant_term_of_expansion_l554_554512

noncomputable def a : ℝ := ∫ x in 1..2, 2 * x

theorem constant_term_of_expansion :
  let exp_term := (a * x - 1 / x)^6 in
  (binom 6 3 : ℝ) * (-1)^3 * 3^(6-3) = -540 :=
by
  sorry

end constant_term_of_expansion_l554_554512


namespace lucky_numbers_count_l554_554696

def is_lucky (n : ℕ) : Prop := 
  ∃ (k q r : ℕ), 
    1 ≤ n ∧ n ≤ 221 ∧
    221 = n * q + r ∧ 
    r = k * q ∧ 
    r < n

def count_lucky_numbers : ℕ := 
  ∑ n in (finset.range 222).filter is_lucky, 1

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
  by 
    sorry

end lucky_numbers_count_l554_554696


namespace discount_percentage_is_25_l554_554582

noncomputable def regular_price_per_can : ℝ := 0.55
noncomputable def price_for_70_cans : ℝ := 70 * regular_price_per_can
noncomputable def discounted_price_for_70_cans : ℝ := 28.875

theorem discount_percentage_is_25 :
  ∃ D : ℝ, D = 25 ∧ price_for_70_cans * (1 - D / 100) = discounted_price_for_70_cans :=
by {
  use 25,
  split,
  { refl, },
  { sorry, }
}

end discount_percentage_is_25_l554_554582


namespace fourth_student_in_sample_l554_554083

def sample_interval (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

def in_sample (student_number : ℕ) (start : ℕ) (interval : ℕ) (n : ℕ) : Prop :=
  student_number = start + n * interval

theorem fourth_student_in_sample :
  ∀ (total_students sample_size : ℕ) (s1 s2 s3 : ℕ),
    total_students = 52 →
    sample_size = 4 →
    s1 = 7 →
    s2 = 33 →
    s3 = 46 →
    ∃ s4, in_sample s4 s1 (sample_interval total_students sample_size) 1 ∧
           in_sample s2 s1 (sample_interval total_students sample_size) 2 ∧
           in_sample s3 s1 (sample_interval total_students sample_size) 3 ∧
           s4 = 20 := 
by
  sorry

end fourth_student_in_sample_l554_554083


namespace walking_speed_correct_l554_554299

-- Definitions based on the conditions
variables (v : ℝ) -- walking speed in km/hr
variables (running_speed : ℝ := 8) -- running speed in km/hr
variables (total_distance : ℝ := 4) -- total distance in km
variables (total_time : ℝ := 0.75) -- total time in hours

-- The person walks for half the distance and runs for the other half
def walking_distance := total_distance / 2
def running_distance := total_distance / 2
def walking_time := walking_distance / v
def running_time := running_distance / running_speed

-- Total time taken
def journey_time := walking_time + running_time

-- The proof goal is that the walking speed is 4 km/hr
theorem walking_speed_correct : journey_time = total_time → v = 4 :=
by
  sorry

end walking_speed_correct_l554_554299


namespace number_of_non_attacking_rook_placements_l554_554063

theorem number_of_non_attacking_rook_placements : 
  let rows := 4
  let columns := 5
  let rooks := 3
  (choose rows rooks) * (choose columns rooks) * (factorial rooks) = 240 := by
  sorry

end number_of_non_attacking_rook_placements_l554_554063


namespace dream_team_distribution_A_maximize_probability_l554_554562

theorem dream_team_distribution_A (
  P_A_A : ℝ,
  P_A_B : ℝ,
  P_B_A : ℝ,
  P_B_B : ℝ,
  h1 : P_A_A = 0.7,
  h2 : P_A_B = 0.5,
  h3 : P_B_A = 0.4,
  h4 : P_B_B = 0.8
) :
  let P_X_0 := (1 - P_A_A) * (1 - P_B_A),
      P_X_1 := P_A_A * (1 - P_A_B) * (1 - P_B_A) + (1 - P_A_A) * P_B_A * (1 - P_B_B),
      P_X_2 := P_A_A * P_A_B * (1 - P_B_A) + (1 - P_A_A) * P_B_A * P_B_B + P_A_A * P_B_A * (1 - P_A_B) * (1 - P_B_B),
      P_X_3 := P_A_A * P_B_A * P_A_B * (1 - P_B_B) + P_A_A * P_B_A * (1 - P_A_B) * P_B_B,
      P_X_4 := P_A_A * P_B_A * P_A_B * P_B_B,
      E_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4
  in
    P_X_0 = 0.18 ∧
    P_X_1 = 0.234 ∧
    P_X_2 = 0.334 ∧
    P_X_3 = 0.14 ∧
    P_X_4 = 0.112 ∧
    E_X = 1.77 :=
  sorry

theorem maximize_probability (
  P_A_A : ℝ,
  P_A_B : ℝ,
  P_B_A : ℝ,
  P_B_B : ℝ,
  h1 : P_A_A = 0.7,
  h2 : P_A_B = 0.5,
  h3 : P_B_A = 0.4,
  h4 : P_B_B = 0.8
) : 
  let P_A_first := ((P_A_A * P_A_B * P_B_A * (1 - P_B_B)) + (P_A_A * P_B_A * (1 - P_A_B) * P_B_B)),
      P_B_first := ((P_A_B * P_B_B * (1 - P_A_A) * P_B_A) + (P_A_A * P_B_B * P_A_A * (1 - P_B_A)))
  in
    P_B_first > P_A_first :=
  sorry

end dream_team_distribution_A_maximize_probability_l554_554562


namespace line_perpendicular_bisector_k_value_l554_554940

theorem line_perpendicular_bisector_k_value :
  let A : ℝ × ℝ := (1, 6)
  let B : ℝ × ℝ := (7, 12)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ k : ℝ, (∀ x y : ℝ, x + y = k ↔ (x, y) = midpoint) := 13 := 
by
  let A := (1, 6)
  let B := (7, 12)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  have midpoint_eq : midpoint = (4, 9) := by sorry
  use 13
  intro x y
  split
  · intro h
    rw [h]
    exact midpoint_eq.symm
  · intro h
    rw [h]
    exact midpoint_eq
  have k_eq : 4 + 9 = 13 := by sorry
  exact k_eq

end line_perpendicular_bisector_k_value_l554_554940


namespace positive_diff_probability_fair_coin_l554_554609

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end positive_diff_probability_fair_coin_l554_554609


namespace count_lucky_numbers_l554_554680

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 221 ∧ let q := 221 / n in let r := 221 % n in r % q = 0

theorem count_lucky_numbers : ∃ (N : ℕ), N = 115 ∧ (∀ n, is_lucky n → n ≤ 221) := 
by
  apply Exists.intro 115
  constructor
  · rfl
  · intros n hn
    cases hn with hpos hrange
    exact hrange.left
  · sorry

end count_lucky_numbers_l554_554680


namespace decreasing_interval_f_l554_554945

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (4*x - x^2)

theorem decreasing_interval_f : ∀ x, (2 < x) ∧ (x < 4) → f x < f (2 : ℝ) :=
by
sorry

end decreasing_interval_f_l554_554945


namespace find_valid_pairs_l554_554958

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def distinct_two_digit_primes : List (ℕ × ℕ) :=
  [(13, 53), (19, 47), (23, 43), (29, 37)]

def average (p q : ℕ) : ℕ := (p + q) / 2

def number1 (p q : ℕ) : ℕ := 100 * p + q
def number2 (p q : ℕ) : ℕ := 100 * q + p

theorem find_valid_pairs (p q : ℕ)
  (hp : is_prime p) (hq : is_prime q)
  (hpq : p ≠ q)
  (havg : average p q ∣ number1 p q ∧ average p q ∣ number2 p q) :
  (p, q) ∈ distinct_two_digit_primes ∨ (q, p) ∈ distinct_two_digit_primes :=
sorry

end find_valid_pairs_l554_554958


namespace negation_proposition_l554_554573

theorem negation_proposition :
  ¬(∀ x : ℝ, x^2 > x) ↔ ∃ x : ℝ, x^2 ≤ x :=
sorry

end negation_proposition_l554_554573


namespace inequality_proof_l554_554502

theorem inequality_proof 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a ≤ 2 * b) 
  (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a ^ 2 + b ^ 2) ∧ 2 * (a ^ 2 + b ^ 2) ≤ 5 * a * b := 
by
  sorry

end inequality_proof_l554_554502


namespace tangent_line_at_one_increasing_function_range_sum_inequality_l554_554048

section
variable {x : ℝ} (a : ℝ)

noncomputable def f (x : ℝ) := a * (x - 1 / x) - log x

/-- Problem 1: For a = 1, the equation of the tangent line to the curve y = f(x) at the point (1, f(1)) is y = x - 1. -/
theorem tangent_line_at_one (a : ℝ) (ha : a = 1) : 
  let f := (λ x, a * (x - 1 / x) - log x)
  in ∀ y, y = (f 1) → y = x - 1 :=
sorry

/-- Problem 2: If the function f(x) is increasing in its domain (0, +∞), then a ≥ 1/2. -/
theorem increasing_function_range (a : ℝ) (hf : ∀ x ∈ set.Ioi 0, deriv (λ x, a * (x - 1 / x) - log x) x ≥ 0) : 
  a ≥ 1/2 :=
sorry

/-- Problem 3: Prove that (2 × 1 + 1) / (1 × 2) + (2 × 2 + 1) / (2 × 3) + ... + (2n + 1) / (n(n+1)) > ln(n+1) for n ∈ ℕ⁺. -/
theorem sum_inequality (n : ℕ) (hn : 0 < n) : 
  ∑ i in finset.range n.succ, (2 * i + 1) / (i * (i + 1)) > log (n + 1) :=
sorry

end

end tangent_line_at_one_increasing_function_range_sum_inequality_l554_554048


namespace simplify_expression_l554_554990

theorem simplify_expression : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 :=
by sorry

end simplify_expression_l554_554990


namespace simplify_sqrt7_pow6_l554_554925

theorem simplify_sqrt7_pow6 : (Real.sqrt 7)^6 = (343 : Real) :=
by 
  -- we'll fill in the proof later
  sorry

end simplify_sqrt7_pow6_l554_554925


namespace fraction_wearing_glasses_l554_554473

theorem fraction_wearing_glasses (n : ℕ) (h_nonzero : n ≠ 0)
  (h_glasses : 2 / 3 ≤ 1) :
  let students_wearing_glasses := (2 / 3 : ℚ) * n,
      students_not_wearing_glasses := (1 / 3 : ℚ) * n,
      new_students_not_wearing_glasses := 3 * students_not_wearing_glasses,
      new_total_students := students_wearing_glasses + new_students_not_wearing_glasses
  in (students_wearing_glasses / new_total_students = 2 / 5) :=
by
  sorry

end fraction_wearing_glasses_l554_554473


namespace first_player_perfect_play_wins_l554_554154

-- Define the game conditions and players
def first_player_wins_with_perfect_play : Prop :=
  ∀ (cards : list ℕ), cards = [0, 1, 2, 3, 4, 5, 6] →
  ∀ (A B : list ℕ → ℕ), 
  (∀ (picked_cards : list ℕ), card ∈ picked_cards → card ∈ cards) → -- each player picks cards from the available cards
  (A_turn : bool) → -- A starts the game
  ∃ (winning_strategy_A : list ℕ), 
    A (winning_strategy_A) % 17 = 0

-- State the theorem
theorem first_player_perfect_play_wins : first_player_wins_with_perfect_play :=
sorry

end first_player_perfect_play_wins_l554_554154


namespace difference_between_mean_and_median_l554_554859

namespace MathProof

noncomputable def percentage_72 := 0.12
noncomputable def percentage_82 := 0.30
noncomputable def percentage_87 := 0.18
noncomputable def percentage_91 := 0.10
noncomputable def percentage_96 := 1 - (percentage_72 + percentage_82 + percentage_87 + percentage_91)

noncomputable def num_students := 20
noncomputable def scores := [72, 72, 82, 82, 82, 82, 82, 82, 87, 87, 87, 87, 91, 91, 96, 96, 96, 96, 96, 96]

noncomputable def mean_score : ℚ := (72 * 2 + 82 * 6 + 87 * 4 + 91 * 2 + 96 * 6) / num_students
noncomputable def median_score : ℚ := 87

theorem difference_between_mean_and_median :
  mean_score - median_score = 0.1 := by
  sorry

end MathProof

end difference_between_mean_and_median_l554_554859


namespace translate_BINARY_l554_554481

-- Define the assignment according to the problem statement
def char_to_digit (c : Char) : Option ℕ :=
  match c with
  | 'M' => some 0
  | 'O' => some 1
  | 'N' => some 2
  | 'I' => some 3
  | 'T' => some 4
  | 'R' => some 6
  | 'K' => some 7
  | 'E' => some 8
  | 'Y' => some 9
  | 'B' => some 0
  | 'A' => some 2
  | 'D' => some 4
  | _ => none

-- Define the final result string as per the question
def BINARY_code : String := "BINARY"

-- Prove that the binary code translates to the correct sequence
theorem translate_BINARY : (BINARY_code.toList.map char_to_digit) = [some 0, some 3, some 2, some 2, some 3, some 9] :=
by
  -- Verification steps
  have h1 : char_to_digit 'B' = some 0 := rfl
  have h2 : char_to_digit 'I' = some 3 := rfl
  have h3 : char_to_digit 'N' = some 2 := rfl
  have h4 : char_to_digit 'A' = some 2 := rfl
  have h5 : char_to_digit 'R' = some 3 := rfl
  have h6 : char_to_digit 'Y' = some 9 := rfl
  rw [BINARY_code]
  simp
  rw [h1, h2, h3, h4, h5, h6]

-- So the final number is
#eval (BINARY_code.toList.map char_to_digit)

end translate_BINARY_l554_554481


namespace solve_ellipse_problem_l554_554788

section EllipseProblem

-- Defining the conditions
variables (a b c : ℝ) -- Parameters of the ellipse
variables (A : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (e : ℝ) (l : ℝ → ℝ)
variable (h_ellipse : A = (2, 3))
variable (h_axes : F1 = (-2, 0) ∧ F2 = (2, 0))
variable (h_eccentricity : e = 1/2)
variable (h_bisector : ∀ (x y : ℝ), (2 * x - y - 1 = 0))

-- Part (I): Equation of the ellipse
def equation_of_ellipse : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    ((a^2 = b^2 + c^2) ∧ (c/a = 1/2) ∧ 
     (4/a^2 + 9/b^2 = 1) ∧ 
     (A.1^2/a^2 + A.2^2/b^2 = 1))

-- Part (II): Coordinates of Q and equation of line l
def coordinates_of_Q_and_line_l : Prop :=
  ∃ Q : ℝ × ℝ, Q = (1/2, 0) ∧ 
                (∀ x, l x = 2 * x - 1)

-- Part (III): Symmetry of points on the ellipse
def no_symmetric_points : Prop :=
  ∀ B C : ℝ × ℝ, 
    (B.1^2 / 16 + B.2^2 / 12 = 1) → 
    (C.1^2 / 16 + C.2^2 / 12 = 1) → 
    ¬ (2 * ((B.1 + C.1) / 2) - ((B.2 + C.2) / 2) - 1 = 0)

end EllipseProblem

theorem solve_ellipse_problem :
  equation_of_ellipse A ∧ coordinates_of_Q_and_line_l Q l ∧ no_symmetric_points :=
by {
  -- Skipping proofs
  sorry
}

end solve_ellipse_problem_l554_554788


namespace each_charity_gets_45_l554_554113

-- Define the conditions
def dozen := 12
def total_cookies := 6 * dozen
def price_per_cookie := 1.5
def cost_per_cookie := 0.25
def total_revenue := total_cookies * price_per_cookie
def total_cost := total_cookies * cost_per_cookie
def total_profit := total_revenue - total_cost

-- Define the expected outcome
def expected_each_charity_gets := 45

-- The theorem to prove
theorem each_charity_gets_45 :
  total_profit / 2 = expected_each_charity_gets :=
by
  sorry

end each_charity_gets_45_l554_554113


namespace largest_final_number_l554_554921

theorem largest_final_number :
  let num := 12
  let rodrigo_final := (num - 3)^2 + 4
  let samantha_final := (num)^2 - 5 + 4
  let leo_final := (num - 3 + 4)^2
  leo_final > rodrigo_final ∧ leo_final > samantha_final :=
by {
  have h_rodrigo : rodrigo_final = 85 := by sorry,
  have h_samantha : samantha_final = 143 := by sorry,
  have h_leo : leo_final = 169 := by sorry,
  exact ⟨by rw [h_leo, h_rodrigo]; exact Nat.lt_of_lt_of_le 85 169 sorry, 
         by rw [h_leo, h_samantha]; exact Nat.lt_of_lt_of_le 143 169 sorry⟩
}

end largest_final_number_l554_554921


namespace community_theater_receipts_l554_554231

def total_receipts (adult_tickets child_tickets : ℕ) (adult_price child_price : ℕ) : ℕ :=
  (adult_tickets * adult_price) + (child_tickets * child_price)

theorem community_theater_receipts :
  let adult_price := 12
  let child_price := 4
  let total_tickets := 130
  let child_tickets := 90
  let adult_tickets := total_tickets - child_tickets
  total_receipts adult_tickets child_tickets adult_price child_price = 840 :=
by
  let adult_price := 12
  let child_price := 4
  let total_tickets := 130
  let child_tickets := 90
  let adult_tickets := total_tickets - child_tickets
  have h1 : adult_tickets = 40, by linarith
  show total_receipts adult_tickets child_tickets adult_price child_price = 840, from sorry

end community_theater_receipts_l554_554231


namespace surface_area_of_large_cube_correct_l554_554236

-- Definition of the surface area problem

def edge_length_of_small_cube := 3 -- centimeters
def number_of_small_cubes := 27
def surface_area_of_large_cube (edge_length_of_small_cube : ℕ) (number_of_small_cubes : ℕ) : ℕ :=
  let edge_length_of_large_cube := edge_length_of_small_cube * (number_of_small_cubes^(1/3))
  6 * edge_length_of_large_cube^2

theorem surface_area_of_large_cube_correct :
  surface_area_of_large_cube edge_length_of_small_cube number_of_small_cubes = 486 := by
  sorry

end surface_area_of_large_cube_correct_l554_554236


namespace fraction_spent_on_candy_l554_554620

theorem fraction_spent_on_candy (initial_quarters : ℕ) (initial_cents remaining_cents cents_per_dollar : ℕ) (fraction_spent : ℝ) :
  initial_quarters = 14 ∧ remaining_cents = 300 ∧ initial_cents = initial_quarters * 25 ∧ cents_per_dollar = 100 →
  fraction_spent = (initial_cents - remaining_cents) / cents_per_dollar →
  fraction_spent = 1 / 2 :=
by
  intro h1 h2
  sorry

end fraction_spent_on_candy_l554_554620


namespace power_sum_prime_eq_l554_554138

theorem power_sum_prime_eq (p a n : ℕ) (hp : p.Prime) (h_eq : 2^p + 3^p = a^n) : n = 1 :=
by sorry

end power_sum_prime_eq_l554_554138


namespace mass_percentage_h_in_ascorbic_acid_l554_554966

-- Define the input values
def num_c : ℕ := 6
def num_h : ℕ := 8
def num_o : ℕ := 6

def molar_mass_c : ℝ := 12.01 -- g/mol
def molar_mass_h : ℝ := 1.008 -- g/mol
def molar_mass_o : ℝ := 16.00 -- g/mol

-- Calculate the molar mass of each element in the compound
def mass_c : ℝ := num_c * molar_mass_c
def mass_h : ℝ := num_h * molar_mass_h
def mass_o : ℝ := num_o * molar_mass_o

-- Calculate the total molar mass of ascorbic acid
def total_molar_mass : ℝ := mass_c + mass_h + mass_o

-- Define the mass percentage of hydrogen
def mass_percentage_hydrogen : ℝ := (mass_h / total_molar_mass) * 100

-- Formalize the statement for the mass percentage of hydrogen in ascorbic acid
theorem mass_percentage_h_in_ascorbic_acid : abs (mass_percentage_hydrogen - 4.58) < 0.01 :=
by
  -- proof here
  sorry

end mass_percentage_h_in_ascorbic_acid_l554_554966


namespace find_smallest_k_l554_554015

theorem find_smallest_k : ∃ k : ℕ, (∑ d in (Nat.digits 10 (9 * (10^k - 1) / 9)), id d) = 1500 ∧ k = 167 := by
  sorry

end find_smallest_k_l554_554015


namespace find_a_l554_554890

def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x) / (x + a)

theorem find_a :
  (∀ x, f x a = (Real.exp x) / (x + a)) →
  (∂ x, ∀ x, f x a) 1 = Real.exp 1 / 4 →
  a = 1 :=
by
  intros H₁ H₂
  sorry

end find_a_l554_554890


namespace polynomial_f_eqn_l554_554143

theorem polynomial_f_eqn 
  (f : ℂ → ℂ)
  (h_poly : ∀ x, f x = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e)
  (h1 : f(1 + (3 : ℂ)^(1/3)) = 1 + (3 : ℂ)^(1/3))
  (h2 : f(1 + (3 : ℂ)^(1/2)) = 7 + (3 : ℂ)^(1/2)) :
  f(x) = x^4 - 3*x^3 + 3*x^2 - 3*x := 
sorry

end polynomial_f_eqn_l554_554143


namespace geometric_sequence_second_term_value_l554_554208

theorem geometric_sequence_second_term_value
  (a : ℝ) 
  (r : ℝ) 
  (h1 : 30 * r = a) 
  (h2 : a * r = 7 / 4) 
  (h3 : 0 < a) : 
  a = 7.5 := 
sorry

end geometric_sequence_second_term_value_l554_554208


namespace baseball_batter_at_bats_left_l554_554636

theorem baseball_batter_at_bats_left (L R H_L H_R : ℕ) (h1 : L + R = 600)
    (h2 : H_L + H_R = 192) (h3 : H_L = 25 / 100 * L) (h4 : H_R = 35 / 100 * R) : 
    L = 180 :=
by
  sorry

end baseball_batter_at_bats_left_l554_554636


namespace general_inequality_l554_554394

theorem general_inequality (n : ℕ) (x : ℕ → ℝ) (hx : ∀ i, 1 ≤ i → i ≤ n → x i > 0)
  (base_case : (x 1 + x 2) * (1 / x 1 + 1 / x 2) ≥ 4) :
  (x 1 + x 2 + ... + x n) * (1 / x 1 + 1 / x 2 + ... + 1 / x n) ≥ n^2 :=
begin
  sorry
end

end general_inequality_l554_554394


namespace mean_of_combined_sets_l554_554944

theorem mean_of_combined_sets
  (S₁ : Finset ℝ) (S₂ : Finset ℝ)
  (h₁ : S₁.card = 7) (h₂ : S₂.card = 8)
  (mean_S₁ : (S₁.sum id) / S₁.card = 15)
  (mean_S₂ : (S₂.sum id) / S₂.card = 26)
  : (S₁.sum id + S₂.sum id) / (S₁.card + S₂.card) = 20.8667 := 
by
  sorry

end mean_of_combined_sets_l554_554944


namespace find_DF_length_l554_554556

noncomputable def DF_length (AB AD DG : ℝ) : ℝ :=
  let x := 2 * real.sqrt 13 in
  x

theorem find_DF_length (AB AD DG : ℝ) (h1 : AB = 8) (h2 : AD = 3) (h3 : DG = 6) :
  DF_length AB AD DG = 2 * real.sqrt 13 :=
by
  sorry

end find_DF_length_l554_554556


namespace parabola_x_intercepts_l554_554447

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end parabola_x_intercepts_l554_554447


namespace fox_starting_coins_l554_554603

theorem fox_starting_coins :
  ∃ x : ℚ, (3 * (3 * (3 * x - 50) - 50) - 50) = 50 ∧ x = 700 / 27 :=
by
  use (700 / 27)
  split
  · sorry
  · sorry

end fox_starting_coins_l554_554603


namespace find_a_sq_plus_b_sq_l554_554151

theorem find_a_sq_plus_b_sq : 
  ∀ (S : ℕ → ℕ) (a b : ℤ),
  (S 1 = 1) →
  (S 2 = 9) →
  (S 3 = 25) →
  (S 4 = 49) →
  (∀ n, S (2 * n - 1) = (4 * n - 3) * (a * n + b)) →
  a^2 + b^2 = 25 :=
by
  intros S a b hS1 hS2 hS3 hS4 hSupp
  have eq1 : a + b = 1 := sorry
  have eq2 : 2 * a + b = 5 := sorry
  have ha : a = 4 := sorry
  have hb : b = -3 := sorry
  have sum_sq : a^2 + b^2 = 16 + 9 := sorry
  show a^2 + b^2 = 25 from
  by rw [sum_sq]

end find_a_sq_plus_b_sq_l554_554151


namespace inequality_quadrilateral_l554_554268

theorem inequality_quadrilateral : 
  ∀ (A B C D K L : Point), convex_quadrilateral A B C D →
  is_angle_bisector A D K → is_angle_bisector D A K →
  is_angle_bisector B C L → is_angle_bisector C B L →
  2 * dist K L ≥ |dist A B - dist B C + dist C D - dist D A| := 
by
  intros A B C D K L hconvex hbis1 hbis2 hbis3 hbis4
  sorry

end inequality_quadrilateral_l554_554268


namespace vector_subtraction_l554_554025

theorem vector_subtraction 
  (u : ℝ^3)
  (v : ℝ^3)
  (h₁ : u = ![-3, 5, 2])
  (h₂ : v = ![1, -1, 3]) :
  u - 2 • v = ![-5, 7, -4] :=
sorry

end vector_subtraction_l554_554025


namespace jayden_coins_received_l554_554323

theorem jayden_coins_received :
  ∃ J : ℕ, (J + (J + 60) = 660 ∧ J = 300) :=
by
  -- Lean statement for conditions and proof goal
  exists_intro 300
  have h₁ : 300 + (300 + 60) = 660,
  {
    norm_num,
  },
  exact ⟨h₁, rfl⟩

end jayden_coins_received_l554_554323


namespace determine_B_l554_554035

-- Declare the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 1}

-- The conditions given in the problem
axiom h1 : A ∩ B = {1}
axiom h2 : A ∪ B = {0, 1, 2}

-- The theorem we want to prove
theorem determine_B : B = {0, 1} :=
by
  sorry

end determine_B_l554_554035


namespace trivia_team_original_members_l554_554711

theorem trivia_team_original_members (x : ℕ) (h1 : 6 * (x - 2) = 18) : x = 5 :=
by
  sorry

end trivia_team_original_members_l554_554711


namespace words_on_each_page_l554_554282

/-- Given a book with 150 pages, where each page has between 50 and 150 words, 
    and the total number of words in the book is congruent to 217 modulo 221, 
    prove that each page has 135 words. -/
theorem words_on_each_page (p : ℕ) (h1 : 50 ≤ p) (h2 : p ≤ 150) (h3 : 150 * p ≡ 217 [MOD 221]) : 
  p = 135 :=
by
  sorry

end words_on_each_page_l554_554282


namespace rectangle_area_l554_554300

theorem rectangle_area (x : ℝ) (l : ℝ) (h1 : 3 * l = x^2 / 10) : 
  3 * l^2 = 3 * x^2 / 10 :=
by sorry

end rectangle_area_l554_554300


namespace hyperbola_eccentricity_hyperbola_equation_l554_554428

variables (a b e c x0 y0 : ℝ)

theorem hyperbola_eccentricity (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_eq : (x0^2 / a^2 - y0^2 / b^2 = 1) ∧ (x0 ≠ a) ∧ (x0 ≠ -a))
    (h_slopes : - b^2 / a^2 = 144 / 25) : e = 13 / 5 :=
sorry

theorem hyperbola_equation (h_e : e = 13 / 5) (h_dist : 12 = abs(b * c) / sqrt(a^2 + b^2)) 
    : x^2 / 25 - y^2 / 144 = 1 :=
sorry

end hyperbola_eccentricity_hyperbola_equation_l554_554428


namespace volume_of_one_pizza_slice_l554_554304

-- Condition definitions
def pizza_thickness := (1 : ℝ) / 4
def pizza_diameter := 16
def pizza_radius := pizza_diameter / 2
def full_pizza_volume := Real.pi * pizza_radius^2 * pizza_thickness
def number_of_slices := 8

-- Problem statement
theorem volume_of_one_pizza_slice :=
  let volume_per_slice := full_pizza_volume / number_of_slices
  volume_per_slice = 2 * Real.pi :=
by
  -- By proving the equivalence
  sorry

end volume_of_one_pizza_slice_l554_554304


namespace algebraic_expression_value_l554_554619

def algebraic_expression (a b : ℤ) :=
  a + 2 * b + 2 * (a + 2 * b) + 1

theorem algebraic_expression_value :
  algebraic_expression 1 (-1) = -2 :=
by
  -- Proof skipped
  sorry

end algebraic_expression_value_l554_554619


namespace coordinates_of_M_l554_554456

theorem coordinates_of_M (x_0 y_0 : ℝ) (h1 : y_0 = 2 * x_0^2)
  (h2 : x_0 ≥ 0) (h3 : y_0 ≥ 0)
  (h4 : (x_0^2 + (y_0 - 1/8)^2) = (1/4)^2) :
  (x_0 = sqrt 2 / 8) ∧ (y_0 = 1 / 16) := 
by 
  sorry

end coordinates_of_M_l554_554456


namespace triangle_area_proof_l554_554106

theorem triangle_area_proof {a b c : ℝ}
    (h1 : c^2 = (a - b)^2 + 6)
    (h2 : ∠C = π / 3)
    (h3 : C = angle_opposite a b c):
    area_triangle a b C = 3 * sqrt 3 / 2 :=
by sorry

end triangle_area_proof_l554_554106


namespace all_cells_same_color_l554_554634

def grid (n : ℕ) := (Fin n → Fin n → bool)

def majority_color (cells : Fin n → bool) : bool :=
  if (cells.toFinset.filter id).card > (cells.toFinset.filter not).card then true else false

def recolor_row (g : grid 99) (i : Fin 99) : grid 99 :=
  λ r c, if r = i then majority_color (λ c, g r c) else g r c

def recolor_column (g : grid 99) (i : Fin 99) : grid 99 :=
  λ r c, if c = i then majority_color (λ r, g r c) else g r c

theorem all_cells_same_color (g : grid 99) : ∃ b : bool, ∀ i j : Fin 99, 
  (recolor_column (recolor_row g ⟨i.1, sorry⟩) ⟨j.1, sorry⟩) i j = b := sorry

end all_cells_same_color_l554_554634


namespace typing_and_editing_time_l554_554264

-- Definitions for typing and editing times for consultants together and for Mary and Jim individually
def combined_typing_time := 12.5
def combined_editing_time := 7.5
def mary_typing_time := 30.0
def jim_editing_time := 12.0

-- The total time when Jim types and Mary edits
def total_time := 42.0

-- Proof statement
theorem typing_and_editing_time :
  (combined_typing_time = 12.5) ∧ 
  (combined_editing_time = 7.5) ∧ 
  (mary_typing_time = 30.0) ∧ 
  (jim_editing_time = 12.0) →
  total_time = 42.0 := 
by
  intro h
  -- Proof to be filled later
  sorry

end typing_and_editing_time_l554_554264


namespace estimate_student_height_with_foot_length_24_l554_554234

theorem estimate_student_height_with_foot_length_24 :
  (∑ i in finRange 10, (i: ℝ) = 225) →
  (∑ i in finRange 10, (i: ℝ) * 2 = 1600) →
  (∀ b: ℝ, b = 4) →
  ∃ (x y a: ℝ), x = 24 ∧ y = 166 ∧ (y = b * x + a) :=
by
  sorry

end estimate_student_height_with_foot_length_24_l554_554234


namespace triangle_obtuse_of_sine_square_inequality_l554_554850

theorem triangle_obtuse_of_sine_square_inequality {A B C : ℝ} (h : sin A ^ 2 + sin B ^ 2 < sin C ^ 2) : ∃ (A B C : ℝ), (sin A ^ 2 + sin B ^ 2 < sin C ^ 2) ∧ is_obtuse (A + B + C) := 
by 
  sorry

end triangle_obtuse_of_sine_square_inequality_l554_554850


namespace correct_statements_count_l554_554329

theorem correct_statements_count :
  (∃ n : ℕ, odd_positive_integer = 4 * n + 1 ∨ odd_positive_integer = 4 * n + 3) ∧
  (∀ k : ℕ, k = 3 * m ∨ k = 3 * m + 1 ∨ k = 3 * m + 2) ∧
  (∀ s : ℕ, odd_positive_integer ^ 2 = 8 * p + 1) ∧
  (∀ t : ℕ, perfect_square = 3 * q ∨ perfect_square = 3 * q + 1) →
  num_correct_statements = 2 :=
by
  sorry

end correct_statements_count_l554_554329


namespace empty_tank_time_l554_554976

def cubic_feet_to_cubic_inches (volume_ft: ℕ) : ℕ := volume_ft * 1728

def net_rate (inlet: ℤ) (outlet1: ℤ) (outlet2: ℤ) : ℤ := (outlet1 + outlet2) - inlet

def time_to_empty (volume_in_cubic_inches: ℕ) (net_rate_in_cubic_inches_per_min: ℤ) : ℤ :=
  volume_in_cubic_inches / net_rate_in_cubic_inches_per_min

theorem empty_tank_time :
  let volume_ft := 30 in
  let inlet_rate := 3 in
  let outlet_rate1 := 9 in
  let outlet_rate2 := 6 in
  let volume_in_cubic_inches := cubic_feet_to_cubic_inches volume_ft in
  let net_rate := net_rate inlet_rate outlet_rate1 outlet_rate2 in
  time_to_empty volume_in_cubic_inches net_rate = 4320 :=
by
  sorry

end empty_tank_time_l554_554976


namespace crayons_left_l554_554225

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
  (initial_crayons_eq : initial_crayons = 48) 
  (kiley_fraction_eq : kiley_fraction = 1/4) 
  (joe_fraction_eq : joe_fraction = 1/2): 
  (initial_crayons - initial_crayons * kiley_fraction - (initial_crayons - initial_crayons * kiley_fraction) * joe_fraction) = 18 := 
by 
  sorry

end crayons_left_l554_554225


namespace projection_to_plane_correct_l554_554509

noncomputable def proj_to_plane (v n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_prod := v.1 * n.1 + v.2 * n.2 + v.3 * n.3
  let norm_sq := n.1 * n.1 + n.2 * n.2 + n.3 * n.3
  (dot_prod / norm_sq * n.1, dot_prod / norm_sq * n.2, dot_prod / norm_sq * n.3)

theorem projection_to_plane_correct :
  let p := (6, 4, 6)
  let pp := (4, 6, 2)
  let projection_onto_plane (v n : ℝ × ℝ × ℝ) := (v.1 - proj_to_plane (v - pp) n).1,
                                             (v.2 - proj_to_plane (v - pp) n).2,
                                             (v.3 - proj_to_plane (v - pp) n).3
  ∃ (n : ℝ × ℝ × ℝ),
    (p - pp) = n ∧
    ∀ (q : ℝ × ℝ × ℝ),
      q = (5, 1, 8) →
      projection_onto_plane q n = (1.5, 4.5, 1) :=
begin
      sorry
end

end projection_to_plane_correct_l554_554509


namespace sum_min_max_distance_l554_554093

def rectangular_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

def line_equation (x : ℝ) : ℝ :=
  -2 * x + 2 * Real.sqrt 5

def distance_on_line_segment (x : ℝ) : ℝ :=
  |x| + |line_equation x|

theorem sum_min_max_distance :
  let f := distance_on_line_segment
  let min_distance := Real.sqrt 5
  let max_distance := 5 * Real.sqrt 5
  -Real.sqrt 5 <= (x : ℝ) ∧ x <= 2 * Real.sqrt 5 →
  (min_distance + max_distance) = 6 * Real.sqrt 5 :=
begin
  sorry
end

end sum_min_max_distance_l554_554093


namespace horse_perimeter_time_l554_554309

theorem horse_perimeter_time :
  let side_length := Real.sqrt 625,
      perimeter := 4 * side_length,
      length_each_terrain := perimeter / 4 in
  let time_flat_grassland := length_each_terrain / 25,
      time_hilly_terrain := length_each_terrain / 15,
      time_rocky_terrain := length_each_terrain / 10,
      time_dense_forest := length_each_terrain / 5 in
  (time_flat_grassland + time_hilly_terrain + time_rocky_terrain + time_dense_forest) = 10.17 :=
by
  have side_length_pos : side_length = 25 := by sorry
  have perimeter_eq : perimeter = 100 := by sorry
  have terrain_length_eq : length_each_terrain = 25 := by sorry
  have time_flat_eq : time_flat_grassland = 1 := by sorry
  have time_hilly_eq : time_hilly_terrain = 1.67 := by sorry
  have time_rocky_eq : time_rocky_terrain = 2.5 := by sorry
  have time_forest_eq : time_dense_forest = 5 := by sorry
  sorry

end horse_perimeter_time_l554_554309


namespace minimum_value_of_f_l554_554844

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∃ a > 2, (∀ x > 2, f x ≥ f a) ∧ a = 3 := by
sorry

end minimum_value_of_f_l554_554844


namespace lucky_numbers_count_l554_554660

def isLucky (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≠ 0 ∧ 221 = n * k + (221 % n) ∧ (221 % n) % k = 0

def countLuckyNumbers : ℕ :=
  Nat.card (Finset.filter isLucky (Finset.range 222))

theorem lucky_numbers_count : countLuckyNumbers = 115 :=
  sorry

end lucky_numbers_count_l554_554660


namespace min_value_f_l554_554383

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / real.sqrt (x^2 + 5)

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f(y) ≥ f(x) ∧ f(x) = 9 / real.sqrt 5 :=
by sorry

end min_value_f_l554_554383


namespace cody_payment_l554_554349

def initial_cost : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8

def tax_amount := initial_cost * tax_rate
def total_with_tax := initial_cost + tax_amount
def final_price := total_with_tax - discount
def cody_share := final_price / 2

theorem cody_payment : cody_share = 17 := by
  sorry

end cody_payment_l554_554349


namespace range_a_close_functions_l554_554773

noncomputable def f1 (a x : ℝ) : ℝ := log a (x - 3 * a)
noncomputable def f2 (a x : ℝ) : ℝ := log a (1 / (x - a))

theorem range_a (a : ℝ) : 
  (∀ x : ℝ, a + 2 ≤ x ∧ x ≤ a + 3 → (x > 3 * a ∧ x > a)) ↔ (0 < a ∧ a < 1) :=
sorry

theorem close_functions (a : ℝ):
  (0 < a ∧ a < 1) → 
  (∀ x : ℝ, a + 2 ≤ x ∧ x ≤ a + 3 → (|f1 a x - f2 a x| ≤ 1)) ↔ (0 < a ∧ a ≤ (9 - (Real.sqrt 57)) / 12) :=
sorry

end range_a_close_functions_l554_554773


namespace arithmetic_sequence_15th_term_l554_554618

theorem arithmetic_sequence_15th_term :
  let a1 := 3
  let d := 7
  let n := 15
  a1 + (n - 1) * d = 101 :=
by
  let a1 := 3
  let d := 7
  let n := 15
  sorry

end arithmetic_sequence_15th_term_l554_554618


namespace minimum_value_expression_l554_554382

theorem minimum_value_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := 
by
  sorry

end minimum_value_expression_l554_554382


namespace coefficient_of_x3_in_product_l554_554245

-- Definitions of the polynomials
def f : Polynomial ℤ := Polynomial.Coeff (Polynomial.monomial 4 1) [-2, 3, -4, 5]
def g : Polynomial ℤ := Polynomial.Coeff (Polynomial.monomial 3 3) [-4, 1, 6]

-- The proof statement
theorem coefficient_of_x3_in_product : (f * g).coeff 3 = 22 := 
by
  sorry

end coefficient_of_x3_in_product_l554_554245


namespace geom_seq_proof_l554_554039

noncomputable def geom_seq (a q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n - 1)

variables {a q : ℝ}

theorem geom_seq_proof (h1 : geom_seq a q 7 = 4) (h2 : geom_seq a q 5 + geom_seq a q 9 = 10) :
  geom_seq a q 3 + geom_seq a q 11 = 17 :=
by
  sorry

end geom_seq_proof_l554_554039


namespace max_true_statements_l554_554527

theorem max_true_statements (x : ℝ) :
  let s1 := (0 < x^2 ∧ x^2 < 2)
  let s2 := (x^2 > 2)
  let s3 := (-2 < x ∧ x < 0)
  let s4 := (0 < x ∧ x < 2)
  let s5 := (0 < x - x^2 ∧ x - x^2 < 2)
  let s6 := (0 < x^3 ∧ x^3 < 2)
  finset.univ.filter
    (λ s, s1 ∨ s2 ∨ s3 ∨ s4 ∨ s5 ∨ s6).card ≤ 4 :=
begin
  sorry
end

end max_true_statements_l554_554527


namespace correct_factorization_option_B_l554_554250

-- Lean 4 statement to prove the correct factorization
theorem correct_factorization_option_B (x : ℝ) : 4 * x ^ 2 - 4 * x + 1 = (2 * x - 1) ^ 2 := by
  sorry

end correct_factorization_option_B_l554_554250


namespace only_prime_such_that_2p_plus_one_is_perfect_power_l554_554374

theorem only_prime_such_that_2p_plus_one_is_perfect_power :
  ∃ (p : ℕ), p ≤ 1000 ∧ Prime p ∧ ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 :=
by
  sorry

end only_prime_such_that_2p_plus_one_is_perfect_power_l554_554374


namespace measure_angle_BCG_l554_554720

variables {A B C D G : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited G]
variables (A B C D G: Point) -- Considering points as variables
variables {AB BC CD DA BD AC AG : ℝ} -- Considering the lengths as real numbers
variables (α : ℝ) 

-- Square ABCD
def is_square (AB CD : ℝ) (BC DA: ℝ) (∠ DAB ∠ ABC ∠ BCD ∠ CDA : ℝ) :=
  AB = CD ∧ BC = DA ∧ ∠ DAB = 90 ∧ ∠ ABC = 90 ∧ ∠ BCD = 90 ∧ ∠ CDA = 90

-- Extensions and properties
def extended_length (AG BD AC : ℝ) := AG = BD ∧ BD = AC

-- Angle in triangle ACG
def angle_triangle (α angle_ACG angle_GCA angle_AGC : ℝ) :=
  α = 67.5 ∧ angle_ACG = α ∧ angle_GCA = α ∧ angle_AGC = 45

noncomputable def angle_BCG (∠ BCA ∠ ACG : ℝ) :=
  ∠ BCA = 45 ∧ ∠ ACG = 22.5

theorem measure_angle_BCG (AB CD BC DA BD AC AG : ℝ) (angle_BCG : ℝ)
  (h1 : is_square AB CD BC DA 90 90 90 90)
  (h2 : extended_length AG BD AC)
  (h3 : ∠ BCA = 45)
  (h4 : α = 67.5)
  (h5 : ∠ ACG = 22.5)
  (h6 : angle_BCG = ∠ BCA + ∠ ACG) : 
  angle_BCG = 67.5 :=
sorry

end measure_angle_BCG_l554_554720


namespace area_of_rectangle_l554_554263

theorem area_of_rectangle (side radius length breadth : ℕ) (h1 : side^2 = 784) (h2 : radius = side) (h3 : length = radius / 4) (h4 : breadth = 5) : length * breadth = 35 :=
by
  -- proof to be filled here
  sorry

end area_of_rectangle_l554_554263


namespace slope_angle_proof_l554_554783

noncomputable def slope_angle_of_line (k : ℝ) (A B : ℝ × ℝ) : ℝ :=
  if k < 0 then 150 else sorry

theorem slope_angle_proof :
  ∀ (l : ℝ → ℝ) (P A B O : ℝ × ℝ),
    (P.1 = 2) ∧ (P.2 = 0) ∧
    (l = λ x, k * (x - 2)) ∧
    (y = sqrt (2 - x^2)) ∧
    (∀ O, O = (0, 0)) ∧
    let A := (a, sqrt (2 - a^2)), B := (b, sqrt (2 - b^2)) in
    (1/2 * sqrt 2 * sqrt 2 * sin (angle A O B) = 1) →
    slope_angle_of_line k A B = 150 :=
begin
  intros,
  sorry,
end

end slope_angle_proof_l554_554783


namespace ratio_of_black_cookies_eaten_to_initial_l554_554356

-- Define the initial conditions
def initial_white_cookies : ℕ := 80
def additional_black_cookies : ℕ := 50
def remaining_cookies : ℕ := 85
def fraction_white_cookies_eaten : ℚ := 3/4

-- Define the initial number of black cookies
def initial_black_cookies : ℕ := initial_white_cookies + additional_black_cookies

-- Calculate the initial number of white cookies eaten
def white_cookies_eaten : ℕ := (fraction_white_cookies_eaten * initial_white_cookies).toNat

-- Calculate the number of white cookies remaining
def white_cookies_remaining : ℕ := initial_white_cookies - white_cookies_eaten

-- Calculate the number of black cookies remaining
def black_cookies_remaining : ℕ := remaining_cookies - white_cookies_remaining

-- Calculate the number of black cookies eaten
def black_cookies_eaten : ℕ := initial_black_cookies - black_cookies_remaining

-- Prove the ratio of black cookies eaten to the initial number of black cookies is 1:2
theorem ratio_of_black_cookies_eaten_to_initial (h: (fraction_white_cookies_eaten * initial_white_cookies).toNat = white_cookies_eaten) :
  black_cookies_eaten * 2 = initial_black_cookies := by
  sorry

end ratio_of_black_cookies_eaten_to_initial_l554_554356


namespace simplify_expression_l554_554272

theorem simplify_expression (x : ℝ) (h : 1 < x ∧ x < 4) : (sqrt((1 - x)^2) - abs(x - 5) = 2 * x - 6) :=
 by sorry

end simplify_expression_l554_554272


namespace cricket_team_matches_in_august_l554_554852

noncomputable def cricket_matches_played_in_august (M W W_new: ℕ) : Prop :=
  W = 26 * M / 100 ∧
  W_new = 52 * (M + 65) / 100 ∧ 
  W_new = W + 65

theorem cricket_team_matches_in_august (M W W_new: ℕ) : cricket_matches_played_in_august M W W_new → M = 120 := 
by
  sorry

end cricket_team_matches_in_august_l554_554852


namespace gas_isothermal_work_l554_554158

noncomputable def gas_ideal_isothermal_work (one_mole : ℤ) (work_isobaric : ℝ) (heat_same : ℝ) : ℝ :=
  let C_p := (5 / 2 : ℝ) * real.R
  let ΔT := work_isobaric / real.R
  let Q_isobaric := (5 / 2 : ℝ) * real.R * ΔT
  have h_Q_isobaric : heat_same = Q_isobaric := by
    sorry
  heat_same

theorem gas_isothermal_work
  (one_mole : ℤ) 
  (work_isobaric : ℝ)
  (h_one_mole : one_mole = 1)
  (h_work_isobaric : work_isobaric = 40)
  (heat_same : ℝ)
  (h_heat_same : heat_same = (5 / 2 : ℝ) * real.R * (40 / real.R)) :
  gas_ideal_isothermal_work one_mole work_isobaric heat_same = 100 := by
  sorry

end gas_isothermal_work_l554_554158


namespace count_base8_integers_with_digit_6_or_7_l554_554829

theorem count_base8_integers_with_digit_6_or_7 : 
  let upper_limit := 512 in 
  let base_8_num_with_6_or_7 := upper_limit - (6 ^ 3) in 
  base_8_num_with_6_or_7 = 296 :=
by 
  sorry

end count_base8_integers_with_digit_6_or_7_l554_554829


namespace cost_price_per_metre_is_45_l554_554112

-- Define the conditions
def total_cost : ℝ := 416.25
def total_metres_of_cloth : ℝ := 9.25

-- Define the cost price per metre calculation
def cost_price_per_metre : ℝ :=
  total_cost / total_metres_of_cloth

-- The theorem to prove the cost price per metre is $45
theorem cost_price_per_metre_is_45 :
  cost_price_per_metre = 45 := by
  sorry

end cost_price_per_metre_is_45_l554_554112


namespace pizza_volume_per_piece_l554_554302

-- Define the given conditions
def pizza_thickness : ℝ := 1 / 4
def pizza_diameter : ℝ := 16
def pizza_pieces : ℕ := 8

-- Compute the radius from the diameter
def pizza_radius : ℝ := pizza_diameter / 2

-- Compute the volume of the entire pizza
def pizza_volume : ℝ := π * pizza_radius^2 * pizza_thickness

-- Volume of one piece of pizza
theorem pizza_volume_per_piece : pizza_volume / pizza_pieces = 2 * π :=
by
  -- Write the proof here
  sorry

end pizza_volume_per_piece_l554_554302


namespace coloring_ways_l554_554605

def is_prime (n : ℕ) : Prop := ∀ d, d ∣ n → d = 1 ∨ d = n

def product (s : set ℕ) : ℕ := s.fold (*) 1

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem coloring_ways :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ n ∧ n ≤ 20 → f n = 0 ∨ f n = 1) ∧
  (∃ w b : set ℕ, (∀ n, n ∈ w ∨ n ∈ b) ∧ (∀ n, n ∈ w → 1 ≤ n ∧ n ≤ 20 ∧ f n = 0) ∧
                   (∀ n, n ∈ b → 1 ≤ n ∧ n ≤ 20 ∧ f n = 1) ∧
                   (product w).gcd (product b) = 1) ∧
  (w ≠ ∅ ∧ b ≠ ∅) ∧
  ∃ f_inv, ∀ n, 1 ≤ n ∧ n ≤ 20 → f_inv (f n) = n :=
  ∃ f : (fin 20) → ℕ, ( ∑ i, Nat.gcd (product (finset.filter (λ x, f x = 0) finset.univ)) (product (finset.filter (λ x, f x = 1) finset.univ))) = 1 ∧ function.bijective (f : fin 20 → ℕ) := 29  :=
sorry

end coloring_ways_l554_554605


namespace fifth_term_second_order_arithmetic_sequence_l554_554703

theorem fifth_term_second_order_arithmetic_sequence :
  let a : ℕ → ℕ := λ n, match n with
                        | 1 => 1
                        | 2 => 3
                        | 3 => 7
                        | 4 => 13
                        | _ => a 4 + 8 -- placeholder for a_5, we'll define the pattern instead
                        end,
      d : ℕ → ℕ := λ n, if n = 1 then 2 else 2 + d (n - 1),
      b : ℕ → ℕ := λ n, match n with
                        | 1 => a 1
                        | 2 => a 2
                        | 3 => a 3
                        | 4 => a 4
                        | _ => b (n-1) + d (n-1)
                        end
  in b 5 = 21 := sorry

end fifth_term_second_order_arithmetic_sequence_l554_554703


namespace cody_payment_l554_554350

def initial_cost : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8

def tax_amount := initial_cost * tax_rate
def total_with_tax := initial_cost + tax_amount
def final_price := total_with_tax - discount
def cody_share := final_price / 2

theorem cody_payment : cody_share = 17 := by
  sorry

end cody_payment_l554_554350


namespace min_small_bottles_l554_554307

theorem min_small_bottles (small_volume : ℕ) (large_volume : ℕ) (absorption_ratio : ℚ) (effective_volume : ℚ) 
  (absorption_volume : ℚ) : 
  (small_volume = 60) → 
  (large_volume = 750) → 
  (absorption_ratio = 0.05) → 
  (absorption_volume = (small_volume : ℚ) * absorption_ratio) → 
  (effective_volume = (small_volume : ℚ) - absorption_volume) → 
  ceil (large_volume / effective_volume) = 14 := 
begin
  intros hsml hlg harr habs heff,
  sorry
end

end min_small_bottles_l554_554307


namespace sum_of_squares_mod_five_eq_zero_l554_554530

theorem sum_of_squares_mod_five_eq_zero (b : ℕ → ℕ) (h1 : ∀ i j, i < j → b i < b j)
  (h2 : ∑ i in finset.range 100, b i = 5050) :
  (∑ i in finset.range 100, (b i)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_mod_five_eq_zero_l554_554530


namespace part_a_part_b_part_c_l554_554568

def f (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), Nat.gcd (k + 1) n

theorem part_a (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
    f (m * n) = f m * f n := sorry

theorem part_b (a : ℕ) : ∃ x : ℕ, f x = a * x := sorry

theorem part_c (a : ℕ) : (∃! x : ℕ, f x = a * x) ↔ ∃ k : ℕ, a = 2^k := sorry

end part_a_part_b_part_c_l554_554568


namespace sufficient_but_not_necessary_l554_554629

theorem sufficient_but_not_necessary (x : ℝ) (h1 : x > 1 → x > 0) (h2 : ¬ (x > 0 → x > 1)) : 
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) := 
by 
  sorry

end sufficient_but_not_necessary_l554_554629


namespace f_val_at_100_l554_554930

theorem f_val_at_100 (f : ℝ → ℝ) (h₀ : ∀ x, f x * f (x + 3) = 12) (h₁ : f 1 = 4) : f 100 = 3 :=
sorry

end f_val_at_100_l554_554930


namespace inequality_l554_554916

variable {n : ℕ}

def x : Fin n → ℝ := sorry

def condition (x : Fin n → ℝ) : Prop :=
  (∀ i, 0 < x i) ∧ (∑ i, x i ≤ ∑ i, (x i)^2)

theorem inequality (x : Fin n → ℝ) (t : ℝ) (h_cond : condition x) (h_t : 1 < t) : 
  ∑ i, (x i)^t ≤ ∑ i, (x i)^(t+1) :=
sorry

end inequality_l554_554916


namespace min_b_value_range_a_monotonic_l554_554424

variables {a b x : ℝ} (f F : ℝ → ℝ)

-- Define f(x) and F(x)
def f (x : ℝ) : ℝ := x^2 + a * x - Real.log x

def F (x : ℝ) : ℝ := f x * Real.exp (-x)

-- 1. Prove the minimum value of b is 0 given a = -1
theorem min_b_value (h : a = -1) : b = 0 :=
sorry

-- 2. Prove the range of a such that F(x) is monotonic on (0,1]
theorem range_a_monotonic (x : ℝ) (h_f : ∀ x, f x = x^2 + a * x - Real.log x) :
  a ≤ 2 ↔ ∀ x ∈ Set.Ioc 0 1, (F' x ≤ 0) :=
sorry

end min_b_value_range_a_monotonic_l554_554424


namespace cover_naturals_l554_554184

noncomputable def irrational_sequences_cover_naturals (x y : ℝ) (hx : Irrational x) (hy : Irrational y) (hxy : 1 / x + 1 / y = 1) : Prop :=
  let seq_x := λ n : ℕ, ⌊n * x⌋
  let seq_y := λ n : ℕ, ⌊n * y⌋
  ∀ n : ℕ, ∃ i : ℕ, seq_x i = n ∨ seq_y i = n

theorem cover_naturals {x y : ℝ} (hx : Irrational x) (hy : Irrational y) (hxy : 1 / x + 1 / y = 1) :
  irrational_sequences_cover_naturals x y hx hy hxy :=
sorry

end cover_naturals_l554_554184


namespace necessary_but_not_sufficient_condition_l554_554412

noncomputable def exists_infinitely_many_even_conditions {α : Type*} (f : α → ℝ) :=
  ∃ (s : Set α), s.Infinite ∧ ∀ x ∈ s, f (-x) = f x

def is_even_function {α : Type*} (f : α → ℝ) :=
  ∀ x, f (-x) = f x

theorem necessary_but_not_sufficient_condition {α : Type*} (f : α → ℝ) (h_inf : ∃ s : Set α, s.Infinite) :
  (exists_infinitely_many_even_conditions f) → ¬ (is_even_function f) :=
begin
  sorry -- Proof not required, only the statement.
end

end necessary_but_not_sufficient_condition_l554_554412


namespace log_base_one_fifth_of_25_l554_554742

theorem log_base_one_fifth_of_25 : log (1/5) 25 = -2 := by
  sorry

end log_base_one_fifth_of_25_l554_554742


namespace net_effect_on_mr_x_l554_554150

theorem net_effect_on_mr_x
  (initial_price : ℝ)
  (profit_percentage : ℝ)
  (loss_percentage : ℝ) :
  let final_sale_price := initial_price * (1 + profit_percentage / 100)
  let repurchase_price := final_sale_price * (1 - loss_percentage / 100)
  final_sale_price - repurchase_price = 1725 :=
by
  assume initial_price profit_percentage loss_percentage
  let final_sale_price := initial_price * (1 + profit_percentage / 100)
  let repurchase_price := final_sale_price * (1 - loss_percentage / 100)
  have h_final_sale_price : final_sale_price = 1.15 * initial_price, by sorry
  have h_repurchase_price : repurchase_price = 0.9 * final_sale_price, by sorry
  have h1 : final_sale_price = 17250, by sorry
  have h2 : repurchase_price = 15525, by sorry
  calc final_sale_price - repurchase_price = 17250 - 15525 : by sorry
                             ... = 1725 : by sorry

end net_effect_on_mr_x_l554_554150


namespace segments_count_l554_554180

/--
Given two concentric circles, with chords of the larger circle that are tangent to the smaller circle,
if each chord subtends an angle of 80 degrees at the center, then the number of such segments 
drawn before returning to the starting point is 18.
-/
theorem segments_count (angle_ABC : ℝ) (circumference_angle_sum : ℝ → ℝ) (n m : ℕ) :
  angle_ABC = 80 → 
  circumference_angle_sum angle_ABC = 360 → 
  100 * n = 360 * m → 
  5 * n = 18 * m →
  n = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end segments_count_l554_554180


namespace fifth_altitude_passes_through_P_l554_554474

-- Define terms and set up the conditions
variables (A B C D E P : Type)
variables [convex_pentagon A B C D E]
variables [concurrent_altitudes A B C D E P]

-- State the theorem
theorem fifth_altitude_passes_through_P 
    (H : altitudes_concurrent A B C D E P) :
  altitude_from E (line_through B C) passes_through P :=
sorry

end fifth_altitude_passes_through_P_l554_554474


namespace integral_quarter_circle_l554_554369

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..1, Real.sqrt (x * (2 - x))

theorem integral_quarter_circle (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 
  (Real.sqrt (x * (2 - x))) = Real.sqrt (1 - (x - 1)^2)) :
  integral_problem = (1/4) * Real.pi :=
  sorry

end integral_quarter_circle_l554_554369


namespace newspapers_sold_correct_l554_554119

def total_sales : ℝ := 425.0
def magazines_sold : ℝ := 150
def newspapers_sold : ℝ := total_sales - magazines_sold

theorem newspapers_sold_correct : newspapers_sold = 275.0 := by
  sorry

end newspapers_sold_correct_l554_554119


namespace find_value_of_M_l554_554785

-- Given conditions
def grid_sequenced (col_diff : Int) (row_diff : Int) (values : List Int) : Prop :=
  values.nth 0 + col_diff = values.nth 1 ∧
  values.nth 1 + col_diff = values.nth 2 ∧
  (values.nth 3) + row_diff = values.nth 4 ∧
  (values.nth 4) + row_diff = values.nth 5

-- Define the problem
theorem find_value_of_M 
  (a b c d e f g : Int)
  (h1 : grid_sequenced 4 (-13/3) [25, 20, 16, M, e, f, g])
  (h2 : (d - g) / 4 = -3)
  (h3 : M = (-1) - ((d - g) / 4)) :
  M = 2 := sorry

end find_value_of_M_l554_554785


namespace sunscreen_cost_l554_554955

theorem sunscreen_cost (hours_at_beach : ℕ) (reapply_interval : ℕ) (ounces_per_application : ℕ) (ounces_per_bottle : ℕ) (cost_per_bottle : ℕ) :
  hours_at_beach = 16 →
  reapply_interval = 2 →
  ounces_per_application = 3 →
  ounces_per_bottle = 12 →
  cost_per_bottle = 7 →
  (hours_at_beach / reapply_interval) * ounces_per_application / ounces_per_bottle * cost_per_bottle = 14 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  rw Nat.div_mul_cancel
  norm_num
  exact Nat.div_mul_cancel
  exact Nat.div_mul_cancel
  sorry

end sunscreen_cost_l554_554955


namespace maximize_triangles_l554_554952

theorem maximize_triangles (n : ℕ) (groups : Fin n → ℕ) (total : Σ (i : Fin n), groups i = 2019) (no_coplanar : ¬ ∃ (a b c d : Fin n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) (distinct_sizes : ∀ i j : Fin n, i ≠ j → groups i ≠ groups j) :
    (Σ (i : Fin n), ∃ j (ij  : i ≠ j), groups j - groups i ≥ 3) → False :=
begin
  sorry
end

end maximize_triangles_l554_554952


namespace probability_product_multiple_of_12_is_1_over_6_l554_554848

variable (s : Finset ℕ) (h : s = {2, 3, 6, 9})

/-- The probability that the product of two randomly chosen numbers from the set {2, 3, 6, 9}
    will be a multiple of 12 is 1/6. -/
theorem probability_product_multiple_of_12_is_1_over_6 :
  (∃ p : ℚ, p = 1/6 ∧
    ∀ (a b ∈ s) (ha : a ≠ b), (a * b) % 12 = 0 ↔ p = 1 / 6) :=
by
  exists sorry

end probability_product_multiple_of_12_is_1_over_6_l554_554848


namespace simplify_sqrt7_to_the_six_l554_554923

theorem simplify_sqrt7_to_the_six : (sqrt 7)^6 = 343 :=
by 
  sorry

end simplify_sqrt7_to_the_six_l554_554923


namespace passed_boys_count_l554_554187

theorem passed_boys_count (total_boys avg_passed avg_failed overall_avg : ℕ) 
  (total_boys_eq : total_boys = 120) 
  (avg_passed_eq : avg_passed = 39) 
  (avg_failed_eq : avg_failed = 15) 
  (overall_avg_eq : overall_avg = 38) :
  let marks_by_passed := total_boys * overall_avg 
                         - (total_boys - passed) * avg_failed;
  let passed := marks_by_passed / avg_passed;
  passed = 115 := 
by
  sorry

end passed_boys_count_l554_554187


namespace polynomial_factors_value_l554_554078

theorem polynomial_factors_value (m n p : ℤ) (h1 : ∃ k, (3 * 3^4 - m * 3^2 + n * 3 - p) = 0)
                                (h2 : ∃ k, (3 * (-4)^4 - m * (-4)^2 + n * (-4) - p) = 0) :
    |m + 2 * n - 4 * p| = 20 := 
sorry

end polynomial_factors_value_l554_554078


namespace at_least_one_room_with_three_doors_l554_554285

def castle_grid (n : ℕ) : Prop :=
  n = 81

def doors_between_adjacent_rooms (grid_size : ℕ) : Prop :=
  ∀ (i j : ℕ), i < grid_size → j < grid_size → 
  (∃ k l : ℕ, (k, l) ∈ [(i+1, j), (i-1, j), (i, j+1), (i, j-1)])

def no_doors_leading_outside (grid_size : ℕ) : Prop :=
  ∀ (i j : ℕ), i = grid_size → j < grid_size → 
  (∃ k l : ℕ, (k, l) ∈ [(i-1, j), (i, j-1), (i, j+1)]) ∧
  (∀ m n : ℕ, m = 0 ∨ n = 0 → (k, l) ∉ [(m-1, n), (m+1, n), (m, n-1), (m, n+1)])

def each_room_at_least_two_doors (grid_size : ℕ) : Prop :=
  ∀ (i j : ℕ), i < grid_size → j < grid_size → 
  (∃ k l m n : ℕ, [(k, l), (m, n)] ⊆ [(i+1, j), (i-1, j), (i, j+1), (i, j-1)])

theorem at_least_one_room_with_three_doors (grid_size : ℕ) 
  (castle : castle_grid grid_size)
  (connections : doors_between_adjacent_rooms grid_size)
  (no_outside_doors : no_doors_leading_outside grid_size)
  (minimum_two_doors : each_room_at_least_two_doors grid_size) :
  ∃ (i j : ℕ), i < grid_size → j < grid_size → 
  (∃ k l m : ℕ, [(k, l), (m, n), (p, q)] ⊆ [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]) :=
sorry

end at_least_one_room_with_three_doors_l554_554285


namespace sin_angle_EAC_l554_554507

variable (A B C D E F G H : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [MetricSpace E] [MetricSpace F] [MetricSpace G] [MetricSpace H]

-- Given conditions
variable (AB CD EF GH BC DA HG : ℝ)
variable (AE BF CG DH : ℝ)
variable hAB : AB = 2
variable hCD : CD = 2
variable hEF : EF = 2
variable hGH : GH = 2
variable hBC : BC = 4
variable hDA : DA = 4
variable hHG : HG = 4
variable hAE : AE = 1
variable hBF : BF = 1
variable hCG : CG = 1
variable hDH : DH = 1

theorem sin_angle_EAC : sin (angle E A C) = 4 / sqrt 17 := by
  sorry

end sin_angle_EAC_l554_554507


namespace harmonic_inequality_l554_554899

noncomputable def harmonic (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ k, 1 / (k + 1 : ℝ))

theorem harmonic_inequality (n : ℕ) (h : n ≥ 2) :
  (harmonic n) ^ 2 > 2 * (Finset.sum (Finset.range (n - 1)) (λ k, harmonic (k + 1) / (k + 2))) :=
by sorry

end harmonic_inequality_l554_554899


namespace right_triangle_c_squared_value_l554_554463

theorem right_triangle_c_squared_value (a b c : ℕ) (h : a = 9) (k : b = 12) (right_triangle : True) :
  c^2 = a^2 + b^2 ∨ c^2 = b^2 - a^2 :=
by sorry

end right_triangle_c_squared_value_l554_554463


namespace induction_inequality_number_of_terms_added_l554_554604

theorem induction_inequality (n : ℕ) (hn : n > 1) : 
  (∑ i in finset.range (2^n - 1), 1 / (i + 1) : ℝ) < n :=
by sorry

theorem number_of_terms_added (k : ℕ) (hk : k > 0) :
  (finset.range (2^(k+1) - 1)).card - (finset.range (2^k - 1)).card = 2^k :=
by sorry

end induction_inequality_number_of_terms_added_l554_554604


namespace proof_problem_l554_554996

theorem proof_problem (a b c : ℤ) (h1 : a > 2) (h2 : b < 10) (h3 : c ≥ 0) (h4 : 32 = a + 2 * b + 3 * c) : 
  a = 4 ∧ b = 8 ∧ c = 4 :=
by
  sorry

end proof_problem_l554_554996


namespace countLuckyNumbers_l554_554665

def isLuckyNumber (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧
  (let q := 221 / n in (221 % n) % q = 0)

theorem countLuckyNumbers : 
  { n : ℕ | isLuckyNumber n }.toFinset.card = 115 :=
by
  sorry

end countLuckyNumbers_l554_554665


namespace escher_prints_probability_l554_554544

theorem escher_prints_probability :
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  probability = 1 / 1320 :=
by
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  sorry

end escher_prints_probability_l554_554544


namespace arithmetic_sequence_n_l554_554129

theorem arithmetic_sequence_n {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  (∃ n : ℕ, a n = 2005) → (∃ n : ℕ, n = 669) :=
by
  sorry

end arithmetic_sequence_n_l554_554129


namespace parallel_PQ_AB_l554_554101

theorem parallel_PQ_AB (ABC : Triangle)
  (D : ABC.BC) (E : ABC.CA) (F : ABC.AB)
  (h1 : FE_parallel_BC : E.F ∥ ABC.BC)
  (h2 : DF_parallel_CA : D.F ∥ ABC.CA)
  (P : ∃ B.E ∩ D.F) 
  (Q : ∃ F.E ∩ A.D) :
  PQ ∥ AB :=
by
  sorry

end parallel_PQ_AB_l554_554101


namespace positive_difference_between_prob_3_and_prob_5_l554_554611

/-- Probability of a coin landing heads up exactly 3 times out of 5 flips -/
def prob_3_heads : ℚ := (nat.choose 5 3) * (1/2)^3 * (1/2)^(5-3)

/-- Probability of a coin landing heads up exactly 5 times out of 5 flips -/
def prob_5_heads : ℚ := (1/2)^5

/-- Positive difference between the probabilities -/
theorem positive_difference_between_prob_3_and_prob_5 : 
  |prob_3_heads - prob_5_heads| = 9 / 32 :=
by sorry

end positive_difference_between_prob_3_and_prob_5_l554_554611


namespace width_decrease_percentage_l554_554203

variable (L W : ℝ) -- original length and width of the rectangle
variable (A : ℝ) -- area of the rectangle
variable (new_length : ℝ) -- new length after increase
variable (new_width : ℝ) -- new width after decrease

-- Conditions
def original_area := L * W
def new_area := new_length * new_width
def length_increase := new_length = 1.40 * L
def area_unchanged := original_area = new_area

-- The theorem to prove
theorem width_decrease_percentage :
  length_increase L new_length →
  area_unchanged L W new_length new_width →
  new_width = (5 / 7) * W :=
sorry

end width_decrease_percentage_l554_554203


namespace compute_expression_l554_554734

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l554_554734


namespace sum_of_solutions_l554_554523

theorem sum_of_solutions (g : ℝ → ℝ) (h : ∀ x, g x = 3 * x - 2) :
  let g_inv x := (x + 2) / 3 in
  ∑ x in {x : ℝ | g_inv x = g (x⁻¹)}, x = -8 :=
by
  -- Definitions based on conditions
  let g_inv (x : ℝ) := (x + 2) / 3
  have inv_eq := (λ x, g_inv x = g (x⁻¹))

  -- Proof setup
  sorry

end sum_of_solutions_l554_554523


namespace cost_price_of_article_l554_554209

theorem cost_price_of_article (C : ℝ) (h1 : 86 - C = C - 42) : C = 64 :=
by
  sorry

end cost_price_of_article_l554_554209


namespace compute_expression_l554_554730

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l554_554730


namespace difference_between_numbers_l554_554948

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 20000) (h2 : b = 2 * a + 6) (h3 : 9 ∣ a) : b - a = 6670 :=
by
  sorry

end difference_between_numbers_l554_554948


namespace sum_of_solutions_l554_554521

theorem sum_of_solutions (g : ℝ → ℝ) (h : ∀ x, g x = 3 * x - 2) :
  let g_inv x := (x + 2) / 3 in
  ∑ x in {x : ℝ | g_inv x = g (x⁻¹)}, x = -8 :=
by
  -- Definitions based on conditions
  let g_inv (x : ℝ) := (x + 2) / 3
  have inv_eq := (λ x, g_inv x = g (x⁻¹))

  -- Proof setup
  sorry

end sum_of_solutions_l554_554521


namespace construct_right_triangle_l554_554784

noncomputable def quadrilateral (A B C D : Type) : Prop :=
∃ (AB BC CA : ℝ), 
AB = BC ∧ BC = CA ∧ 
∃ (angle_D : ℝ), 
angle_D = 30

theorem construct_right_triangle (A B C D : Type) (angle_D: ℝ) (AB BC CA : ℝ) 
    (h1 : AB = BC) (h2 : BC = CA) (h3 : angle_D = 30) : 
    exists DA DB DC : ℝ, (DA * DA) + (DC * DC) = (AD * AD) :=
by sorry

end construct_right_triangle_l554_554784


namespace projection_of_v_on_plane_l554_554012

def projection_of_vector_on_plane 
  (v : ℝ × ℝ × ℝ := (2, 3, 1))
  (plane_normal : ℝ × ℝ × ℝ := (1, 2, -1))
  (p : ℝ × ℝ × ℝ := (5/6, 2/3, 13/6)) : Prop :=
  ∃ (k : ℝ × ℝ × ℝ), 
    k = (plane_normal.1 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2),
        plane_normal.2 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2),
        plane_normal.3 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2)) 
    ∧ (v.1 - k.1, v.2 - k.2, v.3 - k.3) = p

theorem projection_of_v_on_plane :
  projection_of_vector_on_plane :=
sorry

end projection_of_v_on_plane_l554_554012


namespace refill_days_l554_554914

variables (bowl_weight_empty bowl_weight_filled cat_eaten_weight daily_food_amount days : ℕ)

theorem refill_days :
  bowl_weight_empty = 420 →
  bowl_weight_filled = 586 →
  cat_eaten_weight = 14 →
  daily_food_amount = 60 →
  bowl_weight_filled + cat_eaten_weight = bowl_weight_empty + daily_food_amount * days →
  days = 3 :=
by
  intros h1 h2 h3 h4 h5
  have step1 : 600 = bowl_weight_empty + daily_food_amount * days,
    rw [h1, ← h5, h2, h3],
    norm_num,
  have step2 : 600 = 420 + daily_food_amount * days,
    rw h1,
    exact step1,
  have step3 : 180 = daily_food_amount * days,
    linarith,
  have step4 : days = 3,
    rwa h4,
  exact step4

end refill_days_l554_554914


namespace ways_to_make_change_l554_554064

theorem ways_to_make_change :
  ∃ (n : ℕ), n = 42 ∧
    ∃ (ways : set (ℕ × ℕ × ℕ × ℕ)),
      (∀ (p n d q : ℕ), (p * 1 + n * 5 + d * 10 + q * 25 = 50 ∧ ¬ (p * 1 + n * 5 + d * 10 + q * 25 = 50 ∧ q = 2)) ↔ (p, n, d, q) ∈ ways) ∧
      n = ways.to_nat.card :=
begin
  sorry
end

end ways_to_make_change_l554_554064


namespace power_graph_point_l554_554416

theorem power_graph_point :
  ∀ (m n : ℕ), (m = 2 ∧ n = 3 ∧ 8 = (m - 1) * m^n) → n^(-m) = 1 / 9 :=
by
  intros m n h,
  cases h with hm hn,
  cases hn with hn1 hn2,
  have h1 : 8 = (m - 1) * m ^ n := hn2,
  have h2 : m = 2 := hm,
  have h3 : n = 3 := hn1,
  sorry

end power_graph_point_l554_554416


namespace log_base5_rounded_l554_554242

theorem log_base5_rounded (h1: log 5 625 = 4) (h2: 5^4 = 625) (h3: log 5 3125 = 5) (h4: 5^5 = 3125)
  (h5: log 5 625 < log 5 2300) (h6: log 5 2300 < log 5 3125) : Int.round 5 = 5 := 
sorry

end log_base5_rounded_l554_554242


namespace binom_11_1_l554_554736

theorem binom_11_1 : Nat.choose 11 1 = 11 :=
by
  sorry

end binom_11_1_l554_554736


namespace lcm_18_35_l554_554004

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l554_554004


namespace dad_borrowed_quarters_l554_554558

theorem dad_borrowed_quarters (initial_quarters remaining_quarters borrowed_quarters : ℕ)
    (h_initial : initial_quarters = 783)
    (h_remaining : remaining_quarters = 512)
    (h_borrowed : initial_quarters - remaining_quarters = borrowed_quarters) :
  borrowed_quarters = 271 :=
begin
  simp [h_initial, h_remaining, h_borrowed],
  norm_num,
end

end dad_borrowed_quarters_l554_554558


namespace crayons_left_l554_554220

/-- Given initially 48 crayons, if Kiley takes 1/4 and Joe takes half of the remaining,
then 18 crayons are left. -/
theorem crayons_left (initial_crayons : ℕ) (kiley_fraction joe_fraction : ℚ)
    (h_initial : initial_crayons = 48) (h_kiley : kiley_fraction = 1 / 4) (h_joe : joe_fraction = 1 / 2) :
  let kiley_takes := kiley_fraction * initial_crayons,
      remaining_after_kiley := initial_crayons - kiley_takes,
      joe_takes := joe_fraction * remaining_after_kiley,
      crayons_left := remaining_after_kiley - joe_takes
  in crayons_left = 18 :=
by
  sorry

end crayons_left_l554_554220


namespace div_f_iff_l554_554500

variables (m : ℕ) (p : ℕ) [Fact (Nat.Prime p)]
noncomputable def S : Finset (Vector ℕ m) := 
  Finset.univ.filter (λ v, ∀ i, v[i] > 0 ∧ v[i] ≤ p)

noncomputable def f (a : Vector ℕ m) : ℕ :=
  ((Finset.sum S) (λ v, Finset.prod (Finset.univ) (λ i : Fin m, v[i] ^ a[i])))

theorem div_f_iff (a : Vector ℕ m) : 
  p ∣ f a ↔ ∃ i, (a[i] = 0 ∨ ¬ ((p - 1) ∣ a[i])) :=
sorry

end div_f_iff_l554_554500


namespace probability_of_disease_after_two_positive_tests_l554_554931

-- Define the probabilities
def p_D := 1 / 200
def p_Dc := 199 / 200
def p_T1_given_D := 1
def p_T1_given_Dc := 0.05
def p_T2_given_T1_and_D := 1
def p_T2_given_T1_and_Dc := 0.01

-- Define the intermediate probabilities using the conditions
def p_T1 := p_T1_given_D * p_D + p_T1_given_Dc * p_Dc
def p_D_given_T1 := p_T1_given_D * p_D / p_T1
def p_Dc_given_T1 := 1 - p_D_given_T1
def p_T1_and_T2 := (p_T2_given_T1_and_D * p_D_given_T1 + 
                    p_T2_given_T1_and_Dc * p_Dc_given_T1) * p_T1

-- Define the final probability
def p_D_given_T1_and_T2 := p_T2_given_T1_and_D * p_D / p_T1_and_T2

-- Proof statement
theorem probability_of_disease_after_two_positive_tests :
  p_D_given_T1_and_T2 ≈ 1.833 / 1 :=
begin
  sorry
end

end probability_of_disease_after_two_positive_tests_l554_554931


namespace max_value_expression_l554_554888

theorem max_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, 2 * (a - x) * (x + 2 * real.sqrt (x^2 + b^2)) + a * b ≤ a^2 + a * b + 4 * b^2 :=
by
  sorry

end max_value_expression_l554_554888


namespace fewest_reciprocal_presses_l554_554182

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem fewest_reciprocal_presses :
  reciprocal (reciprocal 50) = 50 :=
by
  sorry

end fewest_reciprocal_presses_l554_554182


namespace sequence_general_term_l554_554055

theorem sequence_general_term (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 1/2)
  (h2 : ∀ n ≥ 1, ∑ i in Finset.range (n + 1), a (i + 1) = (n + 1) ^ 2 * a (n + 1)) :
  a n = 1 / (n * (n + 1)) :=
by
  sorry

end sequence_general_term_l554_554055


namespace cards_relationship_l554_554881

-- Definitions from the conditions given in the problem
variables (x y : ℕ)

-- Theorem statement proving the relationship
theorem cards_relationship (h : x + y = 8 * x) : y = 7 * x :=
sorry

end cards_relationship_l554_554881


namespace basketball_team_lineup_l554_554564

def number_of_ways_to_choose_starting_lineup (total_players tall_players : ℕ) : ℕ :=
  let point_guard_choices := total_players - 1
  let shooting_guard_choices := total_players - 2
  let small_forward_choices := total_players - 3
  tall_players * point_guard_choices * shooting_guard_choices * small_forward_choices

theorem basketball_team_lineup (team_members centers : ℕ) (h_team : team_members = 12) (h_center : centers = 4) :
  number_of_ways_to_choose_starting_lineup team_members centers = 3960 :=
by
  rw [h_team, h_center]
  unfold number_of_ways_to_choose_starting_lineup
  norm_num
  sorry

end basketball_team_lineup_l554_554564


namespace simplify_expression_evaluate_at_values_l554_554024

variables (a b : ℝ)
def A : ℝ := 3 * b^2 - 2 * a^2 + 5 * a * b
def B : ℝ := 4 * a * b + 2 * b^2 - a^2

theorem simplify_expression : 2 * A - 3 * B = -a^2 - 2 * a * b :=
by sorry

theorem evaluate_at_values : 2 * A - 3 * B = 7 :=
by 
  let a := -1
  let b := 4
  sorry

end simplify_expression_evaluate_at_values_l554_554024


namespace minimum_shift_proof_l554_554044

-- Definitions based on conditions
def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) + (1 / 2) * Real.cos (2 * x)
def g (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

-- Variable representing the shift
def varphi : ℝ := Real.pi / 12

-- The proof problem statement
theorem minimum_shift_proof (h : ∀ x : ℝ, f x = g x) : varphi > 0 ∧ varphi = Real.pi / 12 :=
by
  -- We skip the proof steps as the focus is on creating the correct statement
  sorry

end minimum_shift_proof_l554_554044


namespace minimum_value_of_f_range_of_t_l554_554050

noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 3)

theorem minimum_value_of_f :
  (∃ x > 3, f x = 9) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x > 3, f x ≥ t / (t + 1) + 7) ↔ (t ≤ -2 ∨ t > -1) :=
by
  sorry

end minimum_value_of_f_range_of_t_l554_554050


namespace total_boys_in_class_l554_554977

-- Conditions definition
def boys_in_circle (total : ℕ) (i j : ℕ) : Prop :=
  i < total ∧ j < total ∧ 2 * (j - i - 1) = total - 2

-- Theorem statement
theorem total_boys_in_class : 
  ∃ n, boys_in_circle n 5 20 ∧ n = 33 :=
by
  -- Providing the existential witness directly
  use 33
  -- Verifying the conditions hold true for n = 33
  have h1 : 5 < 33 := by
    linarith
  have h2 : 20 < 33 := by
    linarith
  have h3 : 2 * (20 - 5 - 1) = 33 - 2 := by
    norm_num
  exact ⟨⟨h1, h2, h3⟩, rfl⟩

end total_boys_in_class_l554_554977


namespace best_project_is_b_l554_554291

-- Define the scores for innovation and practicality for each project.
def innovation : Project → ℕ
| Project.A := 90
| Project.B := 95
| Project.C := 90
| Project.D := 90

def practicality : Project → ℕ
| Project.A := 90
| Project.B := 90
| Project.C := 95
| Project.D := 85

-- Define a function to compute the total score based on the given weights.
def total_score (proj : Project) : ℝ :=
  (innovation proj : ℝ) * 0.6 + (practicality proj : ℝ) * 0.4

-- Define the proposition that project B has the highest total score.
theorem best_project_is_b : ∀ p : Project,
  total_score Project.B ≥ total_score p :=
sorry

end best_project_is_b_l554_554291


namespace triangle_construction_existence_l554_554739

/-- Definition of the given conditions for the triangle construction problem --/
variables {a d : ℝ} {α : ℝ}
  (B C A O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]

/-- Definition of the incenter O and related points --/
variables (incenter : O → B → C → Prop)
  (bc_side : B → C → ℝ)
  (opposite_angle : α → A → B → C → Prop)
  (segment_length : A → O → d → Prop)

theorem triangle_construction_existence :
  ∃ (ABC : (B → C → A)), 
  (bc_side B C = a) ∧ 
  (opposite_angle α A B C) ∧ 
  (segment_length A O d) ∧ 
  (incenter O B C) := 
sorry

end triangle_construction_existence_l554_554739


namespace correct_answer_l554_554825

-- Definition of the correctness condition
def indicates_number (phrase : String) : Prop :=
  (phrase = "Noun + Cardinal Number") ∨ (phrase = "the + Ordinal Number + Noun")

-- Example phrases to be evaluated
def class_first : String := "Class First"
def the_class_one : String := "the Class One"
def class_one : String := "Class One"
def first_class : String := "First Class"

-- The goal is to prove that "Class One" meets the condition
theorem correct_answer : indicates_number "Class One" :=
by {
  -- Insert detailed proof steps here, currently omitted
  sorry
}

end correct_answer_l554_554825


namespace find_b_angle_twice_another_l554_554103

variables {A B C : Type}
variables (a b c : ℝ)
variables (α β γ : ℝ) -- angles A, B, C
variables {𝜋 : ℝ} (h_abSinC_eq_20SinB : a * b * Real.sin γ = 20 * Real.sin β)
variables (h_a2_c2_eq_41 : a^2 + c^2 = 41) 
variables (h_8CosB_eq_1 : 8 * Real.cos β = 1)

-- Question 1: Prove that b = 6
theorem find_b : b = 6 := 
by
  sorry

-- Question 2: Prove that there is an angle twice the size of another in the triangle
theorem angle_twice_another :
  ∃ α β γ, α + β + γ = 𝜋 ∧ (α = 2 * β ∨ β = 2 * γ ∨ γ = 2 * α) :=
by
  sorry

end find_b_angle_twice_another_l554_554103


namespace coefficient_of_x_squared_l554_554097

open BigOperators

theorem coefficient_of_x_squared :
  (∑ k in Finset.range (5 + 1), (Nat.choose 5 k) * (1 : ℤ)^(5 - k) * (2 : ℤ)^k * (x : ℤ)^k).coeff 2 = 40 :=
by
  sorry

end coefficient_of_x_squared_l554_554097


namespace complex_roots_right_triangle_l554_554729

noncomputable def P (z : ℂ) (s t : ℂ) : ℂ := z^3 + s * z + t

theorem complex_roots_right_triangle (p q r : ℂ) (s t k : ℂ)
  (h1 : p + q + r = 0)
  (h2 : |p|^2 + |q|^2 + |r|^2 = 350)
  (h3 : (∀ z, P z s t = 0 ↔ z = p ∨ z = q ∨ z = r))
  (h4 : (p - q).abs^2 + (q - r).abs^2 = k^2) :
  k^2 = 525 := 
sorry

end complex_roots_right_triangle_l554_554729


namespace crayon_count_after_actions_l554_554221

theorem crayon_count_after_actions (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) :
  initial_crayons = 48 → kiley_fraction = 1 / 4 → joe_fraction = 1 / 2 → 
  let crayons_after_kiley := initial_crayons - (kiley_fraction * initial_crayons).to_nat;
      crayons_after_joe := crayons_after_kiley - (joe_fraction * crayons_after_kiley).to_nat
  in crayons_after_joe = 18 :=
by 
  intros h1 h2 h3;
  sorry

end crayon_count_after_actions_l554_554221


namespace train_speed_54_kmph_l554_554318

/-- A train 150 metres long which takes 10 seconds to pass a pole travels at a speed of 54 km/hr. -/
theorem train_speed_54_kmph (length : ℕ) (time : ℕ) (length_eq : length = 150) (time_eq : time = 10) : (length / time) * 3.6 = 54 :=
by
  rw [length_eq, time_eq]
  -- Proof goes here
  sorry

end train_speed_54_kmph_l554_554318


namespace actual_average_height_corrected_l554_554935

theorem actual_average_height_corrected (n : ℕ) (incorrect_avg_height : ℝ) (incorrect_height : ℝ) (actual_height : ℝ) :
  n = 35 → 
  incorrect_avg_height = 181 →
  incorrect_height = 166 →
  actual_height = 106 →
  (let incorrect_total_height := incorrect_avg_height * n in
   let difference := incorrect_height - actual_height in
   let correct_total_height := incorrect_total_height - difference in
   let actual_avg_height := correct_total_height / n in
   Float.round (actual_avg_height * 100) / 100 = 179.29) :=
by
  intros hn hi_avg_height hi_height ha_height
  simp [hn, hi_avg_height, hi_height, ha_height]
  have h1 : Float.round ((6275 / 35) * 100) / 100 = 179.29 := sorry
  exact h1

end actual_average_height_corrected_l554_554935


namespace problem_condition_l554_554043

noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry
noncomputable def x : ℤ := sorry
noncomputable def a : ℤ := 0
noncomputable def b : ℤ := -m + n

theorem problem_condition 
  (h1 : m ≠ 0)
  (h2 : n ≠ 0)
  (h3 : m ≠ n)
  (h4 : (x + m)^2 - (x^2 + n^2) = (m - n)^2) :
  x = a * m + b * n :=
sorry

end problem_condition_l554_554043


namespace find_B_l554_554534

-- conditions
variables {A B C D : ℝ}
hypothesis h1 : C = 4.5 * B
hypothesis h2 : B = 2.5 * A
hypothesis h3 : D = 0.5 * (A + B)
hypothesis h4 : (A + B + C + D) / 4 = 165

-- statement to prove
theorem find_B : B = 100 :=
by sorry

end find_B_l554_554534


namespace lucky_numbers_count_l554_554671

def is_lucky (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧ (221 % n) % (221 / n) = 0

def count_lucky_numbers : ℕ :=
  (Finset.range 222).filter is_lucky |>.card

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
by
  sorry

end lucky_numbers_count_l554_554671


namespace tangent_line_parabola_l554_554364

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  sorry

end tangent_line_parabola_l554_554364


namespace f_is_odd_l554_554107

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.sqrt (1 + x^2))

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end f_is_odd_l554_554107


namespace min_m_value_for_symmetry_l554_554602

theorem min_m_value_for_symmetry (m : ℝ) (h_pos : m > 0) :
  (∀ x : ℝ, 2 * sin (x + m + π / 3) = -2 * sin (-x + m + π / 3)) →
  m = 2 * π / 3 :=
by
  sorry

end min_m_value_for_symmetry_l554_554602


namespace transaction_result_l554_554283

noncomputable def car_cost := 4 * 16000 / 3
noncomputable def motorcycle_cost := 4 * 16000 / 5
noncomputable def total_cost := car_cost + motorcycle_cost
noncomputable def total_selling_price := 2 * 16000
noncomputable def transaction_loss := total_cost - total_selling_price

theorem transaction_result :
  transaction_loss = 2133.33 :=
by
  sorry

end transaction_result_l554_554283


namespace find_a_l554_554049

-- Define the function f
def f (a x : ℝ) := a * x^3 - 2 * x

-- State the theorem, asserting that if f passes through the point (-1, 4) then a = -2.
theorem find_a (a : ℝ) (h : f a (-1) = 4) : a = -2 :=
by {
    sorry
}

end find_a_l554_554049


namespace intersection_complement_A_B_subset_A_C_l554_554431

-- Definition of sets A, B, and complements in terms of conditions
def setA : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def setB : Set ℝ := { x | 2 < x ∧ x < 10 }
def complement_A : Set ℝ := { x | x < 3 ∨ x ≥ 7 }

-- Proof Problem (1)
theorem intersection_complement_A_B :
  ((complement_A) ∩ setB) = { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } := 
  sorry

-- Definition of set C 
def setC (a : ℝ) : Set ℝ := { x | x < a }
-- Proof Problem (2)
theorem subset_A_C {a : ℝ} (h : setA ⊆ setC a) : a ≥ 7 :=
  sorry

end intersection_complement_A_B_subset_A_C_l554_554431


namespace max_value_of_z_l554_554067

theorem max_value_of_z
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + y ≤ 1)
  (h3 : y ≥ -1) :
  ∃ x y, (y ≥ x) ∧ (x + y ≤ 1) ∧ (y ≥ -1) ∧ (2 * x - y = 1 / 2) := by 
  sorry

end max_value_of_z_l554_554067


namespace value_of_k_l554_554947

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem value_of_k
  (a d : ℝ)
  (a1_eq_1 : a = 1)
  (sum_9_eq_sum_4 : 9/2 * (2*a + 8*d) = 4/2 * (2*a + 3*d))
  (k : ℕ)
  (a_k_plus_a_4_eq_0 : arithmetic_sequence a d k + arithmetic_sequence a d 4 = 0) :
  k = 10 :=
by
  sorry

end value_of_k_l554_554947


namespace dice_probability_l554_554176

theorem dice_probability :
  ∀ (die1 die2 die3 die4 : ℕ) (h_pair1 : die1 = die2) (h_pair2 : die3 = die4) (die1 ≠ die3) (die1 ≠ die4)
  (die5 die6 : ℕ),
  let outcomes := 36 in
  let successful_outcome_count := 15 in
  (successful_outcome_count : ℚ) / outcomes = 5 / 12 :=
by
  intros _ _ _ _ h_pair1 h_pair2 h_diff1 h_diff2 _ _
  let outcomes := 36
  let successful_outcome_count := 15
  have h_pos: outcomes > 0 := by decide
  field_simp [h_pos]
  have h : ((successful_outcome_count : ℚ) / outcomes = 5 / 12) = (successful_outcome_count * (12 : ℚ) = 5 * outcomes) :=
    by field_simp [h_pos]; ring
  rw h
  norm_num
  sorry

end dice_probability_l554_554176


namespace find_a_l554_554037

-- Define the function f based on the given conditions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then 2^x - a * x else -2^(-x) - a * x

-- Define the fact that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = -f (-x)

-- State the main theorem that needs to be proven
theorem find_a (a : ℝ) :
  (is_odd_function (f a)) ∧ (f a 2 = 2) → a = -9 / 8 :=
by
  sorry

end find_a_l554_554037


namespace area_triangle_APQ_l554_554186

noncomputable theory
open Real

def A : Point := Point.mk 4 5

def line1 (m₁ b₁ : ℝ) : Line := fun (x y : ℝ) => y = m₁ * x + b₁
def line2 (m₂ b₂ : ℝ) : Line := fun (x y : ℝ) => y = m₂ * x + b₂

def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

def sum_y_intercepts (b₁ b₂ : ℝ) : Prop := b₁ + b₂ = 4

def distance (A B : Point) : ℝ :=
  sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

def area_triangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_triangle_APQ (b₁ b₂ m₁ m₂ : ℝ) 
  (h1 : perpendicular m₁ m₂)
  (h2 : sum_y_intercepts b₁ b₂) 
  (P : Point := (0, b₁)) 
  (Q : Point := (0, b₂)) : 
  area_triangle A P Q = 8 := 
sorry

end area_triangle_APQ_l554_554186


namespace lcm_18_35_is_630_l554_554007

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l554_554007


namespace apples_count_l554_554540

def mangoes_oranges_apples_ratio (mangoes oranges apples : Nat) : Prop :=
  mangoes / 10 = oranges / 2 ∧ mangoes / 10 = apples / 3

theorem apples_count (mangoes oranges apples : Nat) (h_ratio : mangoes_oranges_apples_ratio mangoes oranges apples) (h_mangoes : mangoes = 120) : apples = 36 :=
by
  sorry

end apples_count_l554_554540


namespace projection_correct_l554_554011

def projection_onto_plane : ℝ × ℝ × ℝ :=
  let v : ℝ × ℝ × ℝ := ⟨2, 3, 1⟩
  let n : ℝ × ℝ × ℝ := ⟨1, 2, -1⟩
  let dot (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let scalar_mul (k : ℝ) (a : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ⟨k * a.1, k * a.2, k * a.3⟩
  let sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ⟨a.1 - b.1, a.2 - b.2, a.3 - b.3⟩
  let c := dot v n / dot n n
  let v_proj_n := scalar_mul c n
  sub v v_proj_n

theorem projection_correct :
  projection_onto_plane = ⟨5/6, 4/6, 13/6⟩ :=
sorry

end projection_correct_l554_554011


namespace sum_a_1_to_100_l554_554809

def f (n : ℕ) : ℝ := n^2 * Real.cos (n * Real.pi)

def a (n : ℕ) : ℝ := f n + f (n + 1)

theorem sum_a_1_to_100 : ∑ n in Finset.range 100, a (n + 1) = -100 :=
by
  sorry

end sum_a_1_to_100_l554_554809


namespace number_of_x_intercepts_l554_554440

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end number_of_x_intercepts_l554_554440


namespace percent_increase_salary_l554_554261

theorem percent_increase_salary (new_salary increase : ℝ) (h_new_salary : new_salary = 90000) (h_increase : increase = 25000) :
  (increase / (new_salary - increase)) * 100 = 38.46 := by
  -- Given values
  have h1 : new_salary = 90000 := h_new_salary
  have h2 : increase = 25000 := h_increase
  -- Compute original salary
  let original_salary : ℝ := new_salary - increase
  -- Compute percent increase
  let percent_increase : ℝ := (increase / original_salary) * 100
  -- Show that the percent increase is 38.46
  have h3 : percent_increase = 38.46 := sorry
  exact h3

end percent_increase_salary_l554_554261


namespace parking_savings_l554_554624

theorem parking_savings
  (cost_per_week : ℕ)
  (cost_per_month : ℕ)
  (weeks_per_year : ℕ)
  (months_per_year : ℕ)
  (h_cost_per_week : cost_per_week = 10)
  (h_cost_per_month : cost_per_month = 42)
  (h_weeks_per_year : weeks_per_year = 52)
  (h_months_per_year : months_per_year = 12) :
  (weeks_per_year * cost_per_week) - (months_per_year * cost_per_month) = 16 := by
  rw [h_cost_per_week, h_cost_per_month, h_weeks_per_year, h_months_per_year]
  calc
    52 * 10 - 12 * 42 = 520 - 504 : by norm_num
                   ... = 16 : by norm_num

end parking_savings_l554_554624


namespace odd_function_value_at_2_l554_554388

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

theorem odd_function_value_at_2 : f (-2) + f (2) = 0 :=
by
  sorry

end odd_function_value_at_2_l554_554388


namespace number_of_different_lines_l554_554775

theorem number_of_different_lines :
  let S := {0, 1, 3, 5, 7, 9}
  in ∃ n : ℕ, n = 20 ∧ (∀ A B : ℤ, A ≠ B ∧ A ∈ S ∧ B ∈ S → 
  ∃ f : ℤ × ℤ → Prop, f (A, B) ↔ (A, B) = (0, 1) ∨ (A, B) = (0, 3) ∨ 
  (A, B) = (0, 5) ∨ (A, B) = (0, 7) ∨ (A, B) = (0, 9) ∨ 
  (A, B) = (1, 0) ∨ (A, B) = (3, 0) ∨ (A, B) = (5, 0) ∨ 
  (A, B) = (7, 0) ∨ (A, B) = (9, 0) ∨ 
  (A, B) = (1, 3) ∨ (A, B) = (1, 5) ∨ (A, B) = (1, 7) ∨ 
  (A, B) = (1, 9) ∨ (A, B) = (3, 5) ∨ (A, B) = (3, 7) ∨ 
  (A, B) = (3, 9) ∨
  (A, B) = (5, 7) ∨ (A, B) = (5, 9) ∨ 
  (A, B) = (7, 9)) := 
  sorry

end number_of_different_lines_l554_554775


namespace greatest_possible_radius_of_circle_l554_554839

theorem greatest_possible_radius_of_circle
  (π : Real)
  (r : Real)
  (h : π * r^2 < 100 * π) :
  ∃ (n : ℕ), n = 9 ∧ (r : ℝ) ≤ 10 ∧ (r : ℝ) ≥ 9 :=
by
  sorry

end greatest_possible_radius_of_circle_l554_554839


namespace cube_truncation_edges_l554_554334

-- Define the initial condition: a cube
def initial_cube_edges : ℕ := 12

-- Define the condition of each corner being cut off
def corners_cut (corners : ℕ) (edges_added : ℕ) : ℕ :=
  corners * edges_added

-- Define the proof problem
theorem cube_truncation_edges : initial_cube_edges + corners_cut 8 3 = 36 := by
  sorry

end cube_truncation_edges_l554_554334


namespace solve_equation_l554_554366

variable (a b c : ℝ)

theorem solve_equation (h : (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1) : 
  a * c = 36 * b :=
by 
  -- Proof goes here
  sorry

end solve_equation_l554_554366


namespace integer_solution_existence_l554_554621

theorem integer_solution_existence : ∃ (x y : ℤ), 2 * x + y - 1 = 0 :=
by
  use 1
  use -1
  sorry

end integer_solution_existence_l554_554621


namespace inequality_abc_l554_554898

theorem inequality_abc (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 2) (h₃ : 0 ≤ b) (h₄ : b ≤ 2) (h₅ : 0 ≤ c) (h₆ : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end inequality_abc_l554_554898


namespace time_lent_to_C_eq_l554_554644

variable (principal_B : ℝ := 5000)
variable (time_B : ℕ := 2)
variable (principal_C : ℝ := 3000)
variable (total_interest : ℝ := 1980)
variable (rate_of_interest_per_annum : ℝ := 0.09)

theorem time_lent_to_C_eq (n : ℝ) (H : principal_B * rate_of_interest_per_annum * time_B + principal_C * rate_of_interest_per_annum * n = total_interest) : 
  n = 2 / 3 :=
by
  sorry

end time_lent_to_C_eq_l554_554644


namespace certain_percentage_l554_554068

theorem certain_percentage (P N : ℕ) (h1 : 0.25 * 16 + 2 ≤ P / 100 * N) (h2 : N = 40) : P = 15 :=
by
  sorry

end certain_percentage_l554_554068


namespace probability_at_most_six_distinct_numbers_l554_554247

def roll_eight_dice : ℕ := 6^8

def favorable_cases : ℕ := 3628800

def probability_six_distinct_numbers (n : ℕ) (f : ℕ) : ℚ :=
  f / n

theorem probability_at_most_six_distinct_numbers :
  probability_six_distinct_numbers roll_eight_dice favorable_cases = 45 / 52 := by
  sorry

end probability_at_most_six_distinct_numbers_l554_554247


namespace true_propositions_about_correlation_and_regression_l554_554147

theorem true_propositions_about_correlation_and_regression :
  (∀ (r : ℝ), abs r ≤ 1) ∧
  (∀ (x y : ℝ), (x, y) = (overline x, overline y)) ∧
  (∀ (x y : ℝ), y = 0.2 * x + 10 → y ≠ 12) ∧
  (∀ (R2 : ℝ), R2 ∈ [0, 1] → ∃ SSR : ℝ, SSR ≥ 0) →
  true := by sorry

end true_propositions_about_correlation_and_regression_l554_554147


namespace crayon_count_after_actions_l554_554222

theorem crayon_count_after_actions (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) :
  initial_crayons = 48 → kiley_fraction = 1 / 4 → joe_fraction = 1 / 2 → 
  let crayons_after_kiley := initial_crayons - (kiley_fraction * initial_crayons).to_nat;
      crayons_after_joe := crayons_after_kiley - (joe_fraction * crayons_after_kiley).to_nat
  in crayons_after_joe = 18 :=
by 
  intros h1 h2 h3;
  sorry

end crayon_count_after_actions_l554_554222


namespace count_lucky_numbers_l554_554679

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 221 ∧ let q := 221 / n in let r := 221 % n in r % q = 0

theorem count_lucky_numbers : ∃ (N : ℕ), N = 115 ∧ (∀ n, is_lucky n → n ≤ 221) := 
by
  apply Exists.intro 115
  constructor
  · rfl
  · intros n hn
    cases hn with hpos hrange
    exact hrange.left
  · sorry

end count_lucky_numbers_l554_554679


namespace part1_max_value_l554_554810

variable (f : ℝ → ℝ)
def is_maximum (y : ℝ) := ∀ x : ℝ, f x ≤ y

theorem part1_max_value (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + m*x + 1) :
  m = 0 → (exists y, is_maximum f y ∧ y = 1) := 
sorry

end part1_max_value_l554_554810


namespace smallest_positive_period_and_b_value_l554_554046

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin(2 * x) + cos(2 * x) - 1

def triangle := {A B C : ℝ} (a b c : ℝ)
  (angles : A + B + C = π)
  (sides : b^2 = a^2 + c^2 - 2 * a * c * cos B)

theorem smallest_positive_period_and_b_value :
  (∀ (k : ℤ), -π/3 + k * π ≤ x ∧ x ≤ π/6 + k * π → ∀ x : ℝ, ∀ a c : ℝ, f B = 0 → a + c = 4 → b = sqrt 7) ∧
  (∀ x : ℝ, ∃ T : ℝ, ∀ B : ℝ, T = π) :=
begin
  sorry
end

end smallest_positive_period_and_b_value_l554_554046


namespace maximum_value_expression_l554_554606

-- Definitions
def f (x : ℝ) := -3 * x^2 + 18 * x - 1

-- Lean statement to prove that the maximum value of the function f is 26.
theorem maximum_value_expression : ∃ x : ℝ, f x = 26 :=
sorry

end maximum_value_expression_l554_554606


namespace max_x2_y2_z4_l554_554528

theorem max_x2_y2_z4 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
sorry

end max_x2_y2_z4_l554_554528


namespace complement_A_inter_B_eq_l554_554432

-- Definitions of the sets A and B.
def A : Set ℝ := {x | x^2 - 4 * x > 0}
def B : Set ℝ := {x | x > 1}

-- The theorem statement to prove.
theorem complement_A_inter_B_eq : (∁ A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 4} :=
by
  sorry

end complement_A_inter_B_eq_l554_554432


namespace opposite_of_2_minus_sqrt_5_l554_554946

theorem opposite_of_2_minus_sqrt_5 : -(2 - real.sqrt 5) = real.sqrt 5 - 2 := 
by
  sorry

end opposite_of_2_minus_sqrt_5_l554_554946


namespace red_cookies_count_l554_554543

-- Definitions of the conditions
def total_cookies : ℕ := 86
def pink_cookies : ℕ := 50

-- The proof problem statement
theorem red_cookies_count : ∃ y : ℕ, y = total_cookies - pink_cookies := by
  use 36
  show 36 = total_cookies - pink_cookies
  sorry

end red_cookies_count_l554_554543


namespace log_one_fifth_25_eq_neg2_l554_554751

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end log_one_fifth_25_eq_neg2_l554_554751


namespace probability_valid_arrangement_l554_554021

-- Definitions for the problem conditions
def starts_and_ends_with (lst : List Char) (ch : Char) : Prop :=
  lst.head? = some ch ∧ lst.getLast? = some ch

def no_two_adjacent (lst : List Char) (ch : Char) : Prop :=
  ∀ i, i < lst.length - 1 → lst.nth i = some ch → lst.nth (i + 1) ≠ some ch

def valid_arrangement (lst : List Char) : Prop :=
  starts_and_ends_with lst 'O' ∧ no_two_adjacent lst 'X' ∧ 
  lst.count 'X' = 4 ∧ lst.count 'O' = 3

-- Proposition to prove
theorem probability_valid_arrangement : 
  (∃ lst : List Char, valid_arrangement lst) →
  (number_of_valid_arrangements / number_of_total_arrangements = 2 / 35) :=
sorry

end probability_valid_arrangement_l554_554021


namespace required_samples_l554_554719

def p := 0.85
def q := 1 - p
def epsilon := 0.01
def P := 0.997
def t := 3

theorem required_samples : ∃ n, n ≈ 11475 ∧ P ≈ Φ(epsilon * sqrt(n / (p * q))) where
  n = t^2 * p * q / epsilon^2
  sorry

end required_samples_l554_554719


namespace sin_960_eq_sqrt3_over_2_neg_l554_554589

-- Conditions
axiom sine_periodic : ∀ θ, Real.sin (θ + 360 * Real.pi / 180) = Real.sin θ

-- Theorem to prove
theorem sin_960_eq_sqrt3_over_2_neg : Real.sin (960 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  -- skipping the proof
  sorry

end sin_960_eq_sqrt3_over_2_neg_l554_554589


namespace plane_equation_l554_554127

theorem plane_equation (v w : ℝ × ℝ × ℝ)
  (h_w : w = (3, -1, 3))
  (h_proj_v_w : (v.1 * w.1 + v.2 * w.2 + v.3 * w.3) / (w.1 * w.1 + w.2 * w.2 + w.3 * w.3) * w = (6, -2, 6)) :
  3 * v.1 - v.2 + 3 * v.3 - 38 = 0 :=
sorry

end plane_equation_l554_554127


namespace crayons_left_l554_554224

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
  (initial_crayons_eq : initial_crayons = 48) 
  (kiley_fraction_eq : kiley_fraction = 1/4) 
  (joe_fraction_eq : joe_fraction = 1/2): 
  (initial_crayons - initial_crayons * kiley_fraction - (initial_crayons - initial_crayons * kiley_fraction) * joe_fraction) = 18 := 
by 
  sorry

end crayons_left_l554_554224


namespace concave_number_probability_l554_554317

/-- Definition of a concave number -/
def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

/-- Set of possible digits -/
def digits : Finset ℕ := {4, 5, 6, 7, 8}

 /-- Total number of distinct three-digit combinations -/
def total_combinations : ℕ := 60

 /-- Number of concave numbers -/
def concave_numbers : ℕ := 20

 /-- Probability that a randomly chosen three-digit number is a concave number -/
def probability_concave : ℚ := concave_numbers / total_combinations

theorem concave_number_probability :
  probability_concave = 1 / 3 :=
by
  sorry

end concave_number_probability_l554_554317


namespace infinite_series_sum_l554_554372

theorem infinite_series_sum :
  (∑' n : ℕ, (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l554_554372


namespace increasing_interval_of_f_l554_554943

def quadratic (x : ℝ) : ℝ := x^2 + 2*x - 3
def log_base_half (t : ℝ) : ℝ := Real.log t / Real.log (1/2)
def f (x : ℝ) : ℝ := log_base_half (quadratic x)

theorem increasing_interval_of_f :
  {x : ℝ | quadratic x > 0} = (-∞, -3) ∪ (1, ∞) →
  ∃ I : Set ℝ, I = {x : ℝ | f x > 0} ∧ I = (-∞, -3) := sorry

end increasing_interval_of_f_l554_554943


namespace hyperbola_foci_product_l554_554124

theorem hyperbola_foci_product
  (F1 F2 P : ℝ × ℝ)
  (hF1 : F1 = (-Real.sqrt 5, 0))
  (hF2 : F2 = (Real.sqrt 5, 0))
  (hP : P.1 ^ 2 / 4 - P.2 ^ 2 = 1)
  (hDot : (P.1 + Real.sqrt 5) * (P.1 - Real.sqrt 5) + P.2 ^ 2 = 0) :
  (Real.sqrt ((P.1 + Real.sqrt 5) ^ 2 + P.2 ^ 2)) * (Real.sqrt ((P.1 - Real.sqrt 5) ^ 2 + P.2 ^ 2)) = 2 :=
sorry

end hyperbola_foci_product_l554_554124


namespace product_of_real_roots_l554_554363

theorem product_of_real_roots :
  (∏ x in { x : ℝ | x ^ (Real.log x / Real.log 10) = 100 }, x) = 1 :=
by
  sorry

end product_of_real_roots_l554_554363


namespace cos_double_angle_l554_554832

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2 * n) = 12) : Real.cos (2 * θ) = 5 / 6 := 
sorry

end cos_double_angle_l554_554832


namespace stack_map_view_correct_l554_554759

def heights : list (list ℕ) := [
  [3, 1, 2],  -- First column
  [2, 4, 3],  -- Second column
  [1, 1, 3]   -- Third column
]

def front_view (hs : list (list ℕ)) : list ℕ :=
  hs.map (λ col, col.maximum.toNat)

def right_view (hs : list (list ℕ)) : list ℕ :=
  (list.zipWith (λ a b, nat.max a b) hs.head (hs.tail.getD []).head) ::
  (list.zipWith (λ a b, nat.max a b) hs.head (hs.tail.getD []).tail.getD []).reverse

theorem stack_map_view_correct : 
  front_view heights = [3, 4, 3] ∧ 
  right_view heights = [3, 4, 3] :=
by 
  -- Proof will go here
  sorry

end stack_map_view_correct_l554_554759


namespace nina_money_proof_l554_554915

def total_money_nina_has (W M : ℝ) : Prop :=
  (10 * W = M) ∧ (14 * (W - 1.75) = M)

theorem nina_money_proof (W M : ℝ) (h : total_money_nina_has W M) : M = 61.25 :=
by 
  sorry

end nina_money_proof_l554_554915


namespace increasing_function_range_l554_554199

theorem increasing_function_range (k : ℝ) :
  (∀ x y : ℝ, x < y → (k + 2) * x + 1 < (k + 2) * y + 1) ↔ k > -2 :=
by
  sorry

end increasing_function_range_l554_554199


namespace plates_arrangement_correct_l554_554647

def plates_arrangement := 
  let total_ways := Nat.fact 15 / ((Nat.fact 6) * (Nat.fact 3) * (Nat.fact 3) * (Nat.fact 2) * (Nat.fact 2))
  let green_block_ways := (Nat.fact 14 / ((Nat.fact 6) * (Nat.fact 3) * (Nat.fact 2) * (Nat.fact 2) * (Nat.fact 1))) * (Nat.fact 3)
  total_ways - green_block_ways

theorem plates_arrangement_correct :
  plates_arrangement = (Nat.fact 15 / (Nat.fact 6 * Nat.fact 3 * Nat.fact 3 * Nat.fact 2 * Nat.fact 2)) -
                      ((Nat.fact 14 / (Nat.fact 6 * Nat.fact 3 * Nat.fact 2 * Nat.fact 2 * Nat.fact 1)) * Nat.fact 3) :=
  sorry

end plates_arrangement_correct_l554_554647


namespace problem_statement_l554_554123

def gcd (a b : Nat) : Nat := a.gcd b
def lcm (a b : Nat) : Nat := a.lcm b

theorem problem_statement (a b c d P Q X M N Y : Nat)
  (distinct : ∀ x ∈ [a, b, c, d], x ∉ [a, b, c, d])
  (gcd_ab : gcd a b = P)
  (gcd_cd : gcd c d = Q)
  (lcm_PQ : lcm P Q = X)
  (lcm_26 : lcm 2 6 = M)
  (lcm_cd : lcm c d = N)
  (gcd_MN : gcd M N = Y) :
  ¬(X > 0 ∧ Y > 0 ∧ ( (X % Y = 0 ∧ Y % X ≠ 0) ∨ (X % Y ≠ 0 ∧ Y % X = 0) ∨ (X ≠ Y) )) :=
sorry

end problem_statement_l554_554123


namespace infinite_geometric_series_sum_l554_554727

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def term_1 : ℝ := (sqrt2 + 1) / (sqrt2 - 1)
def term_2 : ℝ := 1 / (2 - sqrt2)
def term_3 : ℝ := 1 / 2

lemma series_is_geometric : (term_1 = (sqrt2 + 1) / (sqrt2 - 1)) ∧ 
                            (term_2 = 1 / (2 - sqrt2)) ∧
                            (term_3 = 1 / 2) :=
by
  unfold term_1 term_2 term_3
  split
  { rfl }
  split
  { rfl }
  { rfl }

lemma common_ratio (q : ℝ) : q = (sqrt2 - 1) / sqrt2 ∧ q < 1 :=
by
  sorry

theorem infinite_geometric_series_sum : 
  (Σ n, if n = 0 then term_1 else if n = 1 then term_2 else if n = 2 then term_3 else 0) 
  = 4 + 3 * sqrt2 :=
by
  sorry

end infinite_geometric_series_sum_l554_554727


namespace work_done_during_second_isothermal_process_l554_554155

open Real

variables (R : ℝ) (n : ℝ) (one_mole : n = 1) (W_iso : ℝ) (Q_iso : ℝ) (W_iso_equal_Q_iso : W_iso = Q_iso)

def isobaric_work_performed (W : ℝ) := W = 40
def heat_added_during_isobaric_process (Q : ℝ) := Q = 100

theorem work_done_during_second_isothermal_process : 
    ∀ (W_iso : ℝ), 
    (isobaric_work_performed 40) →
    (heat_added_during_isobaric_process 100) →
    (W_iso = 100) :=
by {
    intros,
    sorry
}

end work_done_during_second_isothermal_process_l554_554155


namespace simplify_complex_fraction_l554_554173

theorem simplify_complex_fraction : ((5 - 3 * Complex.i) / (2 - 3 * Complex.i)) = (-19 / 5) - (9 / 5) * Complex.i :=
by
  sorry

end simplify_complex_fraction_l554_554173


namespace necessary_but_not_sufficient_condition_l554_554398

variables (m n : Line) (α : Plane)

--- We define the notion of angle between a line and a plane (needs the user to define this notion formally)
def angle_with_plane (l : Line) (p : Plane) : Angle := sorry

--- Assumption that the angle formed by m with α is equal to the angle formed by n with α
axiom equal_angles : angle_with_plane m α = angle_with_plane n α

--- Now we state the theorem that expresses the necessary but not sufficient condition
theorem necessary_but_not_sufficient_condition (h : equal_angles m n α) : m ∥ n :=
sorry

end necessary_but_not_sufficient_condition_l554_554398


namespace andrew_toasts_l554_554722

-- Given conditions
def cost_of_toast : ℕ := 1
def cost_of_egg : ℕ := 3

def dale_toasts : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_eggs : ℕ := 2

def total_cost : ℕ := 15

-- Prove the number of slices of toast Andrew had
theorem andrew_toasts :
  dale_toasts * cost_of_toast + dale_eggs * cost_of_egg + andrew_eggs * cost_of_egg + andrew_toasts * cost_of_toast = total_cost →
  2 * 1 + 2 * 3 + 2 * 3 + andrew_toasts * 1 = 15 →
  andrew_toasts = 1 :=
by
  intros h1 h2
  sorry

end andrew_toasts_l554_554722


namespace parabola_x_intercepts_l554_554446

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end parabola_x_intercepts_l554_554446


namespace integral_eq_solution_l554_554561

theorem integral_eq_solution (ϕ : ℝ → ℝ) :
  (∀ x, ∫ t in 0..x, ϕ(t) * ϕ(x - t) = x^3 / 6) →
  (ϕ = (fun x => x) ∨ ϕ = (fun x => -x)) :=
by
  intro h
  sorry

end integral_eq_solution_l554_554561


namespace strawberries_final_count_l554_554346

def initial_strawberries := 300
def buckets := 5
def strawberries_per_bucket := initial_strawberries / buckets
def strawberries_removed_per_bucket := 20
def redistributed_in_first_two := 15
def redistributed_in_third := 25

-- Defining the final counts after redistribution
def final_strawberries_first := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_second := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_third := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_third
def final_strawberries_fourth := strawberries_per_bucket - strawberries_removed_per_bucket
def final_strawberries_fifth := strawberries_per_bucket - strawberries_removed_per_bucket

theorem strawberries_final_count :
  final_strawberries_first = 55 ∧
  final_strawberries_second = 55 ∧
  final_strawberries_third = 65 ∧
  final_strawberries_fourth = 40 ∧
  final_strawberries_fifth = 40 := by
  sorry

end strawberries_final_count_l554_554346


namespace interstellar_hotel_vip_accommodation_l554_554571

theorem interstellar_hotel_vip_accommodation :
  ∀ (n : ℕ), (n ≤ 8824) → ( ∃ (r1 r2 : ℕ) (rooms : Finset ℕ),
    rooms = Finset.range (200 - 100 + 1) + 101 ∧ 
    r1 ∈ rooms ∧ r2 ∈ rooms ∧ r1 ≠ r2 ∧ 
    rooms.sum id <= 15050) := sorry

end interstellar_hotel_vip_accommodation_l554_554571


namespace rice_and_flour_bags_l554_554937

theorem rice_and_flour_bags (x : ℕ) (y : ℕ) 
  (h1 : x + y = 351)
  (h2 : x + 20 = 3 * (y - 50) + 1) : 
  x = 221 ∧ y = 130 :=
by
  sorry

end rice_and_flour_bags_l554_554937


namespace increasing_function_solve_inequality_find_range_l554_554797

noncomputable def f : ℝ → ℝ := sorry
def a1 := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f x
def a2 := f 1 = 1
def a3 := ∀ m n : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ -1 ≤ n ∧ n ≤ 1 ∧ m + n ≠ 0 → (f m + f n) / (m + n) > 0

-- Statement for question (1)
theorem increasing_function : 
  (∀ x y : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x < f y) :=
by 
  apply sorry

-- Statement for question (2)
theorem solve_inequality (x : ℝ) :
  (f (x^2 - 1) + f (3 - 3*x) < 0 ↔ 1 < x ∧ x ≤ 4/3) :=
by 
  apply sorry

-- Statement for question (3)
theorem find_range (t : ℝ) :
  (∀ x a : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ a ∧ a ≤ 1 → f x ≤ t^2 - 2*a*t + 1) 
  ↔ (2 ≤ t ∨ t ≤ -2 ∨ t = 0) :=
by 
  apply sorry

end increasing_function_solve_inequality_find_range_l554_554797


namespace find_missing_element_l554_554816

open Set

variable (U : Set ℝ) (M : Set ℝ)

theorem find_missing_element (a : ℝ) (hU : U = {-1, 2, 3, a}) (hM : M = {-1, 3}) (hC : U \ M = {2, 5}) :
  a = 5 :=
sorry

end find_missing_element_l554_554816


namespace lcm_18_35_l554_554000

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l554_554000


namespace log_eq_neg_two_l554_554746

theorem log_eq_neg_two : ∀ (x : ℝ), (1 / 5) ^ x = 25 → x = -2 :=
by
  intros x h
  sorry

end log_eq_neg_two_l554_554746


namespace probability_log_is_integer_l554_554054

def log_is_integer (a b : ℕ) : Prop :=
  ∃ k : ℕ, b = a ^ k

def valid_pairs : list (ℕ × ℕ) :=
  [(2, 3), (3, 2), (2, 8), (8, 2), (2, 9), (9, 2), 
   (3, 8), (8, 3), (3, 9), (9, 3), (8, 9), (9, 8)]

theorem probability_log_is_integer : 
  (valid_pairs.filter (λ p, log_is_integer p.1 p.2)).length / valid_pairs.length = 1 / 6 :=
by
  sorry

end probability_log_is_integer_l554_554054


namespace projection_of_v_on_plane_l554_554013

def projection_of_vector_on_plane 
  (v : ℝ × ℝ × ℝ := (2, 3, 1))
  (plane_normal : ℝ × ℝ × ℝ := (1, 2, -1))
  (p : ℝ × ℝ × ℝ := (5/6, 2/3, 13/6)) : Prop :=
  ∃ (k : ℝ × ℝ × ℝ), 
    k = (plane_normal.1 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2),
        plane_normal.2 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2),
        plane_normal.3 * (v.1 * plane_normal.1 + v.2 * plane_normal.2 + v.3 * plane_normal.3) / 
        (plane_normal.1^2 + plane_normal.2^2 + plane_normal.3^2)) 
    ∧ (v.1 - k.1, v.2 - k.2, v.3 - k.3) = p

theorem projection_of_v_on_plane :
  projection_of_vector_on_plane :=
sorry

end projection_of_v_on_plane_l554_554013


namespace value_of_s1_l554_554983

-- Define s(n) as the concatenated number of the first n perfect squares
def s (n : ℕ) : ℕ :=
  (List.range n).map (λ k, (k+1)*(k+1)).foldl (λ acc x, acc * 10 ^ (Nat.digits 10 x).length + x) 0

-- Add the condition about the number of digits in s(99)
axiom digits_s99 : (Nat.digits 10 (s 99)).length = 355

-- The target theorem asserting s(1) = 1
theorem value_of_s1 : s 1 = 1 :=
by 
  sorry

end value_of_s1_l554_554983


namespace sum_b_n_eq_n_squared_sum_S_n_lt_half_l554_554099

noncomputable def a (n : ℕ) : ℝ := if n = 0 then 0 else (3:ℝ)^(n-1)

noncomputable def b (n : ℕ) : ℝ := 1 + 2 * real.log (a n) / real.log 3

def S (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (b i * b (i + 1))

theorem sum_b_n_eq_n_squared (n : ℕ) : 
  (∑ i in finset.range n, b i) = n^2 :=
sorry

theorem sum_S_n_lt_half (n : ℕ) : 
  S n < 1 / 2 :=
sorry

end sum_b_n_eq_n_squared_sum_S_n_lt_half_l554_554099


namespace triangle_chord_theorem_proof_l554_554465

theorem triangle_chord_theorem_proof
  {A B C D : Type*}
  (dist_AB : dist A B = real.sqrt 3)
  (dist_AC : dist A C = real.sqrt 3)
  (D_on_BC : D ∈ segment B C)
  (dist_AD : dist A D = 1) :
  dist B D * dist D C = 2 :=
sorry

end triangle_chord_theorem_proof_l554_554465


namespace vector_dot_product_result_l554_554777

def vec3 := (ℝ × ℝ × ℝ)

def vector_add (u v : vec3) : vec3 :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def dot_product (u v : vec3) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def a : vec3 := (-3, 2, 5)
def b : vec3 := (1, -3, 0)
def c : vec3 := (7, -2, 1)

theorem vector_dot_product_result: dot_product (vector_add a b) c = -7 := by sorry

end vector_dot_product_result_l554_554777


namespace angle_equality_l554_554991

noncomputable def midpoint (A B : Point) : Point := { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

structure Rectangle : Type where
  A B C D : Point

structure Configuration : Type where
  rectangle : Rectangle
  P : Point
  M : Point := midpoint rectangle.A rectangle.D
  N : Point := midpoint rectangle.B rectangle.C
  Q : Point

axiom extension_of_DC (r : Rectangle) : r.P = ↑r.D -- Point P is on the extension of DC beyond D
axiom intersection_PM_AC (r : Rectangle) : r.Q = line_intersection (line_through r.P r.M) (line_through r.A r.C) -- Q is the intersection point of PM and AC

theorem angle_equality (r : Configuration) (h_ext : extension_of_DC r.rectangle) (h_int : intersection_PM_AC r.rectangle) :
  ∠ r.Q r.N r.M = ∠ r.M r.N r.P := 
by
  sorry

end angle_equality_l554_554991


namespace x_intercepts_count_l554_554444

def parabola_x (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem x_intercepts_count : 
  (∃ y : ℝ, parabola_x y = 0) → 1 := sorry

end x_intercepts_count_l554_554444


namespace number_of_pairs_l554_554577

theorem number_of_pairs : 
  (∀ n m : ℕ, (1 ≤ m ∧ m ≤ 2012) → (5^n < 2^m ∧ 2^m < 2^(m+2) ∧ 2^(m+2) < 5^(n+1))) → 
  (∃ c, c = 279) :=
by
  sorry

end number_of_pairs_l554_554577


namespace simplify_part1_1_simplify_part1_2_calculate_part2_calculate_series_part3_l554_554630

-- Definition for simplification part 1
theorem simplify_part1_1 : (1 / Real.sqrt 2) = (Real.sqrt 2 / 2) :=
sorry

theorem simplify_part1_2 : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
sorry

-- Definition for calculation part 2
theorem calculate_part2 : (1 / (2 - Real.sqrt 3) - (2 / (Real.sqrt 3 - 1))) = 1 :=
sorry

-- Definition for series calculation part 3
theorem calculate_series_part3 :
  ∑ k in Finset.range 2018, (1 / (Real.sqrt (k + 1) + Real.sqrt (k + 2))) = (Real.sqrt 2019 - 1) :=
sorry

end simplify_part1_1_simplify_part1_2_calculate_part2_calculate_series_part3_l554_554630


namespace exist_n_consecutive_no_prime_power_l554_554165

theorem exist_n_consecutive_no_prime_power (n : ℕ) (h_pos_n : 0 < n) :
  ∃ (k : ℕ), ∀ i, 0 ≤ i → i < n → ¬ (∃ (p : ℕ) (m : ℕ), p.prime ∧ (k + i) = p ^ m) :=
by sorry

end exist_n_consecutive_no_prime_power_l554_554165


namespace lucky_numbers_count_l554_554695

def is_lucky (n : ℕ) : Prop := 
  ∃ (k q r : ℕ), 
    1 ≤ n ∧ n ≤ 221 ∧
    221 = n * q + r ∧ 
    r = k * q ∧ 
    r < n

def count_lucky_numbers : ℕ := 
  ∑ n in (finset.range 222).filter is_lucky, 1

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
  by 
    sorry

end lucky_numbers_count_l554_554695


namespace factorial_base_a5_l554_554377

theorem factorial_base_a5 (b : ℕ → ℕ) (n : ℕ) (h_repr : 1034 = ∑ k in finset.range (n + 1), b (k + 1) * (k + 1)!)
    (h_bound : ∀ k, k < n + 1 → b (k + 1) ≤ k + 1) : b 5 = 5 :=
sorry

end factorial_base_a5_l554_554377


namespace exists_valid_arrangement_n_eq_3_n_is_odd_l554_554315

open Nat

-- Given conditions translated to definitions
def students (n : ℕ) : ℕ := 3 * n
def isValidArrangement (n : ℕ) (arrangement : List (Fin 3 → Fin (students n))) : Prop :=
  ∀ i j, i < j → ∃ (k₁ k₂), (k₁ < k₂) ∧ arrangement[i] k₁ = arrangement[j] k₂

-- Prove existence of valid arrangement for n = 3
theorem exists_valid_arrangement_n_eq_3 : ∃ (arrangement : List (Fin 3 → Fin (students 3))), isValidArrangement 3 arrangement := 
  sorry

-- Prove that n must be an odd number
theorem n_is_odd (n : ℕ) (arrangement : List (Fin 3 → Fin (students n))) (valid_arrangement : isValidArrangement n arrangement) : Odd n := 
  sorry

end exists_valid_arrangement_n_eq_3_n_is_odd_l554_554315


namespace wire_cut_length_l554_554322

theorem wire_cut_length (S : ℝ) (hS : S = 27.999999999999993) :
  let L := (2 / 3) * S in
  S + L = 46.67 :=
by
  sorry

end wire_cut_length_l554_554322


namespace solve_for_constants_l554_554378

theorem solve_for_constants : 
  ∃ (t s : ℚ), (∀ x : ℚ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + 12) = 15 * x^4 + s * x^3 + 33 * x^2 + 12 * x + 108) ∧ 
  t = 37 / 5 ∧ 
  s = 11 / 5 :=
by
  sorry

end solve_for_constants_l554_554378


namespace kind_wizard_success_l554_554595

-- Define the necessary conditions in Lean 4:
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- Define the theorem with the necessary conditions:
theorem kind_wizard_success (n : ℕ) (h1 : is_odd n) (h2 : 1 < n) :
  (∃ seating : List ℕ, ∀ i, seating.nth i = some (if i < n then i else i - n)) := 
sorry

end kind_wizard_success_l554_554595


namespace find_n_l554_554243

theorem find_n (n : ℕ) (h : (∑ i in Finset.range 21 + 1, (i + 1) * n) / 21 = 44) : n = 4 :=
sorry

end find_n_l554_554243


namespace log_one_fifth_25_eq_neg2_l554_554752

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end log_one_fifth_25_eq_neg2_l554_554752


namespace cost_of_tax_free_items_l554_554262

-- Definitions
def total_expenditure : ℝ := 25
def sales_tax : ℝ := 0.30
def tax_rate : ℝ := 0.06

-- To be proved
theorem cost_of_tax_free_items :
  ∃ (cost_taxable : ℝ) (cost_tax_free : ℝ), 
    sales_tax = tax_rate * cost_taxable ∧
    total_expenditure = cost_tax_exempt + cost_tax_free ∧
    cost_tax_exempt = 5 ∧
    cost_tax_free = 25 - 5 :=
by simp; use cost_taxable, total_expenditure - cost_taxable; exact sorry

end cost_of_tax_free_items_l554_554262


namespace average_speed_correct_l554_554284

def segment1_speed : ℝ := 35 -- kph
def segment1_distance : ℝ := 30 -- km

def segment2_speed : ℝ := 45 -- kph
def segment2_distance : ℝ := 35 -- km

def segment3_speed : ℝ := 55 -- kph
def segment3_time : ℝ := 50 / 60 -- hours

def segment4_speed : ℝ := 42 -- kph
def segment4_time : ℝ := 20 / 60 -- hours

def segment1_time : ℝ := segment1_distance / segment1_speed
def segment2_time : ℝ := segment2_distance / segment2_speed
def segment3_distance : ℝ := segment3_speed * segment3_time
def segment4_distance : ℝ := segment4_speed * segment4_time

def total_distance : ℝ :=
  segment1_distance + segment2_distance + segment3_distance + segment4_distance

def total_time : ℝ :=
  segment1_time + segment2_time + segment3_time + segment4_time

def average_speed : ℝ := total_distance / total_time

theorem average_speed_correct : abs (average_speed - 44.558) < 0.001 :=
by
  -- This is where the detailed steps of the proof would go.
  sorry

end average_speed_correct_l554_554284


namespace angle_between_a_b_is_90_degrees_l554_554818

open Real

variables (a b : ℝ^3)
def vec_a : ℝ^3 := ![0, 2, 1]
def vec_b : ℝ^3 := ![-1, 1, -2]

noncomputable def cos_theta (a b : ℝ^3) :=
  (a ⬝ b) / ((‖a‖) * (‖b‖))

theorem angle_between_a_b_is_90_degrees : 
  cos_theta vec_a vec_b = 0 → real.arccos 0 = π / 2 :=
by
  sorry

end angle_between_a_b_is_90_degrees_l554_554818


namespace intersection_M_N_l554_554057

noncomputable def M : Set ℝ := { x | -2 < x ∧ x < 3 }
noncomputable def N : Set ℝ := { x | 2^(x+1) ≥ 1 }

theorem intersection_M_N :
  M ∩ N = { x | -1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_M_N_l554_554057


namespace student_hourly_wage_at_library_is_7_50_l554_554314

noncomputable def hourly_wage_library : ℝ :=
  let L := (300 - 15 * (25 - 10)) / 10 in
  L

theorem student_hourly_wage_at_library_is_7_50 : hourly_wage_library = 7.50 := by
  sorry

end student_hourly_wage_at_library_is_7_50_l554_554314


namespace terms_added_l554_554489

noncomputable def f (n : ℕ) : ℝ :=
  ∑ i in finset.range (2^n), 1/((i : ℝ) + 1) -- Representing f(n) = ∑_{i=0}^{2^n-1} 1/(i+1)

theorem terms_added (k : ℕ) :
  let fk := ∑ i in finset.range (2^k), 1/((i : ℝ) + 1)
  let fk1 := ∑ i in finset.range (2^(k+1)), 1/((i : ℝ) + 1)
  (fk1 - fk) = ∑ i in finset.Ico (2^k) (2^(k+1)), 1/((i : ℝ) + 1) := 
by
  sorry

end terms_added_l554_554489


namespace expand_expression_l554_554371

theorem expand_expression (x : ℝ) : (2 * x - 3) * (2 * x + 3) * (4 * x ^ 2 + 9) = 4 * x ^ 4 - 81 := by
  sorry

end expand_expression_l554_554371


namespace log_base_one_fifth_twenty_five_l554_554754

theorem log_base_one_fifth_twenty_five : log (1/5) 25 = -2 :=
by
  sorry

end log_base_one_fifth_twenty_five_l554_554754


namespace positive_difference_between_prob_3_and_prob_5_l554_554613

/-- Probability of a coin landing heads up exactly 3 times out of 5 flips -/
def prob_3_heads : ℚ := (nat.choose 5 3) * (1/2)^3 * (1/2)^(5-3)

/-- Probability of a coin landing heads up exactly 5 times out of 5 flips -/
def prob_5_heads : ℚ := (1/2)^5

/-- Positive difference between the probabilities -/
theorem positive_difference_between_prob_3_and_prob_5 : 
  |prob_3_heads - prob_5_heads| = 9 / 32 :=
by sorry

end positive_difference_between_prob_3_and_prob_5_l554_554613


namespace simplify_frac_sqrt3_minus_2_l554_554725

noncomputable def sqrt3_minus_2 : ℝ := sqrt 3 - 2

theorem simplify_frac_sqrt3_minus_2 : (1 : ℝ) / sqrt3_minus_2 = - sqrt 3 - 2 := 
by
  sorry

end simplify_frac_sqrt3_minus_2_l554_554725


namespace no_two_champions_l554_554265

-- Define the problem setup
axiom Team : Type
axiom matches : Team → Team → Prop  -- Represents the win-loss outcome of a match
axiom superior : Team → Team → Prop  -- Represents the superiority relation

-- Conditions based on the problem statement
axiom matches_symmetric : ∀ (a b : Team), matches a b ∨ matches b a  -- No ties, each pair of teams plays exactly one match
axiom superior_def : ∀ (a b : Team), superior a b ↔ (matches a b ∨ ∃ c : Team, matches a c ∧ matches c b)  -- Definition of superiority

-- Formally state the theorem
theorem no_two_champions (t : Finset Team) :
  ¬∃ (a b : Team), a ≠ b ∧ (∀ x ∈ t, superior a x) ∧ (∀ x ∈ t, superior b x) :=
sorry

end no_two_champions_l554_554265


namespace fraction_representation_l554_554870

noncomputable theory

open nat

theorem fraction_representation :
  ∃ (ADA KOK : ℕ), (ADA.gcd KOK = 1) ∧
    ADA < 1000 ∧ KOK < 1000 ∧
    (ADA ≠ KOK) ∧
    (∀ (a d k o : ℕ), (ADA = 100 * a + 10 * d + a) → (KOK = 100 * k + 10 * o + k) →
      (a ≠ d ∧ a ≠ k ∧ a ≠ o ∧ d ≠ k ∧ d ≠ o ∧ k ≠ o) →
      (0.SNELSNELSNEL... = 0.798679867986...) →
      ((SNEL = ADA * 33) ↔ (ADA = 242 ∧ KOK = 303))) :=
sorry

end fraction_representation_l554_554870


namespace annual_manufacturing_costs_l554_554708

noncomputable theory

-- Define total number of machines
def total_machines : ℝ := 14

-- Define annual output 
def annual_output : ℝ := 70000

-- Define establishment charges
def establishment_charges : ℝ := 12000

-- Define profit rate
def profit_rate : ℝ := 0.125

-- Define machines closed
def machines_closed : ℝ := 7.14

-- Define percentage decrease in profit
def percentage_decrease_profit : ℝ := 0.125

theorem annual_manufacturing_costs :
  let M := annual_output - establishment_charges in 
  M = 58000 :=
by
  sorry

end annual_manufacturing_costs_l554_554708


namespace max_digit_sum_l554_554295

/-- Define the range of valid hour values -/
def hour_values : List (ℕ × ℕ) :=
  List.cons (0, 0) $
  List.cons (0, 1) $
  List.cons (0, 2) $
  List.cons (0, 3) $
  List.cons (0, 4) $
  List.cons (0, 5) $
  List.cons (0, 6) $
  List.cons (0, 7) $
  List.cons (0, 8) $
  List.cons (0, 9) $
  List.cons (1, 0) $
  List.cons (1, 1) $
  List.cons (1, 2) $
  List.cons (1, 3) $
  List.cons (1, 4) $
  List.cons (1, 5) $
  List.cons (1, 6) $
  List.cons (1, 7) $
  List.cons (1, 8) $
  List.cons (1, 9) $
  List.cons (2, 0) $
  List.cons (2, 1) $
  List.cons (2, 2) $
  List.cons (2, 3) $
  List.nil

/-- Define the range of valid minute values -/
def minute_values : List (ℕ × ℕ) :=
  List.cons (0, 0) $
  List.cons (0, 1) $
  List.cons (0, 2) $
  List.cons (0, 3) $
  List.cons (0, 4) $
  List.cons (0, 5) $
  List.cons (0, 6) $
  List.cons (0, 7) $
  List.cons (0, 8) $
  List.cons (0, 9) $
  List.cons (1, 0) $
  List.cons (1, 1) $
  List.cons (1, 2) $
  List.cons (1, 3) $
  List.cons (1, 4) $
  List.cons (1, 5) $
  List.cons (1, 6) $
  List.cons (1, 7) $
  List.cons (1, 8) $
  List.cons (1, 9) $
  List.cons (2, 0) $
  List.cons (2, 1) $
  List.cons (2, 2) $
  List.cons (2, 3) $
  List.cons (2, 4) $
  List.cons (2, 5) $
  List.cons (2, 6) $
  List.cons (2, 7) $
  List.cons (2, 8) $
  List.cons (2, 9) $
  List.cons (3, 0) $
  List.cons (3, 1) $
  List.cons (3, 2) $
  List.cons (3, 3) $
  List.cons (3, 4) $
  List.cons (3, 5) $
  List.cons (3, 6) $
  List.cons (3, 7) $
  List.cons (3, 8) $
  List.cons (3, 9) $
  List.cons (4, 0) $
  List.cons (4, 1) $
  List.cons (4, 2) $
  List.cons (4, 3) $
  List.cons (4, 4) $
  List.cons (4, 5) $
  List.cons (4, 6) $
  List.cons (4, 7) $
  List.cons (4, 8) $
  List.cons (4, 9) $
  List.cons (5, 0) $
  List.cons (5, 1) $
  List.cons (5, 2) $
  List.cons (5, 3) $
  List.cons (5, 4) $
  List.cons (5, 5) $
  List.cons (5, 6) $
  List.cons (5, 7) $
  List.cons (5, 8) $
  List.cons (5, 9) $
  List.nil

/-- Function to calculate the sum of the digits of a time given as (hours, minutes) -/
def digit_sum (time : (ℕ × ℕ) × (ℕ × ℕ)) : ℕ :=
  time.fst.fst + time.fst.snd + time.snd.fst + time.snd.snd

/-- The maximum possible sum of the digits of the time displayed in a 24-hour format is 24 -/
theorem max_digit_sum : ∃ (h : (ℕ × ℕ)), (h ∈ hour_values) ∧ ∃ (m : (ℕ × ℕ)), (m ∈ minute_values) ∧ digit_sum (h, m) = 24 :=
by
  exists (1, 9)
  split
  by simp [hour_values]
  exists (5, 9)
  split
  by simp [minute_values]
  by simp [digit_sum]
  sorry

end max_digit_sum_l554_554295


namespace calculate_x_pow_y_l554_554835

theorem calculate_x_pow_y (x y : ℝ) (h : abs (x + 1/2) + (y - 3)^2 = 0) : x^y = -1 / 8 :=
by {
  have h1 : x = -1/2,
  {
    sorry, -- This will be where we solve |x + 1/2| = 0 to x = -1/2
  },
  have h2 : y = 3,
  {
    sorry, -- This will be where we solve (y - 3)^2 = 0 to y = 3
  },
  rw [h1, h2],
  norm_num,
}

end calculate_x_pow_y_l554_554835


namespace lucky_numbers_count_l554_554658

def isLucky (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≠ 0 ∧ 221 = n * k + (221 % n) ∧ (221 % n) % k = 0

def countLuckyNumbers : ℕ :=
  Nat.card (Finset.filter isLucky (Finset.range 222))

theorem lucky_numbers_count : countLuckyNumbers = 115 :=
  sorry

end lucky_numbers_count_l554_554658


namespace total_steps_l554_554545

def steps_on_feet (jason_steps : Nat) (nancy_ratio : Nat) : Nat :=
  jason_steps + (nancy_ratio * jason_steps)

theorem total_steps (jason_steps : Nat) (nancy_ratio : Nat) (h1 : jason_steps = 8) (h2 : nancy_ratio = 3) :
  steps_on_feet jason_steps nancy_ratio = 32 :=
by
  sorry

end total_steps_l554_554545


namespace find_minimum_value_l554_554812

noncomputable def fixed_point_at_2_2 (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : Prop :=
∀ (x : ℝ), a^(2-x) + 1 = 2 ↔ x = 2

noncomputable def point_on_line (m n : ℝ) (hmn_pos : m * n > 0) : Prop :=
2 * m + 2 * n = 1

theorem find_minimum_value (m n : ℝ) (hmn_pos : m * n > 0) :
  (fixed_point_at_2_2 a ha_pos ha_ne) → (point_on_line m n hmn_pos) → (1/m + 1/n ≥ 8) :=
sorry

end find_minimum_value_l554_554812


namespace value_of_d_l554_554581

theorem value_of_d (d : ℝ) :
  (∀ x : ℝ, (x = (-7 + sqrt (2 * d)) / 2 ∨ x = (-7 - sqrt (2 * d)) / 2) ↔ x^2 + 7 * x + d = 0) →
  d = 49 / 6 :=
by {
  sorry
}

end value_of_d_l554_554581


namespace farey_sequence_mediant_l554_554897

theorem farey_sequence_mediant (a b x y c d : ℕ) (h₁ : a * y < b * x) (h₂ : b * x < y * c) (farey_consecutiveness: bx - ay = 1 ∧ cy - dx = 1) : (x / y) = (a+c) / (b+d) := 
by
  sorry

end farey_sequence_mediant_l554_554897


namespace ratio_angles_l554_554713

noncomputable theory

open_locale classical

variables {A B C O E : Type} [inhabited O] [inhabited E]
variables (h_triangle : ∀ (A B C : Type), triangle A B C)
variables (h_circle : ∀ (A B C O : Type), inscribed_circle A B C O)
variables (h_AB : central_angle A B = 100)
variables (h_BC : central_angle B C = 80)
variables (h_perp : ∀ (E : Type) (AC : Type), perpendicular AC E)

theorem ratio_angles (h_OBE : angle O B E = 20) (h_BAC : angle B A C = 40) : 
  ratio (angle O B E) (angle B A C) = 1 / 2 := 
by 
  sorry

end ratio_angles_l554_554713


namespace number_of_intersections_l554_554361

def line₁ (x y : ℝ) := 2 * x - 3 * y + 6 = 0
def line₂ (x y : ℝ) := 5 * x + 2 * y - 10 = 0
def line₃ (x y : ℝ) := x - 2 * y + 1 = 0
def line₄ (x y : ℝ) := 3 * x - 4 * y + 8 = 0

theorem number_of_intersections : 
  ∃! (p₁ p₂ p₃ : ℝ × ℝ),
    (line₁ p₁.1 p₁.2 ∨ line₂ p₁.1 p₁.2) ∧ (line₃ p₁.1 p₁.2 ∨ line₄ p₁.1 p₁.2) ∧
    (line₁ p₂.1 p₂.2 ∨ line₂ p₂.1 p₂.2) ∧ (line₃ p₂.1 p₂.2 ∨ line₄ p₂.1 p₂.2) ∧
    (line₁ p₃.1 p₃.2 ∨ line₂ p₃.1 p₃.2) ∧ (line₃ p₃.1 p₃.2 ∨ line₄ p₃.1 p₃.2) ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ := 
sorry

end number_of_intersections_l554_554361


namespace new_rectangle_area_l554_554700

variable (l w : ℝ)
variable (h₁ : l * w = 1100)

def l_new := 1.1 * l
def w_new := 0.9 * w

theorem new_rectangle_area : l_new * w_new = 1089 := by
  unfold l_new w_new
  calc
    (1.1 * l) * (0.9 * w) = 1.1 * 0.9 * (l * w) : by ring
    ... = 1.1 * 0.9 * 1100 : by rw [h₁]
    ... = 1089 : by norm_num

end new_rectangle_area_l554_554700


namespace trucks_needed_for_coal_transport_l554_554637

def number_of_trucks (total_coal : ℕ) (capacity_per_truck : ℕ) (x : ℕ) : Prop :=
  capacity_per_truck * x = total_coal

theorem trucks_needed_for_coal_transport :
  number_of_trucks 47500 2500 19 :=
by
  sorry

end trucks_needed_for_coal_transport_l554_554637


namespace part1_part2_part3_l554_554408

variables (a b : ℝ^3)
variables (t : ℝ)

-- Given Conditions
def condition1 := ‖a‖ = 4
def condition2 := ‖b‖ = 3
def condition3 := (2 • a - 3 • b) ⬝ (2 • a + b) = 61

-- Prove parts
theorem part1 (h1 : condition1) (h2 : condition2) (h3 : condition3) : ‖a + b‖ = Real.sqrt 13 := 
sorry

theorem part2 (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
projection a b = 2 := 
sorry

theorem part3 (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
t < 15 / 22 ∧ t ≠ -1 :=
sorry

end part1_part2_part3_l554_554408


namespace rationalize_denominator_unique_form_and_sum_l554_554167

theorem rationalize_denominator_unique_form_and_sum :
  let A := 5
  let B := 2
  let C := 2
  let D := 3
  let expr := (5 : ℝ / 3) * Real.sqrt 2 + (2 : ℝ / 3) * Real.sqrt 5
  A + B + C + D = 12 :=
by
  sorry

end rationalize_denominator_unique_form_and_sum_l554_554167


namespace value_at_3000_l554_554017

noncomputable def f (x : ℕ) : ℕ :=
  Nat.recOn x 1 (λ n fn, if n % 3 = 0 then fn + 2 * (n - 3) + 3 else fn)

theorem value_at_3000 : f 3000 = 3000001 :=
  by
    sorry

end value_at_3000_l554_554017


namespace best_project_is_b_l554_554290

-- Define the scores for innovation and practicality for each project.
def innovation : Project → ℕ
| Project.A := 90
| Project.B := 95
| Project.C := 90
| Project.D := 90

def practicality : Project → ℕ
| Project.A := 90
| Project.B := 90
| Project.C := 95
| Project.D := 85

-- Define a function to compute the total score based on the given weights.
def total_score (proj : Project) : ℝ :=
  (innovation proj : ℝ) * 0.6 + (practicality proj : ℝ) * 0.4

-- Define the proposition that project B has the highest total score.
theorem best_project_is_b : ∀ p : Project,
  total_score Project.B ≥ total_score p :=
sorry

end best_project_is_b_l554_554290


namespace log_base_one_fifth_of_25_l554_554744

theorem log_base_one_fifth_of_25 : log (1/5) 25 = -2 := by
  sorry

end log_base_one_fifth_of_25_l554_554744


namespace min_blue_points_needed_l554_554030

noncomputable theory

-- Define the main theorem
theorem min_blue_points_needed (n : ℕ) (hn : n = 998) :
  ∃ k : ℕ, (k = 1991) ∧ (∀ (R : finset (ℝ × ℝ)), R.card = n → (∀ (B : set (ℝ × ℝ)), B.finite → B.card = k → 
  (∀ (t : finset (finset (ℝ × ℝ × ℝ))), t.card = (R.card.choose 3).to_nat → 
  (∀ r ∈ t, ∃ b ∈ B, b ∈ convex_hull (r : set (ℝ × ℝ))))))) ∧ 
  (¬ ∃ (B : set (ℝ × ℝ)), B.finite ∧ B.card = k-1 ∧ 
  (∀ (t : finset (finset (ℝ × ℝ × ℝ))), t.card = (R.card.choose 3).to_nat → 
  (∀ r ∈ t, ∃ b ∈ B, b ∈ convex_hull (r : set (ℝ × ℝ))))) :=
sorry

end min_blue_points_needed_l554_554030


namespace simplify_and_evaluate_l554_554174

theorem simplify_and_evaluate (a b : ℝ) (h_eqn : a^2 + b^2 - 2 * a + 4 * b = -5) :
  (a - 2 * b) * (a^2 + 2 * a * b + 4 * b^2) - a * (a - 5 * b) * (a + 3 * b) = 120 :=
sorry

end simplify_and_evaluate_l554_554174


namespace right_triangle_leg_square_l554_554477

theorem right_triangle_leg_square (a b c : ℝ) 
  (h1 : c = a + 2) 
  (h2 : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 := 
by
  sorry

end right_triangle_leg_square_l554_554477


namespace function_properties_l554_554198

def f (x : ℝ) : ℝ := sin^2 (x + π / 4) - sin^2 (x - π / 4)

theorem function_properties :
  (∀ x, f (-x) = -f (x)) ∧ (∀ x, f (x + π) = f (x)) :=
by
  sorry

end function_properties_l554_554198


namespace total_amount_paid_l554_554235

theorem total_amount_paid :
  let cost_apples := 15 * 85 in
  let cost_mangoes := 12 * 60 in
  let cost_grapes := 10 * 75 in
  let cost_strawberries := 6 * 150 in
  cost_apples + cost_mangoes + cost_grapes + cost_strawberries = 3645 :=
by
  -- Definitions of costs
  let cost_apples := 15 * 85
  let cost_mangoes := 12 * 60
  let cost_grapes := 10 * 75
  let cost_strawberries := 6 * 150

  -- The theorem statement's verification
  have calc1: cost_apples = 15 * 85 := by rfl
  have calc2: cost_mangoes = 12 * 60 := by rfl
  have calc3: cost_grapes = 10 * 75 := by rfl
  have calc4: cost_strawberries = 6 * 150 := by rfl
  
  show cost_apples + cost_mangoes + cost_grapes + cost_strawberries = 3645
  from calc
    cost_apples + cost_mangoes + cost_grapes + cost_strawberries
      = 1275 + 720 + 750 + 900 := by rw [calc1, calc2, calc3, calc4]
      ... = 3645 := by norm_num

end total_amount_paid_l554_554235


namespace range_of_f_l554_554427

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ y, y = f x ∧ (y ≥ -3 / 2 ∧ y ≤ 3) :=
by {
  sorry
}

end range_of_f_l554_554427


namespace ratio_of_percentages_l554_554980

theorem ratio_of_percentages (x y : ℝ) (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by
  sorry

end ratio_of_percentages_l554_554980


namespace initial_number_of_rabbits_is_50_l554_554479

-- Initial number of weasels
def initial_weasels := 100

-- Each fox catches 4 weasels and 2 rabbits per week
def weasels_caught_per_fox_per_week := 4
def rabbits_caught_per_fox_per_week := 2

-- There are 3 foxes
def num_foxes := 3

-- After 3 weeks, 96 weasels and rabbits are left
def weasels_and_rabbits_left := 96
def weeks := 3

theorem initial_number_of_rabbits_is_50 :
  (initial_weasels + (initial_weasels + weasels_and_rabbits_left)) - initial_weasels = 50 :=
by
  sorry

end initial_number_of_rabbits_is_50_l554_554479


namespace far_reaching_quadrilateral_exists_l554_554504

theorem far_reaching_quadrilateral_exists (n m : ℕ) (hn : 1 ≤ n) (hm : 1 ≤ m) (hnM : n ≤ 10^10) (hmM : m ≤ 10^10) :
  ∃ (a b c d : ℕ) (a1 b1 a2 b2 : ℕ),
  (a ≤ n) ∧ (b ≤ m) ∧ (c ≤ n) ∧ (d ≤ m) ∧
  (a1 ≤ n) ∧ (a1 ≥ 0) ∧ (a2 ≤ n) ∧ (a2 ≥ 0) ∧
  (b1 ≤ m) ∧ (b1 ≥ 0) ∧ (b2 ≤ m) ∧ (b2 ≥ 0) ∧
  -- Vertices conditions for far-reaching quadrilateral
  (a1, b1) ≠ (0, 0) ∧ (a1, b1) ≠ (n, 0) ∧ (a1, b1) ≠ (n, m) ∧ (a1, b1) ≠ (0, m) ∧
  (a2, b2) ≠ (0, 0) ∧ (a2, b2) ≠ (n, 0) ∧ (a2, b2) ≠ (n, m) ∧ (a2, b2) ≠ (0, m) ∧
  -- Area condition
  abs ((a*b1 + c*d1 + a2*b2 + n*m) - (a*b2 + c*d2 + a2*b1 + 0)) / 2 ≤ 10^6 := sorry

end far_reaching_quadrilateral_exists_l554_554504


namespace general_term_b_seq_l554_554354

variable (p q r : ℝ) (q_pos : q > 0) (p_gt_r : p > r) (r_pos : r > 0)

def a_seq : ℕ → ℝ
| 1     := p
| (n+1) := p * (a_seq n)

def b_seq : ℕ → ℝ
| 1     := q
| (n+1) := q * (a_seq n) + r * (b_seq n)

theorem general_term_b_seq (n : ℕ) (hn : n ≥ 1) : 
    b_seq p q r n = q * (p^n - r^n) / (p - r) := by
  sorry

end general_term_b_seq_l554_554354


namespace sum_b_n_bound_l554_554033

-- Define the sequence a_n such that a_1 = 1 and a_(n+1) = 3a_n + 1
def a : ℕ → ℤ
| 0     => 1
| (n+1) => 3 * a n + 1

-- Define the sequence b_n using the sequence a_n
def b (n : ℕ) : ℚ :=
3^n / (2 * a n)

-- State the theorem to prove
theorem sum_b_n_bound (n : ℕ) : (finset.range (n + 1)).sum (λ k, b k * (b k - 1)) < 1 :=
sorry

end sum_b_n_bound_l554_554033


namespace intersect_sets_l554_554814

def M : Set ℝ := { x | x ≥ -1 }
def N : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersect_sets :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } := by
  sorry

end intersect_sets_l554_554814


namespace part1_part2_l554_554419

open Set

variable {U : Type} [TopologicalSpace U]

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def set_B (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem part1 (k : ℝ) (hk : k = 1) :
  A ∩ (univ \ set_B k) = {x | 1 < x ∧ x < 3} :=
by
  sorry

theorem part2 (k : ℝ) (h : set_A ∩ set_B k ≠ ∅) :
  k ≥ -1 :=
by
  sorry

end part1_part2_l554_554419


namespace angle_tangent_chord_half_measure_l554_554552

variables {O A B M : Type*} [HasAngle O A B] [HasAngle A M]

-- Given a circle with center O, a chord AB, and a tangent AM at point A
def angle_tangent_chord (O A B M : Type*) [circle_center O] [is_chord O A B] [is_tangent O A M] : Prop :=
  ∃ (α : ℝ), measure_arc O A B = α ∧ measure_angle M A B = α / 2

theorem angle_tangent_chord_half_measure : angle_tangent_chord O A B M :=
sorry

end angle_tangent_chord_half_measure_l554_554552


namespace heptagon_labeling_impossible_l554_554475

/-- 
  Let a heptagon be given with vertices labeled by integers a₁, a₂, a₃, a₄, a₅, a₆, a₇.
  The following two conditions are imposed:
  1. For every pair of consecutive vertices (aᵢ, aᵢ₊₁) (with indices mod 7), 
     at least one of aᵢ and aᵢ₊₁ divides the other.
  2. For every pair of non-consecutive vertices (aᵢ, aⱼ) where i ≠ j ± 1 mod 7, 
     neither aᵢ divides aⱼ nor aⱼ divides aᵢ. 

  Prove that such a labeling is impossible.
-/
theorem heptagon_labeling_impossible :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) ∧
    (∀ {i j : Fin 7}, (i ≠ j + 1 % 7) → (i ≠ j + 6 % 7) → ¬ (a i ∣ a j) ∧ ¬ (a j ∣ a i)) :=
sorry

end heptagon_labeling_impossible_l554_554475


namespace countLuckyNumbers_l554_554668

def isLuckyNumber (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧
  (let q := 221 / n in (221 % n) % q = 0)

theorem countLuckyNumbers : 
  { n : ℕ | isLuckyNumber n }.toFinset.card = 115 :=
by
  sorry

end countLuckyNumbers_l554_554668


namespace cost_per_mile_l554_554256

variable (x : ℝ)
variable (monday_miles : ℝ) (thursday_miles : ℝ) (base_cost : ℝ) (total_spent : ℝ)

-- Given conditions
def car_rental_conditions : Prop :=
  monday_miles = 620 ∧
  thursday_miles = 744 ∧
  base_cost = 150 ∧
  total_spent = 832 ∧
  total_spent = base_cost + (monday_miles + thursday_miles) * x

-- Theorem to prove the cost per mile
theorem cost_per_mile (h : car_rental_conditions x 620 744 150 832) : x = 0.50 :=
  by
    sorry

end cost_per_mile_l554_554256


namespace find_a_l554_554092

open Real

-- Define the line and the circle
def line (a : ℝ) (x y : ℝ) : Prop := a * x + y = 2
def circle (a : ℝ) (x y : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 16 / 3

-- Define that the line intersects the circle at points A and B, and points form equilateral triangle with center
def intersects (a : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), line a x1 y1 ∧ line a x2 y2 ∧ circle a x1 y1 ∧ circle a x2 y2 ∧
  (d x1 y1 x2 y2 = d x1 y1 1 a / 2 ∨ d x2 y2 1 a / 2) ∧
  equilateral_triangle (x1, y1) (x2, y2) (1, a)

def d (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d12 := d A.1 A.2 B.1 B.2
  let d23 := d B.1 B.2 C.1 C.2
  let d31 := d C.1 C.2 A.1 A.2
  d12 = d23 ∧ d23 = d31

theorem find_a (a : ℝ) (h : intersects a) : a = 0 := by
  sorry

end find_a_l554_554092


namespace min_distance_sq_l554_554038

theorem min_distance_sq (x y : ℝ) (h : x - y - 1 = 0) : (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
sorry

end min_distance_sq_l554_554038


namespace calc_length_RS_l554_554355

-- Define the trapezoid properties
def trapezoid (PQRS : Type) (PR QS : ℝ) (h A : ℝ) : Prop :=
  PR = 12 ∧ QS = 20 ∧ h = 10 ∧ A = 180

-- Define the length of the side RS
noncomputable def length_RS (PQRS : Type) (PR QS h A : ℝ) : ℝ :=
  18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3

-- Define the theorem statement
theorem calc_length_RS {PQRS : Type} (PR QS h A : ℝ) :
  trapezoid PQRS PR QS h A → length_RS PQRS PR QS h A = 18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3 :=
by
  intros
  exact Eq.refl (18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3)

end calc_length_RS_l554_554355


namespace cody_paid_amount_l554_554348

/-- Cody buys $40 worth of stuff,
    the tax rate is 5%,
    he receives an $8 discount after taxes,
    and he and his friend split the final price equally.
    Prove that Cody paid $17. -/
theorem cody_paid_amount
  (initial_cost : ℝ)
  (tax_rate : ℝ)
  (discount : ℝ)
  (final_split : ℝ)
  (H1 : initial_cost = 40)
  (H2 : tax_rate = 0.05)
  (H3 : discount = 8)
  (H4 : final_split = 2) :
  (initial_cost * (1 + tax_rate) - discount) / final_split = 17 :=
by
  sorry

end cody_paid_amount_l554_554348


namespace total_people_on_field_trip_l554_554642

theorem total_people_on_field_trip :
  ∀ (num_vans num_buses students_per_van students_per_bus teachers_per_van teachers_per_bus : ℕ),
    num_vans = 6 →
    num_buses = 8 →
    students_per_van = 6 →
    students_per_bus = 18 →
    teachers_per_van = 1 →
    teachers_per_bus = 2 →
    (num_vans * students_per_van + num_buses * students_per_bus) + (num_vans * teachers_per_van + num_buses * teachers_per_bus) = 202 :=
by
  intros num_vans num_buses students_per_van students_per_bus teachers_per_van teachers_per_bus
  intros hnum_vans hnum_buses hstudents_per_van hstudents_per_bus hteachers_per_van hteachers_per_bus
  rw [hnum_vans, hnum_buses, hstudents_per_van, hstudents_per_bus, hteachers_per_van, hteachers_per_bus]
  have hs : 6 * 6 + 8 * 18 = 180 := by norm_num
  have ht : 6 * 1 + 8 * 2 = 22 := by norm_num
  exact (by norm_num : 180 + 22 = 202)

end total_people_on_field_trip_l554_554642


namespace per_capita_expense_percentage_l554_554698

theorem per_capita_expense_percentage :
  ∀ (x y : ℝ),
  (y = 7.675) ∧ (y = 0.66 * x + 1.562) →
  ( (y / x) * 100 = 82.9 ) :=
by
  intros x y h
  cases h with h1 h2
  rw h1 at h2
  sorry

end per_capita_expense_percentage_l554_554698


namespace spider_socks_shoes_order_l554_554706

theorem spider_socks_shoes_order : ∃ (n : ℕ), n = (16! / 2^8) :=
by
  let socks_and_shoes = 16!
  let valid_permutations = socks_and_shoes / 2^8
  exact ⟨valid_permutations, sorry⟩

end spider_socks_shoes_order_l554_554706


namespace tan_product_identity_l554_554273

-- Lean statement for the mathematical problem
theorem tan_product_identity : 
  (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 := by
  sorry

end tan_product_identity_l554_554273


namespace sum_of_inverse_cubes_divisibility_l554_554019

theorem sum_of_inverse_cubes_divisibility (p : ℕ) (a b : ℤ)
  (hp_prime : p.prime) (hp_gt_two : p > 2)
  (h_sum_eq : (1 : ℚ) + 
              (1 : ℚ) / (2 ^ 3) + 
              (1 : ℚ) / (3 ^ 3) + 
              ... + 
              (1 : ℚ) / ((p - 1) ^ 3) = (a : ℚ) / (b : ℚ)) :
  p ∣ a :=
sorry

end sum_of_inverse_cubes_divisibility_l554_554019


namespace correct_sample_size_l554_554973

-- Definitions based on conditions
def investigation_method (method : String) : Prop :=
  method = "sampling survey"

def weather_prediction (event : String) : Prop :=
  event = "random"

def sample_size (n : ℕ) : Prop :=
  ∀ (sample : List ℕ), sample.length = n

def data_variance_property (data : List ℕ) (scaled_factor : ℕ) : Prop :=
  let variance (lst : List ℕ) : ℕ := (lst.median + lst.mean) ^ 2 -- Placeholder for actual variance calculation
  variance (data.map (λ x => x * scaled_factor)) = scaled_factor^2 * variance data

-- Proof Problem
theorem correct_sample_size :
  sample_size =
    (λ n, ∀ (sample : List ℕ), sample.length = n) :=
by sorry

end correct_sample_size_l554_554973


namespace sum_of_remaining_sides_of_triangle_l554_554956

theorem sum_of_remaining_sides_of_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  A = (75 : ℝ) ∧ B = (60 : ℝ) ∧ C = (180 - A - B) ∧ c = 12 →
  b = (c * (Real.sin B / Real.sin A)) →
  a = (c * (Real.sin C / Real.sin A)) →
  a + b ≈ 19.5 :=
by
  intro A B C a b c
  intro h₁ h₂ h₃
  sorry

end sum_of_remaining_sides_of_triangle_l554_554956


namespace distinct_lines_through_point_and_parabola_l554_554828

noncomputable def num_distinct_lines : ℕ :=
  let num_divisors (n : ℕ) : ℕ :=
    have factors := [2^5, 3^2, 7]
    factors.foldl (fun acc f => acc * (f + 1)) 1
  (num_divisors 2016) / 2 -- as each pair (x_1, x_2) corresponds twice

theorem distinct_lines_through_point_and_parabola :
  num_distinct_lines = 36 :=
by
  sorry

end distinct_lines_through_point_and_parabola_l554_554828


namespace log_one_fifth_25_eq_neg2_l554_554750

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end log_one_fifth_25_eq_neg2_l554_554750


namespace number_of_solutions_5x_plus_10y_eq_50_l554_554593

theorem number_of_solutions_5x_plus_10y_eq_50 : 
  (∃! (n : ℕ), ∃ (xy : ℕ × ℕ), xy.1 + 2 * xy.2 = 10 ∧ n = 6) :=
by
  sorry

end number_of_solutions_5x_plus_10y_eq_50_l554_554593


namespace statement_D_is_incorrect_l554_554340

-- Define the initial conditions
def boxA_red_balls := 3
def boxA_yellow_balls := 1
def boxB_red_balls := 1
def boxB_yellow_balls := 3
def totalA := boxA_red_balls + boxA_yellow_balls
def totalB := boxB_red_balls + boxB_yellow_balls

-- Define the expected number of red balls in Box A and Box B after exchange
def E1 (i : ℕ) : ℚ := 
  if i = 1 then (9 * 2 / 16 + 3 * 3 / 8 + 4 * 1 / 16)
  else if i = 2 then (1 * 1 / 4 + 2 * 1 / 2 + 3 * 1 / 4)
  else if i = 3 then (0 * 1 / 16 + 1 * 3 / 8 + 2 * 9 / 16)
  else 0

def E2 (i : ℕ) : ℚ := 
  if i = 1 then (9 * 2 / 16 + 1 * 3 / 8)
  else if i = 2 then (3 * 1 / 4 + 2 * 1 / 2 + 1 * 1 / 4)
  else if i = 3 then (4 * 1 / 16 + 3 * 3 / 8 + 2 * 9 / 16)
  else 0

-- Proof statement
theorem statement_D_is_incorrect : ¬ (E1 3 < E2 1) :=
by {
  have h₁ : E1 3 = 3 / 2 := by refl,
  have h₂ : E2 1 = 3 / 2 := by refl,
  rw [h₁, h₂],
  linarith,
}

end statement_D_is_incorrect_l554_554340


namespace relationship_among_abc_l554_554780

noncomputable def a : ℝ := 2 ^ 0.3
noncomputable def b : ℝ := 0.3 ^ 2
noncomputable def c : ℝ := Real.log 5 / Real.log 2

theorem relationship_among_abc : b < a ∧ a < c :=
by
  have ha : 1 < a := sorry  -- Proof that 1 < 2 ^ 0.3
  have ha2 : a < 2 := sorry -- Proof that 2 ^ 0.3 < 2
  have hb : b < 1 := sorry  -- Proof that 0.3 ^ 2 < 1
  have hc : 2 < c := sorry  -- Proof that log_2 5 > log_2 4 = 2
  exact ⟨hb.trans ha, ha2.trans hc⟩

end relationship_among_abc_l554_554780


namespace preston_total_received_l554_554163

-- Conditions
def cost_per_sandwich := 5
def delivery_fee := 20
def number_of_sandwiches := 18
def tip_percentage := 0.10

-- Correct Answer
def total_amount_received := 121

-- Lean Statement
theorem preston_total_received : 
  (cost_per_sandwich * number_of_sandwiches + delivery_fee) * (1 + tip_percentage) = total_amount_received :=
by 
  sorry

end preston_total_received_l554_554163


namespace find_cost_price_l554_554979

theorem find_cost_price 
  (C : ℝ)
  (h1 : 1.10 * C + 110 = 1.15 * C)
  : C = 2200 :=
sorry

end find_cost_price_l554_554979


namespace exist_identical_2x2_squares_l554_554280

theorem exist_identical_2x2_squares : 
  ∃ sq1 sq2 : Finset (Fin 5 × Fin 5), 
    sq1.card = 4 ∧ sq2.card = 4 ∧ 
    (∀ (i : Fin 5) (j : Fin 5), 
      (i = 0 ∧ j = 0) ∨ (i = 4 ∧ j = 4) → 
      (i, j) ∈ sq1 ∧ (i, j) ∈ sq2 ∧ 
      (sq1 ≠ sq2 → ∃ p ∈ sq1, p ∉ sq2)) :=
sorry

end exist_identical_2x2_squares_l554_554280


namespace rectangular_coordinate_eq_of_polar_eq_range_PA_PB_l554_554488

-- Rectangular coordinate equation of curve C from polar equation
theorem rectangular_coordinate_eq_of_polar_eq :
  ∀ (x y : ℝ),
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) → 
  (ρ^2 = 4 / (4 * Real.sin θ ^ 2 + Real.cos θ ^ 2)) → 
  (x^2 / 4 + y^2 = 1) := 
sorry

-- Range of values for |PA| * |PB|
theorem range_PA_PB :
  ∀ (t α : ℝ),
  (x = -1 + t * Real.cos α ∧ y = 1/2 + t * Real.sin α) → 
  (x^2 / 4 + y^2 = 1) → 
  (P = ⟨(-1 : ℝ), (1/2 : ℝ)⟩) → 
  (1 / (1 + 3 * (Real.sin α) ^ 2)) ∈ Icc (1/2 : ℝ) (2 : ℝ) := 
sorry

end rectangular_coordinate_eq_of_polar_eq_range_PA_PB_l554_554488


namespace lambda_ge_9_over_4_l554_554490

noncomputable def smallest_lambda (λ : ℝ) : Prop :=
  ∀ (n : ℕ), n > 0 → λ * (2^(n-1)) ≥ n^2

theorem lambda_ge_9_over_4 : ∃ (λ : ℝ), smallest_lambda λ ∧ λ = 9 / 4 :=
begin
  use 9 / 4,
  sorry
end

end lambda_ge_9_over_4_l554_554490


namespace inequality_proof_l554_554918

open Real Nat

theorem inequality_proof (n : ℕ) :
  1 - 1 / (n + 1 : ℝ) < ∑ i in range (1, n + 1), (i + sqrt (n^2 + i : ℝ)) / ((n + i) * sqrt (n^2 + i + 2 : ℝ)) ∧ 
  ∑ i in range (1, n + 1), (i + sqrt (n^2 + i : ℝ)) / ((n + i) * sqrt (n^2 + i + 2 : ℝ)) < 1 :=
by
  sorry

end inequality_proof_l554_554918


namespace crayons_left_l554_554219

/-- Given initially 48 crayons, if Kiley takes 1/4 and Joe takes half of the remaining,
then 18 crayons are left. -/
theorem crayons_left (initial_crayons : ℕ) (kiley_fraction joe_fraction : ℚ)
    (h_initial : initial_crayons = 48) (h_kiley : kiley_fraction = 1 / 4) (h_joe : joe_fraction = 1 / 2) :
  let kiley_takes := kiley_fraction * initial_crayons,
      remaining_after_kiley := initial_crayons - kiley_takes,
      joe_takes := joe_fraction * remaining_after_kiley,
      crayons_left := remaining_after_kiley - joe_takes
  in crayons_left = 18 :=
by
  sorry

end crayons_left_l554_554219


namespace value_of_a10_l554_554040

def seq (a : ℕ → ℕ) := a(1) = 2 ∧ ∀ n, a(n+1) = 2 * a(n) 

theorem value_of_a10 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, S (n + 1) = 2 * S n - 1) 
  (h3 : ∀ n, S n = 2 * S (n - 1) - 1) : 
  a 10 = 256 := sorry

end value_of_a10_l554_554040


namespace lucky_numbers_count_l554_554673

def is_lucky (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧ (221 % n) % (221 / n) = 0

def count_lucky_numbers : ℕ :=
  (Finset.range 222).filter is_lucky |>.card

theorem lucky_numbers_count : count_lucky_numbers = 115 := 
by
  sorry

end lucky_numbers_count_l554_554673


namespace balloons_problem_l554_554880

variable (b_J b_S b_J_f b_g : ℕ)

theorem balloons_problem
  (h1 : b_J = 9)
  (h2 : b_S = 5)
  (h3 : b_J_f = 12)
  (h4 : b_g = (b_J + b_S) - b_J_f)
  : b_g = 2 :=
by {
  sorry
}

end balloons_problem_l554_554880


namespace polynomial_fibonacci_property_l554_554806

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem polynomial_fibonacci_property (p : ℕ → ℕ) (h : ∀ k ∈ finset.range 991, p (k + 992) = fibonacci (k + 992)) :
  p 1983 = fibonacci 1983 - 1 :=
sorry

end polynomial_fibonacci_property_l554_554806


namespace polygons_sides_inequality_l554_554551

theorem polygons_sides_inequality (n p q : ℕ) (hpq : p + q ≥ 0) (hshared : ∀ P Q : Type, P ⊂ (n + 4) ∧ Q ⊂ (n + 4) → P ≠ Q ∨ P ∩ Q ≠ ∅):
  p + q ≤ n + 4 :=
  sorry

end polygons_sides_inequality_l554_554551


namespace timeSpentReading_l554_554110

def totalTime : ℕ := 120
def timeOnPiano : ℕ := 30
def timeWritingMusic : ℕ := 25
def timeUsingExerciser : ℕ := 27

theorem timeSpentReading :
  totalTime - timeOnPiano - timeWritingMusic - timeUsingExerciser = 38 := by
  sorry

end timeSpentReading_l554_554110


namespace round_to_nearest_whole_l554_554557

theorem round_to_nearest_whole (x : ℝ) (hx : x = 12345.49999) : round x = 12345 := by
  -- Proof omitted.
  sorry

end round_to_nearest_whole_l554_554557


namespace number_of_people_going_on_trip_l554_554594

theorem number_of_people_going_on_trip
  (bags_per_person : ℕ)
  (weight_per_bag : ℕ)
  (total_luggage_capacity : ℕ)
  (additional_capacity : ℕ)
  (bags_per_additional_capacity : ℕ)
  (h1 : bags_per_person = 5)
  (h2 : weight_per_bag = 50)
  (h3 : total_luggage_capacity = 6000)
  (h4 : additional_capacity = 90) :
  (total_luggage_capacity + (bags_per_additional_capacity * weight_per_bag)) / (weight_per_bag * bags_per_person) = 42 := 
by
  simp [h1, h2, h3, h4]
  repeat { sorry }

end number_of_people_going_on_trip_l554_554594


namespace lucky_numbers_count_l554_554659

def isLucky (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≠ 0 ∧ 221 = n * k + (221 % n) ∧ (221 % n) % k = 0

def countLuckyNumbers : ℕ :=
  Nat.card (Finset.filter isLucky (Finset.range 222))

theorem lucky_numbers_count : countLuckyNumbers = 115 :=
  sorry

end lucky_numbers_count_l554_554659


namespace valid_tuple_count_l554_554737

def isValidTuple (a : Fin 6 → ℤ) : Prop :=
  ∀ i : Fin 6, a i ≤ i

def sumEqSix (a : Fin 6 → ℤ) : Prop :=
  ∑ i, a i = 6

def num_valid_tuples (N : ℕ) : Prop :=
  N = 2002

theorem valid_tuple_count : ∃ N, (num_valid_tuples N) ∧ ∃ a : (Fin 6 → ℤ), isValidTuple a ∧ sumEqSix a :=
  sorry

end valid_tuple_count_l554_554737


namespace probability_black_ball_l554_554081

variable (P_R P_W P_B : ℝ)
variable h1 : P_R = 0.42
variable h2 : P_W = 0.28
variable h3 : P_R + P_W + P_B = 1

theorem probability_black_ball : P_B = 0.3 :=
by
  -- Proof will be provided here.
  sorry

end probability_black_ball_l554_554081


namespace preston_receives_total_amount_l554_554160

theorem preston_receives_total_amount :
  let price_per_sandwich := 5
  let delivery_fee := 20
  let num_sandwiches := 18
  let tip_percent := 0.10
  let sandwich_cost := num_sandwiches * price_per_sandwich
  let initial_total := sandwich_cost + delivery_fee
  let tip := initial_total * tip_percent
  let final_total := initial_total + tip
  final_total = 121 := 
by
  sorry

end preston_receives_total_amount_l554_554160


namespace simplified_equation_has_solution_l554_554570

theorem simplified_equation_has_solution (n : ℤ) :
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x * y - y * z - z * x = n) →
  (∃ x y : ℤ, x^2 + y^2 - x * y = n) :=
by
  intros h
  exact sorry

end simplified_equation_has_solution_l554_554570


namespace sufficient_not_necessary_condition_l554_554270

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = -1) :=
by
  sorry

end sufficient_not_necessary_condition_l554_554270


namespace domain_of_log_function_l554_554939

-- Define the function
def f (x : ℝ) : ℝ := log (2 * x + 4)

-- State the condition
def valid (x : ℝ) : Prop := 2 * x + 4 > 0

-- State the domain in terms of the condition
def domain : set ℝ := { x | valid x }

-- The theorem to prove
theorem domain_of_log_function : domain = { x | x > -2 } := by
  sorry

end domain_of_log_function_l554_554939


namespace coordinate_transformation_l554_554854

theorem coordinate_transformation:
  ∀ (x y t z : ℝ), 
    y = cos(x)^2 → 
    t = 2 * x ∧ y = (z + 1) / 2 → 
    z = cos(t) :=
by
  intros x y t z h1 h2
  sorry

end coordinate_transformation_l554_554854


namespace parabola_standard_form_l554_554769

theorem parabola_standard_form (a : ℝ) (x y : ℝ) :
  (∀ a : ℝ, (2 * a + 3) * x + y - 4 * a + 2 = 0 → 
  x = 2 ∧ y = -8) → 
  (y^2 = 32 * x ∨ x^2 = - (1/2) * y) :=
by 
  intros h
  sorry

end parabola_standard_form_l554_554769


namespace magnitude_of_linear_combination_l554_554435

variables (m : ℝ)
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-2, m)

noncomputable def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem magnitude_of_linear_combination :
  (parallel vec_a vec_b) →
  (|2 * vec_a.1 + 3 * vec_b.1, 2 * vec_a.2 + 3 * vec_b.2| = 4 * Real.sqrt 5) :=
by
  sorry

end magnitude_of_linear_combination_l554_554435


namespace main_problem_l554_554128

noncomputable def problem := 
  ∃ (A B C T X Y : Type) (triangle_ABC_is_acute_scalene : Prop)
    (circumcircle_omega : Prop)
    (tangents_meet_at_T : Prop)
    (BT_eq_20 : ℝ) (CT_eq_20 : ℝ) (BC_eq_30 : ℝ)
    (projection_of_T_onto_AB : Prop) (projection_of_T_onto_AC : Prop)
    (TX_squared_plus_TY_squared_plus_XY_squared_eq_2193 : ℝ), 
  BT_eq_20 = 20 ∧ CT_eq_20 = 20 ∧ BC_eq_30 = 30 ∧ TX_squared_plus_TY_squared_plus_XY_squared_eq_2193 = 2193
   ∧ (TX_squared_plus_TY_squared_eq_1302 : TX^2 + TY^2 = 1302)
   
theorem main_problem : problem := 
by sorry -- this skips the proof, indicating it is not provided here

end main_problem_l554_554128


namespace adoption_event_l554_554718

theorem adoption_event (c : ℕ) 
  (h1 : ∀ d : ℕ, d = 8) 
  (h2 : ∀ fees_dog : ℕ, fees_dog = 15) 
  (h3 : ∀ fees_cat : ℕ, fees_cat = 13)
  (h4 : ∀ donation : ℕ, donation = 53)
  (h5 : fees_dog * 8 + fees_cat * c = 159) :
  c = 3 :=
by 
  sorry

end adoption_event_l554_554718


namespace points_on_same_side_of_line_l554_554872

theorem points_on_same_side_of_line (a : ℝ) 
  (hA : ∃ x y : ℝ, 5 * a^2 - 4 * a * y + 8 * x^2 - 4 * x * y + y^2 + 12 * a * x = 0 ∧ x = -a / 2 ∧ y = a) 
  (hB : ∃ x y : ℝ, a * x^2 - 2 * a^2 * x - a * y + a^3 + 3 = 0 ∧ x = a ∧ y = 3 / a) :
  a ∈ set.Ioo (-5 / 2) (-1 / 2) ∪ set.Ioo 0 3 := 
sorry

end points_on_same_side_of_line_l554_554872


namespace product_properties_l554_554965

-- Define the sequence of odd negative integers strictly greater than -2015
def odd_negative_integers_greater_than_neg_2015 : List Int :=
  List.filter (λ n => n % 2 ≠ 0 ∧ n < 0) (List.range' (-2014) 2014)

-- Define the product of these integers
def product_of_integers (l : List Int) : Int :=
  l.foldl (· * ·) 1

-- Prove that the product is a negative number ending with 5
theorem product_properties :
  let p := product_of_integers odd_negative_integers_greater_than_neg_2015
  p < 0 ∧ (p % 10 = 5) :=
by
  let p := product_of_integers odd_negative_integers_greater_than_neg_2015
  have : p < 0 := sorry -- Proof that the product is negative
  have : p % 10 = 5 := sorry -- Proof that the units digit of the product is 5
  exact ⟨this, this⟩


end product_properties_l554_554965


namespace lifespan_histogram_l554_554249

theorem lifespan_histogram :
  (class_interval = 20) →
  (height_vertical_axis_60_80 = 0.03) →
  (total_people = 1000) →
  (number_of_people_60_80 = 600) :=
by
  intro class_interval height_vertical_axis_60_80 total_people
  -- Perform necessary calculations (omitting actual proof as per instructions)
  sorry

end lifespan_histogram_l554_554249


namespace committee_selection_ways_l554_554090

theorem committee_selection_ways : fintype.card {s : finset (fin 12) // s.card = 5} = 792 :=
by sorry

end committee_selection_ways_l554_554090


namespace find_birth_rate_l554_554476

noncomputable def average_birth_rate (B : ℕ) : Prop :=
  let death_rate := 3
  let net_increase_per_2_seconds := B - death_rate
  let seconds_per_hour := 3600
  let hours_per_day := 24
  let seconds_per_day := seconds_per_hour * hours_per_day
  let net_increase_times := seconds_per_day / 2
  let total_net_increase := net_increase_times * net_increase_per_2_seconds
  total_net_increase = 172800

theorem find_birth_rate (B : ℕ) (h : average_birth_rate B) : B = 7 :=
  sorry

end find_birth_rate_l554_554476


namespace countLuckyNumbers_l554_554669

def isLuckyNumber (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 221 ∧
  (let q := 221 / n in (221 % n) % q = 0)

theorem countLuckyNumbers : 
  { n : ℕ | isLuckyNumber n }.toFinset.card = 115 :=
by
  sorry

end countLuckyNumbers_l554_554669


namespace correct_number_of_propositions_l554_554421

-- Definitions for the given propositions
def prop1 := ∀ (planes : Type) (P Q : planes) (line : Type) (l : line), 
  (P.parallel_to l ∧ Q.parallel_to l) → (P.parallel_to Q)

def prop2 := ∀ (lines : Type) (l m : lines) (plane : Type) (P : plane),
  (l.parallel_to P ∧ m.parallel_to P) → (l.parallel_to m)

def prop3 := ∀ (lines : Type) (l m : lines) (line : Type) (n : line),
  (l.perpendicular_to n ∧ m.perpendicular_to n) → (l.parallel_to m)

def prop4 := ∀ (lines : Type) (l m : lines) (plane : Type) (P : plane),
  (l.perpendicular_to P ∧ m.perpendicular_to P) → (l.parallel_to m)

noncomputable def number_of_true_propositions : Nat :=
  if prop1 then 1 else 0 + if prop2 then 1 else 0 + if prop3 then 1 else 0 + if prop4 then 1 else 0

theorem correct_number_of_propositions : number_of_true_propositions = 1 :=
  sorry

end correct_number_of_propositions_l554_554421


namespace a_n_formula_b_n_formula_T_n_formula_l554_554805

section

variable (S : ℕ → ℕ) (a b : ℕ → ℕ)

-- Conditions
def S_n (n : ℕ) := 3 * n ^ 2 + 8 * n
def a_n (n : ℕ) := if n = 1 then 11 else S (n) - S (n - 1)
def b_n (n : ℕ) := 3 * n + 1
def c_n (n : ℕ) := ((a n) + 1) ^ (n + 1) / ((b n) + 2) ^ n

-- Given
axiom H1 : ∀ n, S n = S_n n
axiom H2 : ∀ n, a n = a_n n
axiom H3 : ∀ n, b n = b_n n

-- Results to prove
theorem a_n_formula : ∀ (n : ℕ), n ≥ 2 → a n = 6 * n + 5 := by sorry
theorem b_n_formula : ∀ (n : ℕ), b n = 3 * n + 1 := by sorry
theorem T_n_formula : ∀ (n : ℕ), (finset.range n).sum (λ i, c_n i) = 3 * n * 2^(n+2) := by sorry

end

end a_n_formula_b_n_formula_T_n_formula_l554_554805


namespace minimum_value_expression_l554_554381

theorem minimum_value_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := 
by
  sorry

end minimum_value_expression_l554_554381


namespace expression_evaluates_to_zero_l554_554016

open Real

noncomputable def equivalent_expression (a b : ℝ) : ℝ :=
  (sqrt (log b (a^4) + log a (b^4) + 2) + 2)^(1 / 2) - log b a - log a b

theorem expression_evaluates_to_zero (a b : ℝ) (ha : 0 < a) (ha_ne_one : a ≠ 1) (hb : 0 < b) (hb_ne_one : b ≠ 1) :
  equivalent_expression a b = 0 :=
by
  sorry

end expression_evaluates_to_zero_l554_554016


namespace lcm_18_35_is_630_l554_554006

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l554_554006


namespace find_x_in_line_segment_l554_554645

theorem find_x_in_line_segment (x : ℝ) (h : x > 0) :
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (x, 6)
  let d : ℝ := 7
  (real.sqrt ((x - 2)^2 + (6 - 2)^2)) = d → x = 2 + real.sqrt 33 :=
by
  sorry

end find_x_in_line_segment_l554_554645


namespace max_num1_max_num2_max_num3_max_num4_l554_554967

open Nat

theorem max_num1 : ∃ (n : ℕ), (n < 8) ∧ (50 * n < 360) :=
by
  use 7
  split
  · exact Nat.lt_succ_self 7
  · exact lt_of_le_of_lt (Nat.mul_le_mul_left 50 (Nat.le_refl 7)) (Nat.succ_pos 350)

theorem max_num2 : ∃ (n : ℕ), (n < 5) ∧ (80 * n < 352) :=
by
  use 4
  split
  · exact Nat.lt_succ_self 4
  · exact lt_of_le_of_lt (Nat.mul_le_mul_left 80 (Nat.le_refl 4)) (Nat.succ_pos 352)

theorem max_num3 : ∃ (n : ℕ), (n < 7) ∧ (70 * n < 424) :=
by
  use 6
  split
  · exact Nat.lt_succ_self 6
  · exact lt_of_le_of_lt (Nat.mul_le_mul_left 70 (Nat.le_refl 6)) (Nat.succ_pos 420)

theorem max_num4 : ∃ (n : ℕ), (n < 5) ∧ (60 * n < 245) :=
by
  use 4
  split
  · exact Nat.lt_succ_self 4
  · exact lt_of_le_of_lt (Nat.mul_le_mul_left 60 (Nat.le_refl 4)) (Nat.succ_pos 240)

end max_num1_max_num2_max_num3_max_num4_l554_554967


namespace find_x_minus_y_l554_554029

theorem find_x_minus_y (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x * y < 0) : x - y = 6 ∨ x - y = -6 :=
by sorry

end find_x_minus_y_l554_554029


namespace positive_difference_between_prob_3_and_prob_5_l554_554612

/-- Probability of a coin landing heads up exactly 3 times out of 5 flips -/
def prob_3_heads : ℚ := (nat.choose 5 3) * (1/2)^3 * (1/2)^(5-3)

/-- Probability of a coin landing heads up exactly 5 times out of 5 flips -/
def prob_5_heads : ℚ := (1/2)^5

/-- Positive difference between the probabilities -/
theorem positive_difference_between_prob_3_and_prob_5 : 
  |prob_3_heads - prob_5_heads| = 9 / 32 :=
by sorry

end positive_difference_between_prob_3_and_prob_5_l554_554612


namespace invalid_votes_percentage_l554_554088

variable (total_votes valid_votes invalid_votes : ℕ)
variable (candidate1_votes candidate2_votes : ℕ)

-- Conditions given in the problem
axiom h1 : total_votes = 7500
axiom h2 : candidate2_votes = 2700
axiom h3 : candidate1_votes = 0.55 * valid_votes
axiom h4 : valid_votes = candidate1_votes + candidate2_votes
axiom h5 : invalid_votes = total_votes - valid_votes

theorem invalid_votes_percentage (h1 h2 h3 h4 h5 : Prop) : 
  let invalid_votes_percentage := (invalid_votes.toFloat / total_votes.toFloat) * 100 in
  invalid_votes_percentage = 20 := 
sorry

end invalid_votes_percentage_l554_554088


namespace simplify_sqrt_24_l554_554559

theorem simplify_sqrt_24 : Real.sqrt 24 = 2 * Real.sqrt 6 :=
sorry

end simplify_sqrt_24_l554_554559


namespace eccentricity_of_ellipse_l554_554330

namespace geometry

-- Definition of constants and given conditions
variables {F1 F2 : ℝ} (c : ℝ) (a : ℝ) (d : ℝ)
variables (h1 : dist F1 F1 = 0) -- ECC wrong
variables (h2 : dist F2 F2 = 0) -- ECC wrong
variables (h3 : dist F1 F2 = 2 * c) -- ECC wrong
variables (h4 : dist d F1 = c)
variables (h5 : dist d F2 = real.sqrt 3 * c)
variables (h6 : 2 * a = real.sqrt 3 * c + c)

-- Proof problem to solve
theorem eccentricity_of_ellipse :
  ∃ e : ℝ, e = (c / a) ∧ e = real.sqrt 3 - 1 :=
sorry

end geometry

end eccentricity_of_ellipse_l554_554330


namespace log_base_one_fifth_of_25_l554_554745

theorem log_base_one_fifth_of_25 : log (1/5) 25 = -2 := by
  sorry

end log_base_one_fifth_of_25_l554_554745


namespace circles_intersect_l554_554437

variable (a b : ℝ)

def circle_O1 (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 4
def circle_O2 (x y : ℝ) : Prop := (x - a - 1)^2 + (y - b - 2)^2 = 1

theorem circles_intersect :
  let center_distance := sqrt ((1:ℝ)^2 + (2:ℝ)^2),
      radii_sum := 2 + 1,
      radii_diff := 2 - 1 in
  1 < center_distance ∧ center_distance < radii_sum → 
  (∃ x y : ℝ, circle_O1 a b ∧ circle_O2 (a + 1) (b + 2)) :=
by
  sorry

end circles_intersect_l554_554437


namespace minimum_value_of_f_l554_554008

open Real

def f (x : ℝ) : ℝ := x + 2/x + 1/(x + 2/x)

theorem minimum_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ (∀ (y : ℝ), y > 0 → f(y) ≥ f(√2)) ∧ f(√2) = (5 * sqrt 2) / 2 :=
by sorry

end minimum_value_of_f_l554_554008


namespace lcm_18_35_l554_554001

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l554_554001


namespace are_propositions_correct_l554_554510

noncomputable def are_correct_propositions (p1 p2 p3 p4 : Prop) : Prop :=
  p2 ∧ p3 ∧ p4 ∧ ¬p1

def proposition_1 (α β γ : Type) [HasPerpendicular α γ] [HasPerpendicular β γ] : Prop :=
  ∀ (α β γ : set ℝ^3), α ⊥ γ → β ⊥ γ → α ⟂ β

def proposition_2 (α β : Type) (a b : Type) [IsSkew a b] [LineContainedInPlane a α]
  [LineContainedInPlane b β] [IsParallel a β] [IsParallel b α] : Prop :=
  ∀ (α β : set ℝ^3) (a b : set ℝ^3), skew_lines a b ∧ a ⊆ α ∧ b ⊆ β
  ∧ a ∥ β ∧ b ∥ α → α ∥ β

def proposition_3 (α β γ a b c : Type) [IsParallel a b] : Prop :=
  ∀ (α β γ : set ℝ^3) (a b c : set ℝ^3), (α ∩ β = a) ∧ (∅ ∩ γ = b)
  ∧ (γ ∩ a = c) ∧ (a ∥ b) → c ∥ β

def proposition_4 (a b c α : Type) [IsSkew a b] [IsParallel a α] [IsParallel b α]
  [IsPerpendicular c a] [IsPerpendicular c b] : Prop :=
  ∀ (a b c α : set ℝ^3), skew_lines a b ∧ a ∥ α ∧ b ∥ α ∧ c ⊥ a ∧ c ⊥ b → c ⊥ α

theorem are_propositions_correct (α β γ a b c : Type) [HasPerpendicular α γ] [HasPerpendicular β γ]
  [LineContainedInPlane a α] [LineContainedInPlane b β]
  [IsSkew a b] [IsParallel a β] [IsParallel b α]
  [IsParallel a b] [IsPerpendicular c a] [IsPerpendicular c b] :
  are_correct_propositions (proposition_1 α β γ)
                          (proposition_2 α β a b)
                          (proposition_3 α β γ a b c)
                          (proposition_4 a b c α) :=
by
  sorry

end are_propositions_correct_l554_554510


namespace product_of_factors_l554_554344

theorem product_of_factors : 
  (\left(1 - \frac{1}{3^2}\right) * \left(1 - \frac{1}{4^2}\right) * \cdots * \left(1 - \frac{1}{12^2}\right)) = \frac{13}{18} := 
sorry

end product_of_factors_l554_554344


namespace combined_age_l554_554542

variable (m y o : ℕ)

noncomputable def younger_brother_age := 5

noncomputable def older_brother_age_based_on_younger := 3 * younger_brother_age

noncomputable def older_brother_age_based_on_michael (m : ℕ) := 1 + 2 * (m - 1)

theorem combined_age (m y o : ℕ) (h1 : y = younger_brother_age) (h2 : o = older_brother_age_based_on_younger) (h3 : o = older_brother_age_based_on_michael m) :
  y + o + m = 28 := by
  sorry

end combined_age_l554_554542


namespace ratio_of_inscribed_squares_l554_554312

theorem ratio_of_inscribed_squares (x y : ℝ) (h1 : ∃ (a b c : ℝ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2)
(h2 : ∃ (a b : ℝ), a = x ∧ b = 6 ∧ c = 10 ∧ 8 - x = (4 / 3) * x ∧ a = 24 / 7)
(h3 : ∃ (a b : ℝ), a = y ∧ b = 10 ∧ (6 / 5) * y + (8 / 5) * y = 10 * (14 / 5) ∧ y = 200 / 37):
\frac{x}{y} = 111 / 175 :=
sorry

end ratio_of_inscribed_squares_l554_554312


namespace matrix_solution_exists_l554_554360

theorem matrix_solution_exists :
  ∃ u v : ℝ, 
    (3 + 4 * u = -3 * v) ∧ 
    (1 - 6 * u = 2 + 4 * v) ∧ 
    (u = 9 / 2) ∧ 
    (v = -7) 
:= by
  use (9 / 2), (-7)
  split
  { rw [← add_eq_of_eq_sub, ← neg_div, neg_neg, mul_div], norm_num, intro h, norm_num at h, rw h }
  split
  { rw [← add_eq_of_eq_sub, ← neg_div, neg_neg, mul_div], norm_num, intro h, norm_num at h, rw h }
  split
  { norm_num }
  { norm_num }

end matrix_solution_exists_l554_554360


namespace isosceles_triangle_count_l554_554827

theorem isosceles_triangle_count : 
  (let valid_triples := { (a, b) | (2 * a + b = 24) ∧ ((b > 0) ∧ (4 * a > 24))} in 
  #[(a, b) ∈ valid_triples | b = 24 - 2 * a ∧ a > 6 ∧ a ∈ (Int.range 12)] = 5) :=
sorry

end isosceles_triangle_count_l554_554827


namespace log_eq_neg_two_l554_554749

theorem log_eq_neg_two : ∀ (x : ℝ), (1 / 5) ^ x = 25 → x = -2 :=
by
  intros x h
  sorry

end log_eq_neg_two_l554_554749


namespace gas_isothermal_work_l554_554157

noncomputable def gas_ideal_isothermal_work (one_mole : ℤ) (work_isobaric : ℝ) (heat_same : ℝ) : ℝ :=
  let C_p := (5 / 2 : ℝ) * real.R
  let ΔT := work_isobaric / real.R
  let Q_isobaric := (5 / 2 : ℝ) * real.R * ΔT
  have h_Q_isobaric : heat_same = Q_isobaric := by
    sorry
  heat_same

theorem gas_isothermal_work
  (one_mole : ℤ) 
  (work_isobaric : ℝ)
  (h_one_mole : one_mole = 1)
  (h_work_isobaric : work_isobaric = 40)
  (heat_same : ℝ)
  (h_heat_same : heat_same = (5 / 2 : ℝ) * real.R * (40 / real.R)) :
  gas_ideal_isothermal_work one_mole work_isobaric heat_same = 100 := by
  sorry

end gas_isothermal_work_l554_554157


namespace periodicity_expression_2_to_4_sum_f_1_to_2013_l554_554133

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f x = - f (-x) := sorry

lemma periodic_function : ∀ x : ℝ, f (x + 1) = f (1 - x) := sorry

lemma given_expression (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : f x = 2 * x - x^2 := sorry

theorem periodicity : ∀ x : ℝ, f (x + 4) = f x := sorry

theorem expression_2_to_4 (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4) : f x = 2 * x - x^2 := sorry

theorem sum_f_1_to_2013 : (List.range 2013).sum (fun n => f (n + 1)) = 1 :=
by
  have h1 : ∀ n, f (n + 1) = f ((n % 4) + 1) := sorry
  have h2 : List.range 2013 = List.range 503 * 4 ++ [0,1,2]++[3] := sorry 
  have h3 : (List.range 0 4).sum (fun n => f (n + 1)) = 0 := sorry
  have h4 : List.range 3.sum (fun n => f (n + 1)) + List.range 503.sum f + 0 := by assumption 
  
  sorry

end periodicity_expression_2_to_4_sum_f_1_to_2013_l554_554133
