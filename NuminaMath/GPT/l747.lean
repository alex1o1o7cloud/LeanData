import Mathlib

namespace probability_of_spinner_greater_than_10_l747_747263

def sections : List ℕ := [4, 6, 7, 11, 12, 13, 17, 18]

def is_greater_than_10 (n : ℕ) : Prop := n > 10

noncomputable def probability_greater_than_10 (s : List ℕ) : ℚ :=
  let favorable := s.filter is_greater_than_10
  (favorable.length : ℚ) / (s.length : ℚ)

theorem probability_of_spinner_greater_than_10 :
  probability_greater_than_10 sections = 5 / 8 :=
sorry

end probability_of_spinner_greater_than_10_l747_747263


namespace jane_exercises_40_hours_l747_747051

-- Define the conditions
def hours_per_day : ℝ := 1
def days_per_week : ℝ := 5
def weeks : ℝ := 8

-- Define total_hours using the conditions
def total_hours : ℝ := (hours_per_day * days_per_week) * weeks

-- The theorem stating the result
theorem jane_exercises_40_hours :
  total_hours = 40 := by
  sorry

end jane_exercises_40_hours_l747_747051


namespace group_photo_arrangement_l747_747161

-- Define the number of teachers, female students, and male students
def num_teachers : Nat := 1
def num_female_students : Nat := 2
def num_male_students : Nat := 2

-- Define the condition for the arrangement: Female students separated by the teacher.
def condition (arrangement : List String) : Prop :=
  let teacher := "T"
  let female_students := ["F1", "F2"]
  ∃ i j, arrangement.nth i = some teacher ∧ arrangement.nth j = some teacher ∧
        i + 2 = j ∧ arrangement.nth (i-1) ∈ female_students ∧ arrangement.nth (i+1) ∈ female_students

-- Define the goal: Total number of different arrangements
def total_arrangements : Nat := 12

-- The theorem to be proved
theorem group_photo_arrangement : 
  ∃ (arrangements : List (List String)), 
    (∀ a ∈ arrangements, condition a) ∧ arrangements.length = total_arrangements :=
sorry

end group_photo_arrangement_l747_747161


namespace average_sales_per_month_l747_747290

theorem average_sales_per_month :
  let sales := [120, 80, 50, 110, 90, 160] in
  let total_sales := list.sum sales in
  let number_of_months := 6 in
  (total_sales / number_of_months : ℤ) = 102 :=
by 
  let sales := [120, 80, 50, 110, 90, 160]
  have total_sales : ℤ := list.sum sales
  have number_of_months : ℤ := 6
  have avg_sales := total_sales / number_of_months
  show avg_sales = 102 from sorry

end average_sales_per_month_l747_747290


namespace plan_y_more_cost_effective_l747_747273

theorem plan_y_more_cost_effective (m : Nat) : 2500 + 7 * m < 15 * m → 313 ≤ m :=
by
  intro h
  sorry

end plan_y_more_cost_effective_l747_747273


namespace unique_solution_xy_l747_747593

theorem unique_solution_xy (x y : ℝ) : 
  (sqrt (x^2 + y^2) + sqrt ((x - 4)^2 + (y - 3)^2) = 5) ∧                                 
  (3 * x^2 + 4 * x * y = 24) → 
  (x = 2 ∧ y = 1.5) :=
by
  sorry

end unique_solution_xy_l747_747593


namespace minimum_common_area_of_triangles_l747_747125

-- Define the problem of finding the minimum area of the common part of the triangles KLM and A1B1C1.
theorem minimum_common_area_of_triangles
  (ABC : Type)
  [is_triangle ABC]
  (A B C A1 B1 C1 K L M : Point ABC)
  (h_area_ABC : area ABC = 1)
  (h_A1 : midpoint A1 B C)
  (h_B1 : midpoint B1 C A)
  (h_C1 : midpoint C1 A B)
  (h_K : on_segment K A B1)
  (h_L : on_segment L C A1)
  (h_M : on_segment M C1 C) :
  ∃(common_area : ℝ), common_area = 1 / 8 :=
begin
  -- Proof to be provided
  sorry
end

end minimum_common_area_of_triangles_l747_747125


namespace find_phi_l747_747996

open Real

-- Define the original function and translation conditions
def f (x ϕ : ℝ) : ℝ := sin (2 * x + ϕ)
def g (x ϕ : ℝ) : ℝ := sin (2 * x - (π / 3) + ϕ)

-- Statement to prove the symmetric condition leads to ϕ = 5π/6
theorem find_phi (ϕ : ℝ) (h₁ : 0 ≤ ϕ) (h₂ : ϕ ≤ π) (h₃ : ∀ x, g (x + π / 6) ϕ = g (-x - π / 6) ϕ) : ϕ = 5 * π / 6 :=
  sorry

end find_phi_l747_747996


namespace dice_sum_probability_l747_747450

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747450


namespace instantaneous_velocity_at_0_l747_747884

def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

theorem instantaneous_velocity_at_0.5 :
  (deriv h 0.5) = 1.6 :=
by
  -- The proof would go here.
  sorry

end instantaneous_velocity_at_0_l747_747884


namespace initial_pipes_num_l747_747594

variable {n : ℕ}

theorem initial_pipes_num (h1 : ∀ t : ℕ, (n * t = 8) → n = 3) (h2 : ∀ t : ℕ, (2 * t = 12) → n = 3) : n = 3 := 
by 
  sorry

end initial_pipes_num_l747_747594


namespace problem_3000_mod_1001_l747_747100

theorem problem_3000_mod_1001 : (300 ^ 3000 - 1) % 1001 = 0 := 
by
  have h1: (300 ^ 3000) % 7 = 1 := sorry
  have h2: (300 ^ 3000) % 11 = 1 := sorry
  have h3: (300 ^ 3000) % 13 = 1 := sorry
  sorry

end problem_3000_mod_1001_l747_747100


namespace total_skateboarded_distance_l747_747054

/-- Define relevant conditions as variables for input values in Lean -/
variables
  (distance_skateboard_to_park : ℕ) -- Distance skateboarded to the park
  (distance_walk_to_park : ℕ) -- Distance walked to the park
  (distance_skateboard_park_to_home : ℕ) -- Distance skateboarded from the park to home

/-- Constraint conditions as hypotheses -/
variables
  (H1 : distance_skateboard_to_park = 10)
  (H2 : distance_walk_to_park = 4)
  (H3 : distance_skateboard_park_to_home = distance_skateboard_to_park + distance_walk_to_park)

/-- The theorem we intend to prove -/
theorem total_skateboarded_distance : 
  distance_skateboard_to_park + distance_skateboard_park_to_home - distance_walk_to_park = 24 := 
by 
  rw [H1, H2, H3] 
  sorry

end total_skateboarded_distance_l747_747054


namespace number_of_females_l747_747262

def population : ℕ := 500
def male_to_female_ratio : ℕ × ℕ := (3, 2)

theorem number_of_females :
  ∀ (population : ℕ) (ratio : ℕ × ℕ), 
  ratio = (3, 2) → 
  population = 500 → 
  ∃ (females : ℕ), females = 200 :=
by
  intros population ratio h_ratio h_population
  use 200
  sorry

end number_of_females_l747_747262


namespace RouteY_not_faster_than_RouteX_l747_747088

-- Define the given problem conditions
def distance_X := 8 -- miles
def speed_X := 40 -- miles per hour
def distance_Y := 6 -- total miles
def distance_construction := 1 -- miles
def speed_Y_normal := 50 -- miles per hour
def speed_Y_construction := 10 -- miles per hour

-- Calculate the time for Route X
def time_X := (distance_X / speed_X) * 60 -- in minutes

-- Calculate the time for Route Y
def time_Y :=
  ((distance_Y - distance_construction) / speed_Y_normal) * 60 + 
  (distance_construction / speed_Y_construction) * 60 -- in minutes

-- Proof that the travel times are equal (0 minutes difference)
theorem RouteY_not_faster_than_RouteX : (time_X - time_Y = 0) :=
by
  have tX_eq : time_X = 12 := by sorry
  have tY_eq : time_Y = 12 := by sorry
  rw [tX_eq, tY_eq]
  exact eq.refl 0

end RouteY_not_faster_than_RouteX_l747_747088


namespace largest_integer_satisfying_inequality_l747_747667

theorem largest_integer_satisfying_inequality :
  ∃ (x : ℤ), (∃ (H : x ∈ {n : ℤ | (n / 5 : ℚ) + 6/7 < 8/5}), ∀ y : ℤ, y > x → ¬(y / 5 : ℚ) + 6/7 < 8/5) ∧ x = 3 :=
by
  -- Variables and conditions
  let P := λ (x : ℤ), (x / 5 : ℚ) + 6/7 < 8/5
  existsi 3
  split
  -- Proving that x = 3 satisfies the condition
  use (by norm_num : 3 / 5 + 6/7 < 8/5)
  intros y hy
  -- Show that no larger integer y satisfies the inequality
  have : (3 < y) → ¬P y := by
    intro h
    linarith [h]
  exact this hy

end largest_integer_satisfying_inequality_l747_747667


namespace pat_kate_ratio_l747_747094

theorem pat_kate_ratio 
  (P K M : ℕ)
  (h1 : P + K + M = 117)
  (h2 : ∃ r : ℕ, P = r * K)
  (h3 : P = M / 3)
  (h4 : M = K + 65) : 
  P / K = 2 :=
by
  sorry

end pat_kate_ratio_l747_747094


namespace angle_A_double_angle_B_iff_l747_747905

variables (A B C M D : Type)
variables [is_midpoint M A B] [is_foot D C A B]

theorem angle_A_double_angle_B_iff :
  ∠ A = 2 * ∠ B ↔ dist A C = 2 * dist M D :=
sorry

end angle_A_double_angle_B_iff_l747_747905


namespace card_prob_product_is_multiple_of_six_l747_747794

open Finset

/-- From 4 cards with numbers 1, 2, 3, 4, two cards are drawn without replacement.
    The probability that the product of the numbers on the two cards is a multiple of 6 is 1/3. -/
theorem card_prob_product_is_multiple_of_six :
  let cards := {1, 2, 3, 4}
  let pairs := (cards.product cards).filter (λ p, p.1 < p.2)
  let favorable := pairs.filter (λ p, (p.1 * p.2) % 6 = 0)
  (favorable.card : ℚ) / pairs.card = 1 / 3 :=
by
  sorry

end card_prob_product_is_multiple_of_six_l747_747794


namespace probability_heads_equals_l747_747212

theorem probability_heads_equals (p q: ℚ) (h1 : q = 1 - p) (h2 : (binomial 10 5) * p^5 * q^5 = (binomial 10 6) * p^6 * q^4) : p = 6 / 11 :=
by {
  sorry
}

end probability_heads_equals_l747_747212


namespace probability_sum_is_10_l747_747454

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747454


namespace probability_sum_10_l747_747482

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747482


namespace central_angle_of_cone_sector_l747_747382

def base_radius : ℝ := 3
def slant_height : ℝ := 12
def base_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem central_angle_of_cone_sector
  (r : ℝ := base_radius) 
  (s : ℝ := slant_height) 
  (L : ℝ := base_circumference r) 
  : (L / s) * (180 / Real.pi) = 90 :=
by
  sorry

end central_angle_of_cone_sector_l747_747382


namespace concurrency_of_midpoints_l747_747819

variables {A B C D E F P Q R O : Type}
variables [AffType A] [AffType B] [AffType C] [AffType D] [AffType E] [AffType F] [AffType P] [AffType Q] [AffType R]

open Affine

-- Assume there is a quadrilateral ABCD
variable {A B C D : AffType}

-- Assume E and F are points on sides BC and CD respectively
variable {E : BC.contains E}
variable {F : CD.contains F}

-- Define P, Q, and R as the midpoints of segments AE, EF, and AF respectively
variable {P : midpoint A E}
variable {Q : midpoint E F}
variable {R : midpoint A F}

-- Definition says BP, CQ, and DR are concurrent
theorem concurrency_of_midpoints :
  Concurrent BP CQ DR := sorry

end concurrency_of_midpoints_l747_747819


namespace probability_sum_10_l747_747487

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747487


namespace quadratic_inequality_solution_l747_747347

-- Definition of the given conditions and the theorem to prove
theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : ∀ x, ax^2 + bx + c < 0 ↔ x < -2 ∨ x > -1/2) :
  ∀ x, ax^2 - bx + c > 0 ↔ 1/2 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l747_747347


namespace pipe_B_empty_time_l747_747096

def rateA := 1 / 6
variable (t : ℝ)
def rateB := 1 / t
def net_rate := rateA - rateB

theorem pipe_B_empty_time :
  (net_rate * 96 + rateA * (-66) = 1) → t = 24 :=
by
  intro h
  sorry

end pipe_B_empty_time_l747_747096


namespace sum_of_factors_of_30_l747_747673

open Nat

theorem sum_of_factors_of_30 : 
  ∑ d in (Finset.filter (λ n, 30 % n = 0) (Finset.range (30 + 1))), d = 72 := by
  sorry

end sum_of_factors_of_30_l747_747673


namespace length_of_side_b_max_area_of_triangle_l747_747903

variable {A B C a b c : ℝ}
variable {triangle_ABC : a + c = 6}
variable {eq1 : (3 - Real.cos A) * Real.sin B = Real.sin A * (1 + Real.cos B)}

-- Theorem for part (1) length of side b
theorem length_of_side_b :
  b = 2 :=
sorry

-- Theorem for part (2) maximum area of the triangle
theorem max_area_of_triangle :
  ∃ (S : ℝ), S = 2 * Real.sqrt 2 :=
sorry

end length_of_side_b_max_area_of_triangle_l747_747903


namespace egg_difference_l747_747577

theorem egg_difference
    (total_eggs : ℕ := 24)
    (broken_eggs : ℕ := 3)
    (cracked_eggs : ℕ := 6)
    (perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs)
    : (perfect_eggs - cracked_eggs = 9) :=
begin
  sorry
end

end egg_difference_l747_747577


namespace second_random_unit_is_068_l747_747983

def random_number_table := [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76]

def is_valid_unit (n : ℕ) : Prop := n ≤ 200

def extract_valid_number (lst : List ℕ) : List ℕ :=
  lst.filter is_valid_unit

def extract_random_units : List ℕ :=
  extract_valid_number (random_number_table.drop 4)

theorem second_random_unit_is_068 :
  extract_random_units.nth 1 = some 68 :=
by
  sorry

end second_random_unit_is_068_l747_747983


namespace denote_loss_of_300_dollars_l747_747868

-- Define the concept of financial transactions
def denote_gain (amount : Int) : Int := amount
def denote_loss (amount : Int) : Int := -amount

-- The condition given in the problem
def earn_500_dollars_is_500 := denote_gain 500 = 500

-- The assertion we need to prove
theorem denote_loss_of_300_dollars : denote_loss 300 = -300 := 
by 
  sorry

end denote_loss_of_300_dollars_l747_747868


namespace find_x_l747_747759

theorem find_x (x : ℤ) : 3^7 * 3^x = 27 ↔ x = -4 := by
  sorry

end find_x_l747_747759


namespace maximum_real_part_of_sum_of_w_l747_747949

theorem maximum_real_part_of_sum_of_w (z : ℂ → Prop) :
  (∀ k : ℕ, k < 12 → z (3 * exp (2 * pi * I * k / 12))) →
  ∀ w : (ℂ → ℂ), (∀ j : ℕ, j < 12 →
    (w (3 * exp (2 * pi * I * j / 12)) = 
    (1 / 2 * (3 * exp (2 * pi * I * j / 12))) ∨ 
    w (3 * exp (2 * pi * I * j / 12)) = 
    (3 * exp (2 * pi * I * j / 12)) ∨ 
    w (3 * exp (2 * pi * I * j / 12)) = 
    (2 * (3 * exp (2 * pi * I * j / 12))))) →
  ∑ j in finset.range 12, (w (3 * exp (2 * pi * I * j / 12))).re ≤ 9 + 9 * real.sqrt 3 := 
by
  sorry

end maximum_real_part_of_sum_of_w_l747_747949


namespace count_integer_solutions_l747_747758

theorem count_integer_solutions : 
  {pairs : ℕ × ℕ // (pairs.1 - 1) * (pairs.2 - 1) = 2}.to_finset.card = 4 := 
by
  sorry

end count_integer_solutions_l747_747758


namespace find_circle_equation_and_slope_range_l747_747356

-- Provided conditions
variable (C : ℝ × ℝ → Prop) -- Circle C
variable (origin : ℝ × ℝ) (centerLine : ℝ → ℝ) (intersectLine : ℝ × ℝ → Prop)
variable [∀ p, decidable (C p)] [∀ p, decidable (intersectLine p)]
variable (A B : ℝ × ℝ)
variable (O : ℝ × ℝ)
variable (M : ℝ × ℝ)

-- Hypotheses: Circle passes through the origin, the center is on y=2x,
--             intersects x+y-3=0 at A, B, and OA ⋅ OB = 0
axiom h1 : C origin
axiom h2 : ∃ (c : ℝ × ℝ), centerLine c.1 = c.2 ∧ 
                             ∃ (p : ℝ), p * c.1 + c.2 = 3 ∧  
                             ∃ (r : ℝ), ∀ (P : ℝ × ℝ), C P ↔ ((P.1 - c.1) ^ 2 + (P.2 - c.2) ^ 2 = r^2)

axiom h3 : ∀ p, intersectLine p ↔ p.1 + p.2 = 3
axiom h4 : C A
axiom h5 : C B
axiom h6 : intersectLine A
axiom h7 : intersectLine B
axiom h8 : O = (0, 0)
axiom h9 : M = (0, 5)
axiom h10 : O ∈ A /\ O ∈ B
axiom h11 : (O.1 - A.1) * (O.1 - B.1) + (O.2 - A.2) * (O.2 - B.2) = 0

-- Questions: Equation of the circle and range of slopes of line MP
theorem find_circle_equation_and_slope_range :
  (∃ (h : ℝ × ℝ), ∀ (P : ℝ × ℝ), C P ↔ (P.1 - h.1)^2 + (P.2 - h.2)^2 = 5) ∧
  (let sl := λ P : ℝ × ℝ, (P.2 - M.2) / (P.1 - M.1) in 
  ∀ P, C P → (sl P ≤ -1/2 ∨ sl P ≥ 2)) :=
by {
  -- Proof replacement
  sorry
}

end find_circle_equation_and_slope_range_l747_747356


namespace probability_sum_is_10_l747_747460

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747460


namespace sum_of_G_that_makes_1234560G_divisible_by_9_l747_747823

theorem sum_of_G_that_makes_1234560G_divisible_by_9 : 
  (∑ G in {G : ℕ | G < 10 ∧ (21 + G) % 9 = 0}, G) = 6 := 
sorry

end sum_of_G_that_makes_1234560G_divisible_by_9_l747_747823


namespace correct_statements_count_l747_747792

theorem correct_statements_count (a b m : ℝ) : 
  let s1 := (am^2 > bm^2 → a > b),
      s2 := (a > b → a * |a| > b * |b|),
      s3 := (b > a ∧ a > 0 ∧ m > 0 → (a + m) / (b + m) > a / b),
      s4 := (a > b ∧ b > 0 ∧ |ln a| = |ln b| → (2 * a + b) ∈ Ioi 3) in
  (count (λ x, x = true) [s1, s2, s3, s4]) = 4 :=
by
  sorry

end correct_statements_count_l747_747792


namespace total_road_signs_l747_747730

-- Definitions for the number of signs at each intersection
def intersection1 : ℕ := 60
def intersection2 : ℕ := intersection1 / 3
def intersection3 : ℕ := (intersection1 + intersection2) / 2
def intersection4 : ℕ := 2 * intersection2 - 5
def intersection5 : ℕ := intersection1 * intersection4
def intersection6 : ℕ := (intersection2 + intersection3) / 2
def intersection7 : ℕ := intersection3 + 15
def intersection8 : ℕ := ((intersection5 + intersection7) / 2).round

-- Statement to prove the total number of road signs
theorem total_road_signs : 
  60 + intersection1 / 3 + (intersection1 + intersection1 / 3) / 2 + 
  (2 * (intersection1 / 3) - 5) + (intersection1 * (2 * (intersection1 / 3) - 5)) + 
  ((intersection1 / 3 + (intersection1 + intersection1 / 3) / 2) / 2) + 
  ((intersection1 + intersection1 / 3) / 2 + 15) + ((intersection5 + intersection7) / 2).round = 3418 :=
by {
  sorry
}

end total_road_signs_l747_747730


namespace distance_center_to_plane_of_trapezoid_l747_747959

variable (R α : ℝ)
variable (h1 : 0 < α) (h2 : α < real.pi / 2)

theorem distance_center_to_plane_of_trapezoid {R α : ℝ} (h1 : 0 < α) (h2 : α < real.pi / 2) : 
  let sin_part1 := real.sin (3 * α / 2) in
  let sqrt_part := real.sqrt (real.sin ((3 * α / 2) + (real.pi / 6)) * real.sin ((3 * α / 2) - (real.pi / 6))) in
  O1O2 = (R / sin_part1) * sqrt_part :=
sorry

end distance_center_to_plane_of_trapezoid_l747_747959


namespace probability_sum_is_10_l747_747510

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747510


namespace existence_of_xy_l747_747781

theorem existence_of_xy {b : ℝ} (h : b ∈ Iio 0 ∨ b ∈ Icc (3/8 : ℝ) (Top : ℝ)) :
  ∃ (a x y : ℝ), x = |y - b| + 3 / b ∧ x^2 + y^2 + 32 = a * (2 * y - a) + 12 * x := 
sorry

end existence_of_xy_l747_747781


namespace diamonds_in_figure_f5_l747_747752

def num_diamonds : ℕ → ℕ
| 1 := 3
| 2 := 19
| (n + 3) := num_diamonds (n + 2) + 4 * (n + 4 + 1)

theorem diamonds_in_figure_f5 : num_diamonds 5 = 91 := by
  sorry

end diamonds_in_figure_f5_l747_747752


namespace find_x_of_product_eq_72_l747_747341

theorem find_x_of_product_eq_72 (x : ℝ) (h : 0 < x) (hx : x * ⌊x⌋₊ = 72) : x = 9 :=
sorry

end find_x_of_product_eq_72_l747_747341


namespace dihedral_angle_between_planes_l747_747628

noncomputable def normal_vector_plane_alpha : EuclideanSpace ℝ (Fin 3) := ![1,0,-1]
noncomputable def normal_vector_plane_beta : EuclideanSpace ℝ (Fin 3) := ![0,-1,1]

def cosine_of_angle (m n : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  (inner_product_space.inner m n) / (norm m * norm n)

theorem dihedral_angle_between_planes :
  let m := normal_vector_plane_alpha
      n := normal_vector_plane_beta in
  ∃ θ : ℝ, (θ = real.acos (cosine_of_angle m n) ∨ θ = π - real.acos (cosine_of_angle m n))
          → θ = π / 3 ∨ θ = 2 * π / 3 :=
sorry

end dihedral_angle_between_planes_l747_747628


namespace karen_group_size_l747_747195

theorem karen_group_size (total_students : ℕ) (zack_group_size number_of_groups : ℕ) (karen_group_size : ℕ) (h1 : total_students = 70) (h2 : zack_group_size = 14) (h3 : number_of_groups = total_students / zack_group_size) (h4 : number_of_groups = total_students / karen_group_size) : karen_group_size = 14 :=
by
  sorry

end karen_group_size_l747_747195


namespace intersection_of_lines_l747_747334

theorem intersection_of_lines :
  ∃ (x y : ℚ), (6 * x - 5 * y = 15) ∧ (8 * x + 3 * y = 1) ∧ x = 25 / 29 ∧ y = -57 / 29 :=
by
  sorry

end intersection_of_lines_l747_747334


namespace angle_equality_l747_747080

variables {V C A B U P Q M : Type} [Incircle V] [Circumcircle Γ] [CircumcircleExt Γ']
variables (AB : line A B)
variables (H : is_inversion V Γ')
variables (H1 : tangent Γ' AB U ∧ tangent Γ V)
variables (H2 : on_same_side C V AB)
variables (H3 : is_bisector (∠ B C A) Γ')
variables (H4 : bisector_intersection (∠ B C A) Γ' P Q ∧ bisector_intersection (∠ B C A) Γ M)

theorem angle_equality :
  ∠(A B P) = ∠(Q B C) := sorry

end angle_equality_l747_747080


namespace probability_p_eq_l747_747208

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l747_747208


namespace sum_of_first_40_terms_l747_747841

def sequence := ℕ → ℝ

noncomputable def a_n : sequence :=
  λ n, if n = 1 then 1 else
       if n = 2 then 3 else
       if n ≥ 3 then sorry else
       0

-- Sum of the first 40 terms of the sequence {a_n}
def S_40 : ℝ := sorry -- The sum function for the sequence in Lean

theorem sum_of_first_40_terms :
  (∃ a_n : sequence, a_n 1 = 1 ∧ a_n 2 = 3 ∧
  (∀ n, n ≥ 2 → a_n (n+1) * a_n (n-1) = a_n n) ∧
  S_40 = ∑ k in (range 40), a_n k) → S_40 = 60 :=
begin
  sorry
end

end sum_of_first_40_terms_l747_747841


namespace probability_sum_is_ten_l747_747435

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747435


namespace minimize_quadratic_expression_l747_747760

theorem minimize_quadratic_expression:
  ∀ x : ℝ, (∃ a b c : ℝ, a = 1 ∧ b = -8 ∧ c = 15 ∧ x^2 + b * x + c ≥ (4 - 4)^2 - 1) :=
by
  sorry

end minimize_quadratic_expression_l747_747760


namespace find_ab_l747_747994

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem find_ab (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f a b 2 = 2) (h₂ : f a b 3 = 5) :
    (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3) :=
by 
  sorry

end find_ab_l747_747994


namespace project_completion_time_l747_747772

-- Defining the conditions of the problem.
def initial_workers : ℕ := 8
def additional_workers : ℕ := 4
def fraction_done : ℚ := 1 / 3
def initial_days : ℕ := 30
def total_work : ℚ := 1

-- Calculating the total number of workers after the addition.
def total_workers : ℕ := initial_workers + additional_workers := by rfl

-- Defining the time taken for the initial work.
def remaining_work : ℚ := total_work - fraction_done := by apply sub_eq_of_eq_add; exact (by exact_mod_cast 1 / 3).symm

-- Define the work rate of one worker per day.
def work_rate_one_worker : ℚ := fraction_done / (initial_days * initial_workers)

-- Define the work rate of all workers combined.
def work_rate_all_workers : ℚ := work_rate_one_worker * total_workers

-- Calculate days needed for the remaining work by additional workers.
def additional_days : ℚ := remaining_work / work_rate_all_workers := by rfl

-- Calculate the total days.
def total_days : ℚ := initial_days + additional_days

-- State the theorem.
theorem project_completion_time : total_days = 70 := by {
  -- Placeholder for detailed proof.
  sorry
}

end project_completion_time_l747_747772


namespace no_1999_primes_in_arithmetic_progression_less_than_12345_l747_747106

noncomputable def no_1999_primes_in_arithmetic_progression (n m d : Nat) : Prop :=
 (∀ k : Nat, k < n → n + k * d < m → Nat.prime (n + k * d)) → False

theorem no_1999_primes_in_arithmetic_progression_less_than_12345 :
  ¬ ∃ (a d : Nat), no_1999_primes_in_arithmetic_progression 1999 12345 d := sorry

end no_1999_primes_in_arithmetic_progression_less_than_12345_l747_747106


namespace infinite_pairwise_coprime_subsets_exists_l747_747065

theorem infinite_pairwise_coprime_subsets_exists (a : ℕ) (h : a > 1) :
  ∃ (s : set ℕ), s ⊆ { n | ∃ k : ℕ, n = a^(k+1) + a^k - 1 } ∧ infinite s ∧ pairwise (coprime on s) :=
sorry

end infinite_pairwise_coprime_subsets_exists_l747_747065


namespace no_non_integer_solution_l747_747765

theorem no_non_integer_solution (x y : ℚ) (m n : ℤ) (h1 : 6 * x + 5 * y = m) (h2 : 13 * x + 11 * y = n) : 
  (x ∉ ℤ) ∧ (y ∉ ℤ) → false :=
by {
  sorry
}

end no_non_integer_solution_l747_747765


namespace selling_price_l747_747265

-- Definitions
def price_coffee_A : ℝ := 10
def price_coffee_B : ℝ := 12
def weight_coffee_A : ℝ := 240
def weight_coffee_B : ℝ := 240
def total_weight : ℝ := 480
def total_cost : ℝ := (weight_coffee_A * price_coffee_A) + (weight_coffee_B * price_coffee_B)

-- Theorem
theorem selling_price (h_total_weight : total_weight = weight_coffee_A + weight_coffee_B) :
  total_cost / total_weight = 11 :=
by
  sorry

end selling_price_l747_747265


namespace no_integer_solution_for_2018_l747_747928

theorem no_integer_solution_for_2018
  {P : ℤ[X]}
  {a b c d : ℤ}
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_poly_vals : P.eval a = 2015 ∧ P.eval b = 2015 ∧ P.eval c = 2015 ∧ P.eval d = 2015) :
  ¬ ∃ x : ℤ, P.eval x = 2018 :=
by
  sorry

end no_integer_solution_for_2018_l747_747928


namespace isosceles_triangle_median_length_l747_747032

open Real

theorem isosceles_triangle_median_length
  (a α : ℝ) :
  let m := (a / 4) * sqrt(9 + tan(α)^2)
  in m = (a / 4) * sqrt(9 + tan(α)^2) :=
by
  sorry

end isosceles_triangle_median_length_l747_747032


namespace probability_sum_is_10_l747_747457

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747457


namespace batsman_average_increase_l747_747224

theorem batsman_average_increase
  (A : ℤ)
  (h1 : (16 * A + 85) / 17 = 37) :
  37 - A = 3 :=
by
  sorry

end batsman_average_increase_l747_747224


namespace cube_root_simplification_l747_747853

theorem cube_root_simplification (N : ℝ) (h : N > 1) : (N^3)^(1/3) * ((N^5)^(1/3) * ((N^3)^(1/3)))^(1/3) = N^(5/3) :=
by sorry

end cube_root_simplification_l747_747853


namespace determine_a_l747_747387

noncomputable def f (a x : ℝ) := a^x * (1 - 2^x)

theorem determine_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, f a (-x) = -f a x) : a = sqrt (2) / 2 :=
by sorry

end determine_a_l747_747387


namespace angle_of_inclination_l747_747531

theorem angle_of_inclination :
  let a := 1
  let b := real.sqrt 3
  let m := -a / b
  let θ := real.pi + real.arctan m
  θ = 5 * real.pi / 6 :=
by
  sorry

end angle_of_inclination_l747_747531


namespace dice_sum_probability_l747_747449

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747449


namespace find_a_minus_3b_l747_747138

theorem find_a_minus_3b (a b : ℤ) (h1 : a - 2 * b + 3 = 0) (h2 : -a - b + 3 = 0) : a - 3 * b = -5 :=  
by
  -- Proof steps go here
  sorry

end find_a_minus_3b_l747_747138


namespace sum_cubed_ge_sum_squared_l747_747697

theorem sum_cubed_ge_sum_squared {n : ℕ} (h : n ≥ 1) (a : Fin n → ℕ) (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∑ i : Fin n, (a i)^3 ≥ (∑ i : Fin n, a i)^2 := 
  sorry

end sum_cubed_ge_sum_squared_l747_747697


namespace coordinates_of_C_l747_747522

def Point : Type := ℝ × ℝ

def is_on_x_axis (P : Point) : Prop := P.2 = 0
def is_on_y_axis (P : Point) : Prop := P.1 = 0
def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

noncomputable def possible_points_C (A B : Point) (area : ℝ) : set Point :=
  { C | (is_on_x_axis C ∨ is_on_y_axis C) ∧ area_of_triangle A B C = area }

theorem coordinates_of_C :
  let A := (2, 0) : Point
  let B := (0, 3) : Point
  possible_points_C A B 6 = { (0, 9), (0, -3), (-2, 0), (6, 0) } :=
by
  sorry

end coordinates_of_C_l747_747522


namespace average_weight_increase_l747_747127

theorem average_weight_increase (A : ℝ) (X : ℝ) (h : (A + X) = (8 * A - 40 + 60) / 8) : X = 2.5 := by
  have h1 : 8 * (A + X) = 8 * A - 40 + 60,
  { linarith },
  have h2 : 8 * A + 8 * X = 8 * A - 40 + 60,
  { exact h1.symm },
  have h3 : 8 * X = 20,
  { linarith },
  show X = 2.5,
  { linarith }

-- We conclude the proof with this "sorry"
sorry

end average_weight_increase_l747_747127


namespace adjacent_even_sum_exists_l747_747971

-- Defining a natural number list of length 7
def seven_numbers_circle (nums : List ℕ) := nums.length = 7

-- Property: Checking if the sum of two adjacent elements is even
def adjacent_even_sum (nums : List ℕ) : Prop :=
  ∃ i, (nums.nth i % 2 + nums.nth (i + 1 % nums.length) % 2) % 2 = 0

theorem adjacent_even_sum_exists (nums : List ℕ) (h : seven_numbers_circle nums) : adjacent_even_sum nums := 
sorry

end adjacent_even_sum_exists_l747_747971


namespace largest_last_digit_l747_747616

/-
The problem is to prove that given the conditions:
1. The first digit of a string of 2015 digits is 2.
2. Any two-digit number formed by consecutive digits within this string must be divisible by either 12 or 15.
The largest possible last digit in this string is 6.
-/

theorem largest_last_digit {s : list ℕ} (h_length : s.length = 2015) 
  (h_first_digit : s.head = some 2)
  (h_consecutive_divisible : ∀ i : ℕ, i < 2014 → 
    (10 * (s.nth i).get_or_else 0 + (s.nth (i + 1)).get_or_else 0) % 12 = 0 ∨ 
    (10 * (s.nth i).get_or_else 0 + (s.nth (i + 1)).get_or_else 0) % 15 = 0) :
  s.last = some 6 :=
sorry

end largest_last_digit_l747_747616


namespace dice_sum_10_probability_l747_747428

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747428


namespace right_triangle_medians_AB_length_l747_747953

theorem right_triangle_medians_AB_length (A B C M N : Point) (h_right_triangle : right_triangle A B C)
    (hM : is_midpoint M B C) (hN : is_midpoint N A C)
    (hAM : dist A M = 5) (hBN : dist B N = 3 * sqrt 5) :
    dist A B = 2 * sqrt 14 := by
  sorry

end right_triangle_medians_AB_length_l747_747953


namespace find_angle_between_a_and_b_l747_747924

noncomputable def angle_between_vectors (a b : ℝ^3) : ℝ :=
  real.arccos (inner_product_space.of_real (inner a b) / (norm a * norm b))

theorem find_angle_between_a_and_b (a b : ℝ^3) (h1 : norm a = 1) (h2 : norm b = 1)
    (orthogonal : inner (a + 2 • b) (5 • a - 4 • b) = 0) :
    angle_between_vectors a b = real.to_degrees (real.pi / 3) :=
begin
  sorry
end

end find_angle_between_a_and_b_l747_747924


namespace octal_734_to_decimal_l747_747259

theorem octal_734_to_decimal : 
  let octal := 7 * 8^2 + 3 * 8^1 + 4 * 8^0 in
  octal = 476 := by
  sorry

end octal_734_to_decimal_l747_747259


namespace prop3_prop4_l747_747552

-- Definitions to represent planes and lines
variable (Plane Line : Type)

-- Predicate representing parallel planes or lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Predicate representing perpendicular planes or lines
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Distinct planes and a line
variables (α β γ : Plane) (l : Line)

-- Proposition 3: If l ⊥ α and l ∥ β, then α ⊥ β
theorem prop3 : perpendicular_line_plane l α ∧ parallel_line_plane l β → perpendicular α β :=
sorry

-- Proposition 4: If α ∥ β and α ⊥ γ, then β ⊥ γ
theorem prop4 : parallel α β ∧ perpendicular α γ → perpendicular β γ :=
sorry

end prop3_prop4_l747_747552


namespace percentage_of_x_minus_y_l747_747418

-- Let's introduce x, y, and P as real numbers
variables (x y P : ℝ)

-- Define the conditions provided in the problem
def condition1 : Prop := P / 100 * (x - y) = 40 / 100 * (x + y)
def condition2 : Prop := y = (1 / 9) * x

-- The theorem to prove
theorem percentage_of_x_minus_y (x y : ℝ) (P : ℝ) (h1 : condition1 x y P) (h2 : condition2 x y):
    P = 6.25 := 
begin
  sorry
end

end percentage_of_x_minus_y_l747_747418


namespace correct_time_fraction_l747_747723

theorem correct_time_fraction : 
  let incorrect_hours := {1, 10, 11, 12}
  let total_hours := 12
  let correct_hours := total_hours - incorrect_hours.size
  let total_minutes := 60
  let incorrect_minutes := 10 + 5 -- minutes with '1' in tens or ones place
  let correct_minutes := total_minutes - incorrect_minutes
  correct_hours * correct_minutes / (total_hours * total_minutes) = 1 / 2 :=
by
  let incorrect_hours := {1, 10, 11, 12}
  let total_hours := 12
  let correct_hours := total_hours - incorrect_hours.size
  let total_minutes := 60
  let incorrect_minutes := 10 + 5
  let correct_minutes := total_minutes - incorrect_minutes
  have hours_fraction := correct_hours / total_hours
  have minutes_fraction := correct_minutes / total_minutes
  have day_fraction := hours_fraction * minutes_fraction
  sorry

end correct_time_fraction_l747_747723


namespace probability_of_neither_event_l747_747202

theorem probability_of_neither_event (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.25) (h2 : P_B = 0.40) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.50 :=
by
  rw [h1, h2, h3]
  sorry

end probability_of_neither_event_l747_747202


namespace theater_ticket_sales_l747_747268

theorem theater_ticket_sales
  (A C : ℕ)
  (h₁ : 8 * A + 5 * C = 236)
  (h₂ : A + C = 34) : A = 22 :=
by
  sorry

end theater_ticket_sales_l747_747268


namespace sin_15_cos_15_l747_747337

theorem sin_15_cos_15 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := by
  sorry

end sin_15_cos_15_l747_747337


namespace hotdogs_sold_correct_l747_747237

def initial_hotdogs : ℕ := 99
def remaining_hotdogs : ℕ := 97
def sold_hotdogs : ℕ := initial_hotdogs - remaining_hotdogs

theorem hotdogs_sold_correct : sold_hotdogs = 2 := by
  sorry

end hotdogs_sold_correct_l747_747237


namespace winter_spending_l747_747141

-- Define the total spending by the end of November
def total_spending_end_november : ℝ := 3.3

-- Define the total spending by the end of February
def total_spending_end_february : ℝ := 7.0

-- Formalize the problem: prove that the spending during December, January, and February is 3.7 million dollars
theorem winter_spending : total_spending_end_february - total_spending_end_november = 3.7 := by
  sorry

end winter_spending_l747_747141


namespace fountains_for_m_4_fountains_for_m_3_l747_747301

noncomputable def ceil_div (a b : ℕ) : ℕ :=
  (a + b - 1) / b

-- Problem for m = 4
theorem fountains_for_m_4 (n : ℕ) : ∃ f : ℕ, f = 2 * ceil_div n 3 := 
sorry

-- Problem for m = 3
theorem fountains_for_m_3 (n : ℕ) : ∃ f : ℕ, f = 3 * ceil_div n 3 :=
sorry

end fountains_for_m_4_fountains_for_m_3_l747_747301


namespace least_common_positive_period_of_f_condition_l747_747306

noncomputable def least_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x : ℝ, f (x + p) = f x

theorem least_common_positive_period_of_f_condition
  (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x + 6) + f(x - 6) = f(x)) : least_period f 36 :=
by
  sorry

end least_common_positive_period_of_f_condition_l747_747306


namespace discount_price_l747_747727

theorem discount_price (original_price : ℝ) (discount_rate : ℝ) (current_price : ℝ) 
  (h1 : original_price = 120) 
  (h2 : discount_rate = 0.8) 
  (h3 : current_price = original_price * discount_rate) : 
  current_price = 96 := 
by
  sorry

end discount_price_l747_747727


namespace sum_of_factors_30_l747_747674

def sum_of_factors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ x => n % x = 0) |>.sum

theorem sum_of_factors_30 : sum_of_factors 30 = 72 := by
  sorry

end sum_of_factors_30_l747_747674


namespace Mickey_horses_per_week_l747_747314

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l747_747314


namespace quadratic_eq_positive_integer_roots_l747_747407

theorem quadratic_eq_positive_integer_roots (k p : ℕ) 
  (h1 : k > 0)
  (h2 : ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k-1) * x1^2 - p * x1 + k = 0 ∧ (k-1) * x2^2 - p * x2 + k = 0) :
  k ^ (k * p) * (p ^ p + k ^ k) + (p + k) = 1989 :=
by
  sorry

end quadratic_eq_positive_integer_roots_l747_747407


namespace dice_sum_probability_l747_747501

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747501


namespace exists_unequal_m_n_l747_747397

theorem exists_unequal_m_n (a b c : ℕ → ℕ) :
  ∃ (m n : ℕ), m ≠ n ∧ a m ≥ a n ∧ b m ≥ b n ∧ c m ≥ c n :=
sorry

end exists_unequal_m_n_l747_747397


namespace find_n_l747_747857

theorem find_n (a b : ℤ) (h₁ : a ≡ 25 [ZMOD 42]) (h₂ : b ≡ 63 [ZMOD 42]) :
  ∃ n, 200 ≤ n ∧ n ≤ 241 ∧ (a - b ≡ n [ZMOD 42]) ∧ n = 214 :=
by
  sorry

end find_n_l747_747857


namespace tyrone_gave_marbles_to_eric_l747_747907

theorem tyrone_gave_marbles_to_eric (initial_tyrone_marbles : ℕ) (initial_eric_marbles : ℕ) (marbles_given : ℕ) :
  initial_tyrone_marbles = 150 ∧ initial_eric_marbles = 30 ∧ (initial_tyrone_marbles - marbles_given = 3 * initial_eric_marbles) → marbles_given = 60 :=
by
  sorry

end tyrone_gave_marbles_to_eric_l747_747907


namespace sally_percentage_raise_l747_747590

theorem sally_percentage_raise (last_month earnings total_earnings : ℝ) (h1 : last_month = 1000) (h2 : total_earnings = 2100) :
  let this_month := total_earnings - last_month in
  let raise := this_month - last_month in
  let percentage_raise := (raise / last_month) * 100 in
  percentage_raise = 10 :=
by
  sorry

end sally_percentage_raise_l747_747590


namespace find_angle_A_find_sin_B_plus_l747_747843

noncomputable theory

variables {A B C a b c : ℝ}
variables (m n : ℝ × ℝ)

def orthogonal (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- Prove that given conditions, angle A is π / 3
theorem find_angle_A
  (h1 : m = (1, 1 - (real.sqrt 3) * real.sin A))
  (h2 : n = (real.cos A, 1))
  (h3 : orthogonal m n) :
  A = π / 3 :=
sorry

-- Prove that given b + c = sqrt(3) * a imply the value of sin(B + π / 6)
theorem find_sin_B_plus
  (h1 : A = π / 3)
  (h2 : b + c = (real.sqrt 3) * a) :
  real.sin (B + π / 6) = real.sqrt 3 / 2 :=
sorry

end find_angle_A_find_sin_B_plus_l747_747843


namespace determine_g_l747_747557

noncomputable def g : ℝ → ℝ := sorry 

lemma g_functional_equation (x y : ℝ) : g (x * y) = g ((x^2 + y^2 + 1) / 3) + (x - y)^2 :=
sorry

lemma g_at_zero : g 0 = 1 :=
sorry

theorem determine_g (x : ℝ) : g x = 2 - 2 * x :=
sorry

end determine_g_l747_747557


namespace peak_infection_day_l747_747881

-- Given conditions
def initial_cases : Nat := 20
def increase_rate : Nat := 50
def decrease_rate : Nat := 30
def total_infections : Nat := 8670
def total_days : Nat := 30

-- Peak Day and infections on that day
def peak_day : Nat := 12

-- Theorem stating what we want to prove
theorem peak_infection_day :
  ∃ n : Nat, n = initial_cases + increase_rate * (peak_day - 1) - decrease_rate * (30 - peak_day) :=
sorry

end peak_infection_day_l747_747881


namespace find_value_of_4_minus_2a_l747_747403

theorem find_value_of_4_minus_2a (a b : ℚ) (h1 : 4 + 2 * a = 5 - b) (h2 : 5 + b = 9 + 3 * a) : 4 - 2 * a = 26 / 5 := 
by
  sorry

end find_value_of_4_minus_2a_l747_747403


namespace volume_of_frustum_cone_l747_747787

-- Given definitions and conditions
variables {h R r : ℝ} (hp : h ≥ 0) (Rp : R ≥ 0) (rp : r ≥ 0)

-- Volume formula for a truncated cone
def volume_frustum (h R r : ℝ) : ℝ :=
  (1 / 3) * real.pi * h * (R^2 + R * r + r^2)

-- Volume of the truncated cone (frustum) with given height and radii
theorem volume_of_frustum_cone (h R r : ℝ) (hp : h ≥ 0) (Rp : R ≥ 0) (rp : r ≥ 0) :
  volume_frustum h R r = (1 / 3) * real.pi * h * (R^2 + R * r + r^2) :=
by
  -- This proof will be provided
  sorry

end volume_of_frustum_cone_l747_747787


namespace dice_sum_probability_l747_747478

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747478


namespace acceleration_at_3_seconds_l747_747248

def position_function (t : ℝ) : ℝ := 2 * t^3 - 6 * t^2 + 4 * t

theorem acceleration_at_3_seconds : 
  let v (t : ℝ) := deriv (position_function t)
  let a (t : ℝ) := deriv (v t)
  a 3 = 24 := by
  sorry

end acceleration_at_3_seconds_l747_747248


namespace college_girls_count_l747_747512

/-- Given conditions:
 1. The ratio of the numbers of boys to girls is 8:5.
 2. The total number of students in the college is 416.
 
 Prove: The number of girls in the college is 160.
 -/
theorem college_girls_count (B G : ℕ) (h1 : B = (8 * G) / 5) (h2 : B + G = 416) : G = 160 :=
by
  sorry

end college_girls_count_l747_747512


namespace count_davids_phone_numbers_l747_747275

theorem count_davids_phone_numbers : ∀ (a b c d e f g : ℕ), 
  (a ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧ (b ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (c ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧ (d ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (e ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧ (f ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (g ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (List.Forall (λ n, n ∈ {2, 3, 4, 5, 6, 7, 8, 9}) [a, b, c, d, e, f, g]) ∧
  (List.Nodup [a, b, c, d, e, f, g]) ∧
  (a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g) →
  (∃ s, s.card = 7 ∧ s ⊆ ({2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
       Finset.card (Finset.image id s) = 8)
:= sorry

end count_davids_phone_numbers_l747_747275


namespace probability_sum_is_10_l747_747506

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747506


namespace anna_earnings_l747_747743

/-- Anna's earnings by selling cupcakes from both odd-numbered and even-numbered trays. -/
theorem anna_earnings :
  let total_trays := 12,
      cupcakes_per_tray := 25,
      price_odd := 2,
      price_even := 3,
      sold_fraction_odd := 2/3,
      sold_fraction_even := 3/5,
      odd_trays := total_trays / 2,
      even_trays := total_trays / 2,
      total_cupcakes_odd := odd_trays * cupcakes_per_tray,
      total_cupcakes_even := even_trays * cupcakes_per_tray,
      cupcakes_sold_odd := sold_fraction_odd * total_cupcakes_odd,
      cupcakes_sold_even := sold_fraction_even * total_cupcakes_even,
      earnings_odd := cupcakes_sold_odd * price_odd,
      earnings_even := cupcakes_sold_even * price_even
  in earnings_odd + earnings_even = 470 := by
  sorry

end anna_earnings_l747_747743


namespace domain_f_l747_747136

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log(x + 1) + Real.sqrt(2 - x)

theorem domain_f :
  {x | x > -1 ∧ x ≠ 0 ∧ x ≤ 2} = {x | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_f_l747_747136


namespace triangle_problem_l747_747536

/--
Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C, respectively, 
where:
1. b * (sin B - sin C) = a * sin A - c * sin C
2. a = 2 * sqrt 3
3. the area of triangle ABC is 2 * sqrt 3

Prove:
1. A = π / 3
2. The perimeter of triangle ABC is 2 * sqrt 3 + 6
-/
theorem triangle_problem 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : 0.5 * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = Real.pi / 3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := 
sorry

end triangle_problem_l747_747536


namespace dice_sum_probability_l747_747472

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747472


namespace correct_time_fraction_l747_747722

theorem correct_time_fraction : 
  let incorrect_hours := {1, 10, 11, 12}
  let total_hours := 12
  let correct_hours := total_hours - incorrect_hours.size
  let total_minutes := 60
  let incorrect_minutes := 10 + 5 -- minutes with '1' in tens or ones place
  let correct_minutes := total_minutes - incorrect_minutes
  correct_hours * correct_minutes / (total_hours * total_minutes) = 1 / 2 :=
by
  let incorrect_hours := {1, 10, 11, 12}
  let total_hours := 12
  let correct_hours := total_hours - incorrect_hours.size
  let total_minutes := 60
  let incorrect_minutes := 10 + 5
  let correct_minutes := total_minutes - incorrect_minutes
  have hours_fraction := correct_hours / total_hours
  have minutes_fraction := correct_minutes / total_minutes
  have day_fraction := hours_fraction * minutes_fraction
  sorry

end correct_time_fraction_l747_747722


namespace harmonic_mean_of_x_and_y_l747_747564

noncomputable def x : ℝ := 88 + (40 / 100) * 88
noncomputable def y : ℝ := x - (25 / 100) * x
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 / ((1 / a) + (1 / b))

theorem harmonic_mean_of_x_and_y :
  harmonic_mean x y = 105.6 :=
by
  sorry

end harmonic_mean_of_x_and_y_l747_747564


namespace mickey_horses_per_week_l747_747307

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l747_747307


namespace figure_perimeter_l747_747140

def semicircle_perimeter (d: ℝ) : ℝ := (π * d) / 2

theorem figure_perimeter : 
  let d := 64 in
  let small_d := d / 4 in
  let large_semi := semicircle_perimeter d in
  let small_semi := semicircle_perimeter small_d in
  large_semi + 4 * small_semi = 64 * π :=
by
  let d := 64
  let small_d := d / 4
  let large_semi := semicircle_perimeter d
  let small_semi := semicircle_perimeter small_d
  show large_semi + 4 * small_semi = 64 * π
  sorry

end figure_perimeter_l747_747140


namespace min_k_48_l747_747095

theorem min_k_48 (board : matrix (fin 8) (fin 8) bool) (cover : finset (fin 8 × fin 8)) :
  (∀ figure : finset (fin 8 × fin 8), finset.card figure = 4 →
    (∃! pos : fin 8 × fin 8, (translated_figure figure pos ∩ cover).card = 4) → k ≥ 48) →
  k = 48 :=
sorry

end min_k_48_l747_747095


namespace distance_planes_eq_sum_radii_l747_747985

-- Definitions of latitude angles for Tropic of Capricorn and Arctic Circle.
def tropic_of_capricorn_latitude : Real := -23 + 27 / 60
def arctic_circle_latitude : Real := 66 + 33 / 60

-- Definition of the spherical Earth (unit radius assumed for simplicity).
def distance_between_latitude_planes (lat1 lat2 : Real) : Real :=
  Real.cos lat1 + Real.cos lat2

-- The statement of the problem.
theorem distance_planes_eq_sum_radii :
  distance_between_latitude_planes tropic_of_capricorn_latitude arctic_circle_latitude =
    Real.cos tropic_of_capricorn_latitude + Real.cos arctic_circle_latitude := by
  sorry

end distance_planes_eq_sum_radii_l747_747985


namespace modulus_sum_l747_747775

def z1 : ℂ := 3 - 5 * Complex.I
def z2 : ℂ := 3 + 5 * Complex.I

theorem modulus_sum : Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := 
by 
  sorry

end modulus_sum_l747_747775


namespace xy_yz_zx_equal_zero_l747_747968

noncomputable def side1 (x y z : ℝ) : ℝ := 1 / abs (x^2 + 2 * y * z)
noncomputable def side2 (x y z : ℝ) : ℝ := 1 / abs (y^2 + 2 * z * x)
noncomputable def side3 (x y z : ℝ) : ℝ := 1 / abs (z^2 + 2 * x * y)

def non_degenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem xy_yz_zx_equal_zero
  (x y z : ℝ)
  (h1 : non_degenerate_triangle (side1 x y z) (side2 x y z) (side3 x y z)) :
  xy + yz + zx = 0 := sorry

end xy_yz_zx_equal_zero_l747_747968


namespace final_price_of_set_l747_747030

theorem final_price_of_set (cost_coffee cost_cheesecake : ℕ) (discount_rate : ℕ) :
  cost_coffee = 6 → cost_cheesecake = 10 → discount_rate = 25 →
  (cost_coffee + cost_cheesecake) - (discount_rate * (cost_coffee + cost_cheesecake) / 100) = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end final_price_of_set_l747_747030


namespace final_price_with_discount_l747_747027

-- Definition of prices and discount
def price_coffee := 6
def price_cheesecake := 10
def discount := 25 / 100

-- Mathematical problem statement
theorem final_price_with_discount :
  let total_price := price_coffee + price_cheesecake in
  let discount_amount := discount * total_price in
  let final_price := total_price - discount_amount in
  final_price = 12 :=
by
  sorry

end final_price_with_discount_l747_747027


namespace constant_sequence_l747_747415

theorem constant_sequence (a : ℕ+ → ℕ+) 
  (h : ∀ i j : ℕ+, i > j → (i - j)^(2 * (i - j)) + 1 ∣ a i - a j) : 
  ∀ n m : ℕ+, a n = a m := 
sorry

end constant_sequence_l747_747415


namespace primes_with_divisors_equality_l747_747401

theorem primes_with_divisors_equality :
  ∃! p : ℕ, Prime p ∧ (∃ d : ℕ, d = p^2 + 23 ∧ nat.factors d.length + 1 = 14) :=
sorry

end primes_with_divisors_equality_l747_747401


namespace dice_sum_probability_l747_747473

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747473


namespace basketball_team_squad_count_l747_747707

theorem basketball_team_squad_count : 
  let n := 12 in 
  let captain_choices := n in 
  let player_choices := Nat.choose (n - 1) 5 in 
  captain_choices * player_choices = 5544 :=
by
  sorry

end basketball_team_squad_count_l747_747707


namespace john_total_skateboarded_distance_l747_747052

noncomputable def total_skateboarded_distance (to_park: ℕ) (back_home: ℕ) : ℕ :=
  to_park + back_home

theorem john_total_skateboarded_distance :
  total_skateboarded_distance 10 10 = 20 :=
by
  sorry

end john_total_skateboarded_distance_l747_747052


namespace ellipse_major_axis_sqrt2_minor_axis_l747_747633

def ellipse_condition (a b : ℝ) : Prop :=
  ∃ (x1 y1 : ℝ), 
    (x1^2 / a^2 + y1^2 / b^2 = 1) ∧
    let Q := (x1, 0) in 
    let R := (x1 / 2, 0) in 
    let O := (0, 0) in 
    let RP := (x1 / 2, y1) in 
    -- condition for RP being perpendicular to the tangent at P
    (∃ (λ : ℝ), RP = λ * ((x1 / a^2), (y1 / b^2)))

theorem ellipse_major_axis_sqrt2_minor_axis (a b : ℝ) :
  (ellipse_condition a b) → a = Real.sqrt 2 * b :=
by
  sorry

end ellipse_major_axis_sqrt2_minor_axis_l747_747633


namespace cafeteria_problem_l747_747879

theorem cafeteria_problem (C : ℕ) 
    (h1 : ∃ h : ℕ, h = 4 * C)
    (h2 : 5 = 5)
    (h3 : C + 4 * C + 5 = 40) : 
    C = 7 := sorry

end cafeteria_problem_l747_747879


namespace intersection_points_ellipse_l747_747420

noncomputable def intersection_points (m n : ℝ) : ℕ :=
if (m^2 + n^2 < 4) then 2 else 0

theorem intersection_points_ellipse 
    (m n : ℝ) 
    (h_line_no_intersection : m^2 + n^2 < 4) :
    intersection_points m n = 2 :=
by
    unfold intersection_points
    simp [h_line_no_intersection]
    sorry

end intersection_points_ellipse_l747_747420


namespace ellipse_properties_l747_747370

open Real

-- Definition of the ellipse
def ellipse (m : ℝ) (x y : ℝ) := (x^2) / (3 * m) + (y^2) / m = 1

-- m in terms of the major axis length
def m_from_major_axis (major_axis_length : ℝ) := 
  (major_axis_length / 2)^2 / 3

-- Equation of the ellipse
def ellipse_equation (x y : ℝ) := 
  ellipse (m_from_major_axis (2 * sqrt 6)) x y

-- Eccentricity calculation
def eccentricity (m : ℝ) : ℝ :=
  sqrt (1 - (m / (3 * m)))

-- Minimum value of OB
def min_ob (y0 : ℝ) : ℝ :=
  abs y0 + (3 / (2 * abs y0))

theorem ellipse_properties :
  (m_from_major_axis (2 * sqrt 6) = 2) ∧
  (∀ x y, ellipse 2 x y ↔ (x^2 / 6 + y^2 / 2 = 1)) ∧ 
  (eccentricity 2 = sqrt 6 / 3) ∧
  (∀ y0, y0 = sqrt 6 / 2 → min_ob y0 = sqrt 6) :=
by {
  sorry
}

end ellipse_properties_l747_747370


namespace eight_distinct_stops_ten_distinct_stops_impossible_l747_747215

-- Define the total number of stops
def stops : ℕ := 14

-- Define the maximum number of passengers allowed on the bus at any time
def max_passengers : ℕ := 25

-- Part (a): Prove that there exist 8 distinct stops such that no passengers travel between certain pairs
theorem eight_distinct_stops :
  ∃ (A1 A2 A3 A4 B1 B2 B3 B4 : ℕ), 
  (∀ (i j : ℕ), i ≠ j → A1 ≠ A2 ∧ A1 ≠ A3 ∧ A1 ≠ A4 ∧ A2 ≠ A3 ∧ A2 ≠ A4 ∧ A3 ≠ A4  ∧ B1 ≠ B2 ∧ B1 ≠ B3 ∧ B1 ≠ B4 ∧ B2 ≠ B3 ∧ B2 ≠ B4 ∧ B3 ≠ B4) ∧
  ∀ i : ℕ, 
  passenger_travels i A1 B1 = false ∧ 
  passenger_travels i A2 B2 = false ∧ 
  passenger_travels i A3 B3 = false ∧ 
  passenger_travels i A4 B4 = false := 
sorry

-- Part (b): Prove it's possible that there do not exist 10 distinct stops with similar properties
theorem ten_distinct_stops_impossible :
  ∀ (A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 : ℕ), 
  (∀ (i j : ℕ), i ≠ j → A1 ≠ A2 ∧ A1 ≠ A3 ∧ A1 ≠ A4 ∧ A1 ≠ A5 ∧ A2 ≠ A3 ∧ A2 ≠ A4 ∧ A2 ≠ A5 ∧ A3 ≠ A4 ∧ A3 ≠ A5 ∧ A4 ≠ A5 ∧ B1 ≠ B2 ∧ B1 ≠ B3 ∧ B1 ≠ B4 ∧ B1 ≠ B5 ∧ B2 ≠ B3 ∧ B2 ≠ B4 ∧ B2 ≠ B5 ∧ B3 ≠ B4 ∧ B3 ≠ B5 ∧ B4 ≠ B5) → 
  ∃ (i : ℕ), 
  passenger_travels i A1 B1 ∧ 
  passenger_travels i A2 B2 ∧ 
  passenger_travels i A3 B3 ∧ 
  passenger_travels i A4 B4 ∧ 
  passenger_travels i A5 B5 := 
sorry

end eight_distinct_stops_ten_distinct_stops_impossible_l747_747215


namespace find_x_value_l747_747339

theorem find_x_value (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 9 :=
begin
  sorry
end

end find_x_value_l747_747339


namespace probability_sum_is_10_l747_747505

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747505


namespace count_isosceles_triangles_eq_27_l747_747798

theorem count_isosceles_triangles_eq_27 : 
  (∃ (a b c : ℕ), a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ 
                  (a = b ∨ b = c ∨ a = c) ∧ 
                  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) → 
  count_isosceles_triangles ({1, 2, 3, 4, 5, 6} : Finset ℕ) = 27 :=
by 
  sorry

end count_isosceles_triangles_eq_27_l747_747798


namespace cannot_be_54_l747_747629

def initial_number : ℕ := 12

def is_integer_div (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def operation (n : ℕ) : set ℕ :=
  {k | k = n * 2 ∨ k = n * 3 ∨ (is_integer_div n 2 ∧ k = n / 2) ∨ (is_integer_div n 3 ∧ k = n / 3)}

theorem cannot_be_54 (n : ℕ) (t : ℕ) (h_initial : n = initial_number) (h_time : t = 60) :
  ∀ k, (k ∈ operation n.t times → n ≠ 54) sorry

end cannot_be_54_l747_747629


namespace find_equations_l747_747805

-- Definitions and conditions
def line (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

def circle (x y r : ℝ) (x0 y0 : ℝ) : Prop := (x - x0)^2 + (y - y0)^2 = r^2

def tangent_line_circle (a b c : ℝ) (x0 y0 r : ℝ) : Prop := 
  ∃ (x1 y1 : ℝ), circle x0 y0 r x1 y1 ∧ line a b c x1 y1

def does_not_pass_second_quadrant (a b c : ℝ) : Prop :=
  ∀ (x y : ℝ), x < 0 → y > 0 → ¬line a b c x y

noncomputable def given (l : ℝ → ℝ → ℝ → ℝ → Prop) (p : ℝ × ℝ) :=
  let (x, y) := p in line (2 : ℝ) (-1 : ℝ) (-4 : ℝ) x y ∧ tangent_line_circle 2 (-1) (-4) 0 1 5

noncomputable def parallel_lines (a b c c' : ℝ) :=
  a = 2 ∧ b = -1 ∧ c ≠ c'

noncomputable def symmetric_with_respect_to_y_equals_1 (a b c x' y' : ℝ) :=
  ∃ (x y : ℝ), line a b c x y ∧ x' = x ∧ y' = 2 - y

noncomputable def line_l1 (a b p : ℝ → ℝ → ℝ) (x y c' : ℝ) (p : ℝ × ℝ) :=
  parallel_lines a b (-4) c' → 
  line a b c' x y ∧ does_not_pass_second_quadrant (a : ℝ) (b : ℝ) (c' : ℝ) p

theorem find_equations :
  ∀ (l l1 l2 : ℝ → ℝ → ℝ → ℝ → Prop) (p p1 : ℝ × ℝ),
    line (2 : ℝ) (-1 : ℝ) (-4 : ℝ) p.1 p.2 →
    tangent_line_circle (2 : ℝ) (-1 : ℝ) (-4 : ℝ) 0 1 5 →
    line_l1 2 -1 4 3 -1 (-7) p1 →
    symmetric_with_respect_to_y_equals_1 2 1 9 4 1 →
    l = line (2:ℝ) (-1 :ℝ) (-4:ℝ) ∧ l2 = line (2:ℝ) (1:ℝ) (-9:ℝ) :=
sorry

end find_equations_l747_747805


namespace dot_product_correct_magnitude_correct_l747_747820

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def angle_between_vectors : ℝ := 120
def magnitude_a : ℝ := ∥a∥ = 1
def magnitude_b : ℝ := ∥b∥ = 5

-- Expected answers
def dot_product_ab := a • b = -5 / 2
def magnitude_3a_minus_b := ∥(3 : ℝ) • a - b∥ = 7

-- Lean theorem statements
theorem dot_product_correct : angle_between_vectors = 120 ∧ magnitude_a ∧ magnitude_b → dot_product_ab := by sorry

theorem magnitude_correct : angle_between_vectors = 120 ∧ magnitude_a ∧ magnitude_b → magnitude_3a_minus_b := by sorry

end dot_product_correct_magnitude_correct_l747_747820


namespace probability_sum_is_10_l747_747511

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747511


namespace hyperbola_equation_l747_747804

theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3 / 3)
  (h4 : c^2 = a^2 + b^2) (h5 : b * c / Real.sqrt (a^2 + b^2) = 2) :
  ∃ (a b : ℝ), a = 2 * Real.sqrt 3 ∧ b = 2 ∧ (a > 0 ∧ b > 0 ∧ (a ≠ 0)) ∧ 
  (Real.sqrt (a^2 + b^2) = c) ∧ (b * c / Real.sqrt (a^2 + b^2) = 2) ∧ 
  ( ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 12 - y^2 / 4 = 1 ) := sorry

end hyperbola_equation_l747_747804


namespace maximum_possible_S_l747_747526

-- Definition of conditions
def number_placement (grid : ℕ × ℕ → ℕ) : Prop :=
  (∀ i j, 1 ≤ grid (i, j) ∧ grid (i, j) ≤ 10000) ∧
  (∀ n, ∃ i j, grid (i, j) = n) ∧
  (∀ i j, (i < 100 → grid (i, j) + 1 = grid (i+1, j) ∨ j < 100 → grid (i, j) + 1 = grid (i, j+1)))

-- Definition of distance formula
def distance (i j k l : ℕ) : ℝ :=
  real.sqrt ((k - i : ℝ) ^ 2 + (l - j : ℝ) ^ 2)

-- Main theorem
theorem maximum_possible_S 
  (grid : ℕ × ℕ → ℕ) 
  (h_placement : number_placement grid) : 
  ∃ S, S = 50 * real.sqrt 2 ∧ 
       ∀ i j k l, 
       grid (i, j) - grid (k, l) = 5000 → 
       S ≥ distance i j k l :=
begin
  sorry
end

end maximum_possible_S_l747_747526


namespace crucian_carps_heavier_l747_747703

-- Variables representing the weights
variables (K O L : ℝ)

-- Given conditions
axiom weight_6K_lt_5O : 6 * K < 5 * O
axiom weight_6K_gt_10L : 6 * K > 10 * L

-- The proof statement
theorem crucian_carps_heavier : 2 * K > 3 * L :=
by
  -- Proof would go here
  sorry

end crucian_carps_heavier_l747_747703


namespace total_volume_of_cubes_l747_747676

theorem total_volume_of_cubes (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 6) :
  a^3 + b^3 + c^3 = 368 :=
by
  rw [h1, h2, h3]
  calc
    3^3 + 5^3 + 6^3 = 27 + 125 + 216 : by norm_num
                ... = 368           : by norm_num

end total_volume_of_cubes_l747_747676


namespace pair_2_n_is_good_l747_747547

-- Definitions
def is_removed {n : ℕ} (pos : ℕ) : Prop := pos % n = 0

def remove_every_nth (seq : List ℕ) (n : ℕ) : List ℕ :=
  seq.filter (λ x, ¬is_removed n (x))

def position_after_removal (seq : List ℕ) (n : ℕ) (k : ℕ) : Option ℕ :=
  seq.enum.filter (λ (p : Nat × ℕ), ¬is_removed n (p.1 + 1)) |>.findIdx (λ p, p.2 = k)

-- Main Theorem (Statement only)
theorem pair_2_n_is_good (n : ℕ) (hpos : n > 0) : 
∀ (k : ℕ), (k ∈ (remove_every_nth (remove_every_nth (List.range (k + 1)) 2) n)) ↔ 
(k ∈ (remove_every_nth (remove_every_nth (List.range (k + 1)) n) 2)) :=
sorry

end pair_2_n_is_good_l747_747547


namespace ratio_proof_l747_747034

variables {E F G H Q R S : Type}
variables [add_comm_group E] [add_comm_group F] [add_comm_group G] [add_comm_group H]
variables [module ℝ E] [module ℝ F] [module ℝ G] [module ℝ H]
variables (parallelogram : E → F → G → H → Prop)
variables (is_parallelogram : parallelogram E F G H)
variables [affine_space ℝ E] [affine_space ℝ F] [affine_space ℝ G] [affine_space ℝ H]

-- definition of Q and R according to the given ratios
def Q_on_EF := 3/7 * (F - E) + E
def R_on_EH := 1/4 * (H - E) + E

-- intersection condition
def is_intersection (S : Type) := ∃ t u : ℝ, t * (G - E) + E = (1 - u) * (Q - R) + Q

theorem ratio_proof (parallelogram E F G H) 
  (hQ : Q = Q_on_EF) 
  (hR : R = R_on_EH) 
  (hS : is_intersection S) :
  (EG/ES) = (1/3) :=
sorry

end ratio_proof_l747_747034


namespace min_value_of_frac_l747_747371

variable {x y a b : ℝ}

noncomputable def ellipse (a b : ℝ) := ∀ (x y : ℝ), (x / a) ^ 2 + (y / b) ^ 2 = 1

noncomputable def eccentricity (a b : ℝ) := (Math.sqrt (a ^ 2 - b ^ 2) / a)

noncomputable def minimum_value (a b : ℝ) := (a ^ 2 + 1) / b

theorem min_value_of_frac (a b : ℝ) (ha : a > b) (hb : b > 0) (he : eccentricity a b = 1 / 2) :
    minimum_value a b = 4 * Real.sqrt 3 / 3 :=
sorry

end min_value_of_frac_l747_747371


namespace ellipse_distance_proof_l747_747302

noncomputable def distance_CD_of_ellipse : ℝ :=
  let center := (-2 : ℝ, 3 : ℝ)
  let semi_major_axis := 4
  let semi_minor_axis := 2
  real.sqrt (semi_major_axis^2 + semi_minor_axis^2)

theorem ellipse_distance_proof :
  ∀ (x y : ℝ) (h : 16*(x + 2)^2 + 4*(y - 3)^2 = 64),
  distance_CD_of_ellipse = 2 * real.sqrt 5 :=
by
  sorry

end ellipse_distance_proof_l747_747302


namespace symmetric_points_origin_l747_747417

theorem symmetric_points_origin {a b : ℝ} (h₁ : a = -(-4)) (h₂ : b = -(3)) : a - b = 7 :=
by 
  -- since this is a statement template, the proof is omitted
  sorry

end symmetric_points_origin_l747_747417


namespace length_of_side_b_max_area_of_triangle_l747_747904

variable {A B C a b c : ℝ}
variable {triangle_ABC : a + c = 6}
variable {eq1 : (3 - Real.cos A) * Real.sin B = Real.sin A * (1 + Real.cos B)}

-- Theorem for part (1) length of side b
theorem length_of_side_b :
  b = 2 :=
sorry

-- Theorem for part (2) maximum area of the triangle
theorem max_area_of_triangle :
  ∃ (S : ℝ), S = 2 * Real.sqrt 2 :=
sorry

end length_of_side_b_max_area_of_triangle_l747_747904


namespace distinct_sequences_count_l747_747234

def binomial_coefficient (n k : Nat) : Nat :=
  Nat.choose n k

def favorable_sequences (n : Nat) : Nat :=
  binomial_coefficient n 6 +
  binomial_coefficient n 7 +
  binomial_coefficient n 8 +
  binomial_coefficient n 9 +
  binomial_coefficient n 10

theorem distinct_sequences_count (n : Nat) (hn : n = 10) :
    favorable_sequences n = 386 :=
by 
  simp [hn, favorable_sequences, binomial_coefficient, Nat.choose]
  sorry

end distinct_sequences_count_l747_747234


namespace chord_PQ_eqn_l747_747412

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9
def midpoint_PQ (M : ℝ × ℝ) : Prop := M = (1, 2)
def line_PQ_eq (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem chord_PQ_eqn : 
  (∃ P Q : ℝ × ℝ, circle_eq P.1 P.2 ∧ circle_eq Q.1 Q.2 ∧ midpoint_PQ ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →
  ∃ x y : ℝ, line_PQ_eq x y := 
sorry

end chord_PQ_eqn_l747_747412


namespace relationship_among_abc_l747_747796

noncomputable def log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem relationship_among_abc :
  let a := log_base 2 5
  let b := log_base 5 (log_base 2 5)
  let c := (1 / 2) ^ (-0.52)
  b < c ∧ c < a :=
by
  intro a b c
  dsimp [a, b, c, log_base]
  sorry

end relationship_among_abc_l747_747796


namespace equivalent_statement_l747_747385

theorem equivalent_statement (x y z w : ℝ)
  (h : (2 * x + y) / (y + z) = (z + w) / (w + 2 * x)) :
  (x = z / 2 ∨ 2 * x + y + z + w = 0) :=
sorry

end equivalent_statement_l747_747385


namespace max_value_g_f_less_than_e_x_div_x_sq_l747_747833

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_g : ∃ x, x = 3 ∧ g x = 2 * Real.log 2 - 7 / 4 := by
  sorry

theorem f_less_than_e_x_div_x_sq (x : ℝ) (hx : x > 0) : f x < (Real.exp x - 1) / x ^ 2 := by
  sorry

end max_value_g_f_less_than_e_x_div_x_sq_l747_747833


namespace probability_sum_10_l747_747491

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747491


namespace kaiden_collected_first_week_l747_747057

variable (collected_first_week collected_second_week goal needed_more : ℕ)

theorem kaiden_collected_first_week :
  collected_second_week = 259 → goal = 500 → needed_more = 83 →
  collected_first_week = goal - needed_more - collected_second_week :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

#check kaiden_collected_first_week

end kaiden_collected_first_week_l747_747057


namespace piece_exits_grid_at_A2_l747_747023

-- Defining the grid size
def gridSize : ℕ := 4

-- Defining possible movements
inductive Direction
| up
| down
| left
| right

-- Describing the piece movement on the grid
structure Move where
  current_pos : ℕ × ℕ
  direction : Direction
  next_move : Move

-- Function to determine the next cell and update the arrow direction
noncomputable def move_piece (pos : ℕ × ℕ) : (ℕ × ℕ) :=
  match pos with
  -- Starting cell is C2 (which is (2, 2) in zero-based indexing)
  | (2, 2) => (2, 3) -- moves right
  -- Each step's movement computation
  | (2, 3) => (1, 3) -- moves up
  | (1, 3) => (0, 3) -- moves up
  | (0, 3) => (0, 4) -- moves right (out of the grid)
  | ... -- add other movements according to the sequence
  | _ => (0, 0) -- termination condition or remaining cells; this should be handled more precisely
  end

-- Function to check if the piece has exited the grid
noncomputable def has_exited (pos : ℕ × ℕ) : Prop :=
  pos.1 < 0 ∨ pos.1 >= gridSize ∨ pos.2 < 0 ∨ pos.2 >= gridSize

-- Main theorem to prove the problem statement
theorem piece_exits_grid_at_A2 :
  ∃ (pos : ℕ × ℕ), pos = (0, 1) ∧ has_exited (move_piece pos) :=
by
  sorry

end piece_exits_grid_at_A2_l747_747023


namespace lateral_surface_area_cylinder_l747_747624

theorem lateral_surface_area_cylinder (α d : ℝ) : 
  let S := (1 / 2) * d^2 * real.sin α in
  true := sorry

end lateral_surface_area_cylinder_l747_747624


namespace find_m_l747_747871

theorem find_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 :=
by
  intros h
  sorry

end find_m_l747_747871


namespace clock_correct_fraction_l747_747724

/--
A 12-hour digital clock displays the hour and minute of a day. 
Whenever it is supposed to display a 1, it mistakenly displays a 9. 
Prove that the fraction of the day the clock shows the correct time is 1/2.
-/
def correct_fraction_hours : ℚ := 2 / 3

def correct_fraction_minutes : ℚ := 3 / 4

theorem clock_correct_fraction : correct_fraction_hours * correct_fraction_minutes = 1 / 2 :=
by
  have hours_correct := correct_fraction_hours
  have minutes_correct := correct_fraction_minutes
  calc
    (correct_fraction_hours * correct_fraction_minutes) = (2 / 3 * 3 / 4) : by sorry
    ... = 1 / 2 : by sorry

end clock_correct_fraction_l747_747724


namespace expression_is_correct_l747_747324

theorem expression_is_correct (a : ℝ) : 2 * (a + 1) = 2 * a + 1 := 
sorry

end expression_is_correct_l747_747324


namespace weather_conditions_l747_747281

variable (Temperature Sunny Wind Crowded : Prop)
variable (T : ℝ)

axiom temperature_condition : T ≥ 75 → Sunny → Wind < 10 → Crowded
axiom not_crowded : ¬Crowded

theorem weather_conditions :
  ¬Crowded → (T < 75 ∨ ¬Sunny ∨ Wind ≥ 10) :=
begin
  intro h,
  contrapose! h,
  from temperature_condition,
end

end weather_conditions_l747_747281


namespace number_of_pink_cookies_l747_747958

def total_cookies : ℕ := 86
def red_cookies : ℕ := 36

def pink_cookies (total red : ℕ) : ℕ := total - red

theorem number_of_pink_cookies : pink_cookies total_cookies red_cookies = 50 :=
by
  sorry

end number_of_pink_cookies_l747_747958


namespace prime_divisor_gt_n_cubed_l747_747716

theorem prime_divisor_gt_n_cubed (n : ℕ) (h : ∀ d : ℕ, d ∣ n → ¬ (n^2 ≤ d^4 ∧ d^4 ≤ n^3)) :
  ∃ p : ℕ, prime p ∧ p ∣ n ∧ p^4 > n^3 :=
by sorry

end prime_divisor_gt_n_cubed_l747_747716


namespace certain_number_l747_747862

theorem certain_number (x : ℝ) (h : 7125 / x = 5700) : x = 1.25 := 
sorry

end certain_number_l747_747862


namespace circle_params_sum_eq_l747_747920

theorem circle_params_sum_eq : 
  let C' := (x : ℝ) ^ 2 + 6 * (y : ℝ) - 4 = -(y : ℝ) ^ 2 + 12 * (x : ℝ) - 12 in
  let a' := 6 in
  let b' := -3 in
  let r' := Real.sqrt 37 in
  a' + b' + r' = 3 + Real.sqrt 37 :=
by
  sorry

end circle_params_sum_eq_l747_747920


namespace females_in_coach_class_l747_747199

theorem females_in_coach_class (total_passengers : ℕ) (percent_female : ℕ) 
    (percent_first_class : ℕ) (fraction_male_first_class : ℚ) : ℕ :=
  let num_females := (percent_female * total_passengers) / 100 in
  let num_first_class := (percent_first_class * total_passengers) / 100 in
  let num_males_first_class := fraction_male_first_class * num_first_class in
  let num_females_first_class := num_first_class - num_males_first_class in
  num_females - num_females_first_class

example : females_in_coach_class 120 55 10 (1/3) = 58 := by
  let total_passengers := 120
  let percent_female := 55
  let percent_first_class := 10
  let fraction_male_first_class := (1 : ℚ) / 3

  let num_females := (percent_female * total_passengers) / 100
  let num_first_class := (percent_first_class * total_passengers) / 100
  let num_males_first_class := fraction_male_first_class * num_first_class
  let num_females_first_class := num_first_class - num_males_first_class

  have num_females_eq : num_females = 66 := by norm_num
  have num_first_class_eq : num_first_class = 12 := by norm_num
  have num_males_first_class_eq : num_males_first_class = (4 : ℚ) := by norm_num
  have num_females_first_class_eq : num_females_first_class = (8 : ℚ) := by norm_num

  show num_females - num_females_first_class = 58
  · norm_num
  sorry

end females_in_coach_class_l747_747199


namespace evaluate_expression_l747_747750

-- Defining the conditions for the cosine and sine values
def cos_0 : Real := 1
def sin_3pi_2 : Real := -1

-- Proving the given expression equals -1
theorem evaluate_expression : 3 * cos_0 + 4 * sin_3pi_2 = -1 :=
by 
  -- Given the definitions, this will simplify as expected.
  sorry

end evaluate_expression_l747_747750


namespace problem_l747_747515

noncomputable theory
open_locale classical

variables (A B C H : Type) [triangle : Triangle A B C] 
  (a b c : ℝ) (AH BH CH : ℝ)

def orthocenter (A B C H : Type) [Triangle A B C] : Prop := sorry
def acute_triangle (A B C : Type) [Triangle A B C] : Prop := sorry

theorem problem
  (h_triangle : acute_triangle A B C)
  (h_orthocenter : orthocenter A B C H)
  (cond1 : a > 0)
  (cond2 : b > 0)
  (cond3 : c > 0)
  (cond4 : AH > 0)
  (cond5 : BH > 0)
  (cond6 : CH > 0)
  : a * BH * CH + b * CH * AH + c * AH * BH = a * b * c :=
sorry

end problem_l747_747515


namespace min_value_abs_function_l747_747625

theorem min_value_abs_function : ∀ (x : ℝ), (|x + 1| + |2 - x|) ≥ 3 :=
by
  sorry

end min_value_abs_function_l747_747625


namespace mapping_has_output_l747_747807

variable (M N : Type) (f : M → N)

theorem mapping_has_output (x : M) : ∃ y : N, f x = y :=
by
  sorry

end mapping_has_output_l747_747807


namespace exponential_inequalities_l747_747353

theorem exponential_inequalities (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : 2^b < 2^a ∧ 2^a < 3^a :=
by
  sorry

end exponential_inequalities_l747_747353


namespace number_on_far_left_of_20th_row_l747_747581

theorem number_on_far_left_of_20th_row : ∀ (n : ℕ), (∑ k in finset.range n, 2*k + 1) = n^2 → (∑ k in finset.range 19, 2*k + 1) + 1 = 362 :=
by
  intros n h
  sorry

end number_on_far_left_of_20th_row_l747_747581


namespace perimeter_of_H_l747_747712

-- Define the dimensions of the rectangles
def width := 3
def height := 5

-- Define the positions (overlapping parts)
def overlap := 2

-- Define the perimeter of the "H"
def calc_perimeter : ℕ :=
  2 * (height + width) + 2 * (width - overlap)

-- Theorem statement
theorem perimeter_of_H:
  calc_perimeter = 26 :=
by 
  -- Calculation skipping proof for now
  sorry

end perimeter_of_H_l747_747712


namespace find_highest_m_l747_747241

def reverse_num (m : ℕ) : ℕ :=
  let digits := m.toString.data.toList.reverse
  digits.foldl (λ acc c => acc * 10 + c.toNat - '0'.toNat) 0

theorem find_highest_m (m : ℕ) :
  (1000 ≤ m ∧ m ≤ 9999) ∧
  (1000 ≤ reverse_num m ∧ reverse_num m ≤ 9999) ∧
  (m % 36 = 0 ∧ reverse_num m % 36 = 0) ∧
  (m % 7 = 0) →
  m = 5796 :=
by
  sorry

end find_highest_m_l747_747241


namespace minimum_cumulative_score_champion_l747_747159

def initial_scores : list ℕ := [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

def ranking : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def result_score (initial: ℕ) (win: ℕ) (loss: ℕ) : ℕ :=
  initial + win

def min_possible_cumulative_score (scores: list ℕ) : ℕ :=
  12  -- From the conditions, we derive this minimum score for the new champion

theorem minimum_cumulative_score_champion : 
  min_possible_cumulative_score initial_scores = 12 :=
by
sory

end minimum_cumulative_score_champion_l747_747159


namespace solution_set_of_inequality_l747_747555

noncomputable theory
open Real

/-- For a function f defined on ℝ, if f(0) = 2 and ∀ x ∈ ℝ, f(x) + f'(x) > 1,
    then the solution set of the inequality e^x * f(x) > e^x + 1 is (0, +∞). -/
theorem solution_set_of_inequality (f : ℝ → ℝ) (hf0 : f 0 = 2)
  (h' : ∀ x : ℝ, f x + (deriv f x) > 1) :
  {x | exp x * f x > exp x + 1} = set.Ioi 0 :=
sorry

end solution_set_of_inequality_l747_747555


namespace boxcar_capacity_ratio_l747_747970

-- The known conditions translated into Lean definitions
def red_boxcar_capacity (B : ℕ) : ℕ := 3 * B
def blue_boxcar_count : ℕ := 4
def red_boxcar_count : ℕ := 3
def black_boxcar_count : ℕ := 7
def black_boxcar_capacity : ℕ := 4000
def total_capacity : ℕ := 132000

-- The mathematical condition as a Lean theorem statement.
theorem boxcar_capacity_ratio 
  (B : ℕ)
  (h_condition : (red_boxcar_count * red_boxcar_capacity B + 
                  blue_boxcar_count * B + 
                  black_boxcar_count * black_boxcar_capacity = 
                  total_capacity)) : 
  black_boxcar_capacity / B = 1 / 2 := 
sorry

end boxcar_capacity_ratio_l747_747970


namespace required_cement_l747_747256

def total_material : ℝ := 0.67
def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem required_cement : cement = total_material - (sand + dirt) := 
by
  sorry

end required_cement_l747_747256


namespace number_of_possible_multisets_of_roots_l747_747615

theorem number_of_possible_multisets_of_roots :
  let roots := {1, -1, 2, -2}
  in ∃ multisets : multiset ℤ, (multisets.card = 8 ∧ ∀ r ∈ multisets, r ∈ roots) ∧ 
  (multiset.card (multisets.filter (λ r, r = 1)) 
   + multiset.card (multisets.filter (λ r, r = -1)) 
   + multiset.card (multisets.filter (λ r, r = 2)) 
   + multiset.card (multisets.filter (λ r, r = -2)) = 8)
  :=
    (multiset.filter (λ r, r = 1) + 
     multiset.filter (λ r, r = -1) + 
     multiset.filter (λ r, r = 2) + 
     multiset.filter (λ r, r = -2)).card = 8 → 
163 :=
sorry

end number_of_possible_multisets_of_roots_l747_747615


namespace seventh_triangular_number_is_28_l747_747631

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_triangular_number_is_28 : triangular_number 7 = 28 :=
by
  /- proof goes here -/
  sorry

end seventh_triangular_number_is_28_l747_747631


namespace sum_of_first_51_decimal_digits_of_9_over_11_l747_747754

theorem sum_of_first_51_decimal_digits_of_9_over_11 :
  let decimal_expansion := "818181818181818181818181818181818181818181818181818" in
  let first_51_digits := take 51 decimal_expansion in
  let digit_sum := first_51_digits.foldl (λ acc x => acc + (x.to_nat - '0'.to_nat)) 0 in
  digit_sum = 233 :=
by
  sorry

end sum_of_first_51_decimal_digits_of_9_over_11_l747_747754


namespace only_three_A_l747_747788

def student := Type
variable (Alan Beth Carlos Diana Eliza : student)

variable (gets_A : student → Prop)

variable (H1 : gets_A Alan → gets_A Beth)
variable (H2 : gets_A Beth → gets_A Carlos)
variable (H3 : gets_A Carlos → gets_A Diana)
variable (H4 : gets_A Diana → gets_A Eliza)
variable (H5 : gets_A Eliza → gets_A Alan)
variable (H6 : ∃ a b c : student, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ gets_A a ∧ gets_A b ∧ gets_A c ∧ ∀ d : student, gets_A d → d = a ∨ d = b ∨ d = c)

theorem only_three_A : gets_A Carlos ∧ gets_A Diana ∧ gets_A Eliza :=
by
  sorry

end only_three_A_l747_747788


namespace find_p1_plus_q1_l747_747063

def x_power_8_minus_98x_power_4_plus_1 : Polynomial ℤ := 
  Polynomial.X ^ 8 - 98 * Polynomial.X ^ 4 + 1

def p : Polynomial ℤ := 
  Polynomial.X ^ 4 + 10 * Polynomial.X ^ 2 + 1

def q : Polynomial ℤ := 
  Polynomial.X ^ 4 - 10 * Polynomial.X ^ 2 + 1

theorem find_p1_plus_q1 : 
  p.eval 1 + q.eval 1 = 4 :=
by
  sorry

end find_p1_plus_q1_l747_747063


namespace cheese_division_l747_747160

-- Define the problem statement
theorem cheese_division (masses : Fin 9 → ℝ) (h_distinct : Function.Injective masses) :
  ∃ (m_cut : ℝ × ℝ) (i : Fin 9), 
    (∃ g1 g2 : Fin 10 → ℝ, 
      (∀ j, j < 5 → (∃ k, g1 j = if j.val < i.val then masses k else if j.val = 4 then m_cut.1 else masses (Fin.ofNat (j.val+1))) ∈ Finset.image masses Finset.univ ∪ {m_cut.1, m_cut.2}) ∧
      (∀ j, j < 5 → (∃ k, g2 j = if j.val < i.val then masses k else if j.val = 4 then m_cut.2 else masses (Fin.ofNat (j.val+1))) ∈ Finset.image masses Finset.univ ∪ {m_cut.1, m_cut.2}) ∧
      (Finset.ofList (List.g1.toList)).sum id = (Finset.ofList (List.g2.toList)).sum id) :=
sorry

end cheese_division_l747_747160


namespace cylinder_height_relationship_l747_747168

variable (r₁ h₁ r₂ h₂ : ℝ)

def radius_relationship := r₂ = 1.1 * r₁

def volume_equal := π * r₁^2 * h₁ = π * r₂^2 * h₂

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) 
     (h_radius : radius_relationship r₁ r₂) 
     (h_volume : volume_equal r₁ h₁ r₂ h₂) : 
     h₁ = 1.21 * h₂ :=
by
  unfold radius_relationship at h_radius
  unfold volume_equal at h_volume
  sorry

end cylinder_height_relationship_l747_747168


namespace egg_difference_l747_747576

theorem egg_difference
    (total_eggs : ℕ := 24)
    (broken_eggs : ℕ := 3)
    (cracked_eggs : ℕ := 6)
    (perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs)
    : (perfect_eggs - cracked_eggs = 9) :=
begin
  sorry
end

end egg_difference_l747_747576


namespace probability_sum_is_10_l747_747508

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747508


namespace dice_sum_prob_10_l747_747463

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747463


namespace find_angle_A_l747_747874

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 2) (h3 : B = Real.pi / 4) : A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l747_747874


namespace prob1_prob2_max_area_prob3_circle_diameter_l747_747355

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def line_through_center (x y : ℝ) : Prop := x - y - 3 = 0
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Problem 1: Line passes through the center of the circle
theorem prob1 (x y : ℝ) : line_through_center x y ↔ circle_eq x y :=
sorry

-- Problem 2: Maximum area of triangle CAB
theorem prob2_max_area (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 0 ∨ m = -6) :=
sorry

-- Problem 3: Circle with diameter AB passes through origin
theorem prob3_circle_diameter (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 1 ∨ m = -4) :=
sorry

end prob1_prob2_max_area_prob3_circle_diameter_l747_747355


namespace discount_difference_l747_747720

theorem discount_difference:
  let price := 30
  let discount_flat := 5
  let discount_percent := 0.25
  let cost_first_flat_then_percent := (price - discount_flat) * (1 - discount_percent)
  let cost_first_percent_then_flat := (price * (1 - discount_percent)) - discount_flat
  in (cost_first_flat_then_percent - cost_first_percent_then_flat) * 100 = 125 := by
  let price := 30
  let discount_flat := 5
  let discount_percent := 0.25
  let cost_first_flat_then_percent := (price - discount_flat) * (1 - discount_percent)
  let cost_first_percent_then_flat := (price * (1 - discount_percent)) - discount_flat
  show (cost_first_flat_then_percent - cost_first_percent_then_flat) * 100 = 125 from sorry

end discount_difference_l747_747720


namespace wrong_answer_more_than_correct_l747_747514

theorem wrong_answer_more_than_correct (n : ℕ) (h_n192 : n = 192) :
  let correct := (5 * n) / 16,
      incorrect := (5 * n) / 6
  in incorrect - correct = 100 := by
  intros
  rw h_n192
  let correct := (5 * 192) / 16
  let incorrect := (5 * 192) / 6
  have h_correct : correct = 60 := by
    -- calculation
    sorry
  have h_incorrect : incorrect = 160 := by
    -- calculation
    sorry
  rw [h_correct, h_incorrect]
  exact rfl

end wrong_answer_more_than_correct_l747_747514


namespace smallest_solution_l747_747783

theorem smallest_solution (x : ℝ) (h₀ : floor x = 2 + 50 * (x - floor x)) (h₁ : 0 ≤ x - floor x ∧ x - floor x < 1) :
  x = 1.96 :=
sorry

end smallest_solution_l747_747783


namespace maximum_visibility_sum_l747_747586

theorem maximum_visibility_sum (X Y : ℕ) (h : X + 2 * Y = 30) :
  X * Y ≤ 112 :=
by
  sorry

end maximum_visibility_sum_l747_747586


namespace probability_sum_is_10_l747_747452

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747452


namespace stratified_sampling_groupD_l747_747228

-- Definitions for the conditions
def totalDistrictCount : ℕ := 38
def groupADistrictCount : ℕ := 4
def groupBDistrictCount : ℕ := 10
def groupCDistrictCount : ℕ := 16
def groupDDistrictCount : ℕ := 8
def numberOfCitiesToSelect : ℕ := 9

-- Define stratified sampling calculation with a floor function or rounding
noncomputable def numberSelectedFromGroupD : ℕ := (groupDDistrictCount * numberOfCitiesToSelect) / totalDistrictCount

-- The theorem to prove 
theorem stratified_sampling_groupD : numberSelectedFromGroupD = 2 := by
  sorry -- This is where the proof would go

end stratified_sampling_groupD_l747_747228


namespace single_solution_quadratic_l747_747118

theorem single_solution_quadratic (b : ℝ) (hb : b ≠ 0) (hdisc : (-24)^2 - 4 * b * 6 = 0) :
  ∃ x : ℝ, bx^2 - 24x + 6 = 0 ∧ x = 1/2 :=
by {
  sorry
}

end single_solution_quadratic_l747_747118


namespace fixed_point_is_P_parallel_lines_distance_l747_747836

noncomputable def fixed_point (k : ℝ) : ℝ × ℝ :=
  (-1, 2)

theorem fixed_point_is_P : fixed_point k = (-1, 2) :=
  sorry

noncomputable def distance_between_lines (l1 : ℝ → ℝ → Prop) (l2 : ℝ → ℝ → Prop) : ℝ :=
  real.abs ((-1*l2.point * l2.slope + l2.constant) / (real.sqrt (l2.slope^2 + 1))) 

def line1 (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1) + 2

def line2 (k : ℝ) (x y : ℝ) : Prop := 3*x - (k-2)*y + 5 = 0

theorem parallel_lines_distance (k : ℝ) (l1 l2 : ℝ → ℝ → Prop) (h1 : l1 = line1 k) (h2 : l2 = line2 (-1))
  (h_parallel : (k ≠ 3 ∧ k ≠ -1) → false) : distance_between_lines l1 l2 = (4 * real.sqrt 2) / 3 :=
  sorry

end fixed_point_is_P_parallel_lines_distance_l747_747836


namespace max_area_of_triangle_l747_747876

theorem max_area_of_triangle (a b c : ℝ) (C : ℝ) (hC : C = 60) (h : 3 * a * b = 25 - c^2) :
  let S := (sqrt 3 / 4) * a * b
  in S <= (25 * sqrt 3) / 16 :=
by
  sorry

end max_area_of_triangle_l747_747876


namespace savings_correct_l747_747143

-- Define the conditions
def in_store_price : ℝ := 320
def discount_rate : ℝ := 0.05
def monthly_payment : ℝ := 62
def monthly_payments : ℕ := 5
def shipping_handling : ℝ := 10

-- Prove that the savings from buying in-store is 16 dollars.
theorem savings_correct : 
  (monthly_payments * monthly_payment + shipping_handling) - (in_store_price * (1 - discount_rate)) = 16 := 
by
  sorry

end savings_correct_l747_747143


namespace distance_greater_than_school_l747_747135

-- Let d1, d2, and d3 be the distances given as the conditions
def distance_orchard_to_house : ℕ := 800
def distance_house_to_pharmacy : ℕ := 1300
def distance_pharmacy_to_school : ℕ := 1700

-- The total distance from orchard to pharmacy via the house
def total_distance_orchard_to_pharmacy : ℕ :=
  distance_orchard_to_house + distance_house_to_pharmacy

-- The difference between the total distance from orchard to pharmacy and the distance from pharmacy to school
def distance_difference : ℕ :=
  total_distance_orchard_to_pharmacy - distance_pharmacy_to_school

-- The theorem to prove
theorem distance_greater_than_school :
  distance_difference = 400 := sorry

end distance_greater_than_school_l747_747135


namespace area_of_triangle_pqr_zero_l747_747298

-- Conditions of the problem
def circle_center (x y r : ℝ) := ∃ l : ℝ, x = l ∧ y = r

theorem area_of_triangle_pqr_zero :
  ∀ (P Q R : ℝ × ℝ),
    (circle_center P.1 P.2 1) ∧
    (circle_center Q.1 Q.2 2) ∧
    (circle_center R.1 R.2 3) ∧
    (dist P Q = 3) ∧
    (dist Q R = 5) →
    triangle_area P Q R = 0 :=
by
  intros P Q R conditions
  sorry

-- Helper definitions
noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))) / 2

end area_of_triangle_pqr_zero_l747_747298


namespace soda_cost_proof_l747_747084

-- Define the main facts about the weeds
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32 / 2  -- Only half the weeds in the grass

-- Define the earning rate
def earning_per_weed : ℕ := 6

-- Define the total earnings and the remaining money conditions
def total_earnings : ℕ := (weeds_flower_bed + weeds_vegetable_patch + weeds_grass) * earning_per_weed
def remaining_money : ℕ := 147

-- Define the cost of the soda
def cost_of_soda : ℕ := total_earnings - remaining_money

-- Problem statement: Prove that the cost of the soda is 99 cents
theorem soda_cost_proof : cost_of_soda = 99 := by
  sorry

end soda_cost_proof_l747_747084


namespace negation_of_proposition_l747_747627

-- Condition: there exists x in ℝ such that 2^x ≤ 0.
def proposition_to_negate : Prop :=
  ∃ x : ℝ, 2^x ≤ 0

-- Statement: The negation of the proposition should be ∀ x in ℝ, 2^x > 0.
theorem negation_of_proposition :
  ¬proposition_to_negate ↔ ∀ x : ℝ, 2^x > 0 :=
by
  sorry

end negation_of_proposition_l747_747627


namespace age_of_youngest_person_l747_747770

-- Definitions
def group_size : ℕ := 10
def sums : list ℕ := [82, 83, 84, 85, 87, 89, 90, 90, 91, 92]

-- Theorem to prove the age of the youngest person
theorem age_of_youngest_person : 
  let total_sum := list.sum sums in
  let combined_age := total_sum / 9 in
  let youngest_age := combined_age - list.maximum sums.to_finset in
  youngest_age = 5 := 
by {
  sorry
}

end age_of_youngest_person_l747_747770


namespace dice_sum_prob_10_l747_747468

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747468


namespace solve_for_a_l747_747018

theorem solve_for_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 16 - 6 * a + a ^ 2) : 
  a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41 := by
  sorry

end solve_for_a_l747_747018


namespace largest_g_8_l747_747927

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_is_polynomial : ∃ m : ℕ, ∃ (b : fin (m + 1) → ℝ), ∀ k, 0 ≤ b k
axiom g_4 : g 4 = 8
axiom g_16 : g 16 = 512

theorem largest_g_8 : g 8 ≤ 64 :=
sorry

end largest_g_8_l747_747927


namespace total_stickers_l747_747109

def stickers_in_first_box : ℕ := 23
def stickers_in_second_box : ℕ := stickers_in_first_box + 12

theorem total_stickers :
  stickers_in_first_box + stickers_in_second_box = 58 := 
by
  sorry

end total_stickers_l747_747109


namespace area_of_quadrilateral_ADEC_l747_747039

-- Define the entities and conditions
variables {Point : Type} [metric_space Point]

noncomputable def square (x : ℝ) := x * x

structure Triangle (A B C : Point) : Prop :=
(rt_angle_c : angle (vector A C) (vector B C) = π / 2)
(ab : dist A B = 30)
(ac : dist A C = 18)

structure QuadsAreaProblem (A B C D E : Point) : Prop :=
(triangle_abc : Triangle A B C)
(midpoint_d : dist A D = dist D B)
(ortho_de : ∠D E A = ∠D E B)

-- Problem statement: Prove the area of quadrilateral ADEC equals 131.625
theorem area_of_quadrilateral_ADEC (A B C D E : Point) 
  [QuadsAreaProblem A B C D E] : 
  area_eq (quadrilateral A D E C) 131.625 :=
sorry

end area_of_quadrilateral_ADEC_l747_747039


namespace valid_integer_values_of_x_l747_747689

theorem valid_integer_values_of_x (x : ℤ) 
  (h1 : 3 < x) (h2 : x < 10)
  (h3 : 5 < x) (h4 : x < 18)
  (h5 : -2 < x) (h6 : x < 9)
  (h7 : 0 < x) (h8 : x < 8) 
  (h9 : x + 1 < 9) : x = 6 ∨ x = 7 :=
by
  sorry

end valid_integer_values_of_x_l747_747689


namespace reflection_correct_l747_747132

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def M : point := (3, 2)

theorem reflection_correct : reflect_x_axis M = (3, -2) :=
  sorry

end reflection_correct_l747_747132


namespace leading_digit_neq_one_a_leading_digit_neq_one_b_l747_747699

theorem leading_digit_neq_one_a (x : ℕ) (hx1 : (∀ y ∈ [x, x^2, x^3], digit y = digit x)) : 
    digit x ≠ 1 ∧ digit x ≠ 9 := 
sorry

theorem leading_digit_neq_one_b (x : ℕ) (hx2 : ∀ n: ℕ, 1 ≤ n ∧ n ≤ 2015 → digit (x^n) = digit x): 
    digit x ≠ 1 ∧ digit x ≠ 9 := 
sorry

end leading_digit_neq_one_a_leading_digit_neq_one_b_l747_747699


namespace number_of_possible_triangle_areas_l747_747764

-- Define the problem setup
noncomputable def points_on_line (dist : ℝ) : Prop :=
  ∀ A B C D : ℝ, A < B ∧ B < C ∧ C < D ∧ B - A = dist ∧ C - B = dist ∧ D - C = dist

noncomputable def points_on_parallel_line (dist : ℝ) : Prop :=
  ∀ E F G : ℝ, E < F ∧ F < G ∧ F - E = dist ∧ G - F = dist

noncomputable def constant_distance_between_lines (height : ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ {A, B, C, D} ∧ y ∈ {E, F, G} → dist_between x y = height

def possible_triangle_areas (height : ℝ) (dist : ℝ) : ℕ :=
  let bases := {1, 2, 3}
  let areas := bases.image (λ b, (1 / 2) * b * height)
  areas.size

-- Statement of the theorem
theorem number_of_possible_triangle_areas
  (h : ℝ) (d : ℝ)
  (h_points : points_on_line d)
  (h_parallel_points : points_on_parallel_line d)
  (h_height : constant_distance_between_lines h) :
  possible_triangle_areas h d = 3 :=
  sorry

end number_of_possible_triangle_areas_l747_747764


namespace cubing_identity_l747_747410

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l747_747410


namespace proposition_check_l747_747253

variable (P : ℕ → Prop)

theorem proposition_check 
  (h : ∀ k : ℕ, ¬ P (k + 1) → ¬ P k)
  (h2012 : P 2012) : P 2013 :=
by
  sorry

end proposition_check_l747_747253


namespace polynomial_bound_exists_l747_747950

theorem polynomial_bound_exists (f : ℂ → ℂ) (n : ℕ) (hn : n ≥ 1) (h_deg : degree f = n) (M : ℝ) (hM : M > 0) :
  ∃ R : ℝ, R > 0 ∧ ∀ z : ℂ, |z| > R → |f z| ≥ M :=
sorry

end polynomial_bound_exists_l747_747950


namespace rowing_time_ratio_l747_747152

theorem rowing_time_ratio
  (V_b : ℝ) (V_s : ℝ) (V_upstream : ℝ) (V_downstream : ℝ) (T_upstream T_downstream : ℝ)
  (h1 : V_b = 39) (h2 : V_s = 13)
  (h3 : V_upstream = V_b - V_s) (h4 : V_downstream = V_b + V_s)
  (h5 : T_upstream * V_upstream = T_downstream * V_downstream) :
  T_upstream / T_downstream = 2 := by
  sorry

end rowing_time_ratio_l747_747152


namespace dice_sum_10_probability_l747_747425

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747425


namespace minimum_colors_l747_747174

theorem minimum_colors (lines : Finset ℝ)
  (h_distinct : no_two_parallel lines)
  (h_no_concurrent: no_three_concurrent lines)
  : ∃ (colors : ℕ), (∀ (p1 p2 : Point), p1 != p2 ∧ on_same_line_no_other_between p1 p2 → colors p1 ≠ colors p2) ∧ colors ≤ 3 :=
sorry

end minimum_colors_l747_747174


namespace real_function_as_sum_of_symmetric_graphs_l747_747102

theorem real_function_as_sum_of_symmetric_graphs (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ), (∀ x, g x + h x = f x) ∧ (∀ x, g x = g (-x)) ∧ (∀ x, h (1 + x) = h (1 - x)) :=
sorry

end real_function_as_sum_of_symmetric_graphs_l747_747102


namespace count_integers_with_permutation_multiple_of_11_l747_747400

theorem count_integers_with_permutation_multiple_of_11 : 
  ∃ (n : ℕ), n = 226 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 → 
    (∃ (y : ℕ), (permutes_digits x y) ∧ (100 ≤ y) ∧ (y ≤ 999) ∧ y % 11 = 0 
    → x ∈ {100, ..., 999}) :=
by
  sorry

/--
Helper function to check if two numbers are permutations of each other.
This is for demonstration purposes as auxiliary functions are generally required to perform the check.
-/
def permutes_digits (a b : ℕ) := sorry


end count_integers_with_permutation_multiple_of_11_l747_747400


namespace exactly_one_box_empty_count_l747_747348

-- Define the setting with four different balls and four boxes.
def numberOfWaysExactlyOneBoxEmpty (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  if (balls.card = 4 ∧ boxes.card = 4) then
     Nat.choose 4 2 * Nat.factorial 3
  else 0

theorem exactly_one_box_empty_count :
  numberOfWaysExactlyOneBoxEmpty {1, 2, 3, 4} {1, 2, 3, 4} = 144 :=
by
  -- The proof is omitted
  sorry

end exactly_one_box_empty_count_l747_747348


namespace other_factor_computation_l747_747229

theorem other_factor_computation (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
  a = 11 → b = 43 → c = 2 → d = 31 → e = 1311 → 33 ∣ 363 →
  a * b * c * d * e = 38428986 :=
by
  intros ha hb hc hd he hdiv
  rw [ha, hb, hc, hd, he]
  -- proof steps go here if required
  sorry

end other_factor_computation_l747_747229


namespace four_digit_numbers_divisible_by_5_l747_747110

theorem four_digit_numbers_divisible_by_5 :
  let odd_digits := {1, 3, 5, 7}
  let even_digits := {0, 2, 4, 6, 8}
  ∃(n : ℕ), n = 300 ∧ 
    (∀ digits : finset ℕ,
      digits.card = 4 ∧ 
      (∃ d1 d2 : ℕ, d1 ∈ odd_digits ∧ d2 ∈ odd_digits ∧ d1 ≠ d2) ∧ 
      (∃ e1 e2 : ℕ, e1 ∈ even_digits ∧ e2 ∈ even_digits ∧ e1 ≠ e2) ∧ 
      (digits ∩ odd_digits).card = 2 ∧
      (digits ∩ even_digits).card = 2 ∧
      (digits.to_list.nth 0 * 1000 + digits.to_list.nth 1 * 100 + digits.to_list.nth 2 * 10 + digits.to_list.nth 3) % 5 = 0) → n = 300 :=
by
  sorry

end four_digit_numbers_divisible_by_5_l747_747110


namespace ellipse_standard_equation_l747_747785

theorem ellipse_standard_equation :
  ∃ a b : ℝ, ∃ (h1 : a > b), ∃ (h2 : b > 0), 
  let c := 2 in
  let focal_distance := 2 * c in
  focal_distance = 4 ∧
  let M := (3, 2 : ℝ × ℝ) in
  let d1 := (M.1^2 + (M.2 + c)^2).sqrt in
  let d2 := (M.1^2 + (M.2 - c)^2).sqrt in
  let a := (d1 + d2) / 2 in
  let b_squared := a^2 - c^2 in
  b_squared = 12 ∧
  a = 4 ∧
  (b_squared).sqrt = b ∧
  ∀ x y : ℝ, ((y^2 / 16) + (x^2 / (b^2)) = 1) :=
sorry

end ellipse_standard_equation_l747_747785


namespace transform_polynomial_l747_747859

variable (x z : ℝ)

theorem transform_polynomial (h1 : z = x - 1 / x) (h2 : x^4 - 3 * x^3 - 2 * x^2 + 3 * x + 1 = 0) :
  x^2 * (z^2 - 3 * z) = 0 :=
sorry

end transform_polynomial_l747_747859


namespace probability_p_eq_l747_747210

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l747_747210


namespace volume_of_pond_l747_747517

theorem volume_of_pond (l w h : ℕ) (hl : l = 20) (hw : w = 10) (hh : h = 5) :
  l * w * h = 1000 :=
by
  rw [hl, hw, hh]
  rfl

end volume_of_pond_l747_747517


namespace omega_range_l747_747621

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + π / 3)

def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + 5 * π / 6)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → Real.sin (ω * x + 5 * π / 6) ≤ Real.sin (ω * (x + ε) + 5 * π / 6)) →
  ω ≤ 4 / 3 := sorry

end omega_range_l747_747621


namespace correct_option_C_l747_747185

variable (x : ℝ)
variable (hx : 0 < x ∧ x < 1)

theorem correct_option_C : 0 < 1 - x^2 ∧ 1 - x^2 < 1 :=
by
  sorry

end correct_option_C_l747_747185


namespace probability_sum_of_5_l747_747170

noncomputable def tetrahedral_dice_probability : ℚ :=
  let outcomes := [(1, 4), (2, 3), (3, 2), (4, 1)]
  let favorable_outcomes := 4 * 2
  let total_outcomes := 4 * 4
  favorable_outcomes / total_outcomes

theorem probability_sum_of_5 : tetrahedral_dice_probability = 1 / 2 := by
  sorry

end probability_sum_of_5_l747_747170


namespace red_blue_tile_difference_is_15_l747_747092

def num_blue_tiles : ℕ := 17
def num_red_tiles_initial : ℕ := 8
def additional_red_tiles : ℕ := 24
def num_red_tiles_new : ℕ := num_red_tiles_initial + additional_red_tiles
def tile_difference : ℕ := num_red_tiles_new - num_blue_tiles

theorem red_blue_tile_difference_is_15 : tile_difference = 15 :=
by
  sorry

end red_blue_tile_difference_is_15_l747_747092


namespace evaluate_expression_l747_747323

theorem evaluate_expression : 
  (3 / 20 - 5 / 200 + 7 / 2000 : ℚ) = 0.1285 :=
by
  sorry

end evaluate_expression_l747_747323


namespace probability_product_multiple_of_4_l747_747188

theorem probability_product_multiple_of_4 :
  (∃ (cards : Finset ℕ) (h : cards = {1, 2, 3, 4, 5, 6}) (drawn : Finset (ℕ × ℕ))
     (h2 : drawn = {⟨1,2⟩, ⟨1,3⟩, ⟨1,4⟩, ⟨1,5⟩, ⟨1,6⟩,
                    ⟨2,3⟩, ⟨2,4⟩, ⟨2,5⟩, ⟨2,6⟩,
                    ⟨3,4⟩, ⟨3,5⟩, ⟨3,6⟩,
                    ⟨4,5⟩, ⟨4,6⟩, ⟨5,6⟩}),
   filter (λ (pair : ℕ × ℕ), (pair.fst * pair.snd) % 4 = 0) drawn).card / drawn.card = 2 / 5 :=
sorry

end probability_product_multiple_of_4_l747_747188


namespace dice_sum_probability_l747_747447

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747447


namespace smallest_bob_number_l747_747272

theorem smallest_bob_number (n m : ℕ) (h_n : n = 36)
    (h_factors : ∀ p : ℕ, prime p → p ∣ n → p ∣ m) : m = 6 :=
sorry

end smallest_bob_number_l747_747272


namespace sum_pow2_xn_eq_neg1989_l747_747640

noncomputable def x : ℕ → ℝ
| 0       := 1989
| (n + 1) := -1989 / (n + 1) * (List.range (n + 1)).sum (λ k => x k)

theorem sum_pow2_xn_eq_neg1989 :
  ∑ n in Finset.range 1989, (2:ℝ)^n * x n = -1989 :=
by
  sorry

end sum_pow2_xn_eq_neg1989_l747_747640


namespace best_fitting_model_l747_747518

/-- A type representing the coefficient of determination of different models -/
def r_squared (m : ℕ) : ℝ :=
  match m with
  | 1 => 0.98
  | 2 => 0.80
  | 3 => 0.50
  | 4 => 0.25
  | _ => 0 -- An auxiliary value for invalid model numbers

/-- The best fitting model is the one with the highest r_squared value --/
theorem best_fitting_model : r_squared 1 = max (r_squared 1) (max (r_squared 2) (max (r_squared 3) (r_squared 4))) :=
by
  sorry

end best_fitting_model_l747_747518


namespace sym_coords_origin_l747_747860

theorem sym_coords_origin (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) :
  (-a, -b) = (-3, 4) :=
sorry

end sym_coords_origin_l747_747860


namespace min_distance_eq_sqrt_two_l747_747404

open Real

def line_eq (x : ℝ) : ℝ := 1 - x
def curve_eq (x : ℝ) : ℝ := -exp x

theorem min_distance_eq_sqrt_two :
  let min_distance : ℝ := (2 : ℝ)/sqrt 2
  min_distance = sqrt 2 :=
by
  sorry

end min_distance_eq_sqrt_two_l747_747404


namespace contradiction_proof_AssumptionA_l747_747966

theorem contradiction_proof_AssumptionA
  (a b c : ℕ) :
  (¬(∃! x, (x = a ∨ x = b ∨ x = c) ∧ ¬even x) ↔ (¬(even a ∧ even b ∧ ¬even c) 
  ∨ ¬(¬even a ∧ even b ∧ even c) ∨ ¬(even a ∧ ¬even b ∧ even c))) :=
by sorry

end contradiction_proof_AssumptionA_l747_747966


namespace fraction_of_money_left_l747_747747

theorem fraction_of_money_left (m : ℝ) (b : ℝ) (h1 : (1 / 4) * m = (1 / 2) * b) :
  m - b - 50 = m / 2 - 50 → (m - b - 50) / m = 1 / 2 - 50 / m :=
by sorry

end fraction_of_money_left_l747_747747


namespace product_of_area_and_perimeter_l747_747091

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let x1 := p1.1
  let y1 := p1.2
  let x2 := p2.1
  let y2 := p2.2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def vertex_A : ℝ × ℝ := (5,5)
def vertex_B : ℝ × ℝ := (2,6)
def vertex_C : ℝ × ℝ := (1,3)
def vertex_D : ℝ × ℝ := (4,2)

def side_length : ℝ := distance vertex_A vertex_D
def area : ℝ := side_length^2
def perimeter : ℝ := 4 * side_length

theorem product_of_area_and_perimeter :
  area * perimeter = 40 * real.sqrt 10 :=
by
  unfold area perimeter side_length distance
  sorry

end product_of_area_and_perimeter_l747_747091


namespace product_is_48_l747_747989

-- Define the conditions and the target product
def problem (x y : ℝ) := 
  x ≠ y ∧ (x + y) / (x - y) = 7 ∧ (x * y) / (x - y) = 24

-- Prove that the product is 48 given the conditions
theorem product_is_48 (x y : ℝ) (h : problem x y) : x * y = 48 :=
sorry

end product_is_48_l747_747989


namespace sum_of_squares_l747_747923

def integer_part (x : ℝ) : ℤ := ⌊x⌋ -- Greatest integer less than or equal to x
def fractional_part (x : ℝ) : ℝ := x - (integer_part x) -- Fractional part of x

theorem sum_of_squares :
  let condition (m : ℕ) := integer_part ((2 * (m:ℤ) + 1) * fractional_part (Real.sqrt (2 * (m:ℤ) + 1))) = m
  ∑ m in { m : ℕ | condition m }, (m ^ 2) = 0 :=
by
  sorry

end sum_of_squares_l747_747923


namespace measure_arc_BC_l747_747024

-- Given conditions
variables {O A B C : Point} -- points on the circle
variable  {circle : Circle} -- the circle with center O
variable  (h1 : circle.center = O) -- O is the center
variable  (h2 : ∠ BAC = 60) -- angle BAC is 60 degrees
variable  (h3 : A B C ∈ circle) -- A, B, and C are on the circle
variable  (h4 : diameter A C circle AC) -- AC is the diameter

-- Proof statement
theorem measure_arc_BC (circle : Circle) (O A B C : Point)
  (h1 : circle.center = O)
  (h2 : ∠ BAC = 60)
  (h3 : A B C ∈ circle)
  (h4 : diameter A C circle AC) : measure_arc B C = 120 :=
by
  sorry

end measure_arc_BC_l747_747024


namespace parallelogram_BD_l747_747963

open Real

theorem parallelogram_BD (AB CD BC AD CE: ℝ)
  (parallelogram_ABCD: parallelogram ABCD)
  (h1: AB = 6)
  (h2: CD = 6)
  (h3: BC = 10)
  (h4: AD = 10)
  (h5: CE = 4)
  (angle_ABC_obtuse: ∠ ABC > π / 2) 
  (circumcircle_triangle_ABD_intersects_BC_at_E : IsOnCircumcircle ⟨B, A, D⟩ E ∧ CE = 4) :
  BD = 4 * sqrt 6 :=
by
  sorry

end parallelogram_BD_l747_747963


namespace dice_sum_probability_l747_747498

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747498


namespace prob_intersection_l747_747012

variable {Ω : Type} [probability_space Ω]

variables (E F : event Ω)
hypothesis h : independent E F
hypothesis hE : Pr[E] = 1 / 4
hypothesis hF : Pr[F] = 1 / 4

theorem prob_intersection : Pr[E ∩ F] = 1 / 16 := by
  sorry

end prob_intersection_l747_747012


namespace log_inequality_iff_l747_747379

variables {a b : ℝ}

theorem log_inequality_iff (ha : a > 1) (hb : b > 1) : 
  a > b ↔ log b a > log a b :=
sorry

end log_inequality_iff_l747_747379


namespace number_of_groups_l747_747956

-- Defining the problem conditions
def total_popsicle_sticks : ℕ := 170
def remaining_popsicle_sticks : ℕ := 20
def popsicle_sticks_per_group : ℕ := 15

-- Proving the number of groups
theorem number_of_groups 
  (total_popsicle_sticks = 170) 
  (remaining_popsicle_sticks = 20) 
  (popsicle_sticks_per_group = 15) : 
  (170 - 20) / 15 = 10 :=
  by sorry

end number_of_groups_l747_747956


namespace number_of_paths_l747_747993

theorem number_of_paths (n : ℕ) (h1 : n > 3) : 
  (2 * (8 * n^3 - 48 * n^2 + 88 * n - 48) + (4 * n^2 - 12 * n + 8) + (2 * n - 2)) = 16 * n^3 - 92 * n^2 + 166 * n - 90 :=
by
  sorry

end number_of_paths_l747_747993


namespace maximum_rooks_l747_747582

-- Define the chessboard size
def chessboard_size : ℕ := 8

-- Define the condition that no rook can attack another rook of different color
def no_attacking_condition (n : ℕ) (white_positions black_positions : list (ℕ × ℕ)) : Prop :=
  n = white_positions.length ∧ n = black_positions.length ∧ (∀ (x : ℕ × ℕ) (y : ℕ × ℕ),
    (x ∈ white_positions → y ∈ black_positions → x.fst ≠ y.fst ∧ x.snd ≠ y.snd))

-- Define the maximum possible value of n
theorem maximum_rooks (n : ℕ) (white_positions black_positions : list (ℕ × ℕ)) :
  no_attacking_condition n white_positions black_positions →
  n ≤ 16 :=
by
  intros h
  sorry

end maximum_rooks_l747_747582


namespace cubing_identity_l747_747409

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l747_747409


namespace meatballs_left_after_eating_l747_747648

def uneaten_meatballs (m_per_plate : ℕ) (sons : ℕ) (fraction_eaten : ℚ) : ℕ :=
  let fraction_uneaten := 1 - fraction_eaten
  let uneaten_per_plate := (fraction_uneaten * m_per_plate)
  let total_uneaten := uneaten_per_plate * sons
  total_uneaten.to_nat

theorem meatballs_left_after_eating :
  uneaten_meatballs 3 3 (2/3) = 3 :=
by
  sorry

end meatballs_left_after_eating_l747_747648


namespace hyperbola_standard_eq_l747_747381

theorem hyperbola_standard_eq (a b : ℝ) (x y : ℝ)
  (h : x = 2 ∧ y = 1 ∧ 
       a^2 = 3 ∧ 
       (x - sqrt 3)^2 + b^2 = 1 ∧ 
       (x + sqrt 3)^2 + b^2 = 1) :
  (x, y) ∈ hyperbola (sqrt 2, b) :=
by
  sorry

end hyperbola_standard_eq_l747_747381


namespace sequence_formula_l747_747642

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ)
  (h_seq : ∀ n : ℕ+, S n = 2 * n - a n) :
  ∀ n : ℕ+, a n = (2^n - 1) / 2^(n - 1) := 
by
  sorry

end sequence_formula_l747_747642


namespace dice_sum_probability_l747_747442

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747442


namespace part_a_l747_747700

theorem part_a (x : ℝ) (hx : x ≥ 1) : x^3 - 5 * x^2 + 8 * x - 4 ≥ 0 := 
  sorry

end part_a_l747_747700


namespace quadratic_root_condition_l747_747873

theorem quadratic_root_condition (k : ℝ) :
  (∀ (x : ℝ), x^2 + k * x + 4 * k^2 - 3 = 0 → ∃ x1 x2 : ℝ, x1 + x2 = (-k) ∧ x1 * x2 = 4 * k^2 - 3 ∧ x1 + x2 = x1 * x2) →
  k = 3 / 4 :=
by
  sorry

end quadratic_root_condition_l747_747873


namespace necessary_but_not_sufficient_l747_747791

theorem necessary_but_not_sufficient (a b c : ℝ) (h : a > b) : 
  (a > b → ac^2 > bc^2) ∧ ¬(ac^2 > bc^2 → a > b) := 
sorry

end necessary_but_not_sufficient_l747_747791


namespace probability_sum_10_l747_747486

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747486


namespace probability_heads_l747_747206

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l747_747206


namespace dice_sum_probability_l747_747492

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747492


namespace egg_difference_l747_747575

theorem egg_difference
    (total_eggs : ℕ := 24)
    (broken_eggs : ℕ := 3)
    (cracked_eggs : ℕ := 6)
    (perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs)
    : (perfect_eggs - cracked_eggs = 9) :=
begin
  sorry
end

end egg_difference_l747_747575


namespace problem_solution_l747_747521

-- Define the conditions of the problem
def parametric_eq_curve (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, Real.sin α)

def polar_eq_line (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2

-- Define the proof problem
theorem problem_solution : 
  (∀ α : ℝ, let (x, y) := parametric_eq_curve α in (x^2 / 9 + y^2 = 1)) ∧
  (∀ (ρ θ : ℝ), (polar_eq_line ρ θ → (y = x + 2) ∧ (θ = Real.pi / 4))) ∧
  (∃ (P : ℝ × ℝ) (A B : ℝ × ℝ),
    P = (0, 2) ∧
    (A = parametric_eq_curve (arccos (A.1 / 3))) ∧ 
    (A.2 = sqrt (1 - (A.1 / 3)^2)) ∧
    (B = parametric_eq_curve (arccos (B.1 / 3))) ∧ 
    (B.2 = sqrt (1 - (B.1 / 3)^2)) ∧
    |PA| + |PB| = 18 / 5 * Real.sqrt 2 := 
    sorry -- Proof omitted

end problem_solution_l747_747521


namespace find_highest_m_l747_747242

def reverse_num (m : ℕ) : ℕ :=
  let digits := m.toString.data.toList.reverse
  digits.foldl (λ acc c => acc * 10 + c.toNat - '0'.toNat) 0

theorem find_highest_m (m : ℕ) :
  (1000 ≤ m ∧ m ≤ 9999) ∧
  (1000 ≤ reverse_num m ∧ reverse_num m ≤ 9999) ∧
  (m % 36 = 0 ∧ reverse_num m % 36 = 0) ∧
  (m % 7 = 0) →
  m = 5796 :=
by
  sorry

end find_highest_m_l747_747242


namespace total_lambs_l747_747087

-- Defining constants
def Merry_lambs : ℕ := 10
def Brother_lambs : ℕ := Merry_lambs + 3

-- Proving the total number of lambs
theorem total_lambs : Merry_lambs + Brother_lambs = 23 :=
  by
    -- The actual proof is omitted and a placeholder is put instead
    sorry

end total_lambs_l747_747087


namespace frame_interior_edge_sum_l747_747255

theorem frame_interior_edge_sum :
  ∀ (w h: ℝ), w = 4 → h = 6 → 2 * (w + h) = 20 :=
by
  intros w h hw hh
  rw [hw, hh]
  rw [two_mul, add_mul, one_mul]
  norm_num
  sorry -- proof skipped

end frame_interior_edge_sum_l747_747255


namespace largest_binomial_coefficient_of_six_inequality_of_derivative_existence_of_a_l747_747561

noncomputable def f (x : ℕ) (n : ℕ) := (1 + (1 / n))^x

-- Problem 1: Verify the term with the largest binomial coefficient
theorem largest_binomial_coefficient_of_six (n : ℕ) (hn : n > 1) : 
  let b := (nat.choose 6 3) * (1 / n)^3 in
  b = 20 / n^3 := sorry

-- Problem 2: Prove the given inequality involving the derivative of f
theorem inequality_of_derivative (x : ℝ) (n : ℕ) (hn : n > 1) :
  let f_real (x : ℝ) := (1 + (1 / (n : ℝ)))^x in
  (f_real (2 * x) + f_real 2) / 2 > deriv f_real x := sorry

-- Problem 3: Prove the existence of an appropriate a in the inequality
theorem existence_of_a (n : ℕ) (hn : n > 1) :
  ∃ (a : ℕ), a = 2 ∧ ∀ k, 1 ≤ k → k ≤ n → k * 2 < ∑ i in finset.range (n + 1), (1 + (1 / (i + 1))) ∧ 
                                 ∑ i in finset.range (n + 1), (1 + (1 / (i + 1))) < k * 3 := sorry

end largest_binomial_coefficient_of_six_inequality_of_derivative_existence_of_a_l747_747561


namespace area_closed_figure_l747_747419

theorem area_closed_figure 
  (k : ℝ) 
  (hk : k = (∑ r in finset.range 4, nat.choose 3 r * (1 : ℝ)^r * (1 : ℝ)^3 / x^r * x^(2*r)) |
  ∑ r in finset.range 4, nat.choose 3 r * (1 : ℝ)^(3-r)) = 3) :
  ((∫ x in 0..3, (3 * x - x^2)) = 9 / 2) :=
sorry

end area_closed_figure_l747_747419


namespace percentage_markup_l747_747632

theorem percentage_markup (sell_price : ℝ) (cost_price : ℝ)
  (h_sell : sell_price = 8450) (h_cost : cost_price = 6500) : 
  (sell_price - cost_price) / cost_price * 100 = 30 :=
by
  sorry

end percentage_markup_l747_747632


namespace find_c_in_triangle_l747_747535

theorem find_c_in_triangle (A : ℝ) (b : ℝ) (S : ℝ) (h1 : A = 60) (h2 : b = 16) (h3 : S = 220 * sqrt 3) : 
  let c := 55 
  in S = (1 / 2) * b * c * real.sin A → c = 55 := 
by sorry

end find_c_in_triangle_l747_747535


namespace simplify_expression_evaluate_expression_when_reciprocal_l747_747115

variable (a b : ℝ)

theorem simplify_expression :
  -( -a^2 + 2 * a * b + b^2 ) + ( -a^2 - a * b + b^2 ) = -3 * a * b := by
  sorry

theorem evaluate_expression_when_reciprocal :
  a * b = 1 → -( -a^2 + 2 * a * b + b^2 ) + ( -a^2 - a * b + b^2 ) = -3 := by
  intro h
  rw [simplify_expression]
  rw [h]
  norm_num

end simplify_expression_evaluate_expression_when_reciprocal_l747_747115


namespace probability_sector_F_l747_747233

theorem probability_sector_F (prob_D prob_E prob_F : ℚ)
    (hD : prob_D = 1/4) 
    (hE : prob_E = 1/3) 
    (hSum : prob_D + prob_E + prob_F = 1) :
    prob_F = 5/12 := by
  sorry

end probability_sector_F_l747_747233


namespace simplify_expression_l747_747350

variable (a : ℝ)

theorem simplify_expression (h : 2 < a ∧ a < 3) : (sqrt ((a - π)^2) + abs (a - 2) = π - 2) :=
  sorry

end simplify_expression_l747_747350


namespace ring_matching_possible_iff_odd_l747_747702

theorem ring_matching_possible_iff_odd (n : ℕ) (hn : n ≥ 3) :
  (∃ f : ℕ → ℕ, (∀ k : ℕ, k < n → ∃ j : ℕ, j < n ∧ f (j + k) % n = k % n) ↔ Odd n) :=
sorry

end ring_matching_possible_iff_odd_l747_747702


namespace percentage_of_y_l747_747639

theorem percentage_of_y (x y : ℝ) (h1 : x = 4 * y) (h2 : 0.80 * x = (P / 100) * y) : P = 320 :=
by
  -- Proof goes here
  sorry

end percentage_of_y_l747_747639


namespace min_ratio_at_6_l747_747809

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0     => 33
  | (n+1) => sequence n + 2 * n

def ratio (n : ℕ) : ℚ :=
  (sequence n : ℚ) / n

theorem min_ratio_at_6 (n : ℕ) (h1 : 1 ≤ n) (h2 : ∀ k, 1 ≤ k → k ≠ n → ratio k ≥ ratio 6) :
  ratio n = 21 / 2 :=
sorry

end min_ratio_at_6_l747_747809


namespace probability_sum_is_10_l747_747458

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747458


namespace quotient_in_first_division_l747_747584

theorem quotient_in_first_division (N Q Q' : ℕ) (h₁ : N = 68 * Q) (h₂ : N % 67 = 1) : Q = 1 :=
by
  -- rest of the proof goes here
  sorry

end quotient_in_first_division_l747_747584


namespace decreasing_interval_f_l747_747999

noncomputable def f (x : ℝ) := √2 * Real.sin (x + π / 4)

def I := Set.Icc (π / 4) π

theorem decreasing_interval_f :
  ∀ x ∈ Set.Icc 0 π, f x ∈ Set.Icc (π / 4) π →  f x ≤ f (x + d) :=
  sorry

end decreasing_interval_f_l747_747999


namespace largest_integral_value_of_k_l747_747668

theorem largest_integral_value_of_k : ∃ k : ℤ, k <= 8 ∧ ∀ m : ℤ, m < 9 → m <= k :=
by
  have h_disc : ∀ k : ℝ, let Δ := (-6)^2 - 4 * (1) * k in Δ > 0 ↔ k < 9 :=
    λ k, by { simp only [pow_two, neg_mul],
              have : 36 - 4 * k > 0 ↔ k < 9, from by linarith,
              exact this, }
  existsi (8 : ℤ)
  split
  { trivially }
  { intros m hm
    have : (m : ℝ) < 9 := by exact_mod_cast hm
    linarith }

end largest_integral_value_of_k_l747_747668


namespace probability_heads_l747_747203

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l747_747203


namespace tea_bags_total_l747_747961

theorem tea_bags_total 
(tea_bag_cups : ℕ → Prop)
(divided_equally : ∀ n, (∃ x, ((x * 2) = n ∨ (x * 3) = n)) ∧ (∀ n, (∃ x, ((x * 2) = n ∨ (x * 3) = n))))
(mila_brewed : 57)
(tanya_brewed : 83)
(mila_tea_bags_range : 19 ≤ mila_bags ∧ mila_bags ≤ 28)
(tanya_tea_bags_range : 28 ≤ tanya_bags ∧ tanya_bags ≤ 41)
(equal_tea_bags : mila_bags = tanya_bags):
(n : ℕ) :
(mila_bags + tanya_bags) = 56 :=
by
    sorry

end tea_bags_total_l747_747961


namespace function_relationship_area_60_maximum_area_l747_747171

-- Definitions and conditions
def perimeter := 32
def side_length (x : ℝ) : ℝ := 16 - x  -- One side of the rectangle
def area (x : ℝ) : ℝ := x * (16 - x)

-- Theorem 1: Function relationship between y and x
theorem function_relationship (x : ℝ) (hx : 0 < x ∧ x < 16) : area x = -x^2 + 16 * x :=
by
  sorry

-- Theorem 2: Values of x when the area is 60 square meters
theorem area_60 (x : ℝ) (hx1 : area x = 60) : x = 6 ∨ x = 10 :=
by
  sorry

-- Theorem 3: Maximum area
theorem maximum_area : ∃ x, area x = 64 ∧ x = 8 :=
by
  sorry

end function_relationship_area_60_maximum_area_l747_747171


namespace propositions_correct_l747_747680

theorem propositions_correct :
  (∃ x : ℝ, x < 0 ∧ x^2 - 2 * x - 3 = 0) ∧
  (¬ ∀ x : ℝ, sqrt (x^2) = x) ∧
  (¬ ∅ ∈ ({0} : set ℕ)) ∧
  ({x : ℤ | ∃ n : ℤ, x = 2 * n - 1} = {x : ℤ | ∃ n : ℤ, x = 2 * n + 1}) :=
by
  sorry

end propositions_correct_l747_747680


namespace b_is_geometric_l747_747361

variable (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) (h_q : q ≠ 1)
variable (h_geometric : ∀ n : ℕ, a n = a1 * q ^ (n-1))

def b (n : ℕ) : ℝ :=
  a (3*n-2) + a (3*n-1) + a (3*n)

theorem b_is_geometric : (h_q : q ≠ 1) →
  (q ≠ 0) → (∀ n : ℕ, a n = a1 * q ^ (n-1)) →
  ∀ n : ℕ, b n = a1 * q ^ (3*n-3) * (1 + q + q^2) ∧ b (n+1) = (b n) * q^3 := by
  intros
  sorry

end b_is_geometric_l747_747361


namespace tan_triple_angle_l747_747006

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l747_747006


namespace probability_heads_equals_l747_747211

theorem probability_heads_equals (p q: ℚ) (h1 : q = 1 - p) (h2 : (binomial 10 5) * p^5 * q^5 = (binomial 10 6) * p^6 * q^4) : p = 6 / 11 :=
by {
  sorry
}

end probability_heads_equals_l747_747211


namespace probability_sum_is_ten_l747_747441

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747441


namespace resulting_expression_l747_747563

def x : ℕ := 1000
def y : ℕ := 10

theorem resulting_expression : 
  (x + 2 * y) + x + 3 * y + x + 4 * y + x + y = 4 * x + 10 * y :=
by
  sorry

end resulting_expression_l747_747563


namespace find_x_l747_747416

theorem find_x (x : ℝ) (h : 1 - 1 / (1 - x) = 1 / (1 - x)) : x = -1 :=
by
  sorry

end find_x_l747_747416


namespace unique_real_root_iff_a_eq_3_l747_747346

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

theorem unique_real_root_iff_a_eq_3 {a : ℝ} (hu : ∃! x : ℝ, f x a = 0) : a = 3 :=
sorry

end unique_real_root_iff_a_eq_3_l747_747346


namespace sum_of_angles_l747_747336

theorem sum_of_angles (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * π) 
    (h_eq : sin x ^ 4 - cos x ^ 4 = 1 / cos x - 1 / sin x) : 
    ∑ solutions = 3 * π / 2 := 
sorry

end sum_of_angles_l747_747336


namespace find_B_l747_747038

noncomputable def A : ℂ := 3 + complex.i

noncomputable def vector_AC : ℂ := -2 - 4 * complex.i

noncomputable def vector_BC : ℂ := -4 - complex.i

noncomputable def C : ℂ := A + vector_AC

noncomputable def B : ℂ := C + vector_BC

theorem find_B : B = 5 - 2 * complex.i := by
  unfold B
  unfold C
  unfold A
  unfold vector_AC
  unfold vector_BC
  sorry

end find_B_l747_747038


namespace dice_sum_prob_10_l747_747469

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747469


namespace largest_internal_external_sphere_volume_l747_747669

noncomputable def unit_sphere_volume : ℝ := 4 / 3 * π * (1/3)^3

theorem largest_internal_external_sphere_volume :
  ∀ (sphere_radius : ℝ) (tetrahedron_edge : ℝ), 
    (sphere_radius = 1) → 
    (tetrahedron_edge = sqrt 3) → 
    4 / 3 * π * (unit_sphere_volume)^3 = 4 * π / 81 :=
by
  intros sphere_radius tetrahedron_edge hr he
  sorry

end largest_internal_external_sphere_volume_l747_747669


namespace cyclic_sum_ge_sqrt_ab_plus_bc_plus_ca_l747_747354

variable (a b c x y z : ℝ)

theorem cyclic_sum_ge_sqrt_ab_plus_bc_plus_ca (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    (x / (y + z) * (b + c) + y / (z + x) * (c + a) + z / (x + y) * (a + b)) ≥ real.sqrt (3 * (a * b + b * c + c * a)) :=
sorry

end cyclic_sum_ge_sqrt_ab_plus_bc_plus_ca_l747_747354


namespace impossible_arrangement_l747_747046

theorem impossible_arrangement :
  ∃ (S : ℕ), ¬(∀ (x y z : ℕ), x ∈ {1, 4, 9, 16, 25, 36, 49} → y ∈ {1, 4, 9, 16, 25, 36, 49} → z ∈ {1, 4, 9, 16, 25, 36, 49} →
  ((x + y + z = S) →
  ((1 + 4 + 9 + 16 + 25 + 36 + 49) = 140) →
  (5 * S = 280 + x))) := sorry

end impossible_arrangement_l747_747046


namespace minimal_possible_length_of_longest_segment_maximal_possible_length_of_shortest_segment_l747_747917

noncomputable def acuteAngledTriangle (A1 A2 A3 : Point) : Prop := sorry

noncomputable def unitCircle (O : Point) : Circle := sorry

noncomputable
def cevianPoint (Ai Bi : Point) (O : Point) : Prop := sorry

theorem minimal_possible_length_of_longest_segment
  (A1 A2 A3 : Point)
  (O : Point)
  (circ : Circle)
  (h1 : acuteAngledTriangle A1 A2 A3)
  (h2 : A1 ∈ circ)
  (h3 : A2 ∈ circ)
  (h4 : A3 ∈ circ) :
  ∃ B1 B2 B3 : Point, (B1, B2, B3 ∈ cevianPoint A1 B1 O) ∧
  (B1, B2, B3 ∈ cevianPoint A2 B2 O) ∧
  (B1, B2, B3 ∈ cevianPoint A3 B3 O) ∧
  (∀ i = 1..3, Bi ≠ B_(if i = 1 then 2 else if i = 2 then 3 else 1)) ∧ 
  (minimal_possible_length_of_longest Bi O = 1/2) :=
sorry

theorem maximal_possible_length_of_shortest_segment
  (A1 A2 A3 : Point)
  (O : Point)
  (circ : Circle)
  (h1 : acuteAngledTriangle A1 A2 A3)
  (h2 : A1 ∈ circ)
  (h3 : A2 ∈ circ)
  (h4 : A3 ∈ circ) :
  ∃ B1 B2 B3 : Point, (B1, B2, B3 ∈ cevianPoint A1 B1 O) ∧
  (B1, B2, B3 ∈ cevianPoint A2 B2 O) ∧
  (B1, B2, B3 ∈ cevianPoint A3 B3 O) ∧
  (∀ i = 1..3, Bi ≠ B_(if i = 1 then 2 else if i = 2 then 3 else 1)) ∧ 
  (maximal_possible_length_of_shortest Bi O = 1/2) :=
sorry


end minimal_possible_length_of_longest_segment_maximal_possible_length_of_shortest_segment_l747_747917


namespace mickey_horses_per_week_l747_747312

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l747_747312


namespace probability_sum_10_l747_747483

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747483


namespace difference_between_perfect_and_cracked_l747_747572

def total_eggs : ℕ := 24
def broken_eggs : ℕ := 3
def cracked_eggs : ℕ := 2 * broken_eggs

def perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs
def difference : ℕ := perfect_eggs - cracked_eggs

theorem difference_between_perfect_and_cracked :
  difference = 9 := by
  sorry

end difference_between_perfect_and_cracked_l747_747572


namespace trapezoid_sides_l747_747886

theorem trapezoid_sides
  {AB BC AD CD : ℝ} 
  (is_trapezoid : ∃ (AB BC AD CD : ℝ), True) -- separate parts of the trapezoid
  (AC_angle_bisector_BA_120 : ∠B = ∠A / 2 ∧ ∠A = 120)
  (circumradius_ABD_√3 : ∃ (R : ℝ), R = √3)
  (AO_ratio_BO_4_1 : ∃ (AOD BOC : ℝ), AOD / BOC = 4) :
  (AB = BC ∧ AB = CD / 2 ≠ AC ∧ 
   AB = 3 / √7 ∧ CD = 3 * √3 / √7 ∧ AD = 6 / √7) := 
by {
  sorry
}

end trapezoid_sides_l747_747886


namespace boxes_of_crackers_last_nights_l747_747295

-- Definitions based on the conditions
def crackers_per_sandwich : ℕ := 2
def sandwiches_per_night : ℕ := 5
def sleeves_per_box : ℕ := 4
def crackers_per_sleeve : ℕ := 28

-- Proof statement for the problem
theorem boxes_of_crackers_last_nights :
  let crackers_per_night := crackers_per_sandwich * sandwiches_per_night in
  let crackers_per_box := sleeves_per_box * crackers_per_sleeve in
  let total_crackers := crackers_per_box * 5 in
  total_crackers / crackers_per_night = 56 :=
by
  sorry

end boxes_of_crackers_last_nights_l747_747295


namespace simplify_expression_l747_747114

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) :=
by
  sorry

end simplify_expression_l747_747114


namespace tangent_line_intersects_x_axis_at_minus_3_l747_747157

open Real

noncomputable def curve (x : ℝ) : ℝ := x^3 + 11

def point_p : ℝ × ℝ := (1, 12)

theorem tangent_line_intersects_x_axis_at_minus_3 :
  let tangent_slope := deriv curve 1 in
  let tangent_line (x : ℝ) : ℝ := tangent_slope * (x - point_p.1) + point_p.2 in
  ∃ x₀, tangent_line x₀ = 0 ∧ x₀ = -3 :=
by
  sorry

end tangent_line_intersects_x_axis_at_minus_3_l747_747157


namespace domain_of_g_l747_747300

variable {x : ℝ}

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt (4 - Real.sqrt (7 - Real.sqrt (2 * x + 1)))

theorem domain_of_g :
  { x : ℝ | g x = g x } = set.Icc (-1/2 : ℝ) 24 :=
by
  sorry

end domain_of_g_l747_747300


namespace pizza_slices_distribution_l747_747108

theorem pizza_slices_distribution (total_slices : ℕ) (ron_eats : ℕ) (scott_eats : ℕ → ℕ) (mark_eats : ℕ) (remaining_friends : ℕ)
  (H1 : total_slices = 24)
  (H2 : ron_eats = 5)
  (H3 : ∀ remaining, scott_eats remaining = remaining / 3) -- Assuming whole number division
  (H4 : mark_eats = 2)
  (H5 : remaining_friends = 3) :
  let remaining_after_ron := total_slices - ron_eats in
  let remaining_after_scott := remaining_after_ron - scott_eats remaining_after_ron in
  let remaining_after_mark := remaining_after_scott - mark_eats in
  remaining_after_mark / remaining_friends = 3 :=
by
  sorry

end pizza_slices_distribution_l747_747108


namespace hyperbola_properties_proof_l747_747144
noncomputable def hyperbola_real_axis_length (a : ℝ) (b : ℝ) (hyp : a = sqrt 2 ∧ b = sqrt 2) : ℝ :=
2 * a

noncomputable def hyperbola_eccentricity (a : ℝ) (b : ℝ) (hyp : a = sqrt 2 ∧ b = sqrt 2) : ℝ :=
let c := sqrt (a^2 + b^2) in
c / a

noncomputable def hyperbola_asymptotes (a : ℝ) (b : ℝ) (hyp : a = sqrt 2 ∧ b = sqrt 2) : Set (ℝ × ℝ) :=
{ (x, y) | y = x ∨ y = -x }

theorem hyperbola_properties_proof :
  let a := sqrt 2 in
  let b := sqrt 2 in
  a = sqrt 2 ∧ b = sqrt 2 →
  hyperbola_real_axis_length a b (by simp) = 2 * sqrt 2 ∧
  hyperbola_eccentricity a b (by simp) = sqrt 2 ∧
  hyperbola_asymptotes a b (by simp) = { (x, y) | y = x ∨ y = -x } :=
by
  intros a b h1 h2
  constructor
  · exact congr_arg (λ a, 2 * a) h1
  constructor
  · simp [hyperbola_eccentricity, h1, h2]; ring
  · exact rfl

end hyperbola_properties_proof_l747_747144


namespace speed_of_sound_l747_747713

theorem speed_of_sound (time_blasts : ℝ) (distance_traveled : ℝ) (time_heard : ℝ) (speed : ℝ) 
  (h_blasts : time_blasts = 30 * 60) -- time between the two blasts in seconds 
  (h_distance : distance_traveled = 8250) -- distance in meters
  (h_heard : time_heard = 30 * 60 + 25) -- time when man heard the second blast
  (h_relationship : speed = distance_traveled / (time_heard - time_blasts)) : 
  speed = 330 :=
sorry

end speed_of_sound_l747_747713


namespace count_valid_3_digit_numbers_l747_747846

def is_valid_3_digit_number (n : ℕ) : Prop :=
  let h := n / 100 in
  let t := (n % 100) / 10 in
  let u := n % 10 in
  (100 ≤ n ∧ n < 1000) ∧ (u ≥ 3 * t)

theorem count_valid_3_digit_numbers : { n : ℕ // is_valid_3_digit_number n }.card = 189 :=
by
  sorry

end count_valid_3_digit_numbers_l747_747846


namespace probability_p_eq_l747_747209

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l747_747209


namespace fernanda_total_savings_l747_747279

def aryan_debt : ℝ := 1200
def kyro_debt : ℝ := aryan_debt / 2
def aryan_payment : ℝ := 0.60 * aryan_debt
def kyro_payment : ℝ := 0.80 * kyro_debt
def initial_savings : ℝ := 300

theorem fernanda_total_savings : 
  initial_savings + aryan_payment + kyro_payment = 1500 :=
by
  sorry

end fernanda_total_savings_l747_747279


namespace pastries_total_l747_747232

theorem pastries_total (total_volunteers : ℕ) 
  (pct_A pct_B pct_C : ℝ) 
  (batches_A batches_B batches_C : ℕ) 
  (trays_A trays_B trays_C : ℕ) 
  (pastries_per_tray_A pastries_per_tray_B pastries_per_tray_C : ℕ) :
  total_volunteers = 1500 →
  pct_A = 0.40 → pct_B = 0.35 → pct_C = 0.25 →
  batches_A = 10 → batches_B = 15 → batches_C = 8 →
  trays_A = 6 → trays_B = 4 → trays_C = 5 →
  pastries_per_tray_A = 20 → pastries_per_tray_B = 12 → pastries_per_tray_C = 15 →
  (nat.floor (pct_A * total_volunteers) * (batches_A * trays_A * pastries_per_tray_A) +
   nat.floor (pct_B * total_volunteers) * (batches_B * trays_B * pastries_per_tray_B) +
   nat.floor (pct_C * total_volunteers) * (batches_C * trays_C * pastries_per_tray_C) = 1323000) :=
by
  sorry

end pastries_total_l747_747232


namespace more_red_flowers_than_white_l747_747738

-- Definitions based on given conditions
def yellow_and_white := 13
def red_and_yellow := 17
def red_and_white := 14
def blue_and_yellow := 16

-- Definitions based on the requirements of the problem
def red_flowers := red_and_yellow + red_and_white
def white_flowers := yellow_and_white + red_and_white

-- Theorem to prove the number of more flowers containing red than white
theorem more_red_flowers_than_white : red_flowers - white_flowers = 4 := by
  sorry

end more_red_flowers_than_white_l747_747738


namespace room_length_l747_747133

def area_four_walls (L: ℕ) (w: ℕ) (h: ℕ) : ℕ :=
  2 * (L * h) + 2 * (w * h)

def area_door (d_w: ℕ) (d_h: ℕ) : ℕ :=
  d_w * d_h

def area_windows (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  num_windows * (win_w * win_h)

def total_area_to_whitewash (L: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  area_four_walls L w h - area_door d_w d_h - area_windows win_w win_h num_windows

theorem room_length (cost: ℕ) (rate: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) (L: ℕ) :
  cost = rate * total_area_to_whitewash L w h d_w d_h win_w win_h num_windows →
  L = 25 :=
by
  have h1 : total_area_to_whitewash 25 15 12 6 3 4 3 3 = 24 * 25 + 306 := sorry
  have h2 : rate * (24 * 25 + 306) = 5436 := sorry
  sorry

end room_length_l747_747133


namespace remainder_of_sum_division_l747_747186

theorem remainder_of_sum_division :
  ∀ (n : ℕ), (∃ q r, n = q * 6 + r ∧ r < 6 ∧ q = 124 ∧ r = 4) →
  let n' := n + 24 in
  ∃ q' r', n' = q' * 8 + r' ∧ r' < 8 ∧ r' = 4 :=
by intros n h; sorry

end remainder_of_sum_division_l747_747186


namespace total_memory_space_l747_747062

def morning_songs : Nat := 10
def afternoon_songs : Nat := 15
def night_songs : Nat := 3
def song_size : Nat := 5

theorem total_memory_space : (morning_songs + afternoon_songs + night_songs) * song_size = 140 := by
  sorry

end total_memory_space_l747_747062


namespace lindy_total_distance_l747_747690

theorem lindy_total_distance (distance_jc : ℝ) (speed_j : ℝ) (speed_c : ℝ) (speed_l : ℝ)
  (h1 : distance_jc = 270) (h2 : speed_j = 4) (h3 : speed_c = 5) (h4 : speed_l = 8) : 
  ∃ time : ℝ, time = distance_jc / (speed_j + speed_c) ∧ speed_l * time = 240 :=
by
  sorry

end lindy_total_distance_l747_747690


namespace percentage_change_in_area_l747_747638

-- Define initial and final radii
def initial_radius := 5 -- in cm
def final_radius := 4   -- in cm

-- Define the formula for the area of a circle
noncomputable def area (r : ℝ) : ℝ := π * r^2

-- Define the initial and final areas based on the radii
noncomputable def initial_area := area initial_radius
noncomputable def final_area := area final_radius

-- Calculate the change in area
noncomputable def change_in_area := final_area - initial_area

-- Calculate the percentage change in area
noncomputable def percentage_change := (change_in_area / initial_area) * 100

-- The theorem to be proved
theorem percentage_change_in_area : percentage_change = -36 := by
  sorry

end percentage_change_in_area_l747_747638


namespace probability_sum_10_l747_747484

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747484


namespace length_AD_proof_l747_747112

noncomputable def length_AD (A B C D : Point) : Real := 5 * Real.sqrt 13

variables (A B C D : Point)
variables (AB : length A B = 3)
variables (BC : length B C = 7)
variables (CD : length C D = 12)
variables (sin_C : Real.sin (angle C) = 3 / 5)
variables (neg_cos_B : -Real.cos (angle B) = 3 / 5)
variables (obtuse_B : angle B > π / 2 ∧ angle B < π)
variables (obtuse_C : angle C > π / 2 ∧ angle C < π)
variables (sum_AD : angle A + angle D = 110 * π / 180)

theorem length_AD_proof :
  AD = 5 * Real.sqrt 13 := sorry

end length_AD_proof_l747_747112


namespace point_D_coordinates_l747_747527

variables (A B C D E : Point)

-- Definition of the points with their respective coordinates.
-- Note: The actual coordinates of A, B, C, E should be defined
-- according to their real values in the given diagram.
def coordinates (p : Point) : Int × Int :=
  match p with
  | A => (x_A, y_A)
  | B => (x_B, y_B)
  | C => (x_C, y_C)
  | D => (-2, -3)
  | E => (x_E, y_E)

theorem point_D_coordinates :
  ∃ p, coordinates p = (-2, -3) → p = D :=
by
  -- The actual theorem or proof would be given here, but for now,
  -- we use sorry as a placeholder for the proof.
  sorry

end point_D_coordinates_l747_747527


namespace find_minimal_sum_l747_747786

theorem find_minimal_sum :
  ∃ c m : ℝ, (∀ x : ℝ, (∑ n in Finset.range 12, (n - x) * (2 * n - x)) ≥ (∑ n in Finset.range 12, (n - 9) * (2 * n - 9))) 
  ∧ m = ∑ n in Finset.range 12, (n - 9) * (2 * n - 9) 
  ∧ c = 9 
  ∧ m = 121 :=
by
  sorry

end find_minimal_sum_l747_747786


namespace percentage_increase_in_income_l747_747541

theorem percentage_increase_in_income
  (income_job_initial : ℕ := 40)
  (income_side_gig : ℕ := 10)
  (income_dividends : ℕ := 5)
  (income_job_after_raise : ℕ := 80)
  (old_total_income := income_job_initial + income_side_gig + income_dividends)
  (new_total_income := income_job_after_raise + income_side_gig + income_dividends)
  (percentage_increase := (new_total_income - old_total_income) * 100 / old_total_income) :
  percentage_increase ≈ 72.73 := 
sorry

end percentage_increase_in_income_l747_747541


namespace cubing_identity_l747_747408

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l747_747408


namespace area_triangle_PQR_l747_747297

open Real

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem area_triangle_PQR :
  ∀ (P Q R : ℝ × ℝ),
    distance P.fst P.snd Q.fst Q.snd = 3 →
    distance Q.fst Q.snd R.fst R.snd = 5 →
    (∃ (P₁ Q₁ R₁ : ℝ × ℝ), P₁ = (-√2, 1) ∧ Q₁ = (0, 2) ∧ R₁ = (√6, 3) ∧
       P = P₁ ∧ Q = Q₁ ∧ R = R₁) →
    0.5 * abs ((P.fst)*(Q.snd - R.snd) + (Q.fst)*(R.snd - P.snd) + (R.fst)*(P.snd - Q.snd)) = (√6 - √2) :=
by
  sorry

end area_triangle_PQR_l747_747297


namespace percentage_reduction_in_price_l747_747728
-- Import the necessary library

-- The Lean statement for this problem
theorem percentage_reduction_in_price :
  let P : ℝ := 4.444444444444445 in
  (∀ R : ℝ, 200 / R - 200 / P = 5 → (P - R) / P * 100 = 10) :=
by
  sorry

end percentage_reduction_in_price_l747_747728


namespace problem_statement_l747_747931

theorem problem_statement (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := 
sorry

end problem_statement_l747_747931


namespace quadrilateral_area_l747_747812

-- Define the ellipse passing through the point (1, e)
def ellipse (a b : ℝ) := λ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the circle centered at the origin with diameter a (major axis of ellipse)
def circle (r : ℝ) := λ (x y : ℝ), x^2 + y^2 = r^2

-- Given parameters
parameter (e a b : ℝ)
hypothesis ha : a > b
hypothesis hb : b > 0
hypothesis hel : 1 / a^2 + e^2 / (a^2 * b^2) = 1
hypothesis hc2 : a = sqrt 2 -- diameter of the circle (major axis length) = 2 * sqrt 2

-- Variables to represent the functions for ellipse and circle
noncomputable def ellipse_C1 := ellipse (sqrt 2) 1 -- major axis sqrt 2, minor axis 1
noncomputable def circle_C2 := circle sqrt 2 -- radius sqrt 2

-- Tangency condition of circle to the line
hypothesis htangent : ∀ x y : ℝ, (x - y + 2 = 0) → (x^2 + y^2 = 2)

-- Area range of quadrilateral
theorem quadrilateral_area :
  ∃ (A B C D : ℝ×ℝ), 
  (ellipse_C1 A.1 A.2) ∧ (ellipse_C1 B.1 B.2) ∧ 
  (circle_C2 C.1 C.2) ∧ (circle_C2 D.1 D.2) ∧ 
  (2 ≤ S A B C D ∧ S A B C D ≤ 2 * sqrt 2) := sorry

end quadrilateral_area_l747_747812


namespace proof_of_math_problem_l747_747285

variable {u : ℝ} {A B F K D C : ℝ}
variable (n : ℕ)

-- Assume the conditions provided in the problem
axiom conditions (h1 : F = (n-2) * u) (h2 : K = (n-1) * u) (h3 : D = (n-1) * (K - B))
  (h4 : C ∈ line_through A B) (h5 : F ∈ line_through A D) (h6 : K ∈ line_through A D) (h7 : D ∈ line_through K B) :
  CB = AB / n

def math_problem : Prop :=
  CB = AB / n

theorem proof_of_math_problem (h1 : F = (n-2) * u) (h2 : K = (n-1) * u) (h3 : D = (n-1) * (K - B))
  (h4 : C ∈ line_through A B) (h5 : F ∈ line_through A D) (h6 : K ∈ line_through A D) (h7 : D ∈ line_through K B) : 
  math_problem :=
sorry

end proof_of_math_problem_l747_747285


namespace maria_initial_savings_l747_747565

-- Define the costs of the items
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20

-- Number of items Maria will buy
def sweaters_bought : ℕ := 6
def scarves_bought : ℕ := 6

-- Remaining savings after purchase
def remaining_savings : ℕ := 200

-- Calculate the total cost of the sweaters and scarves
def total_spent : ℕ := (sweaters_bought * sweater_cost) + (scarves_bought * scarf_cost)

-- Calculate the initial savings
def initial_savings : ℕ := total_spent + remaining_savings

-- The theorem to prove that Maria's initial savings were $500
theorem maria_initial_savings : initial_savings = 500 := by
  -- Introduce the required definitions and a placeholder for the proof
  unfold total_spent initial_savings sweaters_bought scarves_bought sweater_cost scarf_cost remaining_savings
  -- Provide a beginning proof outline 
  sorry

end maria_initial_savings_l747_747565


namespace greatest_least_not_both_roots_l747_747372

theorem greatest_least_not_both_roots (F G : Polynomial ℝ) (hF : F.degree = 3) (hG : G.degree = 3)
  (hMonicF : F.leadingCoeff = 1) (hMonicG : G.leadingCoeff = 1) (hDistinct : F ≠ G)
  (roots_F : Multiset ℝ) (roots_G : Multiset ℝ) (roots_equal : Multiset ℝ)
  (h_roots_F : roots_F.card = 3) (h_roots_G : roots_G.card = 3) (h_roots_equal : roots_equal.card = 2)
  (hDistinctRoots : Multiset.disjoint roots_F roots_G ∧ Multiset.disjoint roots_F roots_equal ∧ Multiset.disjoint roots_G roots_equal)
  (hTotalRoots : (roots_F ∪ roots_G ∪ roots_equal).card = 8) :
  ∀ γ1 γ8, γ1 ∈ roots_F → γ8 ∈ roots_F → γ1 ≠ Multiset.min (roots_F ∪ roots_G ∪ roots_equal) →
∧ γ8 ≠ Multiset.max (roots_F ∪ roots_G ∪ roots_equal) :=
sorry

end greatest_least_not_both_roots_l747_747372


namespace problem_sum_of_solutions_l747_747548

theorem problem_sum_of_solutions :
  ∑ (x_i, y_i : ℝ) in { (x, y) | abs (x - 4) = 2 * abs (y - 8) ∧ abs (x - 6) = 3 * abs (y - 2) }, (x_i + y_i) = 27 :=
  sorry

end problem_sum_of_solutions_l747_747548


namespace dice_sum_probability_l747_747444

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747444


namespace age_difference_is_54_l747_747122
noncomputable theory

-- Define the current ages of Jack and Bill in terms of their digits
def Jack_current_age (a b : ℕ) : ℕ := 10 * a + b
def Bill_current_age (a b : ℕ) : ℕ := 10 * b + a

-- Define the future ages of Jack and Bill
def Jack_future_age (a b : ℕ) : ℕ := Jack_current_age a b + 10
def Bill_future_age (a b : ℕ) : ℕ := Bill_current_age a b + 10

-- Define the condition of the future ages relationship
def future_age_condition (a b : ℕ) : Prop :=
  Jack_future_age a b = 3 * Bill_future_age a b

-- Final proof statement for the difference in their current ages
theorem age_difference_is_54 (a b : ℕ) (h_a : a = 7) (h_b : b = 1)
  (h_condition : future_age_condition a b) :
  Jack_current_age a b - Bill_current_age a b = 54 :=
by {
  -- Add the given conditions
  have ha : a = 7 := h_a,
  have hb : b = 1 := h_b,
  have hfuture := h_condition,
  -- Prove the statement
  sorry
}

end age_difference_is_54_l747_747122


namespace probability_sum_is_10_l747_747502

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747502


namespace number_of_combinations_for_Xiao_Wang_l747_747036

-- Definitions for the subjects
inductive Subject
| Physics
| Chemistry
| Biology
| Politics
| History
| Geography

-- List of all available subjects
def all_subjects : List Subject := [
  Subject.Physics,
  Subject.Chemistry,
  Subject.Biology,
  Subject.Politics,
  Subject.History,
  Subject.Geography
]

-- Definitions for science and humanities subjects
def science_subjects : List Subject := [
  Subject.Physics,
  Subject.Chemistry,
  Subject.Biology
]

def humanities_subjects : List Subject := [
  Subject.Politics,
  Subject.History,
  Subject.Geography
]

-- Theorem statement
theorem number_of_combinations_for_Xiao_Wang : ∃ n: ℕ, n = 10 ∧ (
  let combinations := (Finset.powersetLen 2 (Finset.ofList science_subjects)).card * 
                      (Finset.powersetLen 1 (Finset.ofList humanities_subjects)).card +
                      (Finset.powersetLen 3 (Finset.ofList science_subjects)).card
  in combinations = 10) :=
by
  sorry

end number_of_combinations_for_Xiao_Wang_l747_747036


namespace scalar_product_calculation_l747_747844

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (2, 0, 3)
def c : ℝ × ℝ × ℝ := (0, -1, 2)

theorem scalar_product_calculation :
  let sum := (b.1 + c.1, b.2 + c.2, b.3 + c.3) in
  let dot_product := (a.1 * sum.1) + (a.2 * sum.2) + (a.3 * sum.3) in
  dot_product = 12 :=
by
  sorry

end scalar_product_calculation_l747_747844


namespace full_price_ticket_revenue_l747_747731

theorem full_price_ticket_revenue 
  (f h p : ℕ)
  (h1 : f + h = 160)
  (h2 : f * p + h * (p / 3) = 2400) :
  f * p = 400 := 
sorry

end full_price_ticket_revenue_l747_747731


namespace mickey_horses_per_week_l747_747308

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l747_747308


namespace log2_decreasing_interval_l747_747146

-- Let f be the function y = log_2(1 - x^2)
def f (x : ℝ) : ℝ := log 2 (1 - x^2)

-- Define the condition that the function is defined on the interval (-1, 1)
def is_defined (x : ℝ) : Prop := -1 < x ∧ x < 1

-- State the theorem that y = log_2(1 - x^2) is monotonically decreasing in the interval (0, 1)
theorem log2_decreasing_interval : ∀ x : ℝ, (0 < x ∧ x < 1) → (f x < f (x + ε)) :=
by
  intros x hx
  sorry

end log2_decreasing_interval_l747_747146


namespace BH_perp_QH_l747_747887

variable {Point : Type*} [MetricSpace Point]

structure Triangle (Point : Type*) :=
(a b c : Point)

variables (A B C I M P H Q : Point)
variables (triangle : Triangle Point)

-- Conditions:
def is_isosceles (triangle : Triangle Point) : Prop :=
  dist triangle.a triangle.b = dist triangle.b triangle.c

def is_incenter (I : Point) (triangle : Triangle Point) : Prop :=
  -- Assume necessary property definition for the incenter
  sorry

def midpoint (M B I : Point) : Prop :=
  -- Assume necessary property definition for a midpoint
  sorry

def point_on_AC (P A C : Point) (k : ℚ) : Prop :=
  dist A P = k * dist P C

def perp (M H P : Point) : Prop :=
  -- Assume necessary property definition for perpendicularity
  sorry

def midpoint_arc (Q A B : Point) (triangle : Triangle Point) : Prop :=
  -- Assume necessary property definition for midpoint of minor arc
  sorry

-- Theorem Statement:
theorem BH_perp_QH
  (isosceles_triangle : is_isosceles triangle)
  (incenter_property : is_incenter I triangle)
  (mid_M_BI : midpoint M triangle.b I)
  (point_condition : point_on_AC P triangle.a triangle.c 3)
  (extended_perp : perp M H P)
  (midpoint_arc_property : midpoint_arc Q triangle.a triangle.b triangle):
  ∃ BH QH : Line Point, perp BH QH := 
sorry

end BH_perp_QH_l747_747887


namespace percentage_reduction_in_prices_l747_747957

def original_price_shirt (n : ℕ) := 60 * n
def original_price_jacket (n : ℕ) := 90 * n
def total_original_price (n m : ℕ) := original_price_shshirt(n) + original_price_jacket(m)

-- The condition where Teagan paid 960 dollars for 5 shirts and 10 leather jackets after reduction.
def reduced_price (p : ℝ) (total_original : ℝ) := (1 - p / 100) * total_original = 960

theorem percentage_reduction_in_prices :
    reduced_price 20 (total_original_price 5 10) :=
by
  sorry

end percentage_reduction_in_prices_l747_747957


namespace problem_part1_problem_part2_l747_747600

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  ( (√2 / 2) * t, 3 + (√2 / 2) * t )

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 2 * Real.sin θ

theorem problem_part1 :
  (∀ t : ℝ, line_parametric t = (( √2 / 2 ) * t, 3 + ( √2 / 2 ) * t)) →
  line_parametric t = ((√2 / 2) * t, y) → x - y + 3 = 0 ∧
  (∀ ρ θ : ℝ, curve_polar ρ θ) →
  ∀ ρ θ, curve_polar ρ θ → (ρ * Real.cos θ)^2 = 2 * ρ * Real.sin θ → x^2 = 2 * y :=
sorry

theorem problem_part2 (continous_1 : ∀ t : ℝ, line_parametric t = ((√2 / 2) * t, 3 + (√2 / 2) * t)) :
  ∀ (x1 y1 x2 y2 : ℝ), (x1, y1), (x2, y2) ∈ {(x, y) | y = x + 3 ∧ x^2 = 2 * y } →
  let M := ((x1 + x2) / 2, (y1 + y2) / 2) in
  let P := (1, 1) in
  dist M P = 3 :=
sorry

end problem_part1_problem_part2_l747_747600


namespace activities_equally_popular_l747_747151

def Dodgeball_prefers : ℚ := 10 / 25
def ArtWorkshop_prefers : ℚ := 12 / 30
def MovieScreening_prefers : ℚ := 18 / 45
def QuizBowl_prefers : ℚ := 16 / 40

theorem activities_equally_popular :
  Dodgeball_prefers = ArtWorkshop_prefers ∧
  ArtWorkshop_prefers = MovieScreening_prefers ∧
  MovieScreening_prefers = QuizBowl_prefers :=
by
  sorry

end activities_equally_popular_l747_747151


namespace reflection_across_x_axis_l747_747523

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  reflect_x_axis (-2, -3) = (-2, 3) :=
by
  sorry

end reflection_across_x_axis_l747_747523


namespace local_value_of_4_in_564823_l747_747899

def face_value (d : ℕ) : ℕ := d
def place_value_of_thousands : ℕ := 1000
def local_value (d : ℕ) (p : ℕ) : ℕ := d * p

theorem local_value_of_4_in_564823 :
  local_value (face_value 4) place_value_of_thousands = 4000 :=
by 
  sorry

end local_value_of_4_in_564823_l747_747899


namespace min_value_sqrt_inequality_l747_747939

theorem min_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ c, (x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2) ≥ c ^ 2 ∧
       √((x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2)) / (x * y) = (2 + real.cbrt 4) / real.cbrt 2 := by
  sorry

end min_value_sqrt_inequality_l747_747939


namespace lamps_bulbs_l747_747588

theorem lamps_bulbs (x : ℕ) :
  (∃ (lamps : ℕ) (working_bulbs : ℕ) (burnt_out_factor : ℚ) (burnt_out_per_lamp : ℕ),
    lamps = 20 ∧
    burnt_out_factor = 1 / 4 ∧
    burnt_out_per_lamp = 2 ∧
    working_bulbs = 130 ∧
    (burnt_out_factor * lamps).to_nat * burnt_out_per_lamp = 10 ∧
    working_bulbs + ((burnt_out_factor * lamps).to_nat * burnt_out_per_lamp) = 140 ∧
    (140 / lamps) = x ) → x = 7 := 
by
  intros
  sorry

end lamps_bulbs_l747_747588


namespace triangle_DEF_angles_l747_747533

theorem triangle_DEF_angles (D E F K : Point)
  (h1 : Median D K E F)
  (h2 : Angle D K E = 70)
  (h3 : Angle D K F = 140) :
  (Angle D E F = 70 ∧ Angle E D F = 90 ∧ Angle D F E = 20) :=
sorry

end triangle_DEF_angles_l747_747533


namespace light_intensity_at_10_m_l747_747623

theorem light_intensity_at_10_m (k : ℝ) (d1 d2 : ℝ) (I1 I2 : ℝ)
  (h1: I1 = k / d1^2) (h2: I1 = 200) (h3: d1 = 5) (h4: d2 = 10) :
  I2 = k / d2^2 → I2 = 50 :=
sorry

end light_intensity_at_10_m_l747_747623


namespace find_x_l747_747845

variable (x : ℝ)
def a : ℝ × ℝ × ℝ := (1, 1, x)
def b : ℝ × ℝ × ℝ := (1, 2, 1)
def c : ℝ × ℝ × ℝ := (1, 1, 1)

def vec_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)
def vec_scale (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (k * v.1, k * v.2, k * v.3)
def dot (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_x (h : dot (vec_sub c a) (vec_scale 2 b) = -2) : x = 2 := by
  sorry

end find_x_l747_747845


namespace meals_given_away_l747_747749

def initial_meals_colt_and_curt : ℕ := 113
def additional_meals_sole_mart : ℕ := 50
def remaining_meals : ℕ := 78
def total_initial_meals : ℕ := initial_meals_colt_and_curt + additional_meals_sole_mart
def given_away_meals (total : ℕ) (remaining : ℕ) : ℕ := total - remaining

theorem meals_given_away : given_away_meals total_initial_meals remaining_meals = 85 :=
by
  sorry

end meals_given_away_l747_747749


namespace prime_divisor_gt_n_cubed_l747_747717

theorem prime_divisor_gt_n_cubed (n : ℕ) (h : ∀ d : ℕ, d ∣ n → ¬ (n^2 ≤ d^4 ∧ d^4 ≤ n^3)) :
  ∃ p : ℕ, prime p ∧ p ∣ n ∧ p^4 > n^3 :=
by sorry

end prime_divisor_gt_n_cubed_l747_747717


namespace range_of_x_plus_2y_minus_2z_l747_747867

theorem range_of_x_plus_2y_minus_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) : -6 ≤ x + 2 * y - 2 * z ∧ x + 2 * y - 2 * z ≤ 6 :=
sorry

end range_of_x_plus_2y_minus_2z_l747_747867


namespace chord_length_square_l747_747528

-- Define the problem conditions
variables {O1 O2 P Q R : Type}
variables {r1 r2 c : ℝ} -- Radii and distance between centers
variables [inhabited O1] [inhabited O2] [inhabited P] [inhabited Q] [inhabited R]
variables (o1p : R) (o2p : R)

-- Assume the distance between centers and circle radii
variables (h1 : r1 = 10) (h2 : r2 = 7) (h3 : c = 15)

-- Point P is the intersection point
variable (intersect : P ∈ (λ x, dist O1 x = r1) ∩ (λ x, dist O2 x = r2))

-- Define the chord lengths as equal
variables (qp : ℝ) (pr : ℝ) (hp : qp = pr)

-- Prove the square of the length of chord QP
theorem chord_length_square : qp^2 = 308.5714 := sorry

end chord_length_square_l747_747528


namespace epidemic_prevention_competition_l747_747164

theorem epidemic_prevention_competition :
  let P_A1 := 3/5;
  let P_A2 := 2/3;
  let P_B1 := 3/4;
  let P_B2 := 2/5;
  (P_A1 * P_A2 > P_B1 * P_B2) ∧
  (1 - ((1 - P_A1 * P_A2) * (1 - P_B1 * P_B2)) = 29/50) :=
begin
  -- Proof goes here
  sorry
end

end epidemic_prevention_competition_l747_747164


namespace probability_sum_is_ten_l747_747436

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747436


namespace probability_sum_is_ten_l747_747433

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747433


namespace probability_sum_is_10_l747_747507

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747507


namespace intersection_of_M_and_N_l747_747951

namespace MathProof

open Set Real

-- Conditions as definitions
def M : Set ℝ := { x | 2^(x-1) < 1 }
def N : Set ℝ := { x | log 2 x < 1 }

-- Theorem statement without proof
theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
sorry

end MathProof

end intersection_of_M_and_N_l747_747951


namespace angle_APQ_eq_angle_ANC_l747_747166

variables (A B M N C P Q : Type)
variables [is_intersecting_circles : TwoCirclesIntersectAt A B]
variables [AM_tangent : TangentToCircle A M]
variables [AN_tangent : TangentToCircle A N]
variables [parallelogram_man : IsParallelogram M A N C]
variables [P_on_BN : DividedInEqualRatio B N P]
variables [Q_on_MC : DividedInEqualRatio M C Q]

theorem angle_APQ_eq_angle_ANC :
  ∠ A P Q = ∠ A N C :=
sorry

end angle_APQ_eq_angle_ANC_l747_747166


namespace BO₁CO₂_is_parallelogram_l747_747154

variables {A B C D E O₁ O₂ : Type}
-- Define the conditions of the problem
variables (ABC BDE : Triangle) [IsCongruentABC : CongruentABC]
variables [IsIsocelesABC : IsIsocelesABC]
variables [LiesOnSameLine : BasesLiesOnSameLine ABC BDE]
variables [SameSide : PointsOnSameSide C E]

-- Define the circumcenters O₁ and O₂
variables (O₁ : Circle (TriangleCircumcenter (Triangle A B E)))
variables (O₂ : Circle (TriangleCircumcenter (Triangle C D E)))

-- Define the quadrilateral
def quad_BO₁CO₂ : Quadrilateral B O₁ C O₂ := sorry

-- The theorem to be proved
theorem BO₁CO₂_is_parallelogram :
  quadrilateralType quad_BO₁CO₂ = QuadrilateralType.Parallelogram := sorry

end BO₁CO₂_is_parallelogram_l747_747154


namespace harly_initial_dogs_l747_747398

theorem harly_initial_dogs (x : ℝ) 
  (h1 : 0.40 * x + 0.60 * x + 5 = 53) : 
  x = 80 := 
by 
  sorry

end harly_initial_dogs_l747_747398


namespace travelers_payment_strategy_l747_747663

theorem travelers_payment_strategy :
  ∀ (gold_chain : list ℕ), 
  gold_chain.length = 6 →
  ∃ (cut_link_index : ℕ) (payment_sequence : list ℕ),
    (cut_link_index < 6) ∧
    (∀ (trip_index : ℕ), trip_index < 6 → 
     ∃ (links_paid : ℕ),  links_paid = 1 ∧ -- payment per trip is 1 link
     list.nodup payment_sequence ∧ -- no duplicate payments
     payment_sequence.length = 6 ∧  -- payment sequence covers all trips
     payment_sequence[trip_index].nat_abs < 6 ∧ -- payment is within link range
     -- condition for only one cut link used
     (matches_cut_link_only_once gold_chain cut_link_index payment_sequence[trip_index])
  )
  →
  true :=
sorry

-- Helper definition to verify that the only cut link is used
def matches_cut_link_only_once (chain : list ℕ) (cut_index : ℕ) (link : ℕ) : Prop :=
  link = cut_index

end travelers_payment_strategy_l747_747663


namespace dice_sum_probability_l747_747477

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747477


namespace campaign_donation_ratio_l747_747163

theorem campaign_donation_ratio (max_donation : ℝ) 
  (total_money : ℝ) 
  (percent_donations : ℝ) 
  (num_max_donors : ℕ) 
  (half_max_donation : ℝ) 
  (total_raised : ℝ) 
  (half_donation : ℝ) :
  total_money = total_raised * percent_donations →
  half_donation = max_donation / 2 →
  half_max_donation = num_max_donors * max_donation →
  total_money - half_max_donation = 1500 * half_donation →
  (1500 : ℝ) / (num_max_donors : ℝ) = 3 :=
sorry

end campaign_donation_ratio_l747_747163


namespace transformation_of_function_l747_747562

-- Defining the function transformation as specified in the problem
theorem transformation_of_function 
  (f : ℝ → ℝ)
  (h : ℝ → ℝ)
  (H1 : ∀ x, h(x) = 2 * f(x))
  (H2 : ∀ x, h(x) = h(x) + 3) : 
  ∀ x, h(x) = 2 * f(x) + 3 :=
by
  sorry

end transformation_of_function_l747_747562


namespace incorrect_conclusions_l747_747965

theorem incorrect_conclusions :
  let p := (∀ x y : ℝ, x * y ≠ 6 → x ≠ 2 ∨ y ≠ 3)
  let q := (2, 1) ∈ { p : ℝ × ℝ | p.2 = 2 * p.1 - 3 }
  (p ∨ ¬q) = false ∧ (¬p ∨ q) = false ∧ (p ∧ ¬q) = false :=
by
  sorry

end incorrect_conclusions_l747_747965


namespace average_height_l747_747997

def heights : List ℕ := [178, 179, 181, 182, 176, 183, 180, 183, 175, 181, 185, 180, 184]

theorem average_height (h : List ℕ) : h = heights →
  Nat.round ((h.sum) / (h.length : ℚ)) = 180 :=
by
  intro h_def
  subst h_def
  sorry

end average_height_l747_747997


namespace concurrency_of_lines_l747_747816

variables {A B C O M E D X Y : Type*}
  [inhabited A] [inhabited B] [inhabited C] [inhabited O] [inhabited M] [inhabited E] [inhabited D] [inhabited X] [inhabited Y]

-- Condition 1: O is the circumcenter of triangle ABC
def circumcenter (A B C O : Type*) : Prop := sorry

-- Condition 2: M is the midpoint of BC
def midpoint (M B C : Type*) : Prop := sorry

-- Condition 3: line through M parallel to BO intersects the altitude CE of triangle ABC at X
def line_through_parallel (M B O : Type*) : Prop := sorry

-- Condition 4: line through M parallel to CO intersects the altitude BD of triangle ABC at Y
def line_through_parallel_altitude (M C O D : Type*) : Prop := sorry

theorem concurrency_of_lines (A B C O M E D X Y : Type*) 
  [circumcenter A B C O] 
  [midpoint M B C] 
  [line_through_parallel M B O] 
  [line_through_parallel M B O E X] 
  [line_through_parallel M C O] 
  [line_through_parallel_altitude M C O D Y] 
  : concurrent A O B X C Y := 
sorry

end concurrency_of_lines_l747_747816


namespace evaluate_expression_l747_747756

def diamond (a b : ℚ) : ℚ := a + (1 / b)

theorem evaluate_expression : 
  let result := diamond (diamond 3 4) 5 - diamond 3 (diamond 4 5) 
  in result = 89 / 420 :=
by 
  sorry

end evaluate_expression_l747_747756


namespace volume_of_regular_tetrahedron_l747_747129

theorem volume_of_regular_tetrahedron (a h : ℝ) (h1 : a = 2) (h2 : h = 1) : 
  (a^2 * sqrt 3 * h) / 12 = sqrt 3 / 3 :=
by
    sorry

end volume_of_regular_tetrahedron_l747_747129


namespace minimum_value_sqrt_inequality_l747_747941

theorem minimum_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (sqrt ((x^2 + y^2) * (4 * x^2 + y^2)) / (x * y)) ≥ 3 :=
sorry

end minimum_value_sqrt_inequality_l747_747941


namespace smallest_sum_of_digits_l747_747542

-- Define three-digit positive integers
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

-- Define unique digits constraint
def are_digits_unique (a b : ℕ) : Prop :=
  a.digits 10 ∪ b.digits 10 = a.digits 10 ∪ b.digits 10 ∧
  a.digits 10.to_finset ∩ b.digits 10.to_finset = ∅

-- Define sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main problem statement
theorem smallest_sum_of_digits :
  ∃ (a b S : ℕ), is_three_digit a ∧ is_three_digit b ∧ 
  are_digits_unique a b ∧ S = a + b ∧ is_three_digit S ∧ sum_of_digits S = 4 :=
by
  sorry

end smallest_sum_of_digits_l747_747542


namespace value_of_f_at_2_l747_747618

theorem value_of_f_at_2 (a b : ℝ) (h : (a + -b + 8) = (9 * a + 3 * b + 8)) :
  (a * 2 ^ 2 + b * 2 + 8) = 8 := 
by
  sorry

end value_of_f_at_2_l747_747618


namespace boys_count_l747_747258

def total_pupils : ℕ := 485
def number_of_girls : ℕ := 232
def number_of_boys : ℕ := total_pupils - number_of_girls

theorem boys_count : number_of_boys = 253 := by
  -- The proof is omitted according to instruction
  sorry

end boys_count_l747_747258


namespace snooker_tournament_tickets_l747_747732

theorem snooker_tournament_tickets
    (P_V : ℕ) (P_G : ℕ) (T : ℕ) (R : ℕ)
    (V G : ℕ)
    (h1 : P_V = 45)
    (h2 : P_G = 20)
    (h3 : V + G = 320)
    (h4 : 45 * V + 20 * G = 7500) :
    G - V = 232 :=
by
    subst h1
    subst h2
    have h5 : V = 320 - G, from sorry,
    have h6 : 45 * (320 - G) + 20 * G = 7500, from sorry,
    have : 14400 - (45 + 25) * G = 7500, from sorry,
    have h7 : 25 * G = 14400 - 7500, from sorry,
    have h8 : G = 276, from sorry,
    have h9 : V = 320 - 276, from sorry,
    have : V = 44, from sorry,
    show G - V = 232, from sorry

end snooker_tournament_tickets_l747_747732


namespace hyperbola_standard_form_l747_747645

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

theorem hyperbola_standard_form :
  (foci_of_ellipse : Set (ℝ × ℝ)) 
  (asymptotes_hyperbola : Set ℝ) 
  (3 * x^2 + 13 * y^2 = 39) 
  (foci_of_ellipse = {(sqrt 10, 0), (-sqrt 10, 0)}) 
  (asymptotes_hyperbola = {y = x / 2, y = - x / 2}) :
  ∀ x y, hyperbola_equation x y := 
-- Proof omitted
  sorry

end hyperbola_standard_form_l747_747645


namespace find_k_l747_747377

-- Define the vectors
def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)

def a : ℝ × ℝ := (e1.1 - 2 * e2.1, e1.2 - 2 * e2.2)
def b (k : ℝ) : ℝ × ℝ := (k * e1.1 + e2.1, k * e1.2 + e2.2)

-- Define the parallel condition
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the problem translated to a Lean theorem
theorem find_k (k : ℝ) : 
  parallel a (b k) -> k = -1 / 2 := by
  sorry

end find_k_l747_747377


namespace part1_part2_l747_747831

noncomputable def f (a x : ℝ) : ℝ := (a / x) - Real.log x

theorem part1 (a : ℝ) (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2) 
(hf : f a x1 = -3) (hf2 : f a x2 = -3) : a ∈ Set.Ioo (-Real.exp 2) 0 :=
sorry

theorem part2 (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2)
(hfa : f (-2) x1 = -3) (hfb : f (-2) x2 = -3) : x1 + x2 > 4 :=
sorry

end part1_part2_l747_747831


namespace tamika_vs_carlos_l747_747603

theorem tamika_vs_carlos :
  let tamika_sums := {21, 22, 23}
  let carlos_products := {24, 28, 42}
  ∀ t ∈ tamika_sums, ∀ c ∈ carlos_products, t ≤ c →
  (∃ p : ℚ, p = (0 : ℚ) / 9) :=
by
  let tamika_sums := {21, 22, 23}
  let carlos_products := {24, 28, 42}
  intros t ht c hc htc
  use 0 / 9
  sorry

end tamika_vs_carlos_l747_747603


namespace triplet_sum_not_zero_l747_747192

def sum_triplet (a b c : ℝ) : ℝ := a + b + c

theorem triplet_sum_not_zero :
  ¬ (sum_triplet 3 (-5) 2 = 0) ∧
  (sum_triplet (1/4) (1/4) (-1/2) = 0) ∧
  (sum_triplet 0.3 (-0.1) (-0.2) = 0) ∧
  (sum_triplet 0.5 (-0.3) (-0.2) = 0) ∧
  (sum_triplet (1/3) (-1/6) (-1/6) = 0) :=
by 
  sorry

end triplet_sum_not_zero_l747_747192


namespace value_of_a_l747_747342

theorem value_of_a (a : ℕ) (h : 3 / 11 * ∏ n in finset.Ico 3 (a+1), (1 + (1:ℚ)/n) = 11) : a = 120 :=
sorry

end value_of_a_l747_747342


namespace fractional_part_x_l747_747778

noncomputable def alpha := Real.arcsin (3 / 5)
noncomputable def x := 5 ^ 2003 * Real.sin (2004 * alpha)

theorem fractional_part_x :
  ∃ (t : Set ℝ), t = {0.2, 0.8} ∧ (x - Real.floor x) ∈ t := 
sorry

end fractional_part_x_l747_747778


namespace candidate_lost_by_1350_votes_l747_747227

theorem candidate_lost_by_1350_votes :
  let total_votes := 4500
  let candidate_votes := 0.35 * total_votes
  let rival_votes := 0.65 * total_votes
  in rival_votes - candidate_votes = 1350 :=
by
  let total_votes := 4500
  let candidate_votes := 0.35 * total_votes
  let rival_votes := 0.65 * total_votes
  have h1 : rival_votes - candidate_votes = 0.65 * total_votes - 0.35 * total_votes := by sorry
  have h2 : 0.65 * total_votes - 0.35 * total_votes = (0.65 - 0.35) * total_votes := by sorry
  have h3 : (0.65 - 0.35) * total_votes = 0.3 * total_votes := by sorry
  have h4 : 0.3 * total_votes = 0.3 * 4500 := by sorry
  have h5 : 0.3 * 4500 = 1350 := by sorry
  exact eq.trans h1 (eq.trans h2 (eq.trans h3 (eq.trans h4 h5)))

end candidate_lost_by_1350_votes_l747_747227


namespace meet_time_l747_747693

def track_circumference : ℕ := 594  -- in meters
def deepak_speed : ℚ := 4.5 / 60 * 1000  -- converting to m/min
def wife_speed : ℚ := 3.75 / 60 * 1000  -- converting to m/min
def relative_speed : ℚ := deepak_speed + wife_speed  -- sum their speeds

theorem meet_time : (track_circumference : ℚ) / relative_speed ≈ 4.32 := by
  sorry

end meet_time_l747_747693


namespace clock_angle_8_15_l747_747280

def angle_minute_hand (minutes : ℕ) : ℝ :=
  (minutes / 60.0) * 360.0

def angle_hour_hand (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours * 30.0) + (minutes / 60.0) * 30.0

def smaller_angle (angle1 : ℝ) (angle2 : ℝ) : ℝ :=
  let diff := |angle1 - angle2|
  min diff (360.0 - diff)

theorem clock_angle_8_15 :
  smaller_angle (angle_minute_hand 15) (angle_hour_hand 8 15) = 157.5 :=
by
  sorry

end clock_angle_8_15_l747_747280


namespace teacher_age_l747_747126

theorem teacher_age (avg_student_age : ℕ) (num_students : ℕ) (new_avg_age : ℕ) (num_total : ℕ) (total_student_age : ℕ) (total_age_with_teacher : ℕ) :
  avg_student_age = 22 → 
  num_students = 23 → 
  new_avg_age = 23 → 
  num_total = 24 → 
  total_student_age = avg_student_age * num_students → 
  total_age_with_teacher = new_avg_age * num_total → 
  total_age_with_teacher - total_student_age = 46 :=
by
  intros
  sorry

end teacher_age_l747_747126


namespace license_plate_combinations_l747_747848

theorem license_plate_combinations :
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits = 110250 :=
by
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  sorry

end license_plate_combinations_l747_747848


namespace red_flowers_count_l747_747085

-- Let's define the given conditions
def total_flowers : ℕ := 10
def white_flowers : ℕ := 2
def blue_percentage : ℕ := 40

-- Calculate the number of blue flowers
def blue_flowers : ℕ := (blue_percentage * total_flowers) / 100

-- The property we want to prove is the number of red flowers
theorem red_flowers_count :
  total_flowers - (blue_flowers + white_flowers) = 4 :=
by
  sorry

end red_flowers_count_l747_747085


namespace dawn_monthly_savings_l747_747755

variable (annual_income : ℕ)
variable (months : ℕ)
variable (tax_deduction_percent : ℚ)
variable (variable_expense_percent : ℚ)
variable (savings_percent : ℚ)

def calculate_monthly_savings (annual_income months : ℕ) 
    (tax_deduction_percent variable_expense_percent savings_percent : ℚ) : ℚ :=
  let monthly_income := (annual_income : ℚ) / months;
  let after_tax_income := monthly_income * (1 - tax_deduction_percent);
  let after_expenses_income := after_tax_income * (1 - variable_expense_percent);
  after_expenses_income * savings_percent

theorem dawn_monthly_savings : 
    calculate_monthly_savings 48000 12 0.20 0.30 0.10 = 224 := 
  by 
    sorry

end dawn_monthly_savings_l747_747755


namespace part1_part2_l747_747797

theorem part1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) : 
  a + 2 * b + c ≤ 4 :=
sorry

theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) (h5 : a = 2 * b) : 
  1 / b + 1 / (c - 1) ≥ 3 :=
sorry

end part1_part2_l747_747797


namespace total_clouds_counted_l747_747293

def clouds_counted (carson_clouds : ℕ) (brother_factor : ℕ) : ℕ :=
  carson_clouds + (carson_clouds * brother_factor)

theorem total_clouds_counted (carson_clouds brother_factor total_clouds : ℕ) 
  (h₁ : carson_clouds = 6) (h₂ : brother_factor = 3) (h₃ : total_clouds = 24) :
  clouds_counted carson_clouds brother_factor = total_clouds :=
by
  sorry

end total_clouds_counted_l747_747293


namespace arithmetic_sequence_middle_term_l747_747524

theorem arithmetic_sequence_middle_term 
  (a b c d e : ℕ) 
  (h_seq : a = 23 ∧ e = 53 ∧ (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d)) :
  c = 38 :=
by
  sorry

end arithmetic_sequence_middle_term_l747_747524


namespace dice_sum_prob_10_l747_747464

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747464


namespace diagonal_BD_eq_diagonal_AD_eq_l747_747035

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-1, 2⟩
def C : Point := ⟨5, 4⟩
def line_AB (p : Point) : Prop := p.x - p.y + 3 = 0

theorem diagonal_BD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ BD : Point → Prop, (BD = fun p => 3*p.x + p.y - 9 = 0)) :=
by
  sorry

theorem diagonal_AD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ AD : Point → Prop, (AD = fun p => p.x + 7*p.y - 13 = 0)) :=
by
  sorry

end diagonal_BD_eq_diagonal_AD_eq_l747_747035


namespace find_ordered_pair_l747_747335

theorem find_ordered_pair : ∃ x y, x + 2 * y = 7 ∧ x = 1 ∧ y = 3 :=
by
  use 1
  use 3
  sorry

end find_ordered_pair_l747_747335


namespace geom_sequence_50th_term_l747_747898

theorem geom_sequence_50th_term (a a_2 : ℤ) (n : ℕ) (r : ℤ) (h1 : a = 8) (h2 : a_2 = -16) (h3 : r = a_2 / a) (h4 : n = 50) :
  a * r^(n-1) = -8 * 2^49 :=
by
  sorry

end geom_sequence_50th_term_l747_747898


namespace max_possible_ratio_squared_l747_747935

noncomputable def maxRatioSquared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : ℝ :=
  2

theorem max_possible_ratio_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : maxRatioSquared a b h1 h2 h3 h4 = 2 :=
sorry

end max_possible_ratio_squared_l747_747935


namespace intersection_of_lines_l747_747177

-- Define the given lines
def line1 (x : ℝ) : ℝ := 2 * x + 5
def line2 (x : ℝ) : ℝ := -1/2 * x + 15/2

-- Define the intersection point (x, y)
def intersection_point : ℝ × ℝ := (1, 7)

-- The theorem statement to prove
theorem intersection_of_lines : ∃ x y : ℝ, y = line1 x ∧ y = line2 x ∧ (x, y) = intersection_point :=
by
  sorry

end intersection_of_lines_l747_747177


namespace ratio_of_x_to_y_l747_747249

variable {R x y : ℝ}

-- Given Conditions
def condition1 : Prop := x = R * y
def condition2 : Prop := y = 0.125 * x

-- Theorem statement
theorem ratio_of_x_to_y : condition1 → condition2 → R = 8 := by
  intros h1 h2
  -- h1 : x = R * y
  -- h2 : y = 0.125 * x
  sorry

end ratio_of_x_to_y_l747_747249


namespace marbles_left_after_removal_l747_747226

-- Defining the initial conditions
def total_marbles : ℕ := 180

def percentage_silver : ℚ := 25 / 100
def percentage_gold : ℚ := 20 / 100
def percentage_bronze : ℚ := 15 / 100
def percentage_sapphire : ℚ := 10 / 100
def percentage_ruby : ℚ := 10 / 100
def percentage_diamond : ℚ := 1 - (percentage_silver + percentage_gold + percentage_bronze + percentage_sapphire + percentage_ruby)

def initial_gold_marbles : ℕ := (total_marbles * percentage_gold).toNat -- calculating initial gold marbles
def gold_marbles_removed : ℕ := (initial_gold_marbles * (10 / 100 : ℚ)).round.toNat -- removing 10% of gold marbles, rounded

def remaining_marbles : ℕ := total_marbles - gold_marbles_removed 

-- The theorem to prove
theorem marbles_left_after_removal : remaining_marbles = 176 := by 
  sorry

end marbles_left_after_removal_l747_747226


namespace ceil_sqrt_sum_l747_747322

theorem ceil_sqrt_sum : (Finset.sum (Finset.range 43) (λ n, Nat.ceil (Real.sqrt (n + 8)))) = 244 := by
  sorry

end ceil_sqrt_sum_l747_747322


namespace bisection_initial_interval_l747_747172

def f (x : ℝ) := x^3 + 5

theorem bisection_initial_interval :
  ∃ a b : ℝ, a = -2 ∧ b = 1 ∧ f a = -3 ∧ f b = 6 ∧ f a * f b < 0 :=
by {
  use [-2, 1],
  simp [f],
  norm_num,
  sorry,
}

end bisection_initial_interval_l747_747172


namespace find_multiplier_l747_747011

theorem find_multiplier (x n : ℤ) (h : 2 * n + 20 = x * n - 4) (hn : n = 4) : x = 8 :=
by
  sorry

end find_multiplier_l747_747011


namespace checker_game_even_n_winner_checker_game_odd_n_winner_checker_game_adjacent_to_corner_winner_l747_747614

-- Definitions for the chess game problem
def checker_game_outcome (n : ℕ) (initial_position : (ℕ × ℕ)) : string :=
  if initial_position = (0, 0) then
    if n % 2 = 0 then "first wins" else "second wins"
  else
    "first wins"

-- Theorem statements for the given problem
theorem checker_game_even_n_winner (n : ℕ) (h_even : n % 2 = 0) : 
  checker_game_outcome n (0, 0) = "first wins" :=
sorry

theorem checker_game_odd_n_winner (n : ℕ) (h_odd : n % 2 = 1) : 
  checker_game_outcome n (0, 0) = "second wins" :=
sorry

theorem checker_game_adjacent_to_corner_winner (n : ℕ) : 
  ∀ p : (ℕ × ℕ), p = (0, 1) ∨ p = (1, 0) → checker_game_outcome n p = "first wins" :=
sorry

end checker_game_even_n_winner_checker_game_odd_n_winner_checker_game_adjacent_to_corner_winner_l747_747614


namespace combination_add_l747_747630

def combination (n m : ℕ) : ℕ := n.choose m

theorem combination_add {n : ℕ} (h1 : 4 ≤ 9) (h2 : 5 ≤ 9) :
  combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end combination_add_l747_747630


namespace distinct_convex_polygons_l747_747326

theorem distinct_convex_polygons (n : ℕ) (h : n = 15) :
  let total_subsets := 2^n,
      subsets_0 := nat.choose n 0,
      subsets_1 := nat.choose n 1,
      subsets_2 := nat.choose n 2,
      valid_subsets := total_subsets - (subsets_0 + subsets_1 + subsets_2)
  in valid_subsets = 32647 := by
  sorry

end distinct_convex_polygons_l747_747326


namespace min_value_sqrt_inequality_l747_747937

theorem min_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ c, (x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2) ≥ c ^ 2 ∧
       √((x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2)) / (x * y) = (2 + real.cbrt 4) / real.cbrt 2 := by
  sorry

end min_value_sqrt_inequality_l747_747937


namespace min_value_sqrt_inequality_l747_747938

theorem min_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ c, (x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2) ≥ c ^ 2 ∧
       √((x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2)) / (x * y) = (2 + real.cbrt 4) / real.cbrt 2 := by
  sorry

end min_value_sqrt_inequality_l747_747938


namespace value_of_fraction_l747_747982

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : (4 * x + y) / (x - 4 * y) = -3)

theorem value_of_fraction : (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end value_of_fraction_l747_747982


namespace shift_cos_left_l747_747658

theorem shift_cos_left (x : ℝ) :
  (cos (x - π / 3)) = cos (x + π / 3 - π / 3) := 
by sorry

end shift_cos_left_l747_747658


namespace game_terminates_pebbles_in_hole_0_l747_747245

theorem game_terminates_pebbles_in_hole_0 (N : ℕ) (h0 : ∀ i > 0, 2 * 2009 = i) :
  -- Initial condition: 2009 pebbles in hole 1, all other holes empty.
  (∀ (board : ℕ → ℕ), board 0 = N ∧ board 1 = 2009 ∧ (∀ i > 1, board i = 0)) →
  -- Condition: At each step, remove 2 pebbles from one hole, place one in each neighbor:
  (∀ (step: ℕ → ℕ), ∀ i, step i = if i > 0 ∧ board i ≥ 2 then board i - 2 + board (i - 1) + board (i + 1) else board i) →
  -- Condition: No pebbles are ever removed from hole 0:
  (∀ (board : ℕ → ℕ), board 0 = N) →
  -- Condition: Game ends if no hole with positive label contains at least 2 pebbles:
  (∀ (board : ℕ → ℕ), (∀ i > 0, board i < 2)) →
  -- Prove: The game always terminates with 1953 pebbles in hole 0.
  ∃ (z : ℕ), game_terminates ∧ board 0 = 1953 := sorry

end game_terminates_pebbles_in_hole_0_l747_747245


namespace complex_fraction_eval_l747_747925

theorem complex_fraction_eval (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a * b + b^2 = 0) :
  (a^15 + b^15) / (a + b)^15 = -2 := by
sorry

end complex_fraction_eval_l747_747925


namespace triangle_proof_l747_747044

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) :=
  ∃A B C a b c : ℝ,
    b^2 + c^2 - a^2 = (4 * real.sqrt 2) / 3 * b * c ∧
    sin A = 1 / 3 ∧
    (3 * c) / a = (real.sqrt 2) * (sin B) / (sin A) ∧
    1/2 * b * c * sin A = 2 * real.sqrt 2 ∧
    c = 2 * real.sqrt 2 ∧
    sin (2 * C - real.pi / 6) = (10 * real.sqrt 6 - 23) / 54

theorem triangle_proof :
  triangle_ABC a b c A B C := sorry

end triangle_proof_l747_747044


namespace sum_of_factors_30_l747_747675

def sum_of_factors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ x => n % x = 0) |>.sum

theorem sum_of_factors_30 : sum_of_factors 30 = 72 := by
  sorry

end sum_of_factors_30_l747_747675


namespace max_sum_twin_primes_le_200_l747_747167

def is_prime (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, 1 < a ∧ a < n ∧ 1 < b ∧ b < n ∧ a * b = n)

def is_twin_prime (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ abs (p - q) = 2

theorem max_sum_twin_primes_le_200 : ∃ p q : ℕ, is_twin_prime p q ∧ p ≤ 200 ∧ q ≤ 200 ∧ p + q = 396 :=
  sorry

end max_sum_twin_primes_le_200_l747_747167


namespace difference_between_perfect_and_cracked_l747_747574

def total_eggs : ℕ := 24
def broken_eggs : ℕ := 3
def cracked_eggs : ℕ := 2 * broken_eggs

def perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs
def difference : ℕ := perfect_eggs - cracked_eggs

theorem difference_between_perfect_and_cracked :
  difference = 9 := by
  sorry

end difference_between_perfect_and_cracked_l747_747574


namespace range_of_y_l747_747858

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 120) : y ∈ Set.Ioo (-11 : ℝ) (-10 : ℝ) :=
sorry

end range_of_y_l747_747858


namespace no_valid_placement_for_digits_on_45gon_l747_747047

theorem no_valid_placement_for_digits_on_45gon (f : Fin 45 → Fin 10) :
  ¬ ∀ (a b : Fin 10), a ≠ b → ∃ (i j : Fin 45), i ≠ j ∧ f i = a ∧ f j = b :=
by {
  sorry
}

end no_valid_placement_for_digits_on_45gon_l747_747047


namespace tan_triple_angle_l747_747004

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l747_747004


namespace probability_sum_is_10_l747_747456

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747456


namespace probability_f1_gt_0_l747_747830

/-- Given the function f(x) = -x^2 + a*x - b where a and b are uniformly chosen from the interval [0, 4], 
    the probability that f(1) > 0 is 3/4. -/
theorem probability_f1_gt_0 (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 4) (hb : 0 ≤ b ∧ b ≤ 4) :
  let f (x : ℝ) := -x^2 + a*x - b in
  (∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 4 ∧ (-1 + a - b > 0)) ->
  P(b ∈ [0, 3]) = 3/4 :=
begin
  -- Proof omitted
  sorry
end

end probability_f1_gt_0_l747_747830


namespace tan_triple_angle_l747_747007

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l747_747007


namespace petya_must_have_photo_files_on_portable_hard_drives_l747_747117

theorem petya_must_have_photo_files_on_portable_hard_drives 
    (H F P T : ℕ) 
    (h1 : H > F) 
    (h2 : P > T) 
    : ∃ x, x ≠ 0 ∧ x ≤ H :=
by
  sorry

end petya_must_have_photo_files_on_portable_hard_drives_l747_747117


namespace inscribed_circle_area_percent_l747_747182

theorem inscribed_circle_area_percent (a : ℝ) (h : a > 0) : 
  Real.floor ((π / 4) * 100 + 0.5) = 79 :=
by
  sorry

end inscribed_circle_area_percent_l747_747182


namespace tamika_vs_carlos_l747_747604

theorem tamika_vs_carlos :
  let tamika_sums := {21, 22, 23}
  let carlos_products := {24, 28, 42}
  ∀ t ∈ tamika_sums, ∀ c ∈ carlos_products, t ≤ c →
  (∃ p : ℚ, p = (0 : ℚ) / 9) :=
by
  let tamika_sums := {21, 22, 23}
  let carlos_products := {24, 28, 42}
  intros t ht c hc htc
  use 0 / 9
  sorry

end tamika_vs_carlos_l747_747604


namespace probability_heads_equals_l747_747213

theorem probability_heads_equals (p q: ℚ) (h1 : q = 1 - p) (h2 : (binomial 10 5) * p^5 * q^5 = (binomial 10 6) * p^6 * q^4) : p = 6 / 11 :=
by {
  sorry
}

end probability_heads_equals_l747_747213


namespace all_settings_weight_5040_l747_747566

def weight_per_piece_silverware := 4
def pieces_per_setting_silverware := 3
def weight_per_piece_plate := 12
def plates_per_setting := 2
def num_tables := 15
def settings_per_table := 8
def backup_settings := 20

def total_weight : ℕ :=
  let total_settings := num_tables * settings_per_table + backup_settings
  let weight_per_setting := (pieces_per_setting_silverware * weight_per_piece_silverware)
                           + (plates_per_setting * weight_per_piece_plate)
  in total_settings * weight_per_setting

theorem all_settings_weight_5040 : total_weight = 5040 := by
  sorry

end all_settings_weight_5040_l747_747566


namespace triangle_area_is_rational_l747_747365

-- Conditions: Given vertices of a triangle with specified integer relations
variables {x1 x2 x3 y1 y2 y3 : ℤ}

-- Defining conditions
def condition1 : Prop := x1 = y1 + 1
def condition2 : Prop := x2 = y2 - 1
def condition3 : Prop := x3 = y3 + 2

-- The theorem: The area of the triangle is rational
theorem triangle_area_is_rational (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  ∃ (q : ℚ), q = 1/2 * |(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) : ℤ)| :=
sorry

end triangle_area_is_rational_l747_747365


namespace find_omega_and_a_find_intervals_l747_747396

-- Definitions of vectors and function f
def vector_m (ω x : ℝ) : ℝ × ℝ := (2 * cos (ω * x), 1)
def vector_n (ω x a : ℝ) : ℝ × ℝ := (sqrt 3 * sin (ω * x) - cos (ω * x), a)
def f (ω x a : ℝ) : ℝ := (vector_m ω x).fst * (vector_n ω x a).fst + (vector_m ω x).snd * (vector_n ω x a).snd

theorem find_omega_and_a (ω a : ℝ) (h1 : ω > 0)
  (h2 : ∀ x : ℝ, (x ∈ ℝ) → f ω x a = 3)
  (h3 : ∀ x : ℝ, f ω x a = f ω (x + π) a):
  ω = 1 ∧ a = 2 :=
by sorry

theorem find_intervals (ω a : ℝ) (h1 : ω = 1) (h2 : a = 2) (x : ℝ) :
  ∃ k : ℤ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 :=
by sorry

end find_omega_and_a_find_intervals_l747_747396


namespace dice_sum_probability_l747_747475

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747475


namespace Mike_onions_grew_l747_747580

-- Define the data:
variables (nancy_onions dan_onions total_onions mike_onions : ℕ)

-- Conditions:
axiom Nancy_onions_grew : nancy_onions = 2
axiom Dan_onions_grew : dan_onions = 9
axiom Total_onions_grew : total_onions = 15

-- Theorem to prove:
theorem Mike_onions_grew (h : total_onions = nancy_onions + dan_onions + mike_onions) : mike_onions = 4 :=
by
  -- The proof is not provided, so we use sorry:
  sorry

end Mike_onions_grew_l747_747580


namespace Gloin_is_telling_the_truth_l747_747767

theorem Gloin_is_telling_the_truth 
  (Gnomes : Fin 10 → Prop) -- Gnomes are defined on finite set {0, ..., 9}
  (Knight : Prop) -- A knight is a truth-teller
  (Liar : Prop) -- A liar always lies
  (at_least_one_knight : ∃ n, Gnomes n = Knight)
  (first_nine_statements : ∀ n < 9, Gnomes n → "Knight_left n")
  (Gloin_statement : Gnomes 9 → "Knight_right 9")
  (Knight_left : ∀ n, n > 0 → Gnomes (n-1) = Knight → "Knight_left n")
  (Knight_right : ∀ n, n < 10 → ∃ k, n < k ∧ Gnomes k = Knight)
  : ∃ k, k = 9 ∧ Gnomes k = Knight  :=
by
  -- Proof goes here
  sorry

end Gloin_is_telling_the_truth_l747_747767


namespace problem_statement_l747_747220

noncomputable def is_finite_decimal (r : ℝ) : Prop :=
  r = 3.66666

noncomputable def is_repeating_decimal (r : ℝ) : Prop :=
  ∃ (n : ℕ) (d : ℤ), r = n / d ∧ (∃ k : ℕ, d = 10 ^ k ∨ d / 10^k ∈ ℤ ∧ abs(d / 10^k) ≥ 10)

theorem problem_statement : ¬ is_repeating_decimal 3.66666 :=
by
  -- Placeholder for the proof
  sorry

end problem_statement_l747_747220


namespace num_valid_license_plates_is_55125_l747_747849

-- Define the sets based on problem conditions
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}
def digits : Finset Int := {1, 2, 3, 4, 5}
def consonants : Finset Char := 
  Finset.filter (fun c => c ∈ {range 'A' 'Z'} ∧ c ∉ vowels) {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z'}

theorem num_valid_license_plates_is_55125 :
  vowels.card * digits.card * consonants.card * consonants.card * vowels.card = 55125 := by
  sorry

end num_valid_license_plates_is_55125_l747_747849


namespace max_K_value_l747_747969

theorem max_K_value (x1 x2 x3 x4 : ℝ) (hx1 : 0 ≤ x1) (hx1_ub : x1 ≤ 1) (hx2 : 0 ≤ x2) (hx2_ub : x2 ≤ 1) 
(hx3 : 0 ≤ x3) (hx3_ub : x3 ≤ 1) (hx4 : 0 ≤ x4) (hx4_ub : x4 ≤ 1) :
  let K := |x1 - x2| * |x1 - x3| * |x1 - x4| * |x2 - x3| * |x2 - x4| * |x3 - x4| in
  K ≤ (Real.sqrt 5) / 125 :=
by
  sorry

end max_K_value_l747_747969


namespace log_quadratic_increasing_l747_747626

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 3 * x - 4

def is_increasing_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem log_quadratic_increasing : 
  ∀ y, y = λ x : ℝ, real.logb 2 (quadratic_function x) →
  (∀ x, quadratic_function x > 0) → is_increasing_interval y 4 +∞ :=
sorry

end log_quadratic_increasing_l747_747626


namespace simplify_trig_expression_l747_747113

variables (α : ℝ)

theorem simplify_trig_expression :
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-π - α) * sin (9 * π / 2 + α) * tan (π + α)) = -1 :=
sorry

end simplify_trig_expression_l747_747113


namespace inequality_solution_l747_747784

theorem inequality_solution (x : ℝ) : 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2 / 3 := 
sorry

end inequality_solution_l747_747784


namespace problem_l747_747244

namespace MathProof

def is_non_decreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ (x y : ℕ+), x ≤ y → f x ≤ f y

def is_multiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ (m n : ℕ+), m.coprime n → f (m * n) = f m * f n

theorem problem (f : ℕ+ → ℕ+) 
  (H1 : is_non_decreasing f) 
  (H2 : is_multiplicative f) :
  f 8 * f 13 ≥ (f 10) ^ 2 := 
sorry

end MathProof

end problem_l747_747244


namespace perimeter_triangle_ABC_l747_747894

-- Define the centers of the circles
variables (P Q R S T U : Point) (r : ℝ := 1) 

-- Define the tangency and positioning conditions
variables 
(center_tangent_to_sides : ∀ (X : Point), X ∈ {P, Q, R, S, T, U} → ∃ (line : Line), tangent line X ∧ line ∈ {side_ABC (X)} ) -- tangency condition
(center_tangent_to_each_other : ∀ (X Y : Point), X ∈ {P, Q, R, S, T, U} ∧ Y ∈ {P, Q, R, S, T, U} → X ≠ Y → distance X Y = 2 * r) -- mutual tangency condition

-- Position and Structure Specification
(centres_form_hexagonal_pattern : hexagonal_pattern {P, Q, R, S, T, U}) -- Hexagonal structure condition
(circle_on_boundary_tangent_to_two_others : ∀ (X : Point), X ∈ {P, Q, R, S, T, U} → ∃ (Y Z : Point), Y ≠ Z ∧ tangent X Y ∧ tangent X Z )

-- Proving the perimeter of triangle ABC
theorem perimeter_triangle_ABC : 
  perimeter (triangle ABC) = 6 + 6 * Real.sqrt 3 := 
  sorry

end perimeter_triangle_ABC_l747_747894


namespace friday_can_determine_arrival_date_l747_747042

-- Define the conditions
def Robinson_crusoe (day : ℕ) : Prop := day % 365 = 0

-- Goal: Within 183 days, Friday can determine his arrival date.
theorem friday_can_determine_arrival_date : 
  (∀ day : ℕ, day < 183 → (Robinson_crusoe day ↔ ¬ Robinson_crusoe (day + 1)) ∨ (day % 365 = 0)) :=
sorry

end friday_can_determine_arrival_date_l747_747042


namespace circular_seating_l747_747705

noncomputable def uniqueCircularArrangements : Nat :=
  12.factorial / 12

theorem circular_seating (d r : Fin 6) : uniqueCircularArrangements = 39916800 := by
  -- Using conditions
  let senate_committee : Fin 12 := ⟨d, r⟩
  sorry

end circular_seating_l747_747705


namespace theta_value_l747_747995

theorem theta_value (θ ω: ℝ): (∀ x, Real.sin (ω * x + θ - Real.pi / 6) = Real.sin (ω * x + θ - Real.pi / 6 + Real.pi)) 
    → (∀ x, Real.sin (2 * (x + Real.pi / 6) + θ - Real.pi / 6) = - (Real.sin (2 * x + θ + Real.pi / 6))) 
    → θ = - Real.pi / 6 := 
begin 
    intros h_min_period h_odd_function,
    -- The solution strategy and the steps to solve will go here.
    -- This is essentially a sketch of the process and not the real proof.
    sorry 
end

end theta_value_l747_747995


namespace log_sum_l747_747808

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = r * a n for some r > 0

theorem log_sum (a : ℕ → ℝ) (n : ℕ) (h1 : n ≥ 1) (h2 : ∀ m ≥ 3, a 5 * a (2 * m - 5) = 10^(2 * m)) (h3 : geometric_sequence a) :
  (finset.range n).sum (λ i, real.log (a (i + 1))) = n * (n + 1) / 2 := 
sorry

end log_sum_l747_747808


namespace Leibniz_differentiation_formula_l747_747099

-- Defining the Leibniz formula for differentiation of the product of two functions
theorem Leibniz_differentiation_formula
  (f g : ℝ → ℝ) -- Functions f and g defined on ℝ
  (N : ℕ) -- Natural number N
  :
  (∀ N, deriv^[N] (λ x, (f x) * (g x)) = 
  ∑ n in finset.range (N + 1), (nat.choose N n) * (deriv^[n] f) * (deriv^[N - n] g)) := 
sorry

end Leibniz_differentiation_formula_l747_747099


namespace TriangleAngle_l747_747165

noncomputable def angleQPS (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
  (PQ QR PR RS : ℝ) (h₁ : PQ = QR) (h₂ : PR = RS)
  (R_inside_PQR : Inside R P Q) (anglePQR : ℝ) (anglePRS : ℝ) : ℝ :=
  if anglePQR = 50 ∧ anglePRS = 110 then
    30
  else
    sorry

-- Formal statement of the problem
theorem TriangleAngle (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
  (PQ QR PR RS : ℝ) (h₁ : PQ = QR) (h₂ : PR = RS)
  (R_inside_PQR : Inside R P Q) (anglePQR : ℝ) (anglePRS : ℝ) :
  anglePQR = 50 → anglePRS = 110 → angleQPS P Q R S PQ QR PR RS h₁ h₂ R_inside_PQR anglePQR anglePRS = 30 := 
by
  intros
  simp only [angleQPS]
  split_ifs
  · exact h
  · sorry

end TriangleAngle_l747_747165


namespace discount_percentage_clearance_sale_l747_747261

theorem discount_percentage_clearance_sale
  (SP : ℝ)
  (gain_percent_original : ℝ)
  (gain_percent_sale : ℝ)
  (original_SP : SP = 30)
  (original_gain : gain_percent_original = 0.15)
  (sale_gain : gain_percent_sale = 0.035) :
  let CP := SP / (1 + gain_percent_original) in
  let SP_sale := CP * (1 + gain_percent_sale) in
  let discount := SP - SP_sale in
  let discount_percentage := (discount / SP) * 100 in
  discount_percentage = 10 :=
by
  sorry

end discount_percentage_clearance_sale_l747_747261


namespace re_part_max_value_l747_747596

noncomputable def complex := {z : ℂ // |z| = 1}

theorem re_part_max_value (z w : ℂ) (hz : |z| = 1) (hw : |w| = 1) (hzw : z * conj w + conj z * w = 2) :
  real.re (z + w) ≤ 2 :=
sorry

end re_part_max_value_l747_747596


namespace calculate_remaining_soup_adult_feed_l747_747708
noncomputable def remaining_soup_adults_feed (initial_cans num_children : ℕ) (soup_per_adult soup_per_child leftover_fraction : ℕ → ℚ) : ℚ :=
  let leftover_cans := ⌈num_children / soup_per_child 1⌉
  let unused_cans := initial_cans - leftover_cans
  let remaining_soup := unused_cans + leftover_fraction leftover_cans
  (remaining_soup * soup_per_adult 1)

theorem calculate_remaining_soup_adult_feed : remaining_soup_adults_feed 8 20 (λ x, 4) (λ x, 6) (λ x, x / 3) = 20 :=
by
  sorry

end calculate_remaining_soup_adult_feed_l747_747708


namespace clock_correct_time_fraction_l747_747704

/-- 
  A 24-hour digital clock displays the hour and minute of a day, 
  counting from 00:00 to 23:59. However, due to a glitch, whenever 
  the clock is supposed to display a '2', it mistakenly displays a '5'.

  Prove that the fraction of a day during which the clock shows the correct 
  time is 23/40.
-/
theorem clock_correct_time_fraction :
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  (correct_hours / total_hours) * (correct_minutes / total_minutes) = 23 / 40 :=
by
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  have h1 : correct_hours = 18 := rfl
  have h2 : correct_minutes = 46 := rfl
  have h3 : 18 / 24 = 3 / 4 := by norm_num
  have h4 : 46 / 60 = 23 / 30 := by norm_num
  have h5 : (3 / 4) * (23 / 30) = 23 / 40 := by norm_num
  exact h5

end clock_correct_time_fraction_l747_747704


namespace puppy_cost_l747_747048

variable (P : ℕ)

theorem puppy_cost (hc : 2 * 50 = 100) (hd : 3 * 100 = 300) (htotal : 2 * 50 + 3 * 100 + 2 * P = 700) : P = 150 :=
by
  sorry

end puppy_cost_l747_747048


namespace probability_sum_10_l747_747490

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747490


namespace P_days_completes_job_l747_747962

def work_rate_P (P_days : ℕ) : ℝ := 1 / P_days
def work_rate_Q (P_days : ℕ) : ℝ := 1 / (3 * P_days)
def combined_work_rate (P_days : ℕ) : ℝ := work_rate_P P_days + work_rate_Q P_days

theorem P_days_completes_job :
  (∃ P_days : ℕ, combined_work_rate P_days = 1 / 3) → (P_days = 4) := 
by
  sorry

end P_days_completes_job_l747_747962


namespace largest_integer_base_7_l747_747551

theorem largest_integer_base_7 :
  ∃ N : ℕ, 7^(3/2) ≤ N ∧ N < 7^2 ∧ N = 48 ∧ nat_base_7_repr N = "66" :=
sorry

end largest_integer_base_7_l747_747551


namespace angle_sum_210_degrees_l747_747964

theorem angle_sum_210_degrees
  (A B R D C : Point) 
  (h_circle : SameCircle A B R D C) 
  (arc_BR : arcMeasure B R = 48) 
  (arc_RD : arcMeasure R D = 52) 
  (S : Point) 
  (h_intersection : line_intersection (line_through A D) (line_through B C) S ) :
  angle_measure S + angle_measure R = 210 :=
begin
  sorry
end

end angle_sum_210_degrees_l747_747964


namespace bob_walk_distance_l747_747251

theorem bob_walk_distance (hex_side : ℝ) (walk_distance : ℝ) (hex_angle: ℝ):
  (hex_side = 3) → (walk_distance = 7) → (hex_angle = 2 * π / 3) → 
  let final_point := (1, real.sqrt 3) in dist ⟨0, 0⟩ final_point = 2 :=
by
  intros h_side h_distance h_angle
  have hex_prop : is_regular_hexagon hex_side := sorry
  let final_point := (1, real.sqrt 3)
  exact sorry

end bob_walk_distance_l747_747251


namespace tan_triple_angle_l747_747002

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l747_747002


namespace remainder_is_six_l747_747187

noncomputable def x : ℝ := 96.15 * 40
def y : ℝ := 40
def decimal_part := 0.15

def remainder (x y : ℝ) : ℝ := x - (x / y) * y

theorem remainder_is_six (h1 : x / y = 96.15) (h2 : y = 40) :
  remainder x y = 6 := by
  sorry

end remainder_is_six_l747_747187


namespace john_milk_l747_747540

theorem john_milk : (5:ℤ) - (17 / 4:ℚ) = (3 / 4:ℚ) :=
by
  norm_num
  rw [sub_eq_add_neg, add_assoc, neg_div, neg_eq_iff_add_eq_zero, add_zero]
  norm_cast
  rw [←sub_eq_add_neg]
  norm_num
  sorry

end john_milk_l747_747540


namespace ratio_third_to_first_l747_747649

theorem ratio_third_to_first (F S T : ℕ) (h1 : F = 33) (h2 : S = 4 * F) (h3 : (F + S + T) / 3 = 77) :
  T / F = 2 :=
by
  sorry

end ratio_third_to_first_l747_747649


namespace hexagon_monochromatic_triangle_probability_l747_747768

open Classical

-- Define the total number of edges in the hexagon
def total_edges : ℕ := 15

-- Define the number of triangles from 6 vertices
def total_triangles : ℕ := Nat.choose 6 3

-- Define the probability that a given triangle is not monochromatic
def prob_not_monochromatic_triangle : ℚ := 3 / 4

-- Calculate the probability of having at least one monochromatic triangle
def prob_at_least_one_monochromatic_triangle : ℚ := 
  1 - (prob_not_monochromatic_triangle ^ total_triangles)

theorem hexagon_monochromatic_triangle_probability :
  abs ((prob_at_least_one_monochromatic_triangle : ℝ) - 0.9968) < 0.0001 :=
by
  sorry

end hexagon_monochromatic_triangle_probability_l747_747768


namespace min_value_abs_expression_l747_747852

theorem min_value_abs_expression {p x : ℝ} (hp1 : 0 < p) (hp2 : p < 15) (hx1 : p ≤ x) (hx2 : x ≤ 15) :
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
sorry

end min_value_abs_expression_l747_747852


namespace greatest_m_div_36_and_7_l747_747239

def reverse_digits (m : ℕ) : ℕ :=
  let d1 := (m / 1000) % 10
  let d2 := (m / 100) % 10
  let d3 := (m / 10) % 10
  let d4 := m % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_m_div_36_and_7
  (m : ℕ) (n : ℕ := reverse_digits m)
  (h1 : 1000 ≤ m ∧ m < 10000)
  (h2 : 1000 ≤ n ∧ n < 10000)
  (h3 : 36 ∣ m ∧ 36 ∣ n)
  (h4 : 7 ∣ m) :
  m = 9828 := 
sorry

end greatest_m_div_36_and_7_l747_747239


namespace ball_hits_ground_time_l747_747706

open Real

theorem ball_hits_ground_time {t : ℝ} :
  (-20 * t^2 - 40 * t + 50 = 0) ↔ t = -1 + sqrt(14) / 2 :=
by
  sorry

end ball_hits_ground_time_l747_747706


namespace complex_numbers_condition_l747_747863

theorem complex_numbers_condition (z1 z2 : ℂ) (λ : ℝ) (hk : k = 1 ∨ k = 2)
  (hz1_nonzero : z1 ≠ 0) (hz2_nonzero : z2 ≠ 0) (mod_eq : |z1| = |z2|) (hλ_pos : λ > 0) :
  (z1 + z2)^2 = λ^2 * z1 * z2 ↔ |z1 + z2| = λ * |if k = 1 then z1 else z2| := by
  sorry

end complex_numbers_condition_l747_747863


namespace probability_sum_is_10_l747_747453

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747453


namespace find_middle_number_l747_747219

-- Defining x and the three numbers according to the given ratio.
variable {x : ℝ}
def first_number := 2 * x
def middle_number := 3 * x
def third_number := 4 * x

-- Defining the given condition: the sum of the squares of the extremes is 180.
def sum_of_squares_of_extremes := (first_number ^ 2) + (third_number ^ 2) = 180

-- The theorem to be proven: Given the condition, the middle number is 9.
theorem find_middle_number (h : sum_of_squares_of_extremes) : middle_number = 9 := by
  -- Proof will be provided here
  sorry

end find_middle_number_l747_747219


namespace satisfies_log_a_satisfies_exp_a_satisfies_x_minus_one_over_x_satisfies_piecewise_fn_l747_747864

def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def piecewise_fn (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else - (1 / x)

def inv_neg_transformation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = - f x

theorem satisfies_log_a (a : ℝ) (h : 0 < a ∧ a ≠ 1) : inv_neg_transformation (log_a a) :=
  sorry

theorem satisfies_exp_a (a : ℝ) (h : 0 < a ∧ a ≠ 1) : ¬ inv_neg_transformation (fun x => a^x) :=
  sorry

theorem satisfies_x_minus_one_over_x : inv_neg_transformation (fun x => x - 1 / x) :=
  sorry

theorem satisfies_piecewise_fn : inv_neg_transformation piecewise_fn :=
  sorry

end satisfies_log_a_satisfies_exp_a_satisfies_x_minus_one_over_x_satisfies_piecewise_fn_l747_747864


namespace tetrahedron_cannot_fit_in_parallelepiped_l747_747952

-- Define the notion of a 'good' polyhedron
structure GoodPolyhedron (α : Type*) :=
(volume : α)
(surface_area : α)
(is_good : volume = surface_area)

-- Definitions for a good tetrahedron and a good parallelepiped
def GoodTetrahedron (α : Type*) := GoodPolyhedron α
def GoodParallelepiped (α : Type*) := GoodPolyhedron α

-- Statement: it is impossible to place a good tetrahedron inside a good parallelepiped
theorem tetrahedron_cannot_fit_in_parallelepiped {α : Type*} [linear_ordered_field α] :
  ∀ (tet : GoodTetrahedron α) (par : GoodParallelepiped α), false :=
by sorry

end tetrahedron_cannot_fit_in_parallelepiped_l747_747952


namespace combined_weight_of_boxes_l747_747911

-- Defining the weights of each box as constants
def weight1 : ℝ := 2.5
def weight2 : ℝ := 11.3
def weight3 : ℝ := 5.75
def weight4 : ℝ := 7.2
def weight5 : ℝ := 3.25

-- The main theorem statement
theorem combined_weight_of_boxes : weight1 + weight2 + weight3 + weight4 + weight5 = 30 := by
  sorry

end combined_weight_of_boxes_l747_747911


namespace find_smallest_period_l747_747304

noncomputable def smallest_period (f : ℝ → ℝ) : ℝ :=
  if h : ∃ p > 0, ∀ x, f(x) = f(x + p) then classical.some h else 0

theorem find_smallest_period (f : ℝ → ℝ)
  (h : ∀ x, f(x + 6) + f(x - 6) = f(x)) : smallest_period f = 36 := by
  sorry

end find_smallest_period_l747_747304


namespace final_temp_fahrenheit_correct_l747_747872

noncomputable def initial_temp_celsius : ℝ := 50
noncomputable def conversion_c_to_f (c: ℝ) : ℝ := (c * 9 / 5) + 32
noncomputable def final_temp_celsius := initial_temp_celsius / 2

theorem final_temp_fahrenheit_correct : conversion_c_to_f final_temp_celsius = 77 :=
  by sorry

end final_temp_fahrenheit_correct_l747_747872


namespace prime_divisor_fourth_power_gt_cubic_l747_747719

-- Stating the problem in Lean 4
theorem prime_divisor_fourth_power_gt_cubic (n : ℕ) (h : ∀ (d : ℕ), d ∣ n → ¬ (n^2 ≤ d^4 ∧ d^4 ≤ n^3)) : 
  ∃ (p : ℕ), prime p ∧ p ∣ n ∧ p^4 > n^3 :=
sorry

end prime_divisor_fourth_power_gt_cubic_l747_747719


namespace area_triangle_PQR_l747_747296

open Real

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem area_triangle_PQR :
  ∀ (P Q R : ℝ × ℝ),
    distance P.fst P.snd Q.fst Q.snd = 3 →
    distance Q.fst Q.snd R.fst R.snd = 5 →
    (∃ (P₁ Q₁ R₁ : ℝ × ℝ), P₁ = (-√2, 1) ∧ Q₁ = (0, 2) ∧ R₁ = (√6, 3) ∧
       P = P₁ ∧ Q = Q₁ ∧ R = R₁) →
    0.5 * abs ((P.fst)*(Q.snd - R.snd) + (Q.fst)*(R.snd - P.snd) + (R.fst)*(P.snd - Q.snd)) = (√6 - √2) :=
by
  sorry

end area_triangle_PQR_l747_747296


namespace sum_of_coefficients_l747_747217

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 : ℝ) :
  (∀ x, (x^2 + 1) * (x - 2)^9 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 +
        a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7 + a8 * (x - 1)^8 + a9 * (x - 1)^9 + a10 * (x - 1)^10 + a11 * (x - 1)^11) →
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 = 2 := 
sorry

end sum_of_coefficients_l747_747217


namespace Mickey_horses_per_week_l747_747315

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l747_747315


namespace egg_condition_difference_l747_747570

theorem egg_condition_difference :
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  in perfect_condition - cracked_eggs = 9 :=
by
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  show perfect_condition - cracked_eggs = 9, by sorry

end egg_condition_difference_l747_747570


namespace greatest_value_a_plus_b_l747_747176

theorem greatest_value_a_plus_b (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) : a + b = 2 * Real.sqrt 55 :=
by
  sorry

end greatest_value_a_plus_b_l747_747176


namespace surface_area_of_largest_cube_l747_747020

-- Definitions for the conditions
def width : ℝ := 12
def length : ℝ := 16
def height : ℝ := 14

-- The problem statement
theorem surface_area_of_largest_cube :
  6 * (min width (min length height))^2 = 864 := 
sorry

end surface_area_of_largest_cube_l747_747020


namespace table_to_chair_ratio_l747_747913

noncomputable def price_chair : ℤ := 20
noncomputable def price_table : ℤ := 60
noncomputable def price_couch : ℤ := 300

theorem table_to_chair_ratio 
  (h1 : price_couch = 300)
  (h2 : price_couch = 5 * price_table)
  (h3 : price_chair + price_table + price_couch = 380)
  : price_table / price_chair = 3 := 
by 
  sorry

end table_to_chair_ratio_l747_747913


namespace find_2023rd_digit_of_11_div_13_l747_747330

noncomputable def decimal_expansion_repeating (n d : Nat) : List Nat := sorry

noncomputable def decimal_expansion_digit (n d pos : Nat) : Nat :=
  let repeating_block := decimal_expansion_repeating n d
  repeating_block.get! ((pos - 1) % repeating_block.length)

theorem find_2023rd_digit_of_11_div_13 :
  decimal_expansion_digit 11 13 2023 = 8 := by
  sorry

end find_2023rd_digit_of_11_div_13_l747_747330


namespace mutually_exclusive_one_two_odd_l747_747678

-- Define the event that describes rolling a fair die
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- Event: Exactly one die shows an odd number -/
def exactly_one_odd (d1 d2 : ℕ) : Prop :=
  (is_odd d1 ∧ ¬ is_odd d2) ∨ (¬ is_odd d1 ∧ is_odd d2)

/-- Event: Exactly two dice show odd numbers -/
def exactly_two_odd (d1 d2 : ℕ) : Prop :=
  is_odd d1 ∧ is_odd d2

/-- Main theorem: Exactly one odd number and exactly two odd numbers are mutually exclusive but not converse-/
theorem mutually_exclusive_one_two_odd (d1 d2 : ℕ) :
  (exactly_one_odd d1 d2 ∧ ¬ exactly_two_odd d1 d2) ∧
  (¬ exactly_one_odd d1 d2 ∧ exactly_two_odd d1 d2) ∧
  (exactly_one_odd d1 d2 ∨ exactly_two_odd d1 d2) :=
by
  sorry

end mutually_exclusive_one_two_odd_l747_747678


namespace smallest_five_digit_palindrome_divisible_by_five_l747_747671

def is_palindrome (n : ℕ) : Prop :=
  (n.to_string = n.to_string.reverse)

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

theorem smallest_five_digit_palindrome_divisible_by_five :
  ∃ (n : ℕ), is_palindrome n ∧ is_five_digit n ∧ is_divisible_by_five n ∧ ∀ m, is_palindrome m ∧ is_five_digit m ∧ is_divisible_by_five m → n ≤ m :=
  ∃ (n : ℕ), is_palindrome n ∧ is_five_digit n ∧ is_divisible_by_five n ∧ ∀ m, is_palindrome m ∧ is_five_digit m ∧ is_divisible_by_five m → n = 50005 := 
  sorry

end smallest_five_digit_palindrome_divisible_by_five_l747_747671


namespace part_a_l747_747544

theorem part_a (K : Type*) [field K] (p : ℕ) [char_p K p] (hp : p % 4 = 1) : 
  ∃ a : K, a^2 = -1 := 
sorry

end part_a_l747_747544


namespace probability_sum_10_l747_747485

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747485


namespace probability_p_eq_l747_747207

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l747_747207


namespace dice_sum_probability_l747_747445

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747445


namespace mr_white_money_l747_747579

-- Definitions based on the conditions
def money_mr_black := 75
def money_mr_green := money_mr_black / 4
def money_mr_white := money_mr_green * 1.2

-- Proof statement
theorem mr_white_money :
  money_mr_white = 22.5 :=
sorry

end mr_white_money_l747_747579


namespace problem_statement_l747_747079

def isEquilateralTriangle (A B C : ℂ) : Prop :=
  A - B ≃ B - C ∧ B - C ≃ C - A ∧ ∃ θ : ℂ, θ * (B - A) = C - A

noncomputable def centerOfEquilateralTriangle (A B C : ℂ) [isEquilateralTriangle A B C] : ℂ :=
  (A + B + C) / 3

def isConvexQuadrilateral (A B C D : ℂ) : Prop :=
  (ConvexHull R {A, B, C, D}) = set.univ

def perpendicular (v w : ℂ) : Prop :=
  Complex.Im(v * complex.conj w) = 0

theorem problem_statement (A B C D O1 O2 O3 O4 : ℂ)
  (h1 : isConvexQuadrilateral A B C D)
  (h2 : dist A C = dist B D)
  (h3 : isEquilateralTriangle A B O1)
  (h4 : isEquilateralTriangle B C O2)
  (h5 : isEquilateralTriangle C D O3)
  (h6 : isEquilateralTriangle D A O4)
  (h7 : centerOfEquilateralTriangle A B O1 = O1)
  (h8 : centerOfEquilateralTriangle B C O2 = O2)
  (h9 : centerOfEquilateralTriangle C D O3 = O3)
  (h10 : centerOfEquilateralTriangle D A O4 = O4) :
  perpendicular (O1 - O3) (O2 - O4) := sorry

end problem_statement_l747_747079


namespace probability_digit_5_among_first_n_digits_l747_747721

theorem probability_digit_5_among_first_n_digits (n : ℕ) : 
  let P := 1 - (9 / 10) ^ n in 
  P = 1 - (9 / 10) ^ n :=
by
  sorry

end probability_digit_5_among_first_n_digits_l747_747721


namespace common_tangent_parallel_to_AD_l747_747045

noncomputable theory

-- Definitions based on problem conditions
variables {A B C D P : Type*} [Parallelogram A B C D] 
variable {P : Point (Plane A B C D)}
hypothesis angle_condition : angle P D A = angle P B A
noncomputable def Omega : Type* := Excircle (Triangle P A B)
noncomputable def omega : Type* := Incircle (Triangle P C D)

-- Theorem to prove
theorem common_tangent_parallel_to_AD : 
  ∃ (Tangent : Line), is_common_tangent Omega omega Tangent ∧ is_parallel Tangent (Line A D) :=
sorry

end common_tangent_parallel_to_AD_l747_747045


namespace minimum_value_l747_747948

noncomputable def expr (x y : ℝ) := (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y)

theorem minimum_value :
  (∀ x y : ℝ, 0 < x → 0 < y → expr x y ≥ (2 + 2 * Real.root 3 8)) :=
  sorry

end minimum_value_l747_747948


namespace minimum_value_l747_747945

noncomputable def expr (x y : ℝ) := (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y)

theorem minimum_value :
  (∀ x y : ℝ, 0 < x → 0 < y → expr x y ≥ (2 + 2 * Real.root 3 8)) :=
  sorry

end minimum_value_l747_747945


namespace problem_statement_l747_747933

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
sorry

end problem_statement_l747_747933


namespace number_of_female_only_child_students_l747_747605

def students : Finset ℕ := Finset.range 21 -- Set of students with attendance numbers from 1 to 20

def female_students : Finset ℕ := {1, 3, 4, 6, 7, 10, 11, 13, 16, 17, 18, 20}

def only_child_students : Finset ℕ := {1, 4, 5, 8, 11, 14, 17, 20}

def common_students : Finset ℕ := female_students ∩ only_child_students

theorem number_of_female_only_child_students :
  common_students.card = 5 :=
by
  sorry

end number_of_female_only_child_students_l747_747605


namespace subset_exists_l747_747076

theorem subset_exists (X : Type) (S : set (set X)) (h_cardX : n = Fintype.card X)
  (h_cardS : ∀ s ∈ S, Fintype.card s = 3)
  (h_common : ∀ {s1 s2 : set X}, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → Fintype.card (s1 ∩ s2) ≤ 1):
  ∃ (A : set X), Fintype.card A ≥ ⌊sqrt (2 * n)⌋₊ ∧ ∀ s ∈ S, ¬(s ⊆ A) :=
by sorry

end subset_exists_l747_747076


namespace price_store_a_price_store_b_l747_747231

/-- Price of hardcover notebooks at Store A --/
theorem price_store_a :
  ∃ (x : ℕ), x ≠ 0 ∧ 
    (240 / x = 195 / (x - 3)) ∧ x = 16 :=
begin
  sorry
end

/-- Original price of hardcover notebooks at Store B --/
theorem price_store_b :
  ∃ (y m : ℕ), y ≠ 0 ∧ m ≠ 0 ∧ m < 30 ∧ m + 5 ≥ 30 ∧ 
    (m * y = (m + 5) * (y - 3)) ∧ y = 18 ∧ m = 25 :=
begin
  sorry
end

end price_store_a_price_store_b_l747_747231


namespace problem1_problem2_l747_747395

-- Proof problem (1)
theorem problem1 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 1 < x ∧ x < 2} ∧ m = 1 →
  (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := 
by 
  sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 2 * m - 1 < x ∧ x < m + 1} →
  (B ⊆ A ↔ (m ≥ 2 ∨ (-1 ≤ m ∧ m < 2))) := 
by 
  sorry

end problem1_problem2_l747_747395


namespace total_amount_received_l747_747201

theorem total_amount_received (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ) (A : ℝ) 
  (hCI : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (hCI_value : CI = 370.80)
  (hr : r = 0.06)
  (hn : n = 1)
  (ht : t = 2)
  (hP : P = 3000)
  (hP_value : P = CI / 0.1236) :
  A = P + CI := 
by 
sorry

end total_amount_received_l747_747201


namespace strictly_positive_integer_le_36_l747_747329

theorem strictly_positive_integer_le_36 (n : ℕ) (h_pos : n > 0) :
  (∀ a : ℤ, (a % 2 = 1) → (a * a ≤ n) → (a ∣ n)) → n ≤ 36 := by
  sorry

end strictly_positive_integer_le_36_l747_747329


namespace cubing_identity_l747_747411

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l747_747411


namespace find_x_of_product_eq_72_l747_747340

theorem find_x_of_product_eq_72 (x : ℝ) (h : 0 < x) (hx : x * ⌊x⌋₊ = 72) : x = 9 :=
sorry

end find_x_of_product_eq_72_l747_747340


namespace greatest_possible_individual_award_l747_747197

theorem greatest_possible_individual_award (total_prize : ℕ) (total_winners : ℕ) (min_award : ℕ) (portion_prize : ℕ) (portion_winners : ℕ) :
  total_prize = 800 →
  total_winners = 20 →
  min_award = 20 →
  portion_prize = 2 / 5 →
  portion_winners = 3 / 5 →
  ∃ greatest_individual_award : ℕ, greatest_individual_award = 420 :=
by
  intros _ _ _ _ _ 
  have : 3 / 5 * 20 = 12 := by norm_num
  have : 2 / 5 * 800 = 320 := by norm_num
  have : 20 - 12 = 8 := by norm_num
  have : 8 * 20 = 160 := by norm_num
  have : 800 - 160 = 640 := by norm_num
  have : 11 * 20 = 220 := by norm_num
  have : 640 - 220 = 420 := by norm_num
  use 420
  sorry

end greatest_possible_individual_award_l747_747197


namespace xyz_sum_equation_l747_747078

noncomputable def xyz_expression (x y z : ℝ) : Prop :=
  x^2 + xy + y^2 = 75 ∧ y^2 + yz + z^2 = 36 ∧ z^2 + xz + x^2 = 111

theorem xyz_sum_equation (x y z : ℝ) (h : xyz_expression x y z) :
  xy + yz + xz = 60 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end xyz_sum_equation_l747_747078


namespace inequality_one_solution_inequality_two_solution_cases_l747_747980

-- Setting up the problem for the first inequality
theorem inequality_one_solution :
  {x : ℝ | -1 ≤ x ∧ x ≤ 4} = {x : ℝ |  -x ^ 2 + 3 * x + 4 ≥ 0} :=
sorry

-- Setting up the problem for the second inequality with different cases of 'a'
theorem inequality_two_solution_cases (a : ℝ) :
  (a = 0 ∧ {x : ℝ | true} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a > 0 ∧ {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a < 0 ∧ {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0}) :=
sorry

end inequality_one_solution_inequality_two_solution_cases_l747_747980


namespace speed_of_train_in_kmh_l747_747222

def train_length : ℕ := 200
def time_to_cross : ℕ := 9

theorem speed_of_train_in_kmh : (train_length / time_to_cross) * 3.6 = 80 :=
by
  sorry

end speed_of_train_in_kmh_l747_747222


namespace probability_sum_is_10_l747_747503

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747503


namespace rule1_rule2_rule3_all_rules_correct_l747_747318

def op_add (a b : ℤ) : ℤ := a + b + 1
def op_sub (a b : ℤ) : ℤ := a - b - 1

theorem rule1 (a b : ℤ) : op_add a b = op_add b a := 
by {
  sorry
}

theorem rule2 (a b c : ℤ) : 
  op_add a (op_add b c) = op_add (op_add a b) c :=
by {
  sorry
}

theorem rule3 (a b c : ℤ) : 
  op_sub a (op_add b c) = op_sub (op_sub a b) c :=
by {
  sorry
}

theorem all_rules_correct (a b c : ℤ) : 
  rule1 a b ∧ rule2 a b c ∧ rule3 a b c :=
by {
  exact ⟨rule1 a b, rule2 a b c, rule3 a b c⟩
}

end rule1_rule2_rule3_all_rules_correct_l747_747318


namespace simplify_expression_l747_747744

variable {p q r : ℚ}

theorem simplify_expression (hp : p ≠ 2) (hq : q ≠ 5) (hr : r ≠ 7) :
  ( (p - 2) / (7 - r) * (q - 5) / (2 - p) * (r - 7) / (5 - q) ) = -1 := by
  sorry

end simplify_expression_l747_747744


namespace original_right_triangle_angles_l747_747753

theorem original_right_triangle_angles (a b c s_a s_b s_c : ℝ) (h1 : a^2 + b^2 = c^2)
    (h2 : s_c = c / 2) (h3 : s_a = √((b^2 + (a / 2)^2)))
    (h4 : s_b = √((a^2 + (b / 2)^2))) :
    let β := real.arctan (b / a) * 180 / real.pi in
    let α := 90 - β in
    |α - 35.3| < 0.1 ∧ |β - 54.7| < 0.1 :=
by
  sorry

end original_right_triangle_angles_l747_747753


namespace gcd_of_78_and_36_l747_747622

theorem gcd_of_78_and_36 : Int.gcd 78 36 = 6 := by
  sorry

end gcd_of_78_and_36_l747_747622


namespace seventh_graders_problems_l747_747595

theorem seventh_graders_problems (n : ℕ) (S : ℕ) (a : ℕ) (h1 : a > (S - a) / 5) (h2 : a < (S - a) / 3) : n = 5 :=
  sorry

end seventh_graders_problems_l747_747595


namespace floor_100S_eq_157_l747_747789

noncomputable def sequence (k : ℕ) : ℝ :=
  if h : (k ≥ 1) then 2 ^ k else 0

noncomputable def series : ℝ :=
  ∑' k, if h : (k ≥ 1) then
    Real.arccos (
      (2 * (sequence k) ^ 2 - 6 * (sequence k) + 5) /
      Real.sqrt ((sequence k) ^ 2 - 4 * (sequence k) + 5) *
      Real.sqrt (4 * (sequence k) ^ 2 - 8 * (sequence k) + 5)
    )
  else 0

theorem floor_100S_eq_157 : ⌊100 * series⌋ = 157 :=
sorry

end floor_100S_eq_157_l747_747789


namespace find_y_l747_747591

theorem find_y (AB BC : ℕ) (y x : ℕ) 
  (h1 : AB = 3 * y)
  (h2 : BC = 2 * x)
  (h3 : AB * BC = 2400) 
  (h4 : AB * BC = 6 * x * y) :
  y = 20 := by
  sorry

end find_y_l747_747591


namespace polar_cartesian_conversion_and_line_intersection_l747_747892

noncomputable theory

-- Polar to Cartesian conversion
def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

def circle_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 4*y = 0

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Main statement
theorem polar_cartesian_conversion_and_line_intersection :
  (∀ (rho theta : ℝ), polar_to_cartesian rho (theta + (π / 4)) = (4 * Real.sqrt 2 * Real.cos (theta + π / 4), 4 * Real.sqrt 2 * Real.sin (theta + π / 4)))
  → ∃ t1 t2 : ℝ, 
      ((2 + (Real.sqrt 2 / 2) * t1, (Real.sqrt 2 / 2) * t1) ∈ {p : ℝ × ℝ | circle_cartesian p.1 p.2})
      ∧ ((2 + (Real.sqrt 2 / 2) * t2, (Real.sqrt 2 / 2) * t2) ∈ {p : ℝ × ℝ | circle_cartesian p.1 p.2})
      ∧ t1 + t2 = -2 * Real.sqrt 2 
      ∧ t1 * t2 = -4
      ∧ (1 / Real.abs t1 + 1 / Real.abs t2 = Real.sqrt 6 / 2) :=
sorry

end polar_cartesian_conversion_and_line_intersection_l747_747892


namespace suitable_for_comprehensive_survey_l747_747190

-- Define the conditions
def is_comprehensive_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  is_specific_group ∧ (group_size < 100)  -- assuming "small" means fewer than 100 individuals/items

def is_sampling_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  ¬is_comprehensive_survey group_size is_specific_group

-- Define the surveys
def option_A (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_comprehensive_survey group_size is_specific_group

def option_B (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_C (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_D (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

-- Question: Which of the following surveys is suitable for a comprehensive survey given conditions
theorem suitable_for_comprehensive_survey :
  ∀ (group_size_A group_size_B group_size_C group_size_D : ℕ) 
    (is_specific_group_A is_specific_group_B is_specific_group_C is_specific_group_D : Bool),
  option_A group_size_A is_specific_group_A ↔ 
  ((option_B group_size_B is_specific_group_B = false) ∧ 
   (option_C group_size_C is_specific_group_C = false) ∧ 
   (option_D group_size_D is_specific_group_D = false)) :=
by
  sorry

end suitable_for_comprehensive_survey_l747_747190


namespace interest_rate_proof_l747_747148

noncomputable def compound_interest_rate (P A : ℝ) (t n : ℕ) : ℝ :=
  (((A / P)^(1 / (n * t))) - 1) * n

theorem interest_rate_proof :
  ∀ P A : ℝ, ∀ t n : ℕ, P = 1093.75 → A = 1183 → t = 2 → n = 1 →
  compound_interest_rate P A t n = 0.0399 :=
by
  intros P A t n hP hA ht hn
  rw [hP, hA, ht, hn]
  unfold compound_interest_rate
  sorry

end interest_rate_proof_l747_747148


namespace A_20_is_10946_l747_747740

def fib : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 2) := fib n + fib (n + 1)

theorem A_20_is_10946 : fib 20 = 10946 := by
  sorry

end A_20_is_10946_l747_747740


namespace find_k_l747_747806

theorem find_k (k : ℝ) (h : ∀ x y : ℝ, (x, y) = (-2, -1) → y = k * x + 2) : k = 3 / 2 :=
sorry

end find_k_l747_747806


namespace minimum_value_sqrt_inequality_l747_747942

theorem minimum_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (sqrt ((x^2 + y^2) * (4 * x^2 + y^2)) / (x * y)) ≥ 3 :=
sorry

end minimum_value_sqrt_inequality_l747_747942


namespace rows_needed_correct_l747_747777

variable (pencils rows_needed : Nat)

def total_pencils : Nat := 35
def pencils_per_row : Nat := 5
def rows_expected : Nat := 7

theorem rows_needed_correct : rows_needed = total_pencils / pencils_per_row →
  rows_needed = rows_expected := by
  sorry

end rows_needed_correct_l747_747777


namespace P_three_equals_thirteen_no_n_for_P_equals_2002_l747_747344

def P (n : ℕ) : ℕ :=
  let pairs := { (a, b) : ℕ × ℕ // 
    (a > n) ∧ 
    (1 < (a : ℚ) / b) ∧ ((a : ℚ) / b < 2) ∧ 
    (2 < (b : ℚ) / n) ∧ ((b : ℚ) / n < 3) }
  pairs.to_finset.card

theorem P_three_equals_thirteen : P 3 = 13 := by
  sorry

theorem no_n_for_P_equals_2002 : ¬ ∃ n : ℕ, P n = 2002 := by
  sorry

end P_three_equals_thirteen_no_n_for_P_equals_2002_l747_747344


namespace _l747_747782

example : nat :=
  let gcd (a b c : ℕ) : ℕ := sorry
  let lcm (a b c : ℕ) : ℕ := sorry
  let a := sorry
  let b := sorry
  let c := sorry
  have gcd_condition : gcd a b c = 10 := sorry
  have lcm_condition : lcm a b c = 2^17 * 5^16 := sorry
  have main_theorem : ∃ n, n = 8640 := sorry
  8640

end _l747_747782


namespace real_nums_inequality_l747_747111

theorem real_nums_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a ^ 2000 + b ^ 2000 = a ^ 1998 + b ^ 1998) :
  a ^ 2 + b ^ 2 ≤ 2 :=
sorry

end real_nums_inequality_l747_747111


namespace tan_triple_angle_l747_747003

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l747_747003


namespace find_z_l747_747696

-- Given conditions as Lean definitions
def consecutive (x y z : ℕ) : Prop := x = z + 2 ∧ y = z + 1 ∧ x > y ∧ y > z
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + 3 * z = 5 * y + 11

-- The statement to be proven
theorem find_z (x y z : ℕ) (h1 : consecutive x y z) (h2 : equation x y z) : z = 3 :=
sorry

end find_z_l747_747696


namespace find_dividend_l747_747331

theorem find_dividend (
  divisor : ℕ,
  quotient : ℕ,
  dividend : ℕ,
  h : 12 * 999809 = dividend
) : dividend = 11997708 :=
by
  sorry

end find_dividend_l747_747331


namespace dice_sum_prob_10_l747_747462

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747462


namespace total_skateboarded_distance_l747_747055

/-- Define relevant conditions as variables for input values in Lean -/
variables
  (distance_skateboard_to_park : ℕ) -- Distance skateboarded to the park
  (distance_walk_to_park : ℕ) -- Distance walked to the park
  (distance_skateboard_park_to_home : ℕ) -- Distance skateboarded from the park to home

/-- Constraint conditions as hypotheses -/
variables
  (H1 : distance_skateboard_to_park = 10)
  (H2 : distance_walk_to_park = 4)
  (H3 : distance_skateboard_park_to_home = distance_skateboard_to_park + distance_walk_to_park)

/-- The theorem we intend to prove -/
theorem total_skateboarded_distance : 
  distance_skateboard_to_park + distance_skateboard_park_to_home - distance_walk_to_park = 24 := 
by 
  rw [H1, H2, H3] 
  sorry

end total_skateboarded_distance_l747_747055


namespace european_confidence_95_european_teams_not_face_l747_747606

-- Definitions for the conditions
def european_teams_round_of_16 := 44
def european_teams_not_round_of_16 := 22
def other_regions_round_of_16 := 36
def other_regions_not_round_of_16 := 58
def total_teams := 160

-- Formula for K^2 calculation
def k_value : ℚ := 3.841
def k_squared (n a_d_diff b_c_diff a b c d : ℚ) : ℚ :=
  n * ((a_d_diff - b_c_diff)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Definitions and calculation of K^2
def n1 := (european_teams_round_of_16 + other_regions_round_of_16 : ℚ)
def a_d_diff1 := (european_teams_round_of_16 * other_regions_not_round_of_16 : ℚ)
def b_c_diff1 := (european_teams_not_round_of_16 * other_regions_round_of_16 : ℚ)
def k_squared_result := k_squared n1 a_d_diff1 b_c_diff1
                                 (european_teams_round_of_16 + european_teams_not_round_of_16)
                                 (other_regions_round_of_16 + other_regions_not_round_of_16)
                                 total_teams total_teams

-- Theorem for 95% confidence derived
theorem european_confidence_95 :
  k_squared_result > k_value := sorry

-- Probability calculation setup
def total_ways_to_pair_teams : ℚ := 15
def ways_european_teams_not_face : ℚ := 6
def probability_european_teams_not_face := ways_european_teams_not_face / total_ways_to_pair_teams

-- Theorem for probability
theorem european_teams_not_face :
  probability_european_teams_not_face = 2 / 5 := sorry

end european_confidence_95_european_teams_not_face_l747_747606


namespace parallel_line_closer_to_base_length_l747_747736

noncomputable def triangle_base_length : ℝ := 20

def is_parallel (l1 l2 : ℝ) : Prop :=
  -- Placeholder for the definition of parallel lines, inherently true here for simplicity
  true

def divides_into_equal_areas (base : ℝ) (n : ℕ) : Prop :=
  -- Placeholder for the definition of dividing the triangle into equal areas
  true

theorem parallel_line_closer_to_base_length :
  ∃ L : ℝ, is_parallel L triangle_base_length ∧ divides_into_equal_areas triangle_base_length 4 ∧ L = 10 :=
by
  existsi 10
  split
  case is_parallel => exact trivial
  case divides_into_equal_areas => exact trivial
  case eq => rfl

end parallel_line_closer_to_base_length_l747_747736


namespace find_first_number_l747_747123

theorem find_first_number (HCF LCM num2 num1 : ℕ) (hcf_cond : HCF = 20) (lcm_cond : LCM = 396) (num2_cond : num2 = 220) 
    (relation_cond : HCF * LCM = num1 * num2) : num1 = 36 :=
by
  sorry

end find_first_number_l747_747123


namespace max_value_g_f_less_than_e_x_div_x_sq_l747_747832

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_g : ∃ x, x = 3 ∧ g x = 2 * Real.log 2 - 7 / 4 := by
  sorry

theorem f_less_than_e_x_div_x_sq (x : ℝ) (hx : x > 0) : f x < (Real.exp x - 1) / x ^ 2 := by
  sorry

end max_value_g_f_less_than_e_x_div_x_sq_l747_747832


namespace trigonometric_identities_l747_747378

noncomputable def sin_cos_product (θ : ℝ) : ℝ :=
  sin θ * cos θ

noncomputable def cos2_minus_sin2 (θ : ℝ) : ℝ :=
  cos θ ^ 2 - sin θ ^ 2

noncomputable def sin3_minus_cos3 (θ : ℝ) : ℝ :=
  sin θ ^ 3 - cos θ ^ 3

theorem trigonometric_identities
  (θ : ℝ)
  (h1 : sin θ + cos θ = 1 / 5)
  (h2 : θ ∈ set.Ioo 0 real.pi) :
  (sin_cos_product θ = -12 / 25) ∧
  (cos2_minus_sin2 θ = -7 / 25) ∧
  (sin3_minus_cos3 θ = 91 / 125) :=
by
  sorry

end trigonometric_identities_l747_747378


namespace no_valid_pairs_for_two_digit_number_with_sum_264_and_even_digit_sum_l747_747319

theorem no_valid_pairs_for_two_digit_number_with_sum_264_and_even_digit_sum :
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (a + b = 24) → false :=
begin
  sorry
end

end no_valid_pairs_for_two_digit_number_with_sum_264_and_even_digit_sum_l747_747319


namespace small_sprocket_rotation_angle_l747_747225

-- Definitions based on the given conditions
def large_sprocket_teeth : ℕ := 48
def small_sprocket_teeth : ℕ := 20
def rotation_angle_per_revolution : ℝ := 2 * Real.pi

-- Statement to prove that the angle through which the small sprocket rotates is 4.8π radians
theorem small_sprocket_rotation_angle :
  rotation_angle_per_revolution * (large_sprocket_teeth / small_sprocket_teeth) = 4.8 * Real.pi :=
by
  sorry  -- Proof to be filled in.

end small_sprocket_rotation_angle_l747_747225


namespace number_of_satisfying_values_l747_747554

noncomputable def f (x : ℤ) : ℤ := x^2 + 5 * x + 6

def S : Set ℤ := Set.Icc 5 30

theorem number_of_satisfying_values : (Set.filter (λ s, f s % 8 = 0) S).card = 8 := sorry

end number_of_satisfying_values_l747_747554


namespace problem_statement_l747_747243

open Nat

def isCoprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def f : ℕ → ℕ := sorry

theorem problem_statement (f_prop1 : ∀ (a b : ℕ), isCoprime a b → f (a * b) = f a * f b)
                          (f_prop2 : ∀ (p q : ℕ), Prime p → Prime q → f (p + q) = f p + f q) :
  f 2 = 2 ∧ f 3 = 3 ∧ f 1999 = 1999 :=
by
  sorry

end problem_statement_l747_747243


namespace order_of_quantities_l747_747077

variables {x y : ℝ}
-- Conditions
axiom pos_diff : x > 0 ∧ y > 0 ∧ x ≠ y
def R := sqrt ((x^2 + y^2) / 2)
def A := (x + y) / 2
def G := sqrt (x * y)
def H := 2 * x * y / (x + y)

-- Problem statement
theorem order_of_quantities (R A G H : ℝ) 
  (R_def : R = sqrt ((x^2 + y^2) / 2))
  (A_def : A = (x + y) / 2)
  (G_def : G = sqrt (x * y))
  (H_def : H = 2 * x * y / (x + y)) : 
  G - H < R - A ∧ R - A < A - G :=
sorry

end order_of_quantities_l747_747077


namespace percentage_remaining_income_l747_747746

variable (income petrol house_rent remaining_income : ℝ)

-- Define the conditions
def condition1 : income * 0.30 = petrol := sorry
def condition2 : petrol = 300 := sorry
def condition3 : house_rent = 140 := sorry
def remaining_income_def : remaining_income = income - petrol := sorry

-- Define the percentage of remaining income spent on house rent
def percentage_on_house_rent (income petrol house_rent remaining_income : ℝ) : ℝ :=
  (house_rent / remaining_income) * 100

-- The theorem we aim to prove
theorem percentage_remaining_income (income petrol house_rent remaining_income : ℝ) :
  condition1 → condition2 → condition3 → remaining_income_def → percentage_on_house_rent income petrol house_rent remaining_income = 20 :=
by
  sorry

end percentage_remaining_income_l747_747746


namespace dice_sum_10_probability_l747_747429

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747429


namespace general_term_formula_sum_of_reciprocal_Sn_l747_747368

variable {n : ℕ}
variable {a_n S_n : ℕ → ℕ}
variable {a1 d : ℕ}

-- Define arithmetic sequence condition
def is_arithmetic_seq (a : ℕ → ℕ) (a1 : ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℕ) (a1 : ℕ) (d : ℕ) : Prop :=
  is_arithmetic_seq a a1 d ∧
  (S_n 5 = 4 * a 3 + 6) ∧
  (a 2, a 3, a 9 form a geometric sequence)

-- Problem 1: Prove the general term formula for the sequence {a_n}
theorem general_term_formula (a : ℕ → ℕ) (a1 d : ℕ) (h : given_conditions a a1 d) : 
  a = (λ n, 2 * n) ∨ a = (λ n, 6) := 
sorry

-- Problem 2: Prove the sum of the first n terms of {1 / S_n}
theorem sum_of_reciprocal_Sn (a : ℕ → ℕ) (a1 d : ℕ) (h : given_conditions a a1 d) (h0 : a1 ≠ a 5) :
  ∑ k in range n, 1 / (S_n k) = n / (n + 1) :=
sorry

end general_term_formula_sum_of_reciprocal_Sn_l747_747368


namespace find_AX_l747_747097

-- Define a circle diameter
def diameter : ℝ := 1

-- Define the angles given in the problem
def angle_BAC : ℝ := 20
def angle_BXC : ℝ := 60

-- Define the equality of distances given in the problem
def BX_eq_CX : Prop := sorry

-- Define the length of AB given the circle's diameter and the angle
def len_AB : ℝ := diameter * sin (angle_BAC * Real.pi / 180)

-- Define AX using the law of sines and given conditions
def AX : ℝ := sin (angle_BAC * Real.pi / 180) * sin (10 * Real.pi / 180) * (1 / cos (30 * Real.pi / 180))

-- The statement to prove that AX is equal to the derived trigonometric formula
theorem find_AX : 
  AX = sin (10 * Real.pi / 180) * sin (20 * Real.pi / 180) * (1 / cos (30 * Real.pi / 180)) := 
  sorry

end find_AX_l747_747097


namespace graph_contains_matching_or_star_l747_747246

open Classical 

noncomputable section

def Graph := Type
def isMatching (G : Graph) (E : set (G × G)) := ∀ e1 e2 ∈ E, e1 ≠ e2 → e1.1 ≠ e2.1 ∧ e1.1 ≠ e2.2 ∧ e1.2 ≠ e2.1 ∧ e1.2 ≠ e2.2
def isStar (G : Graph) (E : set (G × G)) := ∃ v : G, ∀ e ∈ E, v = e.1 ∨ v = e.2
def edgeCount (E : set (G × G)) := set.size E

theorem graph_contains_matching_or_star (G : Graph) (E : set (G × G)) (k : ℕ) (h : edgeCount E > 2 * (k - 1) * (k - 1)) :
  ∃ M : set (G × G), isMatching G M ∧ set.size M = k ∨ ∃ S : set (G × G), isStar G S ∧ set.size S = k :=
sorry

end graph_contains_matching_or_star_l747_747246


namespace moves_multiple_of_n_l747_747041

theorem moves_multiple_of_n
  (n m : ℕ)
  (hn : 2 ≤ n)
  (hmn : n < m)
  (initial_chips: ℕ → ℕ)
  (moves : list (ℕ × ℕ))
  (valid_move : ∀ (move : ℕ × ℕ), 0 < move.1 ∧ move.1 < n ∧ 0 < move.2)
  (initial_distribution : list (ℕ × ℕ))
  (final_distribution : list (ℕ × ℕ))
  (same_distribution: ∀ (v : ℕ), initial_chips v = final_distribution.foldr (λ (p : ℕ × ℕ) (acc : ℕ), if p.1 = v then acc + p.2 else acc) 0) :
  ∃ k : ℕ, k = moves.length ∧ n ∣ k :=
sorry

end moves_multiple_of_n_l747_747041


namespace dice_sum_prob_10_l747_747470

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747470


namespace minimum_distance_from_curve_to_line_l747_747068

noncomputable def minimum_distance : ℝ :=
  3 * Real.sqrt 2 / 2

theorem minimum_distance_from_curve_to_line :
  ∀ P : (ℝ × ℝ), (P.1^2 - P.2 - Real.log P.1 = 0) →
  ∃(x: ℝ), (dist (x, x - 3) P = minimum_distance) :=
by
  intro P hP
  use (1, 1)
  have h1 : P = (1, 1) := sorry
  have h2 : dist (1, 1) (1, -2) = minimum_distance := sorry
  exact ⟨1, h1, h2⟩

end minimum_distance_from_curve_to_line_l747_747068


namespace dice_sum_prob_10_l747_747467

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747467


namespace dice_sum_probability_l747_747481

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747481


namespace inequality_holds_for_interval_l747_747998

theorem inequality_holds_for_interval (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 5 → x^2 - 2 * (a - 2) * x + a < 0) → a ≥ 5 :=
by
  intros h
  sorry

end inequality_holds_for_interval_l747_747998


namespace tan_triple_angle_l747_747001

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l747_747001


namespace min_colors_needed_is_3_l747_747175

noncomputable def min_colors_needed (S : Finset (Fin 7)) : Nat :=
  -- function to determine the minimum number of colors needed
  if ∀ (f : Finset (Fin 7) → Fin 3), ∀ (A B : Finset (Fin 7)), A.card = 3 ∧ B.card = 3 →
    A ∩ B = ∅ → f A ≠ f B then
    3
  else
    sorry

theorem min_colors_needed_is_3 :
  ∀ S : Finset (Fin 7), min_colors_needed S = 3 :=
by
  sorry

end min_colors_needed_is_3_l747_747175


namespace minimum_value_occurs_at_4_l747_747762

noncomputable def minimum_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f x ≤ f y

def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem minimum_value_occurs_at_4 :
  minimum_value_at quadratic_expression 4 :=
sorry

end minimum_value_occurs_at_4_l747_747762


namespace total_time_correct_l747_747691

-- Define the base speeds and distance
def speed_boat : ℕ := 8
def speed_stream : ℕ := 6
def distance : ℕ := 210

-- Define the speeds downstream and upstream
def speed_downstream : ℕ := speed_boat + speed_stream
def speed_upstream : ℕ := speed_boat - speed_stream

-- Define the time taken for downstream and upstream
def time_downstream : ℕ := distance / speed_downstream
def time_upstream : ℕ := distance / speed_upstream

-- Define the total time taken
def total_time : ℕ := time_downstream + time_upstream

-- The theorem to be proven
theorem total_time_correct : total_time = 120 := by
  sorry

end total_time_correct_l747_747691


namespace least_common_positive_period_of_f_condition_l747_747305

noncomputable def least_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x : ℝ, f (x + p) = f x

theorem least_common_positive_period_of_f_condition
  (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x + 6) + f(x - 6) = f(x)) : least_period f 36 :=
by
  sorry

end least_common_positive_period_of_f_condition_l747_747305


namespace average_sitting_time_per_student_l747_747683

def total_travel_time_in_minutes : ℕ := 152
def number_of_seats : ℕ := 5
def number_of_students : ℕ := 8

theorem average_sitting_time_per_student :
  (total_travel_time_in_minutes * number_of_seats) / number_of_students = 95 := 
by
  sorry

end average_sitting_time_per_student_l747_747683


namespace triangle_inequality_equality_condition_for_inequality_l747_747757

theorem triangle_inequality
  (A B C : Point)
  (h : ℝ)
  (α : ℝ)
  (h_def : height A B C = h)
  (α_def : angle BAC = α)
  (AB AC BC : ℝ)
  (angles_def : ∃ β γ : ℝ, γ = angle ACB ∧ β = angle ABC ∧ α = angle BAC)
  :
  AB + AC ≥ BC * cos α + 2 * h * sin α := 
  sorry

theorem equality_condition_for_inequality
  (A B C : Point)
  (h : ℝ)
  (α : ℝ)
  (h_def : height A B C = h)
  (α_def : angle BAC = α)
  (AB AC BC : ℝ)
  (isosceles_def : AB = AC)
  :
  (α = π / 2) ↔ (AB + AC = BC * cos α + 2 * h * sin α) := 
  sorry

end triangle_inequality_equality_condition_for_inequality_l747_747757


namespace triangle_circumcenter_incenter_perpendicular_equal_distance_l747_747906

open EuclideanGeometry

theorem triangle_circumcenter_incenter_perpendicular_equal_distance
  (ABC : Triangle)
  (O I : Point)
  (A B C D E : Point)
  (hO : O = circumcenter ABC)
  (hI : I = incenter ABC)
  (hAngleC : angle A B C = 30)
  (hD_on_AC : lies_on D (Line A C))
  (hE_on_BC : lies_on E (Line B C))
  (h_AD_BE_AB : segment_length A D = segment_length B E ∧ segment_length A D = segment_length A B)
  : perpendicular (Line O I) (Line D E) ∧ distance O I = distance D E := sorry

end triangle_circumcenter_incenter_perpendicular_equal_distance_l747_747906


namespace min_triangles_cover_square_l747_747670

noncomputable def minimumTriangles (s t : ℕ) : ℕ :=
  Nat.ceil ((s * s) / ((sqrt 3 / 4) * t * t))

theorem min_triangles_cover_square : minimumTriangles 10 1 = 231 := by
  sorry

end min_triangles_cover_square_l747_747670


namespace polynomial_sum_l747_747926

def f (x : ℝ) : ℝ := -4 * x^3 - 3 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + x - 7
def h (x : ℝ) : ℝ := 3 * x^3 + 6 * x^2 + 3 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = x^3 - 2 * x^2 + 6 * x - 10 := by
  sorry

end polynomial_sum_l747_747926


namespace total_amount_collected_l747_747089

theorem total_amount_collected 
  (num_members : ℕ)
  (annual_fee : ℕ)
  (cost_hardcover : ℕ)
  (num_hardcovers : ℕ)
  (cost_paperback : ℕ)
  (num_paperbacks : ℕ)
  (total_collected : ℕ) :
  num_members = 6 →
  annual_fee = 150 →
  cost_hardcover = 30 →
  num_hardcovers = 6 →
  cost_paperback = 12 →
  num_paperbacks = 6 →
  total_collected = (annual_fee + cost_hardcover * num_hardcovers + cost_paperback * num_paperbacks) * num_members →
  total_collected = 2412 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end total_amount_collected_l747_747089


namespace sum_of_three_rel_prime_pos_integers_l747_747162

theorem sum_of_three_rel_prime_pos_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_rel_prime_ab : Nat.gcd a b = 1) (h_rel_prime_ac : Nat.gcd a c = 1) (h_rel_prime_bc : Nat.gcd b c = 1)
  (h_product : a * b * c = 2700) :
  a + b + c = 56 := by
  sorry

end sum_of_three_rel_prime_pos_integers_l747_747162


namespace bryan_has_more_skittles_than_ben_has_mms_l747_747286

theorem bryan_has_more_skittles_than_ben_has_mms :
  ∀ (bryan_skittles ben_mms : ℕ), bryan_skittles = 50 → ben_mms = 20 → (bryan_skittles - ben_mms) = 30 := 
by
  intros bryan_skittles ben_mms h1 h2
  rw [h1, h2]
  exact rfl

end bryan_has_more_skittles_than_ben_has_mms_l747_747286


namespace difference_between_perfect_and_cracked_l747_747573

def total_eggs : ℕ := 24
def broken_eggs : ℕ := 3
def cracked_eggs : ℕ := 2 * broken_eggs

def perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs
def difference : ℕ := perfect_eggs - cracked_eggs

theorem difference_between_perfect_and_cracked :
  difference = 9 := by
  sorry

end difference_between_perfect_and_cracked_l747_747573


namespace probability_shorts_differ_jersey_l747_747916

theorem probability_shorts_differ_jersey : 
  let colors := ['black', 'white', 'gold'],
      total_configurations := colors.length * colors.length,
      matching_configurations := colors.length,
      mismatched_configurations := total_configurations - matching_configurations in
  (mismatched_configurations / total_configurations : ℚ) = 2 / 3 :=
by
  sorry

end probability_shorts_differ_jersey_l747_747916


namespace repeating_decimal_as_fraction_l747_747413

theorem repeating_decimal_as_fraction :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ Int.natAbs (Int.gcd a b) = 1 ∧ a + b = 15 ∧ (a : ℚ) / b = 0.3636363636363636 :=
by
  sorry

end repeating_decimal_as_fraction_l747_747413


namespace A_share_of_gain_l747_747685

-- Definitions of conditions
variables 
  (x : ℕ) -- Initial investment by A
  (annual_gain : ℕ := 24000) -- Total annual gain
  (A_investment_period : ℕ := 12) -- Months A invested
  (B_investment_period : ℕ := 6) -- Months B invested after 6 months
  (C_investment_period : ℕ := 4) -- Months C invested after 8 months

-- Investment ratios
def A_ratio := x * A_investment_period
def B_ratio := (2 * x) * B_investment_period
def C_ratio := (3 * x) * C_investment_period

-- Proof statement
theorem A_share_of_gain : 
  A_ratio = 12 * x ∧ B_ratio = 12 * x ∧ C_ratio = 12 * x ∧ annual_gain = 24000 →
  annual_gain / 3 = 8000 :=
by
  sorry

end A_share_of_gain_l747_747685


namespace second_pipe_fill_time_l747_747660

theorem second_pipe_fill_time :
  let x := 12 in
  (∀ t1 t2 t3 fill_time, (t1 = 10 ∧ t2 = 40 ∧ fill_time = 6.31578947368421) →
    let fill_rate1 := 1 / t1 in
    let fill_rate2 := 1 / x in
    let empty_rate := 1 / t2 in
    let combined_rate := 1 / fill_time in
    fill_rate1 + fill_rate2 - empty_rate = combined_rate) :=
  sorry

end second_pipe_fill_time_l747_747660


namespace parallelepipeds_count_l747_747801

noncomputable def number_of_parallelepipeds (p1 p2 p3 p4 : Point) : ℕ :=
  if ∃ plane, ∀ p ∈ {p1, p2, p3, p4}, p ∈ plane then 0 else 29

theorem parallelepipeds_count (p1 p2 p3 p4 : Point) (h : ¬ ∃ plane, ∀ p ∈ {p1, p2, p3, p4}, p ∈ plane) :
  number_of_parallelepipeds p1 p2 p3 p4 = 29 :=
by
  simp [number_of_parallelepipeds, h]
  sorry

end parallelepipeds_count_l747_747801


namespace find_angle_BAC_l747_747043

theorem find_angle_BAC (A B C D : Type) [triangle A B C] 
  (tangent_circle : ∀ (X : Type), tangent X D) 
  (angle_ABC : ∠ B A C = 75) 
  (angle_BCD : ∠ B C D = 40) :
  ∠ B A C = 25 :=
sorry

end find_angle_BAC_l747_747043


namespace geometric_sequence_common_ratio_l747_747896

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = -1) 
  (h2 : a 2 + a 3 = -2) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  q = -2 ∨ q = 1 := 
by sorry

end geometric_sequence_common_ratio_l747_747896


namespace semicircle_length_invariant_l747_747960

theorem semicircle_length_invariant :
  ∀ (A B C : ℝ), 
  (2 * ∃ l : ℝ, B = A + 2 * l) ∧ (A < C ∧ C < B)  →
  let AC := C - A in
  let CB := B - C in
  (∃ l : ℝ, (π * AC + π * CB) = π * l) :=
by
  sorry

end semicircle_length_invariant_l747_747960


namespace area_of_AFCH_l747_747583

-- Define the sides of the rectangles ABCD and EFGH
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the area of quadrilateral AFCH
def area_AFCH : ℝ := 52.5

-- The theorem we want to prove
theorem area_of_AFCH :
  AB = 9 ∧ BC = 5 ∧ EF = 3 ∧ FG = 10 → (area_AFCH = 52.5) :=
by
  sorry

end area_of_AFCH_l747_747583


namespace ab_product_l747_747421

open Polynomial

-- Define the polynomial
def poly (a b : ℝ) := X^4 - (a - 2) * X^3 + 5 * X^2 + (b + 3) * X - 1

-- Conditions: coefficients of x^3 and x are zero
def condition1 (a : ℝ) : Prop := (a - 2) = 0
def condition2 (b : ℝ) : Prop := (b + 3) = 0

-- Proof goal: if the conditions are satisfied, then ab = -6
theorem ab_product (a b : ℝ) (h1 : condition1 a) (h2 : condition2 b) : a * b = -6 := by
  sorry

end ab_product_l747_747421


namespace simplify_expr_l747_747975

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : x ^ (-2) + 2 * x ^ (-1) - 3 = (1 + 2 * x - 3 * x ^ 2) / x ^ 2 :=
by sorry

end simplify_expr_l747_747975


namespace find_lambda_l747_747813

-- Define point structures
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define conditions and the proof to be demonstrated
def P : Point := ⟨-1, 2⟩
def M : Point := ⟨1, -1⟩

def collinear (v1 v2 : Point) : Prop :=
  ∃ k : ℤ, v1.x = k * v2.x ∧ v1.y = k * v2.y

theorem find_lambda (λ : ℚ) : 
  ∀ Q : Point, 
  let PQ := ⟨2*(M.x - P.x), 2*(M.y - P.y)⟩ in
  collinear PQ ⟨λ, 1⟩ → λ = -2/3 :=
by
  intro Q
  let PQ := ⟨-4, 6⟩  -- Based on computation in the solution
  sorry

end find_lambda_l747_747813


namespace sum_positive_l747_747771

variable (s : Fin 1997 → ℝ)

-- Condition: Every subset of 97 out of 1997 numbers has a positive sum
def pos_sum (s : Fin 1997 → ℝ) : Prop :=
  ∀ (t : Finset (Fin 1997)), t.card = 97 → 0 < (t.sum (λ i, s i))

-- Theorem to prove
theorem sum_positive (s : Fin 1997 → ℝ) (h : pos_sum s) : 0 < ∑ i, s i :=
by sorry

end sum_positive_l747_747771


namespace part1_part2_l747_747373

noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def line_equation (p1 p2 : (ℝ × ℝ)) : ℝ → ℝ → Prop :=
  λ x y, (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

def slope (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def perpendicular (p1 p2 p3 p4 : (ℝ × ℝ)) : Prop :=
  slope p1 p2 * slope p3 p4 = -1

-- Conditions and definitions
def A : (ℝ × ℝ) := (2, 5)
def B : (ℝ × ℝ) := (6, -1)
def C : (ℝ × ℝ) := (9, 1)
def M : (ℝ × ℝ) := midpoint A C

-- Proposition to be proved
theorem part1 :
  ∀ x y, line_equation A M x y ↔ 8 * x + y - 47 = 0 :=
sorry

theorem part2 :
  perpendicular A B B C :=
sorry

end part1_part2_l747_747373


namespace simplify_and_evaluate_expression_l747_747974

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 3) :
  (1 + 1 / (a + 1)) / ((a^2 - 4) / (2 * a + 2)) = 2 :=
by
  rw h
  sorry

end simplify_and_evaluate_expression_l747_747974


namespace max_value_of_f_l747_747543

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (x * (80 - x)) + real.sqrt (x * (3 - x))

theorem max_value_of_f :
  ∃ x_0 M, 0 ≤ x_0 ∧ x_0 ≤ 3 ∧ M = f x_0 ∧ ∀ x, 0 ≤ x ∧ x ≤ 3 → f x ≤ M ∧ (x = 3 → M = 4 * real.sqrt 15) :=
begin
  sorry
end

end max_value_of_f_l747_747543


namespace tan_inequality_solution_set_l747_747837

theorem tan_inequality_solution_set :
  ∀ (a : ℝ) (k : ℤ), 0 < a ∧ a < 1 ∧ (x ≠ (k * π / 2 + π / 4) ) →
  {x | (k * π / 2 + π / 8 ≤ x) ∧ (x < k * π / 2 + π / 4)} = 
  {x | tan(2 * x) ≥ 2 * a ∧ ∀ y, y = (k * π / 2 + π / 4) → tan(2 * y) ≠ y} :=
by sorry

end tan_inequality_solution_set_l747_747837


namespace total_clouds_l747_747292

theorem total_clouds (C B : ℕ) (h1 : C = 6) (h2 : B = 3 * C) : C + B = 24 := by
  sorry

end total_clouds_l747_747292


namespace tan_triple_angle_l747_747009

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l747_747009


namespace baez_marble_loss_l747_747283

theorem baez_marble_loss :
  ∃ p : ℚ, (p > 0 ∧ (p / 100) * 25 * 2 = 60) ∧ p = 20 :=
by
  sorry

end baez_marble_loss_l747_747283


namespace dice_sum_probability_l747_747499

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747499


namespace max_m_value_for_hyperbola_right_branch_l747_747890

def distance_between_parallel_lines {a b1 b2 : ℝ} (h: a ≠ 0 ∧ b1 ≠ b2) : ℝ := 
  abs (b1 - b2) / real.sqrt (a^2 + 1)

theorem max_m_value_for_hyperbola_right_branch (m : ℝ) : 
  (∀ (P : ℝ × ℝ), P ∈ {P : ℝ × ℝ | P.fst^2 - P.snd^2 = 1} → 
    ∀ (d : ℝ), d = abs (P.fst - P.snd + 1) / real.sqrt 2 → d > m) 
  → m = real.sqrt 2 / 2 :=
by
  sorry

end max_m_value_for_hyperbola_right_branch_l747_747890


namespace new_arithmetic_mean_l747_747260

theorem new_arithmetic_mean (S : Finset ℝ) (hS_card : S.card = 60) (hS_mean : (S : Set ℝ).sum / 60 = 42)
  (a b c : ℝ) (h : a = 48 ∧ b = 52 ∧ c = 60) :
  let S' := S.erase a.erase b.erase c in
  let new_sum := (S' : Set ℝ).sum in
  new_sum / 57 = 41.404 :=
sorry

end new_arithmetic_mean_l747_747260


namespace find_Y_l747_747647

theorem find_Y 
  (a b c d X Y : ℕ)
  (h1 : a + b + c + d = 40)
  (h2 : X + Y + c + b = 40)
  (h3 : a + b + X = 30)
  (h4 : c + d + Y = 30)
  (h5 : X = 9) 
  : Y = 11 := 
by 
  sorry

end find_Y_l747_747647


namespace minimum_value_sqrt_inequality_l747_747943

theorem minimum_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (sqrt ((x^2 + y^2) * (4 * x^2 + y^2)) / (x * y)) ≥ 3 :=
sorry

end minimum_value_sqrt_inequality_l747_747943


namespace number_of_cats_l747_747880

variable (C D : ℕ)

-- Conditions
def condition1 : Prop := C = 15 * D / 7
def condition2 : Prop := C = 15 * (D + 12) / 11

-- Proof problem
theorem number_of_cats (h1 : condition1 C D) (h2 : condition2 C D) : C = 45 := sorry

end number_of_cats_l747_747880


namespace max_area_of_triangle_PAB_l747_747530

def circle (x y : ℝ) : Prop := (x - 5) ^ 2 + y ^ 2 = 25

def line (x y : ℝ) : Prop := x - 7 * y + 20 = 0

theorem max_area_of_triangle_PAB :
  let A := (1 : ℝ, 3 : ℝ)
  let B := (8 : ℝ, 4 : ℝ)
  ∃ P : ℝ × ℝ, circle P.1 P.2 ∧
    ∃ d, 
    let base := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
    let height := (real.abs (5 + 20) / real.sqrt (1 + 7^2)) + 5
    d = 1/2 * base * height → 
    d = 25 / 2 * (1 + real.sqrt 2) := 
sorry

end max_area_of_triangle_PAB_l747_747530


namespace average_of_first_12_is_14_l747_747613

-- Definitions based on given conditions
def average_of_25 := 19
def sum_of_25 := average_of_25 * 25

def average_of_last_12 := 17
def sum_of_last_12 := average_of_last_12 * 12

def result_13 := 103

-- Main proof statement to be checked
theorem average_of_first_12_is_14 (A : ℝ) (h1 : sum_of_25 = sum_of_25) (h2 : sum_of_last_12 = sum_of_last_12) (h3 : result_13 = 103) :
  (A * 12 + result_13 + sum_of_last_12 = sum_of_25) → (A = 14) :=
by
  sorry

end average_of_first_12_is_14_l747_747613


namespace upper_limit_for_y_l747_747861

theorem upper_limit_for_y (x y : ℝ) (hx : 5 < x) (hx' : x < 8) (hy : 8 < y) (h_diff : y - x = 7) : y ≤ 14 :=
by
  sorry

end upper_limit_for_y_l747_747861


namespace root_equation_solution_l747_747014

theorem root_equation_solution (a : ℝ) (h : 3 * a^2 - 5 * a - 2 = 0) : 6 * a^2 - 10 * a = 4 :=
by 
  sorry

end root_equation_solution_l747_747014


namespace fraction_calls_processed_by_team_B_l747_747686

variable (A B C_A C_B : ℕ)

theorem fraction_calls_processed_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : C_A = (2 / 5) * C_B) :
  (B * C_B) / ((A * C_A) + (B * C_B)) = 8 / 9 := by
  sorry

end fraction_calls_processed_by_team_B_l747_747686


namespace _l747_747360

noncomputable theorem geometric_sequence_general_formula (q : ℝ) (h_q : q > 1)
  (a : ℕ → ℝ) (h_seq : a 2 + a 3 + a 4 = 28) 
  (h_mean : a 3 + 2 = (a 2 + a 4) / 2) : ∀ n, a n = 2^n := 
by
  sorry

noncomputable theorem smallest_positive_integer_n 
  (a : ℕ → ℝ) (h_a : ∀ n, a n = 2^n) 
  (b : ℕ → ℝ) (h_b : ∀ n, b n = a n * (Real.log (a n) / Real.log (0.5))) 
  (S : ℕ → ℝ) (h_S : ∀ n, S n = (Finset.range (n+1)).sum (λ i, b i)) : ∃ n, 0 < n ∧ S n + n * 2^(n+1) > 62 := 
by
  sorry

end _l747_747360


namespace captain_age_l747_747513

theorem captain_age (C : ℕ) (h1 : ∀ W : ℕ, W = C + 3) 
                    (h2 : 21 * 11 = 231) 
                    (h3 : 21 - 1 = 20) 
                    (h4 : 20 * 9 = 180)
                    (h5 : 231 - 180 = 51) :
  C = 24 :=
by
  sorry

end captain_age_l747_747513


namespace problem_inequality_l747_747352

variable (a b : ℝ)

theorem problem_inequality (h1 : a < 0) (h2 : -1 < b) (h3 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end problem_inequality_l747_747352


namespace min_value_of_function_l747_747800

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  (∀ x₀ : ℝ, x₀ > -1 → (x₀ + 1 + 1 / (x₀ + 1) - 1) ≥ 1) ∧ (x = 0) :=
sorry

end min_value_of_function_l747_747800


namespace arith_seq_problem_l747_747525

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

theorem arith_seq_problem 
  (a : ℕ → ℝ) (a1 d : ℝ)
  (h1 : arithmetic_sequence a a1 d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 :=
by 
  sorry

end arith_seq_problem_l747_747525


namespace dice_sum_probability_l747_747497

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747497


namespace pete_flag_diff_squares_twice_stripes_l747_747608

theorem pete_flag_diff_squares_twice_stripes :
  let stars := 50
  let stripes := 13
  let circles := (stars / 2 - 3 : ℕ)
  let squares := (2 * stripes + 6 : ℕ)
  circles + squares = 54 →
  (squares - 2 * stripes) = 6 :=
by
  intros
  simp
  exact sorry

end pete_flag_diff_squares_twice_stripes_l747_747608


namespace incorrect_f_2_eq_7_l747_747414

def f (x : ℝ) : ℝ := (x^2 + 3 * x + 2) / (x - 2)

theorem incorrect_f_2_eq_7 : f 2 ≠ 7 :=
by sorry

end incorrect_f_2_eq_7_l747_747414


namespace Mickey_horses_per_week_l747_747313

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l747_747313


namespace range_of_a_l747_747821

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x^2 + 2 * a * x + 1) → -1 ≤ a ∧ a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l747_747821


namespace no_infinite_loops_l747_747537

theorem no_infinite_loops (α : ℝ) (h_small_angle : α > 0 ∧ α ≤ π) :
  ¬ (∃ (n : ℕ), ∀ (m : ℕ), m ≥ n → wrap_tape_infinite_cone m) :=
by
  sorry

-- Definitions for the conditions and problem setup
def wrap_tape_infinite_cone : ℕ → Prop := sorry

end no_infinite_loops_l747_747537


namespace jake_steps_per_second_l747_747282

/-
Conditions:
1. Austin and Jake start descending from the 9th floor at the same time.
2. The stairs have 30 steps across each floor.
3. The elevator takes 1 minute (60 seconds) to reach the ground floor.
4. Jake reaches the ground floor 30 seconds after Austin.
5. Jake descends 8 floors to reach the ground floor.
-/

def floors : ℕ := 8
def steps_per_floor : ℕ := 30
def time_elevator : ℕ := 60 -- in seconds
def additional_time_jake : ℕ := 30 -- in seconds

def total_time_jake := time_elevator + additional_time_jake -- in seconds
def total_steps := floors * steps_per_floor

def steps_per_second_jake := (total_steps : ℚ) / (total_time_jake : ℚ)

theorem jake_steps_per_second :
  steps_per_second_jake = 2.67 := by
  sorry

end jake_steps_per_second_l747_747282


namespace solve_for_t_l747_747978

theorem solve_for_t (t : ℝ) (ht : (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4)) : 
  t = -3 := sorry

end solve_for_t_l747_747978


namespace mickey_horses_per_week_l747_747309

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l747_747309


namespace proof_of_p_and_not_q_l747_747840

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > 1

theorem proof_of_p_and_not_q : p ∧ ¬q :=
by {
  sorry
}

end proof_of_p_and_not_q_l747_747840


namespace clock_correct_fraction_l747_747725

/--
A 12-hour digital clock displays the hour and minute of a day. 
Whenever it is supposed to display a 1, it mistakenly displays a 9. 
Prove that the fraction of the day the clock shows the correct time is 1/2.
-/
def correct_fraction_hours : ℚ := 2 / 3

def correct_fraction_minutes : ℚ := 3 / 4

theorem clock_correct_fraction : correct_fraction_hours * correct_fraction_minutes = 1 / 2 :=
by
  have hours_correct := correct_fraction_hours
  have minutes_correct := correct_fraction_minutes
  calc
    (correct_fraction_hours * correct_fraction_minutes) = (2 / 3 * 3 / 4) : by sorry
    ... = 1 / 2 : by sorry

end clock_correct_fraction_l747_747725


namespace minimize_Tn_l747_747856

variable {a : ℕ → ℝ}
variable {T : ℕ → ℝ}

noncomputable
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → (0 < a n) ∧ (a n < a m)

noncomputable
def product_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∏ i in finset.range n, a i

theorem minimize_Tn 
  (h_geo : is_increasing_geometric_sequence a)
  (h_T4_eq_T8 : product_of_first_n_terms a 4 = product_of_first_n_terms a 8) :
  ∃ n, product_of_first_n_terms a n = min_prod_n :=
begin
  sorry
end

end minimize_Tn_l747_747856


namespace range_of_a_l747_747389

-- Definition of the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x + a else real.log x / real.log a

-- Main theorem statement
theorem range_of_a (a : ℝ) :
  (0 < a ∧ a ≤ 1/3 ∨ 1 < a) ↔ set.range (f a) = set.univ :=
sorry

end range_of_a_l747_747389


namespace find_point_P_l747_747559

noncomputable def squared_distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2

theorem find_point_P :
  let A : ℝ × ℝ × ℝ := (10, 0, 0)
  let B : ℝ × ℝ × ℝ := (0, -6, 0)
  let C : ℝ × ℝ × ℝ := (0, 0, 8)
  let D : ℝ × ℝ × ℝ := (1, 1, 1)
  let P : ℝ × ℝ × ℝ := (3, -2, 5)
  squared_distance P A = squared_distance (3, -2, 5) A ∧ 
  squared_distance P B = squared_distance (3, -2, 5) B ∧ 
  squared_distance P C = squared_distance (3, -2, 5) C ∧ 
  squared_distance P P = squared_distance (3, 1, 1) - 18 - 6 :=
by
  have dx := P.1 - D.1
  have dy := P.2 - D.2
  have dz := P.3 - D.3
  sorry

end find_point_P_l747_747559


namespace angle_C_maximum_area_l747_747877

-- Definitions specifying the conditions
def triangle_side_lengths (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0

def angle_A (A : ℝ) : Prop :=
A = π / 6

def side_a (a : ℝ) : Prop :=
a = 1

def side_b (b : ℝ) : Prop :=
b = sqrt 3

-- Problem Ⅰ
theorem angle_C (a b : ℝ) (A C : ℝ) (h_a : side_a a) (h_b : side_b b) (h_A : angle_A A) :
  triangle_side_lengths a b c → C = π/2 ∨ C = π/6 :=
sorry

-- Problem Ⅱ
theorem maximum_area (a : ℝ) (A : ℝ) (area : ℝ) (h_a : side_a a) (h_A : angle_A A) :
  (∃ b c, triangle_side_lengths a b c ∧ A = π / 6 ∧ area = 1/2 * b * c * sin (π / 6)) → 
  area = (2 + sqrt 3) / 4 :=
sorry

end angle_C_maximum_area_l747_747877


namespace fermats_little_theorem_l747_747103

theorem fermats_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) : 
  (a^p - a) % p = 0 :=
sorry

end fermats_little_theorem_l747_747103


namespace students_enjoy_both_music_and_sports_l747_747883

theorem students_enjoy_both_music_and_sports :
  ∀ (T M S N B : ℕ), T = 55 → M = 35 → S = 45 → N = 4 → B = M + S - (T - N) → B = 29 :=
by
  intros T M S N B hT hM hS hN hB
  rw [hT, hM, hS, hN] at hB
  exact hB

end students_enjoy_both_music_and_sports_l747_747883


namespace rationality_of_expressions_l747_747189

def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem rationality_of_expressions :
  let e := Exp.exp 1 in
  let x1 := Real.sqrt (e ^ 2) in
  let x2 := (0.64)^(1/3 : ℝ) in
  let x3 := (0.0256)^(1/4 : ℝ) in
  let x4 := ((-8)^(1/3 : ℝ)) * (0.25 : ℝ)^(-1/2 : ℝ) in
  ¬ is_rational x1 ∧ ¬ is_rational x2 ∧ is_rational x3 ∧ is_rational x4 :=
by
  sorry

end rationality_of_expressions_l747_747189


namespace tangent_line_ln_curve_l747_747391

noncomputable def find_a (a : ℝ) : Prop :=
∃ x₀ : ℝ, (x₀ + 1 = log (x₀ + a)) ∧ ((1 / (x₀ + a)) = 1)

theorem tangent_line_ln_curve : find_a 2 :=
by
  sorry

end tangent_line_ln_curve_l747_747391


namespace line_a_skew_to_line_b_l747_747013

noncomputable def line_parallel_to_plane {α : Type*} [MetricSpace α] [AffineSpace α] 
  (a : Set α) (α_plane : Set α) : Prop := 
  ∀ (p : α), p ∈ a → ∃ (q : α), q ∈ α_plane ∧ ¬ p = q ∧ ∀ (r : α), r ∈ α_plane → ¬ ∃ k : ℝ, q + k • (p - q) = r

noncomputable def line_in_plane {α : Type*} [MetricSpace α] [AffineSpace α] 
  (b : Set α) (α_plane : Set α) : Prop := 
  ∀ (p : α), p ∈ b → p ∈ α_plane

theorem line_a_skew_to_line_b {α : Type*} [MetricSpace α] [AffineSpace α] 
  (a b : Set α) (α_plane : Set α) :
  line_parallel_to_plane a α_plane → line_in_plane b α_plane → Skew a b :=
sorry

end line_a_skew_to_line_b_l747_747013


namespace number_of_shirts_is_39_l747_747665

-- Define the conditions as Lean definitions.
def washing_machine_capacity : ℕ := 8
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 9

-- Define the total number of pieces of clothing based on the conditions.
def total_pieces_of_clothing : ℕ :=
  number_of_loads * washing_machine_capacity

-- Define the number of shirts.
noncomputable def number_of_shirts : ℕ :=
  total_pieces_of_clothing - number_of_sweaters

-- The actual proof problem statement.
theorem number_of_shirts_is_39 :
  number_of_shirts = 39 := by
  sorry

end number_of_shirts_is_39_l747_747665


namespace find_ordered_pairs_l747_747328

theorem find_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a - b) ^ (a * b) = a ^ b * b ^ a) :
  (a, b) = (4, 2) := by
  sorry

end find_ordered_pairs_l747_747328


namespace mickey_horses_per_week_l747_747310

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l747_747310


namespace complex_number_multiplication_result_l747_747325

theorem complex_number_multiplication_result :
  (6 - 3 * Complex.i) * (-7 + 2 * Complex.i) = -36 + 33 * Complex.i :=
by
  sorry

end complex_number_multiplication_result_l747_747325


namespace find_x_l747_747780

noncomputable def x : ℝ := 10.3

theorem find_x (h1 : x + (⌈x⌉ : ℝ) = 21.3) (h2 : x > 0) : x = 10.3 :=
sorry

end find_x_l747_747780


namespace remainder_17_pow_1499_mod_23_l747_747179

theorem remainder_17_pow_1499_mod_23 : (17 ^ 1499) % 23 = 11 :=
by
  sorry

end remainder_17_pow_1499_mod_23_l747_747179


namespace domain_of_f_l747_747332

noncomputable def f (x : ℝ) : ℝ := 
  real.sqrt (1 - real.rpow 2 x) + (1 / real.sqrt (x + 3))

theorem domain_of_f : 
  {x : ℝ | 1 - real.rpow 2 x ≥ 0 ∧ x + 3 > 0} = {x | -3 < x ∧ x ≤ 0} :=
by
  sorry

end domain_of_f_l747_747332


namespace dice_sum_probability_l747_747480

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747480


namespace not_equivalent_to_0_0000375_l747_747191

theorem not_equivalent_to_0_0000375 : 
    ¬ (3 / 8000000 = 3.75 * 10 ^ (-5)) :=
by sorry

end not_equivalent_to_0_0000375_l747_747191


namespace minimum_number_of_participants_l747_747635

theorem minimum_number_of_participants (a1 a2 a3 a4 : ℕ) (h1 : a1 = 90) (h2 : a2 = 50) (h3 : a3 = 40) (h4 : a4 = 20) 
  (h5 : ∀ (n : ℕ), n * 2 ≥ a1 + a2 + a3 + a4) : ∃ n, (n ≥ 100) :=
by 
  use 100
  sorry

end minimum_number_of_participants_l747_747635


namespace truncated_cone_sphere_radius_l747_747269

theorem truncated_cone_sphere_radius 
  (r1 r2 : ℝ) (hr1 : r1 = 13) (hr2 : r2 = 7) : 
  ∃ (R : ℝ), R = 9.5394 := 
by
  use 9.5394
  sorry

end truncated_cone_sphere_radius_l747_747269


namespace hershel_goldfish_initial_l747_747399

theorem hershel_goldfish_initial (G : ℕ) :
  let betta_fish_initial := 10
  let betta_fish_added := (2/5 : ℚ) * betta_fish_initial
  let total_betta_fish := betta_fish_initial + betta_fish_added
  let goldfish_added := (1/3 : ℚ) * G
  let total_goldfish := G + goldfish_added
  let total_fish_before_gifting := total_betta_fish + total_goldfish
  (total_fish_before_gifting / 2 = 17) →
  G = 15 :=
by {
  sorry,
}

end hershel_goldfish_initial_l747_747399


namespace area_enclosed_by_sin_l747_747987

/-- The area of the figure enclosed by the curve y = sin(x), the lines x = -π/3, x = π/2, and the x-axis is 3/2. -/
theorem area_enclosed_by_sin (x y : ℝ) (h : y = Real.sin x) (a b : ℝ) 
(h1 : a = -Real.pi / 3) (h2 : b = Real.pi / 2) :
  ∫ x in a..b, |Real.sin x| = 3 / 2 := 
sorry

end area_enclosed_by_sin_l747_747987


namespace increasing_probability_l747_747386

def f (a b : ℤ) (x : ℝ) : ℝ := (1 / 3) * x^3 - (a - 1) * x^2 + (b^2 : ℝ) * x

def f_prime (a b : ℤ) (x : ℝ) : ℝ := x^2 - 2 * (a - 1) * x + (b^2 : ℝ)

theorem increasing_probability :
  let n := Finset.card (Finset.filter (λ ab : ℤ × ℤ, 1 ≤ ab.fst ∧ ab.fst ≤ 4 ∧
    1 ≤ ab.snd ∧ ab.snd ≤ 3 ∧ 4 * (ab.fst^2 - 2 * ab.fst + 1 - ab.snd^2) < 0)
    (Finset.product (Finset.range 5) (Finset.range 4)))
  ∈ {n} ∧ (n.to_real / 12 = 3 / 4) :=
by
  sorry

end increasing_probability_l747_747386


namespace sum_prod_eq_even_odd_l747_747919

theorem sum_prod_eq_even_odd (n : ℕ) (x : Fin n → ℝ) (h_diff : ∀ (i j : Fin n), i ≠ j → x i ≠ x j) :
  (∑ i in Finset.univ, ∏ j in Finset.univ.erase i, (1 - (x i * x j)) / (x i - x j)) =
  if n % 2 = 0 then 0 else 1 := 
sorry

end sum_prod_eq_even_odd_l747_747919


namespace cade_marbles_left_l747_747287
noncomputable def cade_final_marbles (initial_cade_marbles : ℕ) (dylan_initial_marbles : ℕ): ℕ :=
  let cade_after_giving := initial_cade_marbles - 8 in
  let dylan_after_giving := dylan_initial_marbles - 8 in
  let cade_after_receiving := cade_after_giving + dylan_after_giving / 2 in
  cade_after_receiving - (2 / 3) * cade_after_receiving

theorem cade_marbles_left (initial_cade_marbles : ℕ) (dylan_initial_marbles : ℕ)
  (h1 : initial_cade_marbles = 87)
  (h2 : dylan_initial_marbles = 16) :
  cade_final_marbles initial_cade_marbles dylan_initial_marbles = 28 :=
by
  rw [h1, h2]
  unfold cade_final_marbles
  norm_num
  sorry

end cade_marbles_left_l747_747287


namespace total_memory_space_l747_747061

def morning_songs : Nat := 10
def afternoon_songs : Nat := 15
def night_songs : Nat := 3
def song_size : Nat := 5

theorem total_memory_space : (morning_songs + afternoon_songs + night_songs) * song_size = 140 := by
  sorry

end total_memory_space_l747_747061


namespace inradius_inequality_l747_747105

-- Definitions and problem statement
variables (a b c : ℝ)

def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def area (a b c : ℝ) : ℝ := 
  let s := semi_perimeter a b c in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

def inradius (a b c : ℝ) : ℝ := 
  let s := semi_perimeter a b c in
  let t := area a b c in
  t / s

theorem inradius_inequality (a b c : ℝ) : 
  inradius a b c ≤ Real.sqrt (a^2 + b^2 + c^2) / 6 := 
by
  sorry

end inradius_inequality_l747_747105


namespace ellipse_equation_and_triangle_area_l747_747810

noncomputable def ellipse {a b : ℝ} (ha : a > 0) (hb : b > 0) := set_of (λ (p : ℝ × ℝ), p.1^2 / a^2 + p.2^2 / b^2 = 1)

theorem ellipse_equation_and_triangle_area {a b : ℝ} (ha : a > b) (hb : b > 0) (maj_ax_len : 2 * a = 4 * real.sqrt 2)
    (A B C : ℝ × ℝ)
    (right_vertex_A : A = (a, 0))
    (B_first_quad : 0 < B.1 ∧ 0 < B.2)
    (BC_through_O : line_segment B C = line_segment (0, 0) B ∪ line_segment (0, 0) C)
    (BC_eq_2AB : dist B C = 2 * dist A B)
    (cos_angle_ABC : real.cos (angle B A C) = 1 / 5) :
  (∃ E, E = ellipse 2 (2 * real.sqrt 2)) ∧
  (∀ (l : ℝ × ℝ → ℝ × ℝ) (h_tangent : ∀ (x y : ℝ), x^2 + y^2 = 1 → tangent l x y)
    (M N : ℝ × ℝ) (h_intersect : ∃ (p1 p2 : ℝ × ℝ), p1 ∈ E ∧ p2 ∈ E ∧ l p1 = l p2), 
    let area_MON := area_triangle M (0, 0) N in
    (real.sqrt 14 / 2) < area_MON ∧ area_MON ≤ real.sqrt 6) :=
begin
  sorry
end

end ellipse_equation_and_triangle_area_l747_747810


namespace number_of_gp_angles_l747_747320

noncomputable def count_angles_in_gp : ℕ :=
  let angles := {θ : ℝ | 0 < θ ∧ θ < 2 * Real.pi ∧ θ / (Real.pi / 2) ∉ ℤ ∧ 
    ( ∃ {a b c}, [sin θ, cos θ, tan θ].permutes [a, b, c] ∧ a = b * c) } in
  Set.card angles

theorem number_of_gp_angles : count_angles_in_gp = 4 :=
  sorry

end number_of_gp_angles_l747_747320


namespace sales_commission_change_point_l747_747257

theorem sales_commission_change_point 
    (total_sales : ℝ) (remitted_amount : ℝ) (X : ℝ) :
    (total_sales = 32500) →
    (remitted_amount = 31100) →
    (∀ X <= total_sales, commission_Calculation_UpTo : ℝ = 0.05 * X) →
    (∀ X <= total_sales, commission_Calculation_After : ℝ = 0.04 * (total_sales - X)) →
    let commission := 0.05 * X + 0.04 * (total_sales - X) in
    (remitted_amount = total_sales - commission) →
    X = 10000 := 
by
  intros h1 h2 h3 h4 h5
  have h6: 3100 = 32500 - (0.05 * X + 0.04 * (32500 - X)) by sorry
  simp only [h6] at *
  sorry

end sales_commission_change_point_l747_747257


namespace area_of_T_is_8_l747_747069

noncomputable def omega : Complex := Complex.exp (Complex.I * Real.pi / 4)

def T : Set Complex :=
  {z : Complex | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
    z = a + b * omega + c * Complex.conj(omega)}

theorem area_of_T_is_8 : measure_theory.volume T.to_measurable = 8 := sorry

end area_of_T_is_8_l747_747069


namespace proportional_segments_in_triangle_l747_747878

theorem proportional_segments_in_triangle
  (XY YZ XZ : ℝ)
  (h1 : XY = 400)
  (h2 : YZ = 450)
  (h3 : XZ = 480)
  (P : in_triangle X Y Z)
  (parallel_segments : (d : ℝ) × (segments_eq : (XY + YZ + XZ) = 3 * d))
  (additional_cond : proportional_length_segments X Y Z d) :
  d = 18000 / 133 := by sorry

end proportional_segments_in_triangle_l747_747878


namespace george_triangles_cover_l747_747729

def regular_hexagon (H : Type*) [metric_space H] [is_regular_hexagon H (1 : ℝ)] : Prop := ∀ P Q : H, dist P Q = 1

theorem george_triangles_cover (H : Type*) [metric_space H] [is_regular_hexagon H (1 : ℝ)] : ∃ n, 
  (∑ i in range n, area_of_largest_equilateral_triangle_not_yet_placed i) ≥ 0.9 := sorry

end george_triangles_cover_l747_747729


namespace solve_x_for_collinear_and_same_direction_l747_747402

-- Define vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (-1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (-x, 2)

-- Define the conditions for collinearity and same direction
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k • b.1, k • b.2)

def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k • b.1, k • b.2)

theorem solve_x_for_collinear_and_same_direction
  (x : ℝ)
  (h_collinear : collinear (vector_a x) (vector_b x))
  (h_same_direction : same_direction (vector_a x) (vector_b x)) :
  x = Real.sqrt 2 :=
sorry

end solve_x_for_collinear_and_same_direction_l747_747402


namespace dice_sum_probability_l747_747496

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747496


namespace normal_distribution_symmetry_proof_l747_747824

noncomputable theory

open Real

variables (σ : ℝ)
def X : Type := ℝ

def P (X : ℝ) := sorry -- Placeholder for probability function on X

theorem normal_distribution_symmetry_proof
  (h1 : ∀ X, X ~ Normal(2, σ^2))
  (h2 : P(X < 4) = 0.84) :
  P(X ≤ 0) = 0.16 :=
sorry

end normal_distribution_symmetry_proof_l747_747824


namespace sequence_sum_2016_l747_747619

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2) + 1
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem sequence_sum_2016 : S 2016 = 3024 := by
  sorry

end sequence_sum_2016_l747_747619


namespace probability_of_sum_greater_than_15_l747_747654

-- Definition of the dice and outcomes
def total_outcomes : ℕ := 6 * 6 * 6
def favorable_outcomes : ℕ := 10

-- Probability calculation
def probability_sum_gt_15 : ℚ := favorable_outcomes / total_outcomes

-- Theorem to be proven
theorem probability_of_sum_greater_than_15 : probability_sum_gt_15 = 5 / 108 := by
  sorry

end probability_of_sum_greater_than_15_l747_747654


namespace num_four_digit_even_numbers_is_12_l747_747664

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def four_digit_even_numbers (digits : List ℕ) (is_even_fn : ℕ → Prop) : ℕ :=
  (digits.prod n (is_even_fn : ℕ → Prop))

theorem num_four_digit_even_numbers_is_12 : 
  ∀ (digits: List ℕ),
  digits = [1, 2, 3, 4] →
  four_digit_even_numbers digits is_even = 12 :=
by {
  sorry
}

end num_four_digit_even_numbers_is_12_l747_747664


namespace find_area_ABD_l747_747893

-- Definitions
variables {A B C P D: Type}
variables {AB BC CA BD: ℝ}
variables [triangle A B C] [tangents A B C P D]
variables [parallel PD AC]

-- Constants from the problem
constant AB_length : AB = 39
constant BC_length : BC = 45
constant CA_length : CA = 42
constant area_ABC : ℝ := 756

theorem find_area_ABD : ∃ (x : ℝ), x = 168 ∧ 
  area_ABC = (39 * 42 * sin (angle BAC)) / 2 ∧
  BD = |BC - DC| ∧
  (DC = (42 * 5) / (3 * 2)) ∧
  (
    area_ABC * ratio BD BC = 168 :=
begin
  sorry
end

end find_area_ABD_l747_747893


namespace f_for_negative_x_l747_747380

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x * abs (x - 2) else 0  -- only assume the given case for x > 0

theorem f_for_negative_x (x : ℝ) (h : x < 0) : 
  f x = x * abs (x + 2) :=
by
  -- Sorry block to bypass the proof
  sorry

end f_for_negative_x_l747_747380


namespace tangent_point_exists_l747_747734

noncomputable theory

open EuclideanGeometry

theorem tangent_point_exists (O : Point) (r : ℝ) (A : Point) (B : Point) (C D : Point)
  (h_circle : circle O r = set_of (λ P, dist O P = r))
  (h_A_on_circle : dist O A = r)
  (h_tangent_at_A : is_tangent (line_through A B) (circle O r))
  (h_secant_intersect : line_through B O ∩ circle O r = {C, D})
  (h_B_on_tangent : B ∈ (line_through A B) ∧ B ∉ circle O r) :
  dist B C = dist C D :=
by
  sorry

end tangent_point_exists_l747_747734


namespace find_hourly_rate_l747_747739

theorem find_hourly_rate (x : ℝ) (h1 : 40 * x + 10.75 * 16 = 622) : x = 11.25 :=
sorry

end find_hourly_rate_l747_747739


namespace correct_statements_euler_formula_l747_747773

theorem correct_statements_euler_formula (x : ℝ) :
  (cos 2 < 0 ∧ sin 2 > 0) ∧
  (¬(e ^ (Real.pi * Real.I) = 0 * Real.I + α * Real.sin.PI)) ∧
  (∀ (x : ℝ), abs (e ^ (x * Real.I) / (Real.sqrt 3 + Real.I)) = 1 / 2) ∧
  (conj (e ^ (Real.pi / 3 * Real.I)) = (1 / 2) - (Real.sqrt 3  / 2) * Real.I) :=
begin
  sorry
end

end correct_statements_euler_formula_l747_747773


namespace trapezoid_area_ratio_l747_747888

noncomputable def isKite (A B C D : Point) :=
  (dist A B = 14) ∧ (dist B C = 16) ∧ (dist C D = 14) ∧ (dist D A = 16)
  ∧ (let O := midpoint (A, C) in 
        (O ∈ line_segment A C) ∧ (O ∈ line_segment B D) ∧ 
        is_right_angle A O B ∧ is_right_angle C O D)
  ∧ (∃ P : Point, P ∈ line_segment A B ∧ is_perpendicular P (line_segment (O, P)))

theorem trapezoid_area_ratio (A B C D X Y : Point) :
  isKite A B C D ∧ midpoint X A D ∧ midpoint Y B C →
  ratio_area_trapezoid A B Y X X Y C D = 1 :=
by
  sorry

end trapezoid_area_ratio_l747_747888


namespace general_term_formula_l747_747826

-- Define the sequence and the sum of the first n terms
def a_n (n : ℕ) : ℕ := sorry
def S_n (n : ℕ) : ℕ := sorry

-- Given conditions
axiom sum_of_terms : ∀ n : ℕ, S_n n = ∑ i in Finset.range (n + 1), a_n i
axiom condition : ∀ n : ℕ, 2 * (S_n n) = 3^n + 3

-- Statement of the theorem
theorem general_term_formula :
  ∀ n : ℕ,
    (a_n 1 = 3) ∧
    (∀ n : ℕ, n ≥ 2 → a_n n = 3^(n-1)) :=
sorry

end general_term_formula_l747_747826


namespace not_all_logarithmic_functions_decreasing_l747_747726

noncomputable def log_increasing (a : ℝ) (h : 1 < a) {x y : ℝ} (hx : 0 < x) (hy : 0 < y) : x < y → log a x < log a y :=
sorry

noncomputable def log_decreasing (a : ℝ) (h : 0 < a ∧ a < 1) {x y : ℝ} (hx : 0 < x) (hy : 0 < y) : x < y → log a x > log a y :=
sorry

theorem not_all_logarithmic_functions_decreasing: ¬ ∀ (a : ℝ), ∀ (x y : ℝ), (\(a > 0 \land a ≠ 1 \land 0 < x \land 0 < y \land x < y) → log a x > log a y) :=
by {
  intro h,
  have contra : ∃ a, 1 < a := ⟨2, by norm_num⟩,
  cases contra with a ha,
  specialize h a,
  have h_inc := log_increasing a ha,
  specialize h_inc (by norm_num) (by norm_num) (by norm_num),
  specialize h (a = 2),
  linarith,
}

end not_all_logarithmic_functions_decreasing_l747_747726


namespace algebraic_expression_value_l747_747815

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^2 + a + 1 = 2 :=
sorry

end algebraic_expression_value_l747_747815


namespace solution_set_of_inequality_l747_747149

theorem solution_set_of_inequality (x : ℝ) : x ≠ -1 → (x + 2 / (x + 1) > 2) ↔ (x ∈ set.Iio (-1) ∪ set.Ioi 1) :=
by
  sorry

end solution_set_of_inequality_l747_747149


namespace probability_sum_is_ten_l747_747434

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747434


namespace surface_area_of_circumscribed_sphere_l747_747359

def cube_edge_length : ℝ := 1

def space_diagonal (a : ℝ) : ℝ := a * Real.sqrt 3

def sphere_radius (d : ℝ) : ℝ := d / 2

def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem surface_area_of_circumscribed_sphere :
  sphere_surface_area (sphere_radius (space_diagonal cube_edge_length)) = 3 * Real.pi := by
  sorry

end surface_area_of_circumscribed_sphere_l747_747359


namespace concave_number_count_l747_747238

theorem concave_number_count :
  {n : Fin 6 | ∃ a1 a2 a3 a4 a5 : Fin 6,
    a1 > a2 ∧ a2 > a3 ∧ a3 < a4 ∧ a4 < a5 ∧
    n = nat_of_digits 10 [a1, a2, a3, a4, a5]}.card = 146 := 
sorry

end concave_number_count_l747_747238


namespace dice_sum_prob_10_l747_747465

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747465


namespace probability_sum_is_ten_l747_747438

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747438


namespace dice_sum_10_probability_l747_747430

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747430


namespace minimize_total_resistance_l747_747376

variable (a1 a2 a3 a4 a5 a6 : ℝ)
variable (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6)

/-- Theorem: Given resistances a1, a2, a3, a4, a5, a6 such that a1 > a2 > a3 > a4 > a5 > a6, 
arranging them in the sequence a1 > a2 > a3 > a4 > a5 > a6 minimizes the total resistance
for the assembled component. -/
theorem minimize_total_resistance : 
  True := 
sorry

end minimize_total_resistance_l747_747376


namespace factorial_difference_l747_747751

theorem factorial_difference (h12 : fact 12 = 479001600) (h11 : fact 11 = 39916800) : 
    fact 12 - fact 11 = 439084800 := by
    sorry

end factorial_difference_l747_747751


namespace remainder_f24_mod_89_l747_747388

def f (x : ℤ) : ℤ := x^2 - 2

def f_iterate (n : ℕ) (x : ℤ) : ℤ :=
nat.rec_on n x (λ _ y, f y)

theorem remainder_f24_mod_89 :
  (f_iterate 24 18) % 89 = 47 :=
by {
  sorry
}

end remainder_f24_mod_89_l747_747388


namespace quadrilateral_side_lengths_l747_747357

theorem quadrilateral_side_lengths
  (R : ℝ)
  (d : ℝ)
  (a : ℝ)
  (AB BC CD DA : ℝ)
  (h₁ : d = (R / 2) * sqrt (4 - sqrt 5))
  (h₂ : a^2 = R^2 * sqrt (5 + 2 * sqrt 5)) :
  AB = (R / 2) * (sqrt 5 - 1) ∧
  BC = (R / 2) * sqrt (10 - 2 * sqrt 5) ∧
  CD = (R / 2) * sqrt (10 + 2 * sqrt 5) ∧
  DA = (R / 2) * (sqrt 5 + 1) := by
  sorry

end quadrilateral_side_lengths_l747_747357


namespace probability_sum_is_10_l747_747455

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747455


namespace estimating_population_using_sample_l747_747677

theorem estimating_population_using_sample :
    ∃ (C1 C2 C3 C4 : Prop),
    C1 = false ∧
    C2 = true ∧
    C3 = false ∧
    C4 = true ∧
    (C1 || C2 || C3 || C4) ∧
    (C1 = false ∨ C1 = true) ∧
    (C2 = false ∨ C2 = true) ∧ 
    (C3 = false ∨ C3 = true) ∧ 
    (C4 = false ∨ C4 = true) ∧
    (C1 ↔ C1 = true ∨ C1 = false) ∧ 
    (C2 ↔ C2 = true ∨ C2 = false) ∧ 
    (C3 ↔ C3 = true ∨ C3 = false) ∧
    (C4 ↔ C4 = true ∨ C4 = false) ∧
    ([C1, C2, C3, C4].count true = 2) :=
by {
    -- Definitions of the statements
    let C1 := false,
    let C2 := true,
    let C3 := false,
    let C4 := true,
    
    -- Check the conditions
    use [C1, C2, C3, C4],
    
    -- Verify each condition
    repeat { split },
    { exact rfl },
    { exact rfl },
    { exact rfl },
    { exact rfl },
    
    -- Check that there are 2 true statements
    { simp },
    sorry
}

end estimating_population_using_sample_l747_747677


namespace probability_sum_is_ten_l747_747432

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747432


namespace number_of_elements_in_intersection_l747_747922

-- Definitions of the conditions
def is_isosceles_triangle (T : Type) : Prop := sorry  -- Define the property of being an isosceles triangle
def has_side_length_one_and_angle_36 (P : Type) : Prop := sorry  -- Define the property of having one side of length 1 and one angle of 36 degrees

-- Sets M and P
def M := {T : Type | is_isosceles_triangle T}
def P := {P : Type | has_side_length_one_and_angle_36 P}

-- Prove the number of elements in the intersection of M and P is 4
theorem number_of_elements_in_intersection : Fintype.card (M ∩ P) = 4 := sorry

end number_of_elements_in_intersection_l747_747922


namespace correct_ordering_l747_747822

variable (f : ℝ → ℝ)

def periodic_shift (x : ℝ) := f (x + 2) = -f x
def monotonic_increase (x1 x2 : ℝ) := 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 < f x2
def symmetric_graph (x : ℝ) := f (x + 2) = f (-x + 2)

theorem correct_ordering :
  periodic_shift f →
  (∀ x1 x2, monotonic_increase f x1 x2) →
  (∀ x, symmetric_graph f x) →
  f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  intros hp hm hs
  sorry

end correct_ordering_l747_747822


namespace dice_sum_probability_l747_747500

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747500


namespace length_of_AB_l747_747534

-- Declare the variables and constants
variables (AB BC AC : ℝ)
constant AC_length : AC = 100
constant slope_ratio : AB / BC = 4 / 3
constant right_triangle : AB^2 + BC^2 = AC^2

-- Define the goal of the proof
theorem length_of_AB : AB = 8 * real.sqrt 10 :=
  by
  -- Utilize the given conditions
  rw [AC_length, right_triangle, slope_ratio]
  -- Calculate and prove the length of AB
  sorry

end length_of_AB_l747_747534


namespace inequality_ineqs_l747_747839

theorem inequality_ineqs (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_cond : x * y + y * z + z * x = 1) :
  (27 / 4) * (x + y) * (y + z) * (z + x) 
  ≥ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2
  ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2 
  ≥ 
  6 * Real.sqrt 3 := by
  sorry

end inequality_ineqs_l747_747839


namespace triangle_shape_l747_747875

-- Let there be a triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively
variables (A B C : ℝ) (a b c : ℝ) (b_ne_1 : b ≠ 1)
          (h1 : (log (b) (C / A)) = (log (sqrt (b)) (2)))
          (h2 : (log (b) (sin B / sin A)) = (log (sqrt (b)) (2)))

-- Define the theorem that states the shape of the triangle
theorem triangle_shape : A = π / 6 ∧ B = π / 2 ∧ C = π / 3 ∧ (A + B + C = π) :=
by
  -- Proof is provided in the solution, skipping proof here
  sorry

end triangle_shape_l747_747875


namespace inscribed_circle_locus_l747_747662

/-- Given a circle with two fixed points A and B,
  and a point C that moves along the circumference of the circle,
  the locus of the center of the inscribed circle of triangle ACB
  is the arcs of circles centered at F1 and F2 within the main circle. -/
theorem inscribed_circle_locus (A B : ℝ) (hA : on_circle A) (hB : on_circle B)
  (C : ℝ) (hC : moves_on_circle C) (F1 F2 : ℝ) (hF1 : is_diameter_endpoint F1 A B) (hF2 : is_diameter_endpoint F2 A B) :
  locus_of_center_of_inscribed_circle (A B C) = arcs_of_circles_centered_at F1 F2 :=
sorry

end inscribed_circle_locus_l747_747662


namespace positional_relationship_l747_747866

-- Defining the concepts of parallelism, containment, and positional relationships
structure Line -- subtype for a Line
structure Plane -- subtype for a Plane

-- Definitions and Conditions
def is_parallel_to (l : Line) (p : Plane) : Prop := sorry  -- A line being parallel to a plane
def is_contained_in (l : Line) (p : Plane) : Prop := sorry  -- A line being contained within a plane
def are_skew (l₁ l₂ : Line) : Prop := sorry  -- Two lines being skew
def are_parallel (l₁ l₂ : Line) : Prop := sorry  -- Two lines being parallel

-- Given conditions
variables (a b : Line) (α : Plane)
axiom Ha : is_parallel_to a α
axiom Hb : is_contained_in b α

-- The theorem to be proved
theorem positional_relationship (a b : Line) (α : Plane) 
  (Ha : is_parallel_to a α) 
  (Hb : is_contained_in b α) : 
  (are_skew a b ∨ are_parallel a b) :=
sorry

end positional_relationship_l747_747866


namespace minimum_phi_l747_747015

-- Definitions based on the problem description
def f (x : ℝ) : ℝ := sin x * cos x + sqrt 3 * (cos x)^2 - sqrt 3 / 2

-- Lean theorem stating the proof problem
theorem minimum_phi (φ : ℝ) (hφ : φ > 0) : (∀ x : ℝ, f (x - φ) = f (-(x - φ))) → φ = 5 * π / 12 := 
sorry

end minimum_phi_l747_747015


namespace lcm_proof_l747_747351

theorem lcm_proof (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) : Nat.lcm b c = 540 :=
sorry

end lcm_proof_l747_747351


namespace egg_condition_difference_l747_747571

theorem egg_condition_difference :
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  in perfect_condition - cracked_eggs = 9 :=
by
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  show perfect_condition - cracked_eggs = 9, by sorry

end egg_condition_difference_l747_747571


namespace sequence_difference_l747_747553

noncomputable def f (n : ℕ) (h : n > 0) : ℝ :=
  ∑ i in finset.range (3 * n), (1 : ℝ) / (i + 1)

theorem sequence_difference (n : ℕ) (h: n > 0) :
  (f (n + 1) (nat.succ_pos n) - f n h) = (1 / (3 * n) + 1 / (3 * n + 1) + 1 / (3 * n + 2)) :=
  sorry

end sequence_difference_l747_747553


namespace tiffany_lives_l747_747657

theorem tiffany_lives (initial_lives lives_lost lives_after_next_level lives_gained : ℕ)
  (h1 : initial_lives = 43)
  (h2 : lives_lost = 14)
  (h3 : lives_after_next_level = 56)
  (h4 : lives_gained = lives_after_next_level - (initial_lives - lives_lost)) :
  lives_gained = 27 :=
by {
  sorry
}

end tiffany_lives_l747_747657


namespace final_price_with_discount_l747_747028

-- Definition of prices and discount
def price_coffee := 6
def price_cheesecake := 10
def discount := 25 / 100

-- Mathematical problem statement
theorem final_price_with_discount :
  let total_price := price_coffee + price_cheesecake in
  let discount_amount := discount * total_price in
  let final_price := total_price - discount_amount in
  final_price = 12 :=
by
  sorry

end final_price_with_discount_l747_747028


namespace hyperbola_eccentricity_l747_747390

open Classical

variable (a b e : ℝ)
variable (C : set (ℝ × ℝ))
variable (A F B : ℝ × ℝ)

def is_hyperbola (C : set (ℝ × ℝ)) (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ C ∧ (x^2 / a^2) - (y^2 / b^2) = 1

def vertex_A (A : ℝ × ℝ) (a : ℝ) : Prop := A = (-a, 0)
def focus_F (F : ℝ × ℝ) (c : ℝ) : Prop := F = (c, 0)
def point_B (B : ℝ × ℝ) (b : ℝ) : Prop := B = (0, b)
def perpendicular_vectors (A F B : ℝ × ℝ) : Prop := (A.1 * F.1) + (A.2 * F.2) = 0

theorem hyperbola_eccentricity (a b e : ℝ) (C : set (ℝ × ℝ))
  (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
  (ha : a > 0) (hb : b > 0)
  (hC : is_hyperbola C a b)
  (hA : vertex_A A a)
  (hF : focus_F F (sqrt (a^2 * (e^2 + 1))))
  (hB : point_B B b)
  (hBA_BF : perpendicular_vectors A F B) :
  e = (sqrt 5 + 1) / 2 :=
by
  -- Proof would go here
  sorry

end hyperbola_eccentricity_l747_747390


namespace pedal_circles_coincide_l747_747073

def Point : Type := ℝ × ℝ

structure Triangle :=
(A B C : Point)

structure PedalTriangle (Δ : Triangle) (P : Point) :=
(PA PB PC : Point)

def isPedalTriangle := sorry

structure Circle :=
(center : Point)
(radius : ℝ)

def pedalCircle (Δ : Triangle) (P : Point) : Circle := sorry

def symmetricReflectionPoint (Δ : Triangle) (D : Point) : Point := sorry

theorem pedal_circles_coincide (ABC : Triangle) (D : Point) :
  pedalCircle ABC D = pedalCircle ABC (symmetricReflectionPoint ABC D) :=
sorry

end pedal_circles_coincide_l747_747073


namespace sum_of_reciprocals_primes_less_2017_l747_747064

open Real

noncomputable def α := sorry -- Assume α is a positive real number such that 2 < α < 2.7
noncomputable def n := floor (α^2015)

def is_prime (p : Nat) : Prop := sorry -- Definition of a prime number

def primes_up_to_n := {p : Nat | is_prime p ∧ p ≤ n}

theorem sum_of_reciprocals_primes_less_2017 :
  (∑ i in primes_up_to_n, ∑ j in primes_up_to_n, if 2 ≤ i ∧ i ≤ j then 1/(i * j) else 0) < 2017 :=
begin
  sorry
end

end sum_of_reciprocals_primes_less_2017_l747_747064


namespace sum_of_cubes_of_roots_eq_sum_of_fractions_roots_eq_l747_747687

-- Part (a)
theorem sum_of_cubes_of_roots_eq :
  let x1 := √((5 / 4) + (17 / 4)^(0.5))
  let x2 := √((5 / 4) - (17 / 4)^(0.5))
  2 * x1^3 - 5 * x1 + 1 = 0 ∧
  2 * x2^3 - 5 * x2 + 1 = 0 → 
  (x1^3 + x2^3) = 95 / 8 := by
  sorry

-- Part (b)
theorem sum_of_fractions_roots_eq :
  let x1 := √((11 / 4) + (17 / 4)^(0.5))
  let x2 := √((11 / 4) - (17 / 4)^(0.5))
  2 * x1^2 - 11 * x1 + 13 = 0 ∧
  2 * x2^2 - 11 * x2 + 13 = 0 →
  (x1 / x2 + x2 / x1) = 69 / 26 := by
  sorry

end sum_of_cubes_of_roots_eq_sum_of_fractions_roots_eq_l747_747687


namespace tamika_carlos_probability_l747_747601

theorem tamika_carlos_probability :
  let tamika_results := [10 + 11, 10 + 12, 11 + 12],
      carlos_results := [4 * 6, 4 * 7, 6 * 7] in
  (∃ t ∈ tamika_results, ∃ c ∈ carlos_results, t > c) = false :=
by sorry

end tamika_carlos_probability_l747_747601


namespace tan_triple_angle_l747_747010

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l747_747010


namespace distance_problem_l747_747714

-- Define the problem
theorem distance_problem
  (x y : ℝ)
  (h1 : x + y = 21)
  (h2 : x / 60 + 21 / 60 = 10 / 60 + y / 4) :
  x = 19 ∧ y = 2 :=
by
  sorry

end distance_problem_l747_747714


namespace angle_sum_inscribed_circle_l747_747254

-- Defining the geometrical setup and properties
variables {O A B C D : Type}
variable [noncomputable] (c : Circumcircle O A B C D) -- Quadrilateral inscribed in circle O

-- Statement of the theorem
theorem angle_sum_inscribed_circle (h : InscribedQuadrilateral O A B C D) :
  ∠AOB + ∠COD = 180 :=
sorry

end angle_sum_inscribed_circle_l747_747254


namespace circle_with_center_and_point_eq_l747_747150

noncomputable def circle_equation_standard (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 1) ^ 2 = 2

theorem circle_with_center_and_point_eq :
  ∀ (x y : ℝ), (circle_equation_standard 2 2) → (x - 1) ^ 2 + (y - 1) ^ 2 = 2 :=
by
  intros x y h
  rw [circle_equation_standard]
  sorry

end circle_with_center_and_point_eq_l747_747150


namespace total_employees_l747_747715

-- Defining the number of part-time and full-time employees
def p : ℕ := 2041
def f : ℕ := 63093

-- Statement that the total number of employees is the sum of part-time and full-time employees
theorem total_employees : p + f = 65134 :=
by
  -- Use Lean's built-in arithmetic to calculate the sum
  rfl

end total_employees_l747_747715


namespace expected_crocodiles_with_cane_approx_l747_747709

noncomputable def expected_crocodiles_with_cane (n : ℕ) (k : ℕ) : ℝ :=
∑ i in range (n - k + 1), 1 / i

theorem expected_crocodiles_with_cane_approx :
  let n := 10 in
  let k := 3 in
  expected_crocodiles_with_cane n k + 1 = 3.59 :=
by
  sorry

end expected_crocodiles_with_cane_approx_l747_747709


namespace correct_options_l747_747134

noncomputable def problem (a n m : ℝ → ℝ → ℝ) 
  (h1 : m ⊥ n) 
  (h2 : a ⊥ n) 
  (h3 : cos (inner a n) = sqrt 3 / 2) 
  (h4 : cos (inner m n) = 1 / 2) 
  : Prop := 
  (a ⊥ m) ∧ 
  (a ⊥ n) ∧ 
  (inner a n = sqrt 3 / 2) ∧ 
  (inner m n = 1 / 2) ∧ 
  (inner m n = π / 3)

theorem correct_options (a n m : ℝ → ℝ → ℝ)
  (h1 : m ⊥ n)
  (h2 : a ⊥ n)
  (h3 : cos (inner a n) = sqrt 3 / 2)
  (h4 : cos (inner m n) = 1 / 2) :
  problem a n m h1 h2 h3 h4 := by
  sorry

end correct_options_l747_747134


namespace hydrochloric_acid_moles_l747_747850

theorem hydrochloric_acid_moles (amyl_alcohol moles_required : ℕ) 
  (h_ratio : amyl_alcohol = moles_required) 
  (h_balanced : amyl_alcohol = 3) :
  moles_required = 3 :=
by
  sorry

end hydrochloric_acid_moles_l747_747850


namespace clipping_time_proof_l747_747908

-- Definitions based on the given conditions
def clip_time (C : ℕ) : ℕ := 16 * C
def clean_ears_time : ℕ := 2 * 90
def shampoo_time : ℕ := 5 * 60
def total_time (C : ℕ) : ℕ := clip_time(C) + clean_ears_time + shampoo_time

-- Theorem statement to prove
theorem clipping_time_proof : ∃ (C : ℕ), total_time(C) = 640 :=
begin
  -- Solution is omitted; only the statement is provided
  sorry
end

end clipping_time_proof_l747_747908


namespace central_angle_of_sector_l747_747384

variable (r θ : ℝ)
variable (r_pos : 0 < r) (θ_pos : 0 < θ)

def perimeter_eq : Prop := 2 * r + r * θ = 5
def area_eq : Prop := (1 / 2) * r^2 * θ = 1

theorem central_angle_of_sector :
  perimeter_eq r θ ∧ area_eq r θ → θ = 1 / 2 :=
sorry

end central_angle_of_sector_l747_747384


namespace area_of_quadrilateral_PQRS_l747_747107

theorem area_of_quadrilateral_PQRS (PQ RS PR PT : ℝ) (T Q R P S : Point) 
  (h1 : ∠ P Q R = 90º) (h2 : ∠ P S R = 90º)
  (h3 : dist P R = 25) (h4 : dist R S = 40) (h5 : dist P T = 10)
  (h6 : P ≠ Q) (h7 : Q ≠ R) (h8 : R ≠ S) (h9 : S ≠ P)
  (h10 : dist P Q = PQ) (h11 : dist R S = RS) :
  area_of_quadrilateral P Q R S = 687.5 := sorry

end area_of_quadrilateral_PQRS_l747_747107


namespace zero_of_f_in_interval_l747_747158

noncomputable def f (x : ℝ) := log 3 x - 8 + 2 * x

theorem zero_of_f_in_interval : 0 ∈ set.Ioo (f 3) (f 4) :=
by
  have : f 3 < 0 := by
    calc
      f 3 = log 3 3 - 8 + 2 * 3 := by sorry
      ... < 0 := by sorry
  have : 0 < f 4 := by
    calc
      f 4 = log 3 4 - 8 + 2 * 4 := by sorry
      ... > 0 := by sorry
  exact sorry

end zero_of_f_in_interval_l747_747158


namespace maximum_number_of_acceptable_ages_l747_747692

theorem maximum_number_of_acceptable_ages (avg_age std_dev : ℝ) (h_avg : avg_age = 31) (h_std : std_dev = 9) :
  let lower_bound := avg_age - std_dev
  let upper_bound := avg_age + std_dev
  let acceptable_ages := [Int.toNat (Int.floor lower_bound) : ℕ] :: 
                         [Int.toNat (Int.floor upper_bound) : ℕ]
  ∃ (n : ℕ), n = 19 ∧ 
  (∀ i ∈ acceptable_ages, (i >= Int.floor lower_bound ∧ i <= Int.ceil upper_bound)) :=
by
  sorry

end maximum_number_of_acceptable_ages_l747_747692


namespace positive_diff_between_two_smallest_prime_factors_296045_positive_diff_between_two_smallest_prime_factors_296045_Sorry_l747_747178

def smallest_prime_factors (n : ℕ) : list ℕ :=
  if n = 296045 then [3, 7, 11, 61] else []

def positive_difference (a b : ℕ) : ℕ :=
  if a > b then a - b else b - a

theorem positive_diff_between_two_smallest_prime_factors_296045 :
  positive_difference 3 7 = 4 :=
by
  simp [positive_difference]
  exact nat.sub_eq_of_eq_add' rfl

-- Sorry version
theorem positive_diff_between_two_smallest_prime_factors_296045_Sorry :
  positive_difference 3 7 = 4 := sorry

end positive_diff_between_two_smallest_prime_factors_296045_positive_diff_between_two_smallest_prime_factors_296045_Sorry_l747_747178


namespace curve_C₂_is_correct_l747_747520

-- Definitions for Cartesian and Polar coordinate systems, not explicitly necessary in the statement
-- Defining the initial condition of curve C₁
def curve_C₁ (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Condition defining the problem
def curve_C₂_eq (α : ℝ) : Prop :=
  curve_C₁ (α + Real.pi / 3) = 4 * Real.sin (α + 60 * Real.pi / 180)

-- Theorem statement
theorem curve_C₂_is_correct (α : ℝ) : curve_C₂_eq α :=
  sorry

end curve_C₂_is_correct_l747_747520


namespace main_theorem_l747_747196

noncomputable def proof_problem : Prop :=
  ∀ (a b c d x : ℝ), 
   (a > b ∧ b > c ∧ c > d ∧ d ≥ 0) → 
   (a + d = b + c) → 
   (x > 0) → 
   x^a + x^d ≥ x^b + x^c

theorem main_theorem : proof_problem := 
begin
  sorry
end

end main_theorem_l747_747196


namespace geometry_synonyms_l747_747587

-- Definitions based on conditions in a)
def Ray := {x : ℝ × ℝ // ∃ p : ℝ × ℝ, ∃ d : ℝ × ℝ, d ≠ (0, 0) ∧ x = p + t • d for some t ≥ 0}

def HalfLine := Ray

def Tetrahedron := {vertices : Finset (ℝ × ℝ × ℝ) // vertices.card = 4 ∧ ∀ σ : Finset (ℝ × ℝ × ℝ), σ ⊆ vertices → σ.card = 3 → ∃! plane : ℝ ∧ ∃ a b c : ℝ, ∀ p ∈ σ, p.1 * a + p.2 * b + p.3 * c = plane}

def TriangularPyramid := Tetrahedron

def Bisector (ℓ : ℝ × ℝ) := {p : ℝ × ℝ // ∃ a : ℝ × ℝ, ∃ b : ℝ × ℝ, p = (a + b) / 2}

def AngleBisector := {x : ℝ × ℝ // ∃ p a b : ℝ × ℝ, ∃ t1 t2 : ℝ, t1 > 0 ∧ t2 > 0 ∧ (x = p + t1 • (a - p)) ∧ (x = p + t2 • (b - p)) ∧ angle a p b = (1 / 2) * angle a p} 

theorem geometry_synonyms :
  (Ray = HalfLine) ∧ (Tetrahedron = TriangularPyramid) ∧ (AngleBisector = Bisector for angles) := by
  sorry

end geometry_synonyms_l747_747587


namespace dice_sum_10_probability_l747_747431

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747431


namespace inscribed_circle_area_percent_l747_747181

theorem inscribed_circle_area_percent (a : ℝ) (h : a > 0) : 
  Real.floor ((π / 4) * 100 + 0.5) = 79 :=
by
  sorry

end inscribed_circle_area_percent_l747_747181


namespace area_not_covered_by_smaller_squares_l747_747264

-- Define the conditions given in the problem
def side_length_larger_square : ℕ := 10
def side_length_smaller_square : ℕ := 4
def area_of_larger_square : ℕ := side_length_larger_square * side_length_larger_square
def area_of_each_smaller_square : ℕ := side_length_smaller_square * side_length_smaller_square

-- Define the total area of the two smaller squares
def total_area_smaller_squares : ℕ := area_of_each_smaller_square * 2

-- Define the uncovered area
def uncovered_area : ℕ := area_of_larger_square - total_area_smaller_squares

-- State the theorem to prove
theorem area_not_covered_by_smaller_squares :
  uncovered_area = 68 := by
  -- Placeholder for the actual proof
  sorry

end area_not_covered_by_smaller_squares_l747_747264


namespace four_digit_even_and_multiple_of_7_sum_l747_747067

def num_four_digit_even_numbers : ℕ := 4500
def num_four_digit_multiples_of_7 : ℕ := 1286
def C : ℕ := num_four_digit_even_numbers
def D : ℕ := num_four_digit_multiples_of_7

theorem four_digit_even_and_multiple_of_7_sum :
  C + D = 5786 := by
  sorry

end four_digit_even_and_multiple_of_7_sum_l747_747067


namespace most_suitable_for_comprehensive_l747_747681

def comprehensive_survey (survey : Type) : Prop :=
  ∀ x : survey, x ∈ survey -- Every member of the population is included in the survey

inductive SurveyOption
| A : SurveyOption
| B : SurveyOption
| C : SurveyOption
| D : SurveyOption

open SurveyOption

theorem most_suitable_for_comprehensive : 
  ∀ S : SurveyOption, 
    (S = A → ¬ comprehensive_survey A) →
    (S = B → ¬ comprehensive_survey B) →
    (S = D → ¬ comprehensive_survey D) →
    (S = C → comprehensive_survey C) :=
by
  intros S hA hB hD hC
  cases S
  case A => exact hA rfl
  case B => exact hB rfl
  case C => exact hC rfl
  case D => exact hD rfl

end most_suitable_for_comprehensive_l747_747681


namespace find_minimum_n_l747_747369

variable {a_1 d : ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a_1 d : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (2 * a_1 + (n - 1) * d)

def condition1 (a_1 : ℝ) : Prop := a_1 < 0

def condition2 (S : ℕ → ℝ) : Prop := S 7 = S 13

theorem find_minimum_n (a_1 d : ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a_1 d S)
  (h_a1_neg : condition1 a_1)
  (h_s7_eq_s13 : condition2 S) :
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, S n ≤ S m := 
sorry

end find_minimum_n_l747_747369


namespace minimum_participants_l747_747637

theorem minimum_participants
  (correct_first : ℕ)
  (correct_second : ℕ)
  (correct_third : ℕ)
  (correct_fourth : ℕ)
  (H_first : correct_first = 90)
  (H_second : correct_second = 50)
  (H_third : correct_third = 40)
  (H_fourth : correct_fourth = 20)
  (H_max_two : ∀ p : ℕ, 1 ≤ p ∧ p ≤ correct_first + correct_second + correct_third + correct_fourth → p ≤ 2 * (correct_first + correct_second + correct_third + correct_fourth))
  : ∃ n : ℕ, (correct_first + correct_second + correct_third + correct_fourth) / 2 = 100 :=
by
  sorry

end minimum_participants_l747_747637


namespace geometry_problem_l747_747247

open EuclideanGeometry -- Ensure Euclidean geometry definitions are in scope

-- Let's define our problem
theorem geometry_problem
  (r : Line ℝ) -- Line r
  (A B C D P : Point ℝ) -- Points A, B, C, D, and P
  (hABCDr : A ∈ r ∧ B ∈ r ∧ C ∈ r ∧ D ∈ r) -- Points A, B, C, D are on line r
  (h_order : collinear [A, B, C, D]) -- Points A, B, C, D are collinear in that order
  (h_notinP : P ∉ r) -- Point P is not on line r
  (h_angles : ∠(A, P, B) = ∠(C, P, D)) -- Angles ∠APB and ∠CPD are equal
  :
  ∃ G : Point ℝ, G ∈ r ∧ bisects_angle A P D G ∧ 
    (1 / (distance G A) + 1 / (distance G C) = 1 / (distance G B) + 1 / (distance G D)) :=
by sorry -- Proof is omitted

end geometry_problem_l747_747247


namespace convex_hexagon_diagonals_sides_l747_747748

noncomputable def convex_hexagon_exists : Prop :=
  ∃ (hexagon : Type) [convex_hexagon hexagon] (f : hexagon → ℝ),
  (∀ (side : hexagon), f side > 1) ∧ (∀ (diagonal : hexagon), f diagonal < 2)

theorem convex_hexagon_diagonals_sides : convex_hexagon_exists := 
sorry

end convex_hexagon_diagonals_sides_l747_747748


namespace solve_fraction_equation_l747_747019

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 0 ↔ x = -3 :=
sorry

end solve_fraction_equation_l747_747019


namespace donut_selection_count_l747_747093

-- Define the variables and their sum condition
def donutSelections : ℕ := ∑ (g c p : ℕ) in {g // g + c + p = 5}, 1

-- Main theorem statement: The count of selections is 21
theorem donut_selection_count : donutSelections = 21 :=
by
  sorry

end donut_selection_count_l747_747093


namespace circle_diameter_l747_747610

theorem circle_diameter (A : ℝ) (π : ℝ) (r : ℝ) (d : ℝ) (h1 : A = 64 * π) (h2 : A = π * r^2) (h3 : d = 2 * r) :
  d = 16 :=
by
  sorry

end circle_diameter_l747_747610


namespace probability_sum_is_ten_l747_747437

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747437


namespace express_scientific_notation_l747_747153

-- Given condition: the specific number we are dealing with
def number := 250000

-- We need to prove that it can be expressed in scientific notation as follows
def scientific_notation (n : ℕ) := ∃ (c : ℝ), c = 2.5 ∧ n = 250000
def notation_value (c : ℝ) (e : ℕ) := c * (10 ^ e)

theorem express_scientific_notation : scientific_notation number → notation_value 2.5 5 = number :=
by
  intro h
  cases h with c hc
  cases hc with hc1 hc2
  rw [←hc1, hc2]
  exact (by norm_num : 2.5 * 100000 = 250000)

end express_scientific_notation_l747_747153


namespace probability_sum_is_10_l747_747509

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747509


namespace purely_imaginary_iff_l747_747827

noncomputable def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0

theorem purely_imaginary_iff (a : ℝ) :
  isPurelyImaginary (Complex.mk ((a * (a + 2)) / (a - 1)) (a ^ 2 + 2 * a - 3))
  ↔ a = 0 ∨ a = -2 := by
  sorry

end purely_imaginary_iff_l747_747827


namespace tangent_circles_l747_747066

noncomputable def triangle (A B C : Point) (H : Point) (M : Point) (F : Point) (D K : Point) : Prop :=
  ∃ (circumcircle : Circle), 
    (orthocenter A B C H) ∧
    (midpoint B C M) ∧
    (foot_of_altitude A B C F) ∧
    (on_circumcircle D A B C circumcircle) ∧
    (angle H D A = 90) ∧
    (on_circumcircle K A B C circumcircle) ∧
    (angle D K H = 90)

theorem tangent_circles 
  {A B C H M F D K : Point} :
  triangle A B C H M F D K → tangent (circle D K H) (circle F K M) K :=
by
  intro h
  sorry

end tangent_circles_l747_747066


namespace total_clouds_l747_747291

theorem total_clouds (C B : ℕ) (h1 : C = 6) (h2 : B = 3 * C) : C + B = 24 := by
  sorry

end total_clouds_l747_747291


namespace sum_and_product_of_roots_l747_747592

-- Define the equation in terms of |x|
def equation (x : ℝ) : ℝ := |x|^3 - |x|^2 - 6 * |x| + 8

-- Lean statement to prove the sum and product of the roots
theorem sum_and_product_of_roots :
  (∀ x, equation x = 0 → (∃ L : List ℝ, L.sum = 0 ∧ L.prod = 16 ∧ ∀ y ∈ L, equation y = 0)) := 
sorry

end sum_and_product_of_roots_l747_747592


namespace maricela_oranges_per_tree_l747_747769

theorem maricela_oranges_per_tree : 
  ∀ (trees : Nat) (oranges_per_tree_gabriela : Nat) (oranges_per_tree_alba : Nat) (oranges_per_cup : Nat) (price_per_cup : Nat) (total_revenue : Nat),
  trees = 110 → 
  oranges_per_tree_gabriela = 600 → 
  oranges_per_tree_alba = 400 → 
  oranges_per_cup = 3 → 
  price_per_cup = 4 → 
  total_revenue = 220000 →
  let total_oranges_gabriela := trees * oranges_per_tree_gabriela in
  let total_oranges_alba := trees * oranges_per_tree_alba in
  let total_oranges := total_oranges_gabriela + total_oranges_alba in
  let total_cups := total_oranges / oranges_per_cup in
  let total_earnings := total_cups * price_per_cup in
  let remaining_revenue := total_revenue - total_earnings in
  let remaining_cups := remaining_revenue / price_per_cup in
  let total_oranges_needed_maricela := remaining_cups * oranges_per_cup in
  total_oranges_needed_maricela / trees = 500 :=
begin
  sorry
end

end maricela_oranges_per_tree_l747_747769


namespace distributive_laws_none_hold_l747_747549

def star (a b : ℝ) : ℝ := a + b + a * b

theorem distributive_laws_none_hold (x y z : ℝ) :
  ¬ (x * (y + z) = (x * y) + (x * z)) ∧
  ¬ (x + (y * z) = (x + y) * (x + z)) ∧
  ¬ (x * (y * z) = (x * y) * (x * z)) :=
by
  sorry

end distributive_laws_none_hold_l747_747549


namespace mode_and_median_of_student_times_l747_747659

-- Define the conditions: number of students and their times
def student_times : List (Nat × Nat) := [(9, 7), (16, 8), (14, 9), (11, 10)]

-- Given the total number of students
def total_students := 50

-- Define the mode and median calculation functions
def mode (data : List (Nat × Nat)) : Nat :=
  data.foldr (λ p acc, if p.1 > acc.1 then p else acc) (0, 0.0)).2

def median (data : List (Nat × Nat)) (total_count : Nat) : Float :=
  let sorted_data := data.foldr (λ p acc, acc ++ List.replicate p.1 p.2) []
  if total_count % 2 = 0 then
    let mid1 := sorted_data.get! (total_count / 2 - 1)
    let mid2 := sorted_data.get! (total_count / 2)
    (mid1 + mid2).toFloat / 2.0
  else
    sorted_data.get! (total_count / 2).toFloat

-- Main theorem
theorem mode_and_median_of_student_times :
  mode student_times = 8 ∧ median student_times total_students = 8.5 := by
  sorry

end mode_and_median_of_student_times_l747_747659


namespace log_comparison_l747_747071

/-- Assuming a = log base 3 of 2, b = natural log of 3, and c = log base 2 of 3,
    prove that c > b > a. -/
theorem log_comparison (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3)
                                (h2 : b = Real.log 3)
                                (h3 : c = Real.log 3 / Real.log 2) :
  c > b ∧ b > a :=
by {
  sorry
}

end log_comparison_l747_747071


namespace find_smallest_period_l747_747303

noncomputable def smallest_period (f : ℝ → ℝ) : ℝ :=
  if h : ∃ p > 0, ∀ x, f(x) = f(x + p) then classical.some h else 0

theorem find_smallest_period (f : ℝ → ℝ)
  (h : ∀ x, f(x + 6) + f(x - 6) = f(x)) : smallest_period f = 36 := by
  sorry

end find_smallest_period_l747_747303


namespace remainder_of_square_l747_747083

theorem remainder_of_square (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := 
by 
  sorry

end remainder_of_square_l747_747083


namespace divisors_of_2004_victors_l747_747851

theorem divisors_of_2004_victors (n k : ℕ) (h_factorization : n = 2^2 * 3 * 167) 
                                  (h_divisors : k = 2004) :
    let d := 2004 ^ 2004
    in ∃ count, count = 54 ∧ (count = ∑ d in (finset.filter (λ d, nat.divisors d = 2004) (nat.divisors d)).card) :=
by
  sorry

end divisors_of_2004_victors_l747_747851


namespace minimum_value_sqrt_inequality_l747_747944

theorem minimum_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (sqrt ((x^2 + y^2) * (4 * x^2 + y^2)) / (x * y)) ≥ 3 :=
sorry

end minimum_value_sqrt_inequality_l747_747944


namespace integral_of_x_minus_one_l747_747216

theorem integral_of_x_minus_one : ∫ x in 0..2, (x - 1) = 0 :=
  sorry

end integral_of_x_minus_one_l747_747216


namespace find_cd_l747_747597

theorem find_cd (c d : ℤ) (h1 : c ≠ 0) (h2 : d ≠ 0)
    (h3 : ∃ u v : ℤ, ∃ f : ℝ → ℝ, (f = (λ x, x^3 + c * x^2 + d * x + 15 * c)) ∧
        (∀ x, f x = (x - u) * (x - u) * (x - v))) :
    |c * d| = 840 := 
sorry

end find_cd_l747_747597


namespace prime_divisor_fourth_power_gt_cubic_l747_747718

-- Stating the problem in Lean 4
theorem prime_divisor_fourth_power_gt_cubic (n : ℕ) (h : ∀ (d : ℕ), d ∣ n → ¬ (n^2 ≤ d^4 ∧ d^4 ≤ n^3)) : 
  ∃ (p : ℕ), prime p ∧ p ∣ n ∧ p^4 > n^3 :=
sorry

end prime_divisor_fourth_power_gt_cubic_l747_747718


namespace sqrt_neg3_exists_iff_l747_747688

theorem sqrt_neg3_exists_iff (p : ℕ) [Fact p.prime] (hp : p > 3) :
  (∃ x : ℤ, ↑x ^ 2 = -3 [MOD p]) ↔ (∃ k : ℕ, p = 3 * k + 1) :=
sorry

end sqrt_neg3_exists_iff_l747_747688


namespace exists_infinitely_many_composite_numbers_of_the_form_l747_747101

theorem exists_infinitely_many_composite_numbers_of_the_form :
  ∃ᶠ n in at_top, ∃ k : ℕ, k ∣ (50^n + (50 * n + 1)^50) ∧ k > 1 := 
begin
  sorry
end

end exists_infinitely_many_composite_numbers_of_the_form_l747_747101


namespace total_clouds_counted_l747_747294

def clouds_counted (carson_clouds : ℕ) (brother_factor : ℕ) : ℕ :=
  carson_clouds + (carson_clouds * brother_factor)

theorem total_clouds_counted (carson_clouds brother_factor total_clouds : ℕ) 
  (h₁ : carson_clouds = 6) (h₂ : brother_factor = 3) (h₃ : total_clouds = 24) :
  clouds_counted carson_clouds brother_factor = total_clouds :=
by
  sorry

end total_clouds_counted_l747_747294


namespace average_coins_collected_l747_747056

-- Given conditions
variable (a1 : ℕ := 5) -- First day's coin collection
variable (d : ℕ := 6)  -- Common difference in coins collected each day
variable (n : ℕ := 7)  -- Number of days

-- Conclusion to prove
theorem average_coins_collected :
  let sequence := List.range n |>.map (λ i, a1 + i * d) in
  let total_coins := sequence.sum in
  (total_coins : ℚ) / n = 23 := 
by 
  sorry

end average_coins_collected_l747_747056


namespace second_smallest_odd_number_l747_747646

-- Define the conditions
def four_consecutive_odd_numbers_sum (n : ℕ) : Prop := 
  n % 2 = 1 ∧ (n + (n + 2) + (n + 4) + (n + 6) = 112)

-- State the theorem
theorem second_smallest_odd_number (n : ℕ) (h : four_consecutive_odd_numbers_sum n) : n + 2 = 27 :=
sorry

end second_smallest_odd_number_l747_747646


namespace minimum_value_l747_747947

noncomputable def expr (x y : ℝ) := (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y)

theorem minimum_value :
  (∀ x y : ℝ, 0 < x → 0 < y → expr x y ≥ (2 + 2 * Real.root 3 8)) :=
  sorry

end minimum_value_l747_747947


namespace solve_for_x_l747_747977

variable (x : ℝ)

theorem solve_for_x (h : 5 * x - 3 = 17) : x = 4 := sorry

end solve_for_x_l747_747977


namespace trumpet_cost_l747_747955

theorem trumpet_cost (net_spent : ℝ) (book_sold : ℝ) (trumpet_cost : ℝ) :
  net_spent = 139.32 → book_sold = 5.84 → trumpet_cost = net_spent + book_sold → trumpet_cost = 145.16 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3
  sorry

end trumpet_cost_l747_747955


namespace general_term_seq_sum_reciprocal_seq_sum_reciprocal_f_l747_747393

-- Problem 1: General term formula of the sequence {a_n}
theorem general_term_seq (a : ℕ → ℤ) (h₁ : a 1 = 2) (h₂ : a 2 = 2) (h₃ : ∀ n ≥ 2, a (n + 1) = a n + 2 * a (n - 1)) : 
  ∀ n, a n = (2 / 3) * (2^n - (-1)^n) := 
sorry

-- Problem 2: Sum of reciprocals of sequence < 3
theorem sum_reciprocal_seq (a : ℕ → ℤ) (h₁ : a 1 = 2) (h₂ : a 2 = 2) (h₃ : ∀ n ≥ 2, a (n + 1) = a n + 2 * a (n - 1)) :
  ∀ n ≥ 2, (finset.sum (finset.range n) (λ k, (1 / a (k + 1))) < 3) :=
sorry

-- Problem 3: Sum of reciprocals of function f(k) + 1 < 1/2
theorem sum_reciprocal_f (f : ℕ → ℤ) (h₁ : f 1 = 2) (h₂ : ∀ n, f (n + 1) = (f n)^2 + f n) : 
  ∀ n, finset.sum (finset.range n) (λ k, (1 / (f (k + 1) + 1))) < 1/2 :=
sorry

end general_term_seq_sum_reciprocal_seq_sum_reciprocal_f_l747_747393


namespace unique_integer_sequence_l747_747972

theorem unique_integer_sequence (a : ℕ → ℤ) :
  a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) →
  ∃! (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) :=
sorry

end unique_integer_sequence_l747_747972


namespace part1_part2_l747_747992

noncomputable def curve := {p : ℝ × ℝ | (p.1^2 + p.2) * (p.1 + p.2) = 0}
noncomputable def line (k b : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + b}

theorem part1 (k : ℝ) :
  (∃ⁿ (A B C : ℝ × ℝ), A ∈ curve ∧ A ∈ line k (1/16) ∧
     B ∈ curve ∧ B ∈ line k (1/16) ∧
     C ∈ curve ∧ C ∈ line k (1/16) ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C) →
  k ∈ (-∞, -17/16) ∪ (-17/16, -1) ∪ (-1, -1/2) ∪ (1/2, ∞) :=
sorry

theorem part2 (k : ℝ) :
  (∃ⁿ (A B C : ℝ × ℝ), A ∈ curve ∧ A ∈ line k 1 ∧
     B ∈ curve ∧ B ∈ line k 1 ∧
     C ∈ curve ∧ C ∈ line k 1 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
     dist A B = dist B C) →
  k = real.cbrt 2 + real.cbrt (1/2) :=
sorry

end part1_part2_l747_747992


namespace mean_median_comparison_l747_747829

-- Definitions for given conditions
def weights : list ℝ := sorry -- list of weights x_1, x_2, ..., x_50
def x : ℝ := (list.sum weights) / 50  -- average of weights
def y : ℝ := sorry -- the median of the given weights

-- New weight added to the list
def new_weights : list ℝ := 500 :: weights
def new_mean : ℝ := (list.sum new_weights) / 51

-- Statement to be proved
theorem mean_median_comparison :
  new_mean > x ∧ (y ≤ median new_weights ∨ median new_weights = y) :=
by
  sorry

end mean_median_comparison_l747_747829


namespace dice_sum_probability_l747_747443

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747443


namespace range_of_f_on_0_3_l747_747556

-- Let g be a periodic function with period 1
def periodic {α : Type*} (g : α → α) : Prop :=
∀ x, g (x + 1) = g x

-- Define function f
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
x + g x

-- Theorem statement
theorem range_of_f_on_0_3 (g : ℝ → ℝ) (h_periodic : periodic g)
  (h_range : ∀ x ∈ Icc 0 1, f g x ∈ Icc (-2 : ℝ) 5) :
  ∀ x ∈ Icc 0 3, f g x ∈ Icc (-2 : ℝ) 7 :=
sorry

end range_of_f_on_0_3_l747_747556


namespace problem_triangle_ABC_l747_747022

open Triangle

noncomputable def midpoint {A B : Point} (D : Point) := dist D A = dist D B ∧ collinear (A::B::D::[])
noncomputable def parallel {A B C D E F: Point} := ∃ (λ : ℝ), λ • (vector_between A B) = vector_between C D

theorem problem_triangle_ABC (A B C D E : Point) (DE BC AC AB : ℝ) 
  (h_triangle: triangle A B C)
  (h_midpoint: midpoint D A B)
  (h_parallel: parallel D E C B)
  (h_intersect: ∃ E, E ∈ line_through D ∨ E ∈ line_through (intersection_point AC))
  (h_de: DE = 4) :
  BC = 8 := 
begin
  sorry
end

end problem_triangle_ABC_l747_747022


namespace solution_set_empty_iff_l747_747644

def quadratic_no_solution (a b c : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)

theorem solution_set_empty_iff (a b c : ℝ) (h : quadratic_no_solution a b c) : a > 0 ∧ (b^2 - 4 * a * c ≤ 0) :=
sorry

end solution_set_empty_iff_l747_747644


namespace john_height_after_one_year_l747_747915

noncomputable def inches_to_meters (inches : ℝ) : ℝ :=
  inches * 0.0254

theorem john_height_after_one_year :
  let initial_height := 66
      growth_rate := 2
      first_3_months_growth := growth_rate * 3
      height_after_first_3_months := initial_height + first_3_months_growth
      a := 2.2
      r := 1.10
      n := 9
      sum_geometric_series := a * (1 - r ^ n) / (1 - r)
      final_height_inches := height_after_first_3_months + sum_geometric_series
      final_height_meters := inches_to_meters final_height_inches
  in abs (final_height_meters - 2.59) < 0.01 :=
by
  sorry

end john_height_after_one_year_l747_747915


namespace polynomial_root_product_def_l747_747598

theorem polynomial_root_product_def :
  (∀ (d e f : ℚ), (∀ (x : ℝ), x^3 + (d : ℝ) * x^2 + ↑e * x + ↑f = 0 → 
    (x = real.cos (2 * real.pi / 9) ∨ x = real.cos (4 * real.pi / 9) ∨ x = real.cos (8 * real.pi / 9))) →
    (d * e * f = 1 / 27)) :=
by
  sorry

end polynomial_root_product_def_l747_747598


namespace minimum_participants_l747_747636

theorem minimum_participants
  (correct_first : ℕ)
  (correct_second : ℕ)
  (correct_third : ℕ)
  (correct_fourth : ℕ)
  (H_first : correct_first = 90)
  (H_second : correct_second = 50)
  (H_third : correct_third = 40)
  (H_fourth : correct_fourth = 20)
  (H_max_two : ∀ p : ℕ, 1 ≤ p ∧ p ≤ correct_first + correct_second + correct_third + correct_fourth → p ≤ 2 * (correct_first + correct_second + correct_third + correct_fourth))
  : ∃ n : ℕ, (correct_first + correct_second + correct_third + correct_fourth) / 2 = 100 :=
by
  sorry

end minimum_participants_l747_747636


namespace cubic_roots_identity_l747_747967

theorem cubic_roots_identity 
  (x1 x2 x3 p q : ℝ) 
  (hq : ∀ x, x^3 + p * x + q = (x - x1) * (x - x2) * (x - x3))
  (h_sum : x1 + x2 + x3 = 0)
  (h_prod : x1 * x2 + x2 * x3 + x3 * x1 = p):
  x2^2 + x2 * x3 + x3^2 = -p ∧ x1^2 + x1 * x3 + x3^2 = -p ∧ x1^2 + x1 * x2 + x2^2 = -p :=
sorry

end cubic_roots_identity_l747_747967


namespace inequality_abc_l747_747929

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_abc_l747_747929


namespace arithmetic_sequence_problem_l747_747367

theorem arithmetic_sequence_problem (a₁ d S₁₀ : ℝ) (h1 : d < 0) (h2 : (a₁ + d) * (a₁ + 3 * d) = 12) 
  (h3 : (a₁ + d) + (a₁ + 3 * d) = 8) (h4 : S₁₀ = 10 * a₁ + 10 * (10 - 1) / 2 * d) : 
  true := sorry

end arithmetic_sequence_problem_l747_747367


namespace polyhedron_volume_calc_l747_747655

-- Define the conditions in Lean
def side_length : ℝ := 12
def cube_volume : ℝ := side_length ^ 3
def polyhedron_volume : ℝ := cube_volume / 2

-- Lean statement to prove the volume of the polyhedron
theorem polyhedron_volume_calc : polyhedron_volume = 864 :=
by
  sorry

end polyhedron_volume_calc_l747_747655


namespace percentage_of_other_sales_l747_747124

theorem percentage_of_other_sales :
  let pensPercentage := 20
  let pencilsPercentage := 15
  let notebooksPercentage := 30
  let totalPercentage := 100
  totalPercentage - (pensPercentage + pencilsPercentage + notebooksPercentage) = 35 :=
by
  sorry

end percentage_of_other_sales_l747_747124


namespace area_union_square_circle_l747_747733

theorem area_union_square_circle (side_length: ℝ) (radius: ℝ) (h1: side_length = 12) (h2: radius = 12):
  let square_area := side_length^2
  let circle_area := π * radius^2
  let union_area := if radius >= side_length / real.sqrt 2 then circle_area
                    else square_area + circle_area - overlapping_area radius square_area
  union_area = 144 * π := 
by
  -- Let the side length of the square and the radius of the circle be given as 12.
  have side_length_12: side_length = 12 := h1
  have radius_12: radius = 12 := h2
  -- Calculate the area of the square.
  have square_area_144: square_area = 144 := 
    by rw [side_length_12, pow_two]; exact rfl
  -- Calculate the area of the circle.
  have circle_area_144pi: circle_area = 144 * π := 
    by rw [radius_12, pow_two, mul_comm]; exact rfl
  -- The circle completely covers the square, so the area of the union is the area of the circle.
  rw [if_pos] -- because radius >= side_length / real.sqrt 2, 12 >= 12 / sqrt 2 is obviously true
  exact circle_area_144pi
  -- skip proof of overlapping_area function because we don't need it in this case
  sorry

end area_union_square_circle_l747_747733


namespace complex_division_result_l747_747131

theorem complex_division_result : (1 - complex.I) / complex.I = -1 - complex.I :=
by sorry

end complex_division_result_l747_747131


namespace problem_statement_l747_747383

-- Definitions corresponding to conditions
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 / 2 * t, 3 + sqrt 3 / 2 * t)

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * (cos θ)^2 = 4 * sin θ

-- Translate the problem conditions and proof goals into a Lean 4 statement
theorem problem_statement : 
  (∀ t, ∃ x y, parametric_line t = (x, y) ∧ y = 3 + sqrt 3 * x) ∧
  (∀ ρ θ, polar_curve ρ θ → ∀ x y, (x = ρ * cos θ ∧ y = ρ * sin θ) → x^2 = 4 * y) ∧
  (let A : ℝ × ℝ := (2 * sqrt 3, 9) in 
   let P : ℝ × ℝ := (2 * sqrt 3, 6) in 
   |dist A P| = 3) :=
by
  sorry

end problem_statement_l747_747383


namespace dice_sum_prob_10_l747_747466

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747466


namespace probability_of_yellow_face_l747_747599

theorem probability_of_yellow_face :
  let total_faces : ℕ := 10
  let yellow_faces : ℕ := 4
  (yellow_faces : ℚ) / (total_faces : ℚ) = 2 / 5 :=
by
  sorry

end probability_of_yellow_face_l747_747599


namespace not_all_monotonic_have_extremum_l747_747737

theorem not_all_monotonic_have_extremum :
  ∃ (f : ℝ → ℝ), monotonic f ∧ ¬(∀ (a b : ℝ), a < b → f.has_extremum_on (set.Ioo a b)) :=
by sorry

end not_all_monotonic_have_extremum_l747_747737


namespace find_t_l747_747147

theorem find_t:
  (∃ t, (∀ (x y: ℝ), (x = 2 ∧ y = 8) ∨ (x = 4 ∧ y = 14) ∨ (x = 6 ∧ y = 20) → 
                (∀ (m b: ℝ), y = m * x + b) ∧ 
                (∀ (m b: ℝ), y = 3 * x + b ∧ b = 2 ∧ (t = 3 * 50 + 2) ∧ t = 152))) := by
  sorry

end find_t_l747_747147


namespace tan_triple_angle_l747_747005

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l747_747005


namespace min_hyperplanes_cover_S_l747_747930

def S (n k : ℕ) : set (fin k → ℕ) :=
  {x | (∀ i, x i ∈ fin (n + 1)) ∧ (0 < (finset.univ.sum (λ i, x i)))}

theorem min_hyperplanes_cover_S 
  (n k : ℕ) (h : 0 < n) : 
  ∃ m, (∀ H : set (set (fin k → ℕ)), (S n k ⊆ ⋃ h ∈ H, h) → (∀ p ∈ H, (0, 0, ..., 0) ∉ p) → card H >= m) := 
  sorry

end min_hyperplanes_cover_S_l747_747930


namespace distance_from_point_to_focus_l747_747362

noncomputable def distance_to_focus {P : ℝ × ℝ} (hP : P.2^2 = 4 * P.1) (dist_to_line : ℝ) 
  (h_dist_to_line : dist_to_line = abs (P.1 + 2)) : ℝ :=
  if h : dist_to_line = 5 then 4 else sorry

theorem distance_from_point_to_focus {P : ℝ × ℝ} (hP : P.2^2 = 4 * P.1)
  (dist_to_line : ℝ) (h_dist_to_line : dist_to_line = abs (P.1 + 2)) :
  dist_to_line = 5 → distance_to_focus hP dist_to_line h_dist_to_line = 4 :=
by { intros h, simp [distance_to_focus, h] }

end distance_from_point_to_focus_l747_747362


namespace find_c_l747_747021

theorem find_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y / 20) + (c * y / 10) = 0.6 * y) : c = 3 :=
by 
  -- Skipping the proof
  sorry

end find_c_l747_747021


namespace lambda_range_monotonic_sequence_l747_747364

theorem lambda_range_monotonic_sequence
  (a : ℕ → ℝ)
  (λ : ℝ)
  (h : ∀ n : ℕ, a n = n^2 + λ * n)
  (strictly_increasing : ∀ n : ℕ, a n < a (n+1)) :
  λ > -3 :=
sorry

end lambda_range_monotonic_sequence_l747_747364


namespace length_of_chord_EF_l747_747037

-- Given conditions
structure GeometrySetup :=
  (A B C D G E F O N P : Point)
  (r : ℝ)
  (radius_eq_20 : r = 20)
  (distAB : Dist A B = 2 * r)
  (distBC : Dist B C = 2 * r)
  (distCD : Dist C D = 2 * r)
  (AG_tangent_to_P_at_G : Tangent AG P G)
  (EF_chord_O_intersection : Chord EF O)

-- Main theorem
theorem length_of_chord_EF {setup : GeometrySetup} :
  ChordLength EF setup.r = 40 * Real.sqrt 11 := 
  sorry

end length_of_chord_EF_l747_747037


namespace max_mow_time_l747_747567

-- Define the conditions
def timeToMow (x : ℕ) : Prop := 
  let timeToFertilize := 2 * x
  x + timeToFertilize = 120

-- State the theorem
theorem max_mow_time (x : ℕ) (h : timeToMow x) : x = 40 := by
  sorry

end max_mow_time_l747_747567


namespace probability_of_odd_ball_probability_of_both_balls_less_than_6_l747_747516

-- Definitions of conditions
def balls : List ℕ := [3, 4, 5, 6]

def number_of_balls : ℕ := balls.length

def odd_balls : List ℕ := balls.filter (λ n, n % 2 = 1)

def number_of_odd_balls : ℕ := odd_balls.length

def all_pairs : List (ℕ × ℕ) :=
  (balls.product balls).filter (λ pair, pair.1 ≠ pair.2)

def pairs_less_than_6 : List (ℕ × ℕ) :=
  all_pairs.filter (λ pair, pair.1 < 6 ∧ pair.2 < 6)

def number_of_favorable_pairs : ℕ := pairs_less_than_6.length

def number_of_total_pairs : ℕ := all_pairs.length

-- Assertions to be proved
theorem probability_of_odd_ball :
  (number_of_odd_balls : ℚ) / (number_of_balls : ℚ) = 1 / 2 :=
by sorry

theorem probability_of_both_balls_less_than_6 :
  (number_of_favorable_pairs : ℚ) / (number_of_total_pairs : ℚ) = 1 / 2 :=
by sorry

end probability_of_odd_ball_probability_of_both_balls_less_than_6_l747_747516


namespace merry_go_round_cost_per_child_l747_747710

-- Definitions
def num_children := 5
def ferris_wheel_cost_per_child := 5
def num_children_on_ferris_wheel := 3
def ice_cream_cost_per_cone := 8
def ice_cream_cones_per_child := 2
def total_spent := 110

-- Totals
def ferris_wheel_total_cost := num_children_on_ferris_wheel * ferris_wheel_cost_per_child
def ice_cream_total_cost := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone
def merry_go_round_total_cost := total_spent - ferris_wheel_total_cost - ice_cream_total_cost

-- Final proof statement
theorem merry_go_round_cost_per_child : 
  merry_go_round_total_cost / num_children = 3 :=
by
  -- We skip the actual proof here
  sorry

end merry_go_round_cost_per_child_l747_747710


namespace tetrahedron_inequality_tetrahedron_equality_condition_l747_747885

variables {A B C D : Type*}
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]
variables (AB BC CA AD BD CD : ℝ)
variables (angle_BDC : angle B D C = π / 2)
variables (S : affine_subspace ℝ (triangle A B C))

-- Given the foot of the perpendicular from D to plane ABC is the orthocenter of ∆ ABC
def orthocenter_S (D : Type*) (triangle_ABC : affine_subspace ℝ (triangle A B C)) : Prop :=
orthogonal_projection (affine_span ℝ (set.range [A, B, C])) D = S

-- Prove the inequality
theorem tetrahedron_inequality
  (h_orthocenter_S : orthocenter_S D (triangle A B C))
  (h_angle_BDC : angle B D C = π / 2)
  (h_lengths : AB = dist A B ∧ BC = dist B C∧ CA = dist C A
               ∧ AD = dist A D ∧ BD = dist B D ∧ CD = dist C D) :
  (AB + BC + CA)^2 ≤ 6 * (AD^2 + BD^2 + CD^2) :=
sorry

-- Show equality conditions
theorem tetrahedron_equality_condition
  (h_orthocenter_S : orthocenter_S D (triangle A B C))
  (h_angle_BDC : angle B D C = π / 2)
  (h_lengths : AB = dist A B ∧ BC = dist B C ∧ CA = dist C A
               ∧ AD = dist A D ∧ BD = dist B D ∧ CD = dist C D) :
  (AB + BC + CA)^2 = 6 * (AD^2 + BD^2 + CD^2) ↔ AB = BC ∧ BC = CA :=
sorry

end tetrahedron_inequality_tetrahedron_equality_condition_l747_747885


namespace january31_2014_is_friday_l747_747317

-- Define the days of the week.
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
deriving DecidableEq, Inhabited, Repr

-- Define a function to calculate the day of the week after a number of days.
def addDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  let days := [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]
  days[(days.indexOf start + n) % 7]

-- Given conditions
def dec21_2013 := DayOfWeek.Saturday

-- Theorem to prove
theorem january31_2014_is_friday : addDays dec21_2013 41 = DayOfWeek.Friday := by
  sorry

end january31_2014_is_friday_l747_747317


namespace dice_sum_probability_l747_747446

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747446


namespace analytical_expression_of_f_increasing_interval_decreasing_interval_l747_747349

noncomputable def ω : ℝ := 1

def vec_a (x : ℝ) : ℝ × ℝ := (2 * Real.sin (ω * x), Real.cos (ω * x) + Real.sin (ω * x))
def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.cos (ω * x) - Real.sin (ω * x))

def f (x : ℝ) : ℝ := 
  let a := vec_a x
  let b := vec_b x
  a.1 * b.1 + a.2 * b.2

theorem analytical_expression_of_f (x : ℝ) :
  f x = Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) := 
sorry  

theorem increasing_interval (x : ℝ) :
  0 ≤ x ∧ x ≤ Real.pi / 8 → f x > f (x - ε) ∀ ε > 0 := 
sorry

theorem decreasing_interval (x : ℝ) :
  Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 2 → f x < f (x + ε) ∀ ε > 0 :=
sorry

end analytical_expression_of_f_increasing_interval_decreasing_interval_l747_747349


namespace angle_sum_90_l747_747620

-- Define the setup with conditions
section proof_problem

variables {x b c x1 x2 : ℝ} (C X1 X2 O : Point ℝ)

-- Given conditions
def y_eq : ℝ → ℝ := λ x, x + b * real.sqrt x + c
def c_pos : c > 0 := sorry
def C_eq : C = (0, c) := sorry
def X1_eq : X1 = (x1, 0) := sorry
def X2_eq : X2 = (x2, 0) := sorry
def roots_eq : x1 + b * real.sqrt x1 + c = 0 ∧ x2 + b * real.sqrt x2 + c = 0 := sorry
def x1_x2_pos : x1 > 0 ∧ x2 > 0 := sorry

-- Prove the required angles sum to 90 degrees
theorem angle_sum_90 : 
  ∠ (C, X1, O) + ∠ (C, X2, O) = 90 := sorry

end proof_problem

end angle_sum_90_l747_747620


namespace PHQ_collinear_l747_747075

variables {A B C P Q H : Type*}

-- Define the triangle ABC and its orthocenter H
variable (triangle_ABC : Triangle A B C)
variable (orthocenter_H : Orthocenter triangle_ABC = H)

-- Define the circle with diameter BC and the tangents from A (giving points P and Q)
variable (circle_BC : Circle (midpoint B C) (dist B C / 2))
variable (tangent_AP : Tangent A circle_BC = P)
variable (tangent_AQ : Tangent A circle_BC = Q)

-- Define the collinearity of points P, H, and Q
def collinear_PHQ : Prop :=
  Collinear P H Q

-- The main theorem to prove
theorem PHQ_collinear (triangle_ABC : Triangle A B C)
                       (orthocenter_H : Orthocenter triangle_ABC = H)
                       (circle_BC : Circle (midpoint B C) (dist B C / 2))
                       (tangent_AP : Tangent A circle_BC = P)
                       (tangent_AQ : Tangent A circle_BC = Q) :
  collinear_PHQ :=
by sorry

end PHQ_collinear_l747_747075


namespace tan3theta_l747_747000

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l747_747000


namespace xiaohua_walks_to_east_l747_747682

theorem xiaohua_walks_to_east (home_north : Prop) (traffic_rules : Prop) :
  home_north ∧ traffic_rules → True := 
begin
  intros h,
  sorry
end

end xiaohua_walks_to_east_l747_747682


namespace find_length_DF_l747_747040

open Real
open Set

variables {A B C D E F G O : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G] [MetricSpace O]

def right_triangle (A B C: Type*) : Prop :=
  ∃ (a b c : Real), a^2 + b^2 = c^2 ∧ a = 21 ∧ b = 28 ∧ c = 35

def square (A B D E : Type*) : Prop :=
  ∃ (s : Real), s = 35

def distance (x y: Type*): Real := sorry

theorem find_length_DF (A B C D E F G O : Type*) [right_triangle A B 35] [square A B D E] 
    (angle_bisector_ACB : Type*): 
    distance D F = 15 :=
by
  -- Proof steps (details)
  sorry

end find_length_DF_l747_747040


namespace candy_mixture_solution_l747_747266

theorem candy_mixture_solution :
  ∃ x y : ℝ, 18 * x + 10 * y = 1500 ∧ x + y = 100 ∧ x = 62.5 ∧ y = 37.5 := by
  sorry

end candy_mixture_solution_l747_747266


namespace caroline_sequence_final_step_l747_747981

theorem caroline_sequence_final_step :
  let init := 10^7 in
  let seq := (alternate_operations 14 (divide_by 5) (multiply_by 3) init) in
  seq = 2^7 * 3^7 :=
sorry

end caroline_sequence_final_step_l747_747981


namespace concurrency_or_parallel_l747_747098

variables {A B C D E F G H P Q : Type*}
variables [Quad ABCD : Type] [Points E F G H : Type]
variables [Line EH : Type] [Line BD : Type] [Line FG : Type]
variables [Concurrent EH BD FG : Prop]

-- Statements to assert points lying on sides
axiom points_on_sides : Points E F G H → 
  (OnSide E A B ∧ OnSide F B C ∧ OnSide G C D ∧ OnSide H D A)

-- Statement lines are concurrent
axiom lines_concurrent : Concurrent EH BD FG

-- The final statement to prove the condition
theorem concurrency_or_parallel :
  (Concurrent (Line_through E F) (Line_through A C) (Line_through H G)) 
  ∨ (Parallel (Line_through E F) (Line_through A C) (Line_through H G)) := 
sorry

end concurrency_or_parallel_l747_747098


namespace multiplicative_inverse_modulo_l747_747081

theorem multiplicative_inverse_modulo {n : ℕ} (h : 0 < n) :
  ∃ (k : ℕ), k ∈ {8, 6, 5, 7, 9} ∧ (5^(2*n) + 4) * k % 11 = 1 :=
by
  sorry

end multiplicative_inverse_modulo_l747_747081


namespace annual_increase_of_chickens_l747_747954

theorem annual_increase_of_chickens 
  (chickens_now : ℕ)
  (chickens_after_9_years : ℕ)
  (years : ℕ)
  (chickens_now_eq : chickens_now = 550)
  (chickens_after_9_years_eq : chickens_after_9_years = 1900)
  (years_eq : years = 9)
  : ((chickens_after_9_years - chickens_now) / years) = 150 :=
by
  sorry

end annual_increase_of_chickens_l747_747954


namespace rectangle_bounds_product_l747_747145

theorem rectangle_bounds_product (b : ℝ) :
  (∃ b, y = 3 ∧ y = 7 ∧ x = -1 ∧ (x = b) 
   → (b = 3 ∨ b = -5) 
    ∧ (3 * -5 = -15)) :=
sorry

end rectangle_bounds_product_l747_747145


namespace cylinder_height_relationship_l747_747169

variable (r₁ h₁ r₂ h₂ : ℝ)

def radius_relationship := r₂ = 1.1 * r₁

def volume_equal := π * r₁^2 * h₁ = π * r₂^2 * h₂

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) 
     (h_radius : radius_relationship r₁ r₂) 
     (h_volume : volume_equal r₁ h₁ r₂ h₂) : 
     h₁ = 1.21 * h₂ :=
by
  unfold radius_relationship at h_radius
  unfold volume_equal at h_volume
  sorry

end cylinder_height_relationship_l747_747169


namespace inradius_of_isosceles_triangle_l747_747986

noncomputable def inradius (S : ℝ) (α : ℝ) : ℝ :=
  sqrt (S * tan α) * tan (π/4 - α/2)

theorem inradius_of_isosceles_triangle (S α : ℝ) :
  let r := inradius S α
  r = sqrt (S * tan α) * tan (π/4 - α/2) :=
by 
  sorry

end inradius_of_isosceles_triangle_l747_747986


namespace probability_sum_10_l747_747488

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747488


namespace dice_sum_10_probability_l747_747427

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747427


namespace equal_division_of_potatoes_l747_747910

theorem equal_division_of_potatoes (p n : ℕ) (h_p : p = 24) (h_n : n = 3) : p / n = 8 :=
by
  rw [h_p, h_n]
  norm_num
  sorry

end equal_division_of_potatoes_l747_747910


namespace bacterium_diameter_in_nanometers_l747_747976

theorem bacterium_diameter_in_nanometers (d_meters : ℝ) (d_nanometers : ℝ) (h1 : d_meters = 0.00000285) (h2 : 1 = 10^9 * 10^(-9)) : 
  d_nanometers = 2850 :=
by
  -- Assume h1: d_meters = 0.00000285 and h2: 1 = 10^9 * 10^(-9)
  have h3 : d_nanometers = d_meters * 10^9,
  { sorry },
  -- Hence d_nanometers = 0.00000285 * 10^9
  calc
  d_nanometers = 0.00000285 * 10^9 : by rw [h1, h3]
              ... = 2850             : sorry

end bacterium_diameter_in_nanometers_l747_747976


namespace geometric_series_sum_l747_747289

theorem geometric_series_sum : (∑ i in Finset.range 64, 2^i) = 2^64 - 1 :=
by
  sorry

end geometric_series_sum_l747_747289


namespace total_poles_needed_l747_747236

theorem total_poles_needed (longer_side_poles : ℕ) (shorter_side_poles : ℕ) (internal_fence_poles : ℕ) :
  longer_side_poles = 35 → 
  shorter_side_poles = 27 → 
  internal_fence_poles = (shorter_side_poles - 1) → 
  ((longer_side_poles * 2) + (shorter_side_poles * 2) - 4 + internal_fence_poles) = 146 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_poles_needed_l747_747236


namespace roots_quadratic_solution_l747_747814

theorem roots_quadratic_solution (α β : ℝ) (hα : α^2 - 3*α - 2 = 0) (hβ : β^2 - 3*β - 2 = 0) :
  3*α^3 + 8*β^4 = 1229 := by
  sorry

end roots_quadratic_solution_l747_747814


namespace ratio_area_triangle_l747_747529

/-- In a plane, given a triangle ABC and a point P such that 
    PA + PB + PC = AB, prove that the ratio of the area of triangle PAB to the area 
    of triangle ABC is 1/3. -/
theorem ratio_area_triangle (A B C P : ℝ × ℝ)
  (h1 : vector_add A P + vector_add B P + vector_add C P = vector_add A B) :
  area_of_triangle PAB / area_of_triangle ABC = 1/3 :=
sorry

end ratio_area_triangle_l747_747529


namespace percentage_of_apples_is_correct_l747_747128

-- Define the number of responses for each fruit
def apples : ℕ := 80
def bananas : ℕ := 90
def cherries : ℕ := 50
def oranges : ℕ := 40
def grapes : ℕ := 40

-- Define the total number of responses
def total_responses : ℕ := apples + bananas + cherries + oranges + grapes

-- Define the percentage of respondents who preferred apples
noncomputable def percentage_of_apples : ℝ := (apples / total_responses.to_real) * 100

-- Prove that the percentage of respondents who preferred apples is 26.67%
theorem percentage_of_apples_is_correct : percentage_of_apples = 26.67 := 
by 
  sorry

end percentage_of_apples_is_correct_l747_747128


namespace numEquilateralTrianglesInRegularNineSidedPolygon_l747_747838

-- Define the nine-sided regular polygon
structure RegularNineSidedPolygon (P : Type) :=
(vertices : Fin 9 → P)

-- Define the function to count distinct equilateral triangles
noncomputable def countDistinctEquilateralTriangles
  (P : Type) [EuclideanGeometry P] (polygon : RegularNineSidedPolygon P) : Nat := 66

-- Define the theorem to prove the count of equilateral triangles is correct
theorem numEquilateralTrianglesInRegularNineSidedPolygon
  (P : Type) [EuclideanGeometry P] (polygon : RegularNineSidedPolygon P) :
  countDistinctEquilateralTriangles P polygon = 66 := 
sorry

end numEquilateralTrianglesInRegularNineSidedPolygon_l747_747838


namespace isosceles_triangle_angle_l747_747550

/-- Let ABC be an isosceles triangle with A as the vertex angle and D the foot of the internal bisector from B.
    It is given that BC = AD + DB. 
    Prove that the measure of ∠BAC is 100 degrees. -/
theorem isosceles_triangle_angle (A B C D : Type) (triangle_ABC_isosceles : ∀ (A B C : Type), ∃ (D : Type),
  internal_bisector_B_D : ∀ (B D: Type), BC = AD + DB) : measure_angle_BAC (angle BAC) = 100 :=
sorry

end isosceles_triangle_angle_l747_747550


namespace robert_bike_time_proof_l747_747267

noncomputable def robert_bike_ride_time (highway_length_miles : ℝ) (highway_width_feet : ℝ) (bike_speed_mph : ℝ) : ℝ :=
  have highway_length_feet : ℝ := highway_length_miles * 5280
  have radius : ℝ := highway_width_feet / 2
  have path_cycle_length : ℝ := 2 * highway_width_feet + highway_width_feet
  have number_of_cycles : ℝ := highway_length_feet / path_cycle_length
  have total_distance_ft : ℝ := number_of_cycles * (2 * (radius * π) + (2 * radius * π))
  have total_distance_miles : ℝ := total_distance_ft / 5280
  total_distance_miles / bike_speed_mph

theorem robert_bike_time_proof : robert_bike_ride_time 2 40 5 = (4 * π / 15) := by
  sorry

end robert_bike_time_proof_l747_747267


namespace greatest_m_div_36_and_7_l747_747240

def reverse_digits (m : ℕ) : ℕ :=
  let d1 := (m / 1000) % 10
  let d2 := (m / 100) % 10
  let d3 := (m / 10) % 10
  let d4 := m % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_m_div_36_and_7
  (m : ℕ) (n : ℕ := reverse_digits m)
  (h1 : 1000 ≤ m ∧ m < 10000)
  (h2 : 1000 ≤ n ∧ n < 10000)
  (h3 : 36 ∣ m ∧ 36 ∣ n)
  (h4 : 7 ∣ m) :
  m = 9828 := 
sorry

end greatest_m_div_36_and_7_l747_747240


namespace jane_exercises_40_hours_l747_747050

-- Define the conditions
def hours_per_day : ℝ := 1
def days_per_week : ℝ := 5
def weeks : ℝ := 8

-- Define total_hours using the conditions
def total_hours : ℝ := (hours_per_day * days_per_week) * weeks

-- The theorem stating the result
theorem jane_exercises_40_hours :
  total_hours = 40 := by
  sorry

end jane_exercises_40_hours_l747_747050


namespace expression_value_l747_747074

noncomputable def hyperbola : Set (ℝ × ℝ) :=
  { p | p.1 ^ 2 / 4 - p.2 ^ 2 / 5 = 1 }

variables {F1 F2 P I Q : ℝ × ℝ}
variable (h_Foci_F1_F2 : ∅) -- Placeholder for the condition defining F1 and F2 as foci
variable (h_P_on_hyperbola : P ∈ hyperbola) -- P is on the hyperbola
variable {h_excenter : I = excenter (triangle P F1 F2)} -- I is the excenter of triangle P F1 F2
variable {h_PI_intersects_Q : line_through P I ∩ line_through (0,0) (1,0) = Q} -- Line PI intersects x-axis at Q

theorem expression_value :
  ∃ (P I Q : ℝ × ℝ), 
  P ∈ hyperbola ∧ 
  (line_through P I ∩ line_through (0,0) (1,0) = Q) ∧ 
  I = excenter (triangle P F1 F2) ∧ 
  (|P Q| / |P I| + |F1 Q| / |F1 P|) = 4 := 
sorry

end expression_value_l747_747074


namespace cheaper_joint_work_l747_747578

theorem cheaper_joint_work (r L P : ℝ) (hr_pos : 0 < r) (hL_pos : 0 < L) (hP_pos : 0 < P) : 
  (2 * P * L) / (3 * r) < (3 * P * L) / (4 * r) :=
by
  sorry

end cheaper_joint_work_l747_747578


namespace find_n_values_l747_747918

-- We need to state the problem conditions first
def points_form_same_distance_sequence (P : ℕ → ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ seq : Fin n → ℕ, 
    (∀ j : ℕ, 1 ≤ j ∧ j ≤ n → seq (⟨j - 1, by linarith⟩) = P i j)
    ∧ (∀ k l : ℕ, 1 ≤ k ∧ k ≤ n → 1 ≤ l ∧ l ≤ n → (P i k ≤ P i l) = (P j k ≤ P j l)))

-- The mathematical proof problem: 
theorem find_n_values (n : ℕ) (P : ℕ → ℕ → ℕ) :
  (1 ≤ n ∧ points_form_same_distance_sequence P n) → 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

end find_n_values_l747_747918


namespace scientific_notation_30000_l747_747984

theorem scientific_notation_30000 :
  ∃ a n, 30000 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
begin
  use [3, 4],
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end scientific_notation_30000_l747_747984


namespace prove_fees_l747_747711

-- Defining the conditions as hypotheses
def flat_fee_and_regular_fee (f n : ℝ) : Prop :=
  f + 3 * n = 180 ∧ f + 6 * n = 332

-- Stating what we need to prove given the conditions
theorem prove_fees :
  ∃ (f n : ℝ), flat_fee_and_regular_fee f n ∧ f = 28 ∧ n = 50.67 :=
by
  -- Hypotheses
  let h1 : 28 + 3 * 50.67 = 180 := sorry
  let h2 : 28 + 6 * 50.67 = 332 := sorry
  -- Conclusion
  exact ⟨28, 50.67, ⟨h1, h2⟩, rfl, rfl⟩

end prove_fees_l747_747711


namespace greatest_integer_of_2_7_l747_747288

-- Theorem: the greatest integer function applied to 2.7 equals 2
theorem greatest_integer_of_2_7 : (⌊2.7⌋ : ℤ) = 2 :=
sorry

end greatest_integer_of_2_7_l747_747288


namespace minimum_number_of_participants_l747_747634

theorem minimum_number_of_participants (a1 a2 a3 a4 : ℕ) (h1 : a1 = 90) (h2 : a2 = 50) (h3 : a3 = 40) (h4 : a4 = 20) 
  (h5 : ∀ (n : ℕ), n * 2 ≥ a1 + a2 + a3 + a4) : ∃ n, (n ≥ 100) :=
by 
  use 100
  sorry

end minimum_number_of_participants_l747_747634


namespace dice_sum_prob_10_l747_747471

/-- A standard six-faced die has faces numbered 1 through 6 --/
def is_standard_die_face (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

/-- The probability of the sum of the numbers rolled on three standard six-faced dice being 10 --/
theorem dice_sum_prob_10 : 
  ∃ (count_10 : ℕ), ∃ (total : ℕ), 
  (total = 6^3) ∧ 
  (count_10 = 27) ∧ 
  (count_10.toRat / total.toRat = 27 / 216) := 
by 
  sorry

end dice_sum_prob_10_l747_747471


namespace minimum_value_abs_sum_l747_747343

theorem minimum_value_abs_sum (x y : ℝ) : ∃ z, (z = 3) ∧ (∀ x y, | x - 1 | + | x | + | y - 1 | + | y + 1 | ≥ z) :=
by
  use 3
  split
  { refl }
  sorry

end minimum_value_abs_sum_l747_747343


namespace part_one_part_two_l747_747790

theorem part_one (a b : ℝ) (h : a ≠ 0) : |a + b| + |a - b| ≥ 2 * |a| :=
by sorry

theorem part_two (x : ℝ) : |x - 1| + |x - 2| ≤ 2 ↔ (1 / 2 : ℝ) ≤ x ∧ x ≤ (5 / 2 : ℝ) :=
by sorry

end part_one_part_two_l747_747790


namespace contradiction_proof_l747_747173

theorem contradiction_proof (a b c d : ℝ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 1) (h4 : d = 1) (h5 : a * c + b * d > 1) : ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_proof_l747_747173


namespace mickey_horses_per_week_l747_747311

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l747_747311


namespace coefficient_of_x3_in_expansion_l747_747825

theorem coefficient_of_x3_in_expansion :
  (2:ℤ)^n = 64 → 
  n = 6 → 
  (∃ (r : ℤ), r = -160 ∧ 
    ∀ (x : ℤ), 
      let expr := (2 * (x^2) - 1 * (x^(-1)))^n in
      extract_coefficient expr x^3 = r) :=
begin
  intros h1 h2,
  sorry
end

end coefficient_of_x3_in_expansion_l747_747825


namespace min_attack_pairs_l747_747889

open Fin

def attack_pairs (rooks: Finset (Fin 8 × Fin 8)) : ℕ :=
  let row_counts := (List.range 8).map (λ i, (rooks.filter (λ r, r.1 = i)).card)
  let col_counts := (List.range 8).map (λ j, (rooks.filter (λ r, r.2 = j)).card)
  let row_pairs := row_counts.map (λ n, n * (n - 1) / 2)
  let col_pairs := col_counts.map (λ n, n * (n - 1) / 2)
  row_pairs.sum + col_pairs.sum

theorem min_attack_pairs (rooks : Finset (Fin 8 × Fin 8)) (h : rooks.card = 16) : 
  attack_pairs rooks = 16 :=
  sorry

end min_attack_pairs_l747_747889


namespace quadratic_point_inequality_l747_747585

theorem quadratic_point_inequality 
  (m y1 y2 : ℝ)
  (hA : y1 = (m - 1)^2)
  (hB : y2 = (m + 1 - 1)^2)
  (hy1_lt_y2 : y1 < y2) :
  m > 1 / 2 :=
by 
  sorry

end quadratic_point_inequality_l747_747585


namespace alice_final_distance_from_origin_l747_747250

-- Definitions of the conditions
def regular_hexagon_side_length : ℝ := 3
def total_distance_walked : ℝ := 10

-- Proposition statement of the problem
theorem alice_final_distance_from_origin :
  ∀ (hexagon_side_length : ℝ) (distance_walked : ℝ),
  hexagon_side_length = 3 → 
  distance_walked = 10 → 
  √(3^2 + (3 √3 - 0)^2 - 2 * 3 * 3 √3 * cos (120 * π / 180)) = 
  √19 :=
by
  intros hexagon_side_length distance_walked h_side_length h_distance_walked
  rw [h_side_length, h_distance_walked]
  sorry

end alice_final_distance_from_origin_l747_747250


namespace actual_profit_percentage_l747_747742

theorem actual_profit_percentage (CP : ℝ) (hCP : CP > 0) :
  let MP := CP * 1.4,
      SP := MP * 0.75,
      P := SP - CP,
      PP := (P / CP) * 100 in
  PP = 5 :=
by
  sorry

end actual_profit_percentage_l747_747742


namespace no_solution_range_of_a_l747_747016

theorem no_solution_range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) → a ≤ 8 :=
by
  sorry

end no_solution_range_of_a_l747_747016


namespace suma_work_rate_l747_747200

theorem suma_work_rate (W : ℕ) : 
  (∀ W, (W / 6) + (W / S) = W / 4) → S = 24 :=
by
  intro h
  -- detailed proof would actually go here
  sorry

end suma_work_rate_l747_747200


namespace remainder_of_k_l747_747546

theorem remainder_of_k (k : ℕ) :
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 5) + (1 / 6) + (1 / 7) + (1 / 8) + (1 / 9) +
  (1 / 10) + (1 / 11) + (1 / 12) + (1 / 13)) * 13.factorial = k →
  k % 7 = 0 :=
by 
  sorry

end remainder_of_k_l747_747546


namespace minimum_value_l747_747946

noncomputable def expr (x y : ℝ) := (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y)

theorem minimum_value :
  (∀ x y : ℝ, 0 < x → 0 < y → expr x y ≥ (2 + 2 * Real.root 3 8)) :=
  sorry

end minimum_value_l747_747946


namespace large_room_capacity_l747_747735

noncomputable def number_of_people_in_large_room (total_people small_room_capacity rented_small_rooms : ℕ) : ℕ :=
  let remaining_people := total_people - (small_room_capacity * rented_small_rooms)
  let factors := [1, 2, 3, 4, 6, 8, 12, 24]
  factors.reverse.find (λ x => remaining_people % x = 0)

theorem large_room_capacity (total_people : ℕ) (small_room_capacity : ℕ) (rented_small_rooms : ℕ) :
  total_people = 26 →
  small_room_capacity = 2 →
  rented_small_rooms = 1 →
  number_of_people_in_large_room total_people small_room_capacity rented_small_rooms = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [number_of_people_in_large_room]
  sorry

end large_room_capacity_l747_747735


namespace batch_size_l747_747277

theorem batch_size (T : ℕ) :
  let A_contrib := 0.20
  let B_contrib := 0.40
  let reduction := 60
  (A_contrib * T + B_contrib * (T - reduction) = B_contrib * T) → 
  T = 1150 :=
by
  sorry

end batch_size_l747_747277


namespace dice_sum_10_probability_l747_747423

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747423


namespace infinite_redistributions_possible_infinite_redistributions_impossible_l747_747558

-- Define the conditions
variable (n : ℕ)
variable (books : Fin n → ℕ)
variable (neighbors : Fin n → Fin n)
variable [decLt : DecidableRel (<)]

-- Define the conditions of part (i)
def can_redistribute (child : Fin n) : Prop :=
  books child ≥ 2 ∧ books child > books (neighbors child) + books (neighbors (neighbors child))

-- Proposition for part (i)
theorem infinite_redistributions_possible (h : n ≥ 3) : 
  ∃ (books : Fin n → ℕ) (redistribute : ∀ (child : Fin n), can_redistribute n books neighbors child),
  (∀ min (hmin : ∃ child : Fin n, can_redistribute n books neighbors child), True) :=
begin
  sorry
end

-- Define the modified condition of part (ii)
def can_redistribute_modified (child : Fin n) : Prop :=
  books child ≥ 3 ∧ books child > books (neighbors child) + books (neighbors (neighbors child))

-- Proposition for part (ii)
theorem infinite_redistributions_impossible (h : n ≥ 3) : 
  ∀ (books : Fin n → ℕ) (redistribute : ∀ (child : Fin n), can_redistribute_modified n books neighbors child),
  (∃ min (hmin : ∃ child : Fin n, can_redistribute_modified n books neighbors child), False) :=
begin
  sorry
end

end infinite_redistributions_possible_infinite_redistributions_impossible_l747_747558


namespace total_cookies_needed_l747_747194

-- Define the conditions
def cookies_per_person : ℝ := 24.0
def number_of_people : ℝ := 6.0

-- Define the goal
theorem total_cookies_needed : cookies_per_person * number_of_people = 144.0 :=
by
  sorry

end total_cookies_needed_l747_747194


namespace problem_statement_l747_747934

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
sorry

end problem_statement_l747_747934


namespace arithmetic_sequence_15th_term_is_171_l747_747617

theorem arithmetic_sequence_15th_term_is_171 :
  ∀ (a d : ℕ), a = 3 → d = 15 - a → a + 14 * d = 171 :=
by
  intros a d h_a h_d
  rw [h_a, h_d]
  -- The proof would follow with the arithmetic calculation to determine the 15th term
  sorry

end arithmetic_sequence_15th_term_is_171_l747_747617


namespace solve_equation_l747_747979

noncomputable def is_integer_part (x y : ℝ) : Prop := y = Real.floor(x)

theorem solve_equation : ∃ x : ℝ, (3 * x^3 - Real.floor x = 3) ∧ x = Real.cbrt (4 / 3) :=
by
  sorry

end solve_equation_l747_747979


namespace roots_quartic_ab_plus_a_plus_b_l747_747070

theorem roots_quartic_ab_plus_a_plus_b (a b : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0) :
  a * b + a + b = -1 := 
sorry

end roots_quartic_ab_plus_a_plus_b_l747_747070


namespace common_diff_arithmetic_seq_min_geom_seq_k_min_arith_seq_k_d_l747_747363

-- Question 1
theorem common_diff_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) (h₀ : a 1 = 10)
  (h₁ : ∀ n : ℕ, n ≥ 1 → a n - 10 ≤ a (n + 1) ∧ a (n + 1) ≤ a n + 10)
  (h₂ : ∀ n : ℕ, n ≥ 1 → S n = ∑ i in finset.range (n + 1), a i)
  (h₃ : ∀ n : ℕ, n ≥ 1 → S n - 10 ≤ S (n + 1) ∧ S (n + 1) ≤ S n + 10)
  (h_arith : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d)
  : d = 0 :=
sorry

-- Question 2
theorem min_geom_seq_k (a : ℕ → ℤ) (q : ℤ) (k : ℕ) (h₀ : a 1 = 10)
  (h₁ : ∀ n : ℕ, 1 ≤ n → a (n + 1) = a n * q)
  (h₂ : 1 < q)
  (h₃ : ∑ i in finset.range (k + 1), a i > 2017)
  : k ≥ 8 :=
sorry

-- Question 3
theorem min_arith_seq_k_d (a : ℕ → ℤ) (d : ℤ) (k : ℕ) (h₀ : a 1 = 10)
  (h₁ : ∀ n : ℕ, 1 ≤ n → a (n + 1) = a n + d)
  (h₂ : ∑ i in finset.range (k + 1), a i = 100)
  : k = 4 ∧ d = 10 :=
sorry

end common_diff_arithmetic_seq_min_geom_seq_k_min_arith_seq_k_d_l747_747363


namespace area_of_triangle_tangent_to_exp_minus_x_l747_747612

open Real

theorem area_of_triangle_tangent_to_exp_minus_x :
  let f := λ x => exp (-x)
  let p := (1 : ℝ, exp (-1))
  let tangent_line_eq := λ x => -exp (-1) * x + 2 * exp (-1)
  let intersection_x := (2 : ℝ, 0)
  let intersection_y := (0 : ℝ, 2 * exp (-1))
  let area := (1/2) * 2 * (2 * exp (-1))
  area = 2 / exp 1 :=
by
  sorry

end area_of_triangle_tangent_to_exp_minus_x_l747_747612


namespace probability_both_even_l747_747909

-- Conditions
def die1 := {1, 2, 3, 4, 5, 6} -- six-sided die
def die2 := {1, 2, 3, 4, 5, 6, 7} -- seven-sided die
def even_numbers (die: Set ℕ) : Set ℕ := {x ∈ die | x % 2 = 0}

-- Define the problem
theorem probability_both_even :
  let favorable_outcomes := (even_numbers die1).card * (even_numbers die2).card in
  let total_outcomes := die1.card * die2.card in
  favorable_outcomes / total_outcomes = 3 / 14 :=
by
  sorry

end probability_both_even_l747_747909


namespace trig_solution_l747_747684

theorem trig_solution (k : ℤ) : 
  (∃ x, 2 * sin (17 * x) + sqrt 3 * cos (5 * x) + sin (5 * x) = 0 ∧ 
  (x = (6 * k - 1) * π / 66 ∨ x = (3 * k + 2) * π / 18)) :=
sorry

end trig_solution_l747_747684


namespace inscribed_circle_percentage_l747_747184

noncomputable def percentage_area_occupied_by_inscribed_circle (a : ℝ) : ℝ :=
  (π / 4) * 100

theorem inscribed_circle_percentage {a : ℝ} (h1 : a > 0) : 
  round ((π / 4) * 100) = 79 :=
by
  sorry

end inscribed_circle_percentage_l747_747184


namespace football_tournament_schedule_l747_747701

theorem football_tournament_schedule
(n : ℕ) (h : n ≥ 5):
  (∃ schedule : list (ℕ × ℕ),
    (∀ d1 d2 t, (d1 ≠ d2 ∧ (d1, t) ∈ schedule ∧ (d2, t) ∈ schedule → abs (d1 - d2) ≥ 1))
    ∧ length schedule = (n * (n - 1)) / 2) ∧
  (∀ t, ∃ i j, 0 ≤ i ∧ i ≤ j ∧ (j - i = ⌊((n - 3) / 2)⌋)) :=
sorry

end football_tournament_schedule_l747_747701


namespace c_plus_d_l747_747072

theorem c_plus_d (c d : ℝ)
  (h1 : c^3 - 12 * c^2 + 15 * c - 36 = 0)
  (h2 : 6 * d^3 - 36 * d^2 - 150 * d + 1350 = 0) :
  c + d = 7 := 
  sorry

end c_plus_d_l747_747072


namespace two_trains_cross_time_l747_747695

/-- Definition for the two trains' parameters -/
structure Train :=
  (length : ℝ)  -- length in meters
  (speed : ℝ)  -- speed in km/hr

/-- The parameters of Train 1 and Train 2 -/
def train1 : Train := { length := 140, speed := 60 }
def train2 : Train := { length := 160, speed := 40 }

noncomputable def relative_speed_mps (t1 t2 : Train) : ℝ :=
  (t1.speed + t2.speed) * (5 / 18)

noncomputable def total_length (t1 t2 : Train) : ℝ :=
  t1.length + t2.length

noncomputable def time_to_cross (t1 t2 : Train) : ℝ :=
  total_length t1 t2 / relative_speed_mps t1 t2

theorem two_trains_cross_time :
  time_to_cross train1 train2 = 10.8 := by
  sorry

end two_trains_cross_time_l747_747695


namespace mosaic_height_is_10_feet_l747_747086

theorem mosaic_height_is_10_feet
  (length_feet : ℕ) (tile_area_inches_squared : ℕ) (num_tiles : ℕ)
  (length_feet_eq : length_feet = 15)
  (tile_area_inches_squared_eq : tile_area_inches_squared = 1)
  (num_tiles_eq : num_tiles = 21600) :
  (num_tiles * tile_area_inches_squared = 21600 * 1) →
  (length_feet * 12 = 180) →
  (21600 * 1 / 180 = 120) →
  (120 / 12 = 10) →
  length_feet = 15 → 
  num_tiles * tile_area_inches_squared / (15 * 12) / 12 = 10 := 
by {
  intros;
  rw [length_feet_eq, tile_area_inches_squared_eq, num_tiles_eq] at *;
  sorry
}

end mosaic_height_is_10_feet_l747_747086


namespace simplify_trig_expr_l747_747973

theorem simplify_trig_expr :
  (sin (Real.pi / 6) + sin (Real.pi / 3)) / (cos (Real.pi / 6) + cos (Real.pi / 3)) = tan (Real.pi / 4) :=
  sorry

end simplify_trig_expr_l747_747973


namespace tetrahedron_sum_eq_14_l747_747914

theorem tetrahedron_sum_eq_14 :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  let edges := 6
  let corners := 4
  let faces := 4
  show edges + corners + faces = 14
  sorry

end tetrahedron_sum_eq_14_l747_747914


namespace ellipse_distance_range_l747_747891

theorem ellipse_distance_range {d : ℝ} :
  let f1 := (-1 : ℝ, 0 : ℝ)
  let f2 := (1 : ℝ, 0 : ℝ)
  let ellipse := { p : ℝ × ℝ | p.1^2 + 2 * p.2^2 = 2 }
  (∃ (k m : ℝ), m ≠ -1 ∧
    let l := { p : ℝ × ℝ | p.1 = k * p.2 + m }
    (∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ l ∧ B ∈ l ∧
     ∃ (a1 a2 : ℝ), 
      a1 = (A.2 - 0) / (A.1 + 1) ∧
      a2 = (B.2 - 0) / (B.1 + 1) ∧
      let mid_slope := (a1 + a2) / 2
      (a1, mid_slope, a2) being an arithmetic_seq ∧
    (distance f2 l) = d)) ∧
  (d > sqrt 3 ∧ d < 2) :=
begin
  sorry
end

end ellipse_distance_range_l747_747891


namespace dice_sum_probability_l747_747476

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747476


namespace ball_selection_count_l747_747650

/-- 
  There are 6 balls of each of the four colors: red, blue, yellow, and green.
  Each set of 6 balls of the same color is numbered from 1 to 6.
  If 3 balls are selected such that:
  1. They have different numbers.
  2. They have different colors.
  3. Their numbers are not consecutive.
  Then, the number of ways to do this is 96.
-/
theorem ball_selection_count : 
  let colors := ["red", "blue", "yellow", "green"]
  ∧ let numbers := [1, 2, 3, 4, 5, 6]
  ∧ ∀ balls : list (string × ℕ), 
      (balls.length = 3) 
      ∧ (∀ (b : (string × ℕ)) (b' : (string × ℕ)), b ≠ b' → (b.1 ≠ b'.1))
      ∧ (∀ (b1 b2 : ℕ), b1 ≠ b2 → abs (b1 - b2) ≠ 1)
  → list.length (filter (λ (balls : list (string × ℕ)),
      (∀ (b1 b2 : ℕ), b1 ≠ b2 → abs (b1 - b2) ≠ 1)
      ∧ (balls.map prod.fst).nodup
      ∧ (balls.map prod.snd).nodup
      ∧ balls.length = 3)
      (list.product colors numbers)) = 96 :=
by
  sorry

end ball_selection_count_l747_747650


namespace age_sum_l747_747274

variables (A B C : ℕ)

theorem age_sum (h1 : A = 20 + B + C) (h2 : A^2 = 2000 + (B + C)^2) : A + B + C = 100 :=
by
  -- Assume the subsequent proof follows here
  sorry

end age_sum_l747_747274


namespace dice_sum_10_probability_l747_747422

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747422


namespace max_g_f_inequality_l747_747835

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - x / 4 - 1

theorem max_g : ∃ x : ℝ, g x = 2 * Real.log 2 - 7 / 4 :=
sorry

theorem f_inequality (x : ℝ) (hx : 0 < x) : f x < (Real.exp x - 1) / x^2 :=
sorry

end max_g_f_inequality_l747_747835


namespace dice_sum_10_probability_l747_747424

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747424


namespace geom_seq_a3_a5_sum_l747_747897

variable (a : ℕ → ℝ)
variable (q : ℝ)
hypothesis (h1 : 0 < a 1)
hypothesis (h2 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)
hypothesis (geo_seq : ∀ n, a (n+1) = a 1 * q^n)

theorem geom_seq_a3_a5_sum : a 3 + a 5 = 5 := by
  sorry

end geom_seq_a3_a5_sum_l747_747897


namespace unique_positive_integer_solution_l747_747779

theorem unique_positive_integer_solution :
  ∀ (x y : ℕ), x > 0 ∧ y > 0 → x^3 - y^3 = x * y + 61 → (x, y) = (6, 5) := 
begin
  intros x y h_pos h_eq,
  sorry
end

end unique_positive_integer_solution_l747_747779


namespace line_XY_passes_through_midpoint_AC_l747_747025

-- Definitions for quadrilateral ABCD and point properties
structure Quadrilateral :=
(A B C D : Point)
(angleA : A.angle B C = 90)
(angleC : C.angle D A = 90)

-- Midpoint definition
def midpoint (p1 p2 : Point) : Point :=
  Point.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2)

-- Problem statement in Lean 4
theorem line_XY_passes_through_midpoint_AC
  (quad : Quadrilateral)
  (circle1 : Circle)
  (circle2 : Circle)
  (intersect1 intersect2 : Point)
  (mid_AC : Point)
  (h1 : is_diameter quad.A quad.B circle1)
  (h2 : is_diameter quad.C quad.D circle2)
  (h3 : circle1.intersects circle2 intersect1)
  (h4 : circle1.intersects circle2 intersect2)
  (h5 : mid_AC = midpoint quad.A quad.C) :
  line_passing_through intersect1 intersect2 ∋ mid_AC := 
sorry

end line_XY_passes_through_midpoint_AC_l747_747025


namespace new_mean_is_correct_l747_747539

-- Definitions of the conditions
def original_mean (nums : List ℝ) : Prop :=
  nums.length = 30 ∧ ∑ num in nums, num / 30 = 45

def transformed_mean (nums : List ℝ) : ℝ :=
  (2 * (∑ num in nums, num + 20)) / 30

-- Theorem statement encapsulating the problem
theorem new_mean_is_correct (nums : List ℝ) (h : original_mean nums) :
  transformed_mean nums = 130 :=
by
  sorry

end new_mean_is_correct_l747_747539


namespace value_of_I_l747_747902

noncomputable def find_I : ℕ :=
  let N := 8 in
  /- 
    Each column must be solved as follows:
    
    Column 1: N + N
    Since N = 8, 8 + 8 = 16, hence S = 6 and carryover is 1.
    
    Column 2: I + I + 1 (carryover from Column 1)
    Therefore, 2*I + 1 should end in 0 (for carryover), so 2*I + 1 = 10.
    
    Solving, I = 5.

    Column 3: N + N + 1 (carryover from Column 2)
    N = 8, so 8 + 8 + 1 = 17, hence T = 1 (with a carryover of 1).
    
    Thus, I = 5 ensures all conditions are met.
  -/
  5

theorem value_of_I : find_I = 5 :=
  sorry

end value_of_I_l747_747902


namespace midpoint_coordinates_l747_747375

theorem midpoint_coordinates (xM yM xN yN : ℝ) (hM : xM = 3) (hM' : yM = -2) (hN : xN = -1) (hN' : yN = 0) :
  (xM + xN) / 2 = 1 ∧ (yM + yN) / 2 = -1 :=
by
  simp [hM, hM', hN, hN']
  sorry

end midpoint_coordinates_l747_747375


namespace find_x_value_l747_747338

theorem find_x_value (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 9 :=
begin
  sorry
end

end find_x_value_l747_747338


namespace evaluate_series_l747_747776

noncomputable def infinite_series :=
  ∑' n, (n^3 + 2*n^2 - 3) / (n+3).factorial

theorem evaluate_series : infinite_series = 1 / 4 :=
by
  sorry

end evaluate_series_l747_747776


namespace probability_team_A_champions_l747_747895

theorem probability_team_A_champions : 
  let p : ℚ := 1 / 2 
  let prob_team_A_win_next := p
  let prob_team_B_win_next_A_win_after := p * p
  prob_team_A_win_next + prob_team_B_win_next_A_win_after = 3 / 4 :=
by
  sorry

end probability_team_A_champions_l747_747895


namespace sequence_ninth_term_eq_256_sum_nine_terms_eq_377_l747_747394

def a_n (n : ℕ) : ℕ :=
  if odd n then 2^(n-1) else 2 * n - 1

def S_n (n : ℕ) : ℕ :=
  (Finset.range n).sum a_n

theorem sequence_ninth_term_eq_256 : a_n 9 = 256 := 
by {
  -- proof here 
  sorry
}

theorem sum_nine_terms_eq_377 : S_n 9 = 377 := 
by {
  -- proof here 
  sorry
}

end sequence_ninth_term_eq_256_sum_nine_terms_eq_377_l747_747394


namespace function_even_and_min_value_l747_747679

theorem function_even_and_min_value :
  (∀ x : ℝ, (2^x + 2^(-x)) = (2^(-x) + 2^x)) ∧ (∀ x : ℝ, (2^x + 2^(-x)) ≥ 2) :=
by
  sorry

end function_even_and_min_value_l747_747679


namespace sum_consecutive_even_l747_747120

theorem sum_consecutive_even (m : ℤ) : m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 :=
by
  sorry

end sum_consecutive_even_l747_747120


namespace polar_equation_C1_max_area_AOB_l747_747900

-- Given conditions
variables (α r θ: Real)
noncomputable def parametric_x : Real := 2 + r * Real.cos α
noncomputable def parametric_y : Real := r * Real.sin α

def polar_equation_C2 (ρ θ: Real) : Prop := ρ * Real.sin (θ + π / 6) = 3
def intersection_condition (ρ: Real) : Prop := ∃ θ: Real, ρ = 4 * Real.cos θ

-- Statement 1: Prove polar equation of C1
theorem polar_equation_C1 : ∃ (ρ θ: Real), (ρ = 4 * Real.cos θ) → (parametric_x = 2 + r * Real.cos α) ∧ (parametric_y = r * Real.sin α) ∧ intersection_condition ρ := 
sorry

-- Given points and area calculation
variables (ρ1 ρ2: Real)

def angle_condition : Prop := ∠AOB = π / 4
def area_AOB (ρ1 ρ2 θ: Real) : Real := (sqrt 2 / 4) * ρ1 * ρ2

-- Statement 2: Prove maximum area of triangle AOB
theorem max_area_AOB : 
  ∃ (ρ1 ρ2 θ: Real), (ρ1 > 0) ∧ (ρ2 > 0) ∧ (angle_condition) → 
  area_AOB ρ1 ρ2 θ = 2 + 2 * sqrt 2 :=
sorry

end polar_equation_C1_max_area_AOB_l747_747900


namespace diff_of_squares_l747_747406

theorem diff_of_squares (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
by
  sorry

end diff_of_squares_l747_747406


namespace sequence_a_n_l747_747532

def sequence_a : ℕ → ℕ
| 1     := 1
| n + 1 := sequence_a n + (n + 1) + (n + 2) + ... + (n + n) -- Arithmetic sum terms

theorem sequence_a_n (n : ℕ) : sequence_a n = (n^3 + n) / 2 := by
  sorry

end sequence_a_n_l747_747532


namespace sum_of_ceiling_sqrt_10_to_34_l747_747774

theorem sum_of_ceiling_sqrt_10_to_34 :
  (∑ n in (Finset.range 25).filter (λ x, x + 10 ≤ 34), ⌈real.sqrt (x + 10)⌉) = 127 := by
  sorry

end sum_of_ceiling_sqrt_10_to_34_l747_747774


namespace area_square_befg_l747_747661

/-- Two squares ABCD and FHIJ have areas 30 cm^2 and 20 cm^2 respectively, with sides AD and HI on a straight line.
If point E on segment AH is such that BEFG is a square, then the area of the square BEFG is 50 cm^2. -/
theorem area_square_befg :
  ∃ (AB BC CD DA : ℝ) (FH HI IJ JF : ℝ), 
  ∀ (E : ℝ), 
    AB^2 = 30 ∧ BC^2 = 30 ∧ CD^2 = 30 ∧ DA^2 = 30 ∧ 
    FH^2 = 20 ∧ HI^2 = 20 ∧ IJ^2 = 20 ∧ JF^2 = 20 ∧ 
    BEFG ∧ E ∈ segment(A, H)
  → square_area BEFG = 50 :=
sorry

end area_square_befg_l747_747661


namespace probability_sum_10_l747_747489

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l747_747489


namespace max_g_f_inequality_l747_747834

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - x / 4 - 1

theorem max_g : ∃ x : ℝ, g x = 2 * Real.log 2 - 7 / 4 :=
sorry

theorem f_inequality (x : ℝ) (hx : 0 < x) : f x < (Real.exp x - 1) / x^2 :=
sorry

end max_g_f_inequality_l747_747834


namespace total_handshakes_l747_747745

section Handshakes

-- Define the total number of players
def total_players : ℕ := 4 + 6

-- Define the number of players in 2 and 3 player teams
def num_2player_teams : ℕ := 2
def num_3player_teams : ℕ := 2

-- Define the number of players per 2 player team and 3 player team
def players_per_2player_team : ℕ := 2
def players_per_3player_team : ℕ := 3

-- Define the total number of players in 2 player teams and in 3 player teams
def total_2player_team_players : ℕ := num_2player_teams * players_per_2player_team
def total_3player_team_players : ℕ := num_3player_teams * players_per_3player_team

-- Calculate handshakes
def handshakes (total_2player : ℕ) (total_3player : ℕ) : ℕ :=
  let h1 := total_2player * (total_players - players_per_2player_team) / 2
  let h2 := total_3player * (total_players - players_per_3player_team) / 2
  h1 + h2

-- Prove the total number of handshakes
theorem total_handshakes : handshakes total_2player_team_players total_3player_team_players = 37 :=
by
  have h1 := total_2player_team_players * (total_players - players_per_2player_team) / 2
  have h2 := total_3player_team_players * (total_players - players_per_3player_team) / 2
  have h_total := h1 + h2
  sorry

end Handshakes

end total_handshakes_l747_747745


namespace probability_sum_is_10_l747_747459

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747459


namespace tamika_carlos_probability_l747_747602

theorem tamika_carlos_probability :
  let tamika_results := [10 + 11, 10 + 12, 11 + 12],
      carlos_results := [4 * 6, 4 * 7, 6 * 7] in
  (∃ t ∈ tamika_results, ∃ c ∈ carlos_results, t > c) = false :=
by sorry

end tamika_carlos_probability_l747_747602


namespace non_adjacent_A_B_arrangements_l747_747651

theorem non_adjacent_A_B_arrangements 
  (people : Finset ℕ) (A B : ℕ) 
  (hA : A ∈ people) (hB : B ∈ people) 
  (h_card : people.card = 6) :
  (number of different arrangements of people where A and B are not next to each other) = 480 := 
sorry

end non_adjacent_A_B_arrangements_l747_747651


namespace minimize_quadratic_expression_l747_747761

theorem minimize_quadratic_expression:
  ∀ x : ℝ, (∃ a b c : ℝ, a = 1 ∧ b = -8 ∧ c = 15 ∧ x^2 + b * x + c ≥ (4 - 4)^2 - 1) :=
by
  sorry

end minimize_quadratic_expression_l747_747761


namespace nearest_area_of_triangle_DEF_l747_747033

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  (√3 / 4) * a^2

theorem nearest_area_of_triangle_DEF :
  ∀ (D E F Q : ℝ) (DQ EQ FQ : ℝ), 
  DQ = 7 → EQ = 9 → FQ = 11 → 
  let s := a := (√93 : ℝ) in 
  let area := equilateral_triangle_area s in
  abs (area - 40) < 1 :=
  sorry

end nearest_area_of_triangle_DEF_l747_747033


namespace log_problem_solution_l747_747795

theorem log_problem_solution :
  2 * log 5 10 + log 5 0.25 = 2 * log 5 2 :=
by sorry

end log_problem_solution_l747_747795


namespace point_on_circle_if_tangent_line_l747_747017

theorem point_on_circle_if_tangent_line (a b : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = 1) → (ax + by - 1 = 0)) →
  a^2 + b^2 = 1 → 
  (a^2 + b^2 = 1) := 
by 
  intro h1 h2
  exact h2

end point_on_circle_if_tangent_line_l747_747017


namespace find_k_l747_747870

theorem find_k (k a : ℤ)
  (h₁ : 49 + k = a^2)
  (h₂ : 361 + k = (a + 2)^2)
  (h₃ : 784 + k = (a + 4)^2) :
  k = 6035 :=
by sorry

end find_k_l747_747870


namespace ellipse_eccentricity_l747_747276

theorem ellipse_eccentricity
  (a b c : ℝ)
  (h_ellipse : (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1))
  (h_focus_dist : ∃ A B L : ℝ×ℝ, -- arbitrary points A, B, and left focus L
    A ≠ B ∧ A ≠ L ∧ B ≠ L ∧ -- distinct points
    (∃ F₂ : ℝ×ℝ, L = (-c, 0) ∧ F₂ = (c,0) ∧
        (A.1 - c) ^ 2 + A.2 ^ 2 = (B.1 - c) ^ 2 + B.2 ^ 2 = a ^ 2 ∧
        (∃ x : ℝ, A = (x, √ (1 - x^2/a^2) * b) ∧ B = (x, -√ (1 - x^2/a^2) * b)))) :
  eccentricity = √3 / 3 :=
sorry

end ellipse_eccentricity_l747_747276


namespace probability_heads_l747_747205

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l747_747205


namespace domain_of_composite_function_l747_747990

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x → x ≤ 2 → f x = f x) →
  (∀ (x : ℝ), -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → f (x^2) = f (x^2)) :=
by
  sorry

end domain_of_composite_function_l747_747990


namespace least_multiple_72_112_199_is_310_l747_747694

theorem least_multiple_72_112_199_is_310 :
  ∃ k : ℕ, (112 ∣ k * 72) ∧ (199 ∣ k * 72) ∧ k = 310 := 
by
  sorry

end least_multiple_72_112_199_is_310_l747_747694


namespace circles_intersect_condition_l747_747811

theorem circles_intersect_condition (a : ℝ) (ha : a > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end circles_intersect_condition_l747_747811


namespace dice_sum_probability_l747_747494

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747494


namespace profit_made_after_two_years_l747_747155

variable (present_value : ℝ) (depreciation_rate : ℝ) (selling_price : ℝ) 

def value_after_one_year (present_value depreciation_rate : ℝ) : ℝ :=
  present_value - (depreciation_rate * present_value)

def value_after_two_years (value_after_one_year : ℝ) (depreciation_rate : ℝ) : ℝ :=
  value_after_one_year - (depreciation_rate * value_after_one_year)

def profit (selling_price value_after_two_years : ℝ) : ℝ :=
  selling_price - value_after_two_years

theorem profit_made_after_two_years
  (h_present_value : present_value = 150000)
  (h_depreciation_rate : depreciation_rate = 0.22)
  (h_selling_price : selling_price = 115260) :
  profit selling_price (value_after_two_years (value_after_one_year present_value depreciation_rate) depreciation_rate) = 24000 := 
by
  sorry

end profit_made_after_two_years_l747_747155


namespace area_of_triangle_pqr_zero_l747_747299

-- Conditions of the problem
def circle_center (x y r : ℝ) := ∃ l : ℝ, x = l ∧ y = r

theorem area_of_triangle_pqr_zero :
  ∀ (P Q R : ℝ × ℝ),
    (circle_center P.1 P.2 1) ∧
    (circle_center Q.1 Q.2 2) ∧
    (circle_center R.1 R.2 3) ∧
    (dist P Q = 3) ∧
    (dist Q R = 5) →
    triangle_area P Q R = 0 :=
by
  intros P Q R conditions
  sorry

-- Helper definitions
noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))) / 2

end area_of_triangle_pqr_zero_l747_747299


namespace min_value_sqrt_inequality_l747_747940

theorem min_value_sqrt_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ c, (x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2) ≥ c ^ 2 ∧
       √((x ^ 2 + y ^ 2) * (4 * x ^ 2 + y ^ 2)) / (x * y) = (2 + real.cbrt 4) / real.cbrt 2 := by
  sorry

end min_value_sqrt_inequality_l747_747940


namespace greatest_prime_factor_of_132_l747_747666

theorem greatest_prime_factor_of_132 : ∃ p : ℕ, prime p ∧ p ∣ 132 ∧ ∀ q : ℕ, prime q ∧ q ∣ 132 → q ≤ p := 
by 
  sorry

end greatest_prime_factor_of_132_l747_747666


namespace trig_inequality_l747_747405

open Real

theorem trig_inequality (α : ℝ) (k : ℤ) (h : α ≠ k * π / 2) : 
  let T := (sin α + tan α) / (cos α + cot α) in T > 0 :=
by
  let T := (sin α + tan α) / (cos α + cot α)
  sorry

end trig_inequality_l747_747405


namespace initial_amount_l747_747284

theorem initial_amount (X : ℝ) : 
  let final_amount := ((X - 600) + 800) - 1200 in 
  final_amount = 1000 → X = 2000 :=
by
  intros final_amount_eq
  have h : X - 600 + 800 - 1200 = 1000 := final_amount_eq
  sorry

end initial_amount_l747_747284


namespace gcd_71_19_l747_747142

theorem gcd_71_19 : Int.gcd 71 19 = 1 := by
  sorry

end gcd_71_19_l747_747142


namespace sum_of_factors_of_30_l747_747672

open Nat

theorem sum_of_factors_of_30 : 
  ∑ d in (Finset.filter (λ n, 30 % n = 0) (Finset.range (30 + 1))), d = 72 := by
  sorry

end sum_of_factors_of_30_l747_747672


namespace airplane_stop_time_l747_747609

-- Define the distance function s(t)
def distance (t : ℝ) : ℝ := 75 * t - 1.5 * t^2

-- The problem statement: Prove that the airplane stops at t = 25
theorem airplane_stop_time : is_maximizer (distance) 25 := 
begin
  sorry
end

end airplane_stop_time_l747_747609


namespace dice_sum_probability_l747_747493

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747493


namespace max_triangles_in_ngon_l747_747366

theorem max_triangles_in_ngon (n : ℕ) (h : n ≥ 4) (T : set (finset (fin n))) 
  (hT1 : ∀ t ∈ T, t.card = 3)
  (hT2 : ∀ t1 t2 ∈ T, t1 ≠ t2 → t1 ∩ t2 = ∅ ∨ t1 ∩ t2 = (t1 ∩ t2).val.card = 2) : 
  T.card ≤ n := 
sorry

end max_triangles_in_ngon_l747_747366


namespace calculation_problem_quadratic_formula_l747_747698

-- Problem (1)
theorem calculation_problem : 
  (2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6) - (Real.sqrt 2 - 1)^2 = 3 + 2 * Real.sqrt 2 :=
begin
  sorry
end

-- Problem (2)
theorem quadratic_formula (a b c : ℝ) (ha : a ≠ 0) (h : b^2 - 4 * a * c ≥ 0) : 
  ∃ x1 x2 : ℝ, (ax^2 + bx + c = 0) ∧ 
               (x1 = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) ∧ 
               (x2 = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) :=
begin
  sorry
end

end calculation_problem_quadratic_formula_l747_747698


namespace problem1_problem2_problem2_equality_l747_747545

variable {a b c d : ℝ}

-- Problem 1
theorem problem1 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a + b + c + d = 6) : d < 0.36 :=
sorry

-- Problem 2
theorem problem2 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a^2 + b^2 + c^2 + d^2 = 14) : (a + c) * (b + d) ≤ 8 :=
sorry

theorem problem2_equality (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) (h4 : d = 0) : (a + c) * (b + d) = 8 :=
sorry

end problem1_problem2_problem2_equality_l747_747545


namespace inequality_proof_l747_747802

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∛((a / (b + c)) ^ 2) + ∛((b / (c + a)) ^ 2) + ∛((c / (a + b)) ^ 2)) ≥ (3 / ∛(4)) :=
  sorry

end inequality_proof_l747_747802


namespace problem_statement_l747_747932

theorem problem_statement (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := 
sorry

end problem_statement_l747_747932


namespace olympic_savings_l747_747252

theorem olympic_savings (a p : ℝ) (h_p : p ≠ 0) :
  (∑ k in finset.range 8, a * (1 + p)^k) = (a / p) * ((1 + p)^8 - (1 + p)) :=
sorry

end olympic_savings_l747_747252


namespace log_product_l747_747854

theorem log_product (x y : ℝ) (hx1 : log (x^3 * y^5) = 2) (hx2 : log (x^4 * y^2) = 2) (hx3 : log (x^2 * y^7) = 3) : 
  log (x * y) = 4 / 7 := 
by 
  sorry

end log_product_l747_747854


namespace dice_sum_probability_l747_747451

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747451


namespace find_valid_k_l747_747327

theorem find_valid_k (k : ℕ) (n : ℕ) (d : ℕ → ℕ) 
  (h1 : k ≥ 5) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n) 
  (h3 : strict_mono_on d (set.Icc 1 k)) 
  (h4 : (d 2) * (d 3) + (d 3) * (d 5) + (d 5) * (d 2) = n) 
  : k = 8 ∨ k = 9 := 
sorry

end find_valid_k_l747_747327


namespace probability_heads_equals_l747_747214

theorem probability_heads_equals (p q: ℚ) (h1 : q = 1 - p) (h2 : (binomial 10 5) * p^5 * q^5 = (binomial 10 6) * p^6 * q^4) : p = 6 / 11 :=
by {
  sorry
}

end probability_heads_equals_l747_747214


namespace N_divisible_by_9_l747_747104

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem N_divisible_by_9 (N : ℕ) (h : sum_of_digits N = sum_of_digits (5 * N)) : N % 9 = 0 := 
sorry

end N_divisible_by_9_l747_747104


namespace toys_in_stock_l747_747741

theorem toys_in_stock (sold_first_week sold_second_week toys_left toys_initial: ℕ) :
  sold_first_week = 38 → 
  sold_second_week = 26 → 
  toys_left = 19 → 
  toys_initial = sold_first_week + sold_second_week + toys_left → 
  toys_initial = 83 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end toys_in_stock_l747_747741


namespace count_numbers_with_remainder_7_dividing_65_l747_747847

theorem count_numbers_with_remainder_7_dividing_65 : 
  (∃ n : ℕ, n > 7 ∧ n ∣ 58 ∧ 65 % n = 7) ∧ 
  (∀ m : ℕ, m > 7 ∧ m ∣ 58 ∧ 65 % m = 7 → m = 29 ∨ m = 58) :=
sorry

end count_numbers_with_remainder_7_dividing_65_l747_747847


namespace complex_number_cubic_l747_747869

theorem complex_number_cubic (z : ℂ) (h : z^2 + 2 = 0) : z^3 = (2 * complex.sqrt 2) * complex.I ∨ z^3 = -(2 * complex.sqrt 2) * complex.I := 
sorry

end complex_number_cubic_l747_747869


namespace find_r_l747_747121

def sum_powers_of_seven := ∑ i in Finset.range 983, 7 ^ (i ^ 2)

def r := sum_powers_of_seven % 983

theorem find_r (h : 0 ≤ r ∧ r ≤ 492) : r = 819 := 
by 
  sorry

end find_r_l747_747121


namespace find_y_given_x_l747_747156

-- Let x and y be real numbers
variables (x y : ℝ)

-- Assume x and y are inversely proportional, so their product is a constant C
variable (C : ℝ)

-- Additional conditions from the problem statement
variable (h1 : x + y = 40) (h2 : x - y = 10) (hx : x = 7)

-- Define the goal: y = 375 / 7
theorem find_y_given_x : y = 375 / 7 :=
sorry

end find_y_given_x_l747_747156


namespace greatest_possible_value_of_a_l747_747991

theorem greatest_possible_value_of_a :
  ∃ a : ℕ, (∀ x : ℤ, x * (x + a) = -12) → a = 13 := by
  sorry

end greatest_possible_value_of_a_l747_747991


namespace probability_sum_is_10_l747_747504

theorem probability_sum_is_10 : 
  let outcomes := [(a, b, c) | a ∈ [1, 2, 3, 4, 5, 6], b ∈ [1, 2, 3, 4, 5, 6], c ∈ [1, 2, 3, 4, 5, 6]],
      favorable_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, a + b + c = 10] in
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 9 := by
  sorry

end probability_sum_is_10_l747_747504


namespace last_digit_of_large_prime_l747_747271

theorem last_digit_of_large_prime : 
  (859433 = 214858 * 4 + 1) → 
  (∃ d, (2 ^ 859433 - 1) % 10 = d ∧ d = 1) :=
by
  intro h
  sorry

end last_digit_of_large_prime_l747_747271


namespace minimum_colored_cells_l747_747090

def L_shaped_tile := set (ℕ × ℕ)
def board := fin 4 × fin 4

def is_L_shaped (tile : L_shaped_tile) : Prop :=
  ∃ r c : ℕ, 
    tile = {(r, c), (r, c+1), (r+1, c)} ∨ 
    tile = {(r, c), (r+1, c), (r+1, c+1)} ∨ 
    tile = {(r, c), (r, c+1), (r+1, c+1)} ∨ 
    tile = {(r, c+1), (r+1, c), (r+1, c+1)}

def covers (tiles : set L_shaped_tile) (colored_cells : set (fin 4 × fin 4)) : Prop :=
  (∀ tile ∈ tiles, is_L_shaped tile) ∧
  disjoint tiles ∧
  ⋃₀ tiles = colored_cells

theorem minimum_colored_cells (colored_cells : set (fin 4 × fin 4))
  (h : ∀ tiles : set L_shaped_tile, ¬covers tiles colored_cells) :
  colored_cells.card = 16 :=
sorry

end minimum_colored_cells_l747_747090


namespace sets_given_to_friend_l747_747912

theorem sets_given_to_friend (total_cards : ℕ) (total_given_away : ℕ) (sets_brother : ℕ) 
  (sets_sister : ℕ) (cards_per_set : ℕ) (sets_friend : ℕ) 
  (h1 : total_cards = 365) 
  (h2 : total_given_away = 195) 
  (h3 : sets_brother = 8) 
  (h4 : sets_sister = 5) 
  (h5 : cards_per_set = 13) 
  (h6 : total_given_away = (sets_brother + sets_sister + sets_friend) * cards_per_set) : 
  sets_friend = 2 :=
by
  sorry

end sets_given_to_friend_l747_747912


namespace solve_problem_l747_747855

theorem solve_problem (x : ℝ) (h : sqrt (2 * x + 8) = 4) : (3 * x + 12)^2 = 576 :=
by
  sorry

end solve_problem_l747_747855


namespace john_total_skateboarded_distance_l747_747053

noncomputable def total_skateboarded_distance (to_park: ℕ) (back_home: ℕ) : ℕ :=
  to_park + back_home

theorem john_total_skateboarded_distance :
  total_skateboarded_distance 10 10 = 20 :=
by
  sorry

end john_total_skateboarded_distance_l747_747053


namespace total_votes_cast_l747_747031

/-- Define the conditions for Elvis's votes and percentage representation -/
def elvis_votes : ℕ := 45
def percentage_representation : ℚ := 1 / 4

/-- The main theorem that proves the total number of votes cast -/
theorem total_votes_cast : (elvis_votes: ℚ) / percentage_representation = 180 := by
  sorry

end total_votes_cast_l747_747031


namespace probability_sum_is_ten_l747_747440

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747440


namespace necessary_not_sufficient_condition_for_parallel_l747_747374

-- Definitions for lines and planes
variable {α : Type} [plane α] (a b : line) (α : plane)

-- Conditions
variables (h1 : makes_equal_angles_with a α)
variables (h2 : makes_equal_angles_with b α)

-- Proposition to be proved
theorem necessary_not_sufficient_condition_for_parallel :
  (∀ a b α, makes_equal_angles_with a α ∧ makes_equal_angles_with b α → ¬(a ∥ b)) ∧
  (∀ a b α, a ∥ b → makes_equal_angles_with a α ∧ makes_equal_angles_with b α) :=
sorry

end necessary_not_sufficient_condition_for_parallel_l747_747374


namespace sum_sequence_floor_div_l747_747082

open Nat

/-- Sum of the sequence defined by the floor of division by 5 -/
theorem sum_sequence_floor_div :
  ∀ n : ℕ, ∑ k in finset.range (5 * n + 1), (k / 5) = (5 * n * (n - 1) / 2) + n :=
begin
  intro n,
  sorry
end

end sum_sequence_floor_div_l747_747082


namespace james_calories_per_minute_l747_747538

-- Define the conditions
def bags : Nat := 3
def ounces_per_bag : Nat := 2
def calories_per_ounce : Nat := 150
def excess_calories : Nat := 420
def run_minutes : Nat := 40

-- Calculate the total consumed calories
def consumed_calories : Nat := (bags * ounces_per_bag) * calories_per_ounce

-- Calculate the calories burned during the run
def run_calories : Nat := consumed_calories - excess_calories

-- Calculate the calories burned per minute
def calories_per_minute : Nat := run_calories / run_minutes

-- The proof problem statement
theorem james_calories_per_minute : calories_per_minute = 12 := by
  -- Due to the proof not required, we use sorry to skip it.
  sorry

end james_calories_per_minute_l747_747538


namespace probability_sum_is_ten_l747_747439

structure Die :=
(value : ℕ)
(h_value : 1 ≤ value ∧ value ≤ 6)

def possible_rolls : List (ℕ × ℕ × ℕ) :=
List.product 
  (List.product [1, 2, 3, 4, 5, 6] [1, 2, 3, 4, 5, 6]) [1, 2, 3, 4, 5, 6]
|>.map (λ ((a, b), c) => (a, b, c))

def favorable_outcomes : List (ℕ × ℕ × ℕ) :=
possible_rolls.filter (λ (x, y, z) => x + y + z = 10)

def probability (n favorable : ℕ) : ℚ := favourable / n

theorem probability_sum_is_ten :
  probability possible_rolls.length favourable_outcomes.length = 1 / 9 :=
by
  sorry

end probability_sum_is_ten_l747_747439


namespace rectangle_perimeter_l747_747641

theorem rectangle_perimeter (side_smallest_square : ℕ) (h : side_smallest_square = 1) 
 : let side_square_A := 4 * side_smallest_square,
       side_square_B := side_square_A + side_smallest_square,
       side_square_C := side_square_B,
       side_largest_square := side_square_B + side_square_B + side_square_A,
       length_rectangle := side_largest_square + side_square_B,
       width_rectangle := side_largest_square
   in 2 * (length_rectangle + width_rectangle) = 66 :=
by
  sorry

end rectangle_perimeter_l747_747641


namespace parallel_line_eq_l747_747333

-- Definition of the given conditions
def passesThrough (x y : ℝ) (a b : ℝ) : Prop := a * x + b * y = 0

-- Given conditions
def line_through_point (x y : ℝ) := passesThrough x y (-2) 1
def line_parallel (lhs rhs : ℝ) := lhs = rhs

-- Main statement to prove: The equation of the line passing through (-2, 1) and parallel to the line 2x - 3y + 5 = 0 is 2x - 3y + 7 = 0
theorem parallel_line_eq :
  (∀ m : ℝ, line_through_point (-2) 1 2 (-3) + m = 0) →
  line_parallel (2 * (-2) - 3 * 1 + 7) 0 :=
sorry

end parallel_line_eq_l747_747333


namespace number_of_distinguishable_arrangements_l747_747589

-- Specify the number of indistinguishable gold and silver coins
def goldCoins : ℕ := 5
def silverCoins : ℕ := 5

-- The total number of coins
def totalCoins : ℕ := goldCoins + silverCoins

-- The two alternating sequences of colors
def colorSequences : ℕ := 2

-- The number of possible configurations of orientations
def orientationConfigurations : ℕ := 11

-- The proof problem statement
theorem number_of_distinguishable_arrangements :
  totalCoins = 10 ∧ colorSequences * orientationConfigurations = 22 := 
by {
  -- Define and verify problem constraints
  have h1 : totalCoins = 10,
  { exact rfl },
  have h2 : colorSequences * orientationConfigurations = 22,
  { exact rfl },
  -- Conclude the main theorem
  exact ⟨h1, h2⟩
}

end number_of_distinguishable_arrangements_l747_747589


namespace least_value_of_d_l747_747119

theorem least_value_of_d (c d : ℕ) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (hc_factors : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a ≠ b ∧ c = a * b) ∨ (∃ p : ℕ, p > 1 ∧ c = p^3))
  (hd_factors : ∃ factors : ℕ, factors = c ∧ ∃ divisors : Finset ℕ, divisors.card = factors ∧ ∀ k ∈ divisors, d % k = 0)
  (div_cd : d % c = 0) : d = 18 :=
sorry

end least_value_of_d_l747_747119


namespace circle_area_difference_l747_747130

theorem circle_area_difference (C1 C2 : ℝ) (hC1 : C1 = 264) (hC2 : C2 = 352) : 
    ∃ d : ℝ, abs (d - 4310.266) < 0.001 ∧ d = (let r1 := C1 / (2 * Real.pi) in
                                               let r2 := C2 / (2 * Real.pi) in
                                               Real.pi * r2^2 - Real.pi * r1^2) :=
by
  sorry

end circle_area_difference_l747_747130


namespace problem_proof_l747_747560

variables (a b : ℝ) (n : ℕ)

theorem problem_proof (h1: a > 0) (h2: b > 0) (h3: a + b = 1) (h4: n >= 2) :
  3/2 < 1/(a^n + 1) + 1/(b^n + 1) ∧ 1/(a^n + 1) + 1/(b^n + 1) ≤ (2^(n+1))/(2^n + 1) := sorry

end problem_proof_l747_747560


namespace ellipse_equation_l747_747137

theorem ellipse_equation (
  (foci_eq : ∀ x y : ℝ, x^2 + y^2 = 50) // one focus of the ellipse is at (0, √50)
  (intersect_line : ∀ x y : ℝ, y = 3 * x - 2) // ellipse intersects the line y = 3x - 2
  (midpoint_x : ∀ x1 x2 : ℝ, (x1 + x2) / 2 = 1 / 2) // x-coordinate of the midpoint of the chord is 1/2
  (a b : ℝ) (h1 : a^2 = 3 * b^2) (h2 : a^2 - b^2 = 50)
) : 
  ∃ a b : ℝ, (a*b > 0) ∧ (a^2 = 75) ∧ (b^2 = 25) ∧ (∀ x y : ℝ, (x^2 / 25 + y^2 / 75) = 1) := 
by 
	subclass sorry -- Proof is yet to be provided

end ellipse_equation_l747_747137


namespace dice_sum_10_probability_l747_747426

open Probability

noncomputable def probability_sum_10 : ℚ := 
  let total_outcomes := 216
  let favorable_outcomes := 21
  favorable_outcomes / total_outcomes

theorem dice_sum_10_probability :
  probability (λ (x : Fin 6 × Fin 6 × Fin 6), x.1.val + x.2.val + x.3.val + 3 = 10) = probability_sum_10 := 
sorry

end dice_sum_10_probability_l747_747426


namespace minimum_value_occurs_at_4_l747_747763

noncomputable def minimum_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f x ≤ f y

def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem minimum_value_occurs_at_4 :
  minimum_value_at quadratic_expression 4 :=
sorry

end minimum_value_occurs_at_4_l747_747763


namespace dice_sum_probability_l747_747448

theorem dice_sum_probability : 
  let die_faces := {1, 2, 3, 4, 5, 6};
      successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
      total_outcomes := (finset.range 6).card ^ 3
  in  successful_outcomes / total_outcomes = 1 / 9 :=
by
  let die_faces := {1, 2, 3, 4, 5, 6};
  let successful_outcomes := { (x, y, z) | x ∈ die_faces ∧ y ∈ die_faces ∧ z ∈ die_faces ∧ x + y + z = 10 }.toFinset.card;
  let total_outcomes := (finset.range 6).card ^ 3;
  sorry

end dice_sum_probability_l747_747448


namespace circle_center_coordinates_min_tangent_length_l747_747392

section problem

noncomputable def line_param_x (t : ℝ) : ℝ := (real.sqrt 2 / 2) * t
noncomputable def line_param_y (t : ℝ) : ℝ := (real.sqrt 2 / 2) * t + 4 * real.sqrt 2

noncomputable def polar_radius (θ : ℝ) : ℝ := 4 * real.cos (θ + real.pi / 4)

theorem circle_center_coordinates: ∃ x y : ℝ, 
  (∀ θ, polar_radius θ = real.sqrt (x^2 + y^2)) ∧
  ((x - real.sqrt 2)^2 + (y + real.sqrt 2)^2 = 4) := sorry

theorem min_tangent_length: ∃ ε > 0, ∀ t : ℝ,
  let d := real.sqrt ((line_param_x t - real.sqrt 2)^2 + 
                      (line_param_y t + real.sqrt 2)^2 - 4) in
  d ≥ 4 * real.sqrt 2 ∧ 
  (∀ ε > 0, ∃ t : ℝ, 
             real.sqrt ((line_param_x t - real.sqrt 2)^2 + 
                        (line_param_y t + real.sqrt 2)^2 - 4) < 4 * real.sqrt 2 + ε) := sorry

end problem

end circle_center_coordinates_min_tangent_length_l747_747392


namespace polar_to_cartesian_min_dist_range_of_a_range_of_m_l747_747218

-- Problem 1
theorem polar_to_cartesian (ρ θ: ℝ) (hρ : ρ = 2) (hθ : θ = Real.pi / 6) :
    (ρ * Real.cos θ, ρ * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  sorry

-- Problem 2
theorem min_dist (M N : ℝ → ℝ) (hM : ∀ θ, M θ = 2 / Real.sin θ) (hN : ∀ θ, N θ = 2 * Real.cos θ) :
    ∃ (min_val : ℝ), min_val = 1 :=
by
  sorry

-- Problem 3
theorem range_of_a (a : ℝ) (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = |x - a| + a) (hg : ∀ x, g x = 4 - x^2) 
  (h : ∃ x, g x ≥ f x) : 
    a ≤ 17 / 8 :=
by
  sorry

-- Problem 4
theorem range_of_m (m : ℝ) (p q : Prop) 
  (hp : ∀ x, p = |1 - (x - 1) / 3| ≤ 2) (hq : ∀ x, q = x^2 - 2x + 1 - m^2 ≤ 0)
  (hn : ¬p → ¬q) :
    9 ≤ m :=
by
  sorry

end polar_to_cartesian_min_dist_range_of_a_range_of_m_l747_747218


namespace product_largest_smallest_using_digits_l747_747793

theorem product_largest_smallest_using_digits (a b : ℕ) (h1 : 100 * 6 + 10 * 2 + 0 = a) (h2 : 100 * 2 + 10 * 0 + 6 = b) : a * b = 127720 := by
  -- The proof will go here
  sorry

end product_largest_smallest_using_digits_l747_747793


namespace max_sum_gcd_lcm_operation_l747_747058

-- Define the initial set of numbers
def initial_numbers : Set ℕ := {n | 1 ≤ n ∧ n ≤ 15}

-- Define the operation involving gcd and lcm
def gcd_lcm_operation (S : Set ℕ) (a b : ℕ) : Set ℕ :=
if a ∈ S ∧ b ∈ S then 
  (S \ {a, b}) ∪ {Nat.gcd a b, Nat.lcm a b}
else S

-- Define the condition that processes until no changes can be made
def final_state (S : Set ℕ) (f : Set ℕ → Set ℕ → Set ℕ) : Set ℕ :=
if ∃ a b ∈ S, S ≠ f S a b then final_state (f S S) f else S

-- The condition to check whether the process has stabilized
def stabilized (S : Set ℕ) : Prop :=
¬ ∃ a b ∈ S, let next_S := gcd_lcm_operation S a b in next_S ≠ S

-- Statement to prove the maximum sum of the numbers on the board after the process
theorem max_sum_gcd_lcm_operation :
  let S := final_state initial_numbers gcd_lcm_operation in
  stabilized S ∧ 
  S.sum = 360864 := sorry

end max_sum_gcd_lcm_operation_l747_747058


namespace possible_repeating_decimals_l747_747221

theorem possible_repeating_decimals (h : ¬∃ r, r ≠ 0 ∧ has_period 0.123456 r) : 
  ∃ n, n = 6 := 
sorry

end possible_repeating_decimals_l747_747221


namespace circle_eq_min_PQ_tangent_line_l747_747358

open Real

theorem circle_eq {C : Point} (x y : ℝ) :
  (x - 3) ^ 2 + (y + 2) ^ 2 = 25 ↔
  (∃ (C : ℝ × ℝ), C.1 + C.2 = 1 ∧ (C.1 - 3) ^ 2 + (C.2 + 2) ^ 2 = 25) ∧
  ((-1 - fst C) ^ 2 + (1 - snd C) ^ 2) = 25 ∧
  ((-2 - fst C) ^ 2 + (-2 - snd C) ^ 2) = 25 :=
sorry

theorem min_PQ {C : Point} (P Q : Point) :
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), (Q.1 - Q.2 + 5 = 0) ∧
  (P.1 - 3) ^ 2 + (P.2 + 2) ^ 2 = 25 ∧
  (Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2 = (5 * sqrt 2 - 5) ^ 2 :=
sorry

theorem tangent_line (x y : ℝ) :
  (15 * x - 8 * y + 24 = 0) ∨ (y = 3) ↔
  ∃ (l : ℝ), ((x - 3) ^ 2 + (y + 2) ^ 2 = 25) ∧ (l = (0, 3)) :=
sorry

end circle_eq_min_PQ_tangent_line_l747_747358


namespace possible_scores_card_eq_17_l747_747223

-- We define a function to compute all possible scores a player could achieve
def possible_scores : Finset ℕ := 
  Finset.image (λ (scores : Fin 9 → ℕ), Finset.sum Finset.univ (λ i, scores i))
  (Finset.filter (λ scores, Finset.sum Finset.univ (λ i, scores i) = 8) 
  (Finset.Icc (λ _, 0) (λ _, 3)))

-- Statement of the problem: Prove that the set of different possible total scores contains exactly 17 elements.
-- Note: We translate the question to a statement about the cardinality of the set of possible scores.
theorem possible_scores_card_eq_17 : Finset.card possible_scores = 17 :=
by
  sorry

end possible_scores_card_eq_17_l747_747223


namespace dice_sum_probability_l747_747474

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747474


namespace probability_sum_is_10_l747_747461

theorem probability_sum_is_10 (d1 d2 d3 : ℕ) : 
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ (1 ≤ d3 ∧ d3 ≤ 6) ∧ (d1 + d2 + d3 = 10) → 
  (((d1 = 1 ∧ d2 = 3 ∧ d3 = 6) ∨ 
    (d1 = 1 ∧ d2 = 4 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 3 ∧ d3 = 5) ∨ 
    (d1 = 2 ∧ d2 = 4 ∧ d3 = 4) ∨ 
    (d1 = 3 ∧ d2 = 3 ∧ d3 = 4) ∨ 
    (d1 = 4 ∧ d2 = 4 ∧ d3 = 2)) 
    ↔ nat.cast (1 / 9)) :=
begin
  sorry,
end

end probability_sum_is_10_l747_747461


namespace tan_triple_angle_l747_747008

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l747_747008


namespace minutes_past_midnight_l747_747049

-- Definitions for the problem

def degree_per_tick : ℝ := 30
def degree_per_minute_hand : ℝ := 6
def degree_per_hour_hand_hourly : ℝ := 30
def degree_per_hour_hand_minutes : ℝ := 0.5

def condition_minute_hand_degree := 300
def condition_hour_hand_degree := 70

-- Main theorem statement
theorem minutes_past_midnight :
  ∃ (h m: ℝ),
    degree_per_hour_hand_hourly * h + degree_per_hour_hand_minutes * m = condition_hour_hand_degree ∧
    degree_per_minute_hand * m = condition_minute_hand_degree ∧
    h * 60 + m = 110 :=
by
  sorry

end minutes_past_midnight_l747_747049


namespace number_of_ways_to_construct_cube_l747_747235

theorem number_of_ways_to_construct_cube :
  let num_white_cubes := 5
  let num_blue_cubes := 3
  let cube_size := (2, 2, 2)
  let num_rotations := 24
  let num_constructions := 4
  ∃ (num_constructions : ℕ), num_constructions = 4 :=
sorry

end number_of_ways_to_construct_cube_l747_747235


namespace inscribed_circle_percentage_l747_747183

noncomputable def percentage_area_occupied_by_inscribed_circle (a : ℝ) : ℝ :=
  (π / 4) * 100

theorem inscribed_circle_percentage {a : ℝ} (h1 : a > 0) : 
  round ((π / 4) * 100) = 79 :=
by
  sorry

end inscribed_circle_percentage_l747_747183


namespace divisible_sum_or_difference_l747_747198

theorem divisible_sum_or_difference (a : Fin 52 → ℤ) :
  ∃ i j, (i ≠ j) ∧ (a i + a j) % 100 = 0 ∨ (a i - a j) % 100 = 0 :=
by
  sorry

end divisible_sum_or_difference_l747_747198


namespace distribution_of_Y_l747_747230

theorem distribution_of_Y (p : ℝ) (X : ℕ) (Y : ℕ) (hX : X = 8) :
  ∀ k, k = 2 → 
  (P(Y = 0) = 1/45 ∧ 
   P(Y = 1) = 16/45 ∧ 
   P(Y = 2) = 28/45) :=
sorry

end distribution_of_Y_l747_747230


namespace complex_root_magnitude_one_divisible_by_6_l747_747803

theorem complex_root_magnitude_one_divisible_by_6 (n : ℕ) :
    (∃ z : ℂ, |z| = 1 ∧ z^(n+1) - z^n - 1 = 0) ↔ 6 ∣ (n + 2) :=
sorry

end complex_root_magnitude_one_divisible_by_6_l747_747803


namespace total_cost_verification_l747_747180

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 8.38

theorem total_cost_verification 
  (sc : sandwich_cost = 2.45)
  (sd : soda_cost = 0.87)
  (ns : num_sandwiches = 2)
  (nd : num_sodas = 4) :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost := 
sorry

end total_cost_verification_l747_747180


namespace distance_between_skew_lines_l747_747026

-- Definitions for the geometric configuration
def AB : ℝ := 4
def AA1 : ℝ := 4
def AD : ℝ := 3

-- Theorem statement to prove the distance between skew lines A1D and B1D1
theorem distance_between_skew_lines:
  ∃ d : ℝ, d = (6 * Real.sqrt 34) / 17 :=
sorry

end distance_between_skew_lines_l747_747026


namespace flowers_sold_difference_l747_747316

theorem flowers_sold_difference :
  ∀ (d1 d2 d3 d4 : ℕ), 
    d1 = 45 →
    d2 = d1 + 20 →
    d4 = 120 →
    d1 + d2 + d3 + d4 = 350 →
    d3 = 230 - (d1 + d2) → 
    2 * d2 - d3 = 10 :=
by
  intros d1 d2 d3 d4 h1 h2 h4 h_tot h3
  rw [←h1, ←h2, ←h3, ←h4, h_tot]  
  sorry

end flowers_sold_difference_l747_747316


namespace probability_heads_l747_747204

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l747_747204


namespace t_shirts_sold_l747_747607

theorem t_shirts_sold (total_money : ℕ) (money_per_tshirt : ℕ) (n : ℕ) 
  (h1 : total_money = 2205) (h2 : money_per_tshirt = 9) (h3 : total_money = n * money_per_tshirt) : 
  n = 245 :=
by
  sorry

end t_shirts_sold_l747_747607


namespace base_length_of_parallelogram_l747_747611

theorem base_length_of_parallelogram 
  (area : ℕ) (height : ℕ) (h₀ : area = 72) (h₁ : height = 6) : ∃ base : ℕ, base * height = area ∧ base = 12 :=
by
  use 12
  split
  sorry
  exact rfl

end base_length_of_parallelogram_l747_747611


namespace dice_sum_probability_l747_747495

theorem dice_sum_probability : (∃ s, (∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = 10)
  ∧ s = {n : ℕ // ∃ x y z : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ x + y + z = n} ∧ s.card = 24)
  → 24 / 216 = 1 / 9 := 
sorry

end dice_sum_probability_l747_747495


namespace golf_balls_proof_l747_747652

variables (G F : ℕ)
-- Condition 1: G is the number of golf balls in Bin G
-- Condition 2: F is the number of golf balls in Bin F
-- given conditions:
-- 1. F = 2/3 * G
-- 2. G + F = 150
def num_golf_balls_diff : Prop :=
  let G := 90 in  -- derived from solution steps
  let F := (2 * G) / 3 in  -- F is 2/3 of G
  G - F = 30
  
theorem golf_balls_proof : num_golf_balls_diff :=
by {
  let G := 90,
  let F := (2 * G) / 3,
  have h1 : G + F = 150, {
    sorry
  },
  show G - F = 30, {
    sorry
  }
}

end golf_balls_proof_l747_747652


namespace kira_memory_space_is_140_l747_747060

def kira_songs_memory_space 
  (n_m : ℕ) -- number of songs downloaded in the morning
  (n_d : ℕ) -- number of songs downloaded later that day
  (n_n : ℕ) -- number of songs downloaded at night
  (s : ℕ) -- size of each song in MB
  : ℕ := (n_m + n_d + n_n) * s

theorem kira_memory_space_is_140 :
  kira_songs_memory_space 10 15 3 5 = 140 := 
by
  sorry

end kira_memory_space_is_140_l747_747060


namespace basil_wins_game_l747_747345

theorem basil_wins_game (x : ℕ → ℝ) (P B : set (finset (fin 10))) (h_distinct : ∀ (p : finset (fin 10)), p ∈ P ∪ B → p.card = 5)
  (h_turns : ∀ n, ((∃ p ∈ finset.range n, p ∉ P ∪ B) → n % 2 = 1) ∧ ((∃ p ∈ finset.range n, p ∉ P ∪ B) → n % 2 = 0))
  (h_assignment : ∀ i j, i ≤ j → x i ≤ x j) (h_outcome : ∀ s, s ∈ P → ∀ s', s' ∈ B → (∑ i in s, x i) < (∑ i in s', x i)) :
  ∃ strategy, ∀ assignment, ∑ p in P, ∏ i in p, x i < ∑ b in B, ∏ i in b, x i :=
begin
  sorry
end

end basil_wins_game_l747_747345


namespace length_of_AB_l747_747901

noncomputable def line (t : ℝ) : ℝ × ℝ :=
  (1/2 * t, (sqrt 2) / 2 + (sqrt 3) / 2 * t)

noncomputable def curve (α : ℝ) : ℝ × ℝ :=
  (sqrt 2 / 2 + cos α, sqrt 2 / 2 + sin α)

def length_AB (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem length_of_AB : 
  ∃ t1 t2 α1 α2, 
  line t1 = curve α1 ∧ line t2 = curve α2 ∧ 
  length_AB (line t1) (line t2) = sqrt 10 / 2 := 
sorry

end length_of_AB_l747_747901


namespace ratio_of_triangle_area_l747_747656

theorem ratio_of_triangle_area (n : ℝ) (x : ℝ) (hypotenuse : ℝ) 
  (h1 : hypotenuse^2 = (2*x)^2 + (x)^2) 
  (area_triangle1 : ℝ)
  (area_rectangle : ℝ)
  (h2 : area_rectangle = 2 * x^2)
  (h3 : area_triangle1 = n * area_rectangle) : 
  let area_triangle2 := (x^2) / (2 * n),
      ratio := area_triangle2 / area_rectangle
  in ratio = 1 / (4 * n) :=
begin
  -- Proof will be provided here
  sorry
end

end ratio_of_triangle_area_l747_747656


namespace volume_box_l747_747193

theorem volume_box (x y : ℝ) :
  (16 - 2 * x) * (12 - 2 * y) * y = 4 * x * y ^ 2 - 24 * x * y + 192 * y - 32 * y ^ 2 :=
by sorry

end volume_box_l747_747193


namespace dice_sum_probability_l747_747479

/-- The probability that the sum of the numbers on three six-faced dice equals 10 is 1/8. -/
theorem dice_sum_probability : 
  (1 / 8 : ℚ) = (∑ x in (finset.univ : finset (fin 6)), 
                ∑ y in (finset.univ : finset (fin 6)), 
                ∑ z in (finset.univ : finset (fin 6)), 
                if x.1 + y.1 + z.1 = 10 then 1 else 0) / 6^3 :=
begin
  sorry
end

end dice_sum_probability_l747_747479


namespace column_of_1985_l747_747278

theorem column_of_1985 : ∃ col: ℕ, col = 5 ∧ (1 <= col ∧ col <= 5) :=
by 
  -- Definitions based on problem conditions:
  let sequence := λ n: ℕ, 2*n - 1         -- odd positive integers
  let row := λ x: ℕ, (x + 4) / 5          -- row number considering zero-based indexing
  let column_position := λ x: ℕ, x % 5    -- column position in the row

  -- Given the number 1985:
  -- Find its row:
  have h1 : row 1985 = 397 := by sorry
  -- Find the column of 1985 in row 397:
  have h2 : column_position 1985 = 4 := by sorry

  -- Therefore the actual column is identified as:
  exact ⟨5, rfl, sorry⟩

end column_of_1985_l747_747278


namespace horizontal_distance_travelled_l747_747270

theorem horizontal_distance_travelled (r : ℝ) (θ : ℝ) (d : ℝ)
  (h_r : r = 2) (h_θ : θ = Real.pi / 6) :
  d = 2 * Real.sqrt 3 * Real.pi := sorry

end horizontal_distance_travelled_l747_747270


namespace find_f_neg_10_l747_747818

-- Define the function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

def problem_conditions (f : ℝ → ℝ) : Prop :=
  is_even_function (λ x : ℝ, f x + x^3) ∧ f 10 = 15

theorem find_f_neg_10 (f : ℝ → ℝ) (h : problem_conditions f) : f (-10) = 2015 :=
by
  sorry

end find_f_neg_10_l747_747818


namespace exists_permutation_l747_747766

theorem exists_permutation (n : ℕ) (hn : n = 1992) : 
  ∃ σ : Fin n → Fin n, ∀ i j k : Fin n, i < j ∧ j < k → (σ i + σ k) / 2 ≠ σ j := by
  have h1992 : n = 1992 := hn
  use fun i => i -- dummy example to illustrate the setup
  sorry

end exists_permutation_l747_747766


namespace object_speed_l747_747865

noncomputable def object_speed_approx (d_feet : ℕ) (t_seconds : ℕ) : ℝ :=
  let miles := d_feet / 5280 
  let hours := t_seconds / 3600
  miles / hours

theorem object_speed (h : object_speed_approx 300 6 ≈ 34.091) : 
  object_speed_approx 300 6 = 34.091 :=
by
  unfold object_speed_approx
  sorry

end object_speed_l747_747865


namespace cost_of_scooter_l747_747568

-- Given conditions
variables (M T : ℕ)
axiom h1 : T = M + 4
axiom h2 : T = 15

-- Proof goal: The cost of the scooter is $26
theorem cost_of_scooter : M + T = 26 :=
by sorry

end cost_of_scooter_l747_747568


namespace smallest_solution_eqn_l747_747643

theorem smallest_solution_eqn : ∃ (x : ℤ), 
  (⌊x / 2⌋ + ⌊x / 3⌋ + ⌊x / 7⌋ = x) ∧ 
  (∀ y : ℤ, (⌊y / 2⌋ + ⌊y / 3⌋ + ⌊y / 7⌋ = y) → x ≤ y) :=
begin
  use -85,
  split,
  { 
    -- Prove that -85 satisfies the equation
    sorry
  },
  {
    -- Prove that -85 is the smallest such solution by comparing with any other y
    intros y hy,
    sorry
  }
end

end smallest_solution_eqn_l747_747643


namespace sample_from_school_B_l747_747882

-- Problem definitions
variables (A B C : ℕ) -- Number of students in schools A, B, and C
variables (x d : ℕ) -- Variables used in the arithmetic sequence

-- Conditions
-- There are three schools with an arithmetic number of senior high school humanities students
def arithmetic_sequence := (A + B + C) = 1500 ∧ (B = A + d)  ∧ (C = A + 2 * d)
-- Total sample size for performance analysis 
def sample_size := (A + B + C) = 120

-- Statement to prove
theorem sample_from_school_B (A B C : ℕ) (x d : ℕ) 
  (h₁ : arithmetic_sequence A B C x d) 
  (h₂ : sample_size A B C) : 
  B = 40 :=
sorry

end sample_from_school_B_l747_747882


namespace coefficient_of_x_sq_l747_747988

theorem coefficient_of_x_sq :
  (∃ c : ℕ, polynomial.coeff ((polynomial.X + polynomial.C (1/polynomial.X) + polynomial.C 2) ^ 5) 2 = c) ∧
  c = 120 := sorry

end coefficient_of_x_sq_l747_747988


namespace find_DF_l747_747519

variables (A B C D E F : Type) [plane_geometry A B C D E F]
variables (AB BC DE DF : ℝ) (x : ℝ)

-- Conditions
axiom parallelogram_ABCD : parallelogram A B C D
axiom DC_eq : AB = DC = 15
axiom EB_eq : EB = 5
axiom DE_eq : DE = 9

-- Theorem statement
theorem find_DF : DF = 9 :=
by
  -- This is where the proof would go, but we use 'sorry' for now
  sorry

end find_DF_l747_747519


namespace q_sufficient_not_necessary_for_p_l747_747799

def p (x : ℝ) : Prop := abs x < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

theorem q_sufficient_not_necessary_for_p (x : ℝ) : (q x → p x) ∧ ¬(p x → q x) := 
by
  sorry

end q_sufficient_not_necessary_for_p_l747_747799


namespace problem_1_problem_2_l747_747828

-- Problem 1
theorem problem_1 (α : ℝ) (h : α ∈ if (π / 2, 3 * π / 2) 
  (h₁ : (cos α - 3, sin α) = sqrt (10 - 6 * cos α)) 
  (h₂ : (cos α, sin α - 3) = sqrt (10 - 6 * sin α)) :
  α = 5 * π / 4 :=
sorry

-- Problem 2
theorem problem_2 (α : ℝ) (h : α ∈ if (π / 2, 3 * π / 2) 
  (h₃ : (cos α - 3) * cos α + sin α * (sin α - 3) = 2 / 5) :
  tan α = -4 / 3 :=
sorry

end problem_1_problem_2_l747_747828


namespace problem_I_problem_II_l747_747653

-- Define the probability space for the first problem
def prob_product_div_3 : ℚ :=
  let outcomes := {(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4)} in
  let eventA := {(1, 3), (3, 1), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)} in
  finset.card eventA.to_finset / finset.card outcomes.to_finset

-- Define the probability space for the second problem
def prob_xiao_wang_wins : ℚ :=
  let outcomes_2 := {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)} in
  let favorable_B := {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (4, 1), (4, 2), (4, 3)} in
  finset.card favorable_B.to_finset / finset.card outcomes_2.to_finset

-- The statements to prove:
theorem problem_I : prob_product_div_3 = 7 / 16 := by
  sorry

theorem problem_II : prob_xiao_wang_wins = 8 / 9 := by
  sorry

end problem_I_problem_II_l747_747653


namespace kira_memory_space_is_140_l747_747059

def kira_songs_memory_space 
  (n_m : ℕ) -- number of songs downloaded in the morning
  (n_d : ℕ) -- number of songs downloaded later that day
  (n_n : ℕ) -- number of songs downloaded at night
  (s : ℕ) -- size of each song in MB
  : ℕ := (n_m + n_d + n_n) * s

theorem kira_memory_space_is_140 :
  kira_songs_memory_space 10 15 3 5 = 140 := 
by
  sorry

end kira_memory_space_is_140_l747_747059


namespace M_gt_N_l747_747817

theorem M_gt_N
  (a : Fin 1991 → ℝ) 
  (h : ∀ i, 0 < a i) :
  let M := (∑ i in Finset.range 1989, a i) * (∑ i in Finset.range 1989, a (i + 1))
  let N := (∑ i in Finset.range 1990, a i) * (∑ i in Finset.range 1988, a (i + 1))
  M > N :=
by {
  let M := (∑ i in Finset.range 1989, a i) * (∑ i in Finset.range 1989, a (i + 1)),
  let N := (∑ i in Finset.range 1990, a i) * (∑ i in Finset.range 1988, a (i + 1)),
  sorry
}

end M_gt_N_l747_747817


namespace inscribed_circle_radius_l747_747116

/-- Define a square SEAN with side length 2. -/
def square_side_length : ℝ := 2

/-- Define a quarter-circle of radius 1. -/
def quarter_circle_radius : ℝ := 1

/-- Hypothesis: The radius of the largest circle that can be inscribed in the remaining figure. -/
theorem inscribed_circle_radius :
  let S : ℝ := square_side_length
  let R : ℝ := quarter_circle_radius
  ∃ (r : ℝ), (r = 5 - 3 * Real.sqrt 2) := 
sorry

end inscribed_circle_radius_l747_747116


namespace factorize_expression_l747_747139

theorem factorize_expression (a b : ℤ) (h1 : 3 * b + a = -1) (h2 : a * b = -18) : a - b = -11 :=
by
  sorry

end factorize_expression_l747_747139


namespace max_pairs_l747_747921

theorem max_pairs (n : ℕ) (hn : n ≥ 2) : 
  ∃ M : Finset (ℕ × ℕ), 
  (∀ (i j k : ℕ), (i, j) ∈ M → (j, k) ∉ M) ∧ 
  (∑ (i, j) in M, 1) = if Even n then n^2 / 4 else (n^2 - 1) / 4 :=
by sorry

end max_pairs_l747_747921


namespace num_divisors_1215_l747_747321

theorem num_divisors_1215 : (Finset.filter (λ d => 1215 % d = 0) (Finset.range (1215 + 1))).card = 12 :=
by
  sorry

end num_divisors_1215_l747_747321


namespace egg_condition_difference_l747_747569

theorem egg_condition_difference :
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  in perfect_condition - cracked_eggs = 9 :=
by
  let initial_eggs := 2 * 12
  let broken_eggs := 3
  let cracked_eggs := 2 * broken_eggs
  let not_perfect_condition := broken_eggs + cracked_eggs
  let perfect_condition := initial_eggs - not_perfect_condition
  show perfect_condition - cracked_eggs = 9, by sorry

end egg_condition_difference_l747_747569


namespace final_price_of_set_l747_747029

theorem final_price_of_set (cost_coffee cost_cheesecake : ℕ) (discount_rate : ℕ) :
  cost_coffee = 6 → cost_cheesecake = 10 → discount_rate = 25 →
  (cost_coffee + cost_cheesecake) - (discount_rate * (cost_coffee + cost_cheesecake) / 100) = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end final_price_of_set_l747_747029


namespace product_of_real_parts_of_roots_l747_747936

def i : ℂ := complex.I

theorem product_of_real_parts_of_roots :
  let z := complex;
  let equation : z → z := λ z, z^2 + 2*z + (3 - 7*i);
  let roots := multiset.roots (polynomial.map complex.of_real (polynomial.of_finsupp (equation z).to_finsupp));
  (roots.map (λ z, z.re)).prod = -1.5 := sorry

end product_of_real_parts_of_roots_l747_747936


namespace complement_union_eq_l747_747842

open Set

variable (U : Set ℝ) (A B : Set ℝ)

-- Conditions
def U : Set ℝ := univ
def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

-- Statement
theorem complement_union_eq :
  compl (A ∪ B) = {x | x < -1} ∪ {x | 3 ≤ x} := by
  sorry

end complement_union_eq_l747_747842
