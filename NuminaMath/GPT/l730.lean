import Mathlib

namespace find_base_b_l730_730982

theorem find_base_b (x : ℝ) (b : ℝ) : (9^(x + 5) = 10^x ∧ x = log b (9^5)) → b = 10 / 9 := by
  sorry

end find_base_b_l730_730982


namespace nonnegative_integer_solution_count_l730_730464

theorem nonnegative_integer_solution_count :
  ∃ n : ℕ, (∀ x : ℕ, x^2 + 6 * x = 0 → x = 0) ∧ n = 1 :=
by
  sorry

end nonnegative_integer_solution_count_l730_730464


namespace sequence_a2_sequence_sum_2023_l730_730227

theorem sequence_a2 :
  let a : ℕ → ℚ := λ n, if n = 0 then 1/2 else if n = 1 then 2 else if n % 3 = 0 then 1/2 else if n % 3 = 1 then 2 else -1
  in a 1 = 2 := sorry

theorem sequence_sum_2023 :
  let a : ℕ → ℚ := λ n, if n = 0 then 1/2 else if n = 1 then 2 else if n % 3 = 0 then 1/2 else if n % 3 = 1 then 2 else -1
  in (List.range 2023).map a |>.sum = 1011.5 := sorry

end sequence_a2_sequence_sum_2023_l730_730227


namespace permutations_of_3_3_3_7_7_l730_730461

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem permutations_of_3_3_3_7_7 : 
  (factorial 5) / (factorial 3 * factorial 2) = 10 :=
by
  sorry

end permutations_of_3_3_3_7_7_l730_730461


namespace sum_first_2005_terms_l730_730218

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 1 then 2004
  else if n = 2 then 2005
  else if n = 3 then 1
  else if n = 4 then -2004
  else if (n - 1) % 6 = 0 then 2004
  else if (n - 2) % 6 = 0 then 2005
  else if (n - 3) % 6 = 0 then 1
  else if (n - 4) % 6 = 0 then -2004
  else if (n - 5) % 6 = 0 then -2005
  else -1

theorem sum_first_2005_terms : 
  (\sum i in Finset.range 2005, sequence (i + 1)) = 2004 :=
  sorry

end sum_first_2005_terms_l730_730218


namespace monotonicity_and_extremes_l730_730827

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonicity_and_extremes :
  (∀ x, f x > f (-3) ∨ f x < f (-3)) ∧
  (∀ x, f x > f 1 ∨ f x < f 1) ∧
  (∀ x, (x < -3 → (∀ y, y < x → f y < f x)) ∧ (x > 1 → (∀ y, y > x → f y < f x))) ∧
  f (-3) = 10 ∧ f 1 = -(2 / 3) :=
sorry

end monotonicity_and_extremes_l730_730827


namespace sum_of_squares_and_product_l730_730986

theorem sum_of_squares_and_product (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) :
  x + y = 22 :=
sorry

end sum_of_squares_and_product_l730_730986


namespace chess_tournament_total_participants_l730_730865

-- Define the problem setup
def chess_tournament_participants : Prop :=
  ∃ (n : ℕ), 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n + 12 → 
  ∃ (points_i : ℝ), 
  (∀ j : ℕ, (1 ≤ j ∧ j ≤ n + 12 ∧ j ≠ i) → 
    match result matches between i and j
    | win -> points_i += 1
    | lose -> points_i += 0
    | draw -> points_i += 0.5))
  ∧
  ∀ k in the lowest scoring 12 players, 
  ∃ points_k, points against lowest 12 is half
  ∧
  ∀ l in the top scoring players, 
  proof summing all points and conditions 
  show resulting total participants as n + 12 == 24.

theorem chess_tournament_total_participants (n : ℕ) (h : chess_tournament_participants) : n + 12 = 24 := by {
  sorry
}

end chess_tournament_total_participants_l730_730865


namespace defective_rate_final_machined_part_l730_730275

open Probability

theorem defective_rate_final_machined_part :
  let p1 := 1 / 70
  let p2 := 1 / 69
  let p3 := 1 / 68
  (1 - (1 - p1) * (1 - p2) * (1 - p3)) = 3 / 70 := 
by
  let p1 := (1: ℝ) / 70
  let p2 := (1: ℝ) / 69
  let p3 := (1: ℝ) / 68
  have : (1 - (1 - p1) * (1 - p2) * (1 - p3)) = 3 / 70 := sorry
  exact this

end defective_rate_final_machined_part_l730_730275


namespace twins_must_be_present_l730_730237

/-
We are given:
- Vasya and Masha got married in 1994.
- They had four children by the New Year of 2015, celebrated with all six of them.
- All children were born on February 6.
- Today is February 7, 2016.
- The age of the oldest child is equal to the product of the ages of the three younger children.

Our goal: Prove that there are twins in this family.
-/

theorem twins_must_be_present 
  (a_1 a_2 a_3 a_4 : ℕ)
  (h1 : a_1 ≤ a_2) (h2 : a_2 ≤ a_3) (h3 : a_3 ≤ a_4)
  (h_conditions : a_1 + a_2 + a_3 + a_4 = 4 * (2016 - 1994) - 1)
  (h_product : a_4 = a_1 * a_2 * a_3) : 
  ∃ x y, x = y ∧ (x ∈ {a_1, a_2, a_3, a_4}) ∧ (y ∈ {a_1, a_2, a_3, a_4}) :=
by
  sorry

end twins_must_be_present_l730_730237


namespace product_divisible_by_Cs_l730_730791
open Nat

/- Define variables -/
variables (k m n : ℤ)

/- Define condition that m+k+1 is a prime greater than n+1 -/
def is_special_prime (m k n : ℤ) : Prop :=
  prime (m + k + 1) ∧ (m + k + 1) > (n + 1)

/- Define the C_n function -/
noncomputable def C (s : ℤ) : ℤ := s * (s + 1)

/- Define the main theorem to be proven -/
theorem product_divisible_by_Cs
  (h1 : k < 0 ∧ m < 0 ∧ n < 0) 
  (h2 : is_special_prime m k n) :
  (∏ i in (range n).map (λ i, C (m + 1 + i) - C k)) % (∏ i in (range n).map (λ i, C (i + 1))) = 0 := 
sorry

end product_divisible_by_Cs_l730_730791


namespace hexagon_angle_CAB_is_30_l730_730485

theorem hexagon_angle_CAB_is_30 (ABCDEF : Type) [hexagon : IsRegularHexagon ABCDEF]
  (A B C D E F : Point ABCDEF) (h1 : InteriorAngle ABCDEF = 120) (h2 : IsDiagonal AC ) :
  ∠CAB = 30 :=
by
  sorry

end hexagon_angle_CAB_is_30_l730_730485


namespace evaluate_polynomial_at_6_eq_1337_l730_730241

theorem evaluate_polynomial_at_6_eq_1337 :
  (3 * 6^2 + 15 * 6 + 7) + (4 * 6^3 + 8 * 6^2 - 5 * 6 + 10) = 1337 := by
  sorry

end evaluate_polynomial_at_6_eq_1337_l730_730241


namespace steve_family_time_l730_730186

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l730_730186


namespace value_of_m_plus_b_l730_730206

-- Define the points
def point1 : ℝ × ℝ := (1, 7)
def point2 : ℝ × ℝ := (-2, -1)

-- Define the function that computes the slope
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

-- Define the function that computes the y-intercept
def y_intercept (p1 : ℝ × ℝ) (m : ℝ) : ℝ :=
  p1.snd - m * p1.fst

-- Define the proof problem
theorem value_of_m_plus_b (p1 p2 : ℝ × ℝ) (h1 : p1 = (1, 7)) (h2 : p2 = (-2, -1)) :
  let m := slope p1 p2 in
  let b := y_intercept p1 m in
  m + b = 7 :=
by
  rw [h1, h2]
  let m := slope (1, 7) (-2, -1)
  let b := y_intercept (1, 7) m
  have h_slope : m = (8 / 3), by sorry
  have h_intercept : b = (13 / 3), by sorry
  rw [h_slope, h_intercept]
  have : (8 / 3) + (13 / 3) = 7 := by norm_num
  exact this

end value_of_m_plus_b_l730_730206


namespace seth_sold_78_candy_bars_l730_730561

def num_candy_sold_by_seth (num_candy_max: Nat): Nat :=
  3 * num_candy_max + 6

theorem seth_sold_78_candy_bars :
  num_candy_sold_by_seth 24 = 78 :=
by
  unfold num_candy_sold_by_seth
  simp
  rfl

end seth_sold_78_candy_bars_l730_730561


namespace correct_option_is_B_l730_730225

-- Define the total number of balls
def total_black_balls : ℕ := 3
def total_red_balls : ℕ := 7
def total_balls : ℕ := total_black_balls + total_red_balls

-- Define the event of drawing balls
def drawing_balls (n : ℕ) : Prop := n = 3

-- Define what a random variable is within this context
def is_random_variable (n : ℕ) : Prop :=
  n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3

-- The main statement to prove
theorem correct_option_is_B (n : ℕ) :
  drawing_balls n → is_random_variable n :=
by
  intro h
  sorry

end correct_option_is_B_l730_730225


namespace equilateral_triangle_coloring_l730_730616

theorem equilateral_triangle_coloring (color : Fin 3 → Prop) :
  (∀ i, color i = true ∨ color i = false) →
  ∃ i j : Fin 3, i ≠ j ∧ color i = color j :=
by
  sorry

end equilateral_triangle_coloring_l730_730616


namespace intersection_M_N_l730_730446

def M := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}
def N := {x : ℝ | Real.log10 x ≥ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1 ≤ x ∧ x ≤ 4} := by
  sorry

end intersection_M_N_l730_730446


namespace trig_identity_unit_circle_coordinates_problem_conditions_l730_730040

noncomputable def r (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

def sin_alpha (y r : ℝ) : ℝ := y / r
def cos_alpha (x r : ℝ) : ℝ := x / r

theorem trig_identity (x y : ℝ) (h : r x y = 5) :
  2 * sin_alpha y 5 - cos_alpha x 5 = -2 :=
sorry

theorem unit_circle_coordinates (x y : ℝ) (h : r x y = 5) :
  (cos_alpha x 5, sin_alpha y 5) = (4/5, -3/5) :=
sorry

-- Given the condition
theorem problem_conditions : r 4 (-3) = 5 :=
by sorry

end trig_identity_unit_circle_coordinates_problem_conditions_l730_730040


namespace odd_n_contains_equilateral_triangle_l730_730766

def isEquilateralPolygon (P : List (Fin n → ℝ × ℝ)) : Prop :=
  ∀ i, dist (P[i]) (P[(i+1) % n]) = 1

def containsEquilateralTriangle (P : List (Fin n → ℝ × ℝ)) : Prop :=
  ∃ (a b c : Fin n), 
    dist (P[a]) (P[b]) = 1 ∧ 
    dist (P[b]) (P[c]) = 1 ∧ 
    dist (P[c]) (P[a]) = 1

theorem odd_n_contains_equilateral_triangle (n : ℕ) (h : n ≥ 3)
    (P : List (Fin n → ℝ × ℝ))
    (h_polygon : isEquilateralPolygon P)
    (h_convex : convex hull P) :
  (containsEquilateralTriangle P) ↔ odd n :=
by
  sorry

end odd_n_contains_equilateral_triangle_l730_730766


namespace sandy_position_l730_730262

structure Position :=
  (x : ℤ)
  (y : ℤ)

def initial_position : Position := { x := 0, y := 0 }
def after_south : Position := { x := 0, y := -20 }
def after_east : Position := { x := 20, y := -20 }
def after_north : Position := { x := 20, y := 0 }
def final_position : Position := { x := 30, y := 0 }

theorem sandy_position :
  final_position.x - initial_position.x = 10 ∧ final_position.y - initial_position.y = 0 :=
by
  sorry

end sandy_position_l730_730262


namespace max_take_home_pay_l730_730866

theorem max_take_home_pay (x : ℝ) (h : 0 ≤ x) : 
    (let income := 1000 * (x + 10) in income - 1000 * x (x + 10) = 55000) :=
sorry

end max_take_home_pay_l730_730866


namespace steve_spent_on_groceries_l730_730570

theorem steve_spent_on_groceries :
  let milk_cost := 3
  let cereal_count := 2
  let cereal_cost := 3.5
  let banana_count := 4
  let banana_cost := 0.25
  let apple_count := 4
  let apple_cost := 0.5
  let cookie_count := 2
  let cookie_cost := 2 * milk_cost
  milk_cost + 
  (cereal_count * cereal_cost) + 
  (banana_count * banana_cost) + 
  (apple_count * apple_cost) + 
  (cookie_count * cookie_cost) = 25 :=
by
  let milk_cost := 3
  let cereal_count := 2
  let cereal_cost := 3.5
  let banana_count := 4
  let banana_cost := 0.25
  let apple_count := 4
  let apple_cost := 0.5
  let cookie_count := 2
  let cookie_cost := 2 * milk_cost
  (milk_cost + 
  (cereal_count * cereal_cost) + 
  (banana_count * banana_cost) + 
  (apple_count * apple_cost) + 
  (cookie_count * cookie_cost)) = 25 :=
by
  sorry

end steve_spent_on_groceries_l730_730570


namespace charlie_original_price_l730_730074

theorem charlie_original_price (acorns_Alice acorns_Bob acorns_Charlie ν_Alice ν_Bob discount price_Charlie_before_discount price_Charlie_after_discount total_paid_by_AliceBob total_acorns_AliceBob average_price_per_acorn price_per_acorn_Alice price_per_acorn_Bob total_paid_Alice total_paid_Bob: ℝ) :
  acorns_Alice = 3600 →
  acorns_Bob = 2400 →
  acorns_Charlie = 4500 →
  ν_Bob = 6000 →
  ν_Alice = 9 * ν_Bob →
  price_per_acorn_Bob = ν_Bob / acorns_Bob →
  price_per_acorn_Alice = ν_Alice / acorns_Alice →
  total_paid_Alice = acorns_Alice * price_per_acorn_Alice →
  total_paid_Bob = ν_Bob →
  total_paid_by_AliceBob = total_paid_Alice + total_paid_Bob →
  total_acorns_AliceBob = acorns_Alice + acorns_Bob →
  average_price_per_acorn = total_paid_by_AliceBob / total_acorns_AliceBob →
  discount = 10 / 100 →
  price_Charlie_after_discount = average_price_per_acorn * (1 - discount) →
  price_Charlie_before_discount = average_price_per_acorn →
  price_Charlie_before_discount = 14.50 →
  price_per_acorn_Alice = 22.50 →
  price_Charlie_before_discount * acorns_Charlie = 4500 * 14.50 :=
by sorry

end charlie_original_price_l730_730074


namespace triangle_AB_length_l730_730975

-- Definitions for the problem
def radius : ℝ := 4
def angle_BOC : ℝ := 45
def right_angle_B : ℝ := 90
def AB : ℝ := 4 * Real.sqrt 2 -- This is what we need to prove

-- Main theorem
theorem triangle_AB_length : 
  ∀ (A B C O D E F : Type) 
  (r : ℝ) (α β : ℝ),
  radius = r →
  angle_BOC = α →
  right_angle_B = β →
  α = 45 →
  ∃ (a b c : ℝ), 
  (right_angle_B = 90) ∧
  (a = 4 * Real.sqrt 2) ∧ 
  true := 
by
  sorry

end triangle_AB_length_l730_730975


namespace max_P_at_10_l730_730386

noncomputable def a₁ : ℕ → ℝ := λ n, if n = 1 then 1002 else 1002 * (1/2)^(n-1)

def P (n : ℕ) : ℝ := ∏ i in Finset.range n, a₁ (i+1)

theorem max_P_at_10 :
  ∀ n : ℕ, (n = 10) → ∀ m : ℕ, P 10 ≥ P m :=
by
  sorry

end max_P_at_10_l730_730386


namespace steve_family_time_l730_730191

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l730_730191


namespace find_b_l730_730133

noncomputable def g (b x : ℝ) : ℝ := b * x^2 - Real.cos (Real.pi * x)

theorem find_b (b : ℝ) (hb : 0 < b) (h : g b (g b 1) = -Real.cos Real.pi) : b = 1 :=
by
  sorry

end find_b_l730_730133


namespace steve_family_time_l730_730187

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l730_730187


namespace area_triangle_ABC_l730_730861

-- Variables and constants
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides
variables {CD : ℝ} -- Height from C to AB

-- Assume conditions given in the problem
-- Condition 1: A > B > C
axiom A_gt_B_gt_C : A > B ∧ B > C

-- Condition 2: 2 * cos(2 * B) - 8 * cos(B) + 5 = 0
axiom cos_equation: 2 * Real.cos (2 * B) - 8 * Real.cos B + 5 = 0

-- Condition 3: tan(A) + tan(C) = 3 + sqrt(3)
axiom tan_sum : Real.tan A + Real.tan C = 3 + Real.sqrt 3

-- Condition 4: height CD from C to AB is 2 * sqrt(3)
def height_CD : ℝ := 2 * Real.sqrt 3

-- Prove that the area of triangle ABC is 12 - 4 * sqrt(3)
theorem area_triangle_ABC : 
  1/2 * a * height_CD = 12 - 4 * Real.sqrt 3 :=
sorry

end area_triangle_ABC_l730_730861


namespace clock_right_angle_l730_730209

theorem clock_right_angle (h m : ℤ) : 
  (m ≠ 0 → m ≠ 30 → m ≠ 60) → 
  h ∈ {3, 9} :=
by sorry

end clock_right_angle_l730_730209


namespace a_plus_b_eq_2007_l730_730814

theorem a_plus_b_eq_2007 (a b : ℕ) (ha : Prime a) (hb : Odd b)
  (h : a^2 + b = 2009) : a + b = 2007 :=
by
  sorry

end a_plus_b_eq_2007_l730_730814


namespace distance_between_ports_l730_730273

-- Define the conditions
def speed_boat_still_water : ℝ := 8 -- km/h
def ratio_time_upstream_downstream : ℝ := 2 / 1
def speed_water_flow_normal : ℝ := 8 / 3 -- km/h (calculated from the condition)
def speed_water_flow_rainy : ℝ := 2 * speed_water_flow_normal
def total_time_round_trip_rainy : ℝ := 9 -- hours

-- Distance between ports A and B
def distance_AB : ℝ := 20 -- km

-- Prove the distance between ports A and B
theorem distance_between_ports : distance_AB = 20 :=
by
  have h1 : speed_boat_still_water = 8 := rfl
  have h2 : ratio_time_upstream_downstream = 2 / 1 := rfl
  have h3 : speed_water_flow_normal = 8 / 3 := rfl
  have h4 : speed_water_flow_rainy = 2 * speed_water_flow_normal := rfl
  have h5 : total_time_round_trip_rainy = 9 := rfl
  -- Sorry placeholder to indicate proof needs to be completed
  sorry

end distance_between_ports_l730_730273


namespace problem_statement_l730_730853

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_statement_l730_730853


namespace find_sin_theta_l730_730528

def direction_vector : ℝ × ℝ × ℝ := (4, 5, 8)
def normal_vector : ℝ × ℝ × ℝ := (5, -4, 7)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def cos_complement_angle : ℝ :=
  dot_product direction_vector normal_vector / (magnitude direction_vector * magnitude normal_vector)

noncomputable def sin_theta : ℝ :=
  cos_complement_angle

theorem find_sin_theta :
  sin_theta = 56 / Real.sqrt 9450 :=
by
  sorry

end find_sin_theta_l730_730528


namespace length_of_semi_minor_axis_l730_730088

-- Define the coordinates of interest
structure Point where
  x : ℝ
  y : ℝ

-- Define given points as constants
def center : Point := { x := -4, y := -2 }
def focus : Point := { x := -1, y := -2 }
def semi_major_axis_endpoint : Point := { x := -10, y := -2 }
def semi_minor_axis_endpoint : Point := { x := -4, y := 2 }

-- Distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem length_of_semi_minor_axis :
  let c := distance center focus
  let a := distance center semi_major_axis_endpoint
  let b := Real.sqrt (a^2 - c^2)
  b = 3 * Real.sqrt 3 :=
by
  -- Conditions
  let center := Point.mk (-4) (-2)
  let focus := Point.mk (-1) (-2)
  let semi_major_axis_endpoint := Point.mk (-10) (-2)

  -- Calculations derived from the given conditions
  let c := distance center focus
  let a := distance center semi_major_axis_endpoint
  let b := Real.sqrt (a^2 - c^2)

  -- Conclusion
  show b = 3 * Real.sqrt 3 from sorry

end length_of_semi_minor_axis_l730_730088


namespace correlation_derivative_l730_730161

-- Conditions: k_x(τ) is the correlation function of X(t), X(t) is differentiable and stationary
variables {X : ℝ → ℝ}
variables {k_x : ℝ → ℝ}

-- Differentiable and stationary condition
axiom X_differentiable : ∀ t, differentiable (X t)
axiom X_stationary : ∀ t1 t2, k_x (t2 - t1) = (correlation X t1 t2)

-- Theorem statement
theorem correlation_derivative (τ : ℝ) : 
  let k_dot_x := λ τ, (correlation (fun t => deriv (X t)) t -(τ)) in
    k_dot_x τ = - (deriv (deriv (k_x τ))) :=
by sorry

end correlation_derivative_l730_730161


namespace correct_inequality_l730_730418

theorem correct_inequality (a b c d : ℝ)
    (hab : a > b) (hb0 : b > 0)
    (hcd : c > d) (hd0 : d > 0) :
    Real.sqrt (a / d) > Real.sqrt (b / c) :=
by
    sorry

end correct_inequality_l730_730418


namespace minimum_path_length_eq_l730_730739

-- Define the equilateral triangle and its properties.
structure EquilateralTriangle (α : Type*) (side_length : ℝ) :=
  (A B C : α)
  (O : α) -- center
  (dist_A_B : dist A B = side_length)
  (dist_A_C : dist A C = side_length)
  (dist_B_C : dist B C = side_length)
  (is_center : ∀ p : α, dist p O = (dist p A + dist p B + dist p C) / 3)

-- Define the path length problem.
def minimum_ray_path_length (α : Type*) [MetricSpace α] (T : EquilateralTriangle α 1) : ℝ :=
  sorry

-- Statement of the theorem we want to prove.
theorem minimum_path_length_eq (α : Type*) [MetricSpace α] (T : EquilateralTriangle α 1) :
  minimum_ray_path_length α T = sqrt 39 / 3 :=
by
  sorry

end minimum_path_length_eq_l730_730739


namespace find_q_l730_730763

theorem find_q (q : ℚ) (h_nonzero: q ≠ 0) :
  ∃ q, (qx^2 - 18 * x + 8 = 0) → (324 - 32*q = 0) :=
begin
  sorry
end

end find_q_l730_730763


namespace scientific_notation_correct_l730_730252

-- Define the required value
def value : ℝ := 0.00000065

-- Define the scientific notation expression
def scientific_notation : ℝ := 6.5 * 10^(-7)

-- Assert the equality to be proven
theorem scientific_notation_correct : value = scientific_notation :=
by sorry

end scientific_notation_correct_l730_730252


namespace logarithm_solution_set_l730_730531

theorem logarithm_solution_set (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log a (x - 1) > 0) ↔ x ∈ (1, 2) :=
by
  sorry

end logarithm_solution_set_l730_730531


namespace negation_of_exponential_inequality_l730_730210

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, exp x > x) ↔ (∃ x : ℝ, exp x ≤ x) :=
by 
  sorry

end negation_of_exponential_inequality_l730_730210


namespace range_of_lambda_l730_730821

noncomputable def point := (ℝ × ℝ)

noncomputable def ellipse_condition (p : point) : Prop := 
  p.1^2 / 4 + p.2^2 = 1

def M : point := (0, 2)

noncomputable def vectors_equal (D C M: point) (λ : ℝ) : Prop :=
  (D.1 - M.1, D.2 - M.2) = (λ * (C.1 - M.1), λ * (C.2 - M.2))

theorem range_of_lambda 
  (C D : point)
  (hC : ellipse_condition C)
  (hD : ellipse_condition D)
  (h : vectors_equal D C M λ) :
  λ ∈ set.Icc (1/3 : ℝ) 3 :=
sorry

end range_of_lambda_l730_730821


namespace smallest_possible_value_of_M_l730_730140

theorem smallest_possible_value_of_M (a b c d e : ℕ) (h1 : a + b + c + d + e = 3060) 
    (h2 : a + e ≥ 1300) :
    ∃ M : ℕ, M = max (max (a + b) (max (b + c) (max (c + d) (d + e)))) ∧ M = 1174 :=
by
  sorry

end smallest_possible_value_of_M_l730_730140


namespace problem_X_plus_Y_l730_730913

def num_five_digit_even_numbers : Nat := 45000
def num_five_digit_multiples_of_7 : Nat := 12857
def X := num_five_digit_even_numbers
def Y := num_five_digit_multiples_of_7

theorem problem_X_plus_Y : X + Y = 57857 :=
by
  sorry

end problem_X_plus_Y_l730_730913


namespace irrational_c_l730_730050

def a : ℚ := 1 / 3
def b : ℝ := Real.sqrt 4
def c : ℝ := Real.pi / 3
def d : ℝ := 0.673232 -- Assuming interpret repeating decimals properly

theorem irrational_c : Irrational c := by
  sorry

end irrational_c_l730_730050


namespace equal_chords_exists_l730_730448

noncomputable def equal_chords (S1 S2 : Circle) (A : Point) : Prop := 
  ∃ (l : Line), ∃ (M N : Point), 
  S2.intersects_line l = [M, N] ∧ 
  S1.intersects_line l = [P, Q] ∧ 
  (dist M N = dist P Q)

theorem equal_chords_exists 
  (S1 S2 : Circle) (A : Point) 
  (O1 O2 : Point) 
  (h1 : S1.center = O1) 
  (h2 : S2.center = O2) : 
  ∃ (l : Line), S1.intersects_line l = S2.intersects_line l := sorry

end equal_chords_exists_l730_730448


namespace trig_expression_zero_l730_730399

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l730_730399


namespace evaluate_x_squared_minus_y_squared_l730_730030

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l730_730030


namespace max_odd_in_triangular_array_l730_730344

noncomputable def maxOddCount (n : ℕ) : ℕ := (n * (n + 1) + 2) / 3

theorem max_odd_in_triangular_array (n : ℕ) :
  ∀ (a : ℕ → ℕ → ℤ),
  (∀ i j, 1 ≤ j → j ≤ i → i < n → a i j = a (i + 1) j + a (i + 1) (j + 1)) →
  ∃ count, 
    (∀ i j, 1 ≤ j → j ≤ i → i ≤ n → 
     if a i j % 2 = 1 then count := count + 1) ∧
    count ≤ maxOddCount n := 
sorry

end max_odd_in_triangular_array_l730_730344


namespace transmission_prob_correct_transmission_scheme_comparison_l730_730099

noncomputable def transmission_prob_single (α β : ℝ) : ℝ :=
  (1 - α) * (1 - β)^2

noncomputable def transmission_prob_triple_sequence (β : ℝ) : ℝ :=
  β * (1 - β)^2

noncomputable def transmission_prob_triple_decoding_one (β : ℝ) : ℝ :=
  β * (1 - β)^2 + (1 - β)^3

noncomputable def transmission_prob_triple_decoding_zero (α : ℝ) : ℝ :=
  3 * α * (1 - α)^2 + (1 - α)^3

noncomputable def transmission_prob_single_decoding_zero (α : ℝ) : ℝ :=
  1 - α

theorem transmission_prob_correct (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  transmission_prob_single α β = (1 - α) * (1 - β)^2 ∧
  transmission_prob_triple_sequence β = β * (1 - β)^2 ∧
  transmission_prob_triple_decoding_one β = β * (1 - β)^2 + (1 - β)^3 :=
sorry

theorem transmission_scheme_comparison (α : ℝ) (hα : 0 < α ∧ α < 0.5) :
  transmission_prob_triple_decoding_zero α > transmission_prob_single_decoding_zero α :=
sorry

end transmission_prob_correct_transmission_scheme_comparison_l730_730099


namespace a_beats_b_time_difference_l730_730090

theorem a_beats_b_time_difference
  (d : ℝ) (d_A : ℝ) (d_B : ℝ)
  (t_A : ℝ)
  (h1 : d = 1000)
  (h2 : d_A = d)
  (h3 : d_B = d - 60)
  (h4 : t_A = 235) :
  (t_A - (d_B * t_A / d_A)) = 14.1 :=
by sorry

end a_beats_b_time_difference_l730_730090


namespace cannot_derive_xn_minus_1_l730_730505

open Polynomial

noncomputable def f := (X^3 - 3 * X^2 + 5 : Polynomial ℝ)
noncomputable def g := (X^2 - 4 * X : Polynomial ℝ)

lemma derivatives_zero_at_two (p : Polynomial ℝ) : (p.derivative.eval 2 = 0) :=
  by sorry

theorem cannot_derive_xn_minus_1 (p : Polynomial ℝ) (n : ℕ) :
  (p = f ∨ p = g ∨ ∃ (a b : Polynomial ℝ), a = f ∧ b = g ∧ p = a + b ∨ p = a - b ∨ p = a * b ∨ p = a.eval b ∨ ∃ c : ℝ, p = c • a) →
  p.derivative.eval 2 = 0 → n > 0 → (X^n - 1 : Polynomial ℝ).derivative.eval 2 ≠ 0 →
  p ≠ X^n - 1 :=
begin
  intros hp hp_deriv hn hn_deriv,
  sorry
end

end cannot_derive_xn_minus_1_l730_730505


namespace complex_exp1990_sum_theorem_l730_730931

noncomputable def complex_exp1990_sum (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : Prop :=
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1

theorem complex_exp1990_sum_theorem (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : complex_exp1990_sum x y h :=
  sorry

end complex_exp1990_sum_theorem_l730_730931


namespace find_a_plus_b_l730_730932

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 1) * x + 3 * a

theorem find_a_plus_b (a b : ℝ) (h1 : ∀ x ∈ set.Icc (a-3) (2*a), a*x^2 + (b-1)*x + 3*a = a*(-x)^2 + (b-1)*(-x) + 3*a) :
  a + b = 2 :=
by
  sorry

end find_a_plus_b_l730_730932


namespace voltage_meter_range_l730_730669

theorem voltage_meter_range
  (avg_rec : ℝ)
  (min_rec : ℝ)
  (total_rec : ℝ)
  (num_rec : ℕ) :
  avg_rec = 4 → min_rec = 2 → num_rec = 3 → total_rec = num_rec * avg_rec →
  ∃ (max_rec : ℝ), total_rec - min_rec = 2 * max_rec ∧ max_rec = 5 := by
  intros avg_eq min_eq num_eq total_eq
  rw [avg_eq, min_eq, num_eq] at total_eq
  have total_eq : total_rec = 3 * 4 := by sorry
  have min_is_2 : min_rec = 2 := by sorry
  have max_is_5 : ∃ (max_rec : ℝ), total_rec - min_rec = 2 * max_rec ∧ max_rec = 5 := by sorry
  exact max_is_5

end voltage_meter_range_l730_730669


namespace find_n_satisfies_equation_l730_730089

theorem find_n_satisfies_equation : 
  ∃ (n : ℤ), 135^6 + 115^6 + 85^6 + 30^6 = n^6 ∧ n > 150 := 
begin
  use 165,
  split,
  { 
    -- Prove the equation
    sorry,
  },
  { 
    -- Prove the inequality
    norm_num,
  }
end

end find_n_satisfies_equation_l730_730089


namespace sum_of_differences_S_l730_730910

def S : Set ℕ := {2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8}

def sum_of_differences (s : Set ℕ) : ℕ :=
  Finset.sum (s.toFinset.product s.toFinset) (λ p, if p.1 > p.2 then p.1 - p.2 else 0)

theorem sum_of_differences_S : sum_of_differences S = 3096 := 
by {
  sorry
}

end sum_of_differences_S_l730_730910


namespace time_with_family_l730_730188

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l730_730188


namespace point_on_segment_ratio_l730_730922

variable (A B P : Type) [AddGroup A] [AddGroup B] [AddGroup P]
variables (a b : P)

def ratio_3_4 (P : P) (A B : Type) [AddGroup A] [AddGroup B] [HasSmul ℚ P] [HasSmul ℚ A] [HasSmul ℚ B] : Prop :=
  ∃ (P : A) (t u : ℚ), (t = 4 / 7) ∧ (u = 3 / 7) ∧ (P = t • a + u • b)

theorem point_on_segment_ratio :
  ratio_3_4 P a b :=
sorry

end point_on_segment_ratio_l730_730922


namespace trains_crossing_time_l730_730270

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 3600

noncomputable def time_to_cross_trains (length_A length_B speed_A_kmph speed_B_kmph : ℝ) : ℝ :=
  let speed_A_mps := kmph_to_mps speed_A_kmph
  let speed_B_mps := kmph_to_mps speed_B_kmph
  let relative_speed_mps := speed_A_mps + speed_B_mps
  let total_length_m := length_A + length_B
  total_length_m / relative_speed_mps

theorem trains_crossing_time
  (length_A : ℝ) (length_B : ℝ)
  (speed_A_kmph : ℝ) (speed_B_kmph : ℝ)
  (h_length_A : length_A = 260)
  (h_length_B : length_B = 240.04)
  (h_speed_A_kmph : speed_A_kmph = 120)
  (h_speed_B_kmph : speed_B_kmph = 80) :
  time_to_cross_trains length_A length_B speed_A_kmph speed_B_kmph ≈ 9 :=
by 
  sorry

end trains_crossing_time_l730_730270


namespace num_valid_N_l730_730389

theorem num_valid_N : 
  ∃ n : ℕ, n = 4 ∧ ∀ (N : ℕ), (N > 0) → (∃ k : ℕ, 60 = (N+3) * k ∧ k % 2 = 0) ↔ (N = 1 ∨ N = 9 ∨ N = 17 ∨ N = 57) :=
sorry

end num_valid_N_l730_730389


namespace evaluate_x2_minus_y2_l730_730019

-- Definitions based on the conditions.
def x : ℝ
def y : ℝ
axiom cond1 : x + y = 12
axiom cond2 : 3 * x + y = 18

-- The main statement we need to prove.
theorem evaluate_x2_minus_y2 : x^2 - y^2 = -72 :=
by
  sorry

end evaluate_x2_minus_y2_l730_730019


namespace sufficient_not_necessary_condition_abs_eq_one_l730_730070

theorem sufficient_not_necessary_condition_abs_eq_one (a : ℝ) :
  (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 ∨ a = -1) :=
by
  sorry

end sufficient_not_necessary_condition_abs_eq_one_l730_730070


namespace workshop_cost_l730_730781

theorem workshop_cost
  (x : ℝ)
  (h1 : 0 < x) -- Given the cost must be positive
  (h2 : (x / 4) - 15 = x / 7) :
  x = 140 :=
by
  sorry

end workshop_cost_l730_730781


namespace remainder_when_divided_by_eleven_l730_730377

-- Definitions from the conditions
def two_pow_five_mod_eleven : ℕ := 10
def two_pow_ten_mod_eleven : ℕ := 1
def ten_mod_eleven : ℕ := 10
def ten_square_mod_eleven : ℕ := 1

-- Proposition we want to prove
theorem remainder_when_divided_by_eleven :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_eleven_l730_730377


namespace number_of_correct_conclusions_l730_730011

noncomputable theory

variables (E : Type*) [add_comm_group E] [module ℝ E]
variables (e1 e2 a b : E) 
variable  (k : ℝ)

def vectors_conditions (e1 e2 a b: E) (k: ℝ) :=
  a = 2 • e1 - e2 ∧ b = k • e1 + e2 ∧ e1 ≠ 0 ∧ e2 ≠ 0

theorem number_of_correct_conclusions (e1 e2 a b : E) (k : ℝ) :
  vectors_conditions E e1 e2 a b k →
  (
    (¬ collinear ({e1, e2} : set E) ∧ collinear ({a, b} : set E) → k = -2) ∧
    (¬ ∃ k, ¬ collinear ({a, b} : set E) ∧ collinear ({e1, e2} : set E))
  ) → 
  2 := 
by
  sorry

end number_of_correct_conclusions_l730_730011


namespace robert_can_read_books_l730_730173

theorem robert_can_read_books (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 →
  book_pages = 360 →
  total_hours = 8 →
  total_hours / (book_pages / pages_per_hour) = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end robert_can_read_books_l730_730173


namespace sum_odd_divisors_240_l730_730243

theorem sum_odd_divisors_240 : 
  let N := 240
  in let prime_factors := multiset.of [2, 2, 2, 2, 3, 5]
  in (∑ d in (finset.filter (λ d, d % 2 = 1) (finset.divisors N)), d) = 24
:= sorry

end sum_odd_divisors_240_l730_730243


namespace gcd_2024_2048_l730_730618

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l730_730618


namespace y_value_when_x_neg_one_l730_730142

theorem y_value_when_x_neg_one (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = t^2 + 3 * t + 6) 
  (h3 : x = -1) : 
  y = 16 := 
by sorry

end y_value_when_x_neg_one_l730_730142


namespace right_triangle_PR_length_l730_730919

theorem right_triangle_PR_length :
  ∀ (P Q R S T : Type) (p r : ℝ),
  right_triangle P Q R ∧ angle P Q R = 90 ∧
  midpoint S P Q ∧ midpoint T P R ∧
  QT_length P Q S T = 30 ∧
  PS_length S P = 25 ∧
  area_triple P R ≃area (triangle_triangle_triple P S T P Q R) 3 →
  PR_length P R = 5 * (real.sqrt 109) :=
begin
  sorry
end

end right_triangle_PR_length_l730_730919


namespace asep_wins_in_at_most_n_minus_5_div_4_steps_l730_730692

theorem asep_wins_in_at_most_n_minus_5_div_4_steps (n : ℕ) (h : n ≥ 14) : 
  ∃ f : ℕ → ℕ, (∀ X d : ℕ, 0 < d → d ∣ X → (X' = X + d ∨ X' = X - d) → (f X' ≤ f X + 1)) ∧ f n ≤ (n - 5) / 4 := 
sorry

end asep_wins_in_at_most_n_minus_5_div_4_steps_l730_730692


namespace tan_arccos_l730_730738

theorem tan_arccos (θ : ℝ) (h : cos θ = 3 / 5) : tan (arccos (3 / 5)) = 4 / 3 := by
  sorry

end tan_arccos_l730_730738


namespace prime_binom_divisible_by_p_l730_730926

open Nat

theorem prime_binom_divisible_by_p 
  (p : ℕ) (hp : Prime p) (m : ℕ) (hm : 1 ≤ m ∧ m ≤ p - 1) : 
  p ∣ (-1 : ℤ)^m * (Nat.choose (p - 1) m : ℤ) - 1 := 
sorry

end prime_binom_divisible_by_p_l730_730926


namespace count_integers_divisible_by_63_l730_730462

theorem count_integers_divisible_by_63 (a b : ℤ) (h₁ : a = 50) (h₂ : b = 500) : 
  ∃ n : ℕ, n = 7 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ n) → 63 * k > a ∧ 63 * k < b := 
by 
  unfold a b at h₁ h₂ 
  sorry

end count_integers_divisible_by_63_l730_730462


namespace percentage_deficit_l730_730098

theorem percentage_deficit
  (L W : ℝ)
  (h1 : ∃(x : ℝ), 1.10 * L * (W * (1 - x / 100)) = L * W * 1.045) :
  ∃ (x : ℝ), x = 5 :=
by
  sorry

end percentage_deficit_l730_730098


namespace elvins_first_month_bill_l730_730357

variable (F C : ℕ)

def total_bill_first_month := F + C
def total_bill_second_month := F + 2 * C

theorem elvins_first_month_bill :
  total_bill_first_month F C = 46 ∧
  total_bill_second_month F C = 76 ∧
  total_bill_second_month F C - total_bill_first_month F C = 30 →
  total_bill_first_month F C = 46 :=
by
  intro h
  sorry

end elvins_first_month_bill_l730_730357


namespace transmission_prob_correct_transmission_scheme_comparison_l730_730100

noncomputable def transmission_prob_single (α β : ℝ) : ℝ :=
  (1 - α) * (1 - β)^2

noncomputable def transmission_prob_triple_sequence (β : ℝ) : ℝ :=
  β * (1 - β)^2

noncomputable def transmission_prob_triple_decoding_one (β : ℝ) : ℝ :=
  β * (1 - β)^2 + (1 - β)^3

noncomputable def transmission_prob_triple_decoding_zero (α : ℝ) : ℝ :=
  3 * α * (1 - α)^2 + (1 - α)^3

noncomputable def transmission_prob_single_decoding_zero (α : ℝ) : ℝ :=
  1 - α

theorem transmission_prob_correct (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  transmission_prob_single α β = (1 - α) * (1 - β)^2 ∧
  transmission_prob_triple_sequence β = β * (1 - β)^2 ∧
  transmission_prob_triple_decoding_one β = β * (1 - β)^2 + (1 - β)^3 :=
sorry

theorem transmission_scheme_comparison (α : ℝ) (hα : 0 < α ∧ α < 0.5) :
  transmission_prob_triple_decoding_zero α > transmission_prob_single_decoding_zero α :=
sorry

end transmission_prob_correct_transmission_scheme_comparison_l730_730100


namespace closest_integer_sqrt33_l730_730586

theorem closest_integer_sqrt33 : 6 = Int.round (Real.sqrt 33) :=
by
  sorry

end closest_integer_sqrt33_l730_730586


namespace collected_crickets_l730_730634

-- Define the number of crickets you've already collected
def crickets_collected (x : ℕ) : Prop :=
  x + 4 = 11

-- State the theorem
theorem collected_crickets : ∃ x : ℕ, crickets_collected x :=
by {
  use 7,
  unfold crickets_collected,
  exact rfl,
}

end collected_crickets_l730_730634


namespace rotate_ln_condition_l730_730559

/-- Given the function y = ln x, rotating its graph counterclockwise around the origin by an angle
θ until it first touches the y-axis, the condition that angle θ must satisfy is sin θ = e * cos θ. -/
theorem rotate_ln_condition (θ : ℝ) :
  (∀ (x : ℝ), 0 < x → ∃ y : ℝ, y = Real.log x / x) →
  tan θ = Real.exp 1 →
  Real.sin θ = Real.exp 1 * Real.cos θ :=
by
  intros _ h
  sorry

end rotate_ln_condition_l730_730559


namespace length_of_VS_l730_730884

-- Define the conditions
variables (PQRS : Type) [has_side_length PQRS 8] (T : Type) [is_midpoint_of T PQRS]
variables (R T : Type) [folded_coincides_with R T] (V : Type) [on_segment V PQRS]

-- Define the question as a theorem
theorem length_of_VS (PQRS : Type) (T : Type) (R T : Type) (V : Type) [has_side_length PQRS 8]
  [is_midpoint_of T PQRS] [folded_coincides_with R T] [on_segment V PQRS] : 
  length_segment V PQRS = 3 := 
sorry

end length_of_VS_l730_730884


namespace corrected_mean_is_36_02_l730_730264

-- Definitions based on conditions
def incorrect_mean : ℝ := 36
def num_observations : ℕ := 50
def incorrect_observation : ℝ := 47
def correct_observation : ℝ := 48

-- Statement to prove the corrected mean
theorem corrected_mean_is_36_02 : 
  let incorrect_total_sum := incorrect_mean * num_observations in
  let difference := correct_observation - incorrect_observation in
  let corrected_total_sum := incorrect_total_sum + difference in
  corrected_total_sum / num_observations = 36.02 := 
by
  sorry

end corrected_mean_is_36_02_l730_730264


namespace functional_equation_solution_l730_730534

theorem functional_equation_solution : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f(x) * f(y) - f(x * y) = x + y) → 
  let f : ℝ → ℝ := λ x, x + 1 in
  let n := 1 in
  let s := f 2 in
  n * s = 3 :=
by
  sorry

end functional_equation_solution_l730_730534


namespace mia_more_miles_than_liam_l730_730632

-- Definitions for the distances, times, and speeds
variables (d_L t_L s_L d_Z d_M : ℝ)
-- Definitions of the conditions
def condition1 := d_L = s_L * t_L
def condition2 := d_Z = (s_L + 7) * (t_L + 2)
def condition3 := d_Z = d_L + 80
def condition4 := d_M = (s_L + 15) * (t_L + 3)

-- The proof goal: The number of additional miles Mia drove than Liam
theorem mia_more_miles_than_liam :
    condition1 ∧ condition2 ∧ condition3 ∧ condition4 → d_M - d_L = 243 :=
by
  sorry

end mia_more_miles_than_liam_l730_730632


namespace find_a8_l730_730152

noncomputable def S : ℕ → ℝ
noncomputable def a : ℕ → ℝ

-- Conditions
axiom sum_arith_seq (n : ℕ) : S n = ∑ k in finset.range (n + 1), a k
axiom sum_difference : S 10 - S 5 = 40

-- Theorem to prove
theorem find_a8 : a 8 = 8 :=
by
  sorry

end find_a8_l730_730152


namespace compute_N_45_l730_730915

open Matrix

variable {α β : Type}
variable [CommRing α]
variable [AddCommGroup β]
variable [Module α β]
variable [DecidableEq α]

noncomputable def matrix_N : Matrix (Fin 2) (Fin 2) α := sorry

theorem compute_N_45 :
  (matrix_N ⬝ (λ i, if i = 0 then (4 : α) else (5 : α))) =
    !(Fin 2) (λ i, if i = 0 then (-18 / 7 : α) else (2 / 7 : α)) :=
  by
  have h1 : matrix_N ⬝ ![λ i, if i = 0 then (1 : α) else (-2 : α)] = !(Fin 2) [4, 1] := sorry
  have h2 : matrix_N ⬝ ![λ i, if i = 0 then (2 : α) else (3 : α)] = !(Fin 2) [-2, 0] := sorry
  sorry

end compute_N_45_l730_730915


namespace polygon_sides_l730_730984

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_sides_l730_730984


namespace members_playing_both_l730_730654

theorem members_playing_both
  (N B T Neither : ℕ)
  (hN : N = 40)
  (hB : B = 20)
  (hT : T = 18)
  (hNeither : Neither = 5) :
  (B + T) - (N - Neither) = 3 := by
-- to complete the proof
sorry

end members_playing_both_l730_730654


namespace geometric_progression_condition_l730_730368

noncomputable def condition_for_geometric_progression (a q : ℝ) (n p : ℤ) : Prop :=
  ∃ m : ℤ, a = q^m

theorem geometric_progression_condition (a q : ℝ) (n p k : ℤ) :
  condition_for_geometric_progression a q n p ↔ a * q^(n + p) = a * q^k :=
by
  sorry

end geometric_progression_condition_l730_730368


namespace find_some_number_l730_730606

theorem find_some_number (d : ℝ) (x : ℝ) (h1 : d = (0.889 * x) / 9.97) (h2 : d = 4.9) :
  x = 54.9 := by
  sorry

end find_some_number_l730_730606


namespace problem_statement_l730_730343

def sequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 1
  | n+1 => let a := sequence n in a / (1 + n * a)

noncomputable def calculate_value : ℝ :=
  1 / sequence 2004 - 2000000

theorem problem_statement : calculate_value = 9011 :=
by
  sorry

end problem_statement_l730_730343


namespace line_intersects_circle_l730_730443

-- Definitions from the conditions
def line (a : ℝ) (x y : ℝ) : Prop := a * x - y = a - 3
def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y - 4 = 0

-- Theorem to be proven
theorem line_intersects_circle (a : ℝ) (h₁ : line a 1 3) (h₂ : circle 1 3) :
  ∃ x y : ℝ, line a x y ∧ circle x y :=
by
  sorry

end line_intersects_circle_l730_730443


namespace integer_solutions_l730_730567

-- Define the equation to be solved
def equation (x y : ℤ) : Prop := x * y + 3 * x - 5 * y + 3 = 0

-- Define the solutions
def solution_set : List (ℤ × ℤ) := 
  [(-13,-2), (-4,-1), (-1,0), (2, 3), (3, 6), (4, 15), (6, -21),
   (7, -12), (8, -9), (11, -6), (14, -5), (23, -4)]

-- The theorem stating the solutions are correct
theorem integer_solutions : ∀ (x y : ℤ), (x, y) ∈ solution_set → equation x y :=
by
  sorry

end integer_solutions_l730_730567


namespace relationship_between_f_sin_alpha_and_f_cos_beta_l730_730213

noncomputable def f : ℝ → ℝ := sorry
variables (α β : ℝ)
variables (hα1 : 0 < α) (hα2 : α < real.pi / 2)
variables (hβ1 : 0 < β) (hβ2 : β < real.pi / 2)
variables (h_odd : ∀ x, f (-x) = -f x)
variables (h_symm : ∀ x, f (2 - x) = f x)
variables (h_decreasing : ∀ x y, -3 ≤ x → x < y → y ≤ -2 → f y < f x)

theorem relationship_between_f_sin_alpha_and_f_cos_beta :
  f (real.sin α) = f (real.cos β) ∨ f (real.sin α) < f (real.cos β) ∨ f (real.sin α) > f (real.cos β) :=
sorry

end relationship_between_f_sin_alpha_and_f_cos_beta_l730_730213


namespace vanya_more_heads_probability_l730_730647

-- Define binomial distributions for Vanya and Tanya's coin flips
def binomial (n : ℕ) (p : ℝ) : distribution :=
  λ k : ℕ, choose n k * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the random variables X_V and X_T
def X_V (n : ℕ) : distribution := binomial (n + 1) (1 / 2)
def X_T (n : ℕ) : distribution := binomial n (1 / 2)

-- Define the probability of Vanya getting more heads than Tanya
def probability_vanya_more_heads (n : ℕ) : ℝ :=
  ∑ (k : ℕ) in finset.range (n + 2), ∑ (j : ℕ) in finset.range (k), X_V n k * X_T n j

-- The main theorem to prove
theorem vanya_more_heads_probability (n : ℕ) : probability_vanya_more_heads n = 1 / 2 :=
begin
  sorry,
end

end vanya_more_heads_probability_l730_730647


namespace pipe_length_difference_l730_730663

theorem pipe_length_difference (total_length shorter_piece : ℕ) (h1 : total_length = 68) (h2 : shorter_piece = 28) : 
  total_length - shorter_piece * 2 = 12 := 
sorry

end pipe_length_difference_l730_730663


namespace trig_expression_zero_l730_730401

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l730_730401


namespace sum_last_two_digits_fib_factorial_series_l730_730626

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_last_two_digits_fib_factorial_series :
  last_two_digits (1!) + last_two_digits (1!) + last_two_digits (2!) + last_two_digits (3!) + last_two_digits (5!) + last_two_digits (8!) = 50 :=
by
  sorry

end sum_last_two_digits_fib_factorial_series_l730_730626


namespace origin_in_ellipse_l730_730038

theorem origin_in_ellipse (k : ℝ):
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ x = 0 ∧ y = 0) →
  0 < abs k ∧ abs k < 1 :=
by
  -- Note: Proof omitted.
  sorry

end origin_in_ellipse_l730_730038


namespace gumball_problem_l730_730456

theorem gumball_problem :
  ∀ (package_size total_gumballs : ℕ),
    package_size = 5 →
    total_gumballs = 20 →
    ∃ (whole_packages left_over_gumballs : ℕ),
      whole_packages = total_gumballs / package_size ∧
      left_over_gumballs = total_gumballs % package_size ∧
      whole_packages = 4 ∧
      left_over_gumballs = 0 :=
by
  intros package_size total_gumballs
  assume h1 h2
  use (total_gumballs / package_size)
  use (total_gumballs % package_size)
  repeat split
  { rw [h1, h2], refl }
  { rw [h1, h2], refl }
  { exact Nat.div_eq_of_lt (by norm_num : 0 < 5) (by norm_num : 20 < 25) }
  { exact Nat.mod_eq_zero_of_dvd (dvd_of_mod_eq_zero (by norm_num)) }
  sorry

end gumball_problem_l730_730456


namespace new_average_height_is_184_l730_730257

-- Define the initial conditions
def original_num_students : ℕ := 35
def original_avg_height : ℕ := 180
def left_num_students : ℕ := 7
def left_avg_height : ℕ := 120
def joined_num_students : ℕ := 7
def joined_avg_height : ℕ := 140

-- Calculate the initial total height
def original_total_height := original_avg_height * original_num_students

-- Calculate the total height of the students who left
def left_total_height := left_avg_height * left_num_students

-- Calculate the new total height after the students left
def new_total_height1 := original_total_height - left_total_height

-- Calculate the total height of the new students who joined
def joined_total_height := joined_avg_height * joined_num_students

-- Calculate the new total height after the new students joined
def new_total_height2 := new_total_height1 + joined_total_height

-- Calculate the new average height
def new_avg_height := new_total_height2 / original_num_students

-- The theorem stating the result
theorem new_average_height_is_184 : new_avg_height = 184 := by
  sorry

end new_average_height_is_184_l730_730257


namespace parabola_line_intersections_l730_730139

theorem parabola_line_intersections (a b c : ℝ) (h_a_nonzero : a ≠ 0) :
  let p := λ x : ℝ, a * x^2 + b * x + c in
  (∀ x : ℝ, (p x = a * x + b → ∃! t : ℝ, t = x)) ∧
  (∀ x : ℝ, (p x = b * x + c → ∃! t : ℝ, t = x)) ∧
  (∀ x : ℝ, (p x = c * x + a → ∃! t : ℝ, t = x)) ∧
  (∀ x : ℝ, (p x = b * x + a → ∃! t : ℝ, t = x)) ∧
  (∀ x : ℝ, (p x = c * x + b → ∃! t : ℝ, t = x)) ∧
  (∀ x : ℝ, (p x = a * x + c → ∃! t : ℝ, t = x)) → 
  1 ≤ c / a ∧ c / a ≤ 5 :=
sorry

end parabola_line_intersections_l730_730139


namespace solve_for_b_l730_730979

theorem solve_for_b : 
  ∀ {x : ℝ}, (9^5 * 9^x = 10^x) → (x = Real.logb (10/9) (9^5)) → (10 / 9 = b) :=
begin
  assume x h1 h2,
  sorry
end

end solve_for_b_l730_730979


namespace tangent_to_circle_parallel_l730_730716

variable (A B C D E F : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variable (ω1 ω2 : Circle) [IsCircumscribedQuadrilateral A B C D ω1]
variable [IsCircleThroughPoints A B ω2]
variable [IntersectsRayAtOtherPoint ω2 (Ray DB E)] 
variable [IntersectsRayAtOtherPoint ω2 (Ray CA F)]
variable [TangentParallel (TangentLine ω1 C) (Line AE)]

theorem tangent_to_circle_parallel {A B C D E F : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] 
  (ω1 ω2 : Circle) [IsCircumscribedQuadrilateral A B C D ω1]
  [IsCircleThroughPoints A B ω2]
  [IntersectsRayAtOtherPoint ω2 (Ray DB E)] 
  [IntersectsRayAtOtherPoint ω2 (Ray CA F)]
  [TangentParallel (TangentLine ω1 C) (Line AE)] :
  TangentParallel (TangentLine ω2 F) (Line AD) :=
sorry

end tangent_to_circle_parallel_l730_730716


namespace compute_expression_l730_730737

theorem compute_expression (x : ℝ) (h : x = 3) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 90 :=
by 
  sorry

end compute_expression_l730_730737


namespace tan_difference_identity_l730_730809

theorem tan_difference_identity (α : ℝ) (h : tan (α + π / 6) = 1) : tan (α - π / 6) = sqrt 3 - 2 := 
sorry

end tan_difference_identity_l730_730809


namespace num_solutions_in_interval_l730_730212

theorem num_solutions_in_interval : 
  ∃ n : ℕ, n = 2 ∧ ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  2 ^ Real.cos θ = Real.sin θ → n = 2 := 
sorry

end num_solutions_in_interval_l730_730212


namespace solid_is_cone_l730_730297

-- Define the conditions as hypotheses
variable {solid : Type}
variable [FrontView : solid → Type] {front_view : Type} [IsoscelesTriangle : front_view → Prop] 
variable [SideView : solid → Type] {side_view : Type} [IsoscelesTriangle : side_view → Prop]
variable [TopView : solid → Type]  {top_view : Type} [Circle : top_view → Prop]

-- Assume the conditions
axiom front_view_is_congruent_isosceles_triangle (s : solid) : IsoscelesTriangle (FrontView s)
axiom side_view_is_congruent_isosceles_triangle (s : solid) : IsoscelesTriangle (SideView s)
axiom top_view_is_circle_with_center (s : solid) : Circle (TopView s)

-- Formalize the problem as a Lean theorem stating that the solid must be a cone
theorem solid_is_cone (s : solid) 
  (H1 : IsoscelesTriangle (FrontView s)) 
  (H2 : IsoscelesTriangle (SideView s)) 
  (H3 : Circle (TopView s)) :
  solid = Cone := sorry

end solid_is_cone_l730_730297


namespace train_a_constant_rate_l730_730232

theorem train_a_constant_rate
  (d : ℕ)
  (v_b : ℕ)
  (d_a : ℕ)
  (v : ℕ)
  (h1 : d = 350)
  (h2 : v_b = 30)
  (h3 : d_a = 200)
  (h4 : v * (d_a / v) + v_b * (d_a / v) = d) :
  v = 40 := by
  sorry

end train_a_constant_rate_l730_730232


namespace cinema_empty_showtime_exists_l730_730179

theorem cinema_empty_showtime_exists :
  ∀ (S : Finset nat) (T : Finset nat) (showtimes : Finset nat)
  (attend : nat → nat → nat → Prop), 
  S.card = 7 ∧ T.card = 7 ∧ showtimes.card = 8 ∧ 
  (∀ (s t st), attend s t st → s ∈ S ∧ t ∈ T ∧ st ∈ showtimes) ∧
  (∀ (st ∈ showtimes), (∃ t ∈ T, ∃ (s1 s2 s3 s4 s5 s6 ∈ S), s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s1 ≠ s6 ∧ 
     s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s2 ≠ s6 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s3 ≠ s6 ∧ s4 ≠ s5 ∧ s4 ≠ s6 ∧ s5 ≠ s6 ∧ 
     attend s1 t st ∧ attend s2 t st ∧ attend s3 t st ∧ attend s4 t st ∧ attend s5 t st ∧ attend s6 t st) ∧
    (∃ t' ∈ T, t' ≠ t ∧ ∃ s7 ∈ S, s7 ≠ s1 ∧ s7 ≠ s2 ∧ s7 ≠ s3 ∧ s7 ≠ s4 ∧ s7 ≠ s5 ∧ s7 ≠ s6 ∧ attend s7 t' st)) ∧
  (∀ s ∈ S, ∀ t ∈ T, ∃ st ∈ showtimes, attend s t st)
  → ∃ t ∈ T, ∃ st ∈ showtimes, ∀ s ∈ S, ¬attend s t st := 
sorry

end cinema_empty_showtime_exists_l730_730179


namespace maximum_value_parabola_l730_730075

theorem maximum_value_parabola (x : ℝ) : 
  ∃ y : ℝ, y = -3 * x^2 + 6 ∧ ∀ z : ℝ, (∃ a : ℝ, z = -3 * a^2 + 6) → z ≤ 6 :=
by
  sorry

end maximum_value_parabola_l730_730075


namespace xiao_ming_vs_xiao_hong_probability_xiao_ming_regression_estimate_l730_730671

theorem xiao_ming_vs_xiao_hong_probability :
  let successful_attempts_xm := [16, 20, 20, 25, 30, 36]
  let successful_attempts_xh := [16, 22, 25, 26, 32, 35, 35]
  let a_values := (36 : ℕ) ≤ a ∧ a ≤ (60 : ℕ)
  let xm_total_day6 := successful_attempts_xm.sum
  let xh_total := successful_attempts_xh.sum
  let possible_a := finset.Icc 36 60
  (xm_total_day6 + a) ≥ xh_total ↔
  ∑ a in possible_a.filter (λ a, (xm_total_day6 + a) ≥ xh_total), 1 / possible_a.card = 17 / 25 :=
by
  let successful_attempts_xm := [16, 20, 20, 25, 30, 36]
  let xm_total_day6 := successful_attempts_xm.sum
  let successful_attempts_xh := [16, 22, 25, 26, 32, 35, 35]
  let xh_total := successful_attempts_xh.sum
  let a_values := (36 : ℕ) ≤ a ∧ a ≤ (60 : ℕ)
  let possible_a := finset.Icc 36 60
  have h1 : xm_total_day6 + a ≥ xh_total ↔ 44 ≤ a := by sorry
  have h2 :∑ a in possible_a.filter (λ a, 44 ≤ a), 1 = 17 := by sorry
  have h3 : possible_a.card = 25 := by sorry
  simp [h1, h2, h3]
  ring

theorem xiao_ming_regression_estimate :
  let x_vals := [1, 2, 3, 4, 5, 6]
  let y_vals := [16, 20, 20, 25, 30, 36]
  let n := x_vals.length
  let x_mean := (list.sum x_vals : ℝ) / n
  let y_mean := (list.sum y_vals : ℝ) / n
  let xy_sum := ∑ i in list.zip x_vals y_vals, i.1 * i.2
  let x_sq_sum := ∑ i in x_vals, i ^ 2
  let b := (xy_sum - n * x_mean * y_mean) / (x_sq_sum - n * x_mean ^ 2)
  let a := y_mean - b * x_mean
  b = (27 : ℝ) / 7 ∧ a = 11 ∧
  (b * 7 + a = 38) :=
by 
  let x_vals := [1, 2, 3, 4, 5, 6]
  let y_vals := [16, 20, 20, 25, 30, 36]
  let n := x_vals.length
  let x_mean := (list.sum x_vals : ℝ) / n
  let y_mean := (list.sum y_vals : ℝ) / n
  let xy_sum := ∑ i in list.zip x_vals y_vals, i.1 * i.2
  let x_sq_sum := ∑ i in x_vals, i ^ 2
  let b := (xy_sum - n * x_mean * y_mean) / (x_sq_sum - n * x_mean ^ 2)
  let a := y_mean - b * x_mean
  have hb : b = (27 : ℝ) / 7 := by sorry
  have ha : a = 11 := by sorry
  have h7 : b * 7 + a = 38 := by 
    simp [hb, ha]
    ring
  exact ⟨hb, ha, h7⟩

end xiao_ming_vs_xiao_hong_probability_xiao_ming_regression_estimate_l730_730671


namespace sphere_surface_area_l730_730697

theorem sphere_surface_area (a b c : ℝ) (r : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : r = (Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)) / 2):
    4 * Real.pi * r ^ 2 = 50 * Real.pi :=
by
  sorry

end sphere_surface_area_l730_730697


namespace hyperbola_eccentricity_l730_730048

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_asymptote_parallel : b = 2 * a)
  (h_c_squared : c^2 = a^2 + b^2)
  (h_e_def : e = c / a) :
  e = Real.sqrt 5 :=
sorry

end hyperbola_eccentricity_l730_730048


namespace leading_coefficient_of_polynomial_is_15_l730_730743

def poly : ℕ → ℕ := λ x, 5 * (x^5 - 2 * x^3 + x^2) - 8 * (x^5 + 3 * x^3 - x) + 6 * (3 * x^5 - x^3 + 4)

theorem leading_coefficient_of_polynomial_is_15 : 
  ∀ x : ℕ, (poly x).leading_coeff = 15 := 
by 
  sorry

end leading_coefficient_of_polynomial_is_15_l730_730743


namespace coin_flips_prob_l730_730652

open Probability

noncomputable def probability_Vanya_more_heads_than_Tanya (n : ℕ) : ℝ :=
  P(X Vanya > X Tanya)

theorem coin_flips_prob (n : ℕ) :
   P(Vanya gets more heads than Tanya | Vanya flips (n+1) times, Tanya flips n times) = 0.5 :=
sorry

end coin_flips_prob_l730_730652


namespace problem_solution_l730_730686

-- Define the function under consideration
def f (x : ℝ) : ℝ := sin (x / 2 + π / 3)

-- Conditions from the problem
def min_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x ∧ ∀ p', p' > 0 → (f (x + p') = f x → p' ≥ p)

def is_symmetrical_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + (a - x)) = f x

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x > f y

-- Main theorem to prove
theorem problem_solution :
  min_positive_period f (4 * π) ∧ is_symmetrical_axis f (π / 3) ∧ is_decreasing_on f {x | 2 * π / 3 < x ∧ x < 5 * π / 6} :=
by
  sorry

end problem_solution_l730_730686


namespace no_poly_of_form_xn_minus_1_l730_730508

theorem no_poly_of_form_xn_minus_1 (f g : ℝ[X])
  (Hf : f = polynomial.X ^ 3 - 3 * polynomial.X ^ 2 + 5)
  (Hg : g = polynomial.X ^ 2 - 4 * polynomial.X)
  (allowed_operations : ∀ (h : ℝ[X]), h = 
    f + g ∨ h = f - g ∨ h = f * g ∨ h = polynomial.c g ∨ h = polynomial.c f ∨ h = polynomial.eval g f) :
  ¬ ∃ n : ℕ, n ≠ 0 ∧ (∃ h : ℝ[X], h = polynomial.X ^ n - 1) :=
by
  sorry

end no_poly_of_form_xn_minus_1_l730_730508


namespace chord_line_eq_l730_730807

noncomputable def circle :=
  { center : ℝ × ℝ // center = (2, 1) }

def is_point_inside_circle (p : ℝ × ℝ) (c : circle) :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < 4

def is_midpoint_of_chord (p : ℝ × ℝ) (line_eq : ℝ → ℝ → Prop) :=
  ∃ a b : ℝ × ℝ, a ≠ b ∧ (line_eq a.1 a.2 ∧ line_eq b.1 b.2) ∧ p = ((a.1 + b.1)/2, (a.2 + b.2)/2)

theorem chord_line_eq :
  ∀ (p : ℝ × ℝ), p = (3, 2) → ∀ (c : circle), is_point_inside_circle p c →
  ∃ k b : ℝ, ∀ x y : ℝ, (y = k * x + b) ↔ (x + y - 5 = 0) 
by
  sorry

end chord_line_eq_l730_730807


namespace cos_arithmetic_sequence_l730_730811

theorem cos_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 5 + a 9 = 5 * real.pi) :
  real.cos (a 2 + a 8) = -1 / 2 := 
sorry

end cos_arithmetic_sequence_l730_730811


namespace permutations_of_3_3_3_7_7_l730_730460

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem permutations_of_3_3_3_7_7 : 
  (factorial 5) / (factorial 3 * factorial 2) = 10 :=
by
  sorry

end permutations_of_3_3_3_7_7_l730_730460


namespace books_read_in_eight_hours_l730_730170

-- Definitions to set up the problem
def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

-- Theorem statement
theorem books_read_in_eight_hours : (available_time * reading_speed) / book_length = 2 := 
by
  sorry

end books_read_in_eight_hours_l730_730170


namespace fuel_consumption_total_earnings_l730_730573

def driving_sequence := [-2, 5, 8, -3, 6, -2]
def fuel_rate_per_km := 0.3
def starting_price := 10
def extra_charge_per_km := 4

/-- 
  Prove that the total fuel consumption for returning to the starting point,
  given the driving sequence and fuel consumption rate,
  is 11.4 liters.
-/
theorem fuel_consumption : 
  (driving_sequence.map (λ x, abs x)).sum + (driving_sequence.sum.abs) = 38 → 
  38 * fuel_rate_per_km = 11.4 :=
by
  sorry

/-- 
  Prove that the total earnings in yuan, given the driving sequence and fare structure, 
  is 100 yuan.
-/
theorem total_earnings :
  let charge (d : Int) := if d <= 3 then starting_price else starting_price + (d - 3) * extra_charge_per_km
  charge 2 + charge 5 + charge 8 + charge (-3) + charge 6 + charge (-2) = 100 :=
by 
  sorry

end fuel_consumption_total_earnings_l730_730573


namespace evaluate_x2_minus_y2_l730_730021

-- Definitions based on the conditions.
def x : ℝ
def y : ℝ
axiom cond1 : x + y = 12
axiom cond2 : 3 * x + y = 18

-- The main statement we need to prove.
theorem evaluate_x2_minus_y2 : x^2 - y^2 = -72 :=
by
  sorry

end evaluate_x2_minus_y2_l730_730021


namespace total_number_of_signals_l730_730279

-- Conditions definitions
def holes : Set (Fin 4) := {0, 1, 2, 3}
def valid_combinations : Set (Set (Fin 4)) := {{0, 2}, {0, 3}, {1, 3}}

-- Theorem statement
theorem total_number_of_signals :
  (∀ h1 h2 h3, h1 ∈ holes → h2 ∈ holes → h3 ∈ holes →
    ({h1, h2} ∈ valid_combinations → ({0, 1} ∩ {h1, h2} = ∅ ∧ {1, 2} ∩ {h1, h2} = ∅ ∧ {2, 3} ∩ {h1, h2} = ∅)) ∧
    ({h2, h3} ∈ valid_combinations → ({0, 1} ∩ {h2, h3} = ∅ ∧ {1, 2} ∩ {h2, h3} = ∅ ∧ {2, 3} ∩ {h2, h3} = ∅))) →
  3 * 4 = 12 :=
by sorry

end total_number_of_signals_l730_730279


namespace find_x_minus_y_l730_730858

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end find_x_minus_y_l730_730858


namespace find_inverse_value_l730_730584

def inverse_of_3_pow (y : ℝ) : ℝ := log y / log 3 

theorem find_inverse_value (H : ∀ x : ℝ, 3^(inverse_of_3_pow (3^x)) = x):
  inverse_of_3_pow 1 = 0 :=
by sorry

end find_inverse_value_l730_730584


namespace salon_customers_l730_730295

theorem salon_customers (C : ℕ) (H : C * 2 + 5 = 33) : C = 14 :=
by {
  sorry
}

end salon_customers_l730_730295


namespace median_room_number_l730_730870

-- Define the range of rooms
def room_numbers : List ℕ := List.range' 1 30  -- room numbers from 1 to 30

-- Define the unoccupied rooms
def unoccupied_rooms : Set ℕ := {17, 25}

-- Define the remaining rooms after removing the unoccupied ones
def remaining_rooms : List ℕ := (room_numbers.filter (λ n, n ∉ unoccupied_rooms)).sorted

-- Define the function to find the median of a list of numbers
def median (l : List ℕ) : ℕ :=
  if h : l.length % 2 = 0 then
    (l.get (l.length / 2 - 1) + l.get (l.length / 2)) / 2  -- average of middle two elements
  else
    l.get (l.length / 2)

-- Theorem to prove the median room number of the remaining mathematicians
theorem median_room_number : median remaining_rooms = 16.5 := 
by sorry

end median_room_number_l730_730870


namespace part1_part2_l730_730958

-- Part 1
theorem part1 {
  (a n : ℚ) : a + n = a → n = 0
} : 
(proof1_unit_element_addition_rationals : forall a : ℚ, (∃ n : ℚ, a + n = a) -> n = 0),
(proof1_unit_element_multiplication_rationals : forall a : ℚ, (∃ n : ℚ, a * n = a) -> n = 1),
(proof1_inverse_element_addition : 3 + n = 0 -> n = -3),
(proof1_inverse_element_multiplication : ¬∃ n : ℚ, 0 * n = 1)

-- Part 2
theorem part2 {
  (x e : ℚ) : (x * e = x) ∧ (∀ a b : ℚ, a * b = a + b - a * b),
  (m n : ℚ) : (m * n = 0) ∧ (m ≠ 1)
} : 
(proof2_unit_element_new_operation : (∀ x : ℚ, ∃ e : ℚ, x * e = x) -> e = 0),
(proof2_inverse_element_new_operation : ∀ m : ℚ, (∃ n : ℚ, m * n = 0 ∧ m ≠ 1) -> n = m / (m - 1))

end part1_part2_l730_730958


namespace find_n_equiv_l730_730370

theorem find_n_equiv :
  ∃ (n : ℕ), 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ (n = 3 ∨ n = 9) :=
by
  sorry

end find_n_equiv_l730_730370


namespace seating_arrangements_l730_730939

theorem seating_arrangements : 
  let family_members := 5 -- Total number of family members
  let driver_choices := 2 -- Choices for driver
  let front_passenger_choices := 4 -- Choices for the front passenger seat
  
  ∃ driver front_passenger backseat_arrangements,
    (driver_choices = 2) ∧
    (front_passenger_choices = 4) ∧
    (backseat_arrangements = 6) ∧
    (driver_choices * front_passenger_choices * backseat_arrangements = 48) :=
by
  -- These value assignments ensure conditions are acknowledged
  let family_members := 5
  let driver_choices := 2
  let front_passenger_choices := 4
  let backseat_arrangements := 3.choose 2 * 2.factorial
  use [driver_choices, front_passenger_choices, backseat_arrangements]
  sorry -- Proof is omitted

end seating_arrangements_l730_730939


namespace find_common_difference_l730_730008

variable (a_n : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Sum of the first n terms of an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℝ := n * (a_n 1 + a_n n) / 2

-- Difference of average sums condition
def average_difference_condition (Sn : ℕ → ℝ) : Prop := (Sn 2017 / 2017) - (Sn 17 / 17) = 100

theorem find_common_difference (h_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h_sum : ∀ n, S n = arithmetic_sum n d)
  (h_cond : average_difference_condition S) :
  d = 1 / 10 :=
by
  sorry

end find_common_difference_l730_730008


namespace smallest_abundant_number_l730_730287

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d ∣ n ∧ d < n)

def is_abundant (n : ℕ) : Prop :=
  n < proper_divisors n).sum id

theorem smallest_abundant_number : ∃ n : ℕ, is_abundant n ∧ ∀ m : ℕ, is_abundant m → n ≤ m :=
  ∃ n : ℕ, is_abundant n ∧ n = 12 :=
sorry

end smallest_abundant_number_l730_730287


namespace certain_number_approx_l730_730073

theorem certain_number_approx (x : ℝ) : 213 * 16 = 3408 → x * 2.13 = 0.3408 → x = 0.1600 :=
by
  intro h1 h2
  sorry

end certain_number_approx_l730_730073


namespace min_value_expr_l730_730537

-- Define that x and y are positive real numbers
variable (x y : ℝ)
variable (hx : 0 < x) (hy : 0 < y)

-- Define the expression
def expr (x y : ℝ) := (sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2))) / (x * y)

-- State the theorem to prove the minimum value of the expression is 3
theorem min_value_expr : (∃ (x y : ℝ), 0 < x → 0 < y → expr x y = 3) :=
sorry

end min_value_expr_l730_730537


namespace min_sum_of_abs_roots_l730_730822

def is_irrational_quadratic_trinomial (p q : ℤ) : Prop :=
  let D := p^2 - 4*q in
  ∃ α₁ α₂ : ℝ, α₁ ≠ α₂ ∧ irrational α₁ ∧ irrational α₂ ∧
    (α₁ + α₂ = -p) ∧ (α₁ * α₂ = q) ∧ 
    (D = (p^2 - 4*q)) ∧ int.irreducible D ∧ ¬int.exists_square D

theorem min_sum_of_abs_roots (p q : ℤ) : 
  is_irrational_quadratic_trinomial p q → 
  ∃ α₁ α₂ : ℝ, α₁ ≠ α₂ ∧ irrational α₁ ∧ irrational α₂ ∧ 
    (α₁ + α₂ = -p) ∧ (α₁ * α₂ = q) ∧
    (|α₁| + |α₂| = sqrt 5) :=
sorry

end min_sum_of_abs_roots_l730_730822


namespace det_matrixB_l730_730928

def matrixB (b e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![b, 2], ![-3, e]]

def matrixI : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

noncomputable def matrixBinv (b e : ℝ) (h : b * e + 6 ≠ 0) : Matrix (Fin 2) (Fin 2) ℝ :=
  ((be+6)⁻¹ : _) • ![![e, -2], ![3, b]]

def condition (b e : ℝ) (h : b * e + 6 ≠ 0) : Prop :=
  matrixB b e + matrixBinv b e h = matrixI

theorem det_matrixB (b e : ℝ) (h : b * e + 6 ≠ 0) (hc : condition b e h) : 
  Matrix.det (matrixB b e) = 1 :=
sorry

end det_matrixB_l730_730928


namespace perfect_squares_in_interval_l730_730490

theorem perfect_squares_in_interval (s : Set Int) (h1 : ∃ a : Nat, ∀ x ∈ s, a^4 ≤ x ∧ x ≤ (a+9)^4)
                                     (h2 : ∃ b : Nat, ∀ x ∈ s, b^3 ≤ x ∧ x ≤ (b+99)^3) :
  ∃ c : Nat, c ≥ 2000 ∧ ∀ x ∈ s, x = c^2 :=
sorry

end perfect_squares_in_interval_l730_730490


namespace inequality_solution_l730_730256

theorem inequality_solution (x : ℝ) :
  (9.269 * (1 / log (1/2) (sqrt (x + 3))) ≤ 1 / log (1/2) (x + 1)) ↔ (x ∈ Set.Ioo (-1) 0 ∪ Set.Ici 1) := by
  sorry

end inequality_solution_l730_730256


namespace number_of_valid_monograms_l730_730550

-- Define the set of vowels
def vowels : finset char := {'A', 'E', 'I', 'O', 'U'}

-- Define the problem conditions
def conditions (first middle last : char) : Prop :=
  first ∈ vowels ∧
  middle ∈ vowels ∧
  first < middle ∧
  last = 'X'

-- Statement of the problem: Prove the number of valid monograms is 10
theorem number_of_valid_monograms : 
  finset.card {p : char × char × char | conditions p.1 p.2 p.3} = 10 :=
sorry

end number_of_valid_monograms_l730_730550


namespace increasing_f_when_a_eq_2_range_of_a_l730_730825

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log x - (a * (x - 1)) / (x + 1)

theorem increasing_f_when_a_eq_2 : 
  ∀ x : ℝ, x > 0 → (∀ a : ℝ, a = 2 → (f x a) ≥ 0) :=
by
  intros x hx a ha,
  sorry

theorem range_of_a :
  ∀ a : ℝ, (-∞ < a) ∧ (a ≤ 2) ↔ ∀ x : ℝ, x ≥ 1 → f x a ≥ 0 :=
by
  intros a,
  split,
  { intros h x hx,
    sorry },
  { intros h,
    sorry }

end increasing_f_when_a_eq_2_range_of_a_l730_730825


namespace different_color_socks_l730_730466

def total_socks := 15
def white_socks := 6
def brown_socks := 5
def blue_socks := 4

theorem different_color_socks (total : ℕ) (white : ℕ) (brown : ℕ) (blue : ℕ) :
  total = white + brown + blue →
  white ≠ 0 → brown ≠ 0 → blue ≠ 0 →
  (white * brown + brown * blue + white * blue) = 74 :=
by
  intros
  -- proof goes here
  sorry

end different_color_socks_l730_730466


namespace evaluate_x2_minus_y2_l730_730018

-- Definitions based on the conditions.
def x : ℝ
def y : ℝ
axiom cond1 : x + y = 12
axiom cond2 : 3 * x + y = 18

-- The main statement we need to prove.
theorem evaluate_x2_minus_y2 : x^2 - y^2 = -72 :=
by
  sorry

end evaluate_x2_minus_y2_l730_730018


namespace trapezoid_mn_l730_730801

variable (a b : ℝ)

theorem trapezoid_mn (ABCD : Trapezoid)
  (AD BC : ABCD.base)
  (h_AD : AD = a)
  (h_BC : BC = b)
  (M N : ABCD.side)
  (MN_par_AD_BC : MN.parallel AD BC)
  (O : Intersection AC MN)
  (equal_areas : Area (Triangle AMO) = Area (Triangle CNO)) :
  MN = Real.sqrt (a * b) := sorry

end trapezoid_mn_l730_730801


namespace john_money_left_l730_730116

-- Definitions for initial conditions
def initial_amount : ℤ := 100
def cost_roast : ℤ := 17
def cost_vegetables : ℤ := 11

-- Total spent calculation
def total_spent : ℤ := cost_roast + cost_vegetables

-- Remaining money calculation
def remaining_money : ℤ := initial_amount - total_spent

-- Theorem stating that John has €72 left
theorem john_money_left : remaining_money = 72 := by
  sorry

end john_money_left_l730_730116


namespace find_value_sqrt2_l730_730445

-- Define the power function f(x) = x^α
def power_function (x : ℝ) (α : ℝ) : ℝ :=
  x ^ α

-- Given condition: the function passes through the point P(2, 4)
def passes_through_point (α : ℝ) : Prop :=
  power_function 2 α = 4

-- Theorem: Find f(√2) when the power function passes through P(2, 4)
theorem find_value_sqrt2 :
  ∃ α : ℝ, passes_through_point α ∧ power_function (Real.sqrt 2) α = 2 :=
by sorry

end find_value_sqrt2_l730_730445


namespace range_combined_set_l730_730178

def isPrime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def setX := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ isPrime n}

def setY := {n : ℕ | n > 0 ∧ n < 100 ∧ 7 ∣ n ∧ n % 2 = 1}

def combinedSet := setX ∪ setY

theorem range_combined_set : 
  let min_comb := 7 in
  let max_comb := 97 in
  set.min' combinedSet sorry = min_comb ∧ set.max' combinedSet sorry = max_comb → 
  max_comb - min_comb = 86 :=
by
  -- Proof to be filled in
  sorry

end range_combined_set_l730_730178


namespace part1_part2_part3_l730_730879

/-
We declare that we are dealing with variables in the real number domain for this problem.
-/

-- Definitions based on conditions
def P (m : ℝ) := (6-3*m, m+1)
def distance_to_y_axis (m : ℝ) := abs (6-3*m)

-- Part (1)
theorem part1 (m : ℝ) : distance_to_y_axis m = 2 → (m = 4/3 ∨ m = 8/3) :=
by
  sorry

-- Part (2)
theorem part2 (m : ℝ) : (fst (P m) = snd (P m)) → (P m = (9/4, 9/4)) :=
by
  sorry

-- Part (3)
theorem part3 (Q : ℝ × ℝ) : (P 5/4 = (9/4, 9/4)) → (snd Q = 9/4) → (abs (fst Q - 9/4) = 3) → (fst Q < 0) → (Q = (-3/4, 9/4)) :=
by
  sorry

end part1_part2_part3_l730_730879


namespace total_votes_election_l730_730096

theorem total_votes_election (V : ℝ)
    (h1 : 0.55 * 0.8 * V + 2520 = 0.8 * V)
    (h2 : 0.36 > 0) :
    V = 7000 :=
  by
  sorry

end total_votes_election_l730_730096


namespace minimum_cost_l730_730877

noncomputable def total_cost (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + 0.5 * x

theorem minimum_cost : 
  (∃ x : ℝ, x = 55 ∧ total_cost x = 57.5) :=
  sorry

end minimum_cost_l730_730877


namespace solution_regions_l730_730842

noncomputable def solutionSet (x y : ℝ) : Prop :=
  (y^2 - Real.arcsin (Real.sin x)) * 
  (y^2 - Real.arcsin (Real.sin (x + π / 6))) * 
  (y^2 - Real.arcsin (Real.sin (x - π / 6))) < 0

theorem solution_regions :
  ∀ x y : ℝ,
  (y = Real.arcsin (Real.sin x) ∨ y = -Real.arcsin (Real.sin x) ∨
   y = Real.arcsin (Real.sin (x + π / 6)) ∨ y = -Real.arcsin (Real.sin (x + π / 6)) ∨
   y = Real.arcsin (Real.sin (x - π / 6)) ∨ y = -Real.arcsin (Real.sin (x - π / 6))) →
  solutionSet x y → 
  -- Specify the exact regions where the inequality holds true
  sorry

end solution_regions_l730_730842


namespace no_odd_cycle_in_graph_l730_730864
open Classical

noncomputable def group_condition (G : SimpleGraph (Fin 12)) : Prop :=
  ∀ (S : Finset (Fin 12)) (hS : S.card = 9), 5 ≤ S.filter (λ (x : Fin 12), S.filter (λ y, G.Adj x y).card ≥ 5).card

theorem no_odd_cycle_in_graph (G : SimpleGraph (Fin 12)) (H : group_condition G) : ¬(∃ C : List (Fin 12), G.IsOddCycle C) :=
begin
  sorry
end

end no_odd_cycle_in_graph_l730_730864


namespace configuration_of_points_l730_730768

-- Define a type for points
structure Point :=
(x : ℝ)
(y : ℝ)

-- Assuming general position in the plane
def general_position (points : List Point) : Prop :=
  -- Add definition of general position, skipping exact implementation
  sorry

-- Define the congruence condition
def triangles_congruent (points : List Point) : Prop :=
  -- Add definition of the congruent triangles condition
  sorry

-- Define the vertices of two equilateral triangles inscribed in a circle
def two_equilateral_triangles (points : List Point) : Prop :=
  -- Add definition to check if points form two equilateral triangles in a circle
  sorry

theorem configuration_of_points (points : List Point) (h6 : points.length = 6) :
  general_position points →
  triangles_congruent points →
  two_equilateral_triangles points :=
by
  sorry

end configuration_of_points_l730_730768


namespace vanya_more_heads_probability_l730_730648

-- Define binomial distributions for Vanya and Tanya's coin flips
def binomial (n : ℕ) (p : ℝ) : distribution :=
  λ k : ℕ, choose n k * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the random variables X_V and X_T
def X_V (n : ℕ) : distribution := binomial (n + 1) (1 / 2)
def X_T (n : ℕ) : distribution := binomial n (1 / 2)

-- Define the probability of Vanya getting more heads than Tanya
def probability_vanya_more_heads (n : ℕ) : ℝ :=
  ∑ (k : ℕ) in finset.range (n + 2), ∑ (j : ℕ) in finset.range (k), X_V n k * X_T n j

-- The main theorem to prove
theorem vanya_more_heads_probability (n : ℕ) : probability_vanya_more_heads n = 1 / 2 :=
begin
  sorry,
end

end vanya_more_heads_probability_l730_730648


namespace three_coloring_four_coloring_l730_730925

-- Step 1: Define the conditions
def D_n (n : ℤ) : set (ℤ × ℤ) := {p | abs p.1 ≤ n ∧ abs p.2 ≤ n}

-- Step 2: First Proof Problem
theorem three_coloring (n : ℤ) (h_n : n > 1) :
  ∀ (coloring : (ℤ × ℤ) → ℕ),
    (∀ p ∈ D_n n, coloring p < 3) →
    ∃ (p1 p2 : ℤ × ℤ), p1 ≠ p2 ∧ p1 ∈ D_n n ∧ p2 ∈ D_n n ∧ coloring p1 = coloring p2 ∧ 
    (∀ (p3 : ℤ × ℤ), (p3 ∈ D_n n ∧ 
    (((p1.2 - p2.2) * (p3.1 - p1.1) = (p1.1 - p2.1) * (p3.2 - p1.2)) ↔ p3 = p1 ∨ p3 = p2))) := 
sorry

-- Step 3: Second Proof Problem
theorem four_coloring (n : ℤ) (h_n : n > 1) :
  ∃ (coloring : (ℤ × ℤ) → ℕ),
    (∀ p ∈ D_n n, coloring p < 4) ∧ 
    (∀ (p1 p2 : ℤ × ℤ), p1 ≠ p2 ∧ p1 ∈ D_n n ∧ p2 ∈ D_n n →
      ((∀ (p3 : ℤ × ℤ), p3 ∈ D_n n ∧ 
      (((p1.2 - p2.2) * (p3.1 - p1.1) = (p1.1 - p2.1) * (p3.2 - p1.2)) ↔ p3 = p1 ∨ p3 = p2)) → 
      coloring p1 ≠ coloring p2)) :=
sorry

end three_coloring_four_coloring_l730_730925


namespace alex_age_div_M_l730_730312

variable {A M : ℕ}

-- Definitions provided by the conditions
def alex_age_current : ℕ := A
def sum_children_age : ℕ := A
def alex_age_M_years_ago (A M : ℕ) : ℕ := A - M
def children_age_M_years_ago (A M : ℕ) : ℕ := A - 4 * M

-- Given condition as a hypothesis
def condition (A M : ℕ) := alex_age_M_years_ago A M = 3 * children_age_M_years_ago A M

-- The theorem to prove
theorem alex_age_div_M (A M : ℕ) (h : condition A M) : A / M = 11 / 2 := 
by
  -- This is a placeholder for the actual proof.
  sorry

end alex_age_div_M_l730_730312


namespace largest_five_digit_number_l730_730772

theorem largest_five_digit_number (digits : List ℕ) (h_set : digits = [0, 4, 6, 7, 8]) (h_unique : digits.nodup) :
  ∃ x : ℕ, x = 87640 ∧ ∀ y ∈ permutations digits, y ≠ 87640 → to_nat y < 87640 :=
by
  -- proof goes here
  sorry

end largest_five_digit_number_l730_730772


namespace max_value_exponent_l730_730784

theorem max_value_exponent {a b : ℝ} (h : 0 < b ∧ b < a ∧ a < 1) :
  max (max (a^b) (b^a)) (max (a^a) (b^b)) = a^b :=
sorry

end max_value_exponent_l730_730784


namespace reciprocal_twice_eq_initial_l730_730935

theorem reciprocal_twice_eq_initial : ∀ x : ℚ, x = 1 / 16 → (1 / (1 / x)) = x :=
by 
  intros x hx
  rw hx
  norm_num
  sorry -- placeholder for manual proof steps if necessary.

end reciprocal_twice_eq_initial_l730_730935


namespace candy_distribution_even_l730_730349

theorem candy_distribution_even (total_candies : ℕ) (num_sisters : ℕ)
  (h_total_candies : total_candies = 30)
  (h_num_sisters : num_sisters = 5) :
  total_candies % num_sisters = 0 :=
by 
  rw [h_total_candies, h_num_sisters]
  norm_num

end candy_distribution_even_l730_730349


namespace count_fractions_single_digit_less_than_1_l730_730844

theorem count_fractions_single_digit_less_than_1 : 
  let single_digit_nat := {n : ℕ | n < 10}
  let valid_fractions := {f : ℚ | 
    ∃ (n d : ℕ), n < d ∧ d ∈ single_digit_nat ∧ n ∈ single_digit_nat ∧ f = (n : ℚ) / (d : ℚ)}
  is_number_of_fractions := valid_fractions.card = 36 :=
by
  sorry

end count_fractions_single_digit_less_than_1_l730_730844


namespace min_b1_b2_sum_l730_730988

def sequence_relation (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (3 * b n + 4073) / (2 + b (n + 1))

theorem min_b1_b2_sum (b : ℕ → ℕ) (h_seq : sequence_relation b) 
  (h_b1_pos : b 1 > 0) (h_b2_pos : b 2 > 0) :
  b 1 + b 2 = 158 :=
sorry

end min_b1_b2_sum_l730_730988


namespace smallest_remaining_number_111_smallest_remaining_number_110_l730_730891

theorem smallest_remaining_number_111 : (n = 111) → minimal_after_operations (1, 111) = 0 := by
  sorry

theorem smallest_remaining_number_110 : (n = 110) → minimal_after_operations (1, 110) = 1 := by
  sorry

end smallest_remaining_number_111_smallest_remaining_number_110_l730_730891


namespace salon_customers_l730_730296

theorem salon_customers (C : ℕ) (H : C * 2 + 5 = 33) : C = 14 :=
by {
  sorry
}

end salon_customers_l730_730296


namespace product_maximized_l730_730354

noncomputable def maximize_product (N : ℝ) (x : ℝ) : Prop :=
  let P := x * (N - x) in
  ∀ y, P ≤ y * (N - y)

theorem product_maximized (N : ℝ) : maximize_product N (N / 2) :=
by
  sorry

end product_maximized_l730_730354


namespace vertical_asymptotes_count_l730_730847

-- Define the function y = (x-1) / (x^2 + 6x - 7)
def function (x : ℝ) : ℝ := (x - 1) / (x^2 + 6x - 7)

-- State the theorem for the number of vertical asymptotes of the function
theorem vertical_asymptotes_count : ∃ x : ℝ, ((function x) = ⊤) :=
  sorry

end vertical_asymptotes_count_l730_730847


namespace crease_length_l730_730694

theorem crease_length (A B C : ℝ) (h1 : A = 5) (h2 : B = 12) (h3 : C = 13) : ∃ D, D = 6.5 :=
by
  sorry

end crease_length_l730_730694


namespace calculation_eq_l730_730732

theorem calculation_eq : (3 + 1) * (3^2 + 1) * (3^4 + 1) * ... * (3^64 + 1) = (3^128 - 1) / 2 :=
by
  sorry

end calculation_eq_l730_730732


namespace robert_can_read_books_l730_730174

theorem robert_can_read_books (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 →
  book_pages = 360 →
  total_hours = 8 →
  total_hours / (book_pages / pages_per_hour) = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end robert_can_read_books_l730_730174


namespace profit_ratio_l730_730597

-- Definitions based on the given conditions
noncomputable def p_investment_ratio := 7
noncomputable def q_investment_ratio := 5
noncomputable def p_investment_time := 5
noncomputable def q_investment_time := 12

-- Theorem stating the profit ratio
theorem profit_ratio (x : ℕ) :
  let p_profit := p_investment_ratio * p_investment_time * x,
      q_profit := q_investment_ratio * q_investment_time * x
  in (p_profit : ℕ) / (q_profit : ℕ) = 7 / 12 := sorry

end profit_ratio_l730_730597


namespace a_20_value_l730_730053

open Nat

def sequence_a : ℕ → ℝ
| 0     := 0
| (n+1) := (sequence_a n - real.sqrt 3) / (1 + real.sqrt 3 * sequence_a n)

theorem a_20_value : sequence_a 20 = -real.sqrt 3 := 
sorry

end a_20_value_l730_730053


namespace simplify_expression_l730_730566

theorem simplify_expression (q : ℤ) : 
  (((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6)) = 76 * q - 44 := by
  sorry

end simplify_expression_l730_730566


namespace smallest_next_divisor_l730_730517

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000) 
  (h2 : is_even m) 
  (h3 : is_divisor 171 m)
  : ∃ k, k > 171 ∧ k = 190 ∧ is_divisor k m := 
by
  sorry

end smallest_next_divisor_l730_730517


namespace cos_monotonically_increasing_range_l730_730477

theorem cos_monotonically_increasing_range (a : ℝ) : 
  (∀ x1 x2 : ℝ, -π ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ a → cos x1 ≤ cos x2) ↔ (a ∈ Icc (-π) 0) := 
by 
  sorry

end cos_monotonically_increasing_range_l730_730477


namespace chrom_replication_not_in_prophase_I_l730_730240

-- Definitions for the conditions
def chrom_replication (stage : String) : Prop := 
  stage = "Interphase"

def chrom_shortening_thickening (stage : String) : Prop := 
  stage = "Prophase I"

def pairing_homologous_chromosomes (stage : String) : Prop := 
  stage = "Prophase I"

def crossing_over (stage : String) : Prop :=
  stage = "Prophase I"

-- Stating the theorem
theorem chrom_replication_not_in_prophase_I :
  chrom_replication "Interphase" ∧ 
  chrom_shortening_thickening "Prophase I" ∧ 
  pairing_homologous_chromosomes "Prophase I" ∧ 
  crossing_over "Prophase I" → 
  ¬ chrom_replication "Prophase I" := 
by
  sorry

end chrom_replication_not_in_prophase_I_l730_730240


namespace chess_tournament_l730_730993

-- Define the number of chess amateurs
def num_amateurs : ℕ := 5

-- Define the number of games each amateur plays
def games_per_amateur : ℕ := 4

-- Define the total number of chess games possible
def total_games : ℕ := num_amateurs * (num_amateurs - 1) / 2

-- The main statement to prove
theorem chess_tournament : total_games = 10 := 
by
  -- here should be the proof, but according to the task, we use sorry to skip
  sorry

end chess_tournament_l730_730993


namespace balloons_initial_count_l730_730749

theorem balloons_initial_count (B : ℕ) (G : ℕ) : ∃ G : ℕ, B = 7 * G + 4 := sorry

end balloons_initial_count_l730_730749


namespace triangle_sin_cos_identity_l730_730035

theorem triangle_sin_cos_identity (A B C : ℝ) (h : A + B + C = π) : 
  sin ( (B + C) / 2 ) = cos (A / 2) := sorry

end triangle_sin_cos_identity_l730_730035


namespace find_a_from_csc_function_l730_730722

/-- Given that the function y = a * csc(b * x) has a minimum positive value of 4, prove that a = 4. -/
theorem find_a_from_csc_function (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (h : ∀ x : ℝ, b * x ≠ 0 → 0 < a * Real.csc(b * x) → 4 ≤ a * Real.csc(b * x) ) :
  a = 4 := 
sorry

end find_a_from_csc_function_l730_730722


namespace age_in_1988_equals_sum_of_digits_l730_730481

def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

def age_in_1988 (birth_year : ℕ) : ℕ := 1988 - birth_year

def sum_of_digits (x y : ℕ) : ℕ := 1 + 9 + x + y

theorem age_in_1988_equals_sum_of_digits (x y : ℕ) (h0 : 0 ≤ x) (h1 : x ≤ 9) (h2 : 0 ≤ y) (h3 : y ≤ 9) 
  (h4 : age_in_1988 (birth_year x y) = sum_of_digits x y) :
  age_in_1988 (birth_year x y) = 22 :=
by {
  sorry
}

end age_in_1988_equals_sum_of_digits_l730_730481


namespace resulting_polygon_sides_l730_730345

theorem resulting_polygon_sides :
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let decagon_sides := 10
  let shared_square_decagon := 2
  let shared_between_others := 2 * 5 -- 2 sides shared for pentagon to nonagon
  let total_shared_sides := shared_square_decagon + shared_between_others
  let total_unshared_sides := 
    square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides + nonagon_sides + decagon_sides
  total_unshared_sides - total_shared_sides = 37 := by
  sorry

end resulting_polygon_sides_l730_730345


namespace pine_cone_weight_on_roof_l730_730306

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end pine_cone_weight_on_roof_l730_730306


namespace sum_binomial_identity_l730_730955

-- Define the binomial coefficient function in Lean
def binomial_coefficient (n k : ℕ) : ℕ :=
nat.choose n k

theorem sum_binomial_identity (n : ℕ) :
  (∑ k in finset.range (n + 1), ((4 ^ k) / (k + 1)) * (binomial_coefficient (2 * n + 1) (2 * k))) =
  ((4 * n + 3) * (3 ^ (2 * n + 2) - 1)) / (4 * (2 * n + 2) * (2 * n + 3)) :=
sorry

end sum_binomial_identity_l730_730955


namespace inequality_solution_l730_730183

theorem inequality_solution (a x : ℝ) :
  (a = 0 → x ≤ -1) ∧
  (0 < a → x ≤ -1 ∨ x ≥ 2 / a) ∧
  (-2 < a ∧ a < 0 → 2 / a ≤ x ∧ x ≤ -1) ∧
  (a = -2 → x = -1) ∧
  (a < -2 → -1 ≤ x ∧ x ≤ 2 / a) ↔ 
  ax^2 + (a - 2)x - 2 ≥ 0 :=
sorry

end inequality_solution_l730_730183


namespace parabola_properties_l730_730577

def parabola (a b x : ℝ) : ℝ :=
  a * x ^ 2 + b * x - 4

theorem parabola_properties :
  ∃ (a b : ℝ), (a = 2) ∧ (b = 2) ∧
  parabola a b (-2) = 0 ∧ 
  parabola a b (-1) = -4 ∧ 
  parabola a b 0 = -4 ∧ 
  parabola a b 1 = 0 ∧ 
  parabola a b 2 = 8 ∧ 
  parabola a b (-3) = 8 ∧ 
  (0, -4) ∈ {(x, y) | ∃ a b, y = parabola a b x} :=
sorry

end parabola_properties_l730_730577


namespace germs_per_dish_l730_730880

theorem germs_per_dish 
  (num_germs : ℝ) 
  (num_dishes : ℝ) 
  (h1 : num_germs = 0.036 * 10^5) 
  (h2 : num_dishes = 75000 * 10^(-3)) : 
  (num_germs / num_dishes) = 48 := 
by 
  sorry

end germs_per_dish_l730_730880


namespace proposition_p_l730_730084

variable (x : ℝ)

-- Define condition
def negation_of_p : Prop := ∃ x, x < 1 ∧ x^2 < 1

-- Define proposition p
def p : Prop := ∀ x, x < 1 → x^2 ≥ 1

-- Theorem statement
theorem proposition_p (h : negation_of_p) : (p) :=
sorry

end proposition_p_l730_730084


namespace corn_height_after_three_weeks_l730_730727

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l730_730727


namespace wallpaper_removal_time_l730_730756

theorem wallpaper_removal_time (time_per_wall : ℕ) (dining_room_walls_remaining : ℕ) (living_room_walls : ℕ) :
  time_per_wall = 2 → dining_room_walls_remaining = 3 → living_room_walls = 4 → 
  time_per_wall * (dining_room_walls_remaining + living_room_walls) = 14 :=
by
  sorry

end wallpaper_removal_time_l730_730756


namespace minimum_value_of_k_minus_b_l730_730425

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := x + (1 / x) + (1 / 2) * x^2 - Real.log x + 1

theorem minimum_value_of_k_minus_b : ∃ x : ℝ,  (x > 0) ∧ g 1 = 7 / 2 :=
begin
  use 1,
  split,
  { norm_num, },
  { norm_num, }
end

end minimum_value_of_k_minus_b_l730_730425


namespace common_point_condition_l730_730265

theorem common_point_condition
  (P A B C A1 B1 C1 A2 B2 C2 : Point)
  (hP_inside_ABC : P ∈ triangle ABC)
  (hA1_on_BC : A1 ∈ line_segment BC)
  (hB1_on_CA : B1 ∈ line_segment CA)
  (hC1_on_AB : C1 ∈ line_segment AB)
  (hPA_A1 : collinear P A A1)
  (hPB_B1 : collinear P B B1)
  (hPC_C1 : collinear P C C1)
  (hA2_on_BC : A2 ∈ line_segment BC)
  (hB2_on_CA : B2 ∈ line_segment CA)
  (hC2_on_AB : C2 ∈ line_segment AB)
  (hB1C1_A2 : collinear B1 C1 A2)
  (hC1A1_B2 : collinear C1 A1 B2)
  (hA1B1_C2 : collinear A1 B1 C2)
  (W1 : Circle) (hW1 : diameter W1 = A1A2)
  (W2 : Circle) (hW2 : diameter W2 = B1B2)
  (W3 : Circle) (hW3 : diameter W3 = C1C2) :
  (∃ K : Point, K ∈ W1 ∧ K ∈ W2) ↔ 
  (∃ K : Point, K ∈ W1 ∧ K ∈ W2 ∧ K ∈ W3) :=
sorry

end common_point_condition_l730_730265


namespace one_fourth_of_2_pow_30_eq_2_pow_x_l730_730473

theorem one_fourth_of_2_pow_30_eq_2_pow_x (x : ℕ) : (1 / 4 : ℝ) * (2:ℝ)^30 = (2:ℝ)^x → x = 28 := by
  sorry

end one_fourth_of_2_pow_30_eq_2_pow_x_l730_730473


namespace cookies_needed_l730_730328

theorem cookies_needed {n c : ℕ} {r : ℝ} {k : ℕ} (hn : n = 150) (hc : c = 3) (hr : r = 0.60) (hk : k = 20) :
  let needed_cookies := (n * c : ℕ) * (r : ℝ)
  let total_recipes := ⌈needed_cookies / k⌉ 
  total_recipes = 14 :=
by
  sorry

end cookies_needed_l730_730328


namespace distinct_triangles_from_chord_intersections_l730_730156

theorem distinct_triangles_from_chord_intersections :
  let points := 9
  let chords := (points.choose 2)
  let intersections := (points.choose 4)
  let triangles := (points.choose 6)
  (chords > 0 ∧ intersections > 0 ∧ triangles > 0) →
  triangles = 84 :=
by
  intros
  sorry

end distinct_triangles_from_chord_intersections_l730_730156


namespace sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l730_730395

open Real

noncomputable def problem_conditions (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧
  cos α = 3/5 ∧ cos (β + α) = 5/13

theorem sin_beta_value 
  {α β : ℝ} (h : problem_conditions α β) : 
  sin β = 16 / 65 :=
sorry

theorem sin2alpha_over_cos2alpha_plus_cos2alpha_value
  {α β : ℝ} (h : problem_conditions α β) : 
  (sin (2 * α)) / (cos α^2 + cos (2 * α)) = 12 :=
sorry

end sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l730_730395


namespace max_min_cos_sin_cos_l730_730419

theorem max_min_cos_sin_cos (x y z : ℝ)
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ real.pi / 12) (h4 : x + y + z = real.pi / 2) :
  (∃ min_val, cos x * sin y * cos z = min_val ∧ min_val = 1/8) ∧
  (∃ max_val, cos x * sin y * cos z = max_val ∧ max_val = 1/4 + real.sqrt 3 / 8) :=
by
  sorry

end max_min_cos_sin_cos_l730_730419


namespace email_difference_l730_730895

def morning_emails_early : ℕ := 10
def morning_emails_late : ℕ := 15
def afternoon_emails_early : ℕ := 7
def afternoon_emails_late : ℕ := 12

theorem email_difference :
  (morning_emails_early + morning_emails_late) - (afternoon_emails_early + afternoon_emails_late) = 6 :=
by
  sorry

end email_difference_l730_730895


namespace solid_ball_performance_l730_730693

theorem solid_ball_performance :
    ∃ (x : ℝ), y = - (1 / 12 : ℝ) * x^2 + (2 / 3 : ℝ) * x + (5 / 3 : ℝ) ∧ y = 0 ∧ x = 10 :=
by
  sorry

end solid_ball_performance_l730_730693


namespace integral_value_l730_730222

theorem integral_value : ∫ x in 0..1, (3 * x ^ 2 - 1 / 2) = 1 / 2 := 
  sorry

end integral_value_l730_730222


namespace inverse_matrices_l730_730453

noncomputable def A (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 1], ![1, 2]]
noncomputable def B (b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![2/3, b], ![-1/3, 2/3]]

theorem inverse_matrices (a b : ℝ) (h : A a ⬝ B b = 1) : a = 2 ∧ b = -1/3 := by
  sorry

end inverse_matrices_l730_730453


namespace false_implications_l730_730480

-- Definitions based on the given condition
variable (Book : Type) (Library : Set Book)
variable [Nonempty Library]

def EveryBookCheckedOut (checkedOut : Book → Prop) : Prop :=
  ∀ b ∈ Library, checkedOut b

-- Translated proof problem
theorem false_implications (checkedOut : Book → Prop) :
  ¬ EveryBookCheckedOut Library checkedOut →
  (∃ b ∈ Library, ¬ checkedOut b) ∧ ¬ EveryBookCheckedOut Library checkedOut :=
by
  sorry

end false_implications_l730_730480


namespace minimum_value_of_k_minus_b_l730_730426

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := x + (1 / x) + (1 / 2) * x^2 - Real.log x + 1

theorem minimum_value_of_k_minus_b : ∃ x : ℝ,  (x > 0) ∧ g 1 = 7 / 2 :=
begin
  use 1,
  split,
  { norm_num, },
  { norm_num, }
end

end minimum_value_of_k_minus_b_l730_730426


namespace cannot_derive_xn_minus_1_l730_730506

open Polynomial

noncomputable def f := (X^3 - 3 * X^2 + 5 : Polynomial ℝ)
noncomputable def g := (X^2 - 4 * X : Polynomial ℝ)

lemma derivatives_zero_at_two (p : Polynomial ℝ) : (p.derivative.eval 2 = 0) :=
  by sorry

theorem cannot_derive_xn_minus_1 (p : Polynomial ℝ) (n : ℕ) :
  (p = f ∨ p = g ∨ ∃ (a b : Polynomial ℝ), a = f ∧ b = g ∧ p = a + b ∨ p = a - b ∨ p = a * b ∨ p = a.eval b ∨ ∃ c : ℝ, p = c • a) →
  p.derivative.eval 2 = 0 → n > 0 → (X^n - 1 : Polynomial ℝ).derivative.eval 2 ≠ 0 →
  p ≠ X^n - 1 :=
begin
  intros hp hp_deriv hn hn_deriv,
  sorry
end

end cannot_derive_xn_minus_1_l730_730506


namespace solution_set_f_greater_than_2x_plus4_l730_730580

theorem solution_set_f_greater_than_2x_plus4 {f : ℝ → ℝ}
  (domain_f : ∀ x : ℝ, f x ∈ ℝ)
  (h1 : f (-1) = 2)
  (h2 : ∀ x : ℝ, has_deriv_at f (≥ 2) x) :
  { x : ℝ | f x > 2 * x + 4 } = { x : ℝ | x > -1 } :=
sorry

end solution_set_f_greater_than_2x_plus4_l730_730580


namespace kite_IO_eq_OJ_l730_730106

-- Definitions and Conditions
variables (A B C D E F G H I J O : Type)
variable [plane_geometry A B C D E F G H I J O] -- placeholder for actual geometric definitions

-- Given conditions in the problem
variables (eq_AB_AD : A = D) (eq_BC_CD : B = C)
variables (O_intersection : ∃ O, is_intersection (diagonal A C) (diagonal B D) O)
variables (line_through_O_EF : is_line_through (E F) O) 
variables (line_through_O_GH : is_line_through (G H) O)
variables (intersection_GF_D : ∃ I, intersect (line G F) (BD) I)
variables (intersection_EH_D : ∃ J, intersect (line E H) (BD) J)

-- Proof statement
theorem kite_IO_eq_OJ
  (eq_AB_AD : A = D)
  (eq_BC_CD : B = C)
  (O_intersection : ∃ O, is_intersection (diagonal A C) (diagonal B D) O)
  (line_through_O_EF : is_line_through (E F) O) 
  (line_through_O_GH : is_line_through (G H) O)
  (intersection_GF_D : ∃ I, intersect (line G F) (BD) I)
  (intersection_EH_D : ∃ J, intersect (line E H) (BD) J):
  distance O I = distance O J :=
sorry

end kite_IO_eq_OJ_l730_730106


namespace line_intersects_but_not_through_center_l730_730599

-- Define the center and radius of the circle
def circle_center := (1 : ℝ, 2 : ℝ)
def circle_radius := 4

-- Define the line equation
def line (x y : ℝ) := 4 * x + 3 * y = 0

-- Define the distance formula from a point to a line
def distance_from_point_to_line (a b c x1 y1 : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a ^ 2 + b ^ 2)

-- Define the calculation of distance from the center to the line
def distance_center_line : ℝ := 
  distance_from_point_to_line 4 3 0 (1 : ℝ) (2 : ℝ)

-- The main theorem stating that the line intersects but does not pass through the center of the circle
theorem line_intersects_but_not_through_center :
  distance_center_line = 2 ∧ distance_center_line < circle_radius := by
  sorry

end line_intersects_but_not_through_center_l730_730599


namespace complex_distance_to_origin_l730_730881

open Complex

-- Define the complex number given in the problem.
noncomputable def given_complex : ℂ :=
  (i^2016 - 2 * i^2014) / (2 - i)^2

-- Define the function to calculate the distance from a complex number to the origin.
noncomputable def distance_to_origin (z : ℂ) : ℝ :=
  Complex.abs z

-- State the theorem to prove the distance to the origin.
theorem complex_distance_to_origin : 
  distance_to_origin given_complex = 3 / 5 :=
sorry

end complex_distance_to_origin_l730_730881


namespace remainder_of_sum_div_11_is_9_l730_730375

def seven_times_ten_pow_twenty : ℕ := 7 * 10 ^ 20
def two_pow_twenty : ℕ := 2 ^ 20
def sum : ℕ := seven_times_ten_pow_twenty + two_pow_twenty

theorem remainder_of_sum_div_11_is_9 : sum % 11 = 9 := by
  sorry

end remainder_of_sum_div_11_is_9_l730_730375


namespace unique_solution_quadratic_l730_730765

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃ x, q * x^2 - 18 * x + 8 = 0 ∧ ∀ y, q * y^2 - 18 * y + 8 = 0 → y = x) →
  q = 81 / 8 :=
by
  sorry

end unique_solution_quadratic_l730_730765


namespace evaluate_x_squared_minus_y_squared_l730_730028

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l730_730028


namespace cannot_obtain_xn_minus_1_l730_730503

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem cannot_obtain_xn_minus_1 :
  ∀ (n : ℕ), n > 0 → ¬∃ (h : ℝ → ℝ) (c : ℝ), h = (x^n - 1) where
    (h = f + g ∨ h = f - g ∨ h = f * g ∨ h = f ∘ g ∨ h = c • f) :=
begin
  sorry
end

end cannot_obtain_xn_minus_1_l730_730503


namespace laura_total_cost_l730_730123

def salad_cost : ℝ := 3
def beef_cost : ℝ := 2 * salad_cost
def potato_cost : ℝ := salad_cost / 3
def juice_cost : ℝ := 1.5

def total_salad_cost : ℝ := 2 * salad_cost
def total_beef_cost : ℝ := 2 * beef_cost
def total_potato_cost : ℝ := 1 * potato_cost
def total_juice_cost : ℝ := 2 * juice_cost

def total_cost : ℝ := total_salad_cost + total_beef_cost + total_potato_cost + total_juice_cost

theorem laura_total_cost : total_cost = 22 := by
  sorry

end laura_total_cost_l730_730123


namespace sinAB_range_x_range_l730_730094

noncomputable def triangle_conditions {A B C : ℝ} (a b c : ℝ) : Prop :=
  A + B + C = π ∧
  sin A * sin B * sin C ≠ 0 ∧
  a^2 + b^2 + c^2 = 1^2

theorem sinAB_range (A B C a b c: ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : triangle_conditions a b c)
  (m n : ℝ × ℝ)
  (hm : m = (a, cos B))
  (hn : n = (b, cos A))
  (mpn: m.1 * n.2 = n.1 * m.2)
  (mne: m ≠ n) :
  1 < sin A + sin B ∧ sin A + sin B ≤ sqrt 2 :=
sorry

theorem x_range (A B C a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc: triangle_conditions a b c)
  (x : ℝ)
  (hx : a*b*x = a + b)
  (h_range_ab : sqrt 2 + 1 = 2 * sqrt 2) :
  2 * sqrt 2 ≤ x :=
sorry

end sinAB_range_x_range_l730_730094


namespace sum_prime_odd_2009_l730_730812

-- Given a, b ∈ ℕ (natural numbers), where a is prime, b is odd, and a^2 + b = 2009,
-- prove that a + b = 2007.
theorem sum_prime_odd_2009 (a b : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Odd b) (h3 : a^2 + b = 2009) :
  a + b = 2007 := by
  sorry

end sum_prime_odd_2009_l730_730812


namespace hexagon_area_eight_triangle_area_l730_730492

open EuclideanGeometry

theorem hexagon_area_eight_triangle_area
  (A B C : Point) (BC AC AB : ℝ)
  (hA1 : ∃ A1, dist A A1 = BC ∧ collinear {A, B, A1})
  (hA2 : ∃ A2, dist A A2 = BC ∧ collinear {A, C, A2})
  (hB1 : ∃ B1, dist B B1 = AC ∧ collinear {B, A, B1})
  (hB2 : ∃ B2, dist B B2 = AC ∧ collinear {B, C, B2})
  (hC1 : ∃ C1, dist C C1 = AB ∧ collinear {C, A, C1})
  (hC2 : ∃ C2, dist C C2 = AB ∧ collinear {C, B, C2}) :
  area (hexagon A1 A2 B1 B2 C1 C2) = 8 * area (triangle A B C) :=
sorry

end hexagon_area_eight_triangle_area_l730_730492


namespace jennifers_sweets_division_l730_730898

theorem jennifers_sweets_division : 
  ∀ (g b y p : ℕ), g = 212 → b = 310 → y = 502 → p = 4 → (g + b + y) / p = 256 :=
by
  intros g b y p hg hb hy hp
  rw [hg, hb, hy, hp]
  norm_num
  sorry

end jennifers_sweets_division_l730_730898


namespace sum_of_abs_b_i_l730_730390

noncomputable def R (x : ℝ) : ℝ := 1 - (1 / 4) * x + (1 / 8) * x^2

noncomputable def S (x : ℝ) : ℝ := R(x) * R(x^2) * R(x^4) * R(x^6) * R(x^8)

theorem sum_of_abs_b_i : (Finset.sum (Finset.range 41) (λ i, |S (Real.of_nat i)|)) = 3125 / 1024 :=
by sorry

end sum_of_abs_b_i_l730_730390


namespace center_of_polar_circle_l730_730107

open Complex

def polar_circle_center (C : ℂ → Prop) :=
  ∀ z : ℂ, C z ↔ abs z = abs (real.cos (arg z + π / 3))

theorem center_of_polar_circle :
  polar_circle_center (λ z, abs z = complex.abs (complex.cos (arg z + π / 3))) →
  ∃ c : ℂ, c = complex.mk (1/2) (-π / 3) :=
by
  sorry

end center_of_polar_circle_l730_730107


namespace probability_of_divisibility_by_11_l730_730471

/-- The probability that a randomly selected five-digit number
whose digits sum to 42 is divisible by 11 is 1/5. -/
theorem probability_of_divisibility_by_11 :
  let digit_sum (n : ℕ) : ℕ := (n / 10000) % 10 + (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10 in
  let is_five_digit (n : ℕ) := 10000 ≤ n ∧ n < 100000 in
  let valid_number (n : ℕ) := is_five_digit n ∧ digit_sum n = 42 in
  let count_valid_numbers := Nat.card {n | valid_number n} in
  let count_valid_numbers_div_by_11 := Nat.card {n | valid_number n ∧ n % 11 = 0} in
  count_valid_numbers > 0 → count_valid_numbers_div_by_11 = count_valid_numbers / 5 :=
by
  sorry

end probability_of_divisibility_by_11_l730_730471


namespace analytical_expression_constant_triangle_area_l730_730656

-- Define the function f(x) as given in the problem
def f (a b x : ℝ) : ℝ := a * x - b / x

-- Analytical expression of f(x)
theorem analytical_expression : (∀ (a b : ℝ), 
  (∃ a b, 
    f a b 2 = x - 3 / x ∧
    a + b / 4 = 7 / 4 ∧ 
    2 * a - b / 2 = 1 / 2)) := sorry

-- Triangle area enclosed by the tangent line at any point, the line x = 0, and the line y = x is constant
theorem constant_triangle_area : 
  (∀ (a b : ℝ), 
    let f := λ (x : ℝ), a * x - b / x in
    ∃ k, 
      k = 6) := sorry

end analytical_expression_constant_triangle_area_l730_730656


namespace abs_C_minus_D_l730_730062

def c_n (d_n : ℕ → ℝ) (n : ℕ) : ℝ := 
  if n = 0 then 0 else (1/3) * d_n (n - 1)

def d_n (c_n : ℕ → ℝ) (n : ℕ) : ℝ := 
  if n = 0 then 3 * (2/3) else (1/3) * c_n (n - 1) + 2

noncomputable def C : ℝ := lim (c_n d_n)
noncomputable def D : ℝ := lim (d_n c_n)

theorem abs_C_minus_D :
  let C := lim (c_n d_n)
  let D := lim (d_n c_n)
  |C - D| = 3 / 2 :=
sorry

end abs_C_minus_D_l730_730062


namespace angle_condition_l730_730522

variable {A B C P R Q T : Type*}

-- The triangle, point on plane, and reflections
variables [triangle A B C]
variable (pointOnPlane: P)
variable (reflectionAB: R = reflection P A B)
variable (reflectionAC: Q = reflection P A C)
variable (intersection: T = line R Q ∩ line B C)

-- The main statement
theorem angle_condition:
  (angle A P B = angle A P C) ↔ (angle A P T = 90) :=
by
  sorry

end angle_condition_l730_730522


namespace sum_lent_l730_730259

theorem sum_lent (P : ℝ) (R : ℝ := 4) (T : ℝ := 8) (I : ℝ) (H1 : I = P - 204) (H2 : I = (P * R * T) / 100) : 
  P = 300 :=
by 
  sorry

end sum_lent_l730_730259


namespace Alyosha_apartment_number_l730_730313

theorem Alyosha_apartment_number 
  (K A : ℕ) 
  (h1 : K + A = 329) 
  (h2 : 9 * K - 8 ≤ A) 
  (h3 : A ≤ 9 * K) 
  (h4 : 32.9 ≤ K) 
  (h5 : K ≤ 33.7)
  : A = 296 := by
  sorry

end Alyosha_apartment_number_l730_730313


namespace pine_cone_weight_on_roof_l730_730308

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end pine_cone_weight_on_roof_l730_730308


namespace incorrect_calculation_l730_730628

theorem incorrect_calculation :
  (sqrt 8 / sqrt 2 = 2) ∧
  (sqrt (1 / 2) / sqrt 2 = 1 / 2) ∧
  (sqrt 3 / sqrt (3 / 2) = sqrt 2) ∧
  (sqrt (2 / 3) / sqrt (3 / 2) ≠ 1) :=
by
  sorry

end incorrect_calculation_l730_730628


namespace proposition_negation_l730_730082

theorem proposition_negation (p : Prop) : 
  (∃ x : ℝ, x < 1 ∧ x^2 < 1) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
sorry

end proposition_negation_l730_730082


namespace larger_of_two_numbers_l730_730066

theorem larger_of_two_numbers (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : max x y = 8 :=
sorry

end larger_of_two_numbers_l730_730066


namespace coin_flips_prob_l730_730653

open Probability

noncomputable def probability_Vanya_more_heads_than_Tanya (n : ℕ) : ℝ :=
  P(X Vanya > X Tanya)

theorem coin_flips_prob (n : ℕ) :
   P(Vanya gets more heads than Tanya | Vanya flips (n+1) times, Tanya flips n times) = 0.5 :=
sorry

end coin_flips_prob_l730_730653


namespace min_value_of_f_on_interval_l730_730788

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3

theorem min_value_of_f_on_interval :
  let f := λ x : ℝ, 2 * x^3 - 6 * x^2 + 3 in
  (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 3) →
  (∃ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x = -37) :=
sorry

end min_value_of_f_on_interval_l730_730788


namespace shortest_distance_midpoint_parabola_chord_l730_730795

theorem shortest_distance_midpoint_parabola_chord
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 = 4 * A.2)
  (hB : B.1 ^ 2 = 4 * B.2)
  (cord_length : dist A B = 6)
  : dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (0, 0) = 2 :=
sorry

end shortest_distance_midpoint_parabola_chord_l730_730795


namespace problem1_m_condition_problem2_a_condition_l730_730820

def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

theorem problem1_m_condition
  (a : ℝ) (h : a > 0) (h1 : a ≠ 1)
  (m : ℝ) :
  (∀ x, x ∈ Ioo (-1:ℝ) 1 → f a x) →
  (f a (1 - m) + f a (1 - m^2) < 0) ↔ (1 < m ∧ m < real.sqrt 2) := sorry

theorem problem2_a_condition
  (a : ℝ) (h : a > 0) (h1 : a ≠ 1) :
  (∀ x, x ∈ Iio (2:ℝ) → f a x < 4) ↔ (a ∈ Ioo (2 - real.sqrt 3) 1 ∪ Ioo 1 (2 + real.sqrt 3)) := sorry

end problem1_m_condition_problem2_a_condition_l730_730820


namespace frog_jump_vertical_side_prob_l730_730284

-- Definitions of the conditions
def square_side_prob {x y : ℕ} (p : ℕ × ℕ → ℚ) := 
  p (0, y) + p (4, y)

-- Main statement
theorem frog_jump_vertical_side_prob :
  ∀ (p : ℕ × ℕ → ℚ), 
    let start: ℕ × ℕ := (1, 2) in
    (∀ y, 0 ≤ y ∧ y ≤ 4 → p (0, y) = 1) → 
    (∀ y, 0 ≤ y ∧ y ≤ 4 → p (4, y) = 1) → 
    (∀ x, 0 ≤ x ∧ x ≤ 4 → p (x, 0) = 0) → 
    (∀ x, 0 ≤ x ∧ x ≤ 4 → p (x, 4) = 0) → 
    square_side_prob p = 5 / 8 :=
by
  intros
  sorry

end frog_jump_vertical_side_prob_l730_730284


namespace sin_2pi_minus_alpha_l730_730467

noncomputable def alpha_condition (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi) ∧ (Real.cos (Real.pi + α) = -1 / 2)

theorem sin_2pi_minus_alpha (α : ℝ) (h : alpha_condition α) : Real.sin (2 * Real.pi - α) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_2pi_minus_alpha_l730_730467


namespace car_wash_without_bulk_discount_car_wash_with_bulk_discount_l730_730897

section CarWash

def price_per_wash (price_per_bottle : ℝ) (washes_per_bottle : ℕ) : ℝ := 
  price_per_bottle / washes_per_bottle

def total_cost (price_per_wash : ℝ) (num_washes : ℕ) : ℝ := 
  price_per_wash * num_washes

def bulk_discount_price (price_per_bottle : ℝ) : ℝ := 
  price_per_bottle * 0.9

def cost_per_wash_with_discount (price_per_bottle : ℝ) (washes_per_bottle : ℕ) : ℝ :=
  bulk_discount_price(price_per_bottle) / washes_per_bottle

def total_cost_with_bulk (price_per_wash : ℝ) (num_washes : ℕ) : ℝ := 
  price_per_wash * num_washes

parameter (price_a : ℝ := 4) (washes_a : ℕ := 4)
parameter (price_b : ℝ := 6) (washes_b : ℕ := 6)
parameter (price_c : ℝ := 8) (washes_c : ℕ := 9)
parameter (num_washes : ℕ := 20)

theorem car_wash_without_bulk_discount :
  total_cost (price_per_wash price_a washes_a) num_washes = 20 ∧
  total_cost (price_per_wash price_b washes_b) num_washes = 20 ∧
  total_cost (price_per_wash price_c washes_c) num_washes = 17.80 :=
by
  sorry

theorem car_wash_with_bulk_discount :
  total_cost_with_bulk (cost_per_wash_with_discount price_a washes_a) num_washes = 18 ∧
  total_cost_with_bulk (cost_per_wash_with_discount price_b washes_b) num_washes = 18 ∧
  total_cost_with_bulk (cost_per_wash_with_discount price_c washes_c) num_washes = 16 :=
by
  sorry

end CarWash

end car_wash_without_bulk_discount_car_wash_with_bulk_discount_l730_730897


namespace image_of_1_plus_sqrt_2_preimage_of_neg_1_l730_730920

-- Definition of the function f
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Problem statement for the image of 1 + sqrt(2)
theorem image_of_1_plus_sqrt_2 : f (1 + real.sqrt 2) = 0 :=
by sorry

-- Problem statement for the preimage of -1
theorem preimage_of_neg_1 : {x : ℝ | f x = -1} = {0, 2} :=
by sorry

end image_of_1_plus_sqrt_2_preimage_of_neg_1_l730_730920


namespace not_algebraic_expression_C_l730_730706

-- Define what it means for something to be an algebraic expression, as per given problem's conditions
def is_algebraic_expression (expr : String) : Prop :=
  expr = "A" ∨ expr = "B" ∨ expr = "D"
  
theorem not_algebraic_expression_C : ¬ (is_algebraic_expression "C") :=
by
  -- This is a placeholder; proof steps are not required per instructions
  sorry

end not_algebraic_expression_C_l730_730706


namespace robert_can_read_books_l730_730172

theorem robert_can_read_books (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 →
  book_pages = 360 →
  total_hours = 8 →
  total_hours / (book_pages / pages_per_hour) = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end robert_can_read_books_l730_730172


namespace perpendiculars_intersect_at_single_point_l730_730327

theorem perpendiculars_intersect_at_single_point
  {A B C A1 A2 B1 B2 C1 C2 : Type}
  (circle_intersects_sides : ∀ {P}, (P = A1 ∨ P = A2) → P ∈ (circle ABC))
  (perpendiculars_meet_at_point_A : ∃ P : Type, is_perpendicular P A1 BC ∧ is_perpendicular P B1 CA ∧ is_perpendicular P C1 AB) :
  ∃ P' : Type, is_perpendicular P' A2 BC ∧ is_perpendicular P' B2 CA ∧ is_perpendicular P' C2 AB :=
sorry

end perpendiculars_intersect_at_single_point_l730_730327


namespace negative_number_among_options_l730_730316

theorem negative_number_among_options :
  ∃ (x : ℤ), x ∈ {(-(-2)), (abs (-2)), ((-2)^2), ((-2)^3)} ∧ x < 0 :=
by
  sorry

end negative_number_among_options_l730_730316


namespace evaluate_x2_minus_y2_l730_730020

-- Definitions based on the conditions.
def x : ℝ
def y : ℝ
axiom cond1 : x + y = 12
axiom cond2 : 3 * x + y = 18

-- The main statement we need to prove.
theorem evaluate_x2_minus_y2 : x^2 - y^2 = -72 :=
by
  sorry

end evaluate_x2_minus_y2_l730_730020


namespace part_a_part_b_l730_730589

open Nat

def nat_rel_prime (a b : ℕ) : Prop := gcd a b = 1
def s (z : ℕ) : ℕ → ℕ
| 0       => 1
| (k + 1) => s k + z^(k + 1)

theorem part_a (n z : ℕ) (h_rel_prime : nat_rel_prime n z) (hn : 1 < n) (hz : 1 < z) :
  ∃ k < n, n ∣ s z k :=
sorry

theorem part_b (n z : ℕ) (h_rel_prime : nat_rel_prime n z) (hz1_rel_prime : nat_rel_prime n (z - 1)) (hn : 1 < n) (hz : 1 < z) :
  ∃ k < n - 1, n ∣ s z k :=
sorry

end part_a_part_b_l730_730589


namespace train_speed_in_kmph_l730_730702

-- Define the conditions from the problem
def length_of_train := 160                          -- Length of train in meters
def length_of_bridge := 215                         -- Length of bridge in meters
def crossing_time := 30                             -- Time to cross bridge in seconds

-- Calculation for total crossing distance
def total_distance := length_of_train + length_of_bridge -- Total distance is train length + bridge length

-- Calculation for speed in meters per second
def speed_m_per_s := total_distance / crossing_time      -- Speed in meters per second

-- Conversion factor: 1 m/s = 3.6 km/hr
def conversion_factor := 3.6

-- Final speed in km/hr
def speed_km_per_hr := speed_m_per_s * conversion_factor

-- Theorem that states the speed of the train
theorem train_speed_in_kmph : speed_km_per_hr = 45 :=
  sorry  -- Proof is skipped

end train_speed_in_kmph_l730_730702


namespace problem1_monotonic_f_problem2a_monotonic_g_small_a_problem2b_monotonic_g_large_a_problem3_max_value_b_l730_730440

noncomputable def f (x b : ℝ) : ℝ := (1/2) * x^2 + b * x + real.log x
noncomputable def g (x b a : ℝ) : ℝ := f x b - b * x - (1 + a) / 2 * x^2

-- Problem (1)
theorem problem1_monotonic_f (x : ℝ) (b : ℝ) (hx0 : 0 < x) (hf : ∀ x, 0 < x → 0 ≤ x + b + 1 / x) : b ≥ -2 :=
sorry

-- Problem (2a)
theorem problem2a_monotonic_g_small_a (x a : ℝ) (hx0 : 0 < x) (ha : a ≤ 0) (hg : ∀ x, 0 < x → 0 ≤ 1 / x - a * x) : ∀ x, 0 < x → monotone_on (g x b) (set.Ioi 0) :=
sorry

-- Problem (2b)
theorem problem2b_monotonic_g_large_a (x a : ℝ) (hx0 : 0 < x) (ha : 0 < a) :
  (monotone_on (g x b a) (set.Icc (0 : ℝ) (real.sqrt a / a))) ∧ (antimono_on (g x b a) (set.Ioi (real.sqrt a / a))) :=
sorry

-- Problem (3)
theorem problem3_max_value_b (x : ℝ) (b : ℝ) (hx0 : 0 < x ∧ x ≤ 1)
  (hf_ineq : ∀ x, 0 < x ∧ x ≤ 1 → f x b ≤ x^2 + (1 / (2 * x^2)) - 3 * x + 1) (hmf : b ≥ -2) : -2 ≤ b ∧ b ≤ -1 :=
sorry

end problem1_monotonic_f_problem2a_monotonic_g_small_a_problem2b_monotonic_g_large_a_problem3_max_value_b_l730_730440


namespace probability_product_multiple_of_4_l730_730184

theorem probability_product_multiple_of_4 :
  let sophie := {n : ℕ | n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}
  let max := {n : ℕ | n ∈ {1, 2, 3, 4, 5, 6, 7, 8}}
  let multiple_of_4 (n m : ℕ) := (n * m) % 4 = 0
  let prob (s : set ℕ) := (s.card : ℚ) / 12
  (sophie.card = 12) →
  (max.card = 8) →
  (∑ n in sophie, ∑ m in max, if multiple_of_4 n m then 1 else 0) / (12 * 8) = 1 / 4 :=
by
  sorry

end probability_product_multiple_of_4_l730_730184


namespace kristy_insurance_allocation_l730_730121

/-- Kristy's allocation to insurance -/
theorem kristy_insurance_allocation (b h c s p : ℝ) (total_earnings insurance_allocation : ℝ)
  (basic_salary : total_earnings = b * h + c * s)
  (budget_percentage : p = 0.95)
  (insurance_percentage : 1 - p = 0.05) :
  insurance_allocation = total_earnings * (1 - p) :=
by
  sorry

-- setting the values to the given conditions and correct answer
#eval kristy_insurance_allocation 7.5 160 0.16 25000 0.95 (7.5 * 160 + 0.16 * 25000) ((7.5 * 160 + 0.16 * 25000) * 0.05)
-- Should evaluate to true because (7.5 * 160 + 0.16 * 25000) * 0.05 = 260

end kristy_insurance_allocation_l730_730121


namespace binom_coeff_divisibility_l730_730555

theorem binom_coeff_divisibility (p : ℕ) (hp : Prime p) : Nat.choose (2 * p) p - 2 ≡ 0 [MOD p^2] := 
sorry

end binom_coeff_divisibility_l730_730555


namespace _l730_730042

structure point :=
  (x : ℝ)
  (y : ℝ)

-- Define the coordinates of points A, B, and C
def A : point := ⟨-1, 0⟩
def B : point := ⟨0, 2⟩
def C : point := ⟨2, 0⟩

-- Define the midpoint function
def midpoint (P Q : point) : point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

-- Define the vector from point P to point Q
def vector (P Q : point) : point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

-- Define point D as the midpoint of B and C
def D : point := midpoint B C

-- The theorem to prove
example : vector A D = ⟨2, 1⟩ :=
by
  sorry

end _l730_730042


namespace total_chairs_in_canteen_l730_730305

theorem total_chairs_in_canteen (numRoundTables : ℕ) (numRectangularTables : ℕ) 
                                (chairsPerRoundTable : ℕ) (chairsPerRectangularTable : ℕ)
                                (h1 : numRoundTables = 2)
                                (h2 : numRectangularTables = 2)
                                (h3 : chairsPerRoundTable = 6)
                                (h4 : chairsPerRectangularTable = 7) : 
                                (numRoundTables * chairsPerRoundTable + numRectangularTables * chairsPerRectangularTable = 26) :=
by
  sorry

end total_chairs_in_canteen_l730_730305


namespace largest_five_digit_integer_prod_2772_l730_730623

theorem largest_five_digit_integer_prod_2772 :
  ∃ n : ℕ, n / 10000 > 0 ∧ n / 10000 < 10 ∧ (∀ d ∈ [n / 10000, (n % 10000) / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10], 0 < d) ∧ 
  (n / 10000) * ((n % 10000) / 1000) * ((n % 1000) / 100) * ((n % 100) / 10) * (n % 10) = 2772 ∧
  ∃ k : ℕ, ∀ m : ℕ, m / 10000 > 0 ∧ m / 10000 < 10 ∧ (∀ d ∈ [m / 10000, (m % 10000) / 1000, (m % 1000) / 100, (m % 100) / 10, m % 10], 0 < d) ∧ 
  (m / 10000) * ((m % 10000) / 1000) * ((m % 1000) / 100) * ((m % 100) / 10) * (m % 10) = 2772 → m ≤ n :=
  ∃ (n : ℕ), n = 98721 ∧ -- The largest integer satisfying the condition
  ((n / 10000) * ((n % 10000) / 1000) * ((n % 1000) / 100) * ((n % 100) / 10) * (n % 10) = 2772) ∧
  ∀ m : ℕ, (m / 10000 > 0) ∧ (m / 10000 < 10) ∧
  ((m / 10000) * ((m % 10000) / 1000) * ((m % 1000) / 100) * ((m % 100) / 10) * (m % 10) = 2772 →
  m ≤ n) :=
sorry

end largest_five_digit_integer_prod_2772_l730_730623


namespace sam_books_l730_730721

theorem sam_books (A : ℝ) 
  (used_mystery_books : ℝ := 17.0) 
  (new_crime_books : ℝ := 15.0) 
  (total_books : ℝ := 45) 
  (h : A + used_mystery_books + new_crime_books = total_books) : 
  A = 13 := 
by 
  simp at h;
  linarith;
  done

end sam_books_l730_730721


namespace complement_of_16deg51min_is_73deg09min_l730_730069

def complement_angle (A : ℝ) : ℝ := 90 - A

theorem complement_of_16deg51min_is_73deg09min :
  complement_angle 16.85 = 73.15 := by
  sorry

end complement_of_16deg51min_is_73deg09min_l730_730069


namespace rowing_upstream_speed_l730_730689

def speed_in_still_water : ℝ := 31
def speed_downstream : ℝ := 37

def speed_stream : ℝ := speed_downstream - speed_in_still_water

def speed_upstream : ℝ := speed_in_still_water - speed_stream

theorem rowing_upstream_speed :
  speed_upstream = 25 := by
  sorry

end rowing_upstream_speed_l730_730689


namespace find_complement_intersection_find_union_complement_subset_implies_a_range_l730_730054

-- Definitions for sets A and B
def A : Set ℝ := { x | 3 ≤ x ∧ x < 6 }
def B : Set ℝ := { x | 2 < x ∧ x < 9 }

-- Definitions for complements and subsets
def complement (S : Set ℝ) : Set ℝ := { x | x ∉ S }
def intersection (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∧ x ∈ T }
def union (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∨ x ∈ T }

-- Definition for set C as a parameterized set by a
def C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Proof statements
theorem find_complement_intersection :
  complement (intersection A B) = { x | x < 3 ∨ x ≥ 6 } :=
by sorry

theorem find_union_complement :
  union (complement B) A = { x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by sorry

theorem subset_implies_a_range (a : ℝ) :
  C a ⊆ B → a ∈ {x | 2 ≤ x ∧ x ≤ 8} :=
by sorry

end find_complement_intersection_find_union_complement_subset_implies_a_range_l730_730054


namespace maximum_value_of_M_l730_730792

noncomputable def M (x : ℝ) : ℝ :=
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x)

theorem maximum_value_of_M : 
  ∃ x : ℝ, M x = (Real.sqrt 3) / 4 :=
sorry

end maximum_value_of_M_l730_730792


namespace sufficient_but_not_necessary_l730_730852

open Real

theorem sufficient_but_not_necessary (a b c : ℝ) (h : a > b) :
  ac^2 > bc^2 → (a = b → false) :=
by 
  sorry

end sufficient_but_not_necessary_l730_730852


namespace sqrt_of_neg_five_squared_l730_730983

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 ∨ Real.sqrt ((-5 : ℝ)^2) = -5 :=
by
  sorry

end sqrt_of_neg_five_squared_l730_730983


namespace boys_neither_happy_sad_anxious_l730_730482

theorem boys_neither_happy_sad_anxious (total_children happy_children sad_children anxious_children total_boys happy_boys sad_girls anxious_girls : ℕ)
  (h1 : total_children = 80)
  (h2 : happy_children = 25)
  (h3 : sad_children = 15)
  (h4 : anxious_children = 20)
  (h5 : total_boys = 35)
  (h6 : happy_boys = 10)
  (h7 : sad_girls = 6)
  (h8 : anxious_girls = 12) :
  (total_boys - happy_boys - (sad_children - sad_girls) - (anxious_children - anxious_girls) = 8) ∧ 
  (happy_children - happy_boys = 15) :=
begin
  sorry
end

end boys_neither_happy_sad_anxious_l730_730482


namespace quadrilateral_circle_area_ratio_l730_730956

theorem quadrilateral_circle_area_ratio (r : ℝ) (A B C D : Type) [IsCircle r C]
  (h1: Segment AC = 2 * r) (h2: ∠DAC = 15) (h3: ∠BAC = 30) :
  let area_ratio := (√6 + √2 + 2*√3) / (4 * π)
  in 0 + 20 + 4 = 24 :=
by
  have h_AreaABCD := sorry -- Details of the area calculations
  have h_AreaCircle := sorry -- Details of the area calculations
  exact sorry

end quadrilateral_circle_area_ratio_l730_730956


namespace solution_exists_and_is_unique_l730_730365

theorem solution_exists_and_is_unique :
  ∃ (x y z : ℝ), (sqrt (x^3 - y) = z - 1) ∧ (sqrt (y^3 - z) = x - 1) ∧ (sqrt (z^3 - x) = y - 1) ∧ x = 1 ∧ y = 1 ∧ z = 1 :=
begin
  use [1, 1, 1],
  split,
  calc sqrt (1 ^ 3 - 1) = sqrt 0 : by norm_num
                     ... = 0        : by norm_num,
  split,
  calc sqrt (1 ^ 3 - 1) = sqrt 0 : by norm_num
                     ... = 0        : by norm_num,
  calc sqrt (1 ^ 3 - 1) = sqrt 0 : by norm_num
                     ... = 0        : by norm_num,
  exact ⟨rfl, rfl, rfl⟩ 
end

end solution_exists_and_is_unique_l730_730365


namespace maximize_magnitude_l730_730393

theorem maximize_magnitude (a x y : ℝ) 
(h1 : 4 * x^2 + 4 * y^2 = -a^2 + 16 * a - 32)
(h2 : 2 * x * y = a) : a = 8 := 
sorry

end maximize_magnitude_l730_730393


namespace number_of_unique_numbers_l730_730459

theorem number_of_unique_numbers :
  let digits := [3, 3, 3, 7, 7] in
  list.permutations digits |>.length = 10 :=
by
  have : multiset.card (multiset.pmap (λ _ _, 1) [3, 3, 3, 7, 7] _) = 10 :=
    sorry
  exact this

end number_of_unique_numbers_l730_730459


namespace difference_of_reciprocals_l730_730857

theorem difference_of_reciprocals (p q : ℝ) (hp : 3 / p = 6) (hq : 3 / q = 15) : p - q = 3 / 10 :=
by
  sorry

end difference_of_reciprocals_l730_730857


namespace area_arccos_cos_l730_730769

noncomputable def area_under_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..(2 * real.pi), real.arccos (real.cos x)

theorem area_arccos_cos :
  area_under_arccos_cos = real.pi ^ 2 :=
sorry

end area_arccos_cos_l730_730769


namespace measure_of_angle_is_60_l730_730850

noncomputable def measure_of_angle_A_44_A_45_A_43 (A : ℕ → ℝ × ℝ) :=
  ∃ (A₁ A₂ A₃ : ℝ × ℝ),
    (euclidean_geometry.equilateral_triangle A₁ A₂ A₃) ∧
    (∀ n : ℕ, A (n + 3) = midpoint (A n) (A (n + 1))) →
    angle (A 44) (A 45) (A 43) = 60

theorem measure_of_angle_is_60 
  {A : ℕ → ℝ × ℝ} 
  (h₁ : ∃ (A₁ A₂ A₃ : ℝ × ℝ), 
         (euclidean_geometry.equilateral_triangle A₁ A₂ A₃) ∧ 
         (∀ n : ℕ, A (n + 3) = midpoint (A n) (A (n + 1)))) :
  measure_of_angle_A_44_A_45_A_43 A :=
begin
  sorry
end

end measure_of_angle_is_60_l730_730850


namespace unique_cube_colorings_l730_730684

theorem unique_cube_colorings :
  let num_black_faces := 1,
      num_yellow_faces := 2,
      num_white_faces := 3 in
  let num_faces_on_cube := 6 in
  let total_colorings := (num_black_faces > 0) ∧ (num_yellow_faces > 0) ∧ (num_white_faces > 0) ∧ (num_black_faces + num_yellow_faces + num_white_faces = num_faces_on_cube) in
  total_colorings → 3 :=
begin
  sorry
end

end unique_cube_colorings_l730_730684


namespace part1_conditions_part2_range_m_l730_730405

-- Part 1
theorem part1_conditions (a b : ℝ) (f : ℝ → ℝ)
  (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (hf_monotonic : ∀ x y, x < y → f x > f y) : 
  (f 1 = -3 / 5 ∧ f x = 0 ∧ f (-x) = -f x) → ((a = 1 / 2) ∧ (b = 0)) := 
sorry

-- Part 2
theorem part2_range_m (f : ℝ → ℝ) (a : ℝ) (h : a = 1 / 2) :
  (∀ x, f x = (1 - 4 ^ x) / (1 + 4 ^ x)) →
  ((∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ [-1, 1] ∧ x₂ ∈ [-1, 1] ∧ f x = m - 4 ^ x₁) ↔ m ∈ (2 * sqrt 2 - 2, 17 / 20]) :=
sorry

end part1_conditions_part2_range_m_l730_730405


namespace matrix_sum_correct_l730_730341

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![4, -3],
  ![2, 5]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-6, 8],
  ![-3, 7]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, 5],
  ![-1, 12]
]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l730_730341


namespace area_triangle_QTS_l730_730613

-- Definitions and conditions
variable (PQ RS : ℝ) (A : ℝ)
variable (h_area : A = 200)
variable (PQ_len : PQ = 15)
variable (RS_len : RS = 25)

-- The goal to prove
theorem area_triangle_QTS (h_diagonals_intersect : ∃ T, true):
    QTS_area PQ RS A = 46.875 :=
by
  -- Definitions from the problem conditions
  have h_trapezoid : PQ + RS = 40 := by rw [PQ_len, RS_len]; norm_num
  have h_area_eq : 2 * A = 400 := by rw [h_area]; norm_num
  -- Sorry placeholder for the actual proof
  sorry

end area_triangle_QTS_l730_730613


namespace steve_family_time_l730_730185

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l730_730185


namespace num_distinct_solutions_cos_eq_l730_730585

theorem num_distinct_solutions_cos_eq (a : ℝ) (h_a : a = 24 * Real.pi) : 
  (∃ n : ℕ, n = 20 ∧ 
    ∀ x : ℝ, 0 < x ∧ x < a →
      cos (x / 4) = cos x → ∃ k : ℕ, x = (8 * k * Real.pi) / 5 ∨ x = (8 * k * Real.pi) / 3) := sorry

end num_distinct_solutions_cos_eq_l730_730585


namespace maximize_profit_l730_730331

-- Define constants for purchase and selling prices
def priceA_purchase : ℝ := 16
def priceA_selling : ℝ := 20
def priceB_purchase : ℝ := 20
def priceB_selling : ℝ := 25

-- Define constant for total weight
def total_weight : ℝ := 200

-- Define profit function
def profit (weightA weightB : ℝ) : ℝ :=
  (priceA_selling - priceA_purchase) * weightA + (priceB_selling - priceB_purchase) * weightB

-- Define constraints
def constraint1 (weightA weightB : ℝ) : Prop :=
  weightA + weightB = total_weight

def constraint2 (weightA weightB : ℝ) : Prop :=
  weightA >= 3 * weightB

open Real

-- Define the maximum profit we aim to prove
def max_profit : ℝ := 850

-- The main theorem to prove
theorem maximize_profit : 
  ∃ weightA weightB : ℝ, constraint1 weightA weightB ∧ constraint2 weightA weightB ∧ profit weightA weightB = max_profit :=
by {
  sorry
}

end maximize_profit_l730_730331


namespace siding_cost_l730_730961

def wall_width := 10
def wall_height := 7
def roof_width := 10
def roof_height := 6
def siding_section_width := 10
def siding_section_height := 12
def siding_section_cost := 30

def wall_area := wall_width * wall_height
def number_of_walls := 2
def total_wall_area := wall_area * number_of_walls

def roof_area := (roof_width * roof_height) / 2
def number_of_roofs := 2
def total_roof_area := roof_area * number_of_roofs

def total_area := total_wall_area + total_roof_area

def siding_section_area := siding_section_width * siding_section_height
def sections_needed := (total_area / siding_section_area).ceil.to_nat
def total_cost := sections_needed * siding_section_cost

theorem siding_cost : total_cost = 60 := by
  sorry

end siding_cost_l730_730961


namespace third_group_members_l730_730677

theorem third_group_members (total_members first_group second_group : ℕ) (h₁ : total_members = 70) (h₂ : first_group = 25) (h₃ : second_group = 30) : (total_members - (first_group + second_group)) = 15 :=
sorry

end third_group_members_l730_730677


namespace findRangeA_l730_730435

noncomputable theory
open Real

def rootsAP (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ (a₂ - a₁) = (a₃ - a₂) ∧ (a₃ - a₂) = (a₄ - a₃)

def arithmeticProgressionRoots (m n : ℝ) (x : ℝ) : Prop :=
  let p := x^2 - 2*x + m
  let q := x^2 - 2*x + n
  ∃ a₁ a₂ a₃ a₄ : ℝ,
    rootsAP a₁ a₂ a₃ a₄ ∧
    p.eval a₁ = 0 ∧ p.eval a₄ = 0 ∧ q.eval a₂ = 0 ∧ q.eval a₃ = 0 ∧
    a₁ = 1/4

def isAcuteTriangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧
  0 < B ∧ B < π / 2 ∧
  0 < C ∧ C < π / 2 ∧
  A + B + C = π

def triangleABC (a b c A B C : ℝ) : Prop :=
  isAcuteTriangle A B C ∧
  A = 2 * B ∧
  b = 4 * abs (a - c)

theorem findRangeA (m n a b c A B C : ℝ) :
  arithmeticProgressionRoots m n a →
  triangleABC a b c A B C →
  2 * sqrt 2 < a ∧ a < 2 * sqrt 3 :=
begin
  -- to be proved
  sorry
end

end findRangeA_l730_730435


namespace area_to_paint_correct_l730_730962

-- Define the measurements used in the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 2

-- Definition of areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length

-- Definition of total area to paint
def total_area_to_paint : ℕ := wall_area - (window1_area + window2_area)

-- Theorem statement to prove the total area to paint is 131 square feet
theorem area_to_paint_correct : total_area_to_paint = 131 := by
  sorry

end area_to_paint_correct_l730_730962


namespace solution_set_f_lt_1_l730_730423

-- Define the function f with the given properties
noncomputable def f : ℝ → ℝ := sorry

-- Hypotheses
axiom even_function : ∀ x, f(-x) = f(x)
axiom monotonic_increasing : ∀ x y, 0 ≤ x → x ≤ y → f(x) ≤ f(y)
axiom point_a : f 0 = -1
axiom point_b : f 3 = 1

-- Main theorem to prove
theorem solution_set_f_lt_1 : { x : ℝ | f(x) < 1 } = set.Ioo (-3 : ℝ) 3 :=
sorry

end solution_set_f_lt_1_l730_730423


namespace vanya_more_heads_probability_l730_730645

-- Define binomial distributions for Vanya and Tanya's coin flips
def binomial (n : ℕ) (p : ℝ) : distribution :=
  λ k : ℕ, choose n k * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the random variables X_V and X_T
def X_V (n : ℕ) : distribution := binomial (n + 1) (1 / 2)
def X_T (n : ℕ) : distribution := binomial n (1 / 2)

-- Define the probability of Vanya getting more heads than Tanya
def probability_vanya_more_heads (n : ℕ) : ℝ :=
  ∑ (k : ℕ) in finset.range (n + 2), ∑ (j : ℕ) in finset.range (k), X_V n k * X_T n j

-- The main theorem to prove
theorem vanya_more_heads_probability (n : ℕ) : probability_vanya_more_heads n = 1 / 2 :=
begin
  sorry,
end

end vanya_more_heads_probability_l730_730645


namespace P_inter_Q_complement_eq_l730_730127

-- Define the universal set
def U := ℝ

-- Define set P as {x | x^2 < 1}
def P : set ℝ := {x | x^2 < 1}

-- Define set Q as {x | x >= 0}
def Q : set ℝ := {x | 0 <= x}

-- Define the complement of Q with respect to the universal set U
def Q_complement : set ℝ := {x | x < 0}

-- State the theorem
theorem P_inter_Q_complement_eq : P ∩ Q_complement = {x | -1 < x ∧ x < 0} :=
sorry

end P_inter_Q_complement_eq_l730_730127


namespace num_factors_m_l730_730921

noncomputable def m : ℕ := 2^5 * 3^6 * 5^7 * 6^8

theorem num_factors_m : ∃ (k : ℕ), k = 1680 ∧ ∀ d : ℕ, d ∣ m ↔ ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 13 ∧ 0 ≤ b ∧ b ≤ 14 ∧ 0 ≤ c ∧ c ≤ 7 ∧ d = 2^a * 3^b * 5^c :=
by 
sorry

end num_factors_m_l730_730921


namespace eva_total_weighted_score_l730_730358

def first_semester_marks (second_semester_marks : ℕ → ℕ) (subject : String) : ℕ :=
  match subject with
  | "Maths" => second_semester_marks 0 + 10
  | "Arts" => second_semester_marks 1 - 15
  | "Science" => (4 * second_semester_marks 2) / 4 - second_semester_marks 2 / 3
  | "History" => second_semester_marks 3 + 5
  | _ => 0

def total_weighted_score (first_marks second_marks : ℕ → ℕ) (weights : ℕ → ℕ) (subjects : List String) : ℝ :=
  (subjects.zipWith first_marks (List.range subjects.length)).zip (subjects.zipWith second_marks (List.range subjects.length)).zip (subjects.zipWith weights (List.range subjects.length)).sum (λ (x : (ℕ × (ℕ × ℕ))) => 
    (x.fst * x.snd.fst * x.snd.snd).toReal →
      ((x.fst * x.snd.fst * x.snd.snd) %
      if x.fst = 90 
      then 1 else 0))

theorem eva_total_weighted_score (marks : ℕ × ℕ × ℕ × ℕ) (weights : ℕ → ℕ) (subjects : List String) :
  first_semester_marks (λ i, (match i with | 0 => marks.1 | 1 => marks.2 | 2 => marks.3 | 3 => marks.4)) "Maths" = 90 →
  first_semester_marks (λ i, (match i with | 0 => marks.1 | 1 => marks.2 | 2 => marks.3 | 3 => marks.4)) "Arts" = 75 →
  first_semester_marks (λ i, (match i with | 0 => marks.1 | 1 => marks.2 | 2 => marks.3 | 3 => marks.4)) "Science" = 67.5 →
  first_semester_marks (λ i, (match i with | 0 => marks.1 | 1 => marks.2 | 2 => marks.3 | 3 => marks.4)) "History" = 90 →
  marks = (80, 90, 90, 85) →
  let total_score := total_weighted_score 
    (λ i, (match i with | 0 => 90 | 1 => 75 | 2 => 67.5 | 3 => 90)) 
    (λ i, (match i with | 0 => 80 | 1 => 90 | 2 => 90 | 3 => 85)) 
    (λ i, (match i with | 0 => 30 | 1 => 25 | 2 => 35 | 3 => 10)) 
    ["Maths", "Arts", "Science", "History"]
  in total_score = 164.875 := 
by sorry

end eva_total_weighted_score_l730_730358


namespace angle_ACE_equals_angle_CEG_l730_730164

-- Define points on the plane
variables (A B C D E F G : ℝ)

-- Define the conditions
-- Rectangles ABCD and DEFG with point D on segment BF
def rectangles_ABCD_DEFG (h1 : D ∈ [B, F]) : Prop := 
  ∀ θ : ℝ, 
    θ = 90 ∧ 
    DW.line [A, B, C, D] ∧ 
    DW.line [D, E, F, G] ∧ 
    D = DW.cross_segment [B, F]

-- Points B, C, E, F lying on the same circle 
def cyclic_quadrilateral (h1 : D ∈ [B, F])
  (B C E F : ℝ) : Prop := 
  circle B C E F

-- Proving the angles are equal
theorem angle_ACE_equals_angle_CEG 
  (h1 : D ∈ [B, F]) 
  (B C E F : ℝ)
  : rectangles_ABCD_DEFG h1 →
    cyclic_quadrilateral h1 B C E F →
    ∀ θ : ℝ, 
      θ = ∠(A, C, E) → 
      θ = ∠(C, E, G) := 
begin
  sorry
end

end angle_ACE_equals_angle_CEG_l730_730164


namespace impossible_to_equalize_sequence_l730_730592

def initial_sequence : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem impossible_to_equalize_sequence (sequence : List ℕ)
    (h1 : sequence = initial_sequence)
    (h2 : ∀ (a b : ℕ), (add_1_to_adjacent sequence) → (multiply_by_3 sequence)) :
  ∀ n, ¬ (∀ x ∈ sequence, x = n) :=
sorry

-- Definitions for allowed actions can simply be placeholders
def add_1_to_adjacent (seq : List ℕ) : Prop := sorry

def multiply_by_3 (seq : List ℕ) : Prop := sorry

end impossible_to_equalize_sequence_l730_730592


namespace sum_of_integers_satisfying_condition_l730_730777

theorem sum_of_integers_satisfying_condition :
  (∑ k in finset.filter (λ k, nat.choose 26 5 + nat.choose 26 6 = nat.choose 27 k) (finset.range 28), k) = 27 :=
by sorry

end sum_of_integers_satisfying_condition_l730_730777


namespace number_of_unique_numbers_l730_730458

theorem number_of_unique_numbers :
  let digits := [3, 3, 3, 7, 7] in
  list.permutations digits |>.length = 10 :=
by
  have : multiset.card (multiset.pmap (λ _ _, 1) [3, 3, 3, 7, 7] _) = 10 :=
    sorry
  exact this

end number_of_unique_numbers_l730_730458


namespace sin_2beta_eq_f_alpha_eq_l730_730394

-- Define the conditions
variables {α β : ℝ}
variables (h1 : 0 < α ∧ α < π/2 ∧ π/2 < β ∧ β < π)
variables (h2 : cos (β - π/4) = 1/3)
variables (h3 : sin (α + β) = 4/5)

-- Proof problem for part 1: Proving sin 2β = -7/9
theorem sin_2beta_eq : sin (2 * β) = -7 / 9 :=
by {
  sorry
}

-- Defining the function f
def f (x : ℝ) : ℝ := cos x - sin x

-- Proof problem for part 2: Proving f(α) = (16 - 3 * sqrt 2) / 15
theorem f_alpha_eq : f α = (16 - 3 * Real.sqrt 2) / 15 :=
by {
  sorry
}

end sin_2beta_eq_f_alpha_eq_l730_730394


namespace tangent_line_at_one_range_of_a_l730_730439

-- Problem 1: The tangent line to \( f(x) = (x+1) \ln x \) at \( x = 1 \)
def f (x : ℝ) : ℝ := (x + 1) * Real.log x 

theorem tangent_line_at_one : ∃ (m b : ℝ), m = 2 ∧ b = -2 ∧ ∀ x, f x = m * (x - 1) + f 1 := 
sorry

-- Problem 2: The range of \( a \) given \( g(x) = \frac{1}{a(1-x)} f(x) \) and \( g(x) < -2 \)
def g (a x : ℝ) : ℝ := (1 / (a * (1 - x))) * f x

theorem range_of_a : ∀ x ∈ Ioo (0:ℝ) 1, ∀ a, g a x < -2 → a ∈ Icc (0:ℝ) 1 :=
sorry

end tangent_line_at_one_range_of_a_l730_730439


namespace collinear_A_M_N_l730_730200

theorem collinear_A_M_N {A B C B₁ C₁ M B₂ C₂ N : Type*}
  [is_angle_bisector BB₁ A B C]
  [is_angle_bisector CC₁ A B C]
  [is_incenter M A B B₁]
  [is_incenter M A C C₁]
  [is_angle_bisector B₁B₂ A B₁ C₁]
  [is_angle_bisector C₁C₂ A B₁ C₁]
  [is_incenter N A B₁ B₂]
  [is_incenter N A C₁ C₂] :
  collinear A M N :=
sorry

end collinear_A_M_N_l730_730200


namespace prob_sampling_l730_730196

def prob_of_sampling (P_A P_B P_C P_S_A P_S_B P_S_C : ℚ) :=
  P_A * P_S_A + P_B * P_S_B + P_C * P_S_C

theorem prob_sampling (h_P_A : 0.40 = 2 / 5) (h_P_B : 0.35 = 7 / 20) (h_P_C : 0.25 = 1 / 4)
  (h_P_S_A : 0.23 = 23 / 100) (h_P_S_B : 0.39 = 39 / 100) (h_P_S_C : 0.53 = 53 / 100) :
  prob_of_sampling 0.40 0.35 0.25 0.23 0.39 0.53 = 0.361 := 
by
  simp [prob_of_sampling, h_P_A, h_P_B, h_P_C, h_P_S_A, h_P_S_B, h_P_S_C]
  norm_num
  sorry

end prob_sampling_l730_730196


namespace remaining_wallpaper_removal_time_l730_730753

theorem remaining_wallpaper_removal_time (dining_walls living_walls : ℕ) (time_per_wall: ℕ) (time_spent: ℕ) :
  dining_walls = 4 →
  living_walls = 4 →
  time_per_wall = 2 →
  time_spent = 2 →
  time_per_wall * dining_walls + time_per_wall * living_walls - time_spent = 14 :=
by
  intros hd hl ht hs
  rw [hd, hl, ht, hs]
  exact dec_trivial

end remaining_wallpaper_removal_time_l730_730753


namespace intersection_A_B_l730_730836

open Set

-- Define the set A
def A : Set ℝ := {x | (2 * x - 6) / (x + 1) ≤ 0}

-- Define the set B
def B : Set ℝ := {-2, -1, 0, 3, 4}

-- State the theorem that A ∩ B = {0, 3}
theorem intersection_A_B : A ∩ B = {0, 3} :=
by
  sorry

end intersection_A_B_l730_730836


namespace sum_of_first_5_terms_b_def_eq_90_l730_730433

def a_n (n : ℕ) := 3 * n

def b_n (n : ℕ) := a_n (2 * n)

def sum_first_5_terms_of_b : ℕ := (Finset.range 5).sum (λ n, b_n (n + 1))

theorem sum_of_first_5_terms_b_def_eq_90 :
  sum_first_5_terms_of_b = 90 :=
  sorry

end sum_of_first_5_terms_b_def_eq_90_l730_730433


namespace points_opposite_sides_line_l730_730036

theorem points_opposite_sides_line (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end points_opposite_sides_line_l730_730036


namespace proof_propositions_l730_730449

-- Define basic geometric axioms and properties

-- Define what it means for lines to be parallel
def parallel (m n : Type) := ∀ P Q R, P ∈ m → Q ∈ m → R ∈ n → P - Q = Q - R

-- Define what it means for a line to be perpendicular to a plane
def perpendicular (m α : Type) := ∀ P Q R, P ∈ m → Q ∈ m → R ∈ α → P - Q ⊥ Q - R

-- Define the three propositions
def proposition1 (m n α : Type) : Prop := (perpendicular m α ∧ perpendicular n α) → parallel m n
def proposition2 (m n α : Type) : Prop := (parallel m α ∧ parallel n α) → parallel m n
def proposition3 (m n α : Type) : Prop := (parallel m α ∧ perpendicular n α) → perpendicular m n

theorem proof_propositions (m n α : Type) :
  proposition1 m n α ∧ ¬ proposition2 m n α ∧ proposition3 m n α :=
by
  sorry

end proof_propositions_l730_730449


namespace laura_total_cost_l730_730122

def salad_cost : ℝ := 3
def beef_cost : ℝ := 2 * salad_cost
def potato_cost : ℝ := salad_cost / 3
def juice_cost : ℝ := 1.5

def total_salad_cost : ℝ := 2 * salad_cost
def total_beef_cost : ℝ := 2 * beef_cost
def total_potato_cost : ℝ := 1 * potato_cost
def total_juice_cost : ℝ := 2 * juice_cost

def total_cost : ℝ := total_salad_cost + total_beef_cost + total_potato_cost + total_juice_cost

theorem laura_total_cost : total_cost = 22 := by
  sorry

end laura_total_cost_l730_730122


namespace find_b_l730_730333

theorem find_b 
  (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) 
  (h₅ : ∀ x, abs (a * sin (b * x + c) + d) ≤ abs (5 * sin (x / 4))) :
  b = 5 / 2 :=
by
  sorry

end find_b_l730_730333


namespace cost_of_bought_movie_l730_730942

theorem cost_of_bought_movie 
  (ticket_cost : ℝ)
  (ticket_count : ℕ)
  (rental_cost : ℝ)
  (total_spent : ℝ)
  (bought_movie_cost : ℝ) :
  ticket_cost = 10.62 →
  ticket_count = 2 →
  rental_cost = 1.59 →
  total_spent = 36.78 →
  bought_movie_cost = total_spent - (ticket_cost * ticket_count + rental_cost) →
  bought_movie_cost = 13.95 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_bought_movie_l730_730942


namespace analyze_system_l730_730154

noncomputable def system_of_equations (t x y : ℝ) :=
  (4 * t^2 + t + 4) * x + (5 * t + 1) * y = 4 * t^2 - t - 3 ∧
  (t + 2) * x + 2 * y = t

theorem analyze_system (t x y : ℝ) (h : system_of_equations t x y) :
  (t ≠ 2 → (system_of_equations t x y)) ∧
  (t = 2 → ¬ ∃ x y, system_of_equations t x y) :=
begin
  sorry
end

end analyze_system_l730_730154


namespace sum_of_solutions_of_fx_eq_zero_l730_730144

noncomputable def f : ℝ → ℝ := λ x, if x < 3 then 7 * x + 21 else 3 * x - 9

theorem sum_of_solutions_of_fx_eq_zero : 
  (∃ x : ℝ, f x = 0) → (x1 = -3 ∧ x2 = 3) ∧ x1 + x2 = 0 := 
by
  sorry

end sum_of_solutions_of_fx_eq_zero_l730_730144


namespace measure_angle_CAB_in_regular_hexagon_l730_730488

-- Definitions used in step a)
def is_regular_hexagon (ABCDEF : Type) : Prop :=
  -- Regular hexagon property: all interior angles are 120 degrees
  ∀ (A B C D E F : ABCDEF), ⦃A, B, C, D, E, F⦄ → 
    ∀ angle, angle ∈ set.interior_angles [A, B, C, D, E, F] → angle = 120

-- Main theorem statement derived from step c)
theorem measure_angle_CAB_in_regular_hexagon
  (ABCDEF : Type)
  (h_reg_hex : is_regular_hexagon ABCDEF)
  (A B C D E F : ABCDEF)
  (h_diag_AC : A ≠ C) : 
  ∃ (angle : ℝ), angle = 30 :=
by 
  sorry

end measure_angle_CAB_in_regular_hexagon_l730_730488


namespace triangle_side_length_l730_730499

   theorem triangle_side_length
   (A B C D E F : Type)
   (angle_bac angle_edf : Real)
   (AB AC DE DF : Real)
   (h1 : angle_bac = angle_edf)
   (h2 : AB = 5)
   (h3 : AC = 4)
   (h4 : DE = 2.5)
   (area_eq : (1 / 2) * AB * AC * Real.sin angle_bac = (1 / 2) * DE * DF * Real.sin angle_edf):
   DF = 8 :=
   by
   sorry
   
end triangle_side_length_l730_730499


namespace vanya_more_heads_probability_l730_730646

-- Define binomial distributions for Vanya and Tanya's coin flips
def binomial (n : ℕ) (p : ℝ) : distribution :=
  λ k : ℕ, choose n k * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the random variables X_V and X_T
def X_V (n : ℕ) : distribution := binomial (n + 1) (1 / 2)
def X_T (n : ℕ) : distribution := binomial n (1 / 2)

-- Define the probability of Vanya getting more heads than Tanya
def probability_vanya_more_heads (n : ℕ) : ℝ :=
  ∑ (k : ℕ) in finset.range (n + 2), ∑ (j : ℕ) in finset.range (k), X_V n k * X_T n j

-- The main theorem to prove
theorem vanya_more_heads_probability (n : ℕ) : probability_vanya_more_heads n = 1 / 2 :=
begin
  sorry,
end

end vanya_more_heads_probability_l730_730646


namespace cycle_cut_space_orthogonal_complement_l730_730990

-- Definitions based on conditions
section graph_theory

variables {V E : Type}

-- Assume G is a graph with vertex set V and edge set E 
variable (G : Type) [graph G]

-- Definition of the cycle space of graph G
def cycle_space (G : Type) : Type := sorry

-- Definition of the cut space of graph G
def cut_space (G : Type) : Type := sorry

-- Orthogonal complement of a space
def orthogonal_complement (S : Type) : Type := sorry

open_locale classical
noncomputable theory

-- Theorem statement
theorem cycle_cut_space_orthogonal_complement :
  cycle_space G = orthogonal_complement (cut_space G) ∧
  cut_space G = orthogonal_complement (cycle_space G) :=
sorry

end graph_theory

end cycle_cut_space_orthogonal_complement_l730_730990


namespace part_a_part_b_l730_730965

-- Part (a)
theorem part_a (x : ℝ) (h1 : x > -1) (h2 : x ≠ 0) : 
  log (x+1) (x^2 - 3 * x + 1) = 1 → x = 4 := 
sorry

-- Part (b)
theorem part_b (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) : 
  log x (2 * x^2 - 3 * x - 4) = 2 → x = 4 := 
sorry

end part_a_part_b_l730_730965


namespace time_with_family_l730_730189

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l730_730189


namespace value_of_expression_l730_730244

theorem value_of_expression :  real.of_intRoot 4 (81 * real.sqrt (27 * real.sqrt 9)) = 4.8 :=
by
  sorry

end value_of_expression_l730_730244


namespace circumcenters_parallel_l730_730657

open EuclideanGeometry

-- Define the points and conditions of the problem
variables {A B C D E F K M N O₁ O₂ : Point}
variables {ABC : Triangle A B C}
variables {AD : Median A B C D}
variables {K := Midpoint A D}
variables {E : Point} {F : Point}
variables {DE_perp_AB : Perpendicular D E .segment A B E}
variables {DF_perp_AC : Perpendicular D F .segment A C F}
variables {KE_M : Intersecting KE BC M}
variables {KF_N : Intersecting KF BC N}
variables {O₁_circumcenter : Circumcenter D E M O₁}
variables {O₂_circumcenter : Circumcenter D F N O₂}

-- The problem statement to prove O₁O₂ parallel to BC
theorem circumcenters_parallel (h_acute : AcuteTriangle ABC) 
(H_AB_neq_AC : A ≠ B ∧ A ≠ C) 
(H_mid : Midpoint A D K) 
(H_DE_perp_AB : Perpendicular D E (lineThrough A B))
(H_DF_perp_AC : Perpendicular D F (lineThrough A C))
(H_KE_BC_M : Intersect KE BC M)
(H_KF_BC_N : Intersect KF BC N)
(H_O₁_circumcenter : IsCircumcenter (Triangle D E M) O₁)
(H_O₂_circumcenter : IsCircumcenter (Triangle D F N) O₂) :
Parallel (lineSegment O₁ O₂) (lineThrough B C) :=
sorry

end circumcenters_parallel_l730_730657


namespace kim_probability_same_color_l730_730120

noncomputable def probability_same_color (total_shoes : ℕ) (pairs_of_shoes : ℕ) : ℚ :=
  let total_selections := (total_shoes * (total_shoes - 1)) / 2
  let successful_selections := pairs_of_shoes
  successful_selections / total_selections

theorem kim_probability_same_color :
  probability_same_color 10 5 = 1 / 9 :=
by
  unfold probability_same_color
  have h_total : (10 * 9) / 2 = 45 := by norm_num
  have h_success : 5 = 5 := by norm_num
  rw [h_total, h_success]
  norm_num
  done

end kim_probability_same_color_l730_730120


namespace purchase_prices_l730_730946

theorem purchase_prices (x y : ℝ) (hx : x + y = 5)
  (ha : ∃ (a : ℝ), a = x + 1)
  (hb : ∃ (b : ℝ), b = 2 * y - 1)
  (hc : 3 * (x + 1) + 2 * (2 * y -1) = 19) : 
  x = 2 ∧ y = 3 := by
sor

end purchase_prices_l730_730946


namespace diophantine_solution_l730_730143

theorem diophantine_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (n : ℕ) (h_n : n > a * b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end diophantine_solution_l730_730143


namespace journey_first_part_time_l730_730065

variable (x : ℝ) -- Distance of the first part in km
variable (speed1 : ℝ := 40) -- Speed for the first part in kmph
variable (speed2 : ℝ := 60) -- Speed for the second part in kmph
variable (total_distance : ℝ := 240) -- Total distance in km
variable (total_time : ℝ := 5) -- Total time in hours

theorem journey_first_part_time :
  let time1 := x / speed1 in
  let time2 := (total_distance - x) / speed2 in
  time1 + time2 = total_time →
  time1 = 3 :=
by
  sorry

end journey_first_part_time_l730_730065


namespace max_remainder_when_divided_by_7_l730_730454

theorem max_remainder_when_divided_by_7 (y : ℕ) (r : ℕ) (h : r = y % 7) : r ≤ 6 ∧ ∃ k, y = 7 * k + r :=
by
  sorry

end max_remainder_when_divided_by_7_l730_730454


namespace find_q_l730_730762

theorem find_q (q : ℚ) (h_nonzero: q ≠ 0) :
  ∃ q, (qx^2 - 18 * x + 8 = 0) → (324 - 32*q = 0) :=
begin
  sorry
end

end find_q_l730_730762


namespace problem_statement_l730_730132

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 2)

def M : set ℝ := { x | f x ≤ 6 }

theorem problem_statement
  (a b : ℝ)
  (hM_a : a ∈ M)
  (hM_b : b ∈ M) :
  sqrt 3 * abs (a + b) ≤ abs (a * b + 3) :=
sorry

end problem_statement_l730_730132


namespace time_with_family_l730_730190

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l730_730190


namespace original_calculation_l730_730465

theorem original_calculation
  (x : ℝ)
  (h : ((x * 3) + 14) * 2 = 946) :
  ((x / 3) + 14) * 2 = 130 :=
sorry

end original_calculation_l730_730465


namespace train_passes_man_in_approx_33_seconds_l730_730638

/-- A train 605 meters long is running with a speed of 60 km/h. A man is running at 6 km/h in the direction 
opposite to that in which the train is going. The time it takes for the train to pass the man is approximately 33 seconds. -/
theorem train_passes_man_in_approx_33_seconds :
  let length_of_train : ℝ := 605
  let speed_of_train_kmh : ℝ := 60
  let speed_of_man_kmh : ℝ := 6
  let speed_of_train_ms := speed_of_train_kmh * (1000 / 3600)
  let speed_of_man_ms := speed_of_man_kmh * (1000 / 3600)
  let relative_speed_ms := speed_of_train_ms + speed_of_man_ms
  let time_to_pass := length_of_train / relative_speed_ms
  time_to_pass ≈ 33 :=
by sorry

end train_passes_man_in_approx_33_seconds_l730_730638


namespace greenfield_teachers_needed_l730_730455

theorem greenfield_teachers_needed :
  ∀ (students_per_school : ℕ) (classes_per_student : ℕ) (classes_per_teacher : ℕ) (students_per_class : ℕ),
  students_per_school = 900 →
  classes_per_student = 6 →
  classes_per_teacher = 5 →
  students_per_class = 25 →
  let total_classes := students_per_school * classes_per_student in
  let unique_classes := total_classes / students_per_class in
  let required_teachers := unique_classes.toNat.ceil / classes_per_teacher in
  required_teachers = 44 :=
by
  intros students_per_school classes_per_student classes_per_teacher students_per_class 
         h_students_per_school h_classes_per_student h_classes_per_teacher h_students_per_class;
  let total_classes := students_per_school * classes_per_student;
  let unique_classes := total_classes / students_per_class;
  let required_teachers := unique_classes.toNat.ceil / classes_per_teacher;
  sorry

end greenfield_teachers_needed_l730_730455


namespace planes_parallel_transitive_l730_730661

-- Defining non-overlapping lines (this might need to be adapted depending on the exact meaning in the mathematical context)
variable (m n l : Type) [Line m] [Line n] [Line l]
variable (α β γ : Type) [Plane α] [Plane β] [Plane γ]

-- Defining the problem conditions
variable (non_overlap_lines : ∀ m n l, m ≠ n ∧ m ≠ l ∧ n ≠ l)
variable (non_overlap_planes : ∀ α β γ, α ≠ β ∧ α ≠ γ ∧ β ≠ γ)

-- Defining parallelism between lines and planes
variable (parallel_planes : ∀ α β γ, (α ∥ γ) → (β ∥ γ) → (α ∥ β))

-- Statement to prove
theorem planes_parallel_transitive (α β γ : Type) [Plane α] [Plane β] [Plane γ]
  (hαγ : α ∥ γ) (hβγ : β ∥ γ) : α ∥ β :=
sorry

end planes_parallel_transitive_l730_730661


namespace seulgi_second_round_score_l730_730457

theorem seulgi_second_round_score
    (h_score1 : Nat) (h_score2 : Nat)
    (hj_score1 : Nat) (hj_score2 : Nat)
    (s_score1 : Nat) (required_second_score : Nat) :
    h_score1 = 23 →
    h_score2 = 28 →
    hj_score1 = 32 →
    hj_score2 = 17 →
    s_score1 = 27 →
    required_second_score = 25 →
    s_score1 + required_second_score > h_score1 + h_score2 ∧ 
    s_score1 + required_second_score > hj_score1 + hj_score2 :=
by
  intros
  sorry

end seulgi_second_round_score_l730_730457


namespace total_weight_is_58kg_l730_730635

-- Definitions and conditions from the problem
def zinc_to_copper_ratio (Z C : ℝ) : Prop := Z / C = 9 / 11

def weight_of_zinc : ℝ := 26.1

def copper_weight_from_zinc (Z : ℝ) : ℝ := (11 / 9) * Z

-- Theorem Statement to prove the total weight is 58 kg
theorem total_weight_is_58kg (Z C : ℝ) (h_ratio : zinc_to_copper_ratio Z C) (h_zinc : Z = weight_of_zinc) : Z + C = 58 := by
  sorry

end total_weight_is_58kg_l730_730635


namespace double_segment_with_straightedge_l730_730840

theorem double_segment_with_straightedge
    (l₁ l₂ : Line)
    (h_parallel : Parallel l₁ l₂)
    (A B : Point)
    (AB_on_l₁ : LiesOn A l₁ ∧ LiesOn B l₁)
    (C D : Point)
    (CD_on_l₂ : LiesOn C l₂ ∧ LiesOn D l₂) :
    ∃ E : Point, E ≠ B ∧ SegmentEq AB AE := 
sorry

end double_segment_with_straightedge_l730_730840


namespace infinite_3_stratum_numbers_l730_730615

-- Condition for 3-stratum number
def is_3_stratum_number (n : ℕ) : Prop :=
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = (Finset.range (n + 1)).filter (λ x => n % x = 0) ∧
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Part (a): Find a 3-stratum number
example : is_3_stratum_number 120 := sorry

-- Part (b): Prove there are infinitely many 3-stratum numbers
theorem infinite_3_stratum_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_3_stratum_number (f n) := sorry

end infinite_3_stratum_numbers_l730_730615


namespace total_number_of_lockers_l730_730720

/-- 
At Triumph High School, the student lockers are numbered consecutively starting from 1 and 
extending to locker number 2999. 
Each plastic digit for numbering the lockers costs three cents. 
Prove that the total number of lockers is 2999. 
-/
theorem total_number_of_lockers : 
  ∃ n, (∀ k, 1 ≤ k ∧ k ≤ 2999 → ∃ d, 3 * d = (number_of_digits k) * 3) ∧ n = 2999 :=
begin
  sorry
end

end total_number_of_lockers_l730_730720


namespace smallest_k_l730_730217

noncomputable def a : ℕ → ℝ
| 0     := 1
| 1     := real.root 21 3
| (n+2) := a (n+1) * (a n)^3

def product_is_integer (k : ℕ) : Prop :=
∃ m : ℤ, (∏ i in finset.range k, a (i + 1)) = m

theorem smallest_k (k : ℕ) : product_is_integer k → k = 7 := sorry

end smallest_k_l730_730217


namespace zero_table_possible_l730_730494

variable (m n : ℕ)
variable (T : Matrix (Fin m) (Fin n) ℕ)

definition double_row (i : Fin m) : Matrix (Fin m) (Fin n) ℕ :=
  T.update_column i (λ j, 2 * T i j)

definition subtract_one_col (j : Fin n) : Matrix (Fin m) (Fin n) ℕ :=
  λ i, T i j - 1

theorem zero_table_possible (m n : ℕ) (T : Matrix (Fin m) (Fin n) ℕ) :
  ∃ (f1 : List (Fin m)), ∃ (f2 : List (Fin n)), (f1.foldl (λ t i, double_row t i) (f2.foldl (λ t j, subtract_one_col t j) T)) = 0 :=
sorry

end zero_table_possible_l730_730494


namespace hexagon_angle_CAB_is_30_l730_730484

theorem hexagon_angle_CAB_is_30 (ABCDEF : Type) [hexagon : IsRegularHexagon ABCDEF]
  (A B C D E F : Point ABCDEF) (h1 : InteriorAngle ABCDEF = 120) (h2 : IsDiagonal AC ) :
  ∠CAB = 30 :=
by
  sorry

end hexagon_angle_CAB_is_30_l730_730484


namespace complex_number_in_second_quadrant_l730_730594

-- Define complex numbers and quadrants
def imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0

-- Problem statement
theorem complex_number_in_second_quadrant
  (i : ℂ) (h₁ : imaginary_unit i) :
  quadrant ((1 + 2 * i) * i) = 2 :=
sorry

end complex_number_in_second_quadrant_l730_730594


namespace tan_identity_l730_730396

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l730_730396


namespace trig_expression_zero_l730_730400

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l730_730400


namespace fraction_of_state_quarters_is_two_fifths_l730_730941

variable (total_quarters state_quarters : ℕ)
variable (is_pennsylvania_percentage : ℚ)
variable (pennsylvania_state_quarters : ℕ)

theorem fraction_of_state_quarters_is_two_fifths
  (h1 : total_quarters = 35)
  (h2 : pennsylvania_state_quarters = 7)
  (h3 : is_pennsylvania_percentage = 1 / 2)
  (h4 : state_quarters = 2 * pennsylvania_state_quarters)
  : (state_quarters : ℚ) / (total_quarters : ℚ) = 2 / 5 :=
sorry

end fraction_of_state_quarters_is_two_fifths_l730_730941


namespace tangent_line_equation_l730_730770

noncomputable def curve := fun x : ℝ => Real.sin (x + Real.pi / 3)

def tangent_line (x y : ℝ) : Prop :=
  x - 2 * y + Real.sqrt 3 = 0

theorem tangent_line_equation :
  tangent_line 0 (curve 0) := by
  unfold curve tangent_line
  sorry

end tangent_line_equation_l730_730770


namespace line_with_equal_intercepts_l730_730581

theorem line_with_equal_intercepts (x y : ℝ) (h : x = 1 ∧ y = 2) :
  (x + y = 3) ∨ (2 * x - y = 0) :=
by
  cases h with
  | intro hx hy =>
    rw [hx, hy]
    left
    norm_num
    right
    norm_num
    sorry

end line_with_equal_intercepts_l730_730581


namespace robert_books_read_l730_730168

theorem robert_books_read (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 → book_pages = 360 → total_hours = 8 → (total_hours * pages_per_hour) / book_pages = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact (nat.div_eq_of_lt sorry)
end

end robert_books_read_l730_730168


namespace jennifers_sweets_division_l730_730899

theorem jennifers_sweets_division : 
  ∀ (g b y p : ℕ), g = 212 → b = 310 → y = 502 → p = 4 → (g + b + y) / p = 256 :=
by
  intros g b y p hg hb hy hp
  rw [hg, hb, hy, hp]
  norm_num
  sorry

end jennifers_sweets_division_l730_730899


namespace alpha_value_l730_730914

open Complex

theorem alpha_value (α β : ℂ) (h1 : β = 2 + 3 * I) (h2 : (α + β).im = 0) (h3 : (I * (2 * α - β)).im = 0) : α = 6 + 4 * I :=
by
  sorry

end alpha_value_l730_730914


namespace salon_customers_l730_730294

variables (n : ℕ) (c : ℕ)

theorem salon_customers :
  ∀ (n = 33) (extra_cans = 5) (cans_per_customer = 2),
  (n - extra_cans) / cans_per_customer = 14 :=
begin
  sorry
end

end salon_customers_l730_730294


namespace melanie_trout_catch_l730_730560

def trout_caught_sara : ℕ := 5
def trout_caught_melanie (sara_trout : ℕ) : ℕ := 2 * sara_trout

theorem melanie_trout_catch :
  trout_caught_melanie trout_caught_sara = 10 :=
by
  sorry

end melanie_trout_catch_l730_730560


namespace find_f_f_neg1_l730_730789

def f (x : Int) : Int :=
  if x >= 0 then x + 2 else 1

theorem find_f_f_neg1 : f (f (-1)) = 3 :=
by
  sorry

end find_f_f_neg1_l730_730789


namespace simplify_expression_l730_730182

theorem simplify_expression :
  (Real.sqrt 600 / Real.sqrt 75 - Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end simplify_expression_l730_730182


namespace trigonometric_identity_l730_730404

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l730_730404


namespace product_of_sequence_2018_l730_730052

noncomputable def sequence (n : ℕ) : ℚ :=
match n with
| 1     => 2
| (n+1) => (1 + sequence n) / (1 - sequence n)

theorem product_of_sequence_2018 :
  ∏ k in finset.range 2018, sequence (k + 1) = -6 :=
sorry

end product_of_sequence_2018_l730_730052


namespace molecular_weight_of_NH4I_correct_l730_730624

-- Define the atomic weights as given conditions
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

-- Define the calculation of the molecular weight of NH4I
def molecular_weight_NH4I : ℝ :=
  atomic_weight_N + 4 * atomic_weight_H + atomic_weight_I

-- Theorem stating the molecular weight of NH4I is 144.95 g/mol
theorem molecular_weight_of_NH4I_correct : molecular_weight_NH4I = 144.95 :=
by
  sorry

end molecular_weight_of_NH4I_correct_l730_730624


namespace books_read_in_eight_hours_l730_730169

-- Definitions to set up the problem
def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

-- Theorem statement
theorem books_read_in_eight_hours : (available_time * reading_speed) / book_length = 2 := 
by
  sorry

end books_read_in_eight_hours_l730_730169


namespace find_x0_l730_730046

theorem find_x0 
  (ω : ℝ)
  (x0 : ℝ)
  (h1 : 0 < ω)
  (h2 : (2 * Real.pi) / ω = Real.pi)
  (h3 : f x0 = 0)
  (f : ℝ → ℝ) : x0 ∈ Set.Icc 0 (Real.pi / 2) → x0 = (5 * Real.pi) / 12 :=
by
  sorry

end find_x0_l730_730046


namespace proposition_p_l730_730083

variable (x : ℝ)

-- Define condition
def negation_of_p : Prop := ∃ x, x < 1 ∧ x^2 < 1

-- Define proposition p
def p : Prop := ∀ x, x < 1 → x^2 ≥ 1

-- Theorem statement
theorem proposition_p (h : negation_of_p) : (p) :=
sorry

end proposition_p_l730_730083


namespace digits_repeat_after_removal_and_addition_l730_730966

theorem digits_repeat_after_removal_and_addition (n : ℕ) (h_n : n = 2^2023) :
  ∃ i j : ℕ, i ≠ j ∧ (digital_removal_addition h_n).digits.drop (digital_removal_addition h_n).digits.length.tail.to_list i =
  (digital_removal_addition h_n).digits.drop (digital_removal_addition h_n).digits.length.tail.to_list j :=
sorry

end digits_repeat_after_removal_and_addition_l730_730966


namespace money_left_l730_730117

noncomputable def initial_amount : ℕ := 100
noncomputable def cost_roast : ℕ := 17
noncomputable def cost_vegetables : ℕ := 11

theorem money_left (init_amt cost_r cost_v : ℕ) 
  (h1 : init_amt = 100)
  (h2 : cost_r = 17)
  (h3 : cost_v = 11) : init_amt - (cost_r + cost_v) = 72 := by
  sorry

end money_left_l730_730117


namespace no_poly_divides_f_l730_730542

noncomputable def f (n : ℕ) (a : Fin n → ℤ) : ℤ[X] :=
  (∏ i in Finset.univ, (X - C (a i))) - 1

theorem no_poly_divides_f (n : ℕ) (a : Fin n → ℤ) (h1 : 1 < n) (h2 : Function.Injective a) :
  ¬ ∃ g : ℤ[X], (∀ x : ℤ, g.leadingCoeff = 1) ∧ g.degree < n ∧ (∃ h : ℤ[X], f n a = g * h) := 
begin
  sorry
end

end no_poly_divides_f_l730_730542


namespace sum_prime_odd_2009_l730_730813

-- Given a, b ∈ ℕ (natural numbers), where a is prime, b is odd, and a^2 + b = 2009,
-- prove that a + b = 2007.
theorem sum_prime_odd_2009 (a b : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Odd b) (h3 : a^2 + b = 2009) :
  a + b = 2007 := by
  sorry

end sum_prime_odd_2009_l730_730813


namespace books_read_in_eight_hours_l730_730171

-- Definitions to set up the problem
def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

-- Theorem statement
theorem books_read_in_eight_hours : (available_time * reading_speed) / book_length = 2 := 
by
  sorry

end books_read_in_eight_hours_l730_730171


namespace Triangle_OH_lt_3R_l730_730412

-- Lean code representation of the problem conditions and question
theorem Triangle_OH_lt_3R
  (ABC : Triangle)
  (nondegenerate : ¬Collinear ABC.A ABC.B ABC.C)
  (O : Point)
  (H : Point)
  (R : ℝ)
  (Circumcenter : isCircumcenter O ABC)
  (Orthocenter : isOrthocenter H ABC)
  (Circumradius : Circumradius ABC = R) :
  dist O H < 3 * R := 
sorry

end Triangle_OH_lt_3R_l730_730412


namespace star_point_angles_sum_to_720_l730_730751

-- Define the circle, points, and tip angles
def circle : Type := sorry
def points (n : ℕ) : set (circle) := sorry

-- Condition: points are evenly spaced
def evenly_spaced (n : ℕ) (p : set (circle)) : Prop := sorry

-- Define the tips of the star and their angles
def tip_angles (p : set (circle)) : list ℝ := sorry

-- Define the sum of angles at the tips
def sum_tip_angles (angles : list ℝ) : ℝ := sorry

-- Define the 8-pointed star related properties
theorem star_point_angles_sum_to_720 :
  ∀ (c : circle) (p : set (circle)),
    evenly_spaced 8 p →
    sum_tip_angles (tip_angles p) = 720 :=
by
  intros c p h
  sorry

end star_point_angles_sum_to_720_l730_730751


namespace distance_from_point_to_line_correct_l730_730369

noncomputable def distance_point_to_line :
  ℝ := 
  let a := ((0 : ℝ), 3, -1)
  let p₁ := (1, 2, 1)
  let p₂ := (2, 4, 0)
  let d := (1, 2, -1)
  let v (t : ℝ) := (1 + t, 2 + 2*t, 1 - t)
  let diff (t : ℝ) := (v t.1 - a.1, v t.2 - a.2, v t.3 - a.3)
  let dot_product (t : ℝ) := (1 + t) + 2*(-1 + 2*t) - (2 - t)
  have t_value : ℝ := 1 / 2
  let point_on_line := v t_value
  let distance :=
    sqrt ((point_on_line.1 - a.1)^2 + (point_on_line.2 - a.2)^2 + (point_on_line.3 - a.3)^2)
  in distance

theorem distance_from_point_to_line_correct :
  distance_point_to_line = 3 * sqrt 2 / 2 :=
  sorry

end distance_from_point_to_line_correct_l730_730369


namespace part1_part2_l730_730442

noncomputable def f (a : ℝ) (x : ℝ) := 
  a * Real.log x - x + 1 / x

theorem part1 (a : ℝ) (h : ∃ x, 2 < x ∧ derivative (f a) x = 0) : 
  a ∈ set.Ioi (5 / 2) :=
sorry

theorem part2 (a : ℝ) (h_a : a > 0) (h : a ≤ Real.exp 1 + Real.exp (-1)) :
  ∃ M, M = 4 / Real.exp 1 :=
sorry

end part1_part2_l730_730442


namespace gcd_2024_2048_l730_730619

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l730_730619


namespace cyclic_quadrilateral_AD_length_l730_730876

theorem cyclic_quadrilateral_AD_length
  (A B C D : Type*)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  (cyclic_ABC : cyclic A B C)
  (AB : distance A B = 7)
  (BC : distance B C = 9)
  (CD : distance C D = 25)
  (angle_B : angle A B C = 90)
  (angle_C : angle B C D = 90) :
  distance A D = sqrt 337 :=
sorry

end cyclic_quadrilateral_AD_length_l730_730876


namespace ellipse_a_plus_k_l730_730709

-- Define the conditions
def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 1), (0, -3))
def point_on_ellipse : ℝ × ℝ := (5, 0)

-- Define the theorem to prove
theorem ellipse_a_plus_k :
  ∃ (a b h k : ℝ), 
    (a > 0) ∧ (b > 0) ∧ (h = 0)  ∧ (k = -1) ∧ 
    (foci = ((0, 1), (0, -3))) ∧ 
    (point_on_ellipse = (5, 0)) ∧
    (a + k = (sqrt 26 + sqrt 34 - 2) / 2) ∧
    (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 → 
                (x, y) = (point_on_ellipse) ∨ 
                (sqrt ((5 - 0)^2 + (0 - 1)^2) + sqrt ((5 - 0)^2 + (0 + 3)^2) = sqrt 26 + sqrt 34))
: sorry

end ellipse_a_plus_k_l730_730709


namespace sweets_per_person_l730_730900

theorem sweets_per_person (green_sweets blue_sweets yellow_sweets : ℕ)
  (total_friends : ℕ) (h_green : green_sweets = 212) (h_blue : blue_sweets = 310) (h_yellow : yellow_sweets = 502) (h_total_friends : total_friends = 4) :
  (green_sweets + blue_sweets + yellow_sweets) / total_friends = 256 :=
by 
suffices h_total_sweets : green_sweets + blue_sweets + yellow_sweets = 1024, from
  by rw [h_total_sweets, h_total_friends]
    exact nat.div_eq_of_eq_mul_right (by norm_num) rfl,
calc
  green_sweets + blue_sweets + yellow_sweets
    = 212 + 310 + 502 : by rw [h_green, h_blue, h_yellow]
... = 1024 : by norm_num

end sweets_per_person_l730_730900


namespace integer_k_values_l730_730767

theorem integer_k_values (k : ℤ) (x : ℤ) :
  ( ∃ x, (√(39 - 6 * √12) + √(k * x * (k * x + √12) + 3) = 2 * k) ) →
  (k = 3 ∨ k = 6) := 
sorry

end integer_k_values_l730_730767


namespace triangle_XYZ_median_l730_730888

theorem triangle_XYZ_median (XYZ : Triangle) (YZ : ℝ) (XM : ℝ) (XY2_add_XZ2 : ℝ) 
  (hYZ : YZ = 12) (hXM : XM = 7) : XY2_add_XZ2 = 170 → N - n = 0 := by
  sorry

end triangle_XYZ_median_l730_730888


namespace exactly_one_equals_xx_plus_xx_l730_730063

theorem exactly_one_equals_xx_plus_xx (x : ℝ) (hx : x > 0) :
  let expr1 := 2 * x^x
  let expr2 := x^(2*x)
  let expr3 := (2*x)^x
  let expr4 := (2*x)^(2*x)
  (expr1 = x^x + x^x) ∧ (¬(expr2 = x^x + x^x)) ∧ (¬(expr3 = x^x + x^x)) ∧ (¬(expr4 = x^x + x^x)) := 
by
  sorry

end exactly_one_equals_xx_plus_xx_l730_730063


namespace proof_angle_CPB_l730_730268

-- Definitions and lemmas used in the conditions
variables {A B C P : Type}
variables (triangle : Type) [is_triangle triangle A B C]
variables (angle_bisector : triangle → Type) [is_angle_bisector angle_bisector A B C]
variables (perpendicular_bisector : triangle → Type) [is_perpendicular_bisector perpendicular_bisector A B]
variables (intersect : Type) [is_intersect intersect angle_bisector perpendicular_bisector]
variables (angle_C : Type) [angle_C_in_triangle angle_C C 60]
variables {angle_CPB : Type}
variables (theorem_angle_CPB : triangle → Type) [is_theorem_angle_CPB theorem_angle_CPB 80]

-- The final statement to be proved
theorem proof_angle_CPB (triangle ABC : Type) (angle_bisector B : Type) (perpendicular_bisector AB : Type) (intersect P AC : Type) (angle_C 60 : Type) :
  angle_CPB = 80 :=
sorry

end proof_angle_CPB_l730_730268


namespace minimum_rolls_for_repeated_sum_l730_730242

-- Define the parameters for the problem
def fair_eight_sided_die := { n : ℕ // 1 ≤ n ∧ n ≤ 8 }

-- Define the number of dice
def number_of_dice := 4

-- The minimum number of times to roll to ensure a repeated sum
def minimum_rolls := 30

-- Define the main theorem statement
theorem minimum_rolls_for_repeated_sum : 
  ∃ n >= minimum_rolls, ∀ (throws : fin n → vector fair_eight_sided_die number_of_dice), 
    ∃ i j : fin n, i ≠ j ∧ (vector.sum (throws i).val = vector.sum (throws j).val) := sorry

end minimum_rolls_for_repeated_sum_l730_730242


namespace lambda_for_pair_of_lines_l730_730746

def equivalent_quadratic (λ : ℝ) : Prop :=
  ∀ x y : ℝ, λ * x^2 + 4 * x * y + y^2 - 4 * x - 2 * y - 3 = 0 →

theorem lambda_for_pair_of_lines (λ : ℝ) : (λ = 4) :=
sorry

end lambda_for_pair_of_lines_l730_730746


namespace division_example_l730_730617

theorem division_example : (0.075 : ℝ) / 0.005 = 15 := 
by 
  have h : (0.075 : ℝ) / 0.005 = (75 : ℝ) / 5 := 
    by num_cases
  have h_div : (75 : ℝ) / 5 = 15 := by linarith
  rwa [h] at h_div

end division_example_l730_730617


namespace transform_center_l730_730338

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def translate_right (p : point) (d : ℝ) : point :=
  (p.1 + d, p.2)

theorem transform_center (C : point) (hx : C = (3, -4)) :
  translate_right (reflect_x_axis C) 3 = (6, 4) :=
by
  sorry

end transform_center_l730_730338


namespace product_of_consecutive_integers_eq_255_l730_730215

theorem product_of_consecutive_integers_eq_255 (x : ℕ) (h : x * (x + 1) = 255) : x + (x + 1) = 31 := 
sorry

end product_of_consecutive_integers_eq_255_l730_730215


namespace cole_runs_7_miles_l730_730631

theorem cole_runs_7_miles
  (xavier_miles : ℕ)
  (katie_miles : ℕ)
  (cole_miles : ℕ)
  (h1 : xavier_miles = 3 * katie_miles)
  (h2 : katie_miles = 4 * cole_miles)
  (h3 : xavier_miles = 84)
  (h4 : katie_miles = 28) :
  cole_miles = 7 := 
sorry

end cole_runs_7_miles_l730_730631


namespace right_triangle_proof_l730_730521

variable (A B C E F K L : Type)
variable [RightTriangle A B C]
variable [IsHypotenuse B C]
variable [AngleBisector B E (Angle A B C)]
variable [CircumcircleIntersection B C E AB F]
variable [Projection A K (Line BC)]
variable [PointOnSegment L AB]
variable (BK : Length K B)
variable (BC : Length B C)

theorem right_triangle_proof (hBL_BK : Length B L = Length B K) :
  (Length A L / Length A F) = Real.sqrt (Length B K / Length B C) :=
sorry

end right_triangle_proof_l730_730521


namespace order_of_a_b_c_l730_730538

-- Define conditions
variable {f : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f (x)

def condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → x * (deriv f x) < f (-x)

noncomputable def a := (sqrt 2) * (f (sqrt 2))
noncomputable def b := (Real.log 2) * (f (Real.log 2))
noncomputable def c := (Real.log 2 (1 / 4)) * (f (Real.log 2 (1 / 4)))

-- Proof problem statement
theorem order_of_a_b_c
  (h1 : odd_function f) 
  (h2 : condition f) :
  c > a ∧ a > b := 
sorry

end order_of_a_b_c_l730_730538


namespace alexa_fractions_l730_730704

theorem alexa_fractions (alexa_days ethans_days : ℕ) 
  (h1 : alexa_days = 9) (h2 : ethans_days = 12) : 
  alexa_days / ethans_days = 3 / 4 := 
by 
  sorry

end alexa_fractions_l730_730704


namespace sufficient_not_necessary_condition_l730_730603

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x < 1 → x < 2) ∧ ¬ (x < 2 → x < 1) :=
by
  sorry

end sufficient_not_necessary_condition_l730_730603


namespace probability_A_in_swimming_pool_l730_730575
-- Ensure necessary libraries are imported

open ProbabilityTheory

-- Define the problem conditions in Lean 4
theorem probability_A_in_swimming_pool :
  let num_volunteers := 5
  let venues := ["gymnasium", "swimming pool", "comprehensive training hall"]
  let volunteers := ["A", "B", "C", "D", "E"]
  let venues_count := venues.length
  let volunteer_count := volunteers.length
  let same_venue_condition (a b : String) (assignment : String → String) : Prop := assignment a = assignment b
  let at_least_one_each_venue (assignment : String → String) : Prop :=
    ∀ v ∈ venues, ∃ vol ∈ volunteers, assignment vol = v

  -- considering all assignments where A and B are in the same venue
  let valid_assignment : Type := {f : String → String // same_venue_condition "A" "B" f ∧ at_least_one_each_venue f}

  -- The probability calculation
  let probability_A_swimming_pool :=
    let total := {f // at_least_one_each_venue f}.length
    let favorable :=
      (filter (λ f : valid_assignment, f.val "A" = "swimming pool") (finset.univ : finset valid_assignment)).card
    favorable / total
    
  -- Show the probability is 1/3
  probability_A_swimming_pool = 1 / 3 :=
sorry

end probability_A_in_swimming_pool_l730_730575


namespace Douglas_weight_correct_l730_730713

def Anne_weight : ℕ := 67
def weight_diff : ℕ := 15
def Douglas_weight : ℕ := 52

theorem Douglas_weight_correct : Douglas_weight = Anne_weight - weight_diff := by
  sorry

end Douglas_weight_correct_l730_730713


namespace connie_correct_answer_l730_730342

theorem connie_correct_answer (y : ℕ) (h1 : y - 8 = 32) : y + 8 = 48 := by
  sorry

end connie_correct_answer_l730_730342


namespace algebraic_expression_value_l730_730557

theorem algebraic_expression_value
  (x y : ℚ)
  (h : |2 * x - 3 * y + 1| + (x + 3 * y + 5)^2 = 0) :
  (-2 * x * y)^2 * (-y^2) * 6 * x * y^2 = 192 :=
  sorry

end algebraic_expression_value_l730_730557


namespace smallest_k_divisible_by_9999_mn_length_l730_730380

-- Problem 1
theorem smallest_k_divisible_by_9999 : 
  ∃ (k : ℕ), (k > 0) ∧ (∀ n, (∃ (m : ℕ), n = (10 ^ k - 1) / 9) → 9999 ∣ n) ∧ k = 180 :=
  sorry

noncomputable def T : ℝ := sorry -- Not given in problem, assumed to exist

-- Setup for Geometric Problem
variables {ω₁ ω₂ : Type} [metric_space ω₁] [metric_space ω₂]
variables (P Q A B X Y M N : ω₁)
variables (AQ BQ AB : ℝ)

-- Problem 2
theorem mn_length {T : ℝ} (hAQ : AQ = real.sqrt T) (hBQ : BQ = 7) (hAB : AB = 8)
                 (hMidM : M = midpoint A Y) (hMidN : N = midpoint B X) :
                 dist M N = 128 / 21 :=
  sorry

end smallest_k_divisible_by_9999_mn_length_l730_730380


namespace g_correct_l730_730135

def g (x : ℝ) : ℝ := sorry

axiom g_zero : g 0 = 2
axiom g_property : ∀ (x y : ℝ), g (xy) = g ((x^2 + y^2) / 2) + 3 * (x - y)^2

theorem g_correct : ∀ x : ℝ, g x = (3 / 2) * x^2 - 6 * x + (3 / 2) := 
by 
  sorry

end g_correct_l730_730135


namespace average_of_data_is_six_l730_730800

def data : List ℕ := [4, 6, 5, 8, 7, 6]

theorem average_of_data_is_six : 
  (data.sum / data.length : ℚ) = 6 := 
by sorry

end average_of_data_is_six_l730_730800


namespace number_of_parallel_lines_l730_730475

theorem number_of_parallel_lines (n : ℕ) 
  (intersecting_lines : ℕ := 8)
  (parallelograms : ℕ := 420) :
  (nat.choose n 2) * (nat.choose intersecting_lines 2) = parallelograms → n = 6 :=
by
  sorry

end number_of_parallel_lines_l730_730475


namespace average_marks_in_second_exam_l730_730862

theorem average_marks_in_second_exam 
  (total_marks_each : ℕ) 
  (num_exams : ℕ) 
  (avg_first_exam : ℕ) 
  (marks_third_exam : ℕ) 
  (desired_avg : ℕ) 
  (total_marks : ℕ) 
  (total_first_exam : ℕ)
  (total_needed_marks : ℕ)
  : ( (total_needed_marks - (total_first_exam + marks_third_exam)) / total_marks_each * 100 = 55 ) :=
begin
  -- Define constants
  let total_marks := 3 * total_marks_each,
  let needed_marks := (desired_avg * total_marks) / 100,
  let first_exam_marks := (avg_first_exam * total_marks_each) / 100,
  
  -- Calculate marks needed in second exam
  let second_exam_marks := needed_marks - (first_exam_marks + marks_third_exam),
  
  -- Find average percentage
  let second_exam_avg := (second_exam_marks * 100) / total_marks_each,
  
  -- Prove the result
  exact (second_exam_avg = 55),
  sorry
end

end average_marks_in_second_exam_l730_730862


namespace mr_castiel_sausages_l730_730938

theorem mr_castiel_sausages (S : ℕ) :
  S * (3 / 5) * (1 / 2) * (1 / 4) * (3 / 4) = 45 → S = 600 :=
by
  sorry

end mr_castiel_sausages_l730_730938


namespace boxes_needed_to_sell_l730_730902

theorem boxes_needed_to_sell (total_bars : ℕ) (bars_per_box : ℕ) (target_boxes : ℕ) (h₁ : total_bars = 710) (h₂ : bars_per_box = 5) : target_boxes = 142 :=
by
  sorry

end boxes_needed_to_sell_l730_730902


namespace single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l730_730101

section transmission_scheme

variables (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1)

-- Part A
theorem single_transmission_probability :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β) ^ 2 :=
by sorry

-- Part B
theorem triple_transmission_probability :
  (1 - β) * β * (1 - β) = β * (1 - β) ^ 2 :=
by sorry

-- Part C
theorem triple_transmission_decoding :
  (3 * β * (1 - β) ^ 2) + (1 - β) ^ 3 = β * (1 - β) ^ 2 + (1 - β) ^ 3 :=
by sorry

-- Part D
theorem decoding_comparison (h : 0 < α ∧ α < 0.5) :
  (1 - α) < (3 * α * (1 - α) ^ 2 + (1 - α) ^ 3) :=
by sorry

end transmission_scheme

end single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l730_730101


namespace solution_set_f_greater_x_l730_730000

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_ℝ : ∀ x : ℝ, f x ≠ ⊥
axiom f_at_1 : f 1 = 1
axiom f_prime_gt_1 : ∀ x : ℝ, deriv f x > 1

theorem solution_set_f_greater_x : {x : ℝ | f x > x} = Ioi 1 := by
  sorry

end solution_set_f_greater_x_l730_730000


namespace original_pension_value_l730_730276

noncomputable def original_annual_pension : ℝ :=
  let k : ℝ := sorry           -- constant multiplier undefined
  let x : ℝ := sorry           -- original years worked undefined
  if h1 : k * real.sqrt (x + 3) = k * real.sqrt x + 100
  if h2 : k * real.sqrt (x + 5) = k * real.sqrt x + 160
  then (k * real.sqrt x)
  else 0

theorem original_pension_value :
  ∃ (k x : ℝ), (k * real.sqrt (x + 3) = k * real.sqrt x + 100) ∧ 
               (k * real.sqrt (x + 5) = k * real.sqrt x + 160) ∧ 
               (k * real.sqrt x = 670) :=
  sorry

end original_pension_value_l730_730276


namespace area_of_triangle_PQR_l730_730593

noncomputable def point := (ℝ × ℝ)

def reflect_over_y_axis (p : point) : point :=
  (-p.1, p.2)

def reflect_over_line_y_eq_neg_x (p : point) : point :=
  (-p.2, -p.1)

def distance (p q : point) : ℝ :=
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

def triangle_area (p q r : point) : ℝ :=
  let base := distance p q
  let height := abs (r.2 - p.2)
  1/2 * base * height

def P := (4, 5) : point
def Q := reflect_over_y_axis P
def R := reflect_over_line_y_eq_neg_x Q

theorem area_of_triangle_PQR : triangle_area P Q R = 36 := by
  sorry

end area_of_triangle_PQR_l730_730593


namespace a_1000_value_l730_730491

def seq (n : ℕ) : ℤ :=
  if n = 1 then 4014
  else if n = 2 then 4015
  else sorry  -- Recurrence relation implementation skipped

theorem a_1000_value :
  (let a_n := seq in
   a_n 1 = 4014 ∧ a_n 2 = 4015 ∧ (∀ n ≥ 1, a_n n + a_n (n + 1) + a_n (n + 2) = 2 * n)
   → a_n 1000 = 4680) :=
by
  sorry

end a_1000_value_l730_730491


namespace douglas_weight_proof_l730_730712

theorem douglas_weight_proof : 
  ∀ (anne_weight douglas_weight : ℕ), 
  anne_weight = 67 →
  anne_weight = douglas_weight + 15 →
  douglas_weight = 52 :=
by 
  intros anne_weight douglas_weight h1 h2 
  sorry

end douglas_weight_proof_l730_730712


namespace num_four_digit_not_div_by_5_correct_l730_730108

noncomputable def num_four_digit_not_div_by_5 : ℕ :=
  let total := (Finset.perm_of_finset (finset.range 6)).filter (λ l, l.length = 4) in
  let not_end_in_0_or_5 := total.filter (λ l, l.last ∈ {0, 5}) in
  total.card - not_end_in_0_or_5.card

theorem num_four_digit_not_div_by_5_correct :
  num_four_digit_not_div_by_5 = 192 :=
by
  sorry

end num_four_digit_not_div_by_5_correct_l730_730108


namespace rita_bought_4_pounds_l730_730165

variable (total_amount : ℝ) (cost_per_pound : ℝ) (amount_left : ℝ)

theorem rita_bought_4_pounds (h1 : total_amount = 70)
                             (h2 : cost_per_pound = 8.58)
                             (h3 : amount_left = 35.68) :
  (total_amount - amount_left) / cost_per_pound = 4 := 
  by
  sorry

end rita_bought_4_pounds_l730_730165


namespace pipe_length_difference_l730_730665

theorem pipe_length_difference 
  (total_length shorter_piece : ℝ)
  (h1: total_length = 68) 
  (h2: shorter_piece = 28) : 
  ∃ longer_piece diff : ℝ, longer_piece = total_length - shorter_piece ∧ diff = longer_piece - shorter_piece ∧ diff = 12 :=
by
  sorry

end pipe_length_difference_l730_730665


namespace triangle_sides_condition_l730_730832

noncomputable def f (x λ : ℝ) : ℝ := x + 2 * Real.cos x + λ

theorem triangle_sides_condition (λ : ℝ) :
  (∀ x₁ x₂ x₃ ∈ Icc 0 (Real.pi / 2), (f x₁ λ + f x₂ λ > f x₃ λ) ∧ (f x₁ λ + f x₃ λ > f x₂ λ) ∧ (f x₂ λ + f x₃ λ > f x₁ λ)) ↔
  λ > Real.sqrt 3 - 5 * Real.pi / 6 :=
sorry

end triangle_sides_condition_l730_730832


namespace other_religion_students_l730_730092

theorem other_religion_students (total_students : ℕ) 
  (muslims_percent hindus_percent sikhs_percent christians_percent buddhists_percent : ℝ) 
  (h1 : total_students = 1200) 
  (h2 : muslims_percent = 0.35) 
  (h3 : hindus_percent = 0.25) 
  (h4 : sikhs_percent = 0.15) 
  (h5 : christians_percent = 0.10) 
  (h6 : buddhists_percent = 0.05) : 
  ∃ other_religion_students : ℕ, other_religion_students = 120 :=
by
  sorry

end other_religion_students_l730_730092


namespace restaurant_problem_l730_730329

theorem restaurant_problem (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 :=
by
  sorry

end restaurant_problem_l730_730329


namespace commission_percentage_is_4_l730_730707

-- Define the given conditions
def commission := 12.50
def total_sales := 312.5

-- The problem is to prove the commission percentage
theorem commission_percentage_is_4 :
  (commission / total_sales) * 100 = 4 := by
  sorry

end commission_percentage_is_4_l730_730707


namespace timothy_read_pages_l730_730230

theorem timothy_read_pages 
    (mon_tue_pages : Nat) (wed_pages : Nat) (thu_sat_pages : Nat) 
    (sun_read_pages : Nat) (sun_review_pages : Nat) : 
    mon_tue_pages = 45 → wed_pages = 50 → thu_sat_pages = 40 → sun_read_pages = 25 → sun_review_pages = 15 →
    (2 * mon_tue_pages + wed_pages + 3 * thu_sat_pages + sun_read_pages + sun_review_pages = 300) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end timothy_read_pages_l730_730230


namespace order_of_variables_l730_730017

section
variable (a b c : ℝ)

-- Define the conditions
def condition_a : a = Real.log 64 / Real.log 16 := sorry
def condition_b : b = Real.log10 0.2 := sorry
def condition_c : c = 2 ^ 0.2 := sorry

-- Theorem stating the desired order
theorem order_of_variables (h_a : a = Real.log 64 / Real.log 16) 
                          (h_b : b = Real.log10 0.2) 
                          (h_c : c = 2 ^ 0.2) : 
                          b < c ∧ c < a := 
sorry

end

end order_of_variables_l730_730017


namespace corn_height_growth_l730_730724

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l730_730724


namespace intersection_is_0_1_l730_730350

-- Define the sets A and B as given conditions
def A : Set ℝ := {x | ∀ f, f x = Real.sqrt (2^x - 1)}
def B : Set ℝ := {y | ∀ x, y = Real.log_base 2 (2^x + 2)}

-- Define the complement of B in ℝ
def compl_B : Set ℝ := {x | x ≤ 1}

-- Define the intersection of A and the complement of B
def intersection : Set ℝ := {x | x ∈ A ∧ x ≤ 1}

-- Prove the statement that the intersection is [0, 1]
theorem intersection_is_0_1 : intersection = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end intersection_is_0_1_l730_730350


namespace polynomial_form_l730_730541

open Nat

noncomputable
def polynomial_with_integer_coefficients (f : ℤ → ℤ) : Prop :=
  ∃ p : Polynomial ℤ, ∀ x, f x = p.eval x

theorem polynomial_form {f : ℤ → ℤ} (h_polynomial : polynomial_with_integer_coefficients f) :
  (∀ p u v : ℤ, Prime p → p ∣ (u * v + u + v) → p ∣ (f u * f v - 1)) →
  ∃ n : ℕ, f = λ x => (x + 1) ^ n ∨ f = λ x => -((x + 1) ^ n) :=
by
  sorry

end polynomial_form_l730_730541


namespace stratified_random_sampling_l730_730673

theorem stratified_random_sampling
 (junior_students senior_students total_sample_size : ℕ)
 (junior_high_count senior_high_count : ℕ) 
 (total_sample : junior_high_count + senior_high_count = total_sample_size)
 (junior_students_ratio senior_students_ratio : ℕ)
 (ratio : junior_students_ratio + senior_students_ratio = 1)
 (junior_condition : junior_students_ratio = 2 * senior_students_ratio)
 (students_distribution : junior_students = 400 ∧ senior_students = 200 ∧ total_sample_size = 60)
 (combination_junior : (nat.choose junior_students junior_high_count))
 (combination_senior : (nat.choose senior_students senior_high_count)) :
 combination_junior * combination_senior = nat.choose 400 40 * nat.choose 200 20 :=
by
  sorry

end stratified_random_sampling_l730_730673


namespace problem_statement_l730_730072

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2004 = 2005 :=
sorry

end problem_statement_l730_730072


namespace largest_possible_integer_sum_of_10_distinct_is_55_l730_730221

theorem largest_possible_integer_sum_of_10_distinct_is_55 :
  ∃ (a b c d e f g h i j : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g ∧ g ≠ h ∧ h ≠ i ∧ i ≠ j ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧ g ≠ i ∧ g ≠ j ∧ h ≠ j ∧
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0 ∧ j ≠ 0 ∧
  a + b + c + d + e + f + g + h + i + j = 100 ∧ max a b c d e f g h i j = 55 := by
  sorry

end largest_possible_integer_sum_of_10_distinct_is_55_l730_730221


namespace find_1_in_less_than_3n_turns_for_2k_minus_1_l730_730524

noncomputable def find_1_condition (n : ℕ) (A : Array (Array ℕ)) : Prop :=
  (n ≥ 3) ∧
  (A.size = n ∧ A[0].size = n) ∧
  (∀ i j, 0 ≤ i ∧ i < n - 1 ∧ 0 ≤ j ∧ j < n → (A[i][j] + 1 = A[i+1][j] ∨ A[i][j] + 1 = A[i][j+1])) ∧
  (∀ k, 1 ≤ k ∧ k ≤ n^2 - 1 → ∃ i j, A[i][j] = k)
  
theorem find_1_in_less_than_3n_turns_for_2k_minus_1 (k : ℕ) (h : k ≥ 1) 
  (A : Array (Array ℕ)) : find_1_condition (2^k - 1) A →
  ∃ S : Finset (Fin (2^k - 1) × Fin (2^k - 1)), S.card < 3 * (2^k - 1) ∧ (1 ∈ S.image (λ (p : Fin (2^k - 1) × Fin (2^k - 1)), A[p.fst][p.snd])) :=
begin
  sorry
end

end find_1_in_less_than_3n_turns_for_2k_minus_1_l730_730524


namespace pine_cones_on_roof_l730_730310

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end pine_cones_on_roof_l730_730310


namespace find_ellipse_eq_l730_730804

noncomputable def ellipse_eq (a b c : ℝ) : Prop := 
  ∀ (x y : ℝ), (x = 2 ∧ y = real.sqrt 3) → 
  a > b ∧ b > 0 ∧
  dist (2, real.sqrt 3) (a, 0) + dist (a, 0) (-a, 0) = dist (2, real.sqrt 3) (-a, 0) + 2 * dist (a, 0) (-a, 0) ∧
  a^2 = b^2 + c^2 ∧
  4 / a^2 + 3 / b^2 = 1

theorem find_ellipse_eq :
  ∃ (a b c : ℝ), ellipse_eq a b c ∧ (∀ (x y : ℝ), (x^2 / 8 + y^2 / 6 = 1)) :=
by
  sorry

end find_ellipse_eq_l730_730804


namespace sec_neg_330_eq_l730_730729

-- Definitions based on conditions
def sec (θ : Real) : Real := 1 / Real.cos θ
def cosine (θ : Real) : Real := Real.cos θ

lemma cosine_neg (θ : Real) : Real.cos (-θ) = Real.cos θ :=
  by sorry

lemma cosine_30 : Real.cos 30 =  sqrt 3 / 2 :=
  by sorry

-- Proof problem statement
theorem sec_neg_330_eq : sec (-330) = 2 * sqrt 3 / 3 :=
  by
    have h1 : sec (-330) = 1 / cosine (-330) := by refl
    have h2 : cosine (-330) = cosine 30 := cosine_neg 330
    have h3 : cosine 30 = sqrt 3 / 2 := cosine_30
    sorry

end sec_neg_330_eq_l730_730729


namespace find_a_l730_730786

open Complex

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the main hypothesis: (ai / (1 - i)) = (-1 + i)
def hypothesis (a : ℂ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Now, we state the theorem we need to prove
theorem find_a (a : ℝ) (ha : hypothesis a) : a = 2 := by
  sorry

end find_a_l730_730786


namespace train_length_l730_730301

/- 
Question: What is the length of the train?
Conditions: 
- The train takes 6 seconds to cross a man walking at 5 kmph.
- The man is walking in a direction opposite to that of the train.
- The speed of the train is 24.997600191984645 kmph.
Answer: The length of the train is 49.996 meters.
-/
theorem train_length
  (train_speed_kmph : ℝ)
  (man_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (train_speed_given : train_speed_kmph = 24.997600191984645)
  (man_speed_given : man_speed_kmph = 5)
  (time_given : time_seconds = 6) :
  ∃ L : ℝ, L = 49.996 :=
by {
  -- Define the relative speed in kmph.
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph,
  have rel_speed_calc : relative_speed_kmph = 24.997600191984645 + 5,
  -- Convert relative speed to m/s.
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600),
  have rel_speed_conv : relative_speed_mps = (24.997600191984645 + 5) * (5 / 18),
  -- Calculate the length.
  let train_length := relative_speed_mps * time_seconds,
  have train_length_calc : train_length = ((24.997600191984645 + 5) * (5 / 18)) * 6,
  -- The answer is the length L.
  exact ⟨train_length, by linarith⟩,
}

end train_length_l730_730301


namespace grid_has_duplicate_l730_730863

-- Define the size of the grid
def n : ℕ := 10

-- Define the condition that each pair of adjacent integers differ by at most 5
def adjacent_difference_condition (tbl : Fin n × Fin n → ℤ) : Prop :=
  ∀ (i j : Fin n × Fin n), (|tbl i - tbl j| ≤ 5) ∧ (
    (i.1 = j.1 ∧ (i.2 ≠ j.2 ∧ |i.2.val - j.2.val| = 1)) ∨ 
    (i.2 = j.2 ∧ (i.1 ≠ j.1 ∧ |i.1.val - j.1.val| = 1))
  )

-- The theorem statement
theorem grid_has_duplicate (tbl : Fin n × Fin n → ℤ) : 
  adjacent_difference_condition tbl → ∃ (i j : Fin n × Fin n), i ≠ j ∧ tbl i = tbl j :=
by
  sorry

end grid_has_duplicate_l730_730863


namespace John_make_money_l730_730518

def trees_planted (rows : ℕ) (columns : ℕ) : ℕ := rows * columns

def apples_produced (trees : ℕ) (apples_per_tree : ℕ) : ℕ := trees * apples_per_tree

def money_made (apples : ℝ) (price_per_apple : ℝ) : ℝ := apples * price_per_apple

theorem John_make_money (rows columns apples_per_tree : ℕ) (price_per_apple : ℝ) :
  let trees := trees_planted rows columns in
  let apples := apples_produced trees apples_per_tree in
  money_made apples price_per_apple = 30 :=
by
  sorry

end John_make_money_l730_730518


namespace remaining_wallpaper_removal_time_l730_730755

theorem remaining_wallpaper_removal_time (dining_walls living_walls : ℕ) (time_per_wall: ℕ) (time_spent: ℕ) :
  dining_walls = 4 →
  living_walls = 4 →
  time_per_wall = 2 →
  time_spent = 2 →
  time_per_wall * dining_walls + time_per_wall * living_walls - time_spent = 14 :=
by
  intros hd hl ht hs
  rw [hd, hl, ht, hs]
  exact dec_trivial

end remaining_wallpaper_removal_time_l730_730755


namespace third_group_members_l730_730680

-- Define the total number of members in the choir
def total_members : ℕ := 70

-- Define the number of members in the first group
def first_group_members : ℕ := 25

-- Define the number of members in the second group
def second_group_members : ℕ := 30

-- Prove that the number of members in the third group is 15
theorem third_group_members : total_members - first_group_members - second_group_members = 15 := 
by 
  sorry

end third_group_members_l730_730680


namespace sum_of_angles_hypotenuse_visible_l730_730292

theorem sum_of_angles_hypotenuse_visible (A B C D M N : Point) (h_sq : square ABCD)
  (h_right_triangle : right_triangle AMN)
  (h_triangle_sum : AN + AM = AB)
  (h_points_on_sides : M ∈ segment A B ∧ N ∈ segment A D) :
  (angle AMC + angle DNC + angle BNC = 90) :=
sorry

end sum_of_angles_hypotenuse_visible_l730_730292


namespace probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730642

open ProbabilityTheory

noncomputable def Vanya_probability_more_heads_than_Tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  X_V 3 = binomial 3 (1/2) ∧
  X_T 2 = binomial 2 (1/2) ∧
  Pr(X_V > X_T) = 1/2

noncomputable def Vanya_probability_more_heads_than_Tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  ∀ n : ℕ,
  X_V (n+1) = binomial (n+1) (1/2) ∧
  X_T n = binomial n (1/2) ∧
  Pr(X_V > X_T) = 1/2

theorem probability_heads_vanya_tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_a X_V X_T := sorry

theorem probability_heads_vanya_tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_b X_V X_T := sorry

end probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730642


namespace vector_b_magnitude_l730_730057

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_b_magnitude (b : ℝ × ℝ) (h1 : (2, 1).1 * b.1 + (2, 1).2 * b.2 = 10)
  (h2 : vector_magnitude (2 + b.1, 1 + b.2) = 5 * sqrt 2) : vector_magnitude b = 5 :=
by
  let a := (2, 1)
  have ha_magnitude : vector_magnitude a = sqrt (2 ^ 2 + 1 ^ 2) := rfl
  have ha_value : sqrt (4 + 1) = sqrt 5 := rfl
  let sum := (a.1 + b.1, a.2 + b.2)
  have h_sum_magnitude : vector_magnitude sum = 5 * sqrt 2 := h2
  sorry

end vector_b_magnitude_l730_730057


namespace correct_statements_l730_730779

noncomputable def f (x : ℝ) : ℝ := Real.sin (abs x) + abs (Real.cos x)

lemma statement_1 : ∀ x, f (-x) = f x :=
by
  sorry

lemma statement_2 : ¬ ∀ x ∈ Ioc (Real.pi / 2) Real.pi, f x ≤ f (x - 1) :=
by
  sorry

lemma statement_3 : Icc 1 (Real.sqrt 2) = {y : ℝ | ∃ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2), f x = y} :=
by
  sorry

lemma statement_4 : ∀ x ∈ Ioo (5 * Real.pi / 4) (7 * Real.pi / 4), f x < 0 :=
by
  sorry

theorem correct_statements : true :=
by
  have h1 : ∀ x, f (-x) = f x := statement_1
  have h2 : ¬ ∀ x ∈ Ioc (Real.pi / 2) Real.pi, f x ≤ f (x - 1) := statement_2
  have h3 : Icc 1 (Real.sqrt 2) = {y : ℝ | ∃ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2), f x = y} := statement_3
  have h4 : ∀ x ∈ Ioo (5 * Real.pi / 4) (7 * Real.pi / 4), f x < 0 := statement_4
  trivial

end correct_statements_l730_730779


namespace ellipse_y_intercept_range_l730_730415

theorem ellipse_y_intercept_range
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) (h3 : e = 1 / 2)
  (M : ℝ × ℝ) (h4 : (M.1 ^ 2) / (a ^ 2) + (M.2 ^ 2) / (b ^ 2) = 1)
  (h5 : 2 * a = 4)
  (N : ℝ × ℝ) (h6 : N = (4, 0))
  (k : ℝ) (h7 : k > 0)
  (A B : ℝ × ℝ) (l : Set (ℝ × ℝ))
  (intersect_cond : ∃ x1 y1 x2 y2, (y1 = k * (x1 - 4)) ∧ ((x1 ^ 2) / 4 + (y1 ^ 2) / 3 = 1) ∧ 
                                  (y2 = k * (x2 - 4)) ∧ ((x2 ^ 2) / 4 + (y2 ^ 2) / 3 = 1))
  (midpoint : ℝ × ℝ) (h8 : midpoint = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (perp_bisector : Set (ℝ × ℝ)) (h9 : perp_bisector = {p | p.2 - midpoint.2 = - (1 / k) * (p.1 - midpoint.1)})
  (y_intercept : ℝ) (h10 : y_intercept = midpoint.2 - (1 / k) * midpoint.1) :
  y_intercept ∈ Ioo (-1 / 2) (1 / 2) :=
sorry

end ellipse_y_intercept_range_l730_730415


namespace area_enclosed_lines_curve_is_2ln2_l730_730576

theorem area_enclosed_lines_curve_is_2ln2 :
  ∫ x in (1/2) .. 2, (1/x) = 2 * Real.log 2 := by
  sorry

end area_enclosed_lines_curve_is_2ln2_l730_730576


namespace solution_set_of_log_inequality_l730_730533

noncomputable def f (a x : ℝ) := a * x ^ 2 + x + 1

theorem solution_set_of_log_inequality (a x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∃ x, f a x = has_max f) :
  log a (x - 1) > 0 ↔ 1 < x ∧ x < 2 :=
sorry

end solution_set_of_log_inequality_l730_730533


namespace probability_sum_m_n_reaches_4_4_l730_730569

theorem probability_sum_m_n_reaches_4_4 :
  let p : ℚ := (1575 : ℚ) / 262144 in
  p.denom.coprime p.num ∧ (p.num + p.denom = 263719) := 
by
  sorry

end probability_sum_m_n_reaches_4_4_l730_730569


namespace sum_of_sequence_b_l730_730835

theorem sum_of_sequence_b (n : ℕ) : 
  let b (k : ℕ) := k * (k + 1) * (k + 2) in
  (∑ k in Finset.range n, b (k + 1)) = (n * (n + 1) * (n + 2) * (n + 3)) / 4 :=
by
  let b (k : ℕ) := k * (k + 1) * (k + 2)
  sorry

end sum_of_sequence_b_l730_730835


namespace sum_of_distances_in_regular_ngon_constant_l730_730162

theorem sum_of_distances_in_regular_ngon_constant
  (n : ℕ) (a S : ℝ)
  (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ)
  (h : Fin n → ℝ)
  (h_area : (∑ i : Fin n, 1/2 * a * h i) = S)
  : (∑ i : Fin n, h i) = 2 * S / a := 
sorry

end sum_of_distances_in_regular_ngon_constant_l730_730162


namespace smallest_possible_number_of_integers_in_domain_of_g_l730_730238

def g : ℕ → ℕ
| n => if n % 2 = 0 then n / 2 else 3 * n + 1

theorem smallest_possible_number_of_integers_in_domain_of_g :
  let domain := {n | ∃ m, (g^[m]) 12 = n}
  set.card domain = 23 := by
  sorry

end smallest_possible_number_of_integers_in_domain_of_g_l730_730238


namespace probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730639

open ProbabilityTheory

noncomputable def Vanya_probability_more_heads_than_Tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  X_V 3 = binomial 3 (1/2) ∧
  X_T 2 = binomial 2 (1/2) ∧
  Pr(X_V > X_T) = 1/2

noncomputable def Vanya_probability_more_heads_than_Tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  ∀ n : ℕ,
  X_V (n+1) = binomial (n+1) (1/2) ∧
  X_T n = binomial n (1/2) ∧
  Pr(X_V > X_T) = 1/2

theorem probability_heads_vanya_tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_a X_V X_T := sorry

theorem probability_heads_vanya_tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_b X_V X_T := sorry

end probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730639


namespace max_ab_eq_quarter_l730_730138

theorem max_ab_eq_quarter (a b : ℝ) (f : ℝ → ℝ)
(hf : ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), abs (f x) ≤ 1)
(hf_def : ∀ x, f x = a * x + b) : a * b ≤ 1 / 4 :=
sorry

end max_ab_eq_quarter_l730_730138


namespace angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l730_730112

open Real

variable {A B C a b c : ℝ}
variable (AM BM MC : ℝ)

-- Conditions
axiom triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)
axiom BM_MC_relation : BM = (1 / 2) * MC

-- Part 1: Measure of angle A
theorem angle_A_is_pi_over_3 (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) : 
  A = π / 3 :=
by sorry

-- Part 2: Minimum value of |AM|^2 / S
noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * sin A

axiom condition_b_eq_2c : b = 2 * c

theorem minimum_value_AM_sq_div_S (AM BM MC : ℝ) (S : ℝ) (H : BM = (1 / 2) * MC) 
  (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) 
  (area : S = area_triangle a b c A)
  (condition_b_eq_2c : b = 2 * c) : 
  (AM ^ 2) / S ≥ (8 * sqrt 3) / 9 :=
by sorry

end angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l730_730112


namespace max_grapes_leftover_l730_730272

-- Define variables and conditions
def total_grapes (n : ℕ) : ℕ := n
def kids : ℕ := 5
def grapes_leftover (n : ℕ) : ℕ := n % kids

-- The proposition we need to prove
theorem max_grapes_leftover (n : ℕ) (h : n ≥ 5) : grapes_leftover n = 4 :=
sorry

end max_grapes_leftover_l730_730272


namespace decreasing_function_range_l730_730969

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y ∧ x ∈ set.Iic 4 → f x ≥ f y)
  ↔ a ≤ -3 :=
by
  let f := λ x : ℝ, x^2 + 2*(a-1)*x + 2
  refine ⟨λ h, _, λ h x y hxy, _⟩
  sorry

end decreasing_function_range_l730_730969


namespace power_equality_l730_730068

theorem power_equality (x : ℝ) (h : (16 : ℝ)^8 = (8 : ℝ)^x) : 2^(-x) = (1 / 2^(32/3 : ℝ)) :=
by
  sorry

end power_equality_l730_730068


namespace red_peppers_weight_l730_730347

theorem red_peppers_weight :
    ∀ (total_weight green_weight : ℝ),
    total_weight = 5.666666666666667 →
    green_weight = 2.8333333333333335 →
    total_weight - green_weight = 2.8333333333333335 :=
by
  intros total_weight green_weight h_total h_green
  rw [h_total, h_green]
  norm_num
  sorry

end red_peppers_weight_l730_730347


namespace division_of_natural_numbers_to_identical_sequences_impossible_l730_730512

theorem division_of_natural_numbers_to_identical_sequences_impossible:
  ∀ k : ℕ, ∀ (A B : finset ℕ), (∀ x, x ∈ finset.range (k + 1) → x ∈ A ∨ x ∈ B) →
  (∀ a b, a ∈ A → b ∈ B → a ≠ b) →
  (A.card = B.card) → ¬(A.val = B.val) :=
by
  sorry

end division_of_natural_numbers_to_identical_sequences_impossible_l730_730512


namespace only_solution_l730_730366

theorem only_solution (a b c : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
    (h_le : a ≤ b ∧ b ≤ c) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_div_a2b : a^3 + b^3 + c^3 % (a^2 * b) = 0)
    (h_div_b2c : a^3 + b^3 + c^3 % (b^2 * c) = 0)
    (h_div_c2a : a^3 + b^3 + c^3 % (c^2 * a) = 0) : 
    a = 1 ∧ b = 1 ∧ c = 1 :=
  by
  sorry

end only_solution_l730_730366


namespace remaining_wallpaper_removal_time_l730_730754

theorem remaining_wallpaper_removal_time (dining_walls living_walls : ℕ) (time_per_wall: ℕ) (time_spent: ℕ) :
  dining_walls = 4 →
  living_walls = 4 →
  time_per_wall = 2 →
  time_spent = 2 →
  time_per_wall * dining_walls + time_per_wall * living_walls - time_spent = 14 :=
by
  intros hd hl ht hs
  rw [hd, hl, ht, hs]
  exact dec_trivial

end remaining_wallpaper_removal_time_l730_730754


namespace remainder_of_sum_div_11_is_9_l730_730376

def seven_times_ten_pow_twenty : ℕ := 7 * 10 ^ 20
def two_pow_twenty : ℕ := 2 ^ 20
def sum : ℕ := seven_times_ten_pow_twenty + two_pow_twenty

theorem remainder_of_sum_div_11_is_9 : sum % 11 = 9 := by
  sorry

end remainder_of_sum_div_11_is_9_l730_730376


namespace symmetry_relative_to_origin_symmetry_relative_to_point_l730_730413

-- Symmetry Relative to the Origin
theorem symmetry_relative_to_origin (x y : ℝ) :
  ∃ x' y' : ℝ, x' = -x ∧ y' = -y :=
by
  use [-x, -y]
  constructor
  · rfl
  · rfl

-- Symmetry Relative to a Point K(a, b)
theorem symmetry_relative_to_point (x y a b : ℝ) :
  ∃ x' y' : ℝ, x' = 2 * a - x ∧ y' = 2 * b - y :=
by
  use [2 * a - x, 2 * b - y]
  constructor
  · rfl
  · rfl

end symmetry_relative_to_origin_symmetry_relative_to_point_l730_730413


namespace isosceles_trapezoid_tangent_ratios_l730_730710

theorem isosceles_trapezoid_tangent_ratios
  {A B C D P Q R S : Point}
  (h_trapezoid : IsoscelesTrapezoid A B C D)
  (h_circumscribed : CircumscribedCircle A B C D)
  (h_tangent_intersections :
    TangentIntersectCircleAtPoints P Q R S)
  : (PQ / QR) = (RS / SR) :=
sorry

end isosceles_trapezoid_tangent_ratios_l730_730710


namespace wallpaper_removal_time_l730_730758

theorem wallpaper_removal_time (time_per_wall : ℕ) (dining_room_walls_remaining : ℕ) (living_room_walls : ℕ) :
  time_per_wall = 2 → dining_room_walls_remaining = 3 → living_room_walls = 4 → 
  time_per_wall * (dining_room_walls_remaining + living_room_walls) = 14 :=
by
  sorry

end wallpaper_removal_time_l730_730758


namespace focal_length_of_ellipse_l730_730583

theorem focal_length_of_ellipse :
  (∀ θ : ℝ, (x = sqrt 2 * Real.cos θ ∧ y = Real.sin θ) → 2 * sqrt ((sqrt 2)^2 - 1^2) = 2) :=
sorry

end focal_length_of_ellipse_l730_730583


namespace logarithm_solution_set_l730_730530

theorem logarithm_solution_set (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log a (x - 1) > 0) ↔ x ∈ (1, 2) :=
by
  sorry

end logarithm_solution_set_l730_730530


namespace problem_statement_l730_730929

noncomputable def g : ℝ → ℝ
| x => if x < 0 then -x
            else if x < 5 then x + 3
            else 2 * x ^ 2

theorem problem_statement : g (-6) + g 3 + g 8 = 140 :=
by
  -- Proof goes here
  sorry

end problem_statement_l730_730929


namespace smallest_number_remainders_l730_730659

theorem smallest_number_remainders :
  ∃ n : ℕ, 
    (n % 2 = 1) ∧
    (n % 3 = 2) ∧
    (n % 4 = 3) ∧
    (n % 5 = 4) ∧
    (n % 6 = 5) ∧
    (∀ m : ℕ, 
      (m % 2 = 1) ∧
      (m % 3 = 2) ∧
      (m % 4 = 3) ∧
      (m % 5 = 4) ∧
      (m % 6 = 5) →
      n ≤ m) :=
begin
  use 59,
  split,
  { exact nat.mod_eq_of_lt 1 0 2, },
  split,
  { exact nat.mod_eq_of_lt 2 0 3, },
  split,
  { exact nat.mod_eq_of_lt 3 0 4, },
  split,
  { exact nat.mod_eq_of_lt 4 0 5, },
  split,
  { exact nat.mod_eq_of_lt 5 0 6, },
  intros m hm,
  cases hm with h2 hm,
  cases hm with h3 hm,
  cases hm with h4 hm,
  cases hm with h5 hm,
  rw [mod_eq_of_lt (nat.lt_of_succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.zero_lt_bit0 (nat.succ_pos ())))))] at h5,
  exact le_refl _,
  sorry
end

end smallest_number_remainders_l730_730659


namespace dan_balloons_correct_l730_730960

-- Define the initial conditions
def sam_initial_balloons : Float := 46.0
def sam_given_fred_balloons : Float := 10.0
def total_balloons : Float := 52.0

-- Calculate Sam's remaining balloons
def sam_current_balloons : Float := sam_initial_balloons - sam_given_fred_balloons

-- Define the target: Dan's balloons
def dan_balloons := total_balloons - sam_current_balloons

-- Statement to prove
theorem dan_balloons_correct : dan_balloons = 16.0 := sorry

end dan_balloons_correct_l730_730960


namespace graph_of_equation_l730_730247

theorem graph_of_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := 
by
  sorry

end graph_of_equation_l730_730247


namespace intersection_complement_eq_find_a_l730_730447

-- Proof Goal 1: A ∩ ¬B = {x : ℝ | x ∈ (-∞, -3] ∪ [14, ∞)}

def setA : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def setB : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def negB : Set ℝ := {x | x ≤ -2 ∨ x ≥ 14}

theorem intersection_complement_eq :
  setA ∩ negB = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Proof Goal 2: The range of a such that E ⊆ B

def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

theorem find_a (a : ℝ) :
  (∀ x, E a x → setB x) → a ≥ -1 :=
by
  sorry

end intersection_complement_eq_find_a_l730_730447


namespace problem_I_inequality_solution_problem_II_condition_on_b_l730_730752

-- Define the function f(x).
def f (x : ℝ) : ℝ := |x - 2|

-- Problem (I): Proving the solution set to the given inequality.
theorem problem_I_inequality_solution (x : ℝ) : 
  f x + f (x + 1) ≥ 5 ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

-- Problem (II): Proving the condition on |b|.
theorem problem_II_condition_on_b (a b : ℝ) (ha : |a| > 1) (h : f (a * b) > |a| * f (b / a)) :
  |b| > 2 :=
sorry

end problem_I_inequality_solution_problem_II_condition_on_b_l730_730752


namespace pencil_fraction_white_part_l730_730288

theorem pencil_fraction_white_part
  (L : ℝ )
  (H1 : L = 9.333333333333332)
  (H2 : (1 / 8) * L + (7 / 12 * 7 / 8) * (7 / 8) * L + W * (7 / 8) * L = L) :
  W = 5 / 12 :=
by
  sorry

end pencil_fraction_white_part_l730_730288


namespace evaluate_x2_y2_l730_730024

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l730_730024


namespace sum_of_sequence_l730_730526

noncomputable def sequence_satisfies_condition (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (∑ i in Finset.range (n + 1), x i ^ 3) = (∑ i in Finset.range (n + 1), x i) ^ 2

theorem sum_of_sequence (x : ℕ → ℝ) (h : sequence_satisfies_condition x) :
  ∀ n : ℕ, ∃ m : ℕ, (∑ i in Finset.range (n + 1), x i) = m * (m + 1) / 2 :=
by
  sorry

end sum_of_sequence_l730_730526


namespace diametrically_opposite_points_l730_730300

theorem diametrically_opposite_points (n : ℕ) (h : (35 - 7 = n / 2)) : n = 56 := by
  sorry

end diametrically_opposite_points_l730_730300


namespace max_viewers_per_week_l730_730271

theorem max_viewers_per_week :
  ∃ (x y : ℕ), 80 * x + 40 * y ≤ 320 ∧ x + y ≥ 6 ∧ 600000 * x + 200000 * y = 2000000 :=
by
  sorry

end max_viewers_per_week_l730_730271


namespace water_percentage_in_nectar_l730_730627

theorem water_percentage_in_nectar:
  (N H : Real) (HN : 1.3 = N) (HH : 1 = H) (honey_water : Real := 0.35 * H) 
  (nectar_water : Real := N - H) :
  (nectar_water / N) * 100 = 23.08 := 
by
  sorry

end water_percentage_in_nectar_l730_730627


namespace gcd_490_910_l730_730971

def a : ℤ := 490
def b : ℤ := 910
def gcd : ℤ := Int.gcd a b

theorem gcd_490_910 : gcd = 70 := by
  sorry

end gcd_490_910_l730_730971


namespace unique_solution_quadratic_l730_730764

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃ x, q * x^2 - 18 * x + 8 = 0 ∧ ∀ y, q * y^2 - 18 * y + 8 = 0 → y = x) →
  q = 81 / 8 :=
by
  sorry

end unique_solution_quadratic_l730_730764


namespace cone_volume_double_height_l730_730989

-- Define the conditions
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
def cone_volume (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

-- Define the given volume for the cylinder
def cylinder_given_volume := 81 * π

-- Prove the equivalent volume for the cone
theorem cone_volume_double_height 
  (r h : ℝ) 
  (cylinder_vol : cylinder_volume r h = cylinder_given_volume) 
  : cone_volume r (2 * h) = 54 * π :=
by
  -- Proof is omitted
  sorry

end cone_volume_double_height_l730_730989


namespace total_percentage_increase_l730_730783

-- Given conditions as variables
variables (a b c : ℝ)

-- Definition of the total percentage increase from t=0 to t=3 given the conditions
def percentage_increase (a b c : ℝ) : ℝ :=
  a + b + c + (a * b / 100) + (b * c / 100) + (a * c / 100) + (a * b * c / 10000)

-- The theorem statement
theorem total_percentage_increase (a b c : ℝ) :
  percentage_increase a b c = 
  a + b + c + (a * b / 100) + (b * c / 100) + (a * c / 100) + (a * b * c / 10000) :=
by
  simp [percentage_increase]
  sorry

end total_percentage_increase_l730_730783


namespace diagonals_bisect_each_other_l730_730595

theorem diagonals_bisect_each_other 
  (R : Type) [rect : Rectangle R] [rhomb : Rhombus R] [sq : Square R] : 
  (∀ r : Rectangle R, diagonals_bisect r) ∧ 
  (∀ rh : Rhombus R, diagonals_bisect rh) ∧ 
  (∀ s : Square R, diagonals_bisect s) := 
by
  sorry

end diagonals_bisect_each_other_l730_730595


namespace average_of_roots_quadratic_l730_730005

theorem average_of_roots_quadratic (c : ℝ) :
  let f : ℝ → ℝ := fun x => 3 * x^2 - 6 * x + c in
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (x₁ + x₂) / 2 = 1 :=
by
  let f : ℝ → ℝ := fun x => 3 * x^2 - 6 * x + c in
  assume h : ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂,
  sorry

end average_of_roots_quadratic_l730_730005


namespace correct_propositions_l730_730806

-- Definitions for the propositions
def prop1 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ M) → a ∧ b
def prop2 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ ¬M) → a ∧ ¬b
def prop3 (a b M : Prop) : Prop := (a ∧ b) ∧ (b ∧ M) → a ∧ M
def prop4 (a M N : Prop) : Prop := (a ∧ ¬M) ∧ (a ∧ N) → ¬M ∧ N

-- Proof problem statement
theorem correct_propositions : 
  ∀ (a b M N : Prop), 
    (prop1 a M b = true) ∨ (prop1 a M b = false) ∧ 
    (prop2 a M b = true) ∨ (prop2 a M b = false) ∧ 
    (prop3 a b M = true) ∨ (prop3 a b M = false) ∧ 
    (prop4 a M N = true) ∨ (prop4 a M N = false) → 
    3 = 3 :=
by
  sorry

end correct_propositions_l730_730806


namespace problem_statement_l730_730854

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_statement_l730_730854


namespace f_prime_neg1_l730_730476

def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

def f' (a b c x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_prime_neg1 (a b c : ℝ) (h : f' a b c 1 = 2) : f' a b c (-1) = -2 :=
by
  sorry

end f_prime_neg1_l730_730476


namespace sum_of_28_terms_l730_730047

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_of_28_terms
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h2 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y)
  (h3 : ∃ d ≠ 0, ∃ a₁, ∀ n, a (n + 1) = a₁ + n * d)
  (h4 : f (a 6) = f (a 23)) :
  sum_arithmetic_sequence 28 (a 1) ((a 2) - (a 1)) = 28 :=
by sorry

end sum_of_28_terms_l730_730047


namespace final_surface_area_l730_730269

noncomputable def surface_area (total_cubes remaining_cubes cube_surface removed_internal_surface : ℕ) : ℕ :=
  (remaining_cubes * cube_surface) + (remaining_cubes * removed_internal_surface)

theorem final_surface_area :
  surface_area 64 55 54 6 = 3300 :=
by
  sorry

end final_surface_area_l730_730269


namespace range_of_a_l730_730145

theorem range_of_a (a : ℝ) (h₁ : 0 < a) (h₂ : ∀ x : ℝ, 0 < x → (1 / a - 1 / x ≤ 2 * x)) : 
  ∀ a : ℝ, a ≥ real.sqrt 2 / 4 :=
by 
  sorry

end range_of_a_l730_730145


namespace remainder_when_divided_by_eleven_l730_730378

-- Definitions from the conditions
def two_pow_five_mod_eleven : ℕ := 10
def two_pow_ten_mod_eleven : ℕ := 1
def ten_mod_eleven : ℕ := 10
def ten_square_mod_eleven : ℕ := 1

-- Proposition we want to prove
theorem remainder_when_divided_by_eleven :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_eleven_l730_730378


namespace largest_n_consecutive_product_l730_730372

theorem largest_n_consecutive_product (n : ℕ) : n = 0 ↔ (n! = (n+1) * (n+2) * (n+3) * (n+4) * (n+5)) := by
  sorry

end largest_n_consecutive_product_l730_730372


namespace calculate_total_customers_l730_730302

theorem calculate_total_customers 
  (initial_customers : ℕ) 
  (new_customers : ℕ) 
  (initial_customers_eq : initial_customers = 3) 
  (new_customers_eq : new_customers = 5) : 
  initial_customers + new_customers = 8 := 
by 
  rw [initial_customers_eq, new_customers_eq] 
  simp
  sorry


end calculate_total_customers_l730_730302


namespace infinite_subset_B_exists_l730_730905

-- Define a predicate to capture the conditions on polynomials in set A
def poly_conditions (f : ℚ[X]) : Prop :=
  ∃ p q : ℚ,
  ¬ (nat.prime p ∧ p ∣ 2004) ∧
  gcd p 2004 = 1 ∧ 
  gcd q 2004 = 1 ∧
  f.eval p = 2004 ∧ f.eval q = 0

-- Define set A as the set of all polynomials of degree 3 with integer coefficients
-- and leading coefficient 1 satisfying poly_conditions
def setA : set ℚ[X] := { f | degree f = 3 ∧ coeff f 3 = 1 ∧ poly_conditions f }

-- The problem statement we want to prove: there exists an infinite subset B ⊆ A
-- such that the function graphs of the members of B are identical except for translations
theorem infinite_subset_B_exists :
  ∃ (B : set ℚ[X]), B ⊆ setA ∧ set.infinite B ∧ 
  (∀ f g ∈ B, ∃ c : ℚ, ∀ x : ℚ, f x = g (x + c)) :=
sorry

end infinite_subset_B_exists_l730_730905


namespace bus_stop_time_l730_730760

theorem bus_stop_time (speed_excl_stops speed_incl_stops : ℝ) (h1 : speed_excl_stops = 50) (h2 : speed_incl_stops = 45) : (60 * ((speed_excl_stops - speed_incl_stops) / speed_excl_stops)) = 6 := 
by
  sorry

end bus_stop_time_l730_730760


namespace determine_v6_l730_730124

variable (v : ℕ → ℝ)

-- Given initial conditions: v₄ = 12 and v₇ = 471
def initial_conditions := v 4 = 12 ∧ v 7 = 471

-- Recurrence relation definition: vₙ₊₂ = 3vₙ₊₁ + vₙ
def recurrence_relation := ∀ n : ℕ, v (n + 2) = 3 * v (n + 1) + v n

-- The target is to prove that v₆ = 142.5
theorem determine_v6 (h1 : initial_conditions v) (h2 : recurrence_relation v) : 
  v 6 = 142.5 :=
sorry

end determine_v6_l730_730124


namespace rulers_in_drawer_l730_730992

-- conditions
def initial_rulers : ℕ := 46
def additional_rulers : ℕ := 25

-- question: total rulers in the drawer
def total_rulers : ℕ := initial_rulers + additional_rulers

-- proof statement: prove that total_rulers is 71
theorem rulers_in_drawer : total_rulers = 71 := by
  sorry

end rulers_in_drawer_l730_730992


namespace larger_of_two_numbers_l730_730655

noncomputable def larger_number (HCF LCM A B : ℕ) : ℕ :=
  if HCF = 23 ∧ LCM = 23 * 9 * 10 ∧ A * B = HCF * LCM ∧ (A = 10 ∧ B = 23 * 9 ∨ B = 10 ∧ A = 23 * 9)
  then max A B
  else 0

theorem larger_of_two_numbers : larger_number (23) (23 * 9 * 10) 230 207 = 230 := by
  sorry

end larger_of_two_numbers_l730_730655


namespace roots_cubic_eq_sum_fraction_l730_730536

theorem roots_cubic_eq_sum_fraction (p q r : ℝ)
  (h1 : p + q + r = 8)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = 3) :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 8 / 69 := 
sorry

end roots_cubic_eq_sum_fraction_l730_730536


namespace base6_to_base10_l730_730322

theorem base6_to_base10 : 
  let n := 2 * 6^2 + 5 * 6^1 + 3 * 6^0 in n = 105 := 
by sorry

end base6_to_base10_l730_730322


namespace solve_for_y_l730_730470

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end solve_for_y_l730_730470


namespace series_inequality_l730_730948

/-- Proof statement: 
  We want to prove the series S = 1/2 - 1/3 + 1/4 - 1/5 + ... + 1/98 - 1/99 + 1/100
  satisfies S > 1/5
-/
theorem series_inequality : 
  let S := (∑ i in (Finset.filter (λ n, even n) (Finset.range 101)), (1/n) - 1/(n+1) ) + 1/100
  in S > 1/5 :=
by
  sorry

end series_inequality_l730_730948


namespace third_group_members_l730_730678

theorem third_group_members (total_members first_group second_group : ℕ) (h₁ : total_members = 70) (h₂ : first_group = 25) (h₃ : second_group = 30) : (total_members - (first_group + second_group)) = 15 :=
sorry

end third_group_members_l730_730678


namespace M_real_l730_730469

def M : Set ℂ := {z | (z - 1) ^ 2 = complex.abs (z - 1) ^ 2}

theorem M_real :
  M = {z : ℂ | ∃ (x : ℝ), z = x} :=
by sorry

end M_real_l730_730469


namespace area_of_triangle_ABC_l730_730451

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3)

lemma period_of_f_and_max_value :
  (∀ x : ℝ, f (x + 2 * π) = f x)
  ∧ (∀ x : ℝ, f x ≤ 2) :=
sorry

theorem area_of_triangle_ABC (A B : ℝ) (ABC : ℝ) (BC : ℝ) (sin_B : ℝ) :
  f (A - π / 3) = sqrt 3 →
  BC = sqrt 7 →
  sin_B = sqrt 21 / 7 →
  ∃ area : ℝ,
  area = (3 * sqrt 3) / 2 :=
sorry

end area_of_triangle_ABC_l730_730451


namespace gold_coins_distribution_l730_730325

theorem gold_coins_distribution (x y : ℝ) (h₁ : x + y = 25) (h₂ : x ≠ y)
  (h₃ : (x^2 - y^2) = k * (x - y)) : k = 25 :=
sorry

end gold_coins_distribution_l730_730325


namespace eval_f_f_at_4_l730_730831

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2 - x
  else x^2 + x - 1

theorem eval_f_f_at_4 : f (f 4) = 1 := by
  sorry

end eval_f_f_at_4_l730_730831


namespace set_union_intersection_subset_B_imp_a_range_l730_730839

open Set

variable {R : Type*} [Preorder R] [BoundedOrder R] [TopologicalSpace R]

def A : Set R := {x | x ≤ 3 ∨ x ≥ 6}
def B : Set R := {x | -2 < x ∧ x < 9}
def C (a : R) : Set R := {x | a < x ∧ x < a + 1}

theorem set_union_intersection (R : Type*) [Preorder R] [BoundedOrder R] :
  (A ∪ B) = univ ∧ (compl A ∩ B) = {x | 3 < x ∧ x < 6} :=
by
  sorry

theorem subset_B_imp_a_range (a : R) : (C a) ⊆ B → (-2 ≤ a ∧ a ≤ 8) :=
by
  intro h
  sorry

end set_union_intersection_subset_B_imp_a_range_l730_730839


namespace math_proof_l730_730015

variables {m x y x1 y1 x3 y3 x4 y4 : ℝ}

-- Definitions based on conditions
def is_origin (O : ℝ × ℝ) : Prop := O = (0, 0)

def on_ellipse (M : ℝ × ℝ) : Prop := (M.1 ^ 2 / 2) + (M.2 ^ 2) = 1

def satisfies_OP_eq_2OM (O P M : ℝ × ℝ) : Prop :=
  (P.1 - O.1) = 2 * (M.1 - O.1) ∧ (P.2 - O.2) = 2 * (M.2 - O.2)

-- The first goal
def trajectory (P : ℝ × ℝ) : Prop := (P.1 ^ 2 / 8) + (P.2 ^ 2 / 4) = 1

-- The second goal
def line_intersects_curve_at_two_points (l C : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ l A.1 A.2 ∧ C A.1 A.2 ∧ l B.1 B.2 ∧ C B.1 B.2

def line_eq (l : ℝ → ℝ → Prop) : Prop := l = λ x y, y = x + m

def curve_eq (C : ℝ → ℝ → Prop) : Prop := C = λ x y, (x ^ 2 / 8) + (y ^ 2 / 4) = 1

def max_area_triangle_OAB : ℝ := 2 * Real.sqrt 2

-- Putting it all together in a Lean 4 statement
theorem math_proof :
  ∀ (O P M : ℝ × ℝ) (l C : ℝ → ℝ → Prop),
    is_origin O →
    on_ellipse M →
    satisfies_OP_eq_2OM O P M →
    trajectory P ∧ 
    line_eq l →
    curve_eq C →
    line_intersects_curve_at_two_points l C →
    max_area_triangle_OAB = 2 * Real.sqrt 2 := 
by sorry

end math_proof_l730_730015


namespace distance_origin_pointB_l730_730874

def origin : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (8, 15)

theorem distance_origin_pointB : real.sqrt ((8 - 0)^2 + (15 - 0)^2) = 17 := by
  sorry

end distance_origin_pointB_l730_730874


namespace complex_abs_eq_sqrt_2_l730_730150

theorem complex_abs_eq_sqrt_2 (z : ℂ) (h : z * (1 - complex.i) = 2) : complex.abs z = real.sqrt 2 := by
  sorry

end complex_abs_eq_sqrt_2_l730_730150


namespace negation_of_sin_geq_one_l730_730947

theorem negation_of_sin_geq_one : (¬ ∀ x : ℝ, sin x ≥ 1) ↔ ∃ x : ℝ, sin x < 1 :=
by sorry

end negation_of_sin_geq_one_l730_730947


namespace limit_of_arithmetic_sequence_l730_730798

open Real
open BigOperators

noncomputable def a (n : ℕ) := 2 * n - 1

noncomputable def S (n : ℕ) := n^2

theorem limit_of_arithmetic_sequence : 
  (tendsto (λ n : ℕ, (S n : ℝ) / (a n)^2) at_top (𝓝 (1 / 4))) :=
begin
  sorry
end

end limit_of_arithmetic_sequence_l730_730798


namespace tan_identity_l730_730398

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l730_730398


namespace cube_volume_in_cubic_yards_l730_730278

def volume_in_cubic_feet := 64
def cubic_feet_per_cubic_yard := 27

theorem cube_volume_in_cubic_yards : 
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 64 / 27 :=
by
  sorry

end cube_volume_in_cubic_yards_l730_730278


namespace parallelepiped_volume_k_l730_730353

theorem parallelepiped_volume_k (k : ℝ) : 
    abs (3 * k^2 - 13 * k + 27) = 20 ↔ k = (13 + Real.sqrt 85) / 6 ∨ k = (13 - Real.sqrt 85) / 6 := 
by sorry

end parallelepiped_volume_k_l730_730353


namespace evaluate_x_squared_minus_y_squared_l730_730026

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l730_730026


namespace probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730640

open ProbabilityTheory

noncomputable def Vanya_probability_more_heads_than_Tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  X_V 3 = binomial 3 (1/2) ∧
  X_T 2 = binomial 2 (1/2) ∧
  Pr(X_V > X_T) = 1/2

noncomputable def Vanya_probability_more_heads_than_Tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  ∀ n : ℕ,
  X_V (n+1) = binomial (n+1) (1/2) ∧
  X_T n = binomial n (1/2) ∧
  Pr(X_V > X_T) = 1/2

theorem probability_heads_vanya_tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_a X_V X_T := sorry

theorem probability_heads_vanya_tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_b X_V X_T := sorry

end probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730640


namespace minimum_k_minus_b_l730_730427

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + Real.log x

noncomputable def tangent_line (x k b : ℝ) : Prop :=
  ∃ x0 : ℝ, f x0 = k * x0 + b ∧ f' x0 = k

theorem minimum_k_minus_b (x k b : ℝ) :
  (∃ x0 : ℝ, tangent_line x0 k b) → 
  ∃ k b : ℝ, k - b = 7 / 2 := by
  sorry

end minimum_k_minus_b_l730_730427


namespace exists_k_2023_solutions_l730_730388

noncomputable def f (n : ℕ) : ℕ := if n > 1 then n.divisors.erase n |>.max' ⟨1, n.zero_lt_succ⟩ else 0

theorem exists_k_2023_solutions :
  ∃ (k : ℕ), (finset.card (finset.filter (λ n, n > 1 ∧ n - f n = k) (finset.range (2024 * 2024)))) = 2023 :=
begin
  sorry
end

end exists_k_2023_solutions_l730_730388


namespace intersection_points_collinear_l730_730610

open EuclideanGeometry

-- Defining the points and conditions
variables {P : Type*} [MetricSpace P] [NormedAddTorsor EuclideanSpace P] 
(O A1 A2 B1 B2 C1 C2 : P)

-- Definition of lines intersecting at O
def line1 := mkLine O A1
def line2 := mkLine O B1
def line3 := mkLine O C1

-- Definitions of points are on respective lines
-- Assuming all points are not collinear with O simultaneously (non-degenerate scenario)
axiom hA1 : A1 ∈ line1
axiom hA2 : A2 ∈ line1
axiom hB1 : B1 ∈ line2
axiom hB2 : B2 ∈ line2
axiom hC1 : C1 ∈ line3
axiom hC2 : C2 ∈ line3

-- Assuming intersecting conditions as given
axiom hIntersec1 : ∃ P1 : P, isIntersectionPoint P1 (mkLine A1 B1) (mkLine A2 B2)
axiom hIntersec2 : ∃ P2 : P, isIntersectionPoint P2 (mkLine B1 C1) (mkLine B2 C2)
axiom hIntersec3 : ∃ P3 : P, isIntersectionPoint P3 (mkLine A1 C1) (mkLine A2 C2)

-- Statement to prove collinearity of intersection points
theorem intersection_points_collinear : ∀ P1 P2 P3 : P, 
  isIntersectionPoint P1 (mkLine A1 B1) (mkLine A2 B2) →
  isIntersectionPoint P2 (mkLine B1 C1) (mkLine B2 C2) →
  isIntersectionPoint P3 (mkLine A1 C1) (mkLine A2 C2) →
  collinear {P1, P2, P3} :=
sorry  -- Proof is omitted

end intersection_points_collinear_l730_730610


namespace cosβ_values_l730_730432

-- Define the given trigonometric conditions
variables {α β : ℝ}
def sinα := -4 / 5
def cosα := -3 / 5
def sina_plus_b := 5 / 13

-- The goal is to prove that cosβ matches one of the specified values.
theorem cosβ_values : 
  let cosβ := λ (cosa_plus_b : ℝ), cosa_plus_b * cosα + sina_plus_b * sinα in
  cosβ (12 / 13) = -56 / 65 ∨ cosβ (-12 / 13) = 16 / 65 :=
by
  sorry

end cosβ_values_l730_730432


namespace vector_CM_expression_l730_730109

variables (O A B C D M : Type)
variables (a b c : ℝ^3)
variables [is_midpoint A B D] [is_midpoint C D M]
variables (OA OB OC : O → ℝ^3)

def vector_a := OA O = a
def vector_b := OB O = b
def vector_c := OC O = c

theorem vector_CM_expression :
  \(\overrightarrow{CM} = \frac{1}{4} \overrightarrow{a} + \frac{1}{4} \overrightarrow{b} - \frac{1}{2} \overrightarrow{c}\) :
  by sorry  

end vector_CM_expression_l730_730109


namespace evaluate_x2_y2_l730_730023

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l730_730023


namespace necessary_not_sufficient_condition_not_sufficient_condition_l730_730630

theorem necessary_not_sufficient_condition (x : ℝ) :
  (1 < x ∧ x < 4) → (|x - 2| < 1) := sorry

theorem not_sufficient_condition (x : ℝ) :
  (|x - 2| < 1) → (1 < x ∧ x < 4) := sorry

end necessary_not_sufficient_condition_not_sufficient_condition_l730_730630


namespace janet_practiced_days_l730_730516

theorem janet_practiced_days (total_miles : ℕ) (miles_per_day : ℕ) (days_practiced : ℕ) :
  total_miles = 72 ∧ miles_per_day = 8 → days_practiced = total_miles / miles_per_day → days_practiced = 9 :=
by
  sorry

end janet_practiced_days_l730_730516


namespace tom_and_elizabeth_climb_ratio_l730_730231

theorem tom_and_elizabeth_climb_ratio :
  let elizabeth_time := 30
  let tom_time_hours := 2
  let tom_time_minutes := tom_time_hours * 60
  (tom_time_minutes / elizabeth_time) = 4 :=
by sorry

end tom_and_elizabeth_climb_ratio_l730_730231


namespace area_of_rectangle_is_200_l730_730878

-- Definitions and conditions
variables (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variables (AB AD : ℝ) (BE AF : ℝ)

def is_rectangle (ABCD : Prop) := AB > 0 ∧ AD > 0

def bisects_angle (C E : Prop) := true  -- Placeholder for bisection property

def E_on_AB (E AB : Prop) := true  -- Placeholder for E on AB property

def F_on_AD (F AD : Prop) := true  -- Placeholder for F on AD property

def BE_eq_ten (BE : ℝ) := BE = 10

def AF_eq_five (AF : ℝ) := AF = 5

theorem area_of_rectangle_is_200 (ABCD : Prop) (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
  (AB AD BE AF : ℝ) (BE_cond : BE_eq_ten BE) (AF_cond : AF_eq_five AF)
  (rect_cond : is_rectangle ABCD) (bisect_cond : bisects_angle C E) (E_AB_cond : E_on_AB E AB)
  (F_AD_cond : F_on_AD F AD) :
  AB * AD = 200 :=
by
  sorry

end area_of_rectangle_is_200_l730_730878


namespace smallest_k_l730_730379

theorem smallest_k (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k > 1 ∧ 
           (∀ n : ℕ, (n > 0 → (n^k - n) % 2010 = 0)) ∧
           (forall k' : ℕ, k' > 1 → (∀ n : ℕ, (n > 0 → (n^k' - n) % 2010 = 0)) → k ≤ k') ∧
           k = 133 :=
begin
  sorry
end

end smallest_k_l730_730379


namespace divides_n3_minus_7n_l730_730565

theorem divides_n3_minus_7n (n : ℕ) : 6 ∣ n^3 - 7 * n := 
sorry

end divides_n3_minus_7n_l730_730565


namespace seq_period_2020_l730_730834

noncomputable def a : ℕ → ℚ
| 0     := 3
| (n+1) := 1 / (1 - a n)

theorem seq_period_2020 : a 2019 = 3 :=
by {
  have p : ∀ n, a (n + 3) = a n,
  { intro n, 
    induction n,
    { 
      show a (n+4) = 1 / (1 - a (n + 3)),
      { 
        repeat { sorry }, -- detail of periodic proof.
      } 
    },
    { 
      show a (n + 3 + 3) = a (n + 3),
      { 
        repeat { sorry }, -- continue induction proof.
      }
    } 
  },
  have h : 2019 = 673 * 3,
  { 
    suffices : 673 * 3 + 1 = 2020,
    { 
      exact (nat.sub_add_eq_add_sub 2019 673).symm,
    },
    { 
      simp [mul_assoc, add_assoc],
    } 
  },
  exact p 0
}

end seq_period_2020_l730_730834


namespace perpendicular_line_through_P_lines_through_P_distance_2_l730_730444

noncomputable def solveSystem (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 6) : ℝ × ℝ :=
  let x := 2
  let y := 4
  (x, y)

theorem perpendicular_line_through_P (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 6)
  (hx : x = 2) (hy : y = 4) : ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a, b, c) = (2, 1, -8) := by
  use (2, 1, -8)
  constructor
  · sorry
  · rfl

theorem lines_through_P_distance_2 (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 6)
  (hx : x = 2) (hy : y = 4) : 
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a, b, c) = (1, 0, -2)) ∨ (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a, b, c) = (3, -4, 10)) := by
  right
  use (3, -4, 10)
  constructor
  · sorry
  · rfl

end perpendicular_line_through_P_lines_through_P_distance_2_l730_730444


namespace intersection_A_B_l730_730474

def setA : Set ℝ := {x | x^2 - 1 < 0}
def setB : Set ℝ := {x | x > 0}

theorem intersection_A_B : setA ∩ setB = {x | 0 < x ∧ x < 1} := 
by 
  sorry

end intersection_A_B_l730_730474


namespace frog_probability_l730_730281

-- Definition of the vertices of the square
def vertices : List (ℕ × ℕ) := [(0, 0), (0, 4), (4, 4), (4, 0)]

-- Definitions of points and their probabilities
def P (x y : ℕ) : ℚ := sorry

-- The starting point for our problem
def start_point : ℕ × ℕ := (1, 2)

-- Conditions extracted from the problem
def move_parallel_to_axis : bool := true
def jump_length : ℕ := 1
def directions_independent : bool := true
def within_square (x y : ℕ) : bool := x ≤ 4 ∧ y ≤ 4

-- The theorem that proves the probability 
theorem frog_probability : P 1 2 = 5/8 :=
  sorry

end frog_probability_l730_730281


namespace proof_num_equations_l730_730705

def is_equation (e : Prop) (contains_unknowns : Bool) : Bool :=
  e ∧ contains_unknowns

def expr1 := 5 + 3 = 8
def expr2 := a = 0
def expr3 := y^2 - 2y
def expr4 := x - 3 = 8

def contains_unknowns1 : Bool := false
def contains_unknowns2 : Bool := true
def contains_unknowns3 : Bool := false
def contains_unknowns4 : Bool := true

def num_equations : Nat :=
  [is_equation expr1 contains_unknowns1, 
   is_equation expr2 contains_unknowns2,
   is_equation expr3 contains_unknowns3, 
   is_equation expr4 contains_unknowns4].count id

theorem proof_num_equations :
  num_equations = 2 := sorry

end proof_num_equations_l730_730705


namespace magician_card_trick_l730_730688

-- Definitions and proof goal
theorem magician_card_trick :
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 :=
by
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  have h : (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 := sorry
  exact h

end magician_card_trick_l730_730688


namespace existence_of_sequences_l730_730748

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def A_seq : ℕ → Point
| n := ⟨n, n^3⟩

noncomputable def B_seq : ℕ → Point
| n := ⟨-n, -(n^3)⟩

def is_on_line (p1 p2 p : Point) : Prop :=
  let slope := if p1.x = p2.x then 0 else (p2.y - p1.y) / (p2.x - p1.x) in
  p.y = p1.y + slope * (p.x - p1.x)

theorem existence_of_sequences :
  ∃ (A B : ℕ → Point), 
    (∀ i j k : ℕ, 1 ≤ i → i < j → j < k → (is_on_line (A i) (A j) (B k) ↔ k = i + j) ) ∧ 
    (∀ i j k : ℕ, 1 ≤ i → i < j → j < k → (is_on_line (B i) (B j) (A k) ↔ k = i + j) ) :=
begin
  use A_seq,
  use B_seq,
  split;
  intros i j k h1 h2 h3,
  {
    -- First check: B_k on line A_i A_j implies k = i + j
    sorry
  },
  {
    -- Second check: A_k on line B_i B_j implies k = i + j
    sorry
  }
end

end existence_of_sequences_l730_730748


namespace first_other_factor_of_lcm_l730_730972

theorem first_other_factor_of_lcm (A B hcf lcm : ℕ) (h1 : A = 368) (h2 : hcf = 23) (h3 : lcm = hcf * 16 * X) :
  X = 1 :=
by
  sorry

end first_other_factor_of_lcm_l730_730972


namespace initial_shirts_count_l730_730974

theorem initial_shirts_count 
  (S T x : ℝ)
  (h1 : 2 * S + x * T = 1600)
  (h2 : S + 6 * T = 1600)
  (h3 : 12 * T = 2400) :
  x = 4 :=
by
  sorry

end initial_shirts_count_l730_730974


namespace import_tax_amount_in_excess_l730_730690

theorem import_tax_amount_in_excess (X : ℝ) 
  (h1 : 0.07 * (2590 - X) = 111.30) : 
  X = 1000 :=
by
  sorry

end import_tax_amount_in_excess_l730_730690


namespace trigonometric_identity_l730_730403

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l730_730403


namespace pentagon_area_fraction_l730_730636

theorem pentagon_area_fraction (PQRS : ℚ) (T U : ℚ) (PQ : ℚ) (QR : ℚ) (RS : ℚ)
    (PT PU : ℚ) (T_mid_QR : 2 * T = QR) (U_mid_RS : 2 * U = RS)
    (W : ℚ) (V : ℚ) (PT_inter_QS : W = PT / 2) (PU_inter_QS : V = PU / 2):
  ∃ RTWVU_area : ℚ, RTWVU_area = \(\frac{1}{3}\) * PQRS :=
begin
    sorry
end

end pentagon_area_fraction_l730_730636


namespace smallest_elements_in_S_l730_730908

noncomputable def S : Set ℕ := {n | ∃ k : ℕ, ∃ a b : ℕ, n = ((2 ^ a) * (3 ^ b)) * k}

theorem smallest_elements_in_S : 
  ∃ (S : Set ℕ), (0 ∈ S) ∧ (∀ n ∈ S, (2 * n + 1) ∈ S ∧ (3 * n + 2) ∈ S) ∧
  (S ∩ {n | n ≤ 2019}).card = 47 :=
by
  sorry

end smallest_elements_in_S_l730_730908


namespace part1_part2_l730_730058

noncomputable def vec_a (x : ℝ) (m : ℝ) : ℝ × ℝ := (Real.sin x, m * Real.cos x)
def vec_b : ℝ × ℝ := (3, -1)

theorem part1 (x : ℝ) (h_parallel : vec_a x 1 ∥ vec_b) : 2 * Real.sin x ^ 2 - 3 * Real.cos x ^ 2 = 3 / 2 :=
sorry

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (vec_a x m).1 * vec_b.1 + (vec_a x m).2 * vec_b.2

theorem part2 (m : ℝ) (hm : m = Real.sqrt 3) :
  ∃ r1 r2 : ℝ, (f (2 * x) m) = (2 * Real.sqrt 3) * Real.sin (2 * x - π / 6) ∨ 
                (f (2 * x) m) = (-2 * Real.sqrt 3) * Real.sin (2 * x - π / 6) ∧ 
                (∀ x ∈ set.Icc (π / 8) (2 * π / 3), (f (2 * x) m) ∈ set.Icc r1 r2) :=
sorry

end part1_part2_l730_730058


namespace value_sum_l730_730332

noncomputable def v (x : ℝ) : ℝ := -x^2 + 3 * sin (x * real.pi / 3)

theorem value_sum :
  v (-2.5) + v (-1) + v (1) + v (2.5) = -14.5 :=
by
  sorry

end value_sum_l730_730332


namespace number_of_integer_pairs_l730_730918

theorem number_of_integer_pairs (ω : ℂ) (h₁ : ω ^ 4 = 1) (h₂ : ω^2 + 1 = 0) :
  {p : ℤ × ℤ | |p.1 * ω + p.2| = 1}.card = 4 :=
sorry

end number_of_integer_pairs_l730_730918


namespace complex_neither_sufficient_nor_necessary_real_l730_730071

noncomputable def quadratic_equation_real_roots (a : ℝ) : Prop := 
  (a^2 - 4 * a ≥ 0)

noncomputable def quadratic_equation_complex_roots (a : ℝ) : Prop := 
  (a^2 - 4 * (-a) < 0)

theorem complex_neither_sufficient_nor_necessary_real (a : ℝ) :
  (quadratic_equation_complex_roots a ↔ quadratic_equation_real_roots a) = false := 
sorry

end complex_neither_sufficient_nor_necessary_real_l730_730071


namespace no_such_f_exists_l730_730181

theorem no_such_f_exists (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
  (h2 : ∀ x y, 0 < x → 0 < y → f x ^ 2 ≥ f (x + y) * (f x + y)) : false :=
sorry

end no_such_f_exists_l730_730181


namespace total_birds_l730_730552

-- Define the conditions
def is_prime_number (n : ℕ) : Prop := sorry -- Assume this is a placeholder for a prime check function

axiom p : ℕ
axiom h1 : 100 < p ∧ p < 200 ∧ is_prime_number p -- p is a prime number between 100 and 200

def ducks := 1.5 * p -- Number of ducks
def total_sum_chickens_ducks := p + ducks -- Sum of chickens and ducks

def turkeys := 0.7 * (3 * total_sum_chickens_ducks) -- 30% less than three times the sum of chickens and ducks

def q := (⌊(turkeys / 7)⌋ + 1) * 7 -- Smallest multiple of 7 greater than the number of turkeys

-- Define the statement we need to prove: total number of birds as 7.75p + q
theorem total_birds (p q : ℕ) (h1 : 100 < p ∧ p < 200 ∧ is_prime_number p)
                    (h2 : q = (⌊(turkeys / 7)⌋ + 1) * 7) : 
                    ∃ (total_birds : ℝ), total_birds = 7.75 * p + q :=
begin
  sorry, -- Proof goes here
end

end total_birds_l730_730552


namespace Ed_more_marbles_than_Doug_l730_730355

-- Definitions based on conditions
def Ed_marbles_initial : ℕ := 45
def Doug_loss : ℕ := 11
def Doug_marbles_initial : ℕ := Ed_marbles_initial - 10
def Doug_marbles_after_loss : ℕ := Doug_marbles_initial - Doug_loss

-- Theorem statement
theorem Ed_more_marbles_than_Doug :
  Ed_marbles_initial - Doug_marbles_after_loss = 21 :=
by
  -- Proof would go here
  sorry

end Ed_more_marbles_than_Doug_l730_730355


namespace smaller_angle_9_15_l730_730731

theorem smaller_angle_9_15 : 
  let degrees_between (h m : ℕ) : ℝ :=
    let hour_angle := h % 12 * 30 + m / 60 * 30
    let minute_angle := m / 60 * 360
    let angle := abs (hour_angle - minute_angle)
    min angle (360 - angle)
  in degrees_between 9 15 = 187.5 := by
  sorry

end smaller_angle_9_15_l730_730731


namespace taxi_faster_than_truck_l730_730229

noncomputable def truck_speed : ℝ := 2.1 / 1
noncomputable def taxi_speed : ℝ := 10.5 / 4

theorem taxi_faster_than_truck :
  taxi_speed / truck_speed = 1.25 :=
by
  sorry

end taxi_faster_than_truck_l730_730229


namespace triangle_shape_l730_730662

noncomputable def is_right_isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ x : ℝ, (b * (x^2 + 1) + c * (x^2 - 1) - 2 * a * x = 0) ∧
           (sin C * cos A - cos C * sin A = 0) ∧
           ((A = C) ∨ (π - (A + C) = B))

theorem triangle_shape (a b c A B C : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0)
  (h4 : is_right_isosceles_triangle a b c A B C) :
  β (a b c A B C).right_isosceles :=
sorry

end triangle_shape_l730_730662


namespace infinite_perfect_squares_l730_730160

theorem infinite_perfect_squares : ∀ (k : ℕ), ∃ (n : ℕ), 2^n + 4^k = 2^(2 * k) * 9 :=
by
  intro k
  use (2 * k + 3)
  rw [pow_add, pow_mul]
  sorry

end infinite_perfect_squares_l730_730160


namespace total_prime_factors_l730_730778

theorem total_prime_factors :
  let expr := (2^25) * (3^17) * (5^11) * (7^8) * (11^4) * (13^3)
  (nat.mul_pow 2 25 1) * (nat.mul_pow 3 17 1) * (nat.mul_pow 5 11 1) * (nat.mul_pow 7 8 1) * (nat.mul_pow 11 4 1) * (nat.mul_pow 13 3 1) = expr ->
  25 + 17 + 11 + 8 + 4 + 3 = 68 :=
by
  sorry

end total_prime_factors_l730_730778


namespace relationship_among_abc_l730_730424

noncomputable def f (x : ℝ) : ℝ := -exp x + 1

def a : ℝ := -2 * f (-2)
def b : ℝ := -1 * f (-1)
def c : ℝ := 3 * f 3

theorem relationship_among_abc : c < a ∧ a < b := by
  sorry

end relationship_among_abc_l730_730424


namespace school_population_l730_730093

variable (b g t a : ℕ)

theorem school_population (h1 : b = 2 * g) (h2 : g = 4 * t) (h3 : a = t / 2) : 
  b + g + t + a = 27 * b / 16 := by
  sorry

end school_population_l730_730093


namespace odd_integers_count_l730_730211

theorem odd_integers_count : 
  let a := 17 / 4
  let b := 35 / 2
  let smallest_int := 5
  let largest_int := 17
  (count_odds_in_range a b smallest_int largest_int) = 7 :=
by
  sorry

def count_odds_in_range (a : ℚ) (b : ℚ) (smallest_int : ℤ) (largest_int : ℤ) : ℕ :=
  let range := List.range' smallest_int (largest_int - smallest_int + 1)
  let odds := range.filter (λ n, n % 2 = 1)
  odds.length

end odd_integers_count_l730_730211


namespace money_left_l730_730118

noncomputable def initial_amount : ℕ := 100
noncomputable def cost_roast : ℕ := 17
noncomputable def cost_vegetables : ℕ := 11

theorem money_left (init_amt cost_r cost_v : ℕ) 
  (h1 : init_amt = 100)
  (h2 : cost_r = 17)
  (h3 : cost_v = 11) : init_amt - (cost_r + cost_v) = 72 := by
  sorry

end money_left_l730_730118


namespace find_expression_and_range_l730_730828

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * x^2 + m * x + 6 * Real.log x

def g (x : ℝ) (m : ℝ) : ℝ := f x m - x^2 + x

theorem find_expression_and_range (m : ℝ) :
  f' 1 m = 2 → -- tangent condition
  (∀ x, 0 < x → x ≤ 1 → g' x m ≥ 0) → -- monotonicity condition
  (f x -5 = (1/2) * x^2 - 5 * x + 6 * Real.log x * x) ∧ -- analytical expression
  m ∈ Ici (-6) -- range of m
  :=
by
  sorry

end find_expression_and_range_l730_730828


namespace evaluate_x2_y2_l730_730025

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l730_730025


namespace median_number_of_children_l730_730970

-- Define the given conditions
def number_of_data_points : Nat := 13
def median_position : Nat := (number_of_data_points + 1) / 2

-- We assert the median value based on information given in the problem
def median_value : Nat := 4

-- Statement to prove the problem
theorem median_number_of_children (h1: median_position = 7) (h2: median_value = 4) : median_value = 4 := 
by
  sorry

end median_number_of_children_l730_730970


namespace vanya_more_heads_probability_l730_730644

-- Define binomial distributions for Vanya and Tanya's coin flips
def binomial (n : ℕ) (p : ℝ) : distribution :=
  λ k : ℕ, choose n k * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the random variables X_V and X_T
def X_V (n : ℕ) : distribution := binomial (n + 1) (1 / 2)
def X_T (n : ℕ) : distribution := binomial n (1 / 2)

-- Define the probability of Vanya getting more heads than Tanya
def probability_vanya_more_heads (n : ℕ) : ℝ :=
  ∑ (k : ℕ) in finset.range (n + 2), ∑ (j : ℕ) in finset.range (k), X_V n k * X_T n j

-- The main theorem to prove
theorem vanya_more_heads_probability (n : ℕ) : probability_vanya_more_heads n = 1 / 2 :=
begin
  sorry,
end

end vanya_more_heads_probability_l730_730644


namespace find_midpoint_distance_to_y_axis_l730_730013

noncomputable def midpoint_distance_to_y_axis
  (A B : ℝ × ℝ) (AF BF : ℝ) (F : ℝ × ℝ)
  (hF : F = (1 / 4, 0))
  (hA : A.1^2 = A.2)
  (hB : B.1^2 = B.2)
  (h_distances : |AF| + |BF| = 3) :
  ℝ :=
let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
abs midpoint.1

theorem find_midpoint_distance_to_y_axis
  (A B : ℝ × ℝ) (AF BF : ℝ) (F : ℝ × ℝ)
  (hF : F = (1 / 4, 0))
  (h1 : A.2^2 = A.1)
  (h2 : B.2^2 = B.1)
  (h_distances : |AF| + |BF| = 3)
  (h_sum_x: A.1 + 1 / 4 + B.1 + 1 / 4 = 3) :
  midpoint_distance_to_y_axis A B AF BF F hF h1 h2 h_distances = 5 / 4 := sorry

end find_midpoint_distance_to_y_axis_l730_730013


namespace S_30_value_l730_730546

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ := sorry

axiom S_10 : geometric_sequence_sum 10 = 10
axiom S_20 : geometric_sequence_sum 20 = 30

theorem S_30_value : geometric_sequence_sum 30 = 70 :=
by
  sorry

end S_30_value_l730_730546


namespace quadratic_roots_equal_l730_730078

theorem quadratic_roots_equal {k : ℝ} (h : (2 * k) ^ 2 - 4 * 1 * (k^2 + k + 3) = 0) : k^2 + k + 3 = 9 :=
by
  sorry

end quadratic_roots_equal_l730_730078


namespace compound_interest_is_correct_l730_730551

noncomputable def compoundInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R)^T - P

theorem compound_interest_is_correct
  (P : ℝ)
  (R : ℝ)
  (T : ℝ)
  (SI : ℝ) : SI = P * R * T / 100 ∧ R = 0.10 ∧ T = 2 ∧ SI = 600 → compoundInterest P R T = 630 :=
by
  sorry

end compound_interest_is_correct_l730_730551


namespace triangle_reciprocal_sum_l730_730113

variables {A B C D L M N : Type} -- Points are types
variables {t_1 t_2 t_3 t_4 t_5 t_6 : ℝ} -- Areas are real numbers

-- Assume conditions as hypotheses
variable (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
variable (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
variable (h3 : ∀ (t1 t5 t3 t4 : ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5))

theorem triangle_reciprocal_sum 
  (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
  (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
  (h3 : ∀ (t1 t5 t3 t4: ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5)) :
  (1 / t_1 + 1 / t_3 + 1 / t_5) = (1 / t_2 + 1 / t_4 + 1 / t_6) :=
sorry

end triangle_reciprocal_sum_l730_730113


namespace avg_speed_train_l730_730703

theorem avg_speed_train {D V : ℝ} (h1 : D = 20 * (90 / 60)) (h2 : 360 = 6 * 60) : 
  V = D / (360 / 60) :=
  by sorry

end avg_speed_train_l730_730703


namespace compare_abc_l730_730436

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
if x ≤ 0 then 1 - x else (1 / 2 : ℝ)^x

-- Define a, b, and c as given in the problem
def a : ℝ := f (Real.log 2 / Real.log 3)
def b : ℝ := f (2 ^ (-1 / 2 : ℝ))
def c : ℝ := f (3 ^ (1 / 2 : ℝ))

-- The theorem representing the problem statement
theorem compare_abc : a > b ∧ b > c := sorry

end compare_abc_l730_730436


namespace stratified_sampling_total_results_l730_730675

theorem stratified_sampling_total_results :
  let junior_students := 400
  let senior_students := 200
  let total_students_to_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (Nat.choose junior_students junior_sample) * (Nat.choose senior_students senior_sample) = Nat.choose 400 40 * Nat.choose 200 20 :=
  sorry

end stratified_sampling_total_results_l730_730675


namespace sum_inverse_square_radii_ellipse_l730_730391

open Real

theorem sum_inverse_square_radii_ellipse (a b : ℝ) (ha: 0 < a) (hb: 0 < b) (n : ℕ) (hn : 0 < n) 
  (r : ℕ → ℝ) (h_r : ∀ i, r i^2 = (a^2 * b^2) / (b^2 * cos (2 * i * π / n)^2 + a^2 * sin (2 * i * π / n)^2)) : 
  ∑ i in finset.range n, 1 / (r i)^2 = n / 2 * (1 / a^2 + 1 / b^2) :=
sorry

end sum_inverse_square_radii_ellipse_l730_730391


namespace pair_exists_l730_730904

theorem pair_exists (x : Fin 670 → ℝ) (h_distinct : Function.Injective x) (h_bounds : ∀ i, 0 < x i ∧ x i < 1) :
  ∃ (i j : Fin 670), 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := 
by
  sorry

end pair_exists_l730_730904


namespace quad_eq_complete_square_l730_730043

theorem quad_eq_complete_square (p q : ℝ) 
  (h : ∀ x : ℝ, (4 * x^2 - p * x + q = 0 ↔ (x - 1/4)^2 = 33/16)) : q / p = -4 := by
  sorry

end quad_eq_complete_square_l730_730043


namespace pine_cone_weight_on_roof_l730_730307

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end pine_cone_weight_on_roof_l730_730307


namespace find_lambda_sum_l730_730819

-- Define the conditions
variables {A B C O : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space O]
variable (a b : A)
variable (λ1 λ2 : ℝ)

-- Assumptions
axiom h1 : dist A B = 2
axiom h2 : dist A C = 1
axiom h3 : angle A B C = (2 / 3) * Real.pi
axiom h4 : vector A B = a
axiom h5 : vector A C = b
axiom h6 : vector A O = λ1 * a + λ2 * b

-- Goal
theorem find_lambda_sum : λ1 + λ2 = 13 / 6 :=
sorry

end find_lambda_sum_l730_730819


namespace closest_perfect_square_to_325_is_324_l730_730625

theorem closest_perfect_square_to_325_is_324 :
  ∃ n : ℕ, n^2 = 324 ∧ (∀ m : ℕ, m * m ≠ 325) ∧
    (n = 18 ∧ (∀ k : ℕ, (k*k < 325 ∧ (325 - k*k) > 325 - 324) ∨ 
               (k*k > 325 ∧ (k*k - 325) > 361 - 325))) :=
by
  sorry

end closest_perfect_square_to_325_is_324_l730_730625


namespace sum_common_terms_l730_730326

-- Define the sequences and conditions
def seq1 (n: ℕ) : ℕ := 2^n
def seq2 (n: ℕ) : ℕ := 3 * n - 2
def common_terms (n: ℕ) : ℕ := if h : ∃ k: ℕ, seq1 k = seq2 n then some h else 0

-- Prove that the sum of the first n terms is (4^(n+1) - 4) / 3
theorem sum_common_terms (n: ℕ) : 
  let a := λ k, (iterate seq1 k 1).some in
  (∑ i in range n, a i) = (4^(n+1) - 4) / 3 := by
  sorry

end sum_common_terms_l730_730326


namespace eliminate_denominators_l730_730336

variable {x : ℝ}

theorem eliminate_denominators (h : 3 / (2 * x) = 1 / (x - 1)) :
  3 * x - 3 = 2 * x := 
by
  sorry

end eliminate_denominators_l730_730336


namespace sqrt_inequality_l730_730954

theorem sqrt_inequality : (sqrt 2 + sqrt 7) < (sqrt 3 + sqrt 6) := 
by
  sorry

end sqrt_inequality_l730_730954


namespace train_crossing_time_l730_730233

theorem train_crossing_time
  (length_train : ℝ)
  (time_first_train_post : ℝ)
  (time_cross_each_other : ℝ) :
  length_train = 120 →
  time_first_train_post = 12 →
  time_cross_each_other = 16 →
  let speed_first_train := length_train / time_first_train_post in
  let relative_speed := (2 * length_train) / time_cross_each_other in
  let speed_second_train := relative_speed - speed_first_train in
  let time_second_train_post := length_train / speed_second_train in
  time_second_train_post = 24 :=
by
  intros
  sorry

end train_crossing_time_l730_730233


namespace vector_perpendicular_l730_730841

open Real

theorem vector_perpendicular (t : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (4, 3)) :
  a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ↔ t = -2 := by
  sorry

end vector_perpendicular_l730_730841


namespace cannot_derive_xn_minus_1_l730_730507

open Polynomial

noncomputable def f := (X^3 - 3 * X^2 + 5 : Polynomial ℝ)
noncomputable def g := (X^2 - 4 * X : Polynomial ℝ)

lemma derivatives_zero_at_two (p : Polynomial ℝ) : (p.derivative.eval 2 = 0) :=
  by sorry

theorem cannot_derive_xn_minus_1 (p : Polynomial ℝ) (n : ℕ) :
  (p = f ∨ p = g ∨ ∃ (a b : Polynomial ℝ), a = f ∧ b = g ∧ p = a + b ∨ p = a - b ∨ p = a * b ∨ p = a.eval b ∨ ∃ c : ℝ, p = c • a) →
  p.derivative.eval 2 = 0 → n > 0 → (X^n - 1 : Polynomial ℝ).derivative.eval 2 ≠ 0 →
  p ≠ X^n - 1 :=
begin
  intros hp hp_deriv hn hn_deriv,
  sorry
end

end cannot_derive_xn_minus_1_l730_730507


namespace determinant_of_matrix_l730_730735

theorem determinant_of_matrix :
  let M := (Matrix ![[7, 3], [-1, 2]] : Matrix (Fin 2) (Fin 2) ℤ)
  M.det = 17 := by
  sorry

end determinant_of_matrix_l730_730735


namespace circle_center_count_l730_730450

noncomputable def num_circle_centers (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) : ℕ :=
  if (c = d) then 4 else 8

-- Here is the theorem statement
theorem circle_center_count (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) :
  num_circle_centers b c d h₁ h₂ = if (c = d) then 4 else 8 :=
sorry

end circle_center_count_l730_730450


namespace redesigned_lock_additional_combinations_l730_730158

-- Definitions for the problem conditions
def original_combinations : ℕ := Nat.choose 10 5
def total_new_combinations : ℕ := (Finset.range 10).sum (λ k => Nat.choose 10 (k + 1)) 
def additional_combinations := total_new_combinations - original_combinations - 2 -- subtract combinations for 0 and 10

-- Statement of the theorem
theorem redesigned_lock_additional_combinations : additional_combinations = 770 :=
by
  -- Proof omitted (insert 'sorry' to indicate incomplete proof state)
  sorry

end redesigned_lock_additional_combinations_l730_730158


namespace radius_of_sphere_l730_730002

noncomputable def lateral_surface_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (h^2 + r^2) in
  π * r * l

noncomputable def surface_area_sphere (rs : ℝ) : ℝ :=
  4 * π * rs^2

theorem radius_of_sphere 
  (h r rs : ℝ)
  (cone_height : h = 3)
  (cone_radius : r = 4)
  (sphere_surface_equals_cone_lateral : surface_area_sphere rs = lateral_surface_area_cone r h) :
  rs = Real.sqrt 5 :=
by
  sorry

end radius_of_sphere_l730_730002


namespace polynomial_coefficients_l730_730761

theorem polynomial_coefficients (r_1 r_2 r_3 r_4 r_5 r_6 : ℝ) :
  let a_1 := -(r_1 + r_2 + r_3 + r_4 + r_5 + r_6),
      a_2 := r_1 * r_2 + r_1 * r_3 + r_1 * r_4 + r_1 * r_5 + r_1 * r_6 + r_2 * r_3 + r_2 * r_4 + r_2 * r_5 + r_2 * r_6 + r_3 * r_4 + r_3 * r_5 + r_3 * r_6 + r_4 * r_5 + r_4 * r_6 + r_5 * r_6,
      a_3 := -(r_1 * r_2 * r_3 + r_1 * r_2 * r_4 + r_1 * r_2 * r_5 + r_1 * r_2 * r_6 + r_1 * r_3 * r_4 + r_1 * r_3 * r_5 + r_1 * r_3 * r_6 + r_1 * r_4 * r_5 + r_1 * r_4 * r_6 + r_1 * r_5 * r_6 + r_2 * r_3 * r_4 + r_2 * r_3 * r_5 + r_2 * r_3 * r_6 + r_2 * r_4 * r_5 + r_2 * r_4 * r_6 + r_2 * r_5 * r_6 + r_3 * r_4 * r_5 + r_3 * r_4 * r_6 + r_3 * r_5 * r_6 + r_4 * r_5 * r_6),
      a_4 := r_1 * r_2 * r_3 * r_4 + r_1 * r_2 * r_3 * r_5 + r_1 * r_2 * r_3 * r_6 + r_1 * r_2 * r_4 * r_5 + r_1 * r_2 * r_4 * r_6 + r_1 * r_2 * r_5 * r_6 + r_1 * r_3 * r_4 * r_5 + r_1 * r_3 * r_4 * r_6 + r_1 * r_3 * r_5 * r_6 + r_1 * r_4 * r_5 * r_6 + r_2 * r_3 * r_4 * r_5 + r_2 * r_3 * r_4 * r_6 + r_2 * r_3 * r_5 * r_6 + r_2 * r_4 * r_5 * r_6 + r_3 * r_4 * r_5 * r_6,
      a_5 := -(r_1 * r_2 * r_3 * r_4 * r_5 + r_1 * r_2 * r_3 * r_4 * r_6 + r_1 * r_2 * r_3 * r_5 * r_6 + r_1 * r_2 * r_4 * r_5 * r_6 + r_1 * r_3 * r_4 * r_5 * r_6 + r_2 * r_3 * r_4 * r_5 * r_6),
      a_6 := r_1 * r_2 * r_3 * r_4 * r_5 * r_6
  in (x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6 = 0) :=
sorry

end polynomial_coefficients_l730_730761


namespace radius_of_circle_l730_730871

noncomputable def find_radius_of_circle (AB CD : ℝ) (hAB : AB = Real.sqrt 7) (hCD : CD = 1) : ℝ :=
  let R := 1.5
  in R

theorem radius_of_circle (AB CD : ℝ) (hAB : AB = Real.sqrt 7) (hCD : CD = 1) : find_radius_of_circle AB CD hAB hCD = 1.5 :=
  sorry

end radius_of_circle_l730_730871


namespace sum_a_2017_eq_1009_l730_730799

theorem sum_a_2017_eq_1009 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, 2 ≤ n → a n + 2 * S (n - 1) = n)
  (hS : ∀ n, S n = ∑ i in finset.range (n + 1), a i) :
  S 2017 = 1009 :=
sorry

end sum_a_2017_eq_1009_l730_730799


namespace sequence_a_n_l730_730605

-- Given conditions from the problem
variable {a : ℕ → ℕ}
variable (S : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequence is given by S_n
axiom sum_Sn : ∀ n : ℕ, n > 0 → S n = 2 * n * n

-- Definition of a_n, the nth term of the sequence
def a_n (n : ℕ) : ℕ :=
  if n = 1 then
    S 1
  else
    S n - S (n - 1)

-- Prove that a_n = 4n - 2 for all n > 0.
theorem sequence_a_n (n : ℕ) (h : n > 0) : a_n S n = 4 * n - 2 :=
by
  sorry

end sequence_a_n_l730_730605


namespace sum_of_roots_eq_2006_l730_730381

-- Define the polynomial as given in the problem statement
noncomputable def p : Polynomial ℂ := 
  (X - 1)^2008 + 2 * (X - 2)^2007 + 3 * (X - 3)^2006 + 
    ∑ i in finset.range (2006), (i + 4) * (X - (i + 4))^(2008 - (i + 4))

-- Statement of the theorem that the sum of the roots is 2006
theorem sum_of_roots_eq_2006 : 
  (∑ r in (p.roots.to_finset : finset ℂ), r) = 2006 :=
by
    sorry

end sum_of_roots_eq_2006_l730_730381


namespace count_integers_sq_between_150_and_300_l730_730367

theorem count_integers_sq_between_150_and_300 : 
  ({ x : ℕ | 150 ≤ x^2 ∧ x^2 ≤ 300 }.card = 5) := 
begin
  -- The proof will go here
  sorry
end

end count_integers_sq_between_150_and_300_l730_730367


namespace average_of_remaining_numbers_l730_730201

theorem average_of_remaining_numbers (S : ℕ) (h1 : S = 12 * 90) :
  ((S - 65 - 75 - 85) / 9) = 95 :=
by
  sorry

end average_of_remaining_numbers_l730_730201


namespace find_m_abc_inequality_l730_730830

-- Define properties and the theorem for the first problem
def f (x m : ℝ) := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x, f (x + 2) m ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) → m = 1 := by
  intros h
  sorry

-- Define properties and the theorem for the second problem
theorem abc_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) → (a + 2 * b + 3 * c ≥ 9) := by
  intros h
  sorry

end find_m_abc_inequality_l730_730830


namespace min_value_of_quadratic_expression_l730_730406

theorem min_value_of_quadratic_expression (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) : a^2 + 4 * b^2 + 9 * c^2 ≥ 12 :=
by
  sorry

end min_value_of_quadratic_expression_l730_730406


namespace min_sum_proof_l730_730894

noncomputable def min_sum_k_5m_n (m n k : ℕ) (h1 : 1 < m) (h2 : 1 < n) (h3 : 1 < k)
  (h4 : m ≠ n) (h5 : m ≠ k) (h6 : n ≠ k) (h7 : ∃ q : ℚ, log m n = q)
  (h8 : k ^ (sqrt (log m n).toReal) = m ^ (sqrt (log n k).toReal)) : ℕ :=
k + 5 * m + n

theorem min_sum_proof (m n k : ℕ) (h1 : 1 < m) (h2 : 1 < n) (h3 : 1 < k)
  (h4 : m ≠ n) (h5 : m ≠ k) (h6 : n ≠ k) (h7 : ∃ q : ℚ, log m n = q)
  (h8 : k ^ (sqrt (log m n).toReal) = m ^ (sqrt (log n k).toReal)) 
  : min_sum_k_5m_n m n k h1 h2 h3 h4 h5 h6 h7 h8 = 278 := sorry

end min_sum_proof_l730_730894


namespace gcd_2024_2048_l730_730621

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := 
by
  sorry

end gcd_2024_2048_l730_730621


namespace tim_baked_sugar_cookies_l730_730996

theorem tim_baked_sugar_cookies :
  ∀ (total_cookies : ℕ) (chocolate_ratio sugar_ratio peanut_butter_ratio : ℕ),
    total_cookies = 30 →
    chocolate_ratio = 2 →
    sugar_ratio = 5 →
    peanut_butter_ratio = 3 →
    (sugar_ratio * total_cookies) / (chocolate_ratio + sugar_ratio + peanut_butter_ratio) = 15 :=
by
  intros total_cookies chocolate_ratio sugar_ratio peanut_butter_ratio
  assume h_total h_chocolate_ratio h_sugar_ratio h_peanut_butter_ratio
  have ratio_sum : chocolate_ratio + sugar_ratio + peanut_butter_ratio = 10,
    { rw [h_chocolate_ratio, h_sugar_ratio, h_peanut_butter_ratio], trivial },
  have each_part : 30 / 10 = 3,
    { trivial, },
  have result : sugar_ratio * (total_cookies / ratio_sum) = 15,
    { rw [h_total, ratio_sum, each_part, h_sugar_ratio], trivial },
  exact result

end tim_baked_sugar_cookies_l730_730996


namespace area_of_square_same_yarn_l730_730254

theorem area_of_square_same_yarn (a : ℕ) (ha : a = 4) :
  let hexagon_perimeter := 6 * a
  let square_side := hexagon_perimeter / 4
  square_side * square_side = 36 :=
by
  sorry

end area_of_square_same_yarn_l730_730254


namespace sequence_properties_l730_730545

-- Assume the existence of the sequence a_n satisfying the given conditions
variable (a : ℕ → ℝ)
axiom a_sequence_condition : ∀ n : ℕ, n > 0 → 
  (∑ i in Finset.range n, (3 : ℝ) ^ i * a (i+1)) = n / 3

-- Define the function b_n in terms of a_n
def b (n : ℕ) : ℝ := if n = 0 then 0 else n / (a n)

-- Define the sum of the first n terms of b
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)

-- The Lean theorem
theorem sequence_properties :
  (∀ n : ℕ, n > 0 → a n = (1 / 3) ^ n) ∧ 
  (∀ n : ℕ, S n = (3 / 4) + (2 * n - 1) / 4 * 3 ^ (n + 1)) :=
by
  sorry

end sequence_properties_l730_730545


namespace incorrect_conclusion_C_l730_730194

variable {a : ℕ → ℝ} {q : ℝ}

-- Conditions
def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q

theorem incorrect_conclusion_C 
  (h_geo: geo_seq a q)
  (h_cond: a 1 * a 2 < 0) : 
  a 1 * a 5 > 0 :=
by 
  sorry

end incorrect_conclusion_C_l730_730194


namespace smallest_distinct_values_required_l730_730667

def table_size : ℕ := 2012
def num_sums : ℕ := 3

theorem smallest_distinct_values_required :
  ∃ v : ℕ, (∀ f : Fin table_size × Fin table_size → ℤ, 
    (∀ i j, distinct (sum (f i)) ∧ distinct (sum (f j)) ∧ distinct (sum (diagonal f))) ∧
    ∀ x ∈ range num_sums, true) ↔ v = num_sums :=
sorry

end smallest_distinct_values_required_l730_730667


namespace michael_meets_truck_exactly_once_l730_730548

-- Define constants and initial conditions
def michael_speed : ℝ := 4  -- feet per second
def truck_speed : ℝ := 12   -- feet per second
def pail_distance : ℝ := 200  -- feet between pails
def truck_stop_time : ℝ := 20 -- seconds stopping at each pail

-- Define a function to calculate cyclic time for truck
def truck_cycle_time : ℝ := (pail_distance / truck_speed) + truck_stop_time

-- Define a function for the relative distance between Michael and the truck after each cycle
def relative_distance (n : ℕ) : ℝ :=
  (200 + (200 - (pail_distance / truck_speed) * michael_speed)) - truck_stop_time * michael_speed

-- Proof statement that Michael and the truck will meet exactly once
theorem michael_meets_truck_exactly_once : (∃ (n : ℕ), relative_distance n = 0) :=
by
  -- Since this is just the statement, we skip the proof with sorry
  sorry

end michael_meets_truck_exactly_once_l730_730548


namespace find_x_l730_730495

variables {A B C D X Y E : Point}
variables {x : ℝ}

-- Given conditions
axiom angle_AXB : ∠ A X B = 110
axiom angle_CYX : ∠ C Y X = 130
axiom angle_XYZ : ∠ X Y Z = 20
axiom angle_YXE : ∠ Y X E = x
axiom AB_straight : is_straight_line A B
axiom CD_straight : is_straight_line C D
axiom E_on_CD : lies_on_line E C D

-- To prove
theorem find_x : x = 110 :=
by sorry

end find_x_l730_730495


namespace Douglas_weight_correct_l730_730714

def Anne_weight : ℕ := 67
def weight_diff : ℕ := 15
def Douglas_weight : ℕ := 52

theorem Douglas_weight_correct : Douglas_weight = Anne_weight - weight_diff := by
  sorry

end Douglas_weight_correct_l730_730714


namespace magnitude_of_z_l730_730149

noncomputable def z_satisfies_condition (z : ℂ) : Prop :=
  (z + 1) / (z - 2) = 1 - 3 * complex.i

theorem magnitude_of_z : ∀ z : ℂ, z_satisfies_condition z → abs z = real.sqrt 5 := by
  sorry

end magnitude_of_z_l730_730149


namespace sum_of_b_satisfying_polynomial_l730_730911

theorem sum_of_b_satisfying_polynomial :
  let S := {b : ℤ | ∃ r s : ℤ, r + s = -b ∧ r * s = 2009 * b},
  |∑ b in S, b| = 72324 := by
  sorry

end sum_of_b_satisfying_polynomial_l730_730911


namespace betty_oranges_l730_730334

-- Define the givens and result as Lean definitions and theorems
theorem betty_oranges (kg_apples : ℕ) (cost_apples_per_kg cost_oranges_per_kg total_cost_oranges num_oranges : ℕ) 
    (h1 : kg_apples = 3)
    (h2 : cost_apples_per_kg = 2)
    (h3 : cost_apples_per_kg * 2 = cost_oranges_per_kg)
    (h4 : 12 = total_cost_oranges)
    (h5 : total_cost_oranges / cost_oranges_per_kg = num_oranges) :
    num_oranges = 3 :=
sorry

end betty_oranges_l730_730334


namespace percentage_discount_thm_l730_730303

variable (total_cost_16oz_pkg : ℝ) (cost_8oz_pkg : ℝ) (single_4oz_pkg_price : ℝ)
variable (total_cost_target : ℝ) (discount_per_4oz_pkg : ℝ) (percentage_discount : ℝ)

-- Conditions
def condition1 : Prop := total_cost_16oz_pkg = 7
def condition2 : Prop := cost_8oz_pkg = 4
def condition3 : Prop := single_4oz_pkg_price = 2
def condition4 : Prop := total_cost_target = 6

-- Define total cost without discount
def total_cost_without_discount := cost_8oz_pkg + 2 * single_4oz_pkg_price

-- Define total discount needed
def total_discount_needed := total_cost_without_discount - total_cost_target

-- Define discount per 4 oz package
def discount_per_package := total_discount_needed / 2

-- Define percentage discount
def percentage_discount_correct := (discount_per_package / single_4oz_pkg_price) * 100 = percentage_discount

-- The theorem to be proved
theorem percentage_discount_thm 
  (H1 : condition1)
  (H2 : condition2)
  (H3 : condition3)
  (H4 : condition4)
  (H5 : total_cost_without_discount = 8)
  (H6 : total_discount_needed = 2)
  (H7 : discount_per_package = 1)
  : percentage_discount = 50 :=
sorry

end percentage_discount_thm_l730_730303


namespace power_combination_l730_730851

variable {R : Type} [CommRing R] {a m n : R}

theorem power_combination (h1 : a^m = 3) (h2 : a^n = 5) : a^(2 * m + n) = 45 :=
  sorry

end power_combination_l730_730851


namespace remainder_when_divided_by_5_l730_730289

theorem remainder_when_divided_by_5 (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3)
  (h3 : k < 41) : k % 5 = 2 :=
sorry

end remainder_when_divided_by_5_l730_730289


namespace num_two_digit_multiples_5_and_7_l730_730846

/-- 
    Theorem: There are exactly 2 positive two-digit integers that are multiples of both 5 and 7.
-/
theorem num_two_digit_multiples_5_and_7 : 
  ∃ (count : ℕ), count = 2 ∧ ∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → 
    (n % 5 = 0 ∧ n % 7 = 0) ↔ (n = 35 ∨ n = 70) := 
by
  sorry

end num_two_digit_multiples_5_and_7_l730_730846


namespace probability_x_plus_y_le_five_l730_730695

theorem probability_x_plus_y_le_five : 
  (finset.Icc (0:ℝ) 4).sum (λ x, (finset.Icc (0:ℝ) 8).count (λ y, x + y ≤ 5))
  / (real.to_nnreal (4 * 8 : ℝ))
  = 5 / 16 := by sorry

end probability_x_plus_y_le_five_l730_730695


namespace cyclic_quadrilateral_inequality_l730_730718

open Real

variables (A B C D E F : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [has_length A] [has_length B] [has_length C] [has_length D] [has_length E] [has_length F]

-- Assume points D, E, and F are on the sides of triangle ABC
variables [D_on_BC : D ∈ segment B C]
variables [E_on_AC : E ∈ segment A C]
variables [F_on_AB : F ∈ segment A B]

-- Assume quadrilateral AFDE is cyclic
variables [cyclic_AFDE : is_cyclic (quadrilateral A F D E)]

-- Prove the desired inequality
theorem cyclic_quadrilateral_inequality :
  4 * area (triangle D E F) / area (triangle A B C) ≤ (distance E F / distance A D) ^ 2 :=
sorry

end cyclic_quadrilateral_inequality_l730_730718


namespace count_valid_numbers_l730_730945

def digits_appear_once (n : ℕ) : Prop :=
  nat.digits 10 n |>.count 1 = 1 ∧ nat.digits 10 n |>.count 2 = 1

def valid_number (n : ℕ) : Prop :=
  n < 10000 ∧ digits_appear_once n

theorem count_valid_numbers :  finset.card (finset.filter valid_number (finset.range 10000)) = 336 := 
by sorry

end count_valid_numbers_l730_730945


namespace top_card_is_joker_probability_l730_730298

theorem top_card_is_joker_probability :
  let totalCards := 54
  let jokerCards := 2
  let probability := (jokerCards : ℚ) / (totalCards : ℚ)
  probability = 1 / 27 :=
by
  sorry

end top_card_is_joker_probability_l730_730298


namespace angle_between_vectors_l730_730810

variables {a b : EuclideanSpace ℝ}

def is_unit_vector (v : EuclideanSpace ℝ) : Prop :=
  ∥v∥ = 1

def orthogonal (u v : EuclideanSpace ℝ) : Prop :=
  inner u v = 0

theorem angle_between_vectors
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (horth : orthogonal a (a - 2 • b)) :
  ∃ θ : ℝ, ∀ θ, θ = real.arccos (1 / 2) ∧ θ = real.pi / 3 :=
sorry

end angle_between_vectors_l730_730810


namespace ratio_areas_triangle_BDE_ABC_l730_730501

variable {α γ : ℝ}
variable (hα : 0 < α ∧ α < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (hα_γ : α > γ)

theorem ratio_areas_triangle_BDE_ABC (α γ : ℝ) (hα : 0 < α ∧ α < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (hα_γ : α > γ) :
  let tan_half_angle := λ x => Real.tan (x / 2)
  in  (tan_half_angle(α - γ)) / (2 * tan_half_angle(α + γ)) = (Real.tan( (α - γ) / 2)) / (2 * Real.tan( (α + γ) / 2)) :=
by sorry

end ratio_areas_triangle_BDE_ABC_l730_730501


namespace rotation_60_deg_counterclockwise_l730_730833

noncomputable def vector1 := (1:ℝ, 1:ℝ)
noncomputable def vector2 := ((1 - Real.sqrt 3) / 2, (1 + Real.sqrt 3) / 2)

theorem rotation_60_deg_counterclockwise :
  vector2 = 
    ((Real.cos (Real.pi / 3) * vector1.1 - Real.sin (Real.pi / 3) * vector1.2),
     (Real.sin (Real.pi / 3) * vector1.1 + Real.cos (Real.pi / 3) * vector1.2)) :=
sorry

end rotation_60_deg_counterclockwise_l730_730833


namespace union_sets_l730_730148

open Set

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem union_sets :
  A ∪ B = {1, 2, 3, 4, 5} :=
by
  sorry

end union_sets_l730_730148


namespace lara_sees_leo_for_six_minutes_l730_730250

-- Define constants for speeds and initial distances
def lara_speed : ℕ := 60
def leo_speed : ℕ := 40
def initial_distance : ℕ := 1
def time_to_minutes (t : ℚ) : ℚ := t * 60
-- Define the condition that proves Lara can see Leo for 6 minutes
theorem lara_sees_leo_for_six_minutes :
  lara_speed > leo_speed ∧
  initial_distance > 0 ∧
  (initial_distance : ℚ) / (lara_speed - leo_speed) * 2 = (6 : ℚ) / 60 :=
by
  sorry

end lara_sees_leo_for_six_minutes_l730_730250


namespace similar_triangles_length_GH_l730_730056

theorem similar_triangles_length_GH (ABC FGH : Triangle)
  (h_similar : Similar ABC FGH)
  (BC AC A B C F G H : ℝ)
  (BC_eq : BC = 24)
  (FG_eq : FG = 15)
  (AC_eq : AC = 18) :
  ∃ (GH : ℝ), Real.round (GH * 10) = Real.round (11.25 * 10) := by
  sorry

end similar_triangles_length_GH_l730_730056


namespace evaluate_x2_y2_l730_730022

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l730_730022


namespace proof_parallel_OI_DF_l730_730916

noncomputable def problem_statement : Prop :=
  ∀ (O A B C D E I F : Type)
    [circle O A B C]
    (C_is_midpoint : midpoint (arc AB) C)
    [collinear A B D]
    (E_intersection : intersection (BC) (line_parallel (AC) (through D)) E)
    [circumscribed_circle O C E I]
    [intersection (circumscribed_circle O A B) (circumscribed_circle O C E) C F],
    parallel (line_segment O I) (line_segment D F)
    
theorem proof_parallel_OI_DF : problem_statement :=
  sorry

end proof_parallel_OI_DF_l730_730916


namespace probability_remainder_is_one_l730_730323

theorem probability_remainder_is_one :
  let num_possible := 2021
  let num_favorable := 809
  in num_favorable / num_possible = 809 / 2021 :=
by
  sorry

end probability_remainder_is_one_l730_730323


namespace variance_X_plus_2Y_l730_730016

/-
The definitions used in Lean 4 should be directly based on the conditions outlined in the given problem.
-/

noncomputable theory

variable {Ω : Type*} [MeasureSpace Ω]

-- Define the random variables X and Y with given distributions
def X : Ω → ℝ := λ ω, if ω = 0 then 0 else 1
def P_X : MeasureTheory.Measure Ω := MeasureTheory.Measure.dirac 0 0.5 + MeasureTheory.Measure.dirac 1 0.5

def Y : Ω → ℝ := λ ω, if ω = 1 then 1 else 2
def P_Y : MeasureTheory.Measure Ω := MeasureTheory.Measure.dirac 1 (2/3) + MeasureTheory.Measure.dirac 2 (1/3)

-- X and Y are independent
axiom independent_X_Y : MeasureTheory.ProbIndep X Y P_X P_Y

-- Given that a = 1/2 and b = 2/3, we aim to prove the variance of X + 2Y
theorem variance_X_plus_2Y :
  MeasureTheory.variance (λ ω, X ω + 2 * Y ω) = (41 / 36) :=
begin
  sorry
end

end variance_X_plus_2Y_l730_730016


namespace Tim_total_money_l730_730611

theorem Tim_total_money :
  let nickels_amount := 3 * 0.05
  let dimes_amount_shoes := 13 * 0.10
  let shining_shoes := nickels_amount + dimes_amount_shoes
  let dimes_amount_tip_jar := 7 * 0.10
  let half_dollars_amount := 9 * 0.50
  let tip_jar := dimes_amount_tip_jar + half_dollars_amount
  let total := shining_shoes + tip_jar
  total = 6.65 :=
by
  sorry

end Tim_total_money_l730_730611


namespace perimeter_is_correct_l730_730568

def side_length : ℕ := 2
def original_horizontal_segments : ℕ := 16
def original_vertical_segments : ℕ := 10

def horizontal_length : ℕ := original_horizontal_segments * side_length
def vertical_length : ℕ := original_vertical_segments * side_length

def perimeter : ℕ := horizontal_length + vertical_length

theorem perimeter_is_correct : perimeter = 52 :=
by 
  -- Proof goes here.
  sorry

end perimeter_is_correct_l730_730568


namespace number_of_correct_propositions_is_zero_l730_730044

theorem number_of_correct_propositions_is_zero :
  (∀ x, (deriv (λ x : ℝ, Real.exp (2 * x)) x ≠ Real.exp (2 * x)) ∧
  (∀ x, (deriv (λ x : ℝ, Real.log x) x = 1 / x) ∧
  (∀ x, (deriv (λ x : ℝ, Real.sqrt x) x ≠ (1 / 2) * Real.sqrt x)) ∧
  (∀ x, (deriv (λ x : ℝ, (Real.sin (Real.pi / 4) * Real.cos x)) x ≠ Real.cos (Real.pi / 4) * Real.cos x - Real.sin (Real.pi / 4) * Real.sin x))) 
  → 0 = 0 := sorry

end number_of_correct_propositions_is_zero_l730_730044


namespace range_of_t_l730_730285

-- Define a decreasing function f : ℝ → ℝ
variable (f : ℝ → ℝ)
variable h_decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0
variable h_symmetric : ∀ x : ℝ, f (x - 1) = f (-(x - 1))

-- Define the condition for ∀ 2 ≤ s ≤ 4, there exists t such that
variable s : ℝ
variable t : ℝ
variable h_range_s : 2 ≤ s ∧ s ≤ 4
variable h_exist_t : ∃ t : ℝ, f (s^2 - 2 * s) ≤ -f (2 * t - t^2)

-- Prove the range of possible values for t is [0, 2]
theorem range_of_t : 0 ≤ t ∧ t ≤ 2 :=
by {
    sorry
}

end range_of_t_l730_730285


namespace polygon_area_correct_l730_730104

noncomputable def area_ABHFGD : ℝ :=
  let s : ℝ := real.sqrt 25 in  -- side length of squares ABCD and EFGD
  let area_square := 25 in      -- area of each square
  let area_overlap := 2 * (1/2 * s * (s / 2)) in  -- area of quadrilateral CDEH
  let area_HIJD := 3^2 in       -- area of additional square HIJD
  let area_pentagon := 25 - area_overlap in  -- area of each pentagon excluding overlap
  area_pentagon + area_pentagon + area_HIJD  -- total area ABHFGD
  
theorem polygon_area_correct :
  let area_total := area_ABHFGD in
  area_total = 46.5 :=
by
  sorry

end polygon_area_correct_l730_730104


namespace value_of_a2_l730_730007

theorem value_of_a2
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (h1 : ∀ n, S n = (finset.range (n + 1)).sum a)
  (h2 : ∀ n, S n = 2 * a n - 1) :
  a 2 = 4 := sorry

end value_of_a2_l730_730007


namespace monotonic_increasing_interval_l730_730588

def function_y (x : ℝ) : ℝ := 3 * x - x^3

def derivative_y := λ x : ℝ, 3 - 3 * x^2

theorem monotonic_increasing_interval :
  ∀ x : ℝ, -1 < x ∧ x < 1 → derivative_y x > 0 :=
by {
  sorry
}

end monotonic_increasing_interval_l730_730588


namespace largest_sum_of_two_angles_of_quadrilateral_l730_730163

-- The given angles form an arithmetic progression, and the triangles are similar
def angles_form_arithmetic_progression (a b c d : ℝ) : Prop :=
  b = a + d ∧ c = b + d ∧ d = c + d

def similar_triangles (a b c : ℝ) : Prop :=
  ∃ (x y z : ℝ), x + y + z = 180 ∧ x = 60 ∧ y + z = 120

-- The internal angles of the quadrilateral
def internal_angles (x d : ℝ) : Prop :=
  4 * x + 6 * d = 360

-- The problem statement
theorem largest_sum_of_two_angles_of_quadrilateral :
  ∀ (x a b c d : ℝ),
  angles_form_arithmetic_progression x (x+d) (x + 2 * d) (x + 3 * d) →
  similar_triangles a b c →
  internal_angles x d →
  b = 70 ∧ c = 50 →
  a = 60 →
  (b + c) = 120 →
  2 * x + 3 * d = 180 →
  x + d + (x + 2 * d) + (x + 3 * d) = 360 →
  (x + 2 * d) + (x + 3 * d) = 240 :=
begin
  intros,
  sorry
end

end largest_sum_of_two_angles_of_quadrilateral_l730_730163


namespace problem_1_a_problem_1_b_problem_2_l730_730837

def set_A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def set_B : Set ℝ := {x | 2 < x ∧ x < 9}
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def set_union (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∨ x ∈ s₂}
def set_inter (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∧ x ∈ s₂}

theorem problem_1_a :
  set_inter set_A set_B = {x : ℝ | 3 ≤ x ∧ x < 6} :=
sorry

theorem problem_1_b :
  set_union complement_B set_A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
sorry

def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_2 (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end problem_1_a_problem_1_b_problem_2_l730_730837


namespace find_triangle_sides_l730_730587

variables {α β m : ℝ} {A B C M : Type}

-- Conditions: let AM be the median of triangle ABC such that AM = m, and forms angles α and β with sides AB and AC respectively.
def is_median (A B C M : Type) [triangle A B C] (m : ℝ) : Prop :=
  point_on_line M A C ∧ distance A M = m / 2

def forms_angle_with_sides (A B C : Type) [triangle A B C] (AM : Type) (α β : ℝ) : Prop :=
  ∃ M, is_median A B C M (2 * m) ∧ (angle A B M = α) ∧ (angle A C M = β)

-- Prove that AB = (2m * sin β) / sin (α + β) and AC = (2m * sin α) / sin (α + β)
theorem find_triangle_sides
  (h₁ : is_median A B C M (2 * m))
  (h₂ : forms_angle_with_sides A B C M α β) :
  distance A B = (2 * m * sin β) / sin (α + β) ∧ distance A C = (2 * m * sin α) / sin (α + β) := sorry

end find_triangle_sides_l730_730587


namespace div_poly_odd_power_l730_730540

theorem div_poly_odd_power (a b : ℤ) (n : ℕ) : (a + b) ∣ (a^(2*n+1) + b^(2*n+1)) :=
sorry

end div_poly_odd_power_l730_730540


namespace number_of_winning_scores_l730_730483

theorem number_of_winning_scores : 
  ∃ (scores: ℕ), scores = 19 := by
  sorry

end number_of_winning_scores_l730_730483


namespace sweets_per_person_l730_730901

theorem sweets_per_person (green_sweets blue_sweets yellow_sweets : ℕ)
  (total_friends : ℕ) (h_green : green_sweets = 212) (h_blue : blue_sweets = 310) (h_yellow : yellow_sweets = 502) (h_total_friends : total_friends = 4) :
  (green_sweets + blue_sweets + yellow_sweets) / total_friends = 256 :=
by 
suffices h_total_sweets : green_sweets + blue_sweets + yellow_sweets = 1024, from
  by rw [h_total_sweets, h_total_friends]
    exact nat.div_eq_of_eq_mul_right (by norm_num) rfl,
calc
  green_sweets + blue_sweets + yellow_sweets
    = 212 + 310 + 502 : by rw [h_green, h_blue, h_yellow]
... = 1024 : by norm_num

end sweets_per_person_l730_730901


namespace no_partition_into_equilateral_triangles_l730_730004

noncomputable def point (n : ℕ) := vector (ℕ) n

def set_A (n : ℕ) : set (point n) := 
  {p | ∀ i, p.nth i < 6064}

def is_even (p : point n) : Prop :=
  (vector.sum p) % 2 = 0

def is_odd (p : point n) : Prop :=
  ¬ is_even p

theorem no_partition_into_equilateral_triangles (n : ℕ) (h : n ≥ 3) :
  ¬∃ (P : set (point n) → set (fin 3) → Prop),
    ∀ (t : fin 3), 
      (∀ (x y z : point n), 
        x ∈ P t → y ∈ P t → z ∈ P t →
        dist x y = dist y z ∧ dist y z = dist z x) →
      (set_A n) = ⋃ t, P t := 
begin
  sorry
end

end no_partition_into_equilateral_triangles_l730_730004


namespace pull_ups_per_time_l730_730549

theorem pull_ups_per_time (pull_ups_week : ℕ) (times_day : ℕ) (days_week : ℕ)
  (h1 : pull_ups_week = 70) (h2 : times_day = 5) (h3 : days_week = 7) :
  pull_ups_week / (times_day * days_week) = 2 := by
  sorry

end pull_ups_per_time_l730_730549


namespace calculate_total_cost_l730_730719

noncomputable def sandwich_cost : ℕ := 4
noncomputable def soda_cost : ℕ := 3
noncomputable def num_sandwiches : ℕ := 7
noncomputable def num_sodas : ℕ := 8
noncomputable def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem calculate_total_cost : total_cost = 52 := by
  sorry

end calculate_total_cost_l730_730719


namespace fgf_one_l730_730535

/-- Define the function f(x) = 5x + 2 --/
def f (x : ℝ) := 5 * x + 2

/-- Define the function g(x) = 3x - 1 --/
def g (x : ℝ) := 3 * x - 1

/-- Prove that f(g(f(1))) = 102 given the definitions of f and g --/
theorem fgf_one : f (g (f 1)) = 102 := by
  sorry

end fgf_one_l730_730535


namespace gcd_2024_2048_l730_730620

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := 
by
  sorry

end gcd_2024_2048_l730_730620


namespace problem_1_problem_2_l730_730452

-- Defining the given vectors
def vec_OA : ℝ × ℝ := (2, -3)
def vec_OB : ℝ × ℝ := (-5, 4)
def vec_OC (λ : ℝ) : ℝ × ℝ := (1 - λ, 3 * λ + 2)

-- Calculating vector differences
def vec_BA : ℝ × ℝ := (vec_OA.1 - vec_OB.1, vec_OA.2 - vec_OB.2)
def vec_BC (λ : ℝ) : ℝ × ℝ := (vec_OC λ.1 - vec_OB.1, vec_OC λ.2 - vec_OB.2)

-- Problem 1: Right-angled triangle at B
theorem problem_1 (λ : ℝ) (h : vec_BA.1 * vec_BC λ.1 + vec_BA.2 * vec_BC λ.2 = 0) : λ = 2 :=
sorry

-- Problem 2: Non-collinear points A, B, and C form a triangle
theorem problem_2 (λ : ℝ) (h : vec_BA ≠ vec_BC λ) : λ ≠ -2 :=
sorry

end problem_1_problem_2_l730_730452


namespace prob_zero_to_one_l730_730006

variable {σ : ℝ} (ξ : ℝ → ℝ)

def is_normal_1_var (ξ : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), ξ x ~ 𝓝(1, σ^2)

def prob_greater_than_2 (ξ : ℝ → ℝ) [is_normal_1_var ξ] : Prop :=
  P(ξ > 2) = 0.15

theorem prob_zero_to_one (ξ : ℝ → ℝ) [is_normal_1_var ξ] [prob_greater_than_2 ξ] :
  P(0 <= ξ ≤ 1) = 0.35 :=
sorry

end prob_zero_to_one_l730_730006


namespace area_of_centroids_traced_curve_l730_730155

noncomputable def area_of_traced_circle (AB : ℝ) (h : AB = 36) : ℝ :=
  let r := AB / 2
  let radius_of_traced_circle := r / 3
  let area := Real.pi * (radius_of_traced_circle ^ 2)
  let rounded_area := Real.toInt(area).truncate : ℝ
  rounded_area

theorem area_of_centroids_traced_curve :
  area_of_traced_circle 36 (by rfl) = 113 :=
sorry

end area_of_centroids_traced_curve_l730_730155


namespace sum_ratios_ge_one_l730_730700

theorem sum_ratios_ge_one (s : ℝ) (n : ℕ) (a b : ℕ → ℝ) (h₁ : ∀ i, a i ≤ b i)
  (h₂ : s^2 = ∑ i in finset.range n, a i * b i) : 1 ≤ ∑ i in finset.range n, a i / b i :=
sorry

end sum_ratios_ge_one_l730_730700


namespace seth_sold_candy_bars_l730_730563

theorem seth_sold_candy_bars (max_sold : ℕ) (seth_sold : ℕ) 
  (h1 : max_sold = 24) 
  (h2 : seth_sold = 3 * max_sold + 6) : 
  seth_sold = 78 := 
by sorry

end seth_sold_candy_bars_l730_730563


namespace part1_part2_l730_730045

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi / 3)

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 : {x | f x < 1 / 4} = { x | ∃ k : ℤ, k * Real.pi + 5 * Real.pi / 12 < x ∧ x < k * Real.pi + 11 * Real.pi / 12 } :=
by
  sorry

end part1_part2_l730_730045


namespace pairs_with_identical_graphs_l730_730317

def identicalGraphs (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x

def domainRestr (f : ℝ → ℝ) (dom : Set ℝ) : ℝ → ℝ :=
  λ x, if x ∈ dom then f x else 0

theorem pairs_with_identical_graphs :
  (∀ x : ℝ, (\sqrt(x))^2 = x → x ≥ 0) ∧
  (identicalGraphs (λ x : ℝ, x) (λ t : ℝ, (∛(t^3)))) ∧
  (identicalGraphs (λ v : ℝ, |v|) (λ x : ℝ, |x|)) ∧
  (∀ x : ℝ, x ≠ 0 → (x/x) = 1) →
  [B, C] :=
by
  sorry

end pairs_with_identical_graphs_l730_730317


namespace seth_sold_candy_bars_l730_730564

theorem seth_sold_candy_bars (max_sold : ℕ) (seth_sold : ℕ) 
  (h1 : max_sold = 24) 
  (h2 : seth_sold = 3 * max_sold + 6) : 
  seth_sold = 78 := 
by sorry

end seth_sold_candy_bars_l730_730564


namespace P_eta_ge_1_l730_730151

noncomputable def zeta : dist → ℝ := sorry
noncomputable def eta : dist → ℝ := sorry

axiom zeta_dist (p : ℝ) : zeta p ∼ Normal 2 p
axiom eta_dist (p : ℝ) : eta p ∼ Normal 3 p

axiom P_zeta_ge_1 (p : ℝ) : P (ζ p ≥ 1) = 5 / 9

theorem P_eta_ge_1 (p : ℝ) : P (η p ≥ 1) = 16 / 27 := sorry

end P_eta_ge_1_l730_730151


namespace no_poly_of_form_xn_minus_1_l730_730509

theorem no_poly_of_form_xn_minus_1 (f g : ℝ[X])
  (Hf : f = polynomial.X ^ 3 - 3 * polynomial.X ^ 2 + 5)
  (Hg : g = polynomial.X ^ 2 - 4 * polynomial.X)
  (allowed_operations : ∀ (h : ℝ[X]), h = 
    f + g ∨ h = f - g ∨ h = f * g ∨ h = polynomial.c g ∨ h = polynomial.c f ∨ h = polynomial.eval g f) :
  ¬ ∃ n : ℕ, n ≠ 0 ∧ (∃ h : ℝ[X], h = polynomial.X ^ n - 1) :=
by
  sorry

end no_poly_of_form_xn_minus_1_l730_730509


namespace compass_legs_switch_impossible_l730_730203

theorem compass_legs_switch_impossible :
  ∀ (x₁ y₁ x₂ y₂ : ℤ) (d : ℤ),
    (∃ k₁ k₂ : ℤ, x₁^2 + y₁^2 = d^2) ∧
    (∃ l₁ l₂ : ℤ, x₂^2 + y₂^2 = d^2) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    (∀ (x₃ y₃ x₄ y₄ : ℤ),
      (x₃^2 + y₃^2 = d^2) →
      (∃ k₃ k₄ : ℤ, (x₄ = x₃ + 1 ∧ y₄ = y₃ + 1) ∨ (x₄ = x₃ - 1 ∧ y₄ = y₃ - 1) ∨ (x₄ = x₃ + 1 ∧ y₄ = y₃ - 1) ∨ (x₄ = x₃ - 1 ∧ y₄ = y₃ + 1)) →
      ¬ (x₄ = x₂ ∧ y₄ = y₂)
    )

end compass_legs_switch_impossible_l730_730203


namespace AB_less_than_AC_l730_730952

variable (A B C D : Type)
variable (triangle_ABC : Triangle A B C)
variable (D_inside : PointInsideTriangle D A B C)
variable (AD_eq_AB : Distance A D = Distance A B)

theorem AB_less_than_AC (triangle_ABC : Triangle A B C) (D_inside : PointInsideTriangle D A B C) (AD_eq_AB : Distance A D = Distance A B) : Distance A B < Distance A C :=
by
  sorry

end AB_less_than_AC_l730_730952


namespace evaluate_x_squared_minus_y_squared_l730_730029

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l730_730029


namespace cosine_product_le_one_eighth_l730_730261

theorem cosine_product_le_one_eighth (α β γ : ℝ) (r R : ℝ) 
  (h1 : sin (α / 2) * sin (β / 2) * sin (γ / 2) ≤ 1 / 8)
  (h2 : sin (α / 2) * sin (β / 2) * sin (γ / 2) = r / (4 * R))
  (h3 : r ≤ R / 2) :
  cos α * cos β * cos γ ≤ 1 / 8 :=
sorry

end cosine_product_le_one_eighth_l730_730261


namespace coeff_x_cubed_l730_730130

def integrand (x : ℝ) : ℝ := cos x - sin x

def a : ℝ := ∫ x in 0..π, integrand x

theorem coeff_x_cubed :
  let b := (x^2 + a / x) in
  ((b)^6).coeff (2 - 3) = -160 := by
    sorry

end coeff_x_cubed_l730_730130


namespace parallelogram_sides_l730_730602

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 5 * x - 7 = 14) 
  (h2 : 3 * y + 4 = 8 * y - 3) : 
  x + y = 5.6 :=
sorry

end parallelogram_sides_l730_730602


namespace product_sum_of_squares_l730_730556

/-- Define the sequence term expression -/
def seq_term (k : ℕ) : ℕ :=
  (3 * k - 2) * (3 * k - 1) + 3 * k

/-- Product of a finite subset of sequence terms is a sum of two squares -/
theorem product_sum_of_squares (S : Finset ℕ) :
  ∃ a b : ℤ, (S.prod seq_term : ℤ) = a^2 + b^2 :=
begin
  sorry,
end

end product_sum_of_squares_l730_730556


namespace ellipse_tangent_and_fixed_point_l730_730434

-- Definitions based on given conditions
def ellipse (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (x^2 / a^2 + y^2 = 1)

def is_tangent_to (l : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) :=
  ∀ x y, l x y → circle x y

-- Define the specific ellipse and circle based on conditions
def ellipseC : ℝ → ℝ → Prop := ellipse (sqrt 2)
def circle : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 2 / 3

-- Main theorem statement
theorem ellipse_tangent_and_fixed_point :
  (∀ x y, (x = sqrt 2 * (1 - y)) → circle x y) →
  ∃ p : ℝ × ℝ, ∀ A B : ℝ × ℝ, 
    (ellipseC A.1 A.2) →
    (ellipseC B.1 B.2) →
    (A.2 = 1 ∨ B.2 = 1 → A.1 + B.1 = -2) →
    (A.2 ≠ 1 ∧ B.2 ≠ 1) →
    (∃ k1 k2 : ℝ, k1 + k2 = 2) →
    (∃ (L : ℝ × ℝ → Prop), L (A.1, A.2) ∧ L (B.1, B.2) ∧ L p) :=
begin
  sorry
end

end ellipse_tangent_and_fixed_point_l730_730434


namespace angle_equality_of_inscribed_quadrilateral_l730_730717

theorem angle_equality_of_inscribed_quadrilateral
  (A B C D O E F P G : Type)
  [Geometry O]
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_on_circle A O)
  (h3 : is_on_circle B O)
  (h4 : is_on_circle C O)
  (h5 : is_on_circle D O)
  (h6 : E = point_of_intersection (extended_line A B) (extended_line D C))
  (h7 : F = point_of_intersection (extended_line B C) (extended_line A D))
  (h8 : P = point_of_intersection (diagonal A C) (diagonal B D))
  (h9 : G = point_of_intersection (line_through O P) (line_through E F)) :
  angle A G B = angle C G D := 
sorry

end angle_equality_of_inscribed_quadrilateral_l730_730717


namespace jack_total_miles_driven_l730_730514

-- Define the conditions based on the problem
def years : ℕ := 13
def additional_months : ℕ := 7
def miles_per_period : ℕ := 56000
def period_duration : ℝ := 5.5

-- Convert years and months into total months
def total_months : ℕ := years * 12 + additional_months

-- Compute the number of complete periods within the total number of months
def num_periods : ℕ := (total_months / period_duration).toInt

-- Compute the total number of miles driven
def total_miles : ℕ := num_periods * miles_per_period

-- Theorem statement
theorem jack_total_miles_driven : total_miles = 1624000 := by
  sorry

end jack_total_miles_driven_l730_730514


namespace find_initial_files_l730_730936

def megan_initial_files (folders : ℝ) (files_per_folder : ℝ) (added_files : ℝ) (initial_files : ℝ) :=
  initial_files + added_files = folders * files_per_folder

theorem find_initial_files : 
  ∃ initial_files, megan_initial_files 14.25 8.0 21.0 initial_files ∧ initial_files = 93.0 := 
by
  -- We assume such an initial_files exists and it equals 93.0
  use 93.0
  -- Proof is skipped since the statement already shows the relationship
  unfold megan_initial_files
  finish

end find_initial_files_l730_730936


namespace angle_bisector_of_circles_l730_730917

variable {α : Type*} [EuclideanGeometry α]

-- Definitions based on problem conditions
def circles (C1 C2 : set (point α)) (M N : point α) : Prop :=
  C1 ∩ C2 = {M, N}

def common_tangent (C1 C2 : set (point α)) (P Q : point α) : Prop :=
  tangent C1 P ∧ tangent C2 Q ∧ line (P, Q)

def points_on_circles (C1 C2 : set (point α)) (M N P Q R : point α) : Prop :=
  M ∈ C1 ∧ M ∈ C2 ∧ P ∈ C1 ∧ Q ∈ C2 ∧ N ∈ C1 ∧ N ∈ C2 ∧ 
  line (P, N) ∩ C2 = {N, R}

def closer_to_M (P Q M N : point α) : Prop :=
  distance (P, M) < distance (P, N)

-- The theorem statement
theorem angle_bisector_of_circles 
  (C1 C2 : set (point α))
  (M N P Q R : point α)
  (intersection_cond : circles C1 C2 M N)
  (tangent_cond : common_tangent C1 C2 P Q)
  (pos_on_circles : points_on_circles C1 C2 M N P Q R)
  (closer_condition : closer_to_M P Q M N)
  : is_angle_bisector (M, Q) (P, M, R) := sorry

end angle_bisector_of_circles_l730_730917


namespace two_le_three_l730_730886

/-- Proof that the proposition "2 ≤ 3" is true given the logical connective. -/
theorem two_le_three : 2 ≤ 3 := 
by
  sorry

end two_le_three_l730_730886


namespace area_of_sector_l730_730037

theorem area_of_sector (θ r : ℝ) (h_θ : θ = 2) (h_r : r = 1) : 
  (1 / 2) * r^2 * θ = 1 :=
by
  -- Using the given conditions h_θ and h_r
  rw [h_θ, h_r]
  -- Simplify to achieve the desired result
  norm_num
  sorry

end area_of_sector_l730_730037


namespace sin_18_deg_approx_l730_730730

noncomputable def sin_approx : ℝ :=
  let π := Real.pi
  let x := π / 10
  (x - x^3 / 6)

theorem sin_18_deg_approx :
  let π := Real.pi
  let x := π / 10
  |sin x - sin_approx| ≤ (x^5 / 120) ∧ sin_approx = 0.3090 :=
by
  sorry

end sin_18_deg_approx_l730_730730


namespace lila_stickers_correct_l730_730558

-- Defining the constants for number of stickers each has
def Kristoff_stickers : ℕ := 85
def Riku_stickers : ℕ := 25 * Kristoff_stickers
def Lila_stickers : ℕ := 2 * (Kristoff_stickers + Riku_stickers)

-- The theorem to prove
theorem lila_stickers_correct : Lila_stickers = 4420 := 
by {
  sorry
}

end lila_stickers_correct_l730_730558


namespace P_X_leq_2_l730_730085

noncomputable def P (X : ℕ → ℝ) (k : ℕ) : ℝ := 
  if k = 1 then 1/6 
  else if k = 2 then 2/6
  else if k = 3 then 3/6
  else 0

theorem P_X_leq_2 : 
  (P X 1 + P X 2 = 1/2) :=
by 
  have h1 : P X 1 = 1/6 := rfl
  have h2 : P X 2 = 2/6 := rfl
  rw [h1, h2]
  norm_num
  exact rfl

end P_X_leq_2_l730_730085


namespace initial_paper_count_l730_730513

theorem initial_paper_count (used left initial : ℕ) (h_used : used = 156) (h_left : left = 744) :
  initial = used + left :=
sorry

end initial_paper_count_l730_730513


namespace find_parts_per_hour_find_min_A_machines_l730_730197

-- Conditions
variable (x y : ℕ) -- x is parts per hour by B, y is parts per hour by A

-- Definitions based on conditions
def machineA_speed_relation (x y : ℕ) : Prop :=
  y = x + 2

def time_relation (x y : ℕ) : Prop :=
  80 / y = 60 / x

def min_A_machines (x y : ℕ) (m : ℕ) : Prop :=
  8 * m + 6 * (10 - m) ≥ 70

-- Problem statements
theorem find_parts_per_hour (x y : ℕ) (h1 : machineA_speed_relation x y) (h2 : time_relation x y) :
  x = 6 ∧ y = 8 :=
sorry

theorem find_min_A_machines (m : ℕ) (h1 : machineA_speed_relation 6 8) (h2 : time_relation 6 8) (h3 : min_A_machines 6 8 m) :
  m ≥ 5 :=
sorry

end find_parts_per_hour_find_min_A_machines_l730_730197


namespace smallest_k_for_repeating_representation_l730_730744

theorem smallest_k_for_repeating_representation:
  ∃ k : ℕ, (k > 0) ∧ (∀ m : ℕ, m > 0 → m < k → ¬(97*(5*m + 6) = 11*(m^2 - 1))) ∧ 97*(5*k + 6) = 11*(k^2 - 1) := by
  sorry

end smallest_k_for_repeating_representation_l730_730744


namespace evaluate_x_squared_minus_y_squared_l730_730033

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l730_730033


namespace child_to_grandmother_ratio_l730_730691

variable (G D C : ℝ)

axiom condition1 : G + D + C = 150
axiom condition2 : D + C = 60
axiom condition3 : D = 42

theorem child_to_grandmother_ratio : (C / G) = (1 / 5) :=
by
  sorry

end child_to_grandmother_ratio_l730_730691


namespace vectors_norm_squared_sum_l730_730129

open Real

variables (a b m : ℝ → ℝ)
noncomputable def midpoint (a b : ℝ → ℝ) := λ i, (a i + b i) / 2
noncomputable def dot_product (u v : ℝ → ℝ) := u 0 * v 0 + u 1 * v 1
noncomputable def norm_sq (u : ℝ → ℝ) := dot_product u u

theorem vectors_norm_squared_sum :
  let a := λ i, if i = 0 then x else y in
  let b := λ i, if i = 0 then z else w in
  ∀ x y z w,
    let m := λ i, if i = 0 then 3 else 7 in
    midpoint a b = m →
    dot_product a b = 6 →
    norm_sq a + norm_sq b = 220 :=
by
  intros x y z w a b m
  sorry

end vectors_norm_squared_sum_l730_730129


namespace initially_arranged_people_is_10_l730_730159

noncomputable def work_done (people : ℕ) (hours : ℕ) : ℚ :=
  people * hours / 60

theorem initially_arranged_people_is_10 : 
  ∃ x : ℕ, work_done x 1 + work_done (x + 15) 2 = 1 ∧ x = 10 :=
begin
  use 10,
  split,
  {
    -- We will prove that the equation holds for x = 10
    have h1 : work_done 10 1 = 10 / 60 := by simp [work_done],
    have h2 : work_done (10 + 15) 2 = 2 * 25 / 60 := by simp [work_done],

    simp at h1 h2,
    rw [h1, h2],

    norm_num,
  },
  {
    -- Proof that x = 10
    refl,
  }
end

end initially_arranged_people_is_10_l730_730159


namespace triangle_find_angle_A_l730_730493

theorem triangle_find_angle_A (a b c : ℝ) (A B C : ℝ) 
    (ha : a = sqrt 3) 
    (hB : B = 45)
    (hc : c = (sqrt 6 + sqrt 2) / 2)
    (angle_sum : A + B + C = 180) 
    (sine_law_a : a / Real.sin (A * Real.pi / 180) = b / Real.sin (B * Real.pi / 180))
    (sine_law_c : c / Real.sin (C * Real.pi / 180) = b / Real.sin (B * Real.pi / 180)) :
    A = 60 := 
begin
  sorry
end

end triangle_find_angle_A_l730_730493


namespace number_of_students_exclusively_in_math_l730_730849

variable (T M F K : ℕ)
variable (students_in_math students_in_foreign_language students_only_music : ℕ)
variable (students_not_in_music total_students_only_non_music : ℕ)

theorem number_of_students_exclusively_in_math (hT: T = 120) (hM: M = 82)
    (hF: F = 71) (hK: K = 20) :
    T - K = 100 →
    (M + F - 53 = T - K) →
    M - 53 = 29 :=
by
  intros
  sorry

end number_of_students_exclusively_in_math_l730_730849


namespace longer_side_length_l730_730681

def radius (r : ℝ) := r = 6

def area_circle (r : ℝ) : ℝ := π * r^2

def area_rectangle (A_circle : ℝ) : ℝ := 3 * A_circle

def shorter_side (r : ℝ) : ℝ := 2 * r

def longer_side (A_rectangle shorter_side : ℝ) : ℝ := A_rectangle / shorter_side

theorem longer_side_length (r : ℝ) (A_circle : ℝ) (A_rectangle : ℝ) (shorter_side : ℝ) :
  radius r -> area_circle r = A_circle -> area_rectangle A_circle = A_rectangle -> shorter_side r = shorter_side ->
  longer_side A_rectangle shorter_side = 9 * π := 
by
  intros h1 h2 h3 h4
  have hr : r = 6 := by exact h1
  rw [hr, h2] at h3
  have Acircle := (show area_circle 6 = 36 * π, by simp [area_circle, hr, Real.pi.repr (6)] : ℝ)
  have Arectangle := (show 3 * (36 * π) = 108 * π, by ring : ℝ)
  rw [Acircle, Arectangle] at h3
  rw [h3, h4] 
  simp [longer_side, Arectangle]
  sorry

end longer_side_length_l730_730681


namespace common_tangents_infinite_l730_730859

noncomputable def common_tangents_inf (O1 O2 : Type) (R1 R2 : ℝ) : Prop :=
  (O1 = O2 ∧ R1 = R2) → (∃ n : ℕ, n = 0 → n = ℕ∞)

/-- If the centers of two circles \( O_1 \) and \( O_2 \) coincide and the radii are equal \( R_1 = R_2 \), then the number of common tangents is infinite. -/
theorem common_tangents_infinite (O1 O2 : Type) (R1 R2 : ℝ) :
  O1 = O2 → R1 = R2 → ∃ n : ℕ, n = 0 → n = ℕ∞ :=
by
  sorry

end common_tangents_infinite_l730_730859


namespace rook_min_turns_14_l730_730848

def rook_movement_min_turns (board_size : ℕ) (visits_all_squares : Bool) : ℕ :=
  if visits_all_squares = true && board_size = 8 then 14 else 0

theorem rook_min_turns_14 
    (board_size : ℕ) : 
    board_size = 8 → rook_movement_min_turns board_size true = 14 :=
by 
  intros board_size_eq_8
  rw [board_size_eq_8]
  simp
  sorry

end rook_min_turns_14_l730_730848


namespace range_a_l730_730086

noncomputable def satisfies_inequality (a : ℝ) : Prop :=
  ∃ x : ℝ, (exp x - a) ^ 2 + x ^ 2 - 2 * a * x + a ^ 2 ≤ 1 / 2

theorem range_a :
  ∀ a : ℝ, satisfies_inequality a ↔ a = 1 / 2 :=
by
  sorry

end range_a_l730_730086


namespace sqrt_x_minus_y_eq_pm_2_l730_730034

variable {x y : ℝ}

theorem sqrt_x_minus_y_eq_pm_2 (h : sqrt (x - 3) + 2 * abs (y + 1) = 0) : sqrt (x - y) = 2 ∨ sqrt (x - y) = -2 :=
sorry

end sqrt_x_minus_y_eq_pm_2_l730_730034


namespace min_of_abs_alpha_plus_abs_gamma_l730_730930

-- Function definition, conditions, and statement of the theorem.
def f (z α γ : ℂ) : ℂ := (5 + 2 * complex.I) * z^2 + α * z + γ

theorem min_of_abs_alpha_plus_abs_gamma
  (α γ : ℂ)
  (h1 : ∃ (α γ : ℂ), (∀ (z : ℂ), f 1 α γ ∈ ℝ) ∧ (f 1 α γ ∈ ℝ) ∧ (f (-complex.I) α γ ∈ ℝ)) :
  |α| + |γ| = 2 * real.sqrt 2 :=
sorry

end min_of_abs_alpha_plus_abs_gamma_l730_730930


namespace corn_height_after_three_weeks_l730_730728

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l730_730728


namespace polar_to_cartesian_inclination_angle_l730_730051

theorem polar_to_cartesian (θ : ℝ) : (ρ : ℝ) (h : ρ = 4 * cos θ) :
  (let x := ρ * cos θ in let y := ρ * sin θ in (x - 2) ^ 2 + y ^ 2 = 4) := by
  sorry

theorem inclination_angle (α : ℝ) (hα : α ∈ set.Icc 0 real.pi) :
  (let t (x y : ℝ) : ℝ := sqrt ((x - 2)^2 + y^2) in
   ((1 + t * cos α) - 2)^2 + (t * sin α)^2 = 4 → |x - y| = sqrt 14 → 
   (α = real.pi / 4 ∨ α = 3 * real.pi / 4)) := by
  sorry

end polar_to_cartesian_inclination_angle_l730_730051


namespace pilot_fish_final_speed_relative_to_ocean_l730_730903

-- Define conditions
def keanu_speed : ℝ := 20 -- Keanu's speed in mph
def wind_speed : ℝ := 5 -- Wind speed in mph
def shark_speed (initial_speed: ℝ) : ℝ := 2 * initial_speed -- Shark doubles its speed

-- The pilot fish increases its speed by half the shark's increase
def pilot_fish_speed (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  initial_pilot_fish_speed + 0.5 * shark_initial_speed

-- Define the speed of the pilot fish relative to the ocean
def pilot_fish_speed_relative_to_ocean (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  pilot_fish_speed initial_pilot_fish_speed shark_initial_speed - wind_speed

-- Initial assumptions
def initial_pilot_fish_speed : ℝ := keanu_speed -- Pilot fish initially swims at the same speed as Keanu
def initial_shark_speed : ℝ := keanu_speed -- Let us assume the shark initially swims at the same speed as Keanu for simplicity

-- Prove the final speed of the pilot fish relative to the ocean
theorem pilot_fish_final_speed_relative_to_ocean : 
  pilot_fish_speed_relative_to_ocean initial_pilot_fish_speed initial_shark_speed = 25 := 
by sorry

end pilot_fish_final_speed_relative_to_ocean_l730_730903


namespace triangle_inequality_l730_730249

theorem triangle_inequality (a b c : ℕ) : a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

example : triangle_inequality 5 13 12 :=
by 
  apply triangle_inequality
  sorry

end triangle_inequality_l730_730249


namespace douglas_weight_proof_l730_730711

theorem douglas_weight_proof : 
  ∀ (anne_weight douglas_weight : ℕ), 
  anne_weight = 67 →
  anne_weight = douglas_weight + 15 →
  douglas_weight = 52 :=
by 
  intros anne_weight douglas_weight h1 h2 
  sorry

end douglas_weight_proof_l730_730711


namespace solve_for_b_l730_730980

theorem solve_for_b : 
  ∀ {x : ℝ}, (9^5 * 9^x = 10^x) → (x = Real.logb (10/9) (9^5)) → (10 / 9 = b) :=
begin
  assume x h1 h2,
  sorry
end

end solve_for_b_l730_730980


namespace root_of_quadratic_eq_l730_730598

theorem root_of_quadratic_eq  (b: ℝ) (h1: 5 = 5) (h2: c = 1) 
(h3: ∀ {a b c: ℝ}, a + c = 2 * b) 
(h4: 5 ≥ 0 ∧ b ≥ 0 ∧ 1 ≥ 0) 
(h5: b^2 - 4 * 5 * 1 = 0): 
∃ x: ℝ, 5 * x^2 + b * x + 1 = 0 ∧ x = -√5 / 5 :=
sorry
 
end root_of_quadratic_eq_l730_730598


namespace A_winning_strategy_l730_730668

def grid := (1, 2, 3, 4, 5, 6, 7, 8, 9)

def valid_move (prev : ℕ) (next : ℕ) : Prop :=
  (prev, next) ∈ [(7,8), (7,9), (7,4), (8,7), 
                   (8,9), (8,5), (9,7), (9,8),
                   (9,6), (4,7), (4,5), (4,1),
                   (5,4), (5,6), (5,2), (6,9),
                   (6,5), (6,3), (1,4), (1,2),
                   (2,1), (2,5), (2,3), (3,6), 
                   (3,2)]

def win_for_A : Prop :=
  ∃ moves : list ℕ, (moves.head = 9) ∧
  (∀ i < moves.length - 1, valid_move (moves.nth i) (moves.nth (i + 1))) ∧
  (∑ i in moves, i > 30) ∧ (moves.length % 2 = 1)

theorem A_winning_strategy : win_for_A :=
sorry

end A_winning_strategy_l730_730668


namespace exist_twins_l730_730234

theorem exist_twins 
  (child_ages : list ℕ)
  (h_length : child_ages.length = 4)
  (all_born_on_Feb6 : ∀ n ∈ child_ages, n ≥ 2)
  (today_is_Feb_7_2016 : ∀ i, i ∈ child_ages → i ≤ 22)
  (oldest_eq_product_of_rest : 
    ∃ a1 a2 a3 a4, 
      a1 ∈ child_ages ∧ 
      a2 ∈ child_ages ∧ 
      a3 ∈ child_ages ∧ 
      a4 ∈ child_ages ∧ 
      a4 = a1 * a2 * a3) :
  ∃ (a : ℕ), 
    ∃ (x y : ℕ), 
      x ∈ child_ages ∧ 
      y ∈ child_ages ∧ 
      x = a ∧ y = a ∧ x ≠ y :=
begin
  sorry
end

end exist_twins_l730_730234


namespace james_total_distance_l730_730515

-- Define the conditions
def speed_part1 : ℝ := 30  -- mph
def time_part1 : ℝ := 0.5  -- hours
def speed_part2 : ℝ := 2 * speed_part1  -- 2 * 30 mph
def time_part2 : ℝ := 2 * time_part1  -- 2 * 0.5 hours

-- Compute distances
def distance_part1 : ℝ := speed_part1 * time_part1
def distance_part2 : ℝ := speed_part2 * time_part2

-- Total distance
def total_distance : ℝ := distance_part1 + distance_part2

-- The theorem to prove
theorem james_total_distance :
  total_distance = 75 := 
sorry

end james_total_distance_l730_730515


namespace solve_x_l730_730637

theorem solve_x : ∃ (x : ℝ), x ≠ 0 ∧ real.sqrt (5 * x / 3) = x ∧ x = 5 / 3 :=
by
  sorry

end solve_x_l730_730637


namespace dot_product_zero_implies_c_eq_25_over_3_sin_A_when_c_eq_5_l730_730055

-- Define the vertices of the triangle
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 3, y := 4}
def B : Point := {x := 0, y := 0}
def C (c : ℝ) : Point := {x := c, y := 0}

-- Define the vectors AB and AC
def AB : Point := {x := B.x - A.x, y := B.y - A.y}
def AC (c : ℝ) : Point := {x := C(c).x - A.x, y := C(c).y - A.y}

-- Dot product of two points regarded as vectors
def dot (P Q : Point) : ℝ := P.x * Q.x + P.y * Q.y

-- Question 1: Find the value of c that makes the dot product of AB and AC zero
theorem dot_product_zero_implies_c_eq_25_over_3 :
  dot AB (AC c) = 0 → c = 25 / 3 := by
  sorry

-- Question 2: Given c = 5, find the value of sin A
def length (P : Point) : ℝ := Real.sqrt (P.x^2 + P.y^2)

def cos_angle (AB AC : Point) : ℝ :=
  dot AB AC / (length AB * length AC)

theorem sin_A_when_c_eq_5 :
  c = 5 → Real.sin (Real.acos (cos_angle AB (AC 5))) = 2 * Real.sqrt 5 / 5 := by
  sorry

end dot_product_zero_implies_c_eq_25_over_3_sin_A_when_c_eq_5_l730_730055


namespace coefficient_x3_l730_730883

-- Define the binomial coefficient
def binomial_coefficient (n k : Nat) : Nat :=
  Nat.choose n k

noncomputable def coefficient_x3_term : Nat :=
  binomial_coefficient 25 3

theorem coefficient_x3 : coefficient_x3_term = 2300 :=
by
  unfold coefficient_x3_term
  unfold binomial_coefficient
  -- Here, one would normally provide the proof steps, but we're adding sorry to skip
  sorry

end coefficient_x3_l730_730883


namespace total_length_of_path_is_correct_l730_730734

-- Assumptions based on the problem conditions
variables (B : Type) (radius : ℝ) (circumference : ℝ)
variable [noncomputable_instance] : radius = 4 / π
variable [noncomputable_instance] : circumference = 2 * π * radius

-- Define total length of the path as a combination of flat and slope paths
def total_path_length (flat_path_ratio slope_path_ratio : ℝ) : ℝ :=
  (flat_path_ratio * circumference) + (slope_path_ratio * circumference)

-- Main theorem to prove
theorem total_length_of_path_is_correct 
  (flat_path_ratio : ℝ := 3/4) (slope_path_ratio : ℝ := 1/4) : 
  total_path_length B radius circumference flat_path_ratio slope_path_ratio = 8 := by
  sorry

end total_length_of_path_is_correct_l730_730734


namespace probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730641

open ProbabilityTheory

noncomputable def Vanya_probability_more_heads_than_Tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  X_V 3 = binomial 3 (1/2) ∧
  X_T 2 = binomial 2 (1/2) ∧
  Pr(X_V > X_T) = 1/2

noncomputable def Vanya_probability_more_heads_than_Tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  ∀ n : ℕ,
  X_V (n+1) = binomial (n+1) (1/2) ∧
  X_T n = binomial n (1/2) ∧
  Pr(X_V > X_T) = 1/2

theorem probability_heads_vanya_tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_a X_V X_T := sorry

theorem probability_heads_vanya_tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_b X_V X_T := sorry

end probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730641


namespace quadrilateral_area_BEIH_l730_730255

-- Define the necessary points in the problem
structure Point :=
(x : ℚ)
(y : ℚ)

-- Definitions of given points and midpoints
def B : Point := ⟨0, 0⟩
def E : Point := ⟨0, 1.5⟩
def F : Point := ⟨1.5, 0⟩

-- Definitions of line equations from points
def line_DE (p : Point) : Prop := p.y = - (1 / 2) * p.x + 1.5
def line_AF (p : Point) : Prop := p.y = -2 * p.x + 3

-- Intersection points
def I : Point := ⟨3 / 5, 9 / 5⟩
def H : Point := ⟨3 / 4, 3 / 4⟩

-- Function to calculate the area using the Shoelace Theorem
def shoelace_area (a b c d : Point) : ℚ :=
  (1 / 2) * ((a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y) - (a.y * b.x + b.y * c.x + c.y * d.x + d.y * a.x))

-- The proof statement
theorem quadrilateral_area_BEIH :
  shoelace_area B E I H = 9 / 16 :=
sorry

end quadrilateral_area_BEIH_l730_730255


namespace time_to_fill_pool_l730_730967

-- Define the conditions given in the problem
def pool_volume_gallons : ℕ := 30000
def num_hoses : ℕ := 5
def hose_flow_rate_gpm : ℕ := 3

-- Define the total flow rate per minute
def total_flow_rate_gpm : ℕ := num_hoses * hose_flow_rate_gpm

-- Define the total flow rate per hour
def total_flow_rate_gph : ℕ := total_flow_rate_gpm * 60

-- Prove that the time to fill the pool is equal to 34 hours
theorem time_to_fill_pool : pool_volume_gallons / total_flow_rate_gph = 34 :=
by {
  -- Insert detailed proof steps here.
  sorry
}

end time_to_fill_pool_l730_730967


namespace justin_reading_ratio_l730_730520

theorem justin_reading_ratio
  (pages_total : ℝ)
  (pages_first_day : ℝ)
  (pages_left : ℝ)
  (days_remaining : ℝ) :
  pages_total = 130 → 
  pages_first_day = 10 → 
  pages_left = pages_total - pages_first_day →
  days_remaining = 6 →
  (∃ R : ℝ, 60 * R = pages_left) → 
  ∃ R : ℝ, 60 * R = pages_left ∧ R = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end justin_reading_ratio_l730_730520


namespace coin_flips_prob_l730_730649

open Probability

noncomputable def probability_Vanya_more_heads_than_Tanya (n : ℕ) : ℝ :=
  P(X Vanya > X Tanya)

theorem coin_flips_prob (n : ℕ) :
   P(Vanya gets more heads than Tanya | Vanya flips (n+1) times, Tanya flips n times) = 0.5 :=
sorry

end coin_flips_prob_l730_730649


namespace find_t_l730_730137

def P (t : ℝ) := (2 * t + 3, t - 5)
def Q (t : ℝ) := (t - 1, 2 * t + 4)
def midpoint (P Q : ℝ × ℝ) := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def distance_sq (P Q : ℝ × ℝ) := (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

theorem find_t (t : ℝ) :
  distance_sq (midpoint (P t) (Q t)) (P t) = (3 * t ^ 2) / 4 →
  t = 29 ∨ t = -3 :=
by
  sorry

end find_t_l730_730137


namespace num_sets_num_sets_count_l730_730591

open Set

theorem num_sets (M : Finset ℕ) : ({0, 1} ⊆ M ∧ M ⊆ {0, 1, 2, 3}) → (M = {0, 1} ∨ M = {0, 1, 2} ∨ M = {0, 1, 3} ∨ M = {0, 1, 2, 3}) := sorry

theorem num_sets_count : Finset.card {M | {0, 1} ⊆ M ∧ M ⊆ {0, 1, 2, 3}} = 4 := sorry

end num_sets_num_sets_count_l730_730591


namespace sum_of_roots_of_f_poly_l730_730383

-- Define the polynomial as given in the conditions
def f_poly (x : ℝ) : ℝ :=
  (x-1)^2008 + 2*(x-2)^2007 + 3*(x-3)^2006 + 
  ∑ i in range 2004, (i+4)*(x-(i+4))^(2008 - i - 4) + 
  2007*(x-2007)^2 + 2008*(x-2008)

-- State the problem to prove the sum of the roots
theorem sum_of_roots_of_f_poly : 
  (∑ r in (roots (f_poly : polynomial ℝ)), r) = 2006 :=
by 
  sorry

end sum_of_roots_of_f_poly_l730_730383


namespace final_concentration_is_10_percent_l730_730267

variable (kgSaltInit kgInit kgWater : ℝ)
variable (concentrationInit : ℝ)

def saline_solution_concentration (kgSaltInit kgInit kgWater : ℝ) : ℝ := 
  (kgSaltInit / (kgInit + kgWater)) * 100

theorem final_concentration_is_10_percent :
  kgSaltInit = 0.30 * kgInit → kgInit = 100 → kgWater = 200 → 
  concentrationInit = 30 →
  saline_solution_concentration kgSaltInit kgInit kgWater = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  unfold saline_solution_concentration
  sorry

end final_concentration_is_10_percent_l730_730267


namespace coin_flips_prob_l730_730650

open Probability

noncomputable def probability_Vanya_more_heads_than_Tanya (n : ℕ) : ℝ :=
  P(X Vanya > X Tanya)

theorem coin_flips_prob (n : ℕ) :
   P(Vanya gets more heads than Tanya | Vanya flips (n+1) times, Tanya flips n times) = 0.5 :=
sorry

end coin_flips_prob_l730_730650


namespace zero_of_g_lies_in_interval_l730_730411

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f(x) - (f' x) - Real.exp 1

theorem zero_of_g_lies_in_interval :
  (∀ x : ℝ, 0 < x → f(f(x) - Real.log x) = Real.exp 1 + 1) →
  monotone f →
  ∃ x ∈ (1 : ℝ, 2 : ℝ), g(x) = 0 :=
begin
  sorry
end

end zero_of_g_lies_in_interval_l730_730411


namespace retailer_profit_percentage_l730_730708

theorem retailer_profit_percentage
  (marked_price : ℝ)
  (h_marked_price_pos : marked_price > 0)
  (cost_price : ℝ := 0.60 * marked_price)
  (first_discount : ℝ := 3 / 20)
  (second_discount : ℝ := 3 / 40)
  (sp1 : ℝ := marked_price - (first_discount * marked_price))
  (sp2 : ℝ := sp1 - (second_discount * sp1))
  (profit : ℝ := sp2 - cost_price)
  (profit_percentage : ℝ := (profit / cost_price) * 100) :
  profit_percentage ≈ 31.04 := sorry

end retailer_profit_percentage_l730_730708


namespace tan_identity_l730_730397

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l730_730397


namespace num_two_digit_numbers_l730_730374

def first_digits : Finset ℕ := {1, 7, 9}
def all_digits : Finset ℕ := {0, 1, 7, 9}

theorem num_two_digit_numbers : (first_digits.card * (all_digits.card - 1)) = 9 :=
by
  have h1 : first_digits.card = 3 := rfl
  have h2 : all_digits.card = 4 := rfl
  calc
    first_digits.card * (all_digits.card - 1)
        = 3 * (4 - 1) : by rw [h1, h2]
    ... = 3 * 3 : by rfl
    ... = 9 : by rfl

end num_two_digit_numbers_l730_730374


namespace perimeter_of_smaller_rectangle_l730_730701

theorem perimeter_of_smaller_rectangle (s t u : ℝ) (h1 : 4 * s = 160) (h2 : t = s / 2) (h3 : u = t / 3) : 
    2 * (t + u) = 400 / 3 := by
  sorry

end perimeter_of_smaller_rectangle_l730_730701


namespace sequence_becomes_negative_from_8th_term_l730_730208

def seq (n : ℕ) : ℤ := 21 + 4 * n - n ^ 2

theorem sequence_becomes_negative_from_8th_term :
  ∀ n, n ≥ 8 ↔ seq n < 0 :=
by
  -- proof goes here
  sorry

end sequence_becomes_negative_from_8th_term_l730_730208


namespace line_intersects_circle_l730_730003

theorem line_intersects_circle
  (P : ℝ × ℝ)
  (C : ℝ × ℝ)
  (r m : ℝ)
  (hP : P = (-4, -3))
  (hC : C = (-1, -2))
  (hr : r = 5)
  (hm : m = 8)
  (h_intersection : ∃ l : ℝ × ℝ → ℝ, intersects l (C, r) ∧ length_chord l (C, r) m)
  : (x = -4) ∨ (4 * x + 3 * y + 25 = 0) :=
sorry

end line_intersects_circle_l730_730003


namespace even_abs_func_necessary_not_sufficient_l730_730010

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem even_abs_func_necessary_not_sufficient (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, |f (-x)| = |f x|) ∧ (∃ g : ℝ → ℝ, (∀ x : ℝ, |g (-x)| = |g x|) ∧ ¬(∀ x : ℝ, g (-x) = -g x)) :=
by
  -- Proof omitted.
  sorry

end even_abs_func_necessary_not_sufficient_l730_730010


namespace nth_equation_l730_730157

theorem nth_equation (n : ℕ) : 
  (∏ i in finset.range n, (n + 1 + i)) = (2^n) * (∏ i in finset.range n, (2 * i + 1)) :=
by sorry

end nth_equation_l730_730157


namespace same_type_as_sqrt2_l730_730318

theorem same_type_as_sqrt2 : 
  (∃ k : ℝ, sqrt 18 = k * sqrt 2) ∧
  ¬ (∃ k : ℝ, sqrt 12 = k * sqrt 2) ∧
  ¬ (∃ k : ℝ, sqrt 16 = k * sqrt 2) ∧
  ¬ (∃ k : ℝ, sqrt 24 = k * sqrt 2) := by 
sorry

end same_type_as_sqrt2_l730_730318


namespace solution_unique_2014_l730_730364

theorem solution_unique_2014 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x - 2 * y + 1 / z = 1 / 2014) ∧
  (2 * y - 2 * z + 1 / x = 1 / 2014) ∧
  (2 * z - 2 * x + 1 / y = 1 / 2014) →
  x = 2014 ∧ y = 2014 ∧ z = 2014 :=
by
  sorry

end solution_unique_2014_l730_730364


namespace relationship_among_abc_l730_730816

noncomputable def a : ℝ := 0.2^3
noncomputable def b : ℝ := Real.logBase 2 0.3
noncomputable def c : ℝ := Real.logBase 0.3 2

theorem relationship_among_abc : b < c ∧ c < a := sorry

end relationship_among_abc_l730_730816


namespace ratio_of_sums_l730_730126

variable {α : Type*} [LinearOrderField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n : α) / 2 * (a 0 + a (n - 1))

theorem ratio_of_sums
  {a : ℕ → α}
  (h_arith : is_arithmetic_sequence a)
  (h_ratio : a 6 / a 4 = 9 / 13) :
  sum_of_first_n_terms a 13 / sum_of_first_n_terms a 9 = 1 :=
sorry

end ratio_of_sums_l730_730126


namespace triangle_angle_equals_60_l730_730266

theorem triangle_angle_equals_60 (T : Triangle) (h : T.can_be_dissected_into_three_equal_triangles) : 
  ∃ angle ∈ T.angles, angle = 60 := 
sorry

end triangle_angle_equals_60_l730_730266


namespace total_number_of_edges_in_hexahedron_is_12_l730_730869

-- Define a hexahedron
structure Hexahedron where
  face_count : Nat
  edges_per_face : Nat
  edge_sharing : Nat

-- Total edges calculation function
def total_edges (h : Hexahedron) : Nat := (h.face_count * h.edges_per_face) / h.edge_sharing

-- The specific hexahedron (cube) in question
def cube : Hexahedron := {
  face_count := 6,
  edges_per_face := 4,
  edge_sharing := 2
}

-- The theorem to prove the number of edges in a hexahedron
theorem total_number_of_edges_in_hexahedron_is_12 : total_edges cube = 12 := by
  sorry

end total_number_of_edges_in_hexahedron_is_12_l730_730869


namespace min_points_tenth_game_l730_730496

def points_6_9 := [25, 14, 15, 22]
def avg_after_first_five_games_between := (16, 17)
def total_required_points_after_ten_games := 10 * 20

theorem min_points_tenth_game (points_6_9 : List ℕ) 
  (avg_after_first_five_games_between : ℕ × ℕ) 
  (total_required_points_after_ten_games : ℕ) : ℕ :=
  let total_points_6_9 := points_6_9.sum
  let max_points_first_five_games := avg_after_first_five_games_between.2 * 5
  total_required_points_after_ten_games - total_points_6_9 - max_points_first_five_games = 39

end min_points_tenth_game_l730_730496


namespace orthocenter_on_radical_axis_of_circles_l730_730136

open EuclideanGeometry

variables {A B C D E M N H : Point} -- Declare variables for points
variables (triangle_ABC : Triangle A B C) -- Consider triangle ABC
variables (H_orthocenter : Orthocenter triangle_ABC H) -- Assume H is the orthocenter
variables (D_on_AB : OnLineSegment A B D) (E_on_AC : OnLineSegment A C E) -- Points D and E on AB and AC respectively
variables (BM_altitude : Altitude triangle_ABC B M) (CN_altitude : Altitude triangle_ABC C N) -- Altitudes BM and CN
variables (circle_BE : Circle B E) (circle_CD : Circle C D) -- Circles with diameters BE and CD

theorem orthocenter_on_radical_axis_of_circles :
  OnRadicalAxis H circle_BE circle_CD := 
sorry -- Proof is omitted

end orthocenter_on_radical_axis_of_circles_l730_730136


namespace steve_family_time_l730_730193

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l730_730193


namespace min_value_m_plus_n_l730_730571

theorem min_value_m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : 45 * m = n^3) : m + n = 90 :=
sorry

end min_value_m_plus_n_l730_730571


namespace frog_probability_l730_730282

-- Definition of the vertices of the square
def vertices : List (ℕ × ℕ) := [(0, 0), (0, 4), (4, 4), (4, 0)]

-- Definitions of points and their probabilities
def P (x y : ℕ) : ℚ := sorry

-- The starting point for our problem
def start_point : ℕ × ℕ := (1, 2)

-- Conditions extracted from the problem
def move_parallel_to_axis : bool := true
def jump_length : ℕ := 1
def directions_independent : bool := true
def within_square (x y : ℕ) : bool := x ≤ 4 ∧ y ≤ 4

-- The theorem that proves the probability 
theorem frog_probability : P 1 2 = 5/8 :=
  sorry

end frog_probability_l730_730282


namespace violet_marbles_unknown_l730_730348

theorem violet_marbles_unknown :
  ∀ (initial_green : ℕ) (taken_green : ℕ) (remaining_green : ℕ),
    initial_green = 32 →
    taken_green = 23 →
    remaining_green = initial_green - taken_green →
    remaining_green = 9 →
    ∃ (violet_marbles : ℕ), true := 
by
  intros initial_green taken_green remaining_green h_initial h_taken h_remaining h_remaining_eq
  use 0
  trivial
  sorry

end violet_marbles_unknown_l730_730348


namespace convex_quadrilateral_inequality_l730_730906

variable (A B C D E : Type)
variable [ConvexQuadrilateral A B C D]
variable (CD_eq_BC : CD = BC)
variable (BC_eq_BE : BC = BE)

theorem convex_quadrilateral_inequality (h1 : CD = BC) (h2 : BC = BE) :
  AD + DC ≥ AB :=
sorry

end convex_quadrilateral_inequality_l730_730906


namespace sum_of_roots_eq_2006_l730_730382

-- Define the polynomial as given in the problem statement
noncomputable def p : Polynomial ℂ := 
  (X - 1)^2008 + 2 * (X - 2)^2007 + 3 * (X - 3)^2006 + 
    ∑ i in finset.range (2006), (i + 4) * (X - (i + 4))^(2008 - (i + 4))

-- Statement of the theorem that the sum of the roots is 2006
theorem sum_of_roots_eq_2006 : 
  (∑ r in (p.roots.to_finset : finset ℂ), r) = 2006 :=
by
    sorry

end sum_of_roots_eq_2006_l730_730382


namespace find_z_l730_730195

-- Definitions of the rectangle and the operations carried out
def rectangle_area (length width : ℝ) := length * width

-- The sides of the rectangle
def length : ℝ := 15
def width : ℝ := 10

-- The condition that the rectangle is divided into two congruent pentagons and then rearranged to form a square
def area_rectangle := rectangle_area length width
def area_square := area_rectangle
def side_square := real.sqrt area_square

-- Definition of the length z of one side of the pentagons that aligns with one side of the square
noncomputable def z := 5 * real.sqrt 3

-- The theorem representing the problem
theorem find_z (length width : ℝ) (h_len : length = 15) (h_wid : width = 10) :
  let area := rectangle_area length width in
  let area_sq := area in
  let side_sq := real.sqrt area_sq in
  z = 5 * real.sqrt 3 :=
by sorry

end find_z_l730_730195


namespace S7_value_l730_730604

def arithmetic_seq_sum (n : ℕ) (a_1 d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

def a_n (n : ℕ) (a_1 d : ℚ) : ℚ :=
  a_1 + (n - 1) * d

theorem S7_value (a_1 d : ℚ) (S_n : ℕ → ℚ)
  (hSn_def : ∀ n, S_n n = arithmetic_seq_sum n a_1 d)
  (h_sum_condition : S_n 7 + S_n 5 = 10)
  (h_a3_condition : a_n 3 a_1 d = 5) :
  S_n 7 = -15 :=
by
  sorry

end S7_value_l730_730604


namespace stratified_random_sampling_l730_730672

theorem stratified_random_sampling
 (junior_students senior_students total_sample_size : ℕ)
 (junior_high_count senior_high_count : ℕ) 
 (total_sample : junior_high_count + senior_high_count = total_sample_size)
 (junior_students_ratio senior_students_ratio : ℕ)
 (ratio : junior_students_ratio + senior_students_ratio = 1)
 (junior_condition : junior_students_ratio = 2 * senior_students_ratio)
 (students_distribution : junior_students = 400 ∧ senior_students = 200 ∧ total_sample_size = 60)
 (combination_junior : (nat.choose junior_students junior_high_count))
 (combination_senior : (nat.choose senior_students senior_high_count)) :
 combination_junior * combination_senior = nat.choose 400 40 * nat.choose 200 20 :=
by
  sorry

end stratified_random_sampling_l730_730672


namespace probability_sum_divisible_by_4_l730_730572

theorem probability_sum_divisible_by_4 :
  let range := {n | 1 ≤ n ∧ n ≤ 25}
  let pairs := { (a, b) | a ∈ range ∧ b ∈ range ∧ a ≠ b }
  let favorable_pairs :=
    { (a, b) ∈ pairs | (a + b) % 4 = 0 }
  (favorable_pairs.to_finset.card : ℚ) / (pairs.to_finset.card : ℚ) = 6 / 25 :=
by sorry

end probability_sum_divisible_by_4_l730_730572


namespace trapezium_CF_AF_FO_l730_730868

-- Definitions of points and geometric conditions as per the given problem
variable (A B C D E F O : Point)
variable (h1 : is_trapezium A B C D) -- ABCD is a trapezium
variable (h2 : is_parallel A B C D) -- AB is parallel to CD and AB > CD
variable (h3 : bisects BD ∠ADC) -- BD bisects ∠ADC
variable (h4 : is_parallel_through C AD E BD) -- A line through C parallel to AD meets BD at E
variable (h5 : is_parallel_through C AD F AB) -- A line through C parallel to AD meets AB at F
variable (h6 : circumcenter O B E F) -- O is the circumcenter of triangle BEF
variable (h7 : angle A C O = 60) -- ∠ACO = 60°

-- We need to prove the equality
theorem trapezium_CF_AF_FO :
  CF = AF + FO :=
sorry

end trapezium_CF_AF_FO_l730_730868


namespace exists_ten_natural_numbers_l730_730893

theorem exists_ten_natural_numbers :
  ∃ (S : Set ℕ), S.card = 10 ∧ (∀ (n1 ∈ S) (n2 ∈ S), n1 ≠ n2 → ¬(n2 ∣ n1)) ∧ (∀ (n ∈ S), ∀ (m ∈ S), (n^2 ∣ m)) :=
by
  sorry

end exists_ten_natural_numbers_l730_730893


namespace evaluate_x_squared_minus_y_squared_l730_730027

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l730_730027


namespace triangle_ABC_properties_l730_730529

theorem triangle_ABC_properties 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * Real.sin B * Real.sin C * Real.cos A + Real.cos A = 3 * Real.sin A ^ 2 - Real.cos (B - C)) : 
  (2 * a = b + c) ∧ 
  (b + c = 2) →
  (Real.cos A = 3/5) → 
  (1 / 2 * b * c * Real.sin A = 3 / 8) :=
by
  sorry

end triangle_ABC_properties_l730_730529


namespace min_value_of_c_l730_730940

theorem min_value_of_c (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : ∃! x, 2 * x + (|x - a| + |x - b| + |x - c|) = 2031) : c = 1016 :=
by
  sorry

end min_value_of_c_l730_730940


namespace curves_intersect_at_one_point_l730_730745

theorem curves_intersect_at_one_point (b : ℝ) :
  (∃ x : ℝ, bx^2 - 2x + 5 = 3x + 4) ∧
  (∀ x y : ℝ, (bx^2 - 2x + 5 = y ∧ 3x + 4 = y) → x = y) →
  b = 25/4 :=
sorry

end curves_intersect_at_one_point_l730_730745


namespace initial_price_hat_l730_730291

variable (P : ℝ)
hypothesis (wholesale_price : P > 0)
hypothesis (markup_diff : 2 * P - 1.60 * P = 6)

theorem initial_price_hat : 1.60 * P = 24 := 
  sorry

end initial_price_hat_l730_730291


namespace weight_of_new_student_l730_730263

-- Define some constants for the problem
def avg_weight_29_students : ℝ := 28
def number_of_students_29 : ℕ := 29
def new_avg_weight_30_students : ℝ := 27.5
def number_of_students_30 : ℕ := 30

-- Calculate total weights
def total_weight_29_students : ℝ := avg_weight_29_students * number_of_students_29
def new_total_weight_30_students : ℝ := new_avg_weight_30_students * number_of_students_30

-- The proposition we need to prove
theorem weight_of_new_student :
  new_total_weight_30_students - total_weight_29_students = 13 := by
  -- Placeholder for the actual proof
  sorry

end weight_of_new_student_l730_730263


namespace solve_expression_l730_730590

theorem solve_expression (y : ℂ) (h : 3 * y^2 + 2 * y + 1 = 0) : 
  (6 * y + 5)^2 = -7 + 12 * y * ℂ.i.sqrt(2) ∨ (6 * y + 5)^2 = -7 - 12 * y * ℂ.i.sqrt(2) := 
sorry

end solve_expression_l730_730590


namespace g_minus_one_eq_zero_l730_730134

def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

theorem g_minus_one_eq_zero (r : ℝ) : g (-1) r = 0 → r = 14 := by
  sorry

end g_minus_one_eq_zero_l730_730134


namespace find_integers_l730_730363

theorem find_integers (a b : ℤ) : 7^a - 3 * 2^b = 1 ↔ (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 4) :=
by
  sorry

end find_integers_l730_730363


namespace percent_change_area_l730_730080

theorem percent_change_area (l w : ℝ) : 
  let new_length := 0.9 * l;
      new_width := 1.3 * w;
      old_area := l * w;
      new_area := new_length * new_width;
      percent_change := ((new_area / old_area) - 1) * 100 in 
  percent_change = 17 :=
by
  let new_length := 0.9 * l
  let new_width := 1.3 * w
  let old_area := l * w
  let new_area := new_length * new_width
  let percent_change := ((new_area / old_area) - 1) * 100
  sorry

end percent_change_area_l730_730080


namespace probability_two_kings_or_at_least_one_ace_l730_730856

theorem probability_two_kings_or_at_least_one_ace :
  let total_cards := 52
  let number_aces := 4
  let number_kings := 4
  probability_two_kings_or_at_least_one_ace (total_cards number_aces number_kings) = (2/13) :=
by
  sorry

end probability_two_kings_or_at_least_one_ace_l730_730856


namespace kittens_weight_problem_l730_730997

theorem kittens_weight_problem
  (w_lightest : ℕ)
  (w_heaviest : ℕ)
  (w_total : ℕ)
  (total_lightest : w_lightest = 80)
  (total_heaviest : w_heaviest = 200)
  (total_weight : w_total = 500) :
  ∃ (n : ℕ), n = 11 :=
by sorry

end kittens_weight_problem_l730_730997


namespace sum_reciprocals_distances_const_l730_730553

-- Define the variables and their types
variables (n : ℕ) (a c b : ℝ)
-- Define the conditions as hypotheses
hypothesis h_pos : n > 1
hypothesis h_angle_eq : ∀ i j : ℕ, (1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j <= n) → ∠ P_i F P_{i+1} = ∠ P_{j} F P_{j+1} = 2 * π / n
hypothesis h_ellipse_params : b^2 = a^2 - c^2

-- Define the distances (this might need ellaborating on the exact meaning in geometry and context)
def d_i (P_i : point) : ℝ := sorry

theorem sum_reciprocals_distances_const (n a c b : ℝ) 
  (h_pos : n > 1)
  (h_angle_eq : ∀ i j : ℕ, (1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j <= n) → ∠ P_i F P_i1 = ∠ P_j F P_j1 = 2 * π / n)
  (h_ellipse_params : b^2 = a^2 - c^2) :
  ∑ i in finset.range(n), 1 / (d_i i) = n * c / b^2 :=
by
  sorry

end sum_reciprocals_distances_const_l730_730553


namespace find_base_b_l730_730981

theorem find_base_b (x : ℝ) (b : ℝ) : (9^(x + 5) = 10^x ∧ x = log b (9^5)) → b = 10 / 9 := by
  sorry

end find_base_b_l730_730981


namespace corn_height_after_three_weeks_l730_730726

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l730_730726


namespace same_function_D_l730_730314

noncomputable def f : ℝ → ℝ := λ x, real.log (1 - x) + real.log (1 + x)
noncomputable def g : ℝ → ℝ := λ x, real.log (1 - x^2)

theorem same_function_D : ∀ x, has_domain x (1 > x ∧ x > -1) → f x = g x := 
by
  intros x domain_x
  sorry

end same_function_D_l730_730314


namespace find_n_l730_730525

theorem find_n (n : ℕ) :
  (∀ (x : ℝ), (1 + x)^n = ∑ i in range (n+1), (nat.choose n i) * (x ^ i)) →
  ∀ (n : ℕ), (nat.choose n 2 = nat.choose n 3) → n = 5 := 
by 
  intros h1 h2;
  sorry

end find_n_l730_730525


namespace general_term_a_sum_first_n_terms_c_l730_730039

-- Definitions based on given conditions
def S (n : ℕ) : ℕ := 2 * a n - 2

-- Question 1: Prove that a_n = 2^n
theorem general_term_a (n : ℕ) : a n = 2^n := 
sorry

-- Question 2: Prove that T_n = n * 2^{n+1}
def c (n : ℕ) : ℕ := (n + 1) * a n

theorem sum_first_n_terms_c (n : ℕ) : T n = n * 2^(n+1) :=
sorry

end general_term_a_sum_first_n_terms_c_l730_730039


namespace sufficient_not_necessary_condition_l730_730790

theorem sufficient_not_necessary_condition (x : ℝ) : (|x - 1/2| < 1/2) → (x^3 < 1) ∧ ¬(x^3 < 1) → (|x - 1/2| < 1/2) :=
sorry

end sufficient_not_necessary_condition_l730_730790


namespace ratio_is_one_quarter_l730_730114

def Joel_garden_area : ℚ := 64
def garden_half : ℚ := Joel_garden_area / 2
def strawberry_area : ℚ := 8
def fruit_section_area : ℚ := garden_half
def ratio_strawberries_to_fruit_section : ℚ := strawberry_area / fruit_section_area

theorem ratio_is_one_quarter :
  garden_half = 32 ∧ strawberry_area = 8 ∧ ratio_strawberries_to_fruit_section = 1 / 4 :=
by
  split
  . exact rfl
  . split
    . exact rfl
    . exact rfl

end ratio_is_one_quarter_l730_730114


namespace triangle_BPE_is_isosceles_l730_730995

-- Definitions based on the conditions

variables {A B C D E P : Point}

-- Given that B is not on line AC, we assume lines are defined as sets of points
axiom h1 : perpendicular (Line.mk B A) (Line.mk A B)  -- Perpendicular to side AB at B
axiom h2 : OnLine D (altitude B (Line.mk B C))        -- D lies on the altitude from B to BC
axiom h3 : OnLine P (perpendicular_bisector B C)     -- P lies on the perpendicular bisector of BC
axiom h4 : OnLine P (Line.mk B D)                    -- P also lies on line BD
axiom h5 : perpendicular (Line.mk D E) (Line.mk A C) -- DE is perpendicular to AC at E
 
-- Theorem to prove
theorem triangle_BPE_is_isosceles :
  isosceles_triangle B P E :=
sorry

end triangle_BPE_is_isosceles_l730_730995


namespace volume_pyramid_with_spheres_eq_l730_730219

theorem volume_pyramid_with_spheres_eq :
  ∀ (a : ℝ), a > 0 →
  let pyramid_height := -a / 2,
      sphere_radius := a / 3 in
  volume_of_pyramid_with_spheres a pyramid_height sphere_radius = (81 - 32 * Real.pi) / 486 * a^3 :=
by
  sorry

end volume_pyramid_with_spheres_eq_l730_730219


namespace Marcus_pebbles_final_count_l730_730547

theorem Marcus_pebbles_final_count : 
  let initial_pebbles := 250
  let pebbles_given := 0.10 * initial_pebbles
  let pebbles_after_giving := initial_pebbles - pebbles_given
  let pebbles_skipped := 1/3 * pebbles_after_giving
  let pebbles_after_skipping := pebbles_after_giving - pebbles_skipped
  let pebbles_after_finding := pebbles_after_skipping + 60
  let pebbles_dropped := 1/2 * pebbles_after_finding
  let final_pebbles := pebbles_after_finding - pebbles_dropped
  in final_pebbles = 105 :=
by
  sorry

end Marcus_pebbles_final_count_l730_730547


namespace probability_of_right_triangle_area_l730_730410

noncomputable def probability_area_between 
  (AB_length : ℝ)
  (P : ℝ → Prop)
  (lower_area_limit : ℝ) 
  (upper_area_limit : ℝ) 
  (base : ℝ) 
  (height_coefficient : ℝ) 
  (area : ℝ) : ℝ :=
  let APlower := (2 : ℝ) in
  let APupper := (8 : ℝ) in
  (APupper - APlower) / AB_length

theorem probability_of_right_triangle_area :
  ∀ (AP : ℝ), ∃ (AB_length : ℝ) (APlower APupper : ℝ),
    (APlower = 2 ∧ APupper = 8 ∧
     AB_length = 20 ∧
     APlower ≤ AP ∧ AP ≤ APupper → 
     probability_area_between 20 (λ AP, True) (sqrt 3) (16 * sqrt 3) AP (sqrt 3 / 2) (sqrt 3 / 4 * AP^2) = 3 / 10) :=
sorry

end probability_of_right_triangle_area_l730_730410


namespace op_4_3_equals_52_div_9_l730_730214

def op (a b : ℝ) : ℝ := a * (1 + a / b^2)

theorem op_4_3_equals_52_div_9 : op 4 3 = 52 / 9 :=
by
  sorry

end op_4_3_equals_52_div_9_l730_730214


namespace rectangle_uniqueness_of_equal_diagonals_l730_730596

/-- A rectangle has diagonals that are equal in length, while a rhombus does not necessarily have this property. -/
theorem rectangle_uniqueness_of_equal_diagonals (rect rhomb : Type) 
  (P1 : ∀ (r : rect), r.diagonals_are_perpendicular = false)
  (P2 : ∀ (r : rect), r.diagonals_bisect_each_other = true)
  (P3 : ∀ (r : rect), r.diagonals_are_equal_in_length = true)
  (P4 : ∀ (r : rect), r.diagonals_bisect_angles = false)
  (P5 : ∀ (h : rhomb), h.diagonals_are_perpendicular = true)
  (P6 : ∀ (h : rhomb), h.diagonals_bisect_each_other = true)
  (P7 : ∀ (h : rhomb), h.diagonals_are_equal_in_length = h.is_square)
  (P8 : ∀ (h : rhomb), h.diagonals_bisect_angles = true) :
  (P3 = true ∧ P7 = false) :=
by
  sorry

end rectangle_uniqueness_of_equal_diagonals_l730_730596


namespace admission_cutoff_score_l730_730698

theorem admission_cutoff_score (n : ℕ) (x : ℚ) (admitted_average non_admitted_average total_average : ℚ)
    (h1 : admitted_average = x + 15)
    (h2 : non_admitted_average = x - 20)
    (h3 : total_average = 90)
    (h4 : (admitted_average * (2 / 5) + non_admitted_average * (3 / 5)) = total_average) : x = 96 := 
by
  sorry

end admission_cutoff_score_l730_730698


namespace conjugate_quadratic_irrationality_in_interval_l730_730951

theorem conjugate_quadratic_irrationality_in_interval
  (A D B : ℚ) (hD_pos : 0 < D) (hB_pos : 0 < B)
  (alpha : ℝ) (h_alpha_def : alpha = (A + real.sqrt D) / B) :
  α_periodic_cf ?purely_periodic α → 
  let α' := (A - real.sqrt D) / B in -1 < α' ∧ α' < 0 :=
begin
  sorry
end

end conjugate_quadratic_irrationality_in_interval_l730_730951


namespace find_f_of_2_l730_730794

theorem find_f_of_2 (a b : ℝ) (h : (λ x : ℝ, a * x ^ 3 + b * x + 2) (-2) = -7) : (λ x : ℝ, a * x ^ 3 + b * x + 2) 2 = 11 :=
by
  sorry

end find_f_of_2_l730_730794


namespace common_difference_of_geometric_progression_l730_730554

noncomputable def geometric_arithmetic_progression (a b c q : ℝ) (d : ℝ) : Prop :=
a = b / q ∧ c = b * q ∧ (log a b - d) = (log b c) - d ∧ (log b c) = (log c a) + d ∧ q ≠ 1

theorem common_difference_of_geometric_progression (a b c q d : ℝ) (h : geometric_arithmetic_progression a b c q d) :
  d = -3 / 2 :=
sorry

end common_difference_of_geometric_progression_l730_730554


namespace compute_expression_l730_730736

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end compute_expression_l730_730736


namespace find_m_eq_4_l730_730125

theorem find_m_eq_4 (m : ℝ) (h₁ : ∃ (A B C : ℝ × ℝ), A = (m, -m+3) ∧ B = (2, m-1) ∧ C = (-1, 4)) (h₂ : (4 - (-m+3)) / (-1-m) = 3 * ((m-1) - 4) / (2 - (-1))) : m = 4 :=
sorry

end find_m_eq_4_l730_730125


namespace rectangle_area_l730_730747

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

-- Define the condition for tangency for (x1, y1)
def tangent_cond (x1 y1 : ℝ) : Prop := y1 = 4 * x1 / 9

-- Define the conditions for x1
def x1_value (x1 : ℝ) : Prop := x1 ^ 2 = 81 / 13

-- Define the conditions for y1 based on x1
def y1_value (x1 y1 : ℝ) : Prop := y1 = 4 * x1 / 9

-- Define the points of tangency
def tangency_points (x1 y1 : ℝ) : Prop :=
  ellipse x1 y1 ∧ tangent_cond x1 y1 ∧ x1_value x1 ∧ y1_value x1 y1

-- The main theorem stating that the area of the rectangle is 144/13
theorem rectangle_area {x1 y1 : ℝ} 
  (h1 : tangency_points x1 y1) : 
  4 * ((9 * real.sqrt(13) / 13) * (4 * real.sqrt(13) / 13)) = 144 / 13 :=
by
  sorry

end rectangle_area_l730_730747


namespace hexagon_angle_CAB_is_30_l730_730486

theorem hexagon_angle_CAB_is_30 (ABCDEF : Type) [hexagon : IsRegularHexagon ABCDEF]
  (A B C D E F : Point ABCDEF) (h1 : InteriorAngle ABCDEF = 120) (h2 : IsDiagonal AC ) :
  ∠CAB = 30 :=
by
  sorry

end hexagon_angle_CAB_is_30_l730_730486


namespace work_completion_l730_730320

theorem work_completion (x y z : ℕ) (hx : x = 30) (hy : y = 24) (hz : z = 18) :
  (4 / y) + 6 * (1 / x + 1 / y) + (9 / x) = 1 ∧
  (y = 3 * x) ∧
  (z = 3 * x / 2) :=
begin
  sorry
end

end work_completion_l730_730320


namespace exists_vertex_with_acute_plane_angles_l730_730953

-- Definition of a tetrahedron and its properties
structure Tetrahedron where
  vertices : List Point 
  faces : List Face
  -- Additional properties defining the tetrahedron could be added here

-- Placeholder for Point and Face definitions
structure Point where
  x : ℝ 
  y : ℝ 
  z : ℝ

structure Face where
  points : List Point

-- The sum of all plane angles in the tetrahedron is 4π
def sum_of_plane_angles (T : Tetrahedron) : Prop :=
  -- This would be an actual calculation of the sum of the internal angles at all vertices
  sorry

-- Statement of the theorem
theorem exists_vertex_with_acute_plane_angles (T : Tetrahedron) (h : sum_of_plane_angles T) : 
  ∃ (v : Point), at_vertex_all_angles_acute v T :=
by
  -- Proof would go here
  sorry

-- Auxiliary definitions
def at_vertex_all_angles_acute (v : Point) (T : Tetrahedron) : Prop :=
  -- Placeholder for the condition that all plane angles at the vertex v are acute
  sorry

end exists_vertex_with_acute_plane_angles_l730_730953


namespace ethanol_percentage_optimal_l730_730280

theorem ethanol_percentage_optimal (initial_volume : ℝ) (ethanol_percentage : ℝ) (added_ethanol : ℝ) 
                                     (total_volume : ℝ) (desired_percentage : ℝ) :
  initial_volume = 27 →
  ethanol_percentage = 0.05 →
  added_ethanol = 1.5 →
  total_volume = initial_volume + added_ethanol →
  desired_percentage = (initial_volume * ethanol_percentage + added_ethanol) / total_volume * 100 →
  desired_percentage = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5


end ethanol_percentage_optimal_l730_730280


namespace quadratic_function_solution_l730_730776

theorem quadratic_function_solution {a b : ℝ} :
  (∀ x : ℝ, (x^2 + (a + 1)*x + b)^2 + a*(x^2 + (a + 1)*x + b) + b = f(f x + x)) →
  (∀ x : ℝ, f(f x + x) = (f x) * (x^2 + 1776*x + 2010)) →
  a = 1774 ∧ b = 235
    → ∀ x : ℝ, f x = x^2 + 1774*x + 235 :=
begin
  -- sorry:
  sorry
end

end quadratic_function_solution_l730_730776


namespace find_X_l730_730385

-- Defining the given conditions and what we need to prove
theorem find_X (X : ℝ) (h : (X + 43 / 151) * 151 = 2912) : X = 19 :=
sorry

end find_X_l730_730385


namespace twins_must_be_present_l730_730236

/-
We are given:
- Vasya and Masha got married in 1994.
- They had four children by the New Year of 2015, celebrated with all six of them.
- All children were born on February 6.
- Today is February 7, 2016.
- The age of the oldest child is equal to the product of the ages of the three younger children.

Our goal: Prove that there are twins in this family.
-/

theorem twins_must_be_present 
  (a_1 a_2 a_3 a_4 : ℕ)
  (h1 : a_1 ≤ a_2) (h2 : a_2 ≤ a_3) (h3 : a_3 ≤ a_4)
  (h_conditions : a_1 + a_2 + a_3 + a_4 = 4 * (2016 - 1994) - 1)
  (h_product : a_4 = a_1 * a_2 * a_3) : 
  ∃ x y, x = y ∧ (x ∈ {a_1, a_2, a_3, a_4}) ∧ (y ∈ {a_1, a_2, a_3, a_4}) :=
by
  sorry

end twins_must_be_present_l730_730236


namespace eccentricity_of_ellipse_l730_730286

def point : Type := ℝ × ℝ
def slope (p1 p2 : point) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
def is_midpoint (m a b : point) : Prop :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

def ellipse (a b : ℝ) : set point :=
  {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 - (b / a)^2)

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 1)
  (M A B : point)
  (hm : M = (1, 1))
  (hslope : slope M A = -1 / 2)
  (hmid : is_midpoint M A B)
  (hA : A ∈ ellipse a b)
  (hB : B ∈ ellipse a b) :
  eccentricity a b = sqrt 1 - (b / a)^2 := 
sorry

end eccentricity_of_ellipse_l730_730286


namespace largest_5_digit_int_congruent_to_18_mod_25_l730_730622

theorem largest_5_digit_int_congruent_to_18_mod_25 :
  ∃ x : ℕ, x < 100000 ∧ x % 25 = 18 ∧ ∀ y : ℕ, y < 100000 ∧ y % 25 = 18 → y ≤ x :=
begin
  use 99993,
  split,
  { -- Proof of 99993 being less than 100000
    sorry
  },
  split,
  { -- Proof of 99993 % 25 = 18
    sorry
  },
  { -- Proof that 99993 is the largest such number
    sorry
  }
end

end largest_5_digit_int_congruent_to_18_mod_25_l730_730622


namespace right_angle_LEM_l730_730907

noncomputable def right_triangle (A B C : Point) := ∃ (K L M N : Point),
(triangle A B C ∧ ∠ACB = 90 ∧ is_square A C K L ∧ is_square B C M N) 

theorem right_angle_LEM (A B C E L M : Point) :
right_triangle A B C →
altitude C E A B →
∠LEM = 90 :=
by 
sorry 

end right_angle_LEM_l730_730907


namespace difference_in_amount_l730_730685

-- Problem conditions
variable (P Q : ℝ)
-- Definitions derived from conditions
def initialTotalCost := P * Q
def newPrice := 1.20 * P
def newQuantity := 0.70 * Q
def secondTotalCost := newPrice * newQuantity
def difference := secondTotalCost - initialTotalCost

-- Statement to prove
theorem difference_in_amount : difference P Q = -0.16 * P * Q :=
by 
  sorry

end difference_in_amount_l730_730685


namespace sine_translation_monotonicity_l730_730079

theorem sine_translation_monotonicity :
  ∀ x, ∀ y, ∀ z,
    g(x) = sin (2 * x - π / 3) →
    (∀ (x ∈ (Ioc -π/12 (5 * π / 12))),
      g' x > 0) ∧
    ¬ (∀ (y ∈ (Ioc -5 * π / 12 -π/6)),
      g' y > 0)
    ∧
    ¬ (g(z) = g(-z - π/3)) ∧
    ¬ (g(z) = g(z - π/3)) :=
by
  sorry

end sine_translation_monotonicity_l730_730079


namespace max_value_of_sinx_over_2_minus_cosx_l730_730927

theorem max_value_of_sinx_over_2_minus_cosx (x : ℝ) : 
  ∃ y_max, y_max = (Real.sqrt 3) / 3 ∧ ∀ y, y = (Real.sin x) / (2 - Real.cos x) → y ≤ y_max :=
sorry

end max_value_of_sinx_over_2_minus_cosx_l730_730927


namespace cost_of_each_candy_bar_l730_730220

theorem cost_of_each_candy_bar
  (p_chips : ℝ)
  (total_cost : ℝ)
  (num_students : ℕ)
  (num_chips_per_student : ℕ)
  (num_candy_bars_per_student : ℕ)
  (h1 : p_chips = 0.50)
  (h2 : total_cost = 15)
  (h3 : num_students = 5)
  (h4 : num_chips_per_student = 2)
  (h5 : num_candy_bars_per_student = 1) :
  ∃ C : ℝ, C = 2 := 
by 
  sorry

end cost_of_each_candy_bar_l730_730220


namespace diagonal_length_of_square_l730_730699

/-- A square has a perimeter of 800 cm. Calculate the length of the diagonal. --/
theorem diagonal_length_of_square (perimeter : ℝ) (side length : ℝ) (diagonal : ℝ) 
    (h1 : perimeter = 800) 
    (h2 : side length = perimeter / 4) 
    (h3 : diagonal = side length * Real.sqrt 2) : 
    diagonal = 200 * Real.sqrt 2 := 
sorry

end diagonal_length_of_square_l730_730699


namespace smallest_sum_divisible_by_3_l730_730740

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def is_consecutive_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4 ∨ p2 = p1 + 6 ∨ p2 = p1 + 2) ∧
  (p3 = p2 + 2 ∨ p3 = p2 + 4) ∧
  (p4 = p3 + 2 ∨ p4 = p3 + 4)

def greater_than_5 (p : ℕ) : Prop := p > 5

theorem smallest_sum_divisible_by_3 :
  ∃ (p1 p2 p3 p4 : ℕ), is_consecutive_prime p1 p2 p3 p4 ∧
                      greater_than_5 p1 ∧
                      (p1 + p2 + p3 + p4) % 3 = 0 ∧
                      (p1 + p2 + p3 + p4) = 48 :=
by sorry

end smallest_sum_divisible_by_3_l730_730740


namespace surface_area_of_inscribed_sphere_l730_730823

theorem surface_area_of_inscribed_sphere (a : ℝ) (h : a = 2) : 4 * Real.pi * (a / 2)^2 = 4 * Real.pi :=
by
  rw h
  norm_num
  ring

end surface_area_of_inscribed_sphere_l730_730823


namespace ellipse_chord_constant_value_l730_730009

theorem ellipse_chord_constant_value :
  ∀ (m : ℝ), 
  let a := 5 
  let b := 4 in 
  (∀ P A B : ℝ × ℝ, 
    ∃ k : ℝ, (k = 4 / 5) ∧ 
      P.1 = m ∧ P.2 = 0 ∧ 
      (A.1, A.2) ∈ {x : ℝ × ℝ | x.2 = k * (x.1 - P.1) + P.2}
      ∧ (A.1^2 / (a^2) + A.2^2 / (b^2) = 1)
      ∧ (B.1, B.2) ∈ {x : ℝ × ℝ | x.2 = k * (x.1 - P.1) + P.2}
      ∧ (B.1^2 / (a^2) + B.2^2 / (b^2) = 1)) → 
  (∀ P A B, (|A.1 - P.1|^2 + A.2^2 + |B.1 - P.1|^2 + B.2^2) = 41) :=
by sorry

end ellipse_chord_constant_value_l730_730009


namespace edge_length_of_prism_l730_730228

-- Definitions based on conditions
def rectangular_prism_edges : ℕ := 12
def total_edge_length : ℕ := 72

-- Proof problem statement
theorem edge_length_of_prism (num_edges : ℕ) (total_length : ℕ) (h1 : num_edges = rectangular_prism_edges) (h2 : total_length = total_edge_length) : 
  (total_length / num_edges) = 6 :=
by {
  -- The proof is omitted here as instructed
  sorry
}

end edge_length_of_prism_l730_730228


namespace _l730_730012

open Real

noncomputable theorem problem_statement 
  {r p q : ℕ} (x : ℝ)
  (h1 : (1 + sin x) * (1 + cos x) = 9 / 4)
  (h2 : (1 - sin x) * (1 - cos x) = p / q - sqrt r)
  (h3 : Nat.Coprime p q)
  (h4 : 0 < r) (h5 : 0 < p) (h6 : 0 < q) :
  r + p + q = 28 :=
sorry

end _l730_730012


namespace ellipse_standard_eqn_line_MN_fixed_point_max_area_BMN_l730_730414

-- Define the ellipse and the given conditions
noncomputable def ellipse_eqn (a b x y : ℝ) (h : a > b ∧ b > 0) :=
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def line_condition (x y : ℝ) :=
  (3 * x + sqrt(3) * y - 3 = 0)

-- The fixed point condition
constant B : ℝ × ℝ := (0, sqrt(3))
constant A : ℝ × ℝ := (1, 0)  -- right focus F2

-- Define the k conditions
noncomputable def k_condition (x1 y1 x2 y2 : ℝ) :=
  (y1 - B.snd) / x1 * (y2 - B.snd) / x2 = 1 / 4

-- Proof problem 1: Standard Equation of Ellipse
theorem ellipse_standard_eqn (a b : ℝ) (x y : ℝ) (h : ellipse_eqn a b x y ∧ a > b ∧ b > 0 ∧ line_condition 1 0 ∧ line_condition 0 (sqrt(3))) :
  (a^2 = 4 ∧ b^2 = 3) →
  (x = y^2 / 4 + x^2 / 3 = 1) := sorry

-- Proof problem 2: Line MN passing through fixed point
theorem line_MN_fixed_point (x1 y1 x2 y2 : ℝ) (h : ellipse_eqn 2 sqrt(3) x1 y1 ∧ ellipse_eqn 2 sqrt(3) x2 y2 ∧ k_condition x1 y1 x2 y2) :
  ∃ (m k : ℝ), (y1 = k * x1 + m ∧ y2 = k * x2 + m ∧ m = 2 * sqrt(3)) := sorry

-- Proof problem 3: Maximum area of triangle BMN
theorem max_area_BMN (x1 y1 x2 y2 k m : ℝ) (h : ellipse_eqn 2 sqrt(3) x1 y1 ∧ ellipse_eqn 2 sqrt(3) x2 y2 ∧ k_condition x1 y1 x2 y2 ∧ m = 2 * sqrt(3)) :
  (4 * k^2 - 9 > 0) →
  (S = (sqrt(13) / 2) ∧ (k = sqrt(21) / 2 ∨ k = -sqrt(21) / 2)) :=
  sorry

end ellipse_standard_eqn_line_MN_fixed_point_max_area_BMN_l730_730414


namespace evaluate_x_squared_minus_y_squared_l730_730031

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l730_730031


namespace polygon_sides_l730_730683

-- Define the given conditions as Lean definitions.
def convex_polygon (n : ℕ) : Prop :=
  ∃ (s : ℕ → ℝ), (∀ i : fin n, 0 < s i ∧ s i < 180) ∧ s.sum.to_list.sum = 180 * (n - 2)

def angle_sum_condition (n : ℕ) : Prop :=
  (∃ s : ℕ → ℝ, ∀ i : fin n, 0 < s i ∧ s i < 180) ∧ (s.to_list.remove_last.sum = 2009)

-- Define the statement we want to prove.
theorem polygon_sides (n : ℕ) (h₁ : convex_polygon n) (h₂ : angle_sum_condition n) : n = 14 :=
sorry

end polygon_sides_l730_730683


namespace hexagon_area_l730_730696

theorem hexagon_area (O : Point) (A1 A2 A3 A4 A5 A6 : Point) (S : ℝ)
  (h_regular : regular_hexagon A1 A2 A3 A4 A5 A6 S)
  (h_point_inside : point_inside_hexagon O A1 A2 A3 A4 A5 A6) :
  ∃ (B1 B2 B3 B4 B5 B6 : Point), 
  formed_hexagon_with_point O A1 A2 A3 A4 A5 A6 B1 B2 B3 B4 B5 B6 ∧
  hexagon_area B1 B2 B3 B4 B5 B6 ≥ (2 / 3) * S :=
sorry

end hexagon_area_l730_730696


namespace even_rows_pascal_triangle_up_to_30_l730_730351

noncomputable def count_even_rows_up_to (n : ℕ) : ℕ :=
  (List.range (n + 1)).count (λ k, k ≠ 0 ∧ (k.lsb = 0 ∧ k ≠ 1))

theorem even_rows_pascal_triangle_up_to_30 :
  count_even_rows_up_to 30 = 4 :=
by {
  sorry
}

end even_rows_pascal_triangle_up_to_30_l730_730351


namespace annipanni_wins_game_l730_730715

theorem annipanni_wins_game :
  ∀ (digits : ℕ → ℕ), (∀ n, digits n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  digits 0 ≠ 0 → 
  (λ (S: ℕ),  S = alternating_sum digits 100 11 ∧ alternating_sum digits 100 0 ) → 
  ¬(alternating_sum digits 100 % 11 = 5)
by sorry

def alternating_sum (digits : ℕ → ℕ) (n : ℕ) : ℕ :=
if n = 0 then digits n
else digits n - alternating_sum digits (n - 1) 

end annipanni_wins_game_l730_730715


namespace common_divisors_cardinality_l730_730843

def divisors (n : ℤ) : Set ℤ := {d | d ∣ n}

theorem common_divisors_cardinality : 
  (divisors 45 ∩ divisors 75).cardinality = 8 :=
sorry

end common_divisors_cardinality_l730_730843


namespace jacob_third_test_score_l730_730896

theorem jacob_third_test_score :
  ∀ (a_1 a_2 a_4 a_5 : ℕ), 
    a_1 = 85 → 
    a_2 = 79 → 
    a_4 = 84 → 
    a_5 = 85 → 
    (∃ x : ℕ, (a_1 + a_2 + x + a_4 + a_5 = 5 * 85) ∧ x = 92) :=
by
  intros a_1 a_2 a_4 a_5 h_a1 h_a2 h_a4 h_a5
  use 92
  split
  . rw [h_a1, h_a2, h_a4, h_a5]
    sorry
  . sorry

end jacob_third_test_score_l730_730896


namespace monomial_type_equivalence_l730_730315

-- Definition of the given monomials
def monomial1 := -3 * a * b ^ 3
def monomial2 := (1 / 2) * b * a ^ 2
def monomial3 := 2 * a * b ^ 2
def monomial4 := 3 * a ^ 2 * b ^ 2
def target_monomial := -3 * a * b ^ 2

-- Definition of the type equivalence condition
def is_same_type (m1 m2 : ℤ → ℤ → ℤ) : Prop :=
  (∃ (k1 k2 : ℤ), m1 = k1 * a ^ 1 * b ^ 2) ∧ 
  (∃ (k3 k4 : ℤ), m2 = k3 * a ^ 1 * b ^ 2)

-- The problem statement to be proven
theorem monomial_type_equivalence : is_same_type monomial3 target_monomial :=
  sorry

end monomial_type_equivalence_l730_730315


namespace sqrt_meaningful_iff_l730_730479

theorem sqrt_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 4)) ↔ x ≥ 4 :=
by
  sorry

end sqrt_meaningful_iff_l730_730479


namespace curvilinear_triangle_area_half_l730_730944

variable {Point : Type*}
variable (O A B C : Point)
variable (r : ℝ) -- radius (half of the diameter, and segments OA, OB, OC are diameters)

noncomputable def triangle_ABC_area (A B C : Point) : ℝ := 
  sorry -- Definition of area calculation for triangle ABC

noncomputable def curvilinear_triangle_area (O A B C : Point) : ℝ := 
  sorry -- Definition of area calculation for curvilinear triangle

-- Conditions
axiom segments_are_equal : dist O A = dist O B ∧ dist O B = dist O C
axiom B_inside_AOC : sorry -- Condition ensuring B is inside angle AOC
axiom circles_constructed : sorry -- Condition describing circles on segments OA, OB, OC

-- Theorem statement
theorem curvilinear_triangle_area_half
  (segments_are_equal : dist O A = dist O B ∧ dist O B = dist O C)
  (B_inside_AOC : sorry)
  (circles_constructed : sorry) :
  curvilinear_triangle_area O A B C = 1 / 2 * triangle_ABC_area A B C :=
sorry

end curvilinear_triangle_area_half_l730_730944


namespace lattice_point_count_l730_730773

theorem lattice_point_count :
  (∃ x y : ℤ, x^2 - y^2 = 75 ∧ x - y = 5) →
  (∃ y : ℤ, ∃ x : ℤ, (x = 10 ∧ y = 5))) →
  ∃ n : ℕ, n = 1 :=
by
  sorry

end lattice_point_count_l730_730773


namespace factorization_1_min_value_l730_730959

-- Problem 1: Prove that m² - 4mn + 3n² = (m - 3n)(m - n)
theorem factorization_1 (m n : ℤ) : m^2 - 4*m*n + 3*n^2 = (m - 3*n)*(m - n) :=
by
  sorry

-- Problem 2: Prove that the minimum value of m² - 3m + 2015 is 2012 3/4
theorem min_value (m : ℝ) : ∃ x : ℝ, x = m^2 - 3*m + 2015 ∧ x = 2012 + 3/4 :=
by
  sorry

end factorization_1_min_value_l730_730959


namespace kittens_weight_problem_l730_730998

theorem kittens_weight_problem
  (w_lightest : ℕ)
  (w_heaviest : ℕ)
  (w_total : ℕ)
  (total_lightest : w_lightest = 80)
  (total_heaviest : w_heaviest = 200)
  (total_weight : w_total = 500) :
  ∃ (n : ℕ), n = 11 :=
by sorry

end kittens_weight_problem_l730_730998


namespace number_of_distinct_stackings_l730_730251

-- Defining the conditions
def cubes : ℕ := 8
def edge_length : ℕ := 1
def valid_stackings (n : ℕ) : Prop := 
  n = 8 -- Stating that we are working with 8 cubes

-- The theorem stating the problem and expected solution
theorem number_of_distinct_stackings : 
  cubes = 8 ∧ edge_length = 1 ∧ valid_stackings cubes → ∃ (count : ℕ), count = 10 :=
by 
  sorry

end number_of_distinct_stackings_l730_730251


namespace find_angle_l730_730793

def quadratic_eq (A x : ℝ) : ℝ := x^2 * cos A - 2 * x + cos A

def diff_of_squares_of_roots (α β : ℝ) : ℝ :=
  α^2 - β^2 = 3 / 8

theorem find_angle (A : ℝ) (hC : 0 < A ∧ A < π) :
  (∃ α β : ℝ, quadratic_eq A α = 0 ∧ quadratic_eq A β = 0 ∧ diff_of_squares_of_roots α β) →
  A = π / 36 :=
by
  sorry

end find_angle_l730_730793


namespace cannot_obtain_xn_minus_1_l730_730504

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem cannot_obtain_xn_minus_1 :
  ∀ (n : ℕ), n > 0 → ¬∃ (h : ℝ → ℝ) (c : ℝ), h = (x^n - 1) where
    (h = f + g ∨ h = f - g ∨ h = f * g ∨ h = f ∘ g ∨ h = c • f) :=
begin
  sorry
end

end cannot_obtain_xn_minus_1_l730_730504


namespace measure_angle_CAB_in_regular_hexagon_l730_730489

-- Definitions used in step a)
def is_regular_hexagon (ABCDEF : Type) : Prop :=
  -- Regular hexagon property: all interior angles are 120 degrees
  ∀ (A B C D E F : ABCDEF), ⦃A, B, C, D, E, F⦄ → 
    ∀ angle, angle ∈ set.interior_angles [A, B, C, D, E, F] → angle = 120

-- Main theorem statement derived from step c)
theorem measure_angle_CAB_in_regular_hexagon
  (ABCDEF : Type)
  (h_reg_hex : is_regular_hexagon ABCDEF)
  (A B C D E F : ABCDEF)
  (h_diag_AC : A ≠ C) : 
  ∃ (angle : ℝ), angle = 30 :=
by 
  sorry

end measure_angle_CAB_in_regular_hexagon_l730_730489


namespace third_group_members_l730_730679

-- Define the total number of members in the choir
def total_members : ℕ := 70

-- Define the number of members in the first group
def first_group_members : ℕ := 25

-- Define the number of members in the second group
def second_group_members : ℕ := 30

-- Prove that the number of members in the third group is 15
theorem third_group_members : total_members - first_group_members - second_group_members = 15 := 
by 
  sorry

end third_group_members_l730_730679


namespace distance_PQ_l730_730110

section DistanceBetweenPoints

open Real

def euclidean_distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (Q.3 - P.3)^2)

def P : ℝ × ℝ × ℝ := (3, 2, 1)
def Q : ℝ × ℝ × ℝ := (-1, 0, 1)

theorem distance_PQ : euclidean_distance P Q = 2 * sqrt 5 :=
  sorry

end DistanceBetweenPoints

end distance_PQ_l730_730110


namespace find_a_l730_730146

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := log (x + a) / log 2

theorem find_a
  (a : ℝ)
  (ha : ∃ x, func a x = 3 ∧ x = 1)
  (h_inv : ∃ y, y = 3 ∧ (∃ x, func a x = y ∧ x = 1)) :
  a = 7 :=
sorry

end find_a_l730_730146


namespace total_amount_Barry_pays_l730_730330

def shirt_price := 80
def pants_price := 100
def shirt_discount := 0.15
def pants_discount := 0.10
def coupon_discount := 0.05

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price - price * discount

def total_price_after_discounts (shirt_price pants_price shirt_discount pants_discount coupon_discount : ℝ) : ℝ :=
  let shirt_final := discounted_price shirt_price shirt_discount
  let pants_final := discounted_price pants_price pants_discount
  let total_before_coupon := shirt_final + pants_final
  total_before_coupon - total_before_coupon * coupon_discount

theorem total_amount_Barry_pays : 
  total_price_after_discounts shirt_price pants_price shirt_discount pants_discount coupon_discount = 150.10 :=
by sorry

end total_amount_Barry_pays_l730_730330


namespace final_label_eq_k_l730_730223

   def A (pile : List ℕ) : List ℕ :=
     match pile with
     | []       => []
     | (x :: xs) => xs ++ [x]

   def B (pile : List ℕ) : List ℕ :=
     match pile with
     | []       => []
     | (x :: xs) => xs

   def apply_operations (pile : List ℕ) (ops : List (List ℕ → List ℕ)) : List ℕ :=
     ops.foldl (λ p op => op p) pile

   def final_card (n : ℕ) : ℕ :=
     let initial_pile := List.range (n + 1) 
     let ops := List.replicate n [A, B, B].join
     match apply_operations initial_pile ops with
     | [x] => x
     | _   => 0 -- this case shouldn't occur

   noncomputable def L (k : ℕ) : ℕ := final_card (3 * k)

   theorem final_label_eq_k (k : ℕ) (i : ℕ) : 
     k = 1 ∨ 
     k = 243 * a_seq i - 35 ∨ 
     k = 243 * b_seq i - 35 → 
     L k = k := by
     sorry

   
end final_label_eq_k_l730_730223


namespace trajectory_N_l730_730014

open Real

noncomputable def M (x0 : ℝ) : ℝ × ℝ := (x0, 0)
noncomputable def P (y0 : ℝ) : ℝ × ℝ := (0, y0)
noncomputable def N (x y : ℝ) : ℝ × ℝ := (x, y)
noncomputable def F : ℝ × ℝ := (1, 0)

theorem trajectory_N : 
  ∀ (x0 y0 x y : ℝ),
  let M := (x0, 0)
  let P := (0, y0)
  let N := (x, y)
  (x0 - y0^2 = 0) → 
  (x - x0 = -2 * x0) → 
  (y = 2 * y0) → 
  y^2 = 4 * x :=
begin
  intros,
  sorry
end

end trajectory_N_l730_730014


namespace sin_beta_equal_neg_three_fifths_l730_730468

theorem sin_beta_equal_neg_three_fifths
  (α β : ℝ)
  (h : sin (α - β) * cos α - cos (α - β) * sin α = 3 / 5) :
  sin β = -3 / 5 := by
  sorry

end sin_beta_equal_neg_three_fifths_l730_730468


namespace problem_solution_l730_730060

def m (x : ℝ) : ℝ × ℝ := ⟨2 * cos x ^ 2, sqrt 3⟩
def n (x : ℝ) : ℝ × ℝ := ⟨1, sin (2 * x)⟩
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem problem_solution : 
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, f (2 * (π / 6) - x) = f x) :=
by
  sorry

end problem_solution_l730_730060


namespace unpacked_books_30_l730_730091

theorem unpacked_books_30 :
  let total_books := 1485 * 42
  let books_per_box := 45
  total_books % books_per_box = 30 :=
by
  let total_books := 1485 * 42
  let books_per_box := 45
  have h : total_books % books_per_box = 30 := sorry
  exact h

end unpacked_books_30_l730_730091


namespace numbers_greater_than_three_count_l730_730226

theorem numbers_greater_than_three_count :
  let s := ({0.8, 1/2, 0.9, 1/3} : set ℝ) in
  (finset.filter (λ x, x > 3) (finset.val s)).card = 0 := 
by
  sorry

end numbers_greater_than_three_count_l730_730226


namespace perimeter_of_hexagon_constructed_outside_equilateral_triangle_l730_730321

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * s ^ 2

noncomputable def smaller_equilateral_triangle_side (area : ℝ) : ℝ :=
  real.sqrt (4 * area / real.sqrt 3)

theorem perimeter_of_hexagon_constructed_outside_equilateral_triangle :
  ∀ (s : ℝ), s = 4 →
      let large_triangle_area := equilateral_triangle_area s in
      let smaller_triangle_area := large_triangle_area / 2 in
      let smaller_triangle_side := smaller_equilateral_triangle_side smaller_triangle_area in
      let hexagon_side := s in
      6 * hexagon_side = 24 :=
by
  intros s hs
  rw hs
  let large_triangle_area := equilateral_triangle_area 4
  let smaller_triangle_area := large_triangle_area / 2
  let smaller_triangle_side := smaller_equilateral_triangle_side smaller_triangle_area
  let hexagon_side := 4
  exact eq.trans (6 * 4) 24 sorry

end perimeter_of_hexagon_constructed_outside_equilateral_triangle_l730_730321


namespace arithmetic_sequence_general_term_l730_730978

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ)
  (h1 : ∀ n m, a (n+1) - a n = a (m+1) - a m)
  (h2 : (a 2 + a 6) / 2 = 5)
  (h3 : (a 3 + a 7) / 2 = 7) :
  ∀ n, a n = 2 * n - 3 :=
by 
  sorry

end arithmetic_sequence_general_term_l730_730978


namespace length_of_CD_l730_730977

-- Define the length of CD
def length_CD : ℝ := 19

-- Define the radius
def radius : ℝ := 4

-- Define the volume of the region within 4 units of the line segment CD
def volume_region (h : ℝ) : ℝ := (π * radius^2 * h) + ((4/3) * π * radius^3)

-- Define the main theorem
theorem length_of_CD (h : ℝ) (V : ℝ) (h_eq : volume_region h = V) : length_CD = 19 := by
  -- Insert actual proof here when required
  sorry

end length_of_CD_l730_730977


namespace measure_angle_CAB_in_regular_hexagon_l730_730487

-- Definitions used in step a)
def is_regular_hexagon (ABCDEF : Type) : Prop :=
  -- Regular hexagon property: all interior angles are 120 degrees
  ∀ (A B C D E F : ABCDEF), ⦃A, B, C, D, E, F⦄ → 
    ∀ angle, angle ∈ set.interior_angles [A, B, C, D, E, F] → angle = 120

-- Main theorem statement derived from step c)
theorem measure_angle_CAB_in_regular_hexagon
  (ABCDEF : Type)
  (h_reg_hex : is_regular_hexagon ABCDEF)
  (A B C D E F : ABCDEF)
  (h_diag_AC : A ≠ C) : 
  ∃ (angle : ℝ), angle = 30 :=
by 
  sorry

end measure_angle_CAB_in_regular_hexagon_l730_730487


namespace solve_y_pos_in_arithmetic_seq_l730_730964

-- Define the first term as 4
def first_term : ℕ := 4

-- Define the third term as 36
def third_term : ℕ := 36

-- Basing on the properties of an arithmetic sequence, 
-- we solve for the positive second term (y) such that its square equals to 20
theorem solve_y_pos_in_arithmetic_seq : ∃ y : ℝ, y > 0 ∧ y ^ 2 = 20 := by
  sorry

end solve_y_pos_in_arithmetic_seq_l730_730964


namespace number_of_permutations_satisfying_condition_l730_730775

theorem number_of_permutations_satisfying_condition : 
  ∃ (ps : Finset (Fin 8 → ℕ)), 
    ps = (Finset.univ : Finset (Fin 8 → ℕ)) .filter (λ (a : Fin 8 → ℕ), 
       (((a 0 + 1)/3) * ((a 1 + 2)/3) * ((a 2 + 3)/3) * ((a 3 + 4)/3) *
        ((a 4 + 5)/3) * ((a 5 + 6)/3) * ((a 6 + 7)/3) * ((a 7 + 8)/3)) > nat.factorial 8)   
    ∧ ps.card = nat.factorial 8 :=
by sorry

end number_of_permutations_satisfying_condition_l730_730775


namespace relative_error_approximation_l730_730614

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  (1 / (1 + y) - (1 - y)) / (1 / (1 + y)) = y^2 :=
by
  sorry

end relative_error_approximation_l730_730614


namespace arithmetic_sequence_product_l730_730601

theorem arithmetic_sequence_product 
  (a d : ℤ)
  (h1 : a + 6 * d = 20)
  (h2 : d = 2) : 
  a * (a + d) * (a + 2 * d) = 960 := 
by
  -- proof goes here
  sorry

end arithmetic_sequence_product_l730_730601


namespace a_plus_b_eq_2007_l730_730815

theorem a_plus_b_eq_2007 (a b : ℕ) (ha : Prime a) (hb : Odd b)
  (h : a^2 + b = 2009) : a + b = 2007 :=
by
  sorry

end a_plus_b_eq_2007_l730_730815


namespace rothschild_ends_with_coins_l730_730175

-- Define the termination conditions and process
def even_condition (n : ℕ) : ℕ := n / 2

def odd_condition (n : ℕ) (num_men : ℕ) : ℕ :=
  if h : n > 0 then
    (n - 1) / 2
  else
    0

def rothschild_ends_with_at_least_one (n : ℕ) : Prop :=
  ∃ N : ℕ, ∀ m ≥ N, let final_coins := 
    if m % 2 = 0 then even_condition m
    else odd_condition m 1 in
  final_coins ≥ 1

-- The main theorem to prove
theorem rothschild_ends_with_coins : ∃ N : ℕ, ∀ n ≥ N, rothschild_ends_with_at_least_one n :=
sorry

end rothschild_ends_with_coins_l730_730175


namespace part1_part2_l730_730001

variable (z : ℂ) (i : ℑ) (m : ℝ)

theorem part1 (h1 : (z + 3 + 4 * i).im = 0) (h2 : ((z / (1 - 2 * i)).im) = 0) : z = 2 - 4 * i :=
sorry

theorem part2 (h3 : ∃ m : ℝ, (z - m * i) = (2 - 4 * i)) :
  (m + 4 < 0) ∧ (4 - (m + 4)^2 < 0) → m < -6 :=
sorry

end part1_part2_l730_730001


namespace sum_of_sequence_l730_730352

theorem sum_of_sequence :
  let sum_odd_squares (n : ℕ) : ℕ :=
    if n % 2 = 0 then ((n - 1)^2 + 2) else ((n - 1)^2 + 1)
  in
  let S (n : ℕ) : ℕ :=
    (n^2 - sum_odd_squares n + 2) / 2 * ((n^2 + sum_odd_squares n) / 2)
  in
  let sequence_term (n : ℕ) : ℕ :=
    (-1) ^ n * n * (n^2 - n + 1)
  in
  (∑ n in range (71), sequence_term n) = "final answer" sorry

end sum_of_sequence_l730_730352


namespace angle_RTQ_is_97_point_5_degrees_l730_730882

noncomputable def is_straight_line (R S P : Point) : Prop := ∡ (R S P) = 180

/-- Definitions for each condition in the math problem. --/
def G (Q S P : Point) : Prop := ∡ Q S P = 70
def H (R S P Q : Point) : Prop := R S P ∧ SP = PQ 
def special_point (T P Q : Point) : Prop := T ∈ segment P Q ∧ PT = TQ

/-- The main theorem based on the problem statement. --/
theorem angle_RTQ_is_97_point_5_degrees
  (R S P Q T : Point)
  (h1 : is_straight_line R S P)
  (h2 : G Q S P) 
  (h3 : H R S P Q)
  (h4 : special_point T P Q) : 
  ∡ R T Q = 97.5 :=
sorry

end angle_RTQ_is_97_point_5_degrees_l730_730882


namespace largest_prime_divisor_base3_num_l730_730373

def base3_to_base10 (n : Nat) : Nat :=
  -- function to convert the given base-3 number to base-10
  2 * 3^9 + 1 * 3^8 + 0 * 3^7 + 0 * 3^6 + 2 * 3^5 + 1 * 3^4 + 0 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem largest_prime_divisor_base3_num :
  let n := base3_to_base10 2100210021 in
  n = 46501 ∧ Prime 46501 := 
sorry

end largest_prime_divisor_base3_num_l730_730373


namespace max_chord_length_l730_730824

noncomputable def family_of_curves (θ x y : ℝ) := 
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line (x y : ℝ) := 2 * x = y

theorem max_chord_length :
  (∀ (θ : ℝ), ∀ (x y : ℝ), family_of_curves θ x y → line x y) → 
  ∃ (L : ℝ), L = 8 * Real.sqrt 5 :=
by
  sorry

end max_chord_length_l730_730824


namespace limit_value_l730_730826

def f (x : ℝ) := 2 * log(3 * x) + 8 * x + 1

theorem limit_value :
  (Real.limit (fun Δx => (f (1 - 2 * Δx) - f 1) / Δx) (0 : ℝ)) = -20 :=
by
  sorry

end limit_value_l730_730826


namespace necessary_condition_abs_sq_necessary_and_sufficient_add_l730_730067

theorem necessary_condition_abs_sq (a b : ℝ) : a^2 > b^2 → |a| > |b| :=
sorry

theorem necessary_and_sufficient_add (a b c : ℝ) :
  (a > b) ↔ (a + c > b + c) :=
sorry

end necessary_condition_abs_sq_necessary_and_sufficient_add_l730_730067


namespace range_of_a_l730_730416

variable {α : Type*} [LinearOrder α]

def setA (x : α) : Prop := x ≤ -1 ∨ x ≥ 5
def setB (x a : α) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

theorem range_of_a (a : α) (h : ∀ x, setB x a → setA x) : a ≤ -3 ∨ a > 2 :=
by
  sorry

end range_of_a_l730_730416


namespace max_value_of_x_plus_y_l730_730478

variable (x y : ℝ)

-- Define the condition
def condition : Prop := x^2 + y + 3 * x - 3 = 0

-- Define the proof statement
theorem max_value_of_x_plus_y (hx : condition x y) : x + y ≤ 4 :=
sorry

end max_value_of_x_plus_y_l730_730478


namespace max_elements_A_l730_730796

open Set

noncomputable def max_elements (p : ℕ) (n : ℕ) [Fact (Nat.Prime p)] (hpn : p ≥ n) (hn3 : n ≥ 3) : ℕ :=
p^(n-2)

theorem max_elements_A (p n : ℕ) [Fact (Nat.Prime p)] (hpn : p ≥ n) (hn3 : n ≥ 3) :
  ∃ (A : Finset (Vector (Fin p) n)), (∀ x y ∈ A, x ≠ y → 
    (∃ k l m : Fin n, k ≠ l ∧ l ≠ m ∧ m ≠ k ∧ x k ≠ y k ∧ x l ≠ y l ∧ x m ≠ y m)) ∧ A.card = max_elements p n :=
begin
  -- Insert the steps or constructions from the problem solution if needed.
  sorry,
end

end max_elements_A_l730_730796


namespace smallest_positive_period_find_a_l730_730544

def f (ω : ℝ) (a : ℝ) (x : ℝ) : ℝ := sqrt 3 * cos (ω * x)^2 + sin (ω * x) * cos (ω * x) + a

theorem smallest_positive_period (ω a : ℝ) (hω : ω = 1 / 2) :
  ∃ p > 0, ∀ x, f ω a (x + p) = f ω a x :=
sorry

theorem find_a (a : ℝ) (h_a : ∀ x ∈ Icc (-π / 3) (5 * π / 6), f (1 / 2) a x ≥ √3) :
  a = (√3 + 1) / 2 :=
sorry

end smallest_positive_period_find_a_l730_730544


namespace stones_in_pile_l730_730991

theorem stones_in_pile (initial_stones : ℕ) (final_stones_A : ℕ) (final_stones_B_min final_stones_B_max final_stones_B : ℕ) (operations : ℕ) :
  initial_stones = 2006 ∧ final_stones_A = 1990 ∧ final_stones_B_min = 2080 ∧ final_stones_B_max = 2100 ∧ operations < 20 ∧ (final_stones_B_min ≤ final_stones_B ∧ final_stones_B ≤ final_stones_B_max) 
  → final_stones_B = 2090 :=
by
  sorry

end stones_in_pile_l730_730991


namespace single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l730_730102

section transmission_scheme

variables (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1)

-- Part A
theorem single_transmission_probability :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β) ^ 2 :=
by sorry

-- Part B
theorem triple_transmission_probability :
  (1 - β) * β * (1 - β) = β * (1 - β) ^ 2 :=
by sorry

-- Part C
theorem triple_transmission_decoding :
  (3 * β * (1 - β) ^ 2) + (1 - β) ^ 3 = β * (1 - β) ^ 2 + (1 - β) ^ 3 :=
by sorry

-- Part D
theorem decoding_comparison (h : 0 < α ∧ α < 0.5) :
  (1 - α) < (3 * α * (1 - α) ^ 2 + (1 - α) ^ 3) :=
by sorry

end transmission_scheme

end single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l730_730102


namespace triangle_area_CBF_l730_730105

theorem triangle_area_CBF (A B C D E F : ℝ × ℝ)
  (h₁: A = (0, 0)) (h₂: B = (1, 0)) (h₃: C = (1, 1)) (h₄: D = (0, 1))
  (h₅: E = (0.5, 0.5)) (h₆: F = (0.75, 0.25)) :
  let area (P Q R : ℝ × ℝ) := 1 / 2 * abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))) in
  area C B F = 0.125 :=
sorry

end triangle_area_CBF_l730_730105


namespace slope_of_AC_l730_730498

theorem slope_of_AC (A B C : ℝ × ℝ)
  (h_right_triangle : ∠B = 90°)
  (h_AC_length : dist A C = 225)
  (h_AB_length : dist A B = 180)
  : slope A C = 4 / 3 :=
sorry

end slope_of_AC_l730_730498


namespace num_factors_42_l730_730845

theorem num_factors_42 : ∃ n, n = 42 ∧ number_of_factors 42 = 8 := sorry

end num_factors_42_l730_730845


namespace train_speed_is_85_kmh_l730_730260

noncomputable def speed_of_train_in_kmh (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_kmh : ℝ) : ℝ :=
  let speed_of_man_mps := speed_of_man_kmh * 1000 / 3600
  let relative_speed_mps := length_of_train / time_to_cross
  let speed_of_train_mps := relative_speed_mps - speed_of_man_mps
  speed_of_train_mps * 3600 / 1000

theorem train_speed_is_85_kmh
  (length_of_train : ℝ)
  (time_to_cross : ℝ)
  (speed_of_man_kmh : ℝ)
  (h1 : length_of_train = 150)
  (h2 : time_to_cross = 6)
  (h3 : speed_of_man_kmh = 5) :
  speed_of_train_in_kmh length_of_train time_to_cross speed_of_man_kmh = 85 :=
by
  sorry

end train_speed_is_85_kmh_l730_730260


namespace magnitude_of_sum_of_scaled_vectors_l730_730059

def vector_magnitude (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem magnitude_of_sum_of_scaled_vectors :
  ∃ y : ℝ, ∃ (3a2b : ℝ × ℝ),
  let a := (1, -2)
  let b := (-2, y)
  3a2b = (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2) ∧
  a.1 * y = (-2) * (-2) ∧ 
  vector_magnitude 3a2b.1 3a2b.2 = Real.sqrt 5 :=
by
  sorry

end magnitude_of_sum_of_scaled_vectors_l730_730059


namespace pine_cones_on_roof_l730_730311

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end pine_cones_on_roof_l730_730311


namespace func_even_domain_calc_l730_730817

theorem func_even_domain_calc a b (h_func_even: ∀ x, (-a <= x ∧ x <= a+1) → (ax^2 - bx + 1 = a(-x)^2 - b(-x) + 1))
: a + a^b = 1 / 2 :=
by {
  sorry
}

end func_even_domain_calc_l730_730817


namespace statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_multiple_choice_proof_l730_730629

-- Define the context for statement A
def context_A (c A a : ℝ) := c = Real.sqrt 6 ∧ A = Real.pi / 4 ∧ a = 2

-- Define the correctness for statement A using the Law of Sines
theorem statement_A_correct (c A a : ℝ) (h : context_A c A a) : 
  -- Proof that triangle ABC has two sets of solutions
  ∃ B C : ℝ, (C = Real.arcsin ((c * Real.sin A) / a) ∨ C = Real.pi - Real.arcsin ((c * Real.sin A) / a)) :=
sorry

-- Define the context for statement B
def context_B (a b A B : ℝ) := a^2 * Real.tan B = b^2 * Real.tan A

-- Define the incorrectness for statement B
theorem statement_B_incorrect (a b A B : ℝ) (h : context_B a b A B) : 
  ¬(a = b ∧ A + B = Real.pi / 2) :=
sorry

-- Define the context for statement C
def context_C (AP AB AC BC : ℝ) := 
  ∃ λ : ℝ, AP = λ * (AB / Real.abs AB * Real.cos BC + AC / Real.abs AC * Real.cos AB)

-- Define the incorrectness for statement C
theorem statement_C_incorrect (AP AB AC BC : ℝ) (h : context_C AP AB AC BC) : 
  ¬(∃ circumcenter : ℝ, AP passes through circumcenter) :=
sorry

-- Define the context for statement D
def context_D (A B : ℝ) := A > B

-- Define the correctness for statement D using the property of the sine function
theorem statement_D_correct (A B : ℝ) (h: context_D A B) : 
  Real.sin A > Real.sin B :=
sorry

-- Combining all, prove statements A and D are correct while B and C are incorrect
theorem multiple_choice_proof :
  (statement_A_correct ∧ statement_D_correct) ∧ (statement_B_incorrect ∧ statement_C_incorrect) :=
sorry

end statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_multiple_choice_proof_l730_730629


namespace harry_galleons_l730_730660

/--
Harry, Hermione, and Ron go to Diagon Alley to buy chocolate frogs. 
If Harry and Hermione spent one-fourth of their own money, they would spend 3 galleons in total. 
If Harry and Ron spent one-fifth of their own money, they would spend 24 galleons in total. 
Everyone has a whole number of galleons, and the total number of galleons between the three of them is a multiple of 7. 
Prove that the only possible number of galleons that Harry can have is 6.
-/
theorem harry_galleons (H He R : ℕ) (k : ℕ) :
    H + He = 12 →
    H + R = 120 → 
    H + He + R = 7 * k → 
    (H = 6) :=
by sorry

end harry_galleons_l730_730660


namespace parallelogram_area_proof_l730_730658

-- Define the parallelogram ABCD with given properties
variables {A B C D E F : Type} 

-- Assume a metric space (could be specific to a plane)
variable [MetricSpace {A B C D E F}]

-- Definitions of side lengths and angles
def side_AB (a b : ℝ) := a = 2
def angle_A (α : ℝ) := α = 45
def angle_AEB (β : ℝ) := β = 90
def angle_CFD (γ : ℝ) := γ = 90
def ratio_BF_BE (bf be : ℝ) := bf = 3 / 2 * be

-- Given conditions:
def conditions (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₁ : side_AB a b) 
  (h₂ : angle_A α) 
  (h₃ : angle_AEB β) 
  (h₄ : angle_CFD γ) 
  (h₅ : ratio_BF_BE b c) : Prop := 
  true

-- The resulting area to be proved
def area_parallelogram (a : ℝ) := a = 2 * real.sqrt 2

-- The theorem statement
theorem parallelogram_area_proof 
  {a b c d e f α β γ : ℝ} 
  (h₁ : side_AB a b) 
  (h₂ : angle_A α) 
  (h₃ : angle_AEB β) 
  (h₄ : angle_CFD γ) 
  (h₅ : ratio_BF_BE b c) : 
  area_parallelogram (2 * real.sqrt 2) :=
begin
  sorry
end

end parallelogram_area_proof_l730_730658


namespace calculate_angle_BDC_l730_730867
noncomputable theory

def given_angles (A E C : ℝ) := (A + E + C = 150)

theorem calculate_angle_BDC (A E C : ℝ) 
  (hA : A = 80) (hE : E = 30) (hC : C = 40) 
  (hABE : 180 - A - E = 70) 
  (hCBD : 180 - 70 = 110) : 
  180 - 110 - C = 30 :=
by {
  sorry
}

end calculate_angle_BDC_l730_730867


namespace max_value_expression_l730_730141

open Real

-- Assume c and d are positive real numbers
variables (c d : ℝ)
variables (hc : 0 < c) (hd : 0 < d)

-- Define the function to be maximized
def expression (y : ℝ) : ℝ :=
  3 * (c - y) * (y + sqrt (y ^ 2 + d ^ 2))

-- Prove the maximum value of the expression is 3 / 2 * (c ^ 2 + d ^ 2)
theorem max_value_expression : 
  (∃ y, y ≥ 0 ∧ ∀ z, z ≥ 0 → expression c d z ≤ 3 / 2 * (c ^ 2 + d ^ 2)) :=
sorry

end max_value_expression_l730_730141


namespace projection_onto_constant_vector_l730_730759

-- Define the problem in Lean 4
theorem projection_onto_constant_vector :
  ∃ (p : ℝ × ℝ),
    ∀ (a : ℝ),
      let v := (a, (3 / 4) * a - 2)
      in let w := (-3 / 4, 1)
         in 
           let proj_w_v := 
             let vw_dot := a * (-3 / 4) + (((3 / 4) * a - 2) * 1)
               in (vw_dot / ((-3 / 4) * (-3 / 4) + 1 * 1)) * (-3 / 4, 1)
            in proj_w_v = p :=
  -- The constant vector p is (24 / 25, -32 / 25)
  p = (24 / 25, -32 / 25)

end projection_onto_constant_vector_l730_730759


namespace june_first_day_of_week_l730_730360

def february_has_five_sundays (y : ℤ) : Prop :=
  ∃ (d : ℕ), d = 1 ∧ (∀ i ∈ range(5), (nweek(i * 7 + d, y) = sunday))

def day_of_week : ℤ -> string
| 0 := "Sunday"
| 1 := "Monday"
| 2 := "Tuesday"
| 3 := "Wednesday"
| 4 := "Thursday"
| 5 := "Friday"
| 6 := "Saturday"
| _ := "Unknown"

def nweek (n : ℤ) : ℤ := n % 7

theorem june_first_day_of_week (y : ℤ) :
  february_has_five_sundays y → day_of_week (nweek (92 + 1)) = "Tuesday" := 
by
  sorry  

end june_first_day_of_week_l730_730360


namespace sum_abs_a_l730_730431

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem sum_abs_a :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + 
   |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 67) :=
by
  sorry

end sum_abs_a_l730_730431


namespace steve_family_time_l730_730192

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l730_730192


namespace exists_li_ge_2018_l730_730523

noncomputable def polynomial_S (i : ℕ) (li : ℕ) : Polynomial ℝ :=
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 2018 * Polynomial.X + Polynomial.C li

theorem exists_li_ge_2018
  (n : ℕ) (h_n : 1 < n ∧ n < 2018)
  (l : Fin n → ℕ)
  (h_distinct : Function.Injective l)
  (h_pos : ∀ i, 0 < l i)
  (h_root : ∃ x : ℤ, ∑ i in Finset.univ, (polynomial_S i (l i)).eval x = 0) :
  ∃ i, 2018 ≤ l i := 
sorry

end exists_li_ge_2018_l730_730523


namespace sum_of_roots_of_f_poly_l730_730384

-- Define the polynomial as given in the conditions
def f_poly (x : ℝ) : ℝ :=
  (x-1)^2008 + 2*(x-2)^2007 + 3*(x-3)^2006 + 
  ∑ i in range 2004, (i+4)*(x-(i+4))^(2008 - i - 4) + 
  2007*(x-2007)^2 + 2008*(x-2008)

-- State the problem to prove the sum of the roots
theorem sum_of_roots_of_f_poly : 
  (∑ r in (roots (f_poly : polynomial ℝ)), r) = 2006 :=
by 
  sorry

end sum_of_roots_of_f_poly_l730_730384


namespace total_distance_walked_l730_730335

theorem total_distance_walked (bomin_km : ℕ) (bomin_m : ℕ) (yunshik_m : ℕ) (h1 : bomin_km = 2) (h2 : bomin_m = 600) (h3 : yunshik_m = 3700) :
  (bomin_km * 1000 + bomin_m + yunshik_m) = 6300 :=
by 
  have bomin_total_m := bomin_km * 1000 + bomin_m
  have total_distance := bomin_total_m + yunshik_m
  calc 
    bomin_total_m + yunshik_m = (2 * 1000 + 600) + 3700 : by rw [h1, h2, h3]
    ... = 6300 : by norm_num
  sorry

end total_distance_walked_l730_730335


namespace lin_cookies_per_batch_l730_730782

theorem lin_cookies_per_batch :
  let art_radius := 2
  let lin_side := 3
  let art_cookies := 10
  (art_cookies * (Real.pi * art_radius ^ 2)) / (lin_side ^ 2) ≈ 14 :=
by sorry

end lin_cookies_per_batch_l730_730782


namespace race_distance_l730_730873

theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end race_distance_l730_730873


namespace mike_baseball_cards_left_l730_730937

theorem mike_baseball_cards_left (initial_cards : ℕ) (packs_sold : ℕ) (cards_per_pack : ℕ) 
  (cards_sold : packs_sold * cards_per_pack = 36) 
  (initial_count : initial_cards = 150) 
  (packs_count : packs_sold = 3)
  (cards_in_pack : cards_per_pack = 12) : 
  initial_cards - cards_sold = 114 :=
by 
  rw [initial_count, packs_count, cards_in_pack, cards_sold]
  sorry

end mike_baseball_cards_left_l730_730937


namespace rafael_weekly_pay_correct_l730_730957

-- Conditions
def monday_hours : ℕ := 10
def tuesday_hours : ℕ := 8
def total_hours_week : ℕ := 40
def max_hours_per_day := 8
def regular_pay_per_hour := 20
def overtime_pay_per_hour := 30
def tax_rate := 0.10

-- Calculation
def rafael_total_pay_before_tax := 
  let monday_pay := 8 * regular_pay_per_hour + 2 * overtime_pay_per_hour
  let tuesday_pay := 8 * regular_pay_per_hour
  let wednesday_pay := 8 * regular_pay_per_hour
  let thursday_pay := 8 * regular_pay_per_hour
  let friday_pay := 6 * regular_pay_per_hour
  monday_pay + tuesday_pay + wednesday_pay + thursday_pay + friday_pay

def rafael_total_pay_after_tax := 
  rafael_total_pay_before_tax * (1 - tax_rate)

theorem rafael_weekly_pay_correct : 
  rafael_total_pay_after_tax = 738 := by
  sorry

end rafael_weekly_pay_correct_l730_730957


namespace greatest_a_inequality_l730_730742

theorem greatest_a_inequality :
  ∃ a : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) ∧
          (∀ b : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) → b ≤ a) ∧
          a = 2 / Real.sqrt 3 :=
sorry

end greatest_a_inequality_l730_730742


namespace moles_Cl2_combined_l730_730774

-- Condition Definitions
def moles_C2H6 := 2
def moles_HCl_formed := 2
def balanced_reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) : Prop :=
  C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Mathematical Equivalent Proof Problem Statement
theorem moles_Cl2_combined (C2H6 Cl2 HCl C2H4Cl2 : ℝ) (h1 : C2H6 = 2) 
(h2 : HCl = 2) (h3 : balanced_reaction C2H6 Cl2 C2H4Cl2 HCl) :
  Cl2 = 1 :=
by
  -- The proof is stated here.
  sorry

end moles_Cl2_combined_l730_730774


namespace closest_integer_to_cube_root_of_500_l730_730246

theorem closest_integer_to_cube_root_of_500 :
  ∃ n : ℤ, (∀ m : ℤ, |m^3 - 500| ≥ |8^3 - 500|) := 
sorry

end closest_integer_to_cube_root_of_500_l730_730246


namespace combinations_15_3_l730_730875

def num_combinations (n k : ℕ) : ℕ := n.choose k

theorem combinations_15_3 :
  num_combinations 15 3 = 455 :=
sorry

end combinations_15_3_l730_730875


namespace triangle_angles_l730_730500

/-- In triangle ABC, AP bisects ∠BAC and intersects BC at P; 
    BQ bisects ∠ABC and intersects CA at Q. 
    Given that ∠BAC = 60° and AB + BP = AQ + QB, 
    prove that the angles of triangle ABC are 60°, 80°, and 40°. -/
theorem triangle_angles (A B C P Q : Type)
  (hAP : bisects AP (angle A B C))
  (hBP : bisects BQ (angle B A C))
  (hBAC : angle A B = 60)
  (hAB_BP_AQ_QB : length AB + length BP = length AQ + length QB) :
  angle A B C = 60 ∧ angle B A C = 80 ∧ angle C A B = 40 :=
sorry

end triangle_angles_l730_730500


namespace stratified_sampling_total_results_l730_730674

theorem stratified_sampling_total_results :
  let junior_students := 400
  let senior_students := 200
  let total_students_to_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (Nat.choose junior_students junior_sample) * (Nat.choose senior_students senior_sample) = Nat.choose 400 40 * Nat.choose 200 20 :=
  sorry

end stratified_sampling_total_results_l730_730674


namespace compare_a_b_c_l730_730787

noncomputable def a : ℝ := log 0.6 0.5
noncomputable def b : ℝ := Real.log 0.5
noncomputable def c : ℝ := 0.6 ^ 0.5

theorem compare_a_b_c : a > c ∧ c > b := by
  sorry

end compare_a_b_c_l730_730787


namespace find_y_l730_730387

theorem find_y (y : ℕ) (hy : y ≤ 100) (hmean_mode : (42 + 74 + 87 + 3 * y) / 6 = 1.25 * y) : y = 45 :=
sorry

end find_y_l730_730387


namespace value_at_1971_l730_730949

def sequence_x (x : ℕ → ℝ) :=
  ∀ n > 1, 3 * x n - x (n - 1) = n

theorem value_at_1971 (x : ℕ → ℝ) (hx : sequence_x x) (h_initial : abs (x 1) < 1971) :
  abs (x 1971 - 985.25) < 0.000001 :=
by sorry

end value_at_1971_l730_730949


namespace robert_books_read_l730_730167

theorem robert_books_read (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 → book_pages = 360 → total_hours = 8 → (total_hours * pages_per_hour) / book_pages = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact (nat.div_eq_of_lt sorry)
end

end robert_books_read_l730_730167


namespace terminal_zeros_l730_730064

theorem terminal_zeros (a b : ℕ) (ha : a = 25) (hb : b = 240) :
  let prime_factorization := (5^2) * (2^4 * 3 * 5)
  number_of_terminal_zeros (prime_factorization : ℕ) = 3 :=
by
  sorry

end terminal_zeros_l730_730064


namespace range_of_f_l730_730204

def f (x : ℤ) : ℤ := x ^ 2 - 2 * x
def domain : Set ℤ := {0, 1, 2, 3}
def expectedRange : Set ℤ := {-1, 0, 3}

theorem range_of_f : (Set.image f domain) = expectedRange :=
  sorry

end range_of_f_l730_730204


namespace scheduling_classes_l730_730682

/-- Represents the number of different scheduling methods for 6 classes
    (Chinese, Mathematics, Politics, English, PE, Art) given that:
    - The Mathematics class must be scheduled within the first 3 periods, and
    - The PE class cannot be scheduled in the first period.
    The expected number of scheduling methods is 312. -/
theorem scheduling_classes : 
  ∃ f : ℕ, 
    (∀ a b c : ℕ, a ∈ {1, 2, 3} ∧ b = 6 → (a ≠ b) → (c ≠ 1)) → f = 312 :=
sorry

end scheduling_classes_l730_730682


namespace marking_ways_10x2_grid_l730_730633

-- Definitions based on conditions
def grid_10x2 := fin 10 × fin 2
def is_adjacent (a b : grid_10x2) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

def can_be_marked (marking : grid_10x2 → Prop) : Prop :=
  (∃! g : grid_10x2, ¬marking g) ∧  -- Exactly one square is not marked.
  (∀ g1 g2 : grid_10x2, marking g1 → marking g2 → ¬is_adjacent g1 g2)  -- No two marked squares are adjacent.

-- The proof problem statement
theorem marking_ways_10x2_grid : ∃ (num_ways : ℕ), num_ways = 36 ∧
  (∃ (marking : grid_10x2 → Prop), can_be_marked marking ∧ 
    ∃! num_ways = 36) :=
begin
  sorry
end

end marking_ways_10x2_grid_l730_730633


namespace area_of_rectangle_l730_730609

theorem area_of_rectangle (width length : ℝ) (h_width : width = 5.4) (h_length : length = 2.5) : width * length = 13.5 :=
by
  -- We are given that the width is 5.4 and the length is 2.5
  -- We need to show that the area (width * length) is 13.5
  sorry

end area_of_rectangle_l730_730609


namespace incorrect_statements_count_l730_730527

def floor_property_1 (x : ℝ) : Prop := ⌊x⌋ ≤ x ∧ x < ⌊x⌋ + 1
def floor_property_2 (x : ℝ) : Prop := x - 1 < ⌊x⌋ ∧ ⌊x⌋ ≤ x
def floor_property_3 (x : ℝ) : Prop := ⌊-x⌋ = -⌊x⌋
def floor_property_4 (x : ℝ) : Prop := ⌊2 * x⌋ = 2 * ⌊x⌋
def floor_property_5 (x : ℝ) : Prop := ⌊x⌋ + ⌊1 - x⌋ = 1

theorem incorrect_statements_count : (∃ x : ℝ, ¬ floor_property_1 x) ∧ 
                                    (∃ x : ℝ, ¬ floor_property_2 x) ∧ 
                                    (∃ x : ℝ, ¬ floor_property_3 x) ∧ 
                                    (∃ x : ℝ, ¬ floor_property_4 x) ∧ 
                                    (∃ x : ℝ, ¬ floor_property_5 x) → 
                                    3 := 
  sorry

end incorrect_statements_count_l730_730527


namespace salon_customers_l730_730293

variables (n : ℕ) (c : ℕ)

theorem salon_customers :
  ∀ (n = 33) (extra_cans = 5) (cans_per_customer = 2),
  (n - extra_cans) / cans_per_customer = 14 :=
begin
  sorry
end

end salon_customers_l730_730293


namespace calculate_expression_l730_730733

theorem calculate_expression : |(-5 : ℤ)| + (1 / 3 : ℝ)⁻¹ - (Real.pi - 2) ^ 0 = 7 := by
  sorry

end calculate_expression_l730_730733


namespace exist_twins_l730_730235

theorem exist_twins 
  (child_ages : list ℕ)
  (h_length : child_ages.length = 4)
  (all_born_on_Feb6 : ∀ n ∈ child_ages, n ≥ 2)
  (today_is_Feb_7_2016 : ∀ i, i ∈ child_ages → i ≤ 22)
  (oldest_eq_product_of_rest : 
    ∃ a1 a2 a3 a4, 
      a1 ∈ child_ages ∧ 
      a2 ∈ child_ages ∧ 
      a3 ∈ child_ages ∧ 
      a4 ∈ child_ages ∧ 
      a4 = a1 * a2 * a3) :
  ∃ (a : ℕ), 
    ∃ (x y : ℕ), 
      x ∈ child_ages ∧ 
      y ∈ child_ages ∧ 
      x = a ∧ y = a ∧ x ≠ y :=
begin
  sorry
end

end exist_twins_l730_730235


namespace cosine_product_24_l730_730180

noncomputable def angle_n (n : ℕ) : ℝ := 56 * 2^n
noncomputable def cosine_product (n : ℕ) : ℝ := ∏ i in Finset.range n, Real.cos (angle_n i)

theorem cosine_product_24 :
  cosine_product 24 = 1 / 2^24 :=
by
  sorry

end cosine_product_24_l730_730180


namespace closed_chain_of_61_gears_l730_730892

def isPossibleToFormClosedChain (n : Nat) (angleThreshold : ℝ) : Prop :=
  ∃ gears : Fin n → Circle,
    ∀ i : Fin n,
      let next := Fin.succ i
      | prevGears := gears i in
      | nextGears := gears next in
      meshed prevGears nextGears ∧
      angleBetweenMeshedGears prevGears nextGears ≥ angleThreshold

theorem closed_chain_of_61_gears : isPossibleToFormClosedChain 61 150 := by
  sorry

end closed_chain_of_61_gears_l730_730892


namespace cylinder_original_radius_inch_l730_730356

theorem cylinder_original_radius_inch (r : ℝ) :
  (∃ r : ℝ, (π * (r + 4)^2 * 3 = π * r^2 * 15) ∧ (r > 0)) →
  r = 1 + Real.sqrt 5 :=
by 
  sorry

end cylinder_original_radius_inch_l730_730356


namespace volume_maximized_ratio_l730_730239

noncomputable def max_volume_ratio (D : ℝ) : ℝ :=
  let x := (1 + Real.sqrt 17) / 2 -- The ratio that maximizes the volume
  in x

theorem volume_maximized_ratio (D a b : ℝ) (h : a^2 + (a + 2 * b)^2 = D^2) :
  (a / b) = max_volume_ratio D :=
sorry

end volume_maximized_ratio_l730_730239


namespace no_member_T_divisible_by_7_some_member_divisible_by_5_l730_730912

def is_member_of_T (x : ℤ) : Prop :=
  ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2

theorem no_member_T_divisible_by_7_some_member_divisible_by_5 :
  (∀ x ∈ T, ¬(x % 7 = 0)) ∧ (∃ x ∈ T, x % 5 = 0) :=  
by
  sorry

end no_member_T_divisible_by_7_some_member_divisible_by_5_l730_730912


namespace arithmetic_sequence_terms_l730_730803

theorem arithmetic_sequence_terms (a : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + 2)
  (h2 : ∃ k, ∀ n, a (2 * n + 1) + a (2 * n + 2) = a (2 * k + 1)) 
  (h3 : (∑ i in finset.range (2 * k + 1), if i % 2 = 0 then 0 else a i) = 15)
  (h4 : (∑ i in finset.range (2 * k + 1), if i % 2 = 1 then 0 else a i) = 25) : 
  2 * k + 1 = 10 := 
by 
  sorry

end arithmetic_sequence_terms_l730_730803


namespace cannot_obtain_xn_minus_1_l730_730502

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem cannot_obtain_xn_minus_1 :
  ∀ (n : ℕ), n > 0 → ¬∃ (h : ℝ → ℝ) (c : ℝ), h = (x^n - 1) where
    (h = f + g ∨ h = f - g ∨ h = f * g ∨ h = f ∘ g ∨ h = c • f) :=
begin
  sorry
end

end cannot_obtain_xn_minus_1_l730_730502


namespace no_poly_of_form_xn_minus_1_l730_730510

theorem no_poly_of_form_xn_minus_1 (f g : ℝ[X])
  (Hf : f = polynomial.X ^ 3 - 3 * polynomial.X ^ 2 + 5)
  (Hg : g = polynomial.X ^ 2 - 4 * polynomial.X)
  (allowed_operations : ∀ (h : ℝ[X]), h = 
    f + g ∨ h = f - g ∨ h = f * g ∨ h = polynomial.c g ∨ h = polynomial.c f ∨ h = polynomial.eval g f) :
  ¬ ∃ n : ℕ, n ≠ 0 ∧ (∃ h : ℝ[X], h = polynomial.X ^ n - 1) :=
by
  sorry

end no_poly_of_form_xn_minus_1_l730_730510


namespace math_problem_equivalent_l730_730392

section

variables {f : ℝ → ℝ} {g : ℝ → ℝ} {h : ℝ → ℝ} 
variables (m n a : ℝ)

-- Conditions
def f_def := f = λ x, x ^ 3 + m * x ^ 2 + n * x - 2
def point_condition := f (-1) = -6
def g_def := g = λ x, deriv f x + 6 * x
def symmetric_y := ∀ x, g x = g (-x)

-- Questions
def f_inc_intervals := (∀ x, (x < 0 ∨ x > 2) → deriv f x > 0)
def f_dec_intervals := (∀ x, (0 < x ∧ x < 2) → deriv f x < 0)
def h_dec_condition := (∀ x, (-1 < x ∧ x < 1) → deriv h x ≤ 0)

-- Proof statement
theorem math_problem_equivalent : 
  (f_def ∧ point_condition ∧ g_def ∧ symmetric_y) → 
  (m = -3 ∧ n = 0 ∧ f_inc_intervals ∧ f_dec_intervals ∧ (∀ x, (-1 < x ∧ x < 1) → a ≥ 3 * x^2 - 6 * x)) :=
by { sorry }

end

end math_problem_equivalent_l730_730392


namespace red_envelope_prob_correct_l730_730574

def prob_sum_not_less_than_4_yuan : ℚ :=
  let total_amount : ℚ := 10
  let portions : List ℚ := [1.49, 1.81, 2.19, 3.41, 0.62, 0.48]
  (portions.sum = total_amount) →
  let num_people : ℕ := 6
  let num_combinations : ℕ := num_people.choose 2
  let valid_combinations : List (ℚ × ℚ) := 
    [(0.62, 3.41), (1.49, 3.41), (1.81, 2.19), (1.81, 3.41), (2.19, 3.41)]
  (valid_combinations.length = 5) →
  let probability := (5 : ℚ) / (num_combinations : ℚ)
  probability

theorem red_envelope_prob_correct : 
  prob_sum_not_less_than_4_yuan = 1 / 3 :=
sorry

end red_envelope_prob_correct_l730_730574


namespace grid_sum_of_ones_l730_730924

theorem grid_sum_of_ones (m n : ℕ) (h_even : Even n)
  (f : ℕ → ℕ → ℤ)
  (h_values : ∀ i j, 1 ≤ f i j ∨ f i j = -1)
  (h_cols : ∀ j₁ j₂, ∑ i in Finset.range n, (f i j₁) * (f i j₂) ≤ 0) :
  (∑ i in Finset.range n, (∑ j in Finset.range m, if f i j = 1 then 1 else 0))
  ≤ (1 / 2 : ℚ) * n * (m + Real.sqrt m) := sorry

end grid_sum_of_ones_l730_730924


namespace pipe_length_difference_l730_730664

theorem pipe_length_difference (total_length shorter_piece : ℕ) (h1 : total_length = 68) (h2 : shorter_piece = 28) : 
  total_length - shorter_piece * 2 = 12 := 
sorry

end pipe_length_difference_l730_730664


namespace candy_distribution_l730_730176

-- Defining the conditions of the problem
def total_candies : ℕ := 53
def bags : fin 3 → ℕ  -- three bags have a different number of candies

-- Ensuring each bag has a different number of candies
def distinct (a b c : ℕ) : Prop := (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

-- Ensuring any two bags together have more candies than the third one
def valid_distribution (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Main theorem
theorem candy_distribution : 
  ∃ (A B C : ℕ), 
  distinct A B C ∧ 
  valid_distribution A B C ∧ 
  A + B + C = total_candies ∧ 
  finset.card 
   (finset.filter (λ b, (valid_distribution b.1 b.2 (total_candies - b.1 - b.2)) 
                   ∧ distinct b.1 b.2 (total_candies - b.1 - b.2)) 
     ((finset.finpairs 1 (total_candies - 1)).product 
      (finset.finpairs 1 (total_candies - 1)))) 
  = 52 := 
sorry

end candy_distribution_l730_730176


namespace total_students_l730_730339

theorem total_students (n : ℕ) (total : ℕ) (middle : fin (n + 2))
  (from_left : middle = ⟨5, by decide⟩) (from_right : n - middle.1 = 6) : total = 3 * n := by
  sorry

end total_students_l730_730339


namespace max_markers_with_20_dollars_l730_730934

theorem max_markers_with_20_dollars (single_marker_cost : ℕ) (four_pack_cost : ℕ) (eight_pack_cost : ℕ) :
  single_marker_cost = 2 → four_pack_cost = 6 → eight_pack_cost = 10 → (∃ n, n = 16) := by
    intros h1 h2 h3
    existsi 16
    sorry

end max_markers_with_20_dollars_l730_730934


namespace seats_per_section_correct_l730_730224

-- Define the total number of seats
def total_seats : ℕ := 270

-- Define the number of sections
def sections : ℕ := 9

-- Define the number of seats per section
def seats_per_section (total_seats sections : ℕ) : ℕ := total_seats / sections

theorem seats_per_section_correct : seats_per_section total_seats sections = 30 := by
  sorry

end seats_per_section_correct_l730_730224


namespace focal_distances_equal_l730_730780

variable (x y t : ℝ)

-- Define the two hyperbolas
def hyperbola1 : Prop := (x^2 / 16) - (y^2 / 9) = 1
def hyperbola2 : Prop := (x^2 / (16 - t)) - (y^2 / (t + 9)) = 1

-- Conditions
def condition_t : Prop := -9 < t ∧ t < 16

-- Focal distance calculation for hyperbolas
def focal_distance1 : ℝ := 2 * Real.sqrt ((4 : ℝ)^2 + (3 : ℝ)^2)
def focal_distance2 := 2 * Real.sqrt ((16 - t) + (t + 9))

-- The proof problem
theorem focal_distances_equal (h1 : hyperbola1) (h2 : hyperbola2) (ht : condition_t) :
    focal_distance1 = focal_distance2 :=
by
  sorry

end focal_distances_equal_l730_730780


namespace angle_between_a_b_45_degrees_l730_730855

noncomputable def unit_vector_angle_90 (e1 e2 : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (∥e1∥ = 1) ∧ (∥e2∥ = 1) ∧ (inner e1 e2 = 0)

noncomputable def vector_a (e1 e2 : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  2 • e1 + e2

noncomputable def vector_b (e1 e2 : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  3 • e1 - e2

theorem angle_between_a_b_45_degrees (e1 e2 : EuclideanSpace ℝ (Fin 2)) 
  (h : unit_vector_angle_90 e1 e2) : 
  real.angle (vector_a e1 e2) (vector_b e1 e2) = real.pi / 4 :=
sorry

end angle_between_a_b_45_degrees_l730_730855


namespace sixth_term_is_constant_value_of_n_coefficient_of_x2_rational_terms_in_expansion_l730_730805

noncomputable def general_term (n r : ℕ) (x : ℝ) : ℝ :=
  (real.pow (-1/2 : ℝ) r) * (nat.choose n r : ℝ) * real.pow x ((n:ℝ - 2 * r) / 3)

theorem sixth_term_is_constant (n : ℕ) :
  general_term n 5 1 = 1 :=
by 
    sorry

theorem value_of_n :
  ∃ n, general_term n 5 1 = 1 ∧ n = 10 :=
by
    sorry

theorem coefficient_of_x2 :
  general_term 10 2 = (45 / 4 : ℝ) * real.pow x 2 :=
by
    sorry

theorem rational_terms_in_expansion :
  ∃ (c1 c2 c3 : ℝ), 
    general_term 10 2 = c1 * real.pow x 2 ∧ 
    general_term 10 5 = c2 ∧ 
    general_term 10 8 = c3 * real.pow x (-2) ∧ 
    c1 = 45 / 4 ∧ 
    c2 = -63 / 8 ∧ 
    c3 = 45 / 256 :=
by
    sorry

end sixth_term_is_constant_value_of_n_coefficient_of_x2_rational_terms_in_expansion_l730_730805


namespace robert_books_read_l730_730166

theorem robert_books_read (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 → book_pages = 360 → total_hours = 8 → (total_hours * pages_per_hour) / book_pages = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact (nat.div_eq_of_lt sorry)
end

end robert_books_read_l730_730166


namespace find_angle_A_find_triangle_area_l730_730890

theorem find_angle_A (b : ℝ) (A B : ℝ) (h : b + b * Real.cos A = Real.sqrt 3 * Real.sin B) : 
  A = π / 3 :=
sorry

theorem find_triangle_area (a b c A : ℝ) (hA : A = π / 3) (ha : a = Real.sqrt 21) 
  (hb : b = 4) (hcos := λ b c, b * Real.cos A = Real.sqrt 3 * Real.sin B) 
  (hcoslaw : a^2 = b^2 + c^2 - 2*b*c*Real.cos A) : 
  ∃ area : ℝ, area = 5 * Real.sqrt 3 :=
sorry

end find_angle_A_find_triangle_area_l730_730890


namespace pipe_length_difference_l730_730666

theorem pipe_length_difference 
  (total_length shorter_piece : ℝ)
  (h1: total_length = 68) 
  (h2: shorter_piece = 28) : 
  ∃ longer_piece diff : ℝ, longer_piece = total_length - shorter_piece ∧ diff = longer_piece - shorter_piece ∧ diff = 12 :=
by
  sorry

end pipe_length_difference_l730_730666


namespace evaluate_x_squared_minus_y_squared_l730_730032

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l730_730032


namespace three_digit_factorial_sum_l730_730245

theorem three_digit_factorial_sum : ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ (∃ a b c : ℕ, x = 100 * a + 10 * b + c ∧ (a = 0 ∨ b = 0 ∨ c = 0) ∧ x = a.factorial + b.factorial + c.factorial) := 
sorry

end three_digit_factorial_sum_l730_730245


namespace smallest_circle_equation_correct_l730_730687

noncomputable def smallest_circle_through_intersections : Prop :=
  let line := λ x y : ℝ, 2 * x - y + 3 = 0
  let circle := λ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0
  ∃ (x1 y1 x2 y2 : ℝ), 
    line x1 y1 ∧ circle x1 y1 ∧
    line x2 y2 ∧ circle x2 y2 ∧ 
    ∀ (x y : ℝ),
      (x + (3 / 5))^2 + (y - (9 / 5))^2 = (19 / 5)
      ↔ 5 * x^2 + 5 * y^2 + 6 * x - 18 * y - 1 = 0

theorem smallest_circle_equation_correct :
  smallest_circle_through_intersections :=
sorry

end smallest_circle_equation_correct_l730_730687


namespace tan_A_eq_one_l730_730111

noncomputable def trigonometric_condition (A : ℝ) : Prop :=
  (3.0^.5 * Real.cos A + Real.sin A) / (3.0^.5 * Real.sin A - Real.cos A) = Real.tan (-7 * Real.pi / 12)

theorem tan_A_eq_one (A : ℝ) (h1 : trigonometric_condition A) : Real.tan A = 1 := sorry

end tan_A_eq_one_l730_730111


namespace range_of_m_l730_730860

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), ¬((x - m) < -3) ∧ (1 + 2*x)/3 ≥ x - 1) ∧ 
  (∀ (x1 x2 x3 : ℤ), 
    (¬((x1 - m) < -3) ∧ (1 + 2 * x1)/3 ≥ x1 - 1) ∧
    (¬((x2 - m) < -3) ∧ (1 + 2 * x2)/3 ≥ x2 - 1) ∧
    (¬((x3 - m) < -3) ∧ (1 + 2 * x3)/3 ≥ x3 - 1)) →
  (4 ≤ m ∧ m < 5) :=
by 
  sorry

end range_of_m_l730_730860


namespace solution_set_of_log_inequality_l730_730532

noncomputable def f (a x : ℝ) := a * x ^ 2 + x + 1

theorem solution_set_of_log_inequality (a x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∃ x, f a x = has_max f) :
  log a (x - 1) > 0 ↔ 1 < x ∧ x < 2 :=
sorry

end solution_set_of_log_inequality_l730_730532


namespace analytical_expression_of_f_range_of_k_l730_730207

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then (-3 * x) / (x + 2) else (3 * x) / (x - 2)

theorem analytical_expression_of_f :
  ∀ x : ℝ, f x = if x ≥ 0 then (-3 * x) / (x + 2) else (3 * x) / (x - 2) :=
by intro x; exact if_neg (sorry)

theorem range_of_k (k : ℝ) :
  (∀ t : ℝ, f (2 * k - t^2) + f (2 * t - 2 * t^2 - 3) > 0) ↔  k < 4 / 3 :=
by exact sorry

end analytical_expression_of_f_range_of_k_l730_730207


namespace speed_difference_l730_730304

-- Definitions from the conditions
def distance : ℝ := 6         -- both travel 6 miles
def naomi_time : ℝ := 10 / 60 -- Naomi travels in 10 minutes (converted to hours)
def naomi_speed : ℝ := distance / naomi_time -- Naomi's speed
def maya_speed : ℝ := 12      -- Maya's speed in miles per hour

-- The statement to prove
theorem speed_difference : naomi_speed - maya_speed = 24 :=
by
  sorry

end speed_difference_l730_730304


namespace find_g_find_m_range_l730_730437

noncomputable def f (a : ℝ) (x : ℝ) [fact (1 < a)] := real.log a (x + 1) 

noncomputable def g (a : ℝ) (x : ℝ) [fact (1 < a)] := -real.log a (1 - x)

theorem find_g (a : ℝ) [fact (1 < a)] {x : ℝ} (hx : x < 1) :
  g a x = -real.log a (1 - x) :=
by sorry

theorem find_m_range (a : ℝ) [fact (1 < a)] {x : ℝ} (hx : 0 ≤ x ∧ x < 1) (m : ℝ) :
  f a x + g a x ≥ m ↔ m ∈ Iic 0 :=
by sorry

end find_g_find_m_range_l730_730437


namespace point_D_eq_1_2_l730_730422

-- Definitions and conditions
def point : Type := ℝ × ℝ

def A : point := (-1, 4)
def B : point := (-4, -1)
def C : point := (4, 7)

-- Translate function
def translate (p : point) (dx dy : ℝ) := (p.1 + dx, p.2 + dy)

-- The translation distances found from A to C
def dx := C.1 - A.1
def dy := C.2 - A.2

-- The point D
def D : point := translate B dx dy

-- Proof objective
theorem point_D_eq_1_2 : D = (1, 2) := by
  sorry

end point_D_eq_1_2_l730_730422


namespace count_multiples_of_15_between_15_and_180_l730_730463

theorem count_multiples_of_15_between_15_and_180: (set.Icc 15 180).filter (λ n, n % 15 = 0).card = 12 := 
by
  sorry

end count_multiples_of_15_between_15_and_180_l730_730463


namespace juice_packs_in_box_l730_730994

theorem juice_packs_in_box 
  (W_box L_box H_box W_juice_pack L_juice_pack H_juice_pack : ℕ)
  (hW_box : W_box = 24) (hL_box : L_box = 15) (hH_box : H_box = 28)
  (hW_juice_pack : W_juice_pack = 4) (hL_juice_pack : L_juice_pack = 5) (hH_juice_pack : H_juice_pack = 7) : 
  (W_box * L_box * H_box) / (W_juice_pack * L_juice_pack * H_juice_pack) = 72 :=
by
  sorry

end juice_packs_in_box_l730_730994


namespace parallel_iff_dot_product_condition_l730_730061

-- Define the condition for parallel vectors
def are_parallel (a b : Vector ℝ 3) : Prop :=
  ∃ k : ℝ, a = k • b ∨ b = k • a

-- Define the condition |a • b| = |a||b|
def dot_product_condition (a b : Vector ℝ 3) : Prop :=
  |a • b| = ‖a‖ * ‖b‖

-- The proposition to be proven:
theorem parallel_iff_dot_product_condition (a b : Vector ℝ 3) :
  dot_product_condition a b ↔ are_parallel a b :=
by
  sorry

end parallel_iff_dot_product_condition_l730_730061


namespace smallest_number_divisible_by_6_with_perfect_square_product_l730_730290

-- Definition of a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Definition of divisibility by 6
def is_divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

-- Definition of the product of digits being a perfect square
def digits_product (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = m

-- The problem statement as a Lean theorem
theorem smallest_number_divisible_by_6_with_perfect_square_product : ∃ (n : ℕ), is_two_digit n ∧ is_divisible_by_6 n ∧ is_perfect_square (digits_product n) ∧ ∀ (m : ℕ), (is_two_digit m ∧ is_divisible_by_6 m ∧ is_perfect_square (digits_product m)) → n ≤ m :=
by
  sorry

end smallest_number_divisible_by_6_with_perfect_square_product_l730_730290


namespace curve_is_circle_and_line_l730_730579

noncomputable def is_combination_of_circle_and_line (ρ θ : ℝ) : Prop :=
  ρ^2 * cos θ - 3 * ρ * cos θ + ρ - 3 = 0

theorem curve_is_circle_and_line (ρ θ : ℝ) :
  is_combination_of_circle_and_line ρ θ → 
  ∃ c l : GeometryEntity, (c.is_circle ∧ l.is_line ∧ curve.is_combination_of c l) :=
sorry

end curve_is_circle_and_line_l730_730579


namespace river_road_cars_l730_730216

theorem river_road_cars
  (B C : ℕ)
  (h1 : B * 17 = C)
  (h2 : C = B + 80) :
  C = 85 := by
  sorry

end river_road_cars_l730_730216


namespace correct_judgments_count_l730_730438

noncomputable def f (m n x : ℝ) : ℝ := m^x - log n x

theorem correct_judgments_count (m n a b c d : ℝ) (h_m_lt_1 : 0 < m) (h_m_lt_n : m < 1)
  (h_n_gt_1 : 1 < n) (h_a_gt_b : a > b) (h_b_gt_c : b > c) (h_c_gt_0 : c > 0)
  (h_f_prod_lt_0 : f m n a * f m n b * f m n c < 0) (h_d_root : f m n d = 0) :
  (d > 1) ∧ (d < a) ∧ (d > b) ∧ (d < b) ∧ (d > c) :=
sorry

end correct_judgments_count_l730_730438


namespace trigonometric_identity_l730_730402

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l730_730402


namespace avg_salary_rest_of_workers_l730_730202

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_tech : ℝ) (total_workers : ℕ)
  (total_avg_salary : avg_all = 8000) (tech_avg_salary : avg_tech = 12000) (workers_count : total_workers = 30) :
  (20 * (total_workers * avg_all - 10 * avg_tech) / 20) = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l730_730202


namespace corn_height_growth_l730_730725

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l730_730725


namespace total_distinct_pairs_subcommittees_l730_730205

variable (members : Finset ℕ) 
variable (teachers : Finset ℕ)

def is_subcommittee (s : Finset ℕ) := s.card = 4
def has_teacher (s : Finset ℕ) := ∃ t ∈ s, t ∈ teachers

theorem total_distinct_pairs_subcommittees :
  members.card = 12 →
  teachers.card = 5 →
  (∀ (s t : Finset ℕ), s ≠ t → Disjoint s t) →
  (∀ s : Finset ℕ, is_subcommittee members s → has_teacher s) →
  ∃ pairs : Finset (Finset (Finset ℕ)),
  pairs.card = 29900 :=
by
  intros
  sorry

end total_distinct_pairs_subcommittees_l730_730205


namespace sum_a2_a4_a6_l730_730095

-- Define the arithmetic sequence with a positive common difference
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ (d : ℝ), d > 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define that a_1 and a_7 are roots of the quadratic equation x^2 - 10x + 16 = 0
def roots_condition (a : ℕ → ℝ) : Prop :=
(a 1) * (a 7) = 16 ∧ (a 1) + (a 7) = 10

-- The main theorem we want to prove
theorem sum_a2_a4_a6 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : roots_condition a) :
  a 2 + a 4 + a 6 = 15 :=
sorry

end sum_a2_a4_a6_l730_730095


namespace triangle_is_obtuse_l730_730818

-- Define the problem conditions
variables (α : ℝ) (h_internal_angle : 0 < α ∧ α < π)
variable (h_sum_sin_cos : sin α + cos α = 2/3)

-- Define the theorem statement
theorem triangle_is_obtuse : ∀ α, (0 < α ∧ α < π) → (sin α + cos α = 2/3) → (π/2 < α ∧ α < π) :=
begin
  assume α,
  assume h_internal_angle,
  assume h_sum_sin_cos,
  sorry -- proof to be filled in
end

end triangle_is_obtuse_l730_730818


namespace point_on_circle_l730_730409

-- Define the points A and M and radius r.
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 2, y := -3 }
def M : Point := { x := 5, y := -7 }

def radius : ℝ := 5

-- Define the function to compute the distance between two points.
def distance (P1 P2 : Point) : ℝ :=
  real.sqrt ((P2.x - P1.x)^2 + (P2.y - P1.y)^2)

-- Define the theorem for the positional relationship between point M and circle O.
theorem point_on_circle :
  distance A M = radius := by
  sorry

end point_on_circle_l730_730409


namespace minimum_k_minus_b_l730_730428

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + Real.log x

noncomputable def tangent_line (x k b : ℝ) : Prop :=
  ∃ x0 : ℝ, f x0 = k * x0 + b ∧ f' x0 = k

theorem minimum_k_minus_b (x k b : ℝ) :
  (∃ x0 : ℝ, tangent_line x0 k b) → 
  ∃ k b : ℝ, k - b = 7 / 2 := by
  sorry

end minimum_k_minus_b_l730_730428


namespace interest_rate_is_correct_l730_730771

variable (A P I : ℝ)
variable (T R : ℝ)

theorem interest_rate_is_correct
  (hA : A = 1232)
  (hP : P = 1100)
  (hT : T = 12 / 5)
  (hI : I = A - P) :
  R = I * 100 / (P * T) :=
by
  sorry

end interest_rate_is_correct_l730_730771


namespace hyperbola_eccentricity_l730_730741

open Real

-- Definitions
variables (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
noncomputable def hyperbola := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

-- Given Conditions
variable (h_product_slopes : (b^2 / a^2 = 7 / 9))

-- Statement to prove
theorem hyperbola_eccentricity : 
  ∀ (a b : ℝ), 0 < a → 0 < b → (b^2 / a^2 = 7 / 9) → 
  let e := sqrt (1 + b^2 / a^2) in e = 4 / 3 := sorry

end hyperbola_eccentricity_l730_730741


namespace find_k_l730_730258

-- Define the points and slope
def point1 : ℝ × ℝ := (-1, -4)
def point2 (k : ℝ) : ℝ × ℝ := (4, k)
def slope (k : ℝ) : ℝ := k

-- The main theorem statement
theorem find_k : ∃ k : ℝ, slope k = k ∧ 
  (point2 k).2 = k ∧
  let x1 := point1.1 in
  let y1 := point1.2 in
  let x2 := (point2 k).1 in
  let y2 := (point2 k).2 in
  slope k = (y2 - y1) / (x2 - x1) :=
sorry

end find_k_l730_730258


namespace frog_jump_vertical_side_prob_l730_730283

-- Definitions of the conditions
def square_side_prob {x y : ℕ} (p : ℕ × ℕ → ℚ) := 
  p (0, y) + p (4, y)

-- Main statement
theorem frog_jump_vertical_side_prob :
  ∀ (p : ℕ × ℕ → ℚ), 
    let start: ℕ × ℕ := (1, 2) in
    (∀ y, 0 ≤ y ∧ y ≤ 4 → p (0, y) = 1) → 
    (∀ y, 0 ≤ y ∧ y ≤ 4 → p (4, y) = 1) → 
    (∀ x, 0 ≤ x ∧ x ≤ 4 → p (x, 0) = 0) → 
    (∀ x, 0 ≤ x ∧ x ≤ 4 → p (x, 4) = 0) → 
    square_side_prob p = 5 / 8 :=
by
  intros
  sorry

end frog_jump_vertical_side_prob_l730_730283


namespace John_other_trip_length_l730_730119

theorem John_other_trip_length :
  ∀ (fuel_per_km total_fuel first_trip_length other_trip_length : ℕ),
    fuel_per_km = 5 →
    total_fuel = 250 →
    first_trip_length = 20 →
    total_fuel / fuel_per_km - first_trip_length = other_trip_length →
    other_trip_length = 30 :=
by
  intros fuel_per_km total_fuel first_trip_length other_trip_length h1 h2 h3 h4
  sorry

end John_other_trip_length_l730_730119


namespace solve_for_a_l730_730933

open Complex

theorem solve_for_a : ∀ (a : ℝ), (let z := (1+2*I) * (a+I) in (z.re = z.im)) → a = -3 := by
  intro a h
  have z_exp : (1 + 2 * I) * (a + I) = a - 2 + (2 * a + 1) * I := by
    simp [I_sq, Complex.mul_re, Complex.mul_im]
  rw z_exp at h
  -- Continuing the proof steps by step:
  have real_eq_im : (a - 2) = (2 * a + 1) := by
    simp at h
    exact h
  have : a = -3 := by
    linarith
  exact this

end solve_for_a_l730_730933


namespace find_area_ABC_find_length_AC_given_angle_ABC_l730_730319

-- Given conditions in Lean definitions

-- Define constant areas for the given triangles
def area_APK : ℝ := 15
def area_CPK : ℝ := 13

-- Define the angles
def angle_B := Real.arctan (4 / 7)

-- Theorem statements for proof problems

-- Part (a): Finding the area of triangle ABC
theorem find_area_ABC {A B C O P T K : Type} [IsTriangle A B C] [Circumcircle O A C] 
  (in_omega : Inscribed A C ω) (circle_AOC : Circle O A C) (intersects_BC_at_P : IntersectsAt P B C) 
  (tangent_AT_TC : Tangent A T ω) (tangent_CT_TC : Tangent C T ω) 
  (intersections_TP_AC : IntersectsAt K T A C) 
  (area_APK_eq_15 : area_APK) (area_CPK_eq_13 : area_CPK) :
  area_ABC = 784 / 13 :=
  sorry

-- Part (b): Finding the length of AC given angle ABC
theorem find_length_AC_given_angle_ABC {A B C O P T K : Type} [IsTriangle A B C] 
  [Circumcircle O A C] (in_omega : Inscribed A C ω) (circle_AOC : Circle O A C) 
  (intersects_BC_at_P : IntersectsAt P B C) (tangent_AT_TC : Tangent A T ω) 
  (tangent_CT_TC : Tangent C T ω) (intersections_TP_AC : IntersectsAt K T A C) 
  (area_APK_eq_15 : area_APK) (area_CPK_eq_13 : area_CPK) (angle_ABC_arctan : angle_B) :
  length_AC = 14 / Real.sqrt 3 :=
  sorry

end find_area_ABC_find_length_AC_given_angle_ABC_l730_730319


namespace green_vs_blue_tile_difference_l730_730511

-- Initial condition definitions
def initialBlueTiles : ℕ := 20
def initialGreenTiles : ℕ := 10
def firstLayerBorderGreenTiles : ℕ := 12
def secondLayerBorderGreenTiles : ℕ := 18

-- The proposition to prove
theorem green_vs_blue_tile_difference :
  let totalGreenTiles := initialGreenTiles + firstLayerBorderGreenTiles + secondLayerBorderGreenTiles
  let totalBlueTiles := initialBlueTiles
  totalGreenTiles - totalBlueTiles = 20 := 
by 
  let totalGreenTiles := 10 + 12 + 18
  let totalBlueTiles := 20
  show totalGreenTiles - totalBlueTiles = 20 from sorry

end green_vs_blue_tile_difference_l730_730511


namespace problem_statement_l730_730441

noncomputable def f (x : ℝ) := 1 / (3^x + 1)

def log_base2_3 : ℝ := Real.log 3 / Real.log 2
def log_base4_inv9 : ℝ := Real.log (1/9) / Real.log 4

theorem problem_statement : 
  f (log_base2_3) + f (log_base4_inv9) = 1 := 
by
  sorry

end problem_statement_l730_730441


namespace average_monthly_growth_rate_price_reduction_for_desired_profit_l730_730676

theorem average_monthly_growth_rate
  (initial_sales: ℕ) (final_sales: ℕ): 
  initial_sales = 256 →
  final_sales = 400 →
  let x := ((final_sales/initial_sales: ℚ) ^ (1/2) - 1) in
  (x * 100) = 25 := sorry

def profit_from_price_reduction
  (initial_price: ℕ) (cost_price: ℕ) (initial_sales: ℕ) (desired_profit: ℕ) (price_reduction: ℕ): Prop
  := ((initial_price - cost_price - price_reduction) * (initial_sales + 5 * price_reduction) = desired_profit)

theorem price_reduction_for_desired_profit 
  (initial_price: ℕ) (cost_price: ℕ) (initial_sales: ℕ) (desired_profit: ℕ) (y: ℕ):
  initial_price = 40 →
  cost_price = 25 →
  initial_sales = 400 →
  desired_profit = 4250 →
  let y := 5 in
  profit_from_price_reduction initial_price cost_price initial_sales desired_profit y := 
  sorry

end average_monthly_growth_rate_price_reduction_for_desired_profit_l730_730676


namespace determine_M_l730_730987

theorem determine_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 := 
sorry

end determine_M_l730_730987


namespace swap_colors_possible_l730_730802

theorem swap_colors_possible (n : ℕ) (initial_checkerboard : Array (Array Color))
    (corner_black : initial_checkerboard[0][0] = Color.black ∨ initial_checkerboard[n-1][0] = Color.black ∨ initial_checkerboard[0][n-1] = Color.black ∨ initial_checkerboard[n-1][n-1] = Color.black)
    (is_checkerboard : ∀ i j, (i + j) % 2 = 0 → initial_checkerboard[i][j] = Color.black ∧ (i + j) % 2 = 1 → initial_checkerboard[i][j] = Color.white) : 
    (∃ k : ℕ, n = 3 * k) ↔ (∀ i j, checkerboard_color_after_moves i j = initial_checkerboard[i][j] ∧ (i % 2 = 0 ∧ j % 2 = 0) ? Color.white : Color.black) ∨ ((i % 2 = 1 ∧ j % 2 = 1) ? Color.black : Color.white) :=
sorry

-- Define required data structures and types
inductive Color
| black
| white
| green

noncomputable def checkerboard_color_after_moves : Array (Array Color) := sorry

end swap_colors_possible_l730_730802


namespace problem_proof_l730_730963

noncomputable def verify_polar_equation_C1 : Prop :=
  ∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4 → 
  ∃ (ρ θ : ℝ), ρ = 4 * Real.cos θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

noncomputable def verify_cartesian_equation_line_l : Prop :=
  ∀ (ρ θ : ℝ) (a : ℝ), 
    ∃ (x y : ℝ), 
      ρ * Real.cos (θ - Real.pi / 4) = a → 
      a = 3 * Real.sqrt 2 → 
      x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧
      x + y = 6

noncomputable def verify_translated_line_and_length : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ = 4 * Real.cos θ →
    (∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4 → x + y = 0) →
    θ = 3 * Real.pi / 4 → 
    (∃ (Μ Ν : ℝ), Μ = -2 * Real.sqrt 2 ∧ Ν = 2 * Real.sqrt 2 ∧ 
    abs (Ν - Μ) = 4 * Real.sqrt 2)

theorem problem_proof :
  verify_polar_equation_C1 ∧ verify_cartesian_equation_line_l ∧ verify_translated_line_and_length :=
by
  split
  sorry
  split
  sorry
  sorry

end problem_proof_l730_730963


namespace main_theorem_l730_730430

noncomputable def polar_eq_to_rect (ρ θ : ℝ) : Prop :=
  ρ - 2 * Real.sin θ = 0

noncomputable def rect_eq (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

noncomputable def parametric_line_eq (t : ℝ) : ℝ × ℝ :=
  (1 - t * (1 / 2), t * (Real.sqrt 3 / 2))

noncomputable def distance_sum_eq (t1 t2 : ℝ) : ℝ :=
  Real.abs t1 + Real.abs t2

theorem main_theorem 
  (ρ θ t1 t2 : ℝ) 
  (point_M : ℝ × ℝ := (1, 0)) 
  (line_slope_angle : ℝ := 2 * Real.pi / 3) :
  polar_eq_to_rect ρ θ → -- condition (polar equation)
  (rect_eq (ρ * Real.cos θ) (ρ * Real.sin θ)) → -- conversion to rectangular
  (parametric_line_eq t1 = (ρ * Real.cos θ, ρ * Real.sin θ)) → -- parametric for point A
  (parametric_line_eq t2 = (ρ * Real.cos θ, ρ * Real.sin θ)) → -- parametric for point B
  (distance_sum_eq t1 t2 = √3 + 1) := 
by sorry

end main_theorem_l730_730430


namespace seth_sold_78_candy_bars_l730_730562

def num_candy_sold_by_seth (num_candy_max: Nat): Nat :=
  3 * num_candy_max + 6

theorem seth_sold_78_candy_bars :
  num_candy_sold_by_seth 24 = 78 :=
by
  unfold num_candy_sold_by_seth
  simp
  rfl

end seth_sold_78_candy_bars_l730_730562


namespace calculation_l730_730885

def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x^3 + 2 * x^2 - 3 * x)

def p (f : ℝ → ℝ) : ℕ := 1 -- Number of holes
def q (f : ℝ → ℝ) : ℕ := 2 -- Number of vertical asymptotes
def r (f : ℝ → ℝ) : ℕ := 1 -- Number of horizontal asymptotes
def s (f : ℝ → ℝ) : ℕ := 0 -- Number of oblique asymptotes

theorem calculation (f : ℝ → ℝ) (hp : p f = 1) (hq : q f = 2) (hr : r f = 1) (hs : s f = 0) : 
p f + 2 * q f + 3 * r f + 4 * s f = 8 := by
  sorry

end calculation_l730_730885


namespace range_of_a_l730_730543

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ (a ≤ 1) :=
by
  sorry

end range_of_a_l730_730543


namespace tangent_line_eq_l730_730472

theorem tangent_line_eq (
  x y : ℝ,
  h_curve : y = x^4,
  h_perpendicular : 4 * x - y = 3,
  h_line : ∀ (x₀ : ℝ), (4 * (x₀ : ℝ)^3 = 4)
) : 4 * x - y - 3 = 0 :=
sorry

end tangent_line_eq_l730_730472


namespace sin_A_value_l730_730087

theorem sin_A_value
  (a b A B : ℝ)
  (h1 : a = real.sqrt 6)
  (h2 : b = 4)
  (h3 : B = 2 * A) :
  real.sin A = real.sqrt 3 / 3 :=
sorry

end sin_A_value_l730_730087


namespace probability_at_least_one_prize_proof_l730_730607

noncomputable def probability_at_least_one_wins_prize
  (total_tickets : ℕ) (prize_tickets : ℕ) (people : ℕ) :
  ℚ :=
1 - ((@Nat.choose (total_tickets - prize_tickets) people) /
      (@Nat.choose total_tickets people))

theorem probability_at_least_one_prize_proof :
  probability_at_least_one_wins_prize 10 3 5 = 11 / 12 :=
by
  sorry

end probability_at_least_one_prize_proof_l730_730607


namespace valid_lists_count_12_l730_730909

def valid_list (n : ℕ) (l : List ℕ) : Prop :=
  (∀ (k : ℕ), k < n → l.get? k ≠ none) ∧
  (∀ (i : ℕ), 1 < i ∧ i < n →
    (∃ j, j < i ∧ ((l.get? j = some (l.get! i + 2)) ∨ (l.get? j = some (l.get! i - 2)))))

def count_valid_lists_of_length (n : ℕ) : ℕ :=
  if n = 3 then 3
  else 2 * count_valid_lists_of_length (n - 2)

theorem valid_lists_count_12 :
  count_valid_lists_of_length 12 = 96 :=
sorry

end valid_lists_count_12_l730_730909


namespace no_integer_k_with_Pk_eq_3_l730_730923

theorem no_integer_k_with_Pk_eq_3 
  (P : polynomial ℤ) 
  (a b c : ℤ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_Pa2 : P.eval a = 2) 
  (h_Pb2 : P.eval b = 2) 
  (h_Pc2 : P.eval c = 2) :
  ¬ ∃ k : ℤ, P.eval k = 3 :=
sorry

end no_integer_k_with_Pk_eq_3_l730_730923


namespace concurrency_of_AX_BY_CZ_l730_730539

open EuclideanGeometry

variables {A B C : Point} [triangle_ABC : NonEquilateralTriangle A B C]
variables (Ma Mb Mc : Point)

/-- Define the midpoints Ma, Mb, Mc of sides BC, CA, AB respectively. -/
def isMidpoint (Ma B C : Point) := Midpoint B C Ma
def isMidpoint (Mb C A : Point) := Midpoint C A Mb
def isMidpoint (Mc A B : Point) := Midpoint A B Mc

variable (S : Point)

/-- S is a point on the Euler line of triangle ABC. -/
def isOnEulerLine (S : Point) := OnEulerLine S A B C

variables (Xa Ya Za : Point)

/-- Define X, Y, Z as the second intersections of MaS, MbS, McS with the nine-point circle. -/
def isSecondIntersectionWithNinePointCircle (M S X : Point) :=
  SecondIntersectionWithNinePointCircle M S X

/-- Prove that AX, BY and CZ are concurrent. -/
theorem concurrency_of_AX_BY_CZ 
  (hMa : isMidpoint Ma B C)
  (hMb : isMidpoint Mb C A)
  (hMc : isMidpoint Mc A B)
  (hEuler : isOnEulerLine S)
  (hXa : isSecondIntersectionWithNinePointCircle Ma S Xa)
  (hYa : isSecondIntersectionWithNinePointCircle Mb S Ya)
  (hZa : isSecondIntersectionWithNinePointCircle Mc S Za) :
  Concurrent [Line.through A Xa, Line.through B Ya, Line.through C Za] :=
sorry

end concurrency_of_AX_BY_CZ_l730_730539


namespace correct_operation_l730_730248

variable (a b : ℝ)

theorem correct_operation : 
  ¬ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧
  ¬ ((a^3) ^ 2 = a ^ 5) ∧
  (a ^ 5 / a ^ 3 = a ^ 2) ∧
  ¬ (a ^ 3 + a ^ 2 = a ^ 5) :=
by
  sorry

end correct_operation_l730_730248


namespace abc_order_l730_730131

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := 0.5^3
noncomputable def c : Real := Real.log 3 / Real.log 0.5 -- log_0.5 3 is written as (log 3) / (log 0.5) in Lean

theorem abc_order : a > b ∧ b > c :=
by
  have h1 : a = Real.sqrt 3 := rfl
  have h2 : b = 0.5^3 := rfl
  have h3 : c = Real.log 3 / Real.log 0.5 := rfl
  sorry

end abc_order_l730_730131


namespace Jan_drove_210_more_l730_730253

variables (d_I s_I s_H s_J t : ℕ)

-- Conditions
def Han_speed_condition := s_H = s_I + 10
def Han_distance_condition := d_I + 90 = s_H * t
def Jan_speed_condition := s_J = s_I + 15
def Jan_time_condition := t_J = t + 3

-- Definitions
def d_H := s_H * t
def d_J := s_J * (t + 3)

-- Question
theorem Jan_drove_210_more : d_J - d_I = 210 :=
by
  rw [Han_speed_condition, Jan_speed_condition, Han_distance_condition, Jan_time_condition]
  rw t_J,
  sorry

end Jan_drove_210_more_l730_730253


namespace wuyang_football_school_l730_730968

def is_approx_250 (n: ℕ) := abs (n - 250) <= 10

theorem wuyang_football_school (x : ℕ) (h1 : (x - 4) % 2 = 0) (h2 : (x - 5) % 3 = 0) (h3 : x % 5 = 0) (h4 : ∃ k : ℕ, k^2 = (x / 5)) (h5 : is_approx_250 (x - 3)) : x = 260 :=
by
  sorry

end wuyang_football_school_l730_730968


namespace kittens_total_number_l730_730999

theorem kittens_total_number (W L H R : ℕ) (k : ℕ) 
  (h1 : W = 500) 
  (h2 : L = 80) 
  (h3 : H = 200) 
  (h4 : L + H + R = W) 
  (h5 : 40 * k ≤ R) 
  (h6 : R ≤ 50 * k) 
  (h7 : ∀ m, m ≠ 4 → m ≠ 6 → m ≠ k →
        40 * m ≤ R → R ≤ 50 * m → False) : 
  k = 5 ∧ 2 + 4 + k = 11 := 
by {
  -- The proof would go here
  sorry 
}

end kittens_total_number_l730_730999


namespace relation_among_three_numbers_l730_730600

theorem relation_among_three_numbers :
  7^0.3 > 1 ∧ 0.3^7 > 0 ∧ 0.3^7 < 1 ∧ ln 0.3 < 0 →
  7^0.3 > 0.3^7 ∧ 0.3^7 > ln 0.3 := by
  intros h
  sorry

end relation_among_three_numbers_l730_730600


namespace abs_neg_three_l730_730198

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l730_730198


namespace distinct_integer_sums_l730_730750

-- Define the conditions.
def x_values : List ℝ := List.replicate 1002 (ℝ.sqrt 2 - 1) ++ List.replicate 1002 (ℝ.sqrt 2 + 1)

noncomputable def sum_x_pairs (x : List ℝ) : ℝ := 
  (List.finRange 1002).sum (λ k, x.get! (2*k) * x.get! (2*k + 1))

-- Statement of the proof problem.
theorem distinct_integer_sums : 
  ∃ N : ℕ, N = 502 ∧ ∀ x : List ℝ, 
    (∀ i ∈ x, i = ℝ.sqrt 2 - 1 ∨ i = ℝ.sqrt 2 + 1) → 
    (x.length = 2004) → 
    (finset.univ.image (λ (k : ℕ), sum_x_pairs x)).card = N :=
begin
  sorry
end

end distinct_integer_sums_l730_730750


namespace negation_equiv_l730_730973

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 + 1 ≥ 1

-- Negation of the original proposition
def negated_prop : Prop := ∃ x : ℝ, x^2 + 1 < 1

-- Main theorem stating the equivalence
theorem negation_equiv :
  (¬ (∀ x : ℝ, original_prop x)) ↔ negated_prop :=
by sorry

end negation_equiv_l730_730973


namespace white_balls_count_l730_730097

theorem white_balls_count {T W : ℕ} (h1 : 3 * 4 = T) (h2 : T - 3 = W) : W = 9 :=
by 
    sorry

end white_balls_count_l730_730097


namespace wallpaper_removal_time_l730_730757

theorem wallpaper_removal_time (time_per_wall : ℕ) (dining_room_walls_remaining : ℕ) (living_room_walls : ℕ) :
  time_per_wall = 2 → dining_room_walls_remaining = 3 → living_room_walls = 4 → 
  time_per_wall * (dining_room_walls_remaining + living_room_walls) = 14 :=
by
  sorry

end wallpaper_removal_time_l730_730757


namespace perimeter_of_larger_triangle_is_65_l730_730324

noncomputable def similar_triangle_perimeter : ℝ :=
  let a := 7
  let b := 7
  let c := 12
  let longest_side_similar := 30
  let perimeter_small := a + b + c
  let ratio := longest_side_similar / c
  ratio * perimeter_small

theorem perimeter_of_larger_triangle_is_65 :
  similar_triangle_perimeter = 65 := by
  sorry

end perimeter_of_larger_triangle_is_65_l730_730324


namespace eval_expression_l730_730359

theorem eval_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end eval_expression_l730_730359


namespace C_gets_more_than_D_by_500_l730_730299

-- Definitions based on conditions
def proportionA := 5
def proportionB := 2
def proportionC := 4
def proportionD := 3

def totalProportion := proportionA + proportionB + proportionC + proportionD

def A_share := 2500
def totalMoney := A_share * (totalProportion / proportionA)

def C_share := (proportionC / totalProportion) * totalMoney
def D_share := (proportionD / totalProportion) * totalMoney

-- The theorem stating the final question
theorem C_gets_more_than_D_by_500 : C_share - D_share = 500 := by
  sorry

end C_gets_more_than_D_by_500_l730_730299


namespace problem1_problem2_l730_730808

open Real

-- Given points A and B, and a moving point P such that |PA| = 2|PB|, prove the equation of the curve C
theorem problem1 (x y : ℝ) :
  let A := (-3 : ℝ, 0 : ℝ)
  let B := (3 : ℝ, 0 : ℝ)
  let PA := sqrt ((x + 3)^2 + y^2)
  let PB := sqrt ((x - 3)^2 + y^2)
  |PA| = 2 * |PB| →
  (x - 5)^2 + y^2 = 16 :=
by sorry

-- Given line l1, point Q on l1, and curve C, prove the equation of the line QM that minimizes |QM|
theorem problem2 (Q : ℝ × ℝ) (x y : ℝ) :
  let l1 := λ x y : ℝ, x + y + 3
  let C := (x - 5)^2 + y^2 = 16
  Q.1 + Q.2 + 3 = 0 →
  (∃ (M : ℝ × ℝ), M ∈ C ∧ ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ C → |<M-Q>| ≥ |<x₀, y₀-Q>|) →
  (x = 1 ∨ y = -4) :=
by sorry

end problem1_problem2_l730_730808


namespace expansion_coefficients_l730_730421

theorem expansion_coefficients (n : ℕ) (h : n = 7) :
  let sum_of_coeffs := (1 + 2 * (1 : ℝ)^(1/3))^7,
      sum_of_binomial_coeffs := 2^7,
      rational_terms := [1, 560 * (1 : ℝ), 448 * (1 : ℝ)^2, 2016 * (1 : ℝ)^3] 
  in 
  sum_of_coeffs = 2187 ∧ 
  sum_of_binomial_coeffs = 128 ∧ 
  rational_terms = [1, 560, 448, 2016] := 
by 
  sorry

end expansion_coefficients_l730_730421


namespace rhombus_construction_possible_l730_730346

-- Definitions for points, lines, and distances
variables {Point : Type} {Line : Type}
def is_parallel (l1 l2 : Line) : Prop := sorry
def distance_between (l1 l2 : Line) : ℝ := sorry
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Given parallel lines l₁ and l₂ and their distance a
variable {l1 l2 : Line}
variable (a : ℝ)
axiom parallel_lines : is_parallel l1 l2
axiom distance_eq_a : distance_between l1 l2 = a

-- Given points A and B
variable (A B : Point)

-- Definition of a rhombus that meets the criteria
noncomputable def construct_rhombus (A B : Point) (l1 l2 : Line) (a : ℝ) : Prop :=
  ∃ C1 C2 D1 D2 : Point, 
    point_on_line C1 l1 ∧ 
    point_on_line D1 l2 ∧ 
    point_on_line C2 l1 ∧ 
    point_on_line D2 l2 ∧ 
    sorry -- additional conditions ensuring sides passing through A and B and forming a rhombus

theorem rhombus_construction_possible : 
  construct_rhombus A B l1 l2 a :=
sorry

end rhombus_construction_possible_l730_730346


namespace squirrels_in_tree_l730_730608

-- Definitions based on the conditions
def nuts : Nat := 2
def squirrels : Nat := nuts + 2

-- Theorem stating the main proof problem
theorem squirrels_in_tree : squirrels = 4 := by
  -- Proof steps would go here, but we're adding sorry to skip them
  sorry

end squirrels_in_tree_l730_730608


namespace volume_of_remaining_solid_after_removing_tetrahedra_l730_730943

theorem volume_of_remaining_solid_after_removing_tetrahedra :
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  cube_volume - 8 * tetrahedron_volume = 5 / 6 := by
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  have h : cube_volume - 8 * tetrahedron_volume = 5 / 6 := sorry
  exact h

end volume_of_remaining_solid_after_removing_tetrahedra_l730_730943


namespace smallest_positive_angle_correct_l730_730340

noncomputable def smallest_positive_angle (x : ℝ) : Prop := 
  tan (6 * x) = (sin x - cos x) / (sin x + cos x)

theorem smallest_positive_angle_correct : ∃ x : ℝ, smallest_positive_angle x ∧ 0 < x ∧ x = 7.5 :=
begin
  sorry
end

end smallest_positive_angle_correct_l730_730340


namespace probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730643

open ProbabilityTheory

noncomputable def Vanya_probability_more_heads_than_Tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  X_V 3 = binomial 3 (1/2) ∧
  X_T 2 = binomial 2 (1/2) ∧
  Pr(X_V > X_T) = 1/2

noncomputable def Vanya_probability_more_heads_than_Tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  ∀ n : ℕ,
  X_V (n+1) = binomial (n+1) (1/2) ∧
  X_T n = binomial n (1/2) ∧
  Pr(X_V > X_T) = 1/2

theorem probability_heads_vanya_tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_a X_V X_T := sorry

theorem probability_heads_vanya_tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_b X_V X_T := sorry

end probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l730_730643


namespace cost_price_of_book_l730_730076

theorem cost_price_of_book
  (C : ℝ)
  (h : 1.09 * C - 0.91 * C = 9) :
  C = 50 :=
sorry

end cost_price_of_book_l730_730076


namespace john_money_left_l730_730115

-- Definitions for initial conditions
def initial_amount : ℤ := 100
def cost_roast : ℤ := 17
def cost_vegetables : ℤ := 11

-- Total spent calculation
def total_spent : ℤ := cost_roast + cost_vegetables

-- Remaining money calculation
def remaining_money : ℤ := initial_amount - total_spent

-- Theorem stating that John has €72 left
theorem john_money_left : remaining_money = 72 := by
  sorry

end john_money_left_l730_730115


namespace pine_cones_on_roof_l730_730309

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end pine_cones_on_roof_l730_730309


namespace find_k_range_l730_730407

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x - (⌊x⌋ : ℝ) else f (x - 1)

theorem find_k_range (k : ℝ) (h : k > 0) :
  (∀ x, f x = k * x + k → f x ≠ f (x + 0.5) ∧ f x ≠ f (x + 1) ∧ f x ≠ f (x + 1.5)) ↔
  (1/4 < k ∧ k < 1/3) :=
sorry

end find_k_range_l730_730407


namespace max_length_of_permissible_word_l730_730199

def is_permissible (alphabet : Type) (word : List alphabet) : Prop :=
  ∀ i j : ℕ, i < word.length → j < word.length → 
  (word[i] ≠ word[i+1] ∧ 
   ¬∃ a b : alphabet, a ≠ b ∧ List.is_subsequence_of [a, b, a, b] word)

noncomputable def max_length_permissible (n : ℕ) (alphabet : Type) : ℕ :=
  2 * n - 1

theorem max_length_of_permissible_word 
  (n : ℕ) (alphabet : Fin n → Type) : 
  ∃ word : List (Fin n), is_permissible (Fin n) word ∧ 
  word.length = max_length_permissible n (Fin n) :=
sorry

end max_length_of_permissible_word_l730_730199


namespace det_proj_matrix_zero_l730_730128

theorem det_proj_matrix_zero :
  let u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-5]]
  let uuT : Matrix (Fin 2) (Fin 2) ℝ := u.mul u.transpose
  let uTu : ℝ := (u.transpose).mul u
  let Q : Matrix (Fin 2) (Fin 2) ℝ := (1 / uTu) • uuT
  det Q = 0 :=
by
  let u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-5]]
  let uuT : Matrix (Fin 2) (Fin 2) ℝ := u.mul u.transpose
  let uTu : ℝ := (u.transpose).mul u
  let Q : Matrix (Fin 2) (Fin 2) ℝ := (1 / uTu) • uuT
  have Q_def : Q = (1 / 34) • uuT := by sorry
  have det_Q : det Q = 0 := by sorry
  exact det_Q

end det_proj_matrix_zero_l730_730128


namespace find_f_when_x_lt_1_l730_730785

noncomputable def f : ℝ → ℝ := sorry

axiom symm_property : ∀ x : ℝ, f(1 + x) = f(1 - x)
axiom value_for_x_geq_1 : ∀ x : ℝ, x ≥ 1 → f(x) = Real.log(x + 1)

theorem find_f_when_x_lt_1 (x : ℝ) (h : x < 1) : f(x) = Real.log (3 - x) := by
  sorry

end find_f_when_x_lt_1_l730_730785


namespace bc_ao_dot_product_l730_730147

noncomputable def vector_dot_product := sorry /- Define vector dot product function if not pre-defined in Mathlib -/

variables {A B C O : Type} [metric_space A B C O]

variables (circumcenter : Π (a b c : A), A)
variables (AB AC BC AO : ℝ)

theorem bc_ao_dot_product (h₀ : circumcenter A B C = O)
  (h₁ : AB = 13)
  (h₂ : AC = 12)
  : vector_dot_product BC AO = -25 / 2 := sorry

end bc_ao_dot_product_l730_730147


namespace union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l730_730838

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Statement 1: Prove that when \( m = 3 \), \( A \cup B \) = \( \{ x \mid -3 \leq x \leq 5 \} \).
theorem union_of_A_and_B_at_m_equals_3 : set_A ∪ set_B 3 = { x | -3 ≤ x ∧ x ≤ 5 } :=
sorry

-- Statement 2: Prove that if \( A ∪ B = A \), then the range of \( m \) is \( (-\infty, \frac{5}{2}] \).
theorem range_of_m_if_A_union_B_equals_A (m : ℝ) : (set_A ∪ set_B m = set_A) → m ≤ 5 / 2 :=
sorry

end union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l730_730838


namespace find_quadruple_l730_730362

theorem find_quadruple :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a^3 + b^4 + c^5 = d^11 ∧ a * b * c < 10^5 :=
sorry

end find_quadruple_l730_730362


namespace even_marked_squares_9x9_l730_730408

open Nat

theorem even_marked_squares_9x9 :
  let n := 9
  let total_squares := n * n
  let odd_rows_columns := [1, 3, 5, 7, 9]
  let odd_squares := odd_rows_columns.length * odd_rows_columns.length
  total_squares - odd_squares = 56 :=
by
  sorry

end even_marked_squares_9x9_l730_730408


namespace lambda_plus_x_value_l730_730153

def vector (α : Type) := α × α

section

variables (a b c : vector ℝ) (λ x : ℝ)

-- Definition of the vectors
def a : vector ℝ := (1, 2)
def b : vector ℝ := (-3, 5)
def c : vector ℝ := (4, x)

-- Condition: a + b = λ * c
def condition : Prop := (1 - 3, 2 + 5) = (4 * λ, λ * x)

-- Statement to prove: λ + x equals -29/2
theorem lambda_plus_x_value : condition → λ + x = -29 / 2 :=
by
  sorry -- Proof to be filled in

end lambda_plus_x_value_l730_730153


namespace total_cards_across_decks_l730_730519

-- Conditions
def DeckA_cards : ℕ := 52
def DeckB_cards : ℕ := 40
def DeckC_cards : ℕ := 50
def DeckD_cards : ℕ := 48

-- Question as a statement
theorem total_cards_across_decks : (DeckA_cards + DeckB_cards + DeckC_cards + DeckD_cards = 190) := by
  sorry

end total_cards_across_decks_l730_730519


namespace line_plane_intersection_l730_730371

theorem line_plane_intersection :
  ∃ (x y z : ℝ), (∃ t : ℝ, x = -3 + 2 * t ∧ y = 1 + 3 * t ∧ z = 1 + 5 * t) ∧ (2 * x + 3 * y + 7 * z - 52 = 0) ∧ (x = -1) ∧ (y = 4) ∧ (z = 6) :=
sorry

end line_plane_intersection_l730_730371


namespace surface_area_of_solid_block_l730_730361

theorem surface_area_of_solid_block :
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  top_bottom_area + front_back_area + left_right_area = 66 :=
by
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  sorry

end surface_area_of_solid_block_l730_730361


namespace coin_flips_prob_l730_730651

open Probability

noncomputable def probability_Vanya_more_heads_than_Tanya (n : ℕ) : ℝ :=
  P(X Vanya > X Tanya)

theorem coin_flips_prob (n : ℕ) :
   P(Vanya gets more heads than Tanya | Vanya flips (n+1) times, Tanya flips n times) = 0.5 :=
sorry

end coin_flips_prob_l730_730651


namespace find_additional_student_number_l730_730612

def classSize : ℕ := 52
def sampleSize : ℕ := 4
def sampledNumbers : List ℕ := [5, 31, 44]
def additionalStudentNumber : ℕ := 18

theorem find_additional_student_number (classSize sampleSize : ℕ) 
    (sampledNumbers : List ℕ) : additionalStudentNumber ∈ (5 :: 31 :: 44 :: []) →
    (sampledNumbers = [5, 31, 44]) →
    (additionalStudentNumber = 18) := by
  sorry

end find_additional_student_number_l730_730612


namespace sum_count_in_cube_l730_730497

-- Define the structure and conditions
def vertex_values := Fin 8 → ℤ 
def face_value (vs : vertex_values) (f : List (Fin 8)) : ℤ := f.map vs |> List.prod

noncomputable def sum_cube_values (vs : vertex_values) (faces : List (List (Fin 8))) : ℤ := 
  (List.range 8).map vs |> List.sum + faces.map (face_value vs) |> List.sum

-- Define the possible sums set and their product
def possible_sums_set := {14, 10, 6, 2, -2, -6, -10}
def possible_sums_product := -20160

-- Statement to be proved
theorem sum_count_in_cube :
  ∃ vs : vertex_values, ∃ faces : List (List (Fin 8)), 
    sum_cube_values vs faces ∈ possible_sums_set ∧ possible_sums_set.prod == possible_sums_product :=
sorry

end sum_count_in_cube_l730_730497


namespace cylinder_surface_area_l730_730420

theorem cylinder_surface_area (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 2 * r) : 
  2 * Real.pi * r^2 + 2 * Real.pi * r * l = 24 * Real.pi :=
by
  subst h1
  subst h2
  sorry

end cylinder_surface_area_l730_730420


namespace sample_stddev_is_2_l730_730177

open Finset

def masses : Finset ℝ := {125, 124, 121, 123, 127}

def mean (s : Finset ℝ) : ℝ :=
  s.sum id / s.card

def stddev (s : Finset ℝ) : ℝ :=
  let m := mean s
  let squared_deviations := s.map (λ x, (x - m) * (x - m))
  real.sqrt (squared_deviations.sum id / (s.card - 1))

theorem sample_stddev_is_2 : stddev masses = 2 :=
  sorry

end sample_stddev_is_2_l730_730177


namespace find_a_b_l730_730582

-- Define the function and its derivative
def f (a b x : ℝ) : ℝ := a * x - b / x
def f' (a b x : ℝ) : ℝ := a + b / (x * x)

-- Given the point where the tangent is evaluated
def point_x : ℝ := 2
def tangent_line_slope : ℝ := 7 / 4
def tangent_line_y_int : ℝ := (7 * point_x - 12) / 4

-- Given system of equations
def eq1 (a b : ℝ) : Prop := a + b / 4 = 7 / 4
def eq2 (a b : ℝ) : Prop := 2 * a - b / 2 = 1 / 2

theorem find_a_b (a b : ℝ) (h1 : eq1 a b) (h2 : eq2 a b) : a = 1 ∧ b = 3 :=
by
  sorry

end find_a_b_l730_730582


namespace corn_height_growth_l730_730723

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l730_730723


namespace max_distance_center_line_l730_730049

theorem max_distance_center_line (m : ℝ) :
  let l : ℝ → ℝ → Prop := λ x y, m * x + (5 - 2 * m) * y - 2 = 0
  let inside (x y : ℝ) : Prop := x^2 + y^2 = 4
  -- Let d be the distance from the origin (0, 0) to the line l.
  let d (l : ℝ → ℝ → Prop) : ℝ := real.dist (5 * ∥(5 - 2 * m) / (m^2 + (5 - 2 * m)^2)^2∥) 0
  -- Prove that the maximum distance is given when d = 2√5/5.
  max_distance_center_line := d l = 2 * real.sqrt 5 / 5 :=
sorry

end max_distance_center_line_l730_730049


namespace sequence_formula_l730_730797

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 2^n - 1 :=
by
  sorry

end sequence_formula_l730_730797


namespace ratio_percentages_l730_730976

theorem ratio_percentages (A B : ℕ) (h : A / B = 5 / 8) :
  ((B - A) / B * 100 = 37.5) ∧ ((B - A) / A * 100 = 60) :=
by
  -- Proof will be constructed here
  sorry

end ratio_percentages_l730_730976


namespace bc_value_l730_730578

noncomputable def curve := { p : ℝ × ℝ // p.1 * p.2 = 1 }
def line_y_eq_2x := { p : ℝ × ℝ // p.2 = 2 * p.1 }
def reflected_curve (p : curve) : ℝ × ℝ := sorry -- Reflection transformation details

theorem bc_value :
  ∃ b c d : ℝ, (∀ (x y : ℝ), 
    ∃ (u v : ℝ), (u * v = 1) ∧ 
    x = (reflected_curve ⟨(u, v), _⟩).1 ∧ 
    y = (reflected_curve ⟨(u, v), _⟩).2 ∧ 
    12 * x^2 + b * x * y + c * y^2 + d = 0) ∧ 
  (b * c = 84) := sorry

end bc_value_l730_730578


namespace projection_magnitude_is_l730_730077

def Vector (α : Type u) := α × α

variables (a b : Vector ℝ)
variables (theta : ℝ)
variables (abs_a abs_b : ℝ)
variables (angle_ab in_degrees: ℝ)

-- Defining the given conditions
def given_conditions : Prop :=
  angle_ab = 30 ∧ abs_a = 4 ∧ abs_b = 2

-- Define the magnitude of the projection vector
def proj_vector_magnitude (a b : Vector ℝ) (abs_a angle_ab: ℝ) : ℝ :=
  abs_a * Real.cos (angle_ab.toRadians)

noncomputable def toRadians (x : ℝ) : ℝ := x * Real.pi / 180

#eval (toRadians 30 : ℝ) -- Converting degrees to radians for consistent use with cosine function

-- The statement to be proven
theorem projection_magnitude_is (h : given_conditions) : proj_vector_magnitude a b abs_a angle_ab = 2 * Real.sqrt 3 :=
  sorry

end projection_magnitude_is_l730_730077


namespace polygon_sides_l730_730985

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : 
  n = 8 := 
sorry

end polygon_sides_l730_730985


namespace monotonic_intervals_a_eq_1_minimum_value_a_real_roots_of_equation_l730_730829

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + a / x

theorem monotonic_intervals_a_eq_1 :
  (∀ x, 0 < x → (f x 1 > f x 1 → 1 < x) ∧ (f x 1 < f x 1 → 0 < x ∧ x < 1))
    := sorry

theorem minimum_value_a (x0 : ℝ) (y0 : ℝ) :
  f' x0 := (1 / x0 - a / x0^2)
  (x0 > 0) → ((x0 - a) / x0^2 ≤ 1 / 2) → a ≥ -(1 / 2) * x0^2 + x0 − 1 / 2
    := sorry

theorem real_roots_of_equation (a b x : ℝ) (h : 0 < x) :
  let g := λ x, x^3 + 2 * (b * x + a) in
  (f x a = (g x) / (2 * x) - 1 / 2) →
  (b < 0 ∧ ∃ x1 x2, x1 ≠ x2 ∧ f x1 a = (g x1) / (2 * x1) - 1 / 2 ∧ f x2 a = (g x2) / (2 * x2) - 1 / 2) ∧
  (b = 0 ∧ ∃ x1, f x1 a = (g x1) / (2 * x1) - 1 / 2) ∧
  (b > 0 ∧ ∀ x, f x a ≠ (g x) / (2 * x) - 1 / 2)
    := sorry

end monotonic_intervals_a_eq_1_minimum_value_a_real_roots_of_equation_l730_730829


namespace ellipse_equation_l730_730429

theorem ellipse_equation (a b c : ℝ) :
  (2 * a = 10) ∧ (c / a = 4 / 5) →
  ((x:ℝ)^2 / 25 + (y:ℝ)^2 / 9 = 1) ∨ ((x:ℝ)^2 / 9 + (y:ℝ)^2 / 25 = 1) :=
by
  sorry

end ellipse_equation_l730_730429


namespace cos_alpha_val_trigonometric_identity_l730_730417

variable (α : ℝ)
variable (sin_alpha : ℝ)
variable (cos_alpha : ℝ)
hypothesis sin_alpha_def : sin α = sqrt 5 / 5
hypothesis quadrant_one : 0 < α ∧ α < π / 2

theorem cos_alpha_val (sin_alpha_def : sin α = sqrt 5 / 5) (quadrant_one : 0 < α ∧ α < π / 2) : 
  cos α = 2 * sqrt 5 / 5 :=
sorry

theorem trigonometric_identity (sin_alpha_def : sin α = sqrt 5 / 5) (quadrant_one : 0 < α ∧ α < π / 2) :
  (tan (α + π) + (sin (3 * π / 2 - α) / cos (π - α))) = 3 / 2 :=
sorry

end cos_alpha_val_trigonometric_identity_l730_730417


namespace new_shoes_cost_proof_l730_730274

-- Define the conditions
def repair_cost : ℝ := 10.50
def new_shoes_lifetime : ℝ := 2
def cost_increase_factor : ℝ := 1 + 0.42857142857142854

-- Define the cost of new shoes
noncomputable def new_shoes_cost : ℝ := 30

-- The theorem to prove the cost of new shoes
theorem new_shoes_cost_proof : new_shoes_cost = 
  let new_shoes_cost_per_year := new_shoes_cost / new_shoes_lifetime in
  let expected_cost_per_year := repair_cost * cost_increase_factor in
  if new_shoes_cost_per_year = expected_cost_per_year then new_shoes_cost else 0 :=
sorry

end new_shoes_cost_proof_l730_730274


namespace arithmetic_expression_l730_730337

theorem arithmetic_expression : 125 - 25 * 4 = 25 := 
by
  sorry

end arithmetic_expression_l730_730337


namespace analytical_expression_and_minimum_cost_l730_730670

noncomputable def total_items := 30
noncomputable def basketball_cost := 80
noncomputable def soccerball_cost := 60
noncomputable def soccerball_discount := 0.8
noncomputable def discounted_soccerball_cost := soccerball_cost * soccerball_discount

theorem analytical_expression_and_minimum_cost (m : ℕ) (h : 0 < m ∧ m < total_items ∧ m ≥ 2 * (total_items - m)) :
  ∃ w : ℕ, w = 32 * m + 1440 ∧ (m = 20 → w = 2080) :=
by {
  sorry
}

end analytical_expression_and_minimum_cost_l730_730670


namespace triangle_ABC_is_equilateral_l730_730887

-- Define the problem settings and conditions
variables {A B C K L M : Type}
variables [triangle A B C]
variables (is_angle_bisector A K B C : Type) (is_median B L A C : Type) (is_altitude C M A B : Type)
variables (is_equilateral_triangle K L M : Type)

-- The theorem we want to prove
theorem triangle_ABC_is_equilateral 
  (h1 : is_angle_bisector A K B C) 
  (h2 : is_median B L A C) 
  (h3 : is_altitude C M A B) 
  (h4 : is_equilateral_triangle K L M) :  
  is_equilateral_triangle A B C :=
sorry

end triangle_ABC_is_equilateral_l730_730887


namespace company_spends_less_l730_730277

noncomputable def total_spending_reduction_in_dollars : ℝ :=
  let magazine_initial_cost := 840.00
  let online_resources_initial_cost_gbp := 960.00
  let exchange_rate := 1.40
  let mag_cut_percentage := 0.30
  let online_cut_percentage := 0.20

  let magazine_cost_cut := magazine_initial_cost * mag_cut_percentage
  let online_resource_cost_cut_gbp := online_resources_initial_cost_gbp * online_cut_percentage
  
  let new_magazine_cost := magazine_initial_cost - magazine_cost_cut
  let new_online_resource_cost_gbp := online_resources_initial_cost_gbp - online_resource_cost_cut_gbp

  let online_resources_initial_cost := online_resources_initial_cost_gbp * exchange_rate
  let new_online_resource_cost := new_online_resource_cost_gbp * exchange_rate

  let mag_cut_amount := magazine_initial_cost - new_magazine_cost
  let online_cut_amount := online_resources_initial_cost - new_online_resource_cost
  
  mag_cut_amount + online_cut_amount

theorem company_spends_less : total_spending_reduction_in_dollars = 520.80 :=
by
  sorry

end company_spends_less_l730_730277


namespace irrational_has_inf_rationals_approx_l730_730950

theorem irrational_has_inf_rationals_approx (ξ : ℝ) (h_irrational : irrational ξ) : ∃ᶠ (m n : ℕ) in at_top, ∃ (hn : n ≠ 0), ∃ (hm : m ∈ ℤ), (| ξ - (m / n) | < 1 / (sqrt 5 * n^2)) :=
by
  sorry

end irrational_has_inf_rationals_approx_l730_730950


namespace find_angle_B_find_area_triangle_l730_730889

-- Define the problem's conditions
variables {A B C a b c : ℝ}
hypothesis sum_angles : A + B + C = Real.pi
hypothesis altitude_ratio : ∃ k : ℝ, k = (3 / 5) ∧ ∀ x y : ℝ, x = a ∧ y = c → x = k * y
hypothesis side_b : b = 7
hypothesis cosB_sin_halfAC : Real.cos B + Real.sin ((A + C) / 2) = 0

-- Part 1: Proving the value of angle B
theorem find_angle_B : B = (2 * Real.pi) / 3 :=
by
  sorry

-- Part 2: Proving the area of triangle ABC
theorem find_area_triangle (ha_ratio : altitude_ratio) :
  let c_val := sqrt ((Real.pow b 2) * (25 / 49)) 
  area = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end find_angle_B_find_area_triangle_l730_730889


namespace length_of_solution_set_l730_730103

noncomputable def greatest_int_not_greater (x : ℝ) : ℤ := floor x

noncomputable def fractional_part (x : ℝ) : ℝ := x - greatest_int_not_greater x

noncomputable def f (x : ℝ) : ℝ := (greatest_int_not_greater x : ℝ) * fractional_part x

noncomputable def g (x : ℝ) : ℝ := x - 1

theorem length_of_solution_set : ∀ (a b : ℝ), 
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x < g x → x ∈ set.Icc a b) →
  b - a = 1 :=
by 
  sorry

end length_of_solution_set_l730_730103


namespace quiz_competition_l730_730872

theorem quiz_competition (N : ℕ) (h1 : 0.1 * N = 30) : N = 300 :=
by
  sorry

end quiz_competition_l730_730872


namespace conjugate_imag_part_l730_730041

open Complex

theorem conjugate_imag_part (z : ℂ) (hz: (1 + 2 * I) / z = I) : (conj z).im = 1 := 
sorry

end conjugate_imag_part_l730_730041


namespace proposition_negation_l730_730081

theorem proposition_negation (p : Prop) : 
  (∃ x : ℝ, x < 1 ∧ x^2 < 1) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
sorry

end proposition_negation_l730_730081
