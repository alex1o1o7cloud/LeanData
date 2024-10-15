import Mathlib

namespace NUMINAMATH_GPT_accounting_vs_calling_clients_l1844_184420

/--
Given:
1. Total time Maryann worked today is 560 minutes.
2. Maryann spent 70 minutes calling clients.

Prove:
Maryann spends 7 times longer doing accounting than calling clients.
-/
theorem accounting_vs_calling_clients 
  (total_time : ℕ) 
  (calling_time : ℕ) 
  (h_total : total_time = 560) 
  (h_calling : calling_time = 70) : 
  (total_time - calling_time) / calling_time = 7 :=
  sorry

end NUMINAMATH_GPT_accounting_vs_calling_clients_l1844_184420


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1844_184429

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  ( (x^2 - 1) / (x^2 - 6 * x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) ) = - (Real.sqrt 2 / 2) :=
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1844_184429


namespace NUMINAMATH_GPT_product_between_21st_and_24th_multiple_of_3_l1844_184486

theorem product_between_21st_and_24th_multiple_of_3 : 
  (66 * 69 = 4554) :=
by
  sorry

end NUMINAMATH_GPT_product_between_21st_and_24th_multiple_of_3_l1844_184486


namespace NUMINAMATH_GPT_retrievers_count_l1844_184462

-- Definitions of given conditions
def huskies := 5
def pitbulls := 2
def retrievers := Nat
def husky_pups := 3
def pitbull_pups := 3
def retriever_extra_pups := 2
def total_pups_excess := 30

-- Equation derived from the problem conditions
def total_pups (G : Nat) := huskies * husky_pups + pitbulls * pitbull_pups + G * (husky_pups + retriever_extra_pups)
def total_adults (G : Nat) := huskies + pitbulls + G

theorem retrievers_count : ∃ G : Nat, G = 4 ∧ total_pups G = total_adults G + total_pups_excess :=
by
  sorry

end NUMINAMATH_GPT_retrievers_count_l1844_184462


namespace NUMINAMATH_GPT_number_of_piles_l1844_184441

-- Defining the number of walnuts in total
def total_walnuts : Nat := 55

-- Defining the number of walnuts in the first pile
def first_pile_walnuts : Nat := 7

-- Defining the number of walnuts in each of the rest of the piles
def other_pile_walnuts : Nat := 12

-- The proposition we want to prove
theorem number_of_piles (n : Nat) :
  (n > 1) →
  (other_pile_walnuts * (n - 1) + first_pile_walnuts = total_walnuts) → n = 5 :=
sorry

end NUMINAMATH_GPT_number_of_piles_l1844_184441


namespace NUMINAMATH_GPT_cylindrical_container_volume_increase_l1844_184450

theorem cylindrical_container_volume_increase (R H : ℝ)
  (initial_volume : ℝ)
  (x : ℝ) : 
  R = 10 ∧ H = 5 ∧ initial_volume = π * R^2 * H →
  π * (R + 2 * x)^2 * H = π * R^2 * (H + 3 * x) →
  x = 5 :=
by
  -- Given conditions
  intro conditions volume_equation
  obtain ⟨hR, hH, hV⟩ := conditions
  -- Simplifying and solving the resulting equation
  sorry

end NUMINAMATH_GPT_cylindrical_container_volume_increase_l1844_184450


namespace NUMINAMATH_GPT_gcd_930_868_l1844_184458

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end NUMINAMATH_GPT_gcd_930_868_l1844_184458


namespace NUMINAMATH_GPT_cross_fills_space_without_gaps_l1844_184433

structure Cube :=
(x : ℤ)
(y : ℤ)
(z : ℤ)

structure Cross :=
(center : Cube)
(adjacent : List Cube)

def is_adjacent (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ abs (c1.z - c2.z) = 1) ∨
  (c1.x = c2.x ∧ abs (c1.y - c2.y) = 1 ∧ c1.z = c2.z) ∨
  (abs (c1.x - c2.x) = 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

def valid_cross (c : Cross) : Prop :=
  ∀ (adj : Cube), adj ∈ c.adjacent → is_adjacent c.center adj

def fills_space (crosses : List Cross) : Prop :=
  ∀ (pos : Cube), ∃ (c : Cross), c ∈ crosses ∧ 
    (pos = c.center ∨ pos ∈ c.adjacent)

theorem cross_fills_space_without_gaps 
  (crosses : List Cross) 
  (Hcross : ∀ c ∈ crosses, valid_cross c) : 
  fills_space crosses :=
sorry

end NUMINAMATH_GPT_cross_fills_space_without_gaps_l1844_184433


namespace NUMINAMATH_GPT_sum_m_n_zero_l1844_184489

theorem sum_m_n_zero (m n p : ℝ) (h1 : mn + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 :=
sorry

end NUMINAMATH_GPT_sum_m_n_zero_l1844_184489


namespace NUMINAMATH_GPT_problem1_problem2_l1844_184475

variables (a b : ℝ)

-- Problem 1: Prove that 3a^2 - 6a^2 - a^2 = -4a^2
theorem problem1 : (3 * a^2 - 6 * a^2 - a^2 = -4 * a^2) :=
by sorry

-- Problem 2: Prove that (5a - 3b) - 3(a^2 - 2b) = -3a^2 + 5a + 3b
theorem problem2 : ((5 * a - 3 * b) - 3 * (a^2 - 2 * b) = -3 * a^2 + 5 * a + 3 * b) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1844_184475


namespace NUMINAMATH_GPT_expression_meaningful_l1844_184415

theorem expression_meaningful (x : ℝ) : 
  (x - 1 ≠ 0 ∧ true) ↔ x ≠ 1 := 
sorry

end NUMINAMATH_GPT_expression_meaningful_l1844_184415


namespace NUMINAMATH_GPT_smallest_positive_integer_y_l1844_184484

theorem smallest_positive_integer_y
  (y : ℕ)
  (h_pos : 0 < y)
  (h_ineq : y^3 > 80) :
  y = 5 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_y_l1844_184484


namespace NUMINAMATH_GPT_storks_more_than_birds_l1844_184439

theorem storks_more_than_birds :
  let initial_birds := 3
  let additional_birds := 2
  let storks := 6
  storks - (initial_birds + additional_birds) = 1 :=
by
  sorry

end NUMINAMATH_GPT_storks_more_than_birds_l1844_184439


namespace NUMINAMATH_GPT_problem_l1844_184437

theorem problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 + a*b*c = 4) : 
  a + b + c ≤ 3 := 
sorry

end NUMINAMATH_GPT_problem_l1844_184437


namespace NUMINAMATH_GPT_min_value_of_inverse_proportional_function_l1844_184490

theorem min_value_of_inverse_proportional_function 
  (x y : ℝ) (k : ℝ) 
  (h1 : y = k / x) 
  (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → y ≤ 4) :
  (∀ x, x ≥ 8 → y = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_inverse_proportional_function_l1844_184490


namespace NUMINAMATH_GPT_total_spent_l1844_184404

/-- Define the prices of the rides in the morning and the afternoon --/
def morning_price (ride : String) (age : Nat) : Nat :=
  match ride, age with
  | "bumper_car", n => if n < 18 then 2 else 3
  | "space_shuttle", n => if n < 18 then 4 else 5
  | "ferris_wheel", n => if n < 18 then 5 else 6
  | _, _ => 0

def afternoon_price (ride : String) (age : Nat) : Nat :=
  (morning_price ride age) + 1

/-- Define the number of rides taken by Mara and Riley --/
def rides_morning (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 2
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 2
  | _, _ => 0

def rides_afternoon (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 1
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 1
  | _, _ => 0

/-- Define the ages of Mara and Riley --/
def age (person : String) : Nat :=
  match person with
  | "Mara" => 17
  | "Riley" => 19
  | _ => 0

/-- Calculate the total expenditure --/
def total_cost (person : String) : Nat :=
  List.sum ([
    (rides_morning person "bumper_car") * (morning_price "bumper_car" (age person)),
    (rides_afternoon person "bumper_car") * (afternoon_price "bumper_car" (age person)),
    (rides_morning person "space_shuttle") * (morning_price "space_shuttle" (age person)),
    (rides_afternoon person "space_shuttle") * (afternoon_price "space_shuttle" (age person)),
    (rides_morning person "ferris_wheel") * (morning_price "ferris_wheel" (age person)),
    (rides_afternoon person "ferris_wheel") * (afternoon_price "ferris_wheel" (age person))
  ])

/-- Prove the total cost for Mara and Riley is $62 --/
theorem total_spent : total_cost "Mara" + total_cost "Riley" = 62 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_l1844_184404


namespace NUMINAMATH_GPT_roots_polynomial_identity_l1844_184448

theorem roots_polynomial_identity (a b x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + b*x₁ + b^2 + a = 0) 
  (h₂ : x₂^2 + b*x₂ + b^2 + a = 0) : x₁^2 + x₁*x₂ + x₂^2 + a = 0 :=
by 
  sorry

end NUMINAMATH_GPT_roots_polynomial_identity_l1844_184448


namespace NUMINAMATH_GPT_sarah_interviewed_students_l1844_184442

theorem sarah_interviewed_students :
  let oranges := 70
  let pears := 120
  let apples := 147
  let strawberries := 113
  oranges + pears + apples + strawberries = 450 := by
sorry

end NUMINAMATH_GPT_sarah_interviewed_students_l1844_184442


namespace NUMINAMATH_GPT_sides_of_triangle_inequality_l1844_184443

theorem sides_of_triangle_inequality {a b c : ℝ} (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_sides_of_triangle_inequality_l1844_184443


namespace NUMINAMATH_GPT_MrsHiltCanTakeFriendsToMovies_l1844_184400

def TotalFriends : ℕ := 15
def FriendsCantGo : ℕ := 7
def FriendsCanGo : ℕ := 8

theorem MrsHiltCanTakeFriendsToMovies : TotalFriends - FriendsCantGo = FriendsCanGo := by
  -- The proof will show that 15 - 7 = 8.
  sorry

end NUMINAMATH_GPT_MrsHiltCanTakeFriendsToMovies_l1844_184400


namespace NUMINAMATH_GPT_no_real_solutions_for_equation_l1844_184446

theorem no_real_solutions_for_equation (x : ℝ) : ¬(∃ x : ℝ, (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7) :=
sorry

end NUMINAMATH_GPT_no_real_solutions_for_equation_l1844_184446


namespace NUMINAMATH_GPT_unit_digit_of_power_of_two_l1844_184406

theorem unit_digit_of_power_of_two (n : ℕ) :
  (2 ^ 2023) % 10 = 8 := 
by
  sorry

end NUMINAMATH_GPT_unit_digit_of_power_of_two_l1844_184406


namespace NUMINAMATH_GPT_aardvark_total_distance_l1844_184409

noncomputable def total_distance (r_small r_large : ℝ) : ℝ :=
  let small_circumference := 2 * Real.pi * r_small
  let large_circumference := 2 * Real.pi * r_large
  let half_small_circumference := small_circumference / 2
  let half_large_circumference := large_circumference / 2
  let radial_distance := r_large - r_small
  let total_radial_distance := radial_distance + r_large
  half_small_circumference + radial_distance + half_large_circumference + total_radial_distance

theorem aardvark_total_distance :
  total_distance 15 30 = 45 * Real.pi + 45 :=
by
  sorry

end NUMINAMATH_GPT_aardvark_total_distance_l1844_184409


namespace NUMINAMATH_GPT_total_number_of_marbles_is_1050_l1844_184423

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end NUMINAMATH_GPT_total_number_of_marbles_is_1050_l1844_184423


namespace NUMINAMATH_GPT_rectangle_area_l1844_184417

theorem rectangle_area (AB AD AE : ℝ) (S_trapezoid S_triangle : ℝ) (perim_triangle perim_trapezoid : ℝ)
  (h1 : AD - AB = 9)
  (h2 : S_trapezoid = 5 * S_triangle)
  (h3 : perim_triangle + 68 = perim_trapezoid)
  (h4 : S_trapezoid + S_triangle = S_triangle * 6)
  (h5 : perim_triangle = AB + AE + (AE - AB))
  (h6 : perim_trapezoid = AB + AD + AE + (2 * (AD - AE))) :
  AD * AB = 3060 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1844_184417


namespace NUMINAMATH_GPT_triangle_inequality_l1844_184418

theorem triangle_inequality (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_triangle : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2)) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1844_184418


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1844_184427

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (x^2 - 4*x + 3) < 0 ↔ 1 < x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1844_184427


namespace NUMINAMATH_GPT_evaluate_expression_l1844_184447

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a * b + b^2) = 97 / 7 := by
  sorry

example : (3^4 + 2^4) / (3^2 - 3 * 2 + 2^2) = 97 / 7 := evaluate_expression 3 2 rfl rfl

end NUMINAMATH_GPT_evaluate_expression_l1844_184447


namespace NUMINAMATH_GPT_six_digit_mod_27_l1844_184459

theorem six_digit_mod_27 (X : ℕ) (hX : 100000 ≤ X ∧ X < 1000000) (Y : ℕ) (hY : ∃ a b : ℕ, 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ X = 1000 * a + b ∧ Y = 1000 * b + a) :
  X % 27 = Y % 27 := 
by
  sorry

end NUMINAMATH_GPT_six_digit_mod_27_l1844_184459


namespace NUMINAMATH_GPT_charlie_max_success_ratio_l1844_184419

-- Given:
-- Alpha scored 180 points out of 360 attempted on day one.
-- Alpha scored 120 points out of 240 attempted on day two.
-- Charlie did not attempt 360 points on the first day.
-- Charlie's success ratio on each day was less than Alpha’s.
-- Total points attempted by Charlie on both days are 600.
-- Alpha's two-day success ratio is 300/600 = 1/2.
-- Find the largest possible two-day success ratio that Charlie could have achieved.

theorem charlie_max_success_ratio:
  ∀ (x y z w : ℕ),
  0 < x ∧ 0 < z ∧ 0 < y ∧ 0 < w ∧
  y + w = 600 ∧
  (2 * x < y) ∧ (2 * z < w) ∧
  (x + z < 300) -> (299 / 600 = 299 / 600) :=
by
  sorry

end NUMINAMATH_GPT_charlie_max_success_ratio_l1844_184419


namespace NUMINAMATH_GPT_max_difference_two_digit_numbers_l1844_184436

theorem max_difference_two_digit_numbers (A B : ℤ) (hA : 10 ≤ A ∧ A ≤ 99) (hB : 10 ≤ B ∧ B ≤ 99) (h : 2 * A * 3 = 2 * B * 7) : 
  56 ≤ A - B :=
sorry

end NUMINAMATH_GPT_max_difference_two_digit_numbers_l1844_184436


namespace NUMINAMATH_GPT_ceil_square_of_neg_five_thirds_l1844_184444

theorem ceil_square_of_neg_five_thirds : Int.ceil ((-5 / 3:ℚ)^2) = 3 := by
  sorry

end NUMINAMATH_GPT_ceil_square_of_neg_five_thirds_l1844_184444


namespace NUMINAMATH_GPT_cafeteria_green_apples_l1844_184430

def number_of_green_apples (G : ℕ) : Prop :=
  42 + G - 9 = 40 → G = 7

theorem cafeteria_green_apples
  (red_apples : ℕ)
  (students_wanting_fruit : ℕ)
  (extra_fruit : ℕ)
  (G : ℕ)
  (h1 : red_apples = 42)
  (h2 : students_wanting_fruit = 9)
  (h3 : extra_fruit = 40)
  : number_of_green_apples G :=
by
  -- Place for proof omitted intentionally
  sorry

end NUMINAMATH_GPT_cafeteria_green_apples_l1844_184430


namespace NUMINAMATH_GPT_total_cupcakes_baked_l1844_184481

theorem total_cupcakes_baked
    (boxes : ℕ)
    (cupcakes_per_box : ℕ)
    (left_at_home : ℕ)
    (total_given_away : ℕ)
    (total_baked : ℕ)
    (h1 : boxes = 17)
    (h2 : cupcakes_per_box = 3)
    (h3 : left_at_home = 2)
    (h4 : total_given_away = boxes * cupcakes_per_box)
    (h5 : total_baked = total_given_away + left_at_home) :
    total_baked = 53 := by
  sorry

end NUMINAMATH_GPT_total_cupcakes_baked_l1844_184481


namespace NUMINAMATH_GPT_part_a_l1844_184408

theorem part_a (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 2) : 
  (1 / x + 1 / y) ≤ (1 / x^2 + 1 / y^2) := 
sorry

end NUMINAMATH_GPT_part_a_l1844_184408


namespace NUMINAMATH_GPT_green_duck_percentage_l1844_184460

theorem green_duck_percentage (G_small G_large : ℝ) (D_small D_large : ℕ)
    (H1 : G_small = 0.20) (H2 : D_small = 20)
    (H3 : G_large = 0.15) (H4 : D_large = 80) : 
    ((G_small * D_small + G_large * D_large) / (D_small + D_large)) * 100 = 16 := 
by
  sorry

end NUMINAMATH_GPT_green_duck_percentage_l1844_184460


namespace NUMINAMATH_GPT_find_a_add_b_l1844_184465

theorem find_a_add_b (a b : ℝ) 
  (h1 : ∀ (x : ℝ), y = a + b / (x^2 + 1))
  (h2 : (y = 3) → (x = 1)) 
  (h3 : (y = 2) → (x = 0)) : a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_add_b_l1844_184465


namespace NUMINAMATH_GPT_prove_relationship_l1844_184479

noncomputable def relationship_x_y_z (x y z : ℝ) (t : ℝ) : Prop :=
  (x / Real.sin t) = (y / Real.sin (2 * t)) ∧ (x / Real.sin t) = (z / Real.sin (3 * t))

theorem prove_relationship (x y z t : ℝ) (h : relationship_x_y_z x y z t) : x^2 - y^2 + x * z = 0 :=
by
  sorry

end NUMINAMATH_GPT_prove_relationship_l1844_184479


namespace NUMINAMATH_GPT_range_of_a_l1844_184428

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f (2^x) = x^2 - 2 * a * x + a^2 - 1) →
  (∀ x, 2^(a-1) ≤ x ∧ x ≤ 2^(a^2 - 2*a + 2) → -1 ≤ f x ∧ f x ≤ 0) →
  ((3 - Real.sqrt 5) / 2 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a ∧ a ≤ (3 + Real.sqrt 5) / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1844_184428


namespace NUMINAMATH_GPT_f_lg_equality_l1844_184491

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_lg_equality : f (Real.log 2) + f (Real.log (1 / 2)) = 2 := sorry

end NUMINAMATH_GPT_f_lg_equality_l1844_184491


namespace NUMINAMATH_GPT_p_or_q_iff_not_p_and_not_q_false_l1844_184414

variables (p q : Prop)

theorem p_or_q_iff_not_p_and_not_q_false : (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
by sorry

end NUMINAMATH_GPT_p_or_q_iff_not_p_and_not_q_false_l1844_184414


namespace NUMINAMATH_GPT_student_solved_correctly_l1844_184494

theorem student_solved_correctly (x : ℕ) :
  (x + 2 * x = 36) → x = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_student_solved_correctly_l1844_184494


namespace NUMINAMATH_GPT_base3_sum_example_l1844_184403

noncomputable def base3_add (a b : ℕ) : ℕ := sorry  -- Function to perform base-3 addition

theorem base3_sum_example : 
  base3_add (base3_add (base3_add (base3_add 2 120) 221) 1112) 1022 = 21201 := sorry

end NUMINAMATH_GPT_base3_sum_example_l1844_184403


namespace NUMINAMATH_GPT_claire_flour_cost_l1844_184461

def num_cakes : ℕ := 2
def flour_per_cake : ℕ := 2
def cost_per_flour : ℕ := 3
def total_cost (num_cakes flour_per_cake cost_per_flour : ℕ) : ℕ := 
  num_cakes * flour_per_cake * cost_per_flour

theorem claire_flour_cost : total_cost num_cakes flour_per_cake cost_per_flour = 12 := by
  sorry

end NUMINAMATH_GPT_claire_flour_cost_l1844_184461


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l1844_184435

theorem cylinder_volume_ratio (h_C r_D : ℝ) (V_C V_D : ℝ) :
  h_C = 3 * r_D →
  r_D = h_C →
  V_C = 3 * V_D →
  V_C = (1 / 9) * π * h_C^3 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l1844_184435


namespace NUMINAMATH_GPT_minimum_guests_at_banquet_l1844_184499

theorem minimum_guests_at_banquet (total_food : ℝ) (max_food_per_guest : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 411) (h2 : max_food_per_guest = 2.5) : min_guests = 165 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_minimum_guests_at_banquet_l1844_184499


namespace NUMINAMATH_GPT_kristen_turtles_l1844_184492

variable (K : ℕ)
variable (T : ℕ)
variable (R : ℕ)

-- Conditions
def kris_turtles (K : ℕ) : ℕ := K / 4
def trey_turtles (R : ℕ) : ℕ := 7 * R
def trey_more_than_kristen (T K : ℕ) : Prop := T = K + 9

-- Theorem to prove 
theorem kristen_turtles (K : ℕ) (R : ℕ) (T : ℕ) (h1 : R = kris_turtles K) (h2 : T = trey_turtles R) (h3 : trey_more_than_kristen T K) : K = 12 :=
by
  sorry

end NUMINAMATH_GPT_kristen_turtles_l1844_184492


namespace NUMINAMATH_GPT_largest_four_digit_divisible_by_14_l1844_184452

theorem largest_four_digit_divisible_by_14 :
  ∃ (A : ℕ), A = 9898 ∧ 
  (∃ a b : ℕ, A = 1010 * a + 101 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) ∧
  (A % 14 = 0) ∧
  (A = (d1 * 100 + d2 * 10 + d1) * 101)
  :=
sorry

end NUMINAMATH_GPT_largest_four_digit_divisible_by_14_l1844_184452


namespace NUMINAMATH_GPT_find_d_l1844_184456

namespace NineDigitNumber

variables {A B C D E F G : ℕ}

theorem find_d 
  (h1 : 6 + A + B = 13) 
  (h2 : A + B + C = 13)
  (h3 : B + C + D = 13)
  (h4 : C + D + E = 13)
  (h5 : D + E + F = 13)
  (h6 : E + F + G = 13)
  (h7 : F + G + 3 = 13) :
  D = 4 :=
sorry

end NineDigitNumber

end NUMINAMATH_GPT_find_d_l1844_184456


namespace NUMINAMATH_GPT_grace_can_reach_target_sum_l1844_184487

theorem grace_can_reach_target_sum :
  ∃ (half_dollars dimes pennies : ℕ),
    half_dollars ≤ 5 ∧ dimes ≤ 20 ∧ pennies ≤ 25 ∧
    (5 * 50 + 13 * 10 + 5) = 385 :=
sorry

end NUMINAMATH_GPT_grace_can_reach_target_sum_l1844_184487


namespace NUMINAMATH_GPT_amount_subtracted_correct_l1844_184440

noncomputable def find_subtracted_amount (N : ℝ) (A : ℝ) : Prop :=
  0.40 * N - A = 23

theorem amount_subtracted_correct :
  find_subtracted_amount 85 11 :=
by
  sorry

end NUMINAMATH_GPT_amount_subtracted_correct_l1844_184440


namespace NUMINAMATH_GPT_bank_card_payment_technology_order_l1844_184477

-- Conditions as definitions
def action_tap := 1
def action_pay_online := 2
def action_swipe := 3
def action_insert_into_terminal := 4

-- Corresponding proof problem statement
theorem bank_card_payment_technology_order :
  [action_insert_into_terminal, action_swipe, action_tap, action_pay_online] = [4, 3, 1, 2] := by
  sorry

end NUMINAMATH_GPT_bank_card_payment_technology_order_l1844_184477


namespace NUMINAMATH_GPT_tablets_taken_l1844_184426

theorem tablets_taken (total_time interval_time : ℕ) (h1 : total_time = 60) (h2 : interval_time = 15) : total_time / interval_time = 4 :=
by
  sorry

end NUMINAMATH_GPT_tablets_taken_l1844_184426


namespace NUMINAMATH_GPT_roberto_current_salary_l1844_184498

theorem roberto_current_salary (starting_salary current_salary : ℝ) (h₀ : starting_salary = 80000)
(h₁ : current_salary = (starting_salary * 1.4) * 1.2) : 
current_salary = 134400 := by
  sorry

end NUMINAMATH_GPT_roberto_current_salary_l1844_184498


namespace NUMINAMATH_GPT_length_of_plot_l1844_184468

theorem length_of_plot (breadth length : ℕ) 
                       (h1 : length = breadth + 26)
                       (fencing_cost total_cost : ℝ)
                       (h2 : fencing_cost = 26.50)
                       (h3 : total_cost = 5300)
                       (perimeter : ℝ) 
                       (h4 : perimeter = 2 * (breadth + length)) 
                       (h5 : total_cost = perimeter * fencing_cost) :
                       length = 63 :=
by
  sorry

end NUMINAMATH_GPT_length_of_plot_l1844_184468


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1844_184476

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 1) (h2 : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1844_184476


namespace NUMINAMATH_GPT_even_function_value_l1844_184482

-- Define the function condition
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the main problem with given conditions
theorem even_function_value (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : ∀ x : ℝ, x < 0 → f x = x * (x + 1)) 
  (x : ℝ) (hx : x > 0) : f x = x * (x - 1) :=
  sorry

end NUMINAMATH_GPT_even_function_value_l1844_184482


namespace NUMINAMATH_GPT_range_of_m_l1844_184405

-- Defining the point P and the required conditions for it to lie in the fourth quadrant
def point_in_fourth_quadrant (m : ℝ) : Prop :=
  let P := (m + 3, m - 1)
  P.1 > 0 ∧ P.2 < 0

-- Defining the range of m for which the point lies in the fourth quadrant
theorem range_of_m (m : ℝ) : point_in_fourth_quadrant m ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1844_184405


namespace NUMINAMATH_GPT_equivalent_form_l1844_184472

theorem equivalent_form (p q : ℝ) (hp₁ : p ≠ 0) (hp₂ : p ≠ 5) (hq₁ : q ≠ 0) (hq₂ : q ≠ 7) :
  (3/p + 4/q = 1/3) ↔ (p = 9*q/(q - 12)) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_form_l1844_184472


namespace NUMINAMATH_GPT_days_to_fulfill_order_l1844_184454

theorem days_to_fulfill_order (bags_per_batch : ℕ) (total_order : ℕ) (initial_bags : ℕ) (required_days : ℕ) :
  bags_per_batch = 10 →
  total_order = 60 →
  initial_bags = 20 →
  required_days = (total_order - initial_bags) / bags_per_batch →
  required_days = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_days_to_fulfill_order_l1844_184454


namespace NUMINAMATH_GPT_math_problem_l1844_184425

variable (a b c : ℝ)

theorem math_problem (h1 : -10 ≤ a ∧ a < 0) (h2 : 0 < a ∧ a < b ∧ b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1844_184425


namespace NUMINAMATH_GPT_train_speed_is_correct_l1844_184485

noncomputable def train_length : ℕ := 900
noncomputable def platform_length : ℕ := train_length
noncomputable def time_in_minutes : ℕ := 1
noncomputable def distance_covered : ℕ := train_length + platform_length
noncomputable def speed_m_per_minute : ℕ := distance_covered / time_in_minutes
noncomputable def speed_km_per_hr : ℕ := (speed_m_per_minute * 60) / 1000

theorem train_speed_is_correct :
  speed_km_per_hr = 108 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l1844_184485


namespace NUMINAMATH_GPT_solve_floor_equation_l1844_184496

noncomputable def x_solution_set : Set ℚ := 
  {x | x = 1 ∨ ∃ k : ℕ, 16 ≤ k ∧ k ≤ 22 ∧ x = (k : ℚ)/23 }

theorem solve_floor_equation (x : ℚ) (hx : x ∈ x_solution_set) : 
  (⌊20*x + 23⌋ : ℚ) = 20 + 23*x :=
sorry

end NUMINAMATH_GPT_solve_floor_equation_l1844_184496


namespace NUMINAMATH_GPT_prime_factorial_division_l1844_184474

theorem prime_factorial_division (p k n : ℕ) (hp : Prime p) (h : p^k ∣ n!) : (p!)^k ∣ n! :=
sorry

end NUMINAMATH_GPT_prime_factorial_division_l1844_184474


namespace NUMINAMATH_GPT_smallest_multiple_1_through_10_l1844_184480

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end NUMINAMATH_GPT_smallest_multiple_1_through_10_l1844_184480


namespace NUMINAMATH_GPT_negation_proposition_l1844_184407

theorem negation_proposition :
  (¬ ∃ x : ℝ, (x > -1 ∧ x < 3) ∧ (x^2 - 1 ≤ 2 * x)) ↔ 
  (∀ x : ℝ, (x > -1 ∧ x < 3) → (x^2 - 1 > 2 * x)) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_proposition_l1844_184407


namespace NUMINAMATH_GPT_approx_val_l1844_184416

variable (x : ℝ) (y : ℝ)

-- Definitions based on rounding condition
def approx_0_000315 : ℝ := 0.0003
def approx_7928564 : ℝ := 8000000

-- Main theorem statement
theorem approx_val (h1: x = approx_0_000315) (h2: y = approx_7928564) :
  x * y = 2400 := by
  sorry

end NUMINAMATH_GPT_approx_val_l1844_184416


namespace NUMINAMATH_GPT_mass_percentage_C_in_CaCO3_is_correct_l1844_184464

structure Element where
  name : String
  molar_mass : ℚ

def Ca : Element := ⟨"Ca", 40.08⟩
def C : Element := ⟨"C", 12.01⟩
def O : Element := ⟨"O", 16.00⟩

def molar_mass_CaCO3 : ℚ :=
  Ca.molar_mass + C.molar_mass + 3 * O.molar_mass

def mass_percentage_C_in_CaCO3 : ℚ :=
  (C.molar_mass / molar_mass_CaCO3) * 100

theorem mass_percentage_C_in_CaCO3_is_correct :
  mass_percentage_C_in_CaCO3 = 12.01 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_C_in_CaCO3_is_correct_l1844_184464


namespace NUMINAMATH_GPT_toby_money_share_l1844_184467

theorem toby_money_share (initial_money : ℕ) (fraction : ℚ) (brothers : ℕ) (money_per_brother : ℚ)
  (total_shared : ℕ) (remaining_money : ℕ) :
  initial_money = 343 →
  fraction = 1/7 →
  brothers = 2 →
  money_per_brother = fraction * initial_money →
  total_shared = brothers * money_per_brother →
  remaining_money = initial_money - total_shared →
  remaining_money = 245 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_toby_money_share_l1844_184467


namespace NUMINAMATH_GPT_total_amount_spent_l1844_184431

def cost_of_tshirt : ℕ := 100
def cost_of_pants : ℕ := 250
def num_of_tshirts : ℕ := 5
def num_of_pants : ℕ := 4

theorem total_amount_spent : (num_of_tshirts * cost_of_tshirt) + (num_of_pants * cost_of_pants) = 1500 := by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1844_184431


namespace NUMINAMATH_GPT_fraction_value_l1844_184470

theorem fraction_value (x : ℝ) (h₀ : x^2 - 3 * x - 1 = 0) (h₁ : x ≠ 0) : 
  x^2 / (x^4 + x^2 + 1) = 1 / 12 := 
by
  sorry

end NUMINAMATH_GPT_fraction_value_l1844_184470


namespace NUMINAMATH_GPT_largest_value_satisfies_abs_equation_l1844_184422

theorem largest_value_satisfies_abs_equation (x : ℝ) : |5 - x| = 15 + x → x = -5 := by
  intros h
  sorry

end NUMINAMATH_GPT_largest_value_satisfies_abs_equation_l1844_184422


namespace NUMINAMATH_GPT_find_num_apples_l1844_184497

def num_apples (A P : ℕ) : Prop :=
  P = (3 * A) / 5 ∧ A + P = 240

theorem find_num_apples (A : ℕ) (P : ℕ) :
  num_apples A P → A = 150 :=
by
  intros h
  -- sorry for proof
  sorry

end NUMINAMATH_GPT_find_num_apples_l1844_184497


namespace NUMINAMATH_GPT_periodic_modulo_h_l1844_184469

open Nat

-- Defining the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Defining the sequence as per the problem
def x_seq (n : ℕ) : ℕ :=
  binom (2 * n) n

-- The main theorem stating the required condition
theorem periodic_modulo_h (h : ℕ) (h_gt_one : h > 1) :
  (∃ N, ∀ n ≥ N, x_seq n % h = x_seq (n + 1) % h) ↔ h = 2 :=
by
  sorry

end NUMINAMATH_GPT_periodic_modulo_h_l1844_184469


namespace NUMINAMATH_GPT_function_relationship_value_of_x_l1844_184455

variable {x y : ℝ}

-- Given conditions:
-- Condition 1: y is inversely proportional to x
def inversely_proportional (p : ℝ) (q : ℝ) (k : ℝ) : Prop := p = k / q

-- Condition 2: y(2) = -3
def specific_value (x_val y_val : ℝ) : Prop := y_val = -3 ∧ x_val = 2

-- Questions rephrased as Lean theorems:

-- The function relationship between y and x is y = -6 / x
theorem function_relationship (k : ℝ) (hx : x ≠ 0) 
  (h_inv_prop: inversely_proportional y x k) (h_spec : specific_value 2 (-3)) : k = -6 :=
by
  sorry

-- When y = 2, x = -3
theorem value_of_x (hx : x ≠ 0) (hy : y = 2)
  (h_inv_prop : inversely_proportional y x (-6)) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_function_relationship_value_of_x_l1844_184455


namespace NUMINAMATH_GPT_range_of_k_l1844_184473

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k - 1) * x^2 + (k - 1) * x + 2 > 0) ↔ 1 ≤ k ∧ k < 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1844_184473


namespace NUMINAMATH_GPT_geometric_series_sum_condition_l1844_184410

def geometric_series_sum (a q n : ℕ) : ℕ := a * (1 - q^n) / (1 - q)

theorem geometric_series_sum_condition (S : ℕ → ℕ) (a : ℕ) (q : ℕ) (h1 : a = 1) 
  (h2 : ∀ n, S n = geometric_series_sum a q n)
  (h3 : S 7 - 4 * S 6 + 3 * S 5 = 0) : 
  S 4 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_series_sum_condition_l1844_184410


namespace NUMINAMATH_GPT_determine_a_l1844_184421

theorem determine_a (a : ℝ) : (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) := 
sorry

end NUMINAMATH_GPT_determine_a_l1844_184421


namespace NUMINAMATH_GPT_tan_x_eq_sqrt3_l1844_184478

theorem tan_x_eq_sqrt3 (x : Real) (h : Real.sin (x + 20 * Real.pi / 180) = Real.cos (x + 10 * Real.pi / 180) + Real.cos (x - 10 * Real.pi / 180)) : Real.tan x = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_x_eq_sqrt3_l1844_184478


namespace NUMINAMATH_GPT_union_of_sets_l1844_184471

-- Definitions based on conditions
def A : Set ℕ := {2, 3}
def B (a : ℕ) : Set ℕ := {1, a}
def condition (a : ℕ) : Prop := A ∩ (B a) = {2}

-- Main theorem to be proven
theorem union_of_sets (a : ℕ) (h : condition a) : A ∪ (B a) = {1, 2, 3} :=
sorry

end NUMINAMATH_GPT_union_of_sets_l1844_184471


namespace NUMINAMATH_GPT_uniform_prob_correct_l1844_184424

noncomputable def uniform_prob_within_interval 
  (α β γ δ : ℝ) 
  (h₁ : α ≤ β) 
  (h₂ : α ≤ γ) 
  (h₃ : γ < δ) 
  (h₄ : δ ≤ β) : ℝ :=
  (δ - γ) / (β - α)

theorem uniform_prob_correct 
  (α β γ δ : ℝ) 
  (hαβ : α ≤ β) 
  (hαγ : α ≤ γ) 
  (hγδ : γ < δ) 
  (hδβ : δ ≤ β) :
  uniform_prob_within_interval α β γ δ hαβ hαγ hγδ hδβ = (δ - γ) / (β - α) := sorry

end NUMINAMATH_GPT_uniform_prob_correct_l1844_184424


namespace NUMINAMATH_GPT_find_a7_l1844_184413

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -4/3 ∧ (∀ n, a (n + 2) = 1 / (a n + 1))

theorem find_a7 (a : ℕ → ℚ) (h : seq a) : a 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a7_l1844_184413


namespace NUMINAMATH_GPT_determine_n_l1844_184412

theorem determine_n (k : ℕ) (n : ℕ) (h1 : 21^k ∣ n) (h2 : 7^k - k^7 = 1) : n = 1 :=
sorry

end NUMINAMATH_GPT_determine_n_l1844_184412


namespace NUMINAMATH_GPT_exponentiation_81_5_4_eq_243_l1844_184488

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end NUMINAMATH_GPT_exponentiation_81_5_4_eq_243_l1844_184488


namespace NUMINAMATH_GPT_books_on_shelves_l1844_184453

-- Definitions based on the problem conditions.
def bookshelves : ℕ := 1250
def books_per_shelf : ℕ := 45
def total_books : ℕ := 56250

-- Theorem statement
theorem books_on_shelves : bookshelves * books_per_shelf = total_books := 
by
  sorry

end NUMINAMATH_GPT_books_on_shelves_l1844_184453


namespace NUMINAMATH_GPT_points_per_game_l1844_184449

theorem points_per_game (total_points : ℝ) (num_games : ℝ) (h1 : total_points = 120.0) (h2 : num_games = 10.0) : (total_points / num_games) = 12.0 :=
by 
  rw [h1, h2]
  norm_num
  -- sorry


end NUMINAMATH_GPT_points_per_game_l1844_184449


namespace NUMINAMATH_GPT_find_number_of_eggs_l1844_184438

namespace HalloweenCleanup

def eggs (E : ℕ) (seconds_per_egg : ℕ) (minutes_per_roll : ℕ) (total_time : ℕ) (num_rolls : ℕ) : Prop :=
  seconds_per_egg = 15 ∧
  minutes_per_roll = 30 ∧
  total_time = 225 ∧
  num_rolls = 7 ∧
  E * (seconds_per_egg / 60) + num_rolls * minutes_per_roll = total_time

theorem find_number_of_eggs : ∃ E : ℕ, eggs E 15 30 225 7 :=
  by
    use 60
    unfold eggs
    simp
    exact sorry

end HalloweenCleanup

end NUMINAMATH_GPT_find_number_of_eggs_l1844_184438


namespace NUMINAMATH_GPT_overlap_percentage_l1844_184483

noncomputable def square_side_length : ℝ := 10
noncomputable def rectangle_length : ℝ := 18
noncomputable def rectangle_width : ℝ := square_side_length
noncomputable def overlap_length : ℝ := 2
noncomputable def overlap_width : ℝ := rectangle_width

noncomputable def rectangle_area : ℝ :=
  rectangle_length * rectangle_width

noncomputable def overlap_area : ℝ :=
  overlap_length * overlap_width

noncomputable def percentage_shaded : ℝ :=
  (overlap_area / rectangle_area) * 100

theorem overlap_percentage :
  percentage_shaded = 100 * (1 / 9) :=
sorry

end NUMINAMATH_GPT_overlap_percentage_l1844_184483


namespace NUMINAMATH_GPT_sequence_problem_l1844_184493

noncomputable def b_n (n : ℕ) : ℝ := 5 * (5/3)^(n-2)

theorem sequence_problem 
  (a_n : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : a_n 8 = a_n 5 + 3 * d)
  (h4 : a_n 13 = a_n 8 + 5 * d)
  (b_2 : ℝ)
  (hb2 : b_2 = 5)
  (h5 : ∀ n, b_n n = (match n with | 2 => b_2 | _ => sorry))
  (conseq_terms : ∀ (n : ℕ), (a_n 5 + 3 * d)^2 = a_n 5 * (a_n 5 + 8 * d)) 
  : ∀ n, b_n n = b_n 2 * (5/3)^(n-2) := 
by 
  sorry

end NUMINAMATH_GPT_sequence_problem_l1844_184493


namespace NUMINAMATH_GPT_lines_intersect_l1844_184432

theorem lines_intersect :
  ∃ x y : ℚ, 
  8 * x - 5 * y = 40 ∧ 
  6 * x - y = -5 ∧ 
  x = 15 / 38 ∧ 
  y = 140 / 19 :=
by { sorry }

end NUMINAMATH_GPT_lines_intersect_l1844_184432


namespace NUMINAMATH_GPT_max_n_sum_pos_largest_term_seq_l1844_184401

-- Define the arithmetic sequence {a_n} and sum of first n terms S_n along with given conditions
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d
def sum_arith_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

variable (a_1 d : ℤ)
-- Conditions from problem
axiom a8_pos : arithmetic_seq a_1 d 8 > 0
axiom a8_a9_neg : arithmetic_seq a_1 d 8 + arithmetic_seq a_1 d 9 < 0

-- Prove the maximum n for which Sum S_n > 0 is 15
theorem max_n_sum_pos : ∃ n_max : ℤ, sum_arith_seq a_1 d n_max > 0 ∧ 
  ∀ n : ℤ, n > n_max → sum_arith_seq a_1 d n ≤ 0 := by
    exact ⟨15, sorry⟩  -- Substitute 'sorry' for the proof part

-- Determine the largest term in the sequence {S_n / a_n} for 1 ≤ n ≤ 15
theorem largest_term_seq : ∃ n_largest : ℤ, ∀ n : ℤ, 1 ≤ n → n ≤ 15 → 
  (sum_arith_seq a_1 d n / arithmetic_seq a_1 d n) ≤ (sum_arith_seq a_1 d n_largest / arithmetic_seq a_1 d n_largest) := by
    exact ⟨8, sorry⟩  -- Substitute 'sorry' for the proof part

end NUMINAMATH_GPT_max_n_sum_pos_largest_term_seq_l1844_184401


namespace NUMINAMATH_GPT_sandy_spent_on_shirt_l1844_184457

-- Define the conditions
def cost_of_shorts : ℝ := 13.99
def cost_of_jacket : ℝ := 7.43
def total_spent_on_clothes : ℝ := 33.56

-- Define the amount spent on the shirt
noncomputable def cost_of_shirt : ℝ :=
  total_spent_on_clothes - (cost_of_shorts + cost_of_jacket)

-- Prove that Sandy spent $12.14 on the shirt
theorem sandy_spent_on_shirt : cost_of_shirt = 12.14 :=
by
  sorry

end NUMINAMATH_GPT_sandy_spent_on_shirt_l1844_184457


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l1844_184466

theorem isosceles_triangle_base_angle
    (X : ℝ)
    (h1 : 0 < X)
    (h2 : 2 * X + X + X = 180)
    (h3 : X + X + 2 * X = 180) :
    X = 45 ∨ X = 72 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l1844_184466


namespace NUMINAMATH_GPT_smallest_of_consecutive_even_numbers_l1844_184451

theorem smallest_of_consecutive_even_numbers (n : ℤ) (h : ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ c = 2 * n + 1) :
  ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ a = 2 * n - 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_of_consecutive_even_numbers_l1844_184451


namespace NUMINAMATH_GPT_stuffed_animal_cost_is_6_l1844_184445

-- Definitions for the costs of items
def sticker_cost (s : ℕ) := s
def magnet_cost (m : ℕ) := m
def stuffed_animal_cost (a : ℕ) := a

-- Conditions given in the problem
def conditions (m s a : ℕ) :=
  (m = 3) ∧
  (m = 3 * s) ∧
  (m = (2 * a) / 4)

-- The theorem stating the cost of a single stuffed animal
theorem stuffed_animal_cost_is_6 (s m a : ℕ) (h : conditions m s a) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_stuffed_animal_cost_is_6_l1844_184445


namespace NUMINAMATH_GPT_exist_students_with_comparable_scores_l1844_184463

theorem exist_students_with_comparable_scores :
  ∃ (A B : ℕ) (a1 a2 a3 b1 b2 b3 : ℕ), 
    A ≠ B ∧ A < 49 ∧ B < 49 ∧
    (0 ≤ a1 ∧ a1 ≤ 7) ∧ (0 ≤ a2 ∧ a2 ≤ 7) ∧ (0 ≤ a3 ∧ a3 ≤ 7) ∧ 
    (0 ≤ b1 ∧ b1 ≤ 7) ∧ (0 ≤ b2 ∧ b2 ≤ 7) ∧ (0 ≤ b3 ∧ b3 ≤ 7) ∧ 
    (a1 ≥ b1) ∧ (a2 ≥ b2) ∧ (a3 ≥ b3) := 
sorry

end NUMINAMATH_GPT_exist_students_with_comparable_scores_l1844_184463


namespace NUMINAMATH_GPT_ratio_of_votes_l1844_184402

theorem ratio_of_votes (votes_A votes_B total_votes : ℕ) (hA : votes_A = 14) (hTotal : votes_A + votes_B = 21) : votes_A / Nat.gcd votes_A votes_B = 2 ∧ votes_B / Nat.gcd votes_A votes_B = 1 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_votes_l1844_184402


namespace NUMINAMATH_GPT_positive_integer_fraction_l1844_184495

theorem positive_integer_fraction (p : ℕ) (h1 : p > 0) (h2 : (3 * p + 25) / (2 * p - 5) > 0) :
  3 ≤ p ∧ p ≤ 35 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_fraction_l1844_184495


namespace NUMINAMATH_GPT_total_spent_l1844_184411

def price_almond_croissant : ℝ := 4.50
def price_salami_cheese_croissant : ℝ := 4.50
def price_plain_croissant : ℝ := 3.00
def price_focaccia : ℝ := 4.00
def price_latte : ℝ := 2.50
def num_lattes : ℕ := 2

theorem total_spent :
  price_almond_croissant + price_salami_cheese_croissant + price_plain_croissant +
  price_focaccia + (num_lattes * price_latte) = 21.00 := by
  sorry

end NUMINAMATH_GPT_total_spent_l1844_184411


namespace NUMINAMATH_GPT_sum_of_four_consecutive_integers_is_even_l1844_184434

theorem sum_of_four_consecutive_integers_is_even (n : ℤ) : 2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end NUMINAMATH_GPT_sum_of_four_consecutive_integers_is_even_l1844_184434
