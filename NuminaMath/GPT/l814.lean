import Mathlib

namespace find_range_a_l814_81439

def setA (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def setB (x a : ℝ) : Prop := |x - a| < 5
def real_line (x : ℝ) : Prop := True

theorem find_range_a (a : ℝ) :
  (∀ x, setA x ∨ setB x a) ↔ (-3:ℝ) ≤ a ∧ a ≤ 1 := by
sorry

end find_range_a_l814_81439


namespace product_increased_five_times_l814_81408

variables (A B : ℝ)

theorem product_increased_five_times (h : A * B = 1.6) : (5 * A) * (5 * B) = 40 :=
by
  sorry

end product_increased_five_times_l814_81408


namespace prices_and_subsidy_l814_81459

theorem prices_and_subsidy (total_cost : ℕ) (price_leather_jacket : ℕ) (price_sweater : ℕ) (subsidy_percentage : ℕ) 
  (leather_jacket_condition : price_leather_jacket = 5 * price_sweater + 600)
  (cost_condition : price_leather_jacket + price_sweater = total_cost)
  (total_sold : ℕ) (max_subsidy : ℕ) :
  (total_cost = 3000 ∧
   price_leather_jacket = 2600 ∧
   price_sweater = 400 ∧
   subsidy_percentage = 10) ∧ 
  ∃ a : ℕ, (2200 * a ≤ 50000 ∧ total_sold - a ≥ 128) :=
by
  sorry

end prices_and_subsidy_l814_81459


namespace pie_eating_contest_difference_l814_81419

-- Definition of given conditions
def num_students := 8
def emma_pies := 8
def sam_pies := 1

-- Statement to prove
theorem pie_eating_contest_difference :
  emma_pies - sam_pies = 7 :=
by
  -- Omitting the proof, as requested.
  sorry

end pie_eating_contest_difference_l814_81419


namespace max_value_of_g_on_interval_l814_81405

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g_on_interval : ∃ x : ℝ, (0 ≤ x ∧ x ≤ Real.sqrt 2) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ Real.sqrt 2) → g y ≤ g x) ∧ g x = 25 / 8 := by
  sorry

end max_value_of_g_on_interval_l814_81405


namespace problem_statement_l814_81424

theorem problem_statement (w x y z : ℕ) (h : 2^w * 3^x * 5^y * 7^z = 882) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
sorry

end problem_statement_l814_81424


namespace room_length_perimeter_ratio_l814_81427

theorem room_length_perimeter_ratio :
  ∀ (L W : ℕ), L = 19 → W = 11 → (L : ℚ) / (2 * (L + W)) = 19 / 60 := by
  intros L W hL hW
  sorry

end room_length_perimeter_ratio_l814_81427


namespace total_dots_not_visible_l814_81418

noncomputable def total_dots_on_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
noncomputable def total_dice : ℕ := 3
noncomputable def total_visible_faces : ℕ := 5

def visible_faces : List ℕ := [1, 2, 3, 3, 4]

theorem total_dots_not_visible :
  (total_dots_on_die * total_dice) - (visible_faces.sum) = 50 := by
  sorry

end total_dots_not_visible_l814_81418


namespace calculateRemainingMoney_l814_81497

def initialAmount : ℝ := 100
def actionFiguresCount : ℕ := 3
def actionFigureOriginalPrice : ℝ := 12
def actionFigureDiscount : ℝ := 0.25
def boardGamesCount : ℕ := 2
def boardGamePrice : ℝ := 11
def puzzleSetsCount : ℕ := 4
def puzzleSetPrice : ℝ := 6
def salesTax : ℝ := 0.05

theorem calculateRemainingMoney :
  initialAmount - (
    (actionFigureOriginalPrice * (1 - actionFigureDiscount) * actionFiguresCount) +
    (boardGamePrice * boardGamesCount) +
    (puzzleSetPrice * puzzleSetsCount)
  ) * (1 + salesTax) = 23.35 :=
by
  sorry

end calculateRemainingMoney_l814_81497


namespace trebled_resultant_is_correct_l814_81453

-- Let's define the initial number and the transformations
def initial_number := 17
def doubled (n : ℕ) := n * 2
def added_five (n : ℕ) := n + 5
def trebled (n : ℕ) := n * 3

-- Finally, we state the problem to prove
theorem trebled_resultant_is_correct : 
  trebled (added_five (doubled initial_number)) = 117 :=
by
  -- Here we just print sorry which means the proof is expected but not provided yet.
  sorry

end trebled_resultant_is_correct_l814_81453


namespace jackson_hermit_crabs_l814_81400

theorem jackson_hermit_crabs (H : ℕ) (total_souvenirs : ℕ) 
  (h1 : total_souvenirs = H + 3 * H + 6 * H) 
  (h2 : total_souvenirs = 450) : H = 45 :=
by {
  sorry
}

end jackson_hermit_crabs_l814_81400


namespace f_2019_value_l814_81451

noncomputable def B : Set ℚ := {q : ℚ | q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1}

noncomputable def g (x : ℚ) (h : x ∈ B) : ℚ :=
  1 - (2 / x)

noncomputable def f (x : ℚ) (h : x ∈ B) : ℝ :=
  sorry

theorem f_2019_value (h2019 : 2019 ∈ B) :
  f 2019 h2019 = Real.log ((2019 - 0.5) ^ 2 / 2018.5) :=
sorry

end f_2019_value_l814_81451


namespace product_of_differences_l814_81425

-- Define the context where x and y are real numbers
variables (x y : ℝ)

-- State the theorem to be proved
theorem product_of_differences (x y : ℝ) : 
  (-x + y) * (-x - y) = x^2 - y^2 :=
sorry

end product_of_differences_l814_81425


namespace possible_values_of_a_l814_81481

-- Define the sets P and Q under the conditions given
def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

-- Prove that if Q ⊆ P, then a ∈ {0, 1/3, -1/2}
theorem possible_values_of_a (a : ℝ) (h : Q a ⊆ P) : a = 0 ∨ a = 1/3 ∨ a = -1/2 :=
sorry

end possible_values_of_a_l814_81481


namespace find_p_plus_q_l814_81466

noncomputable def f (k p : ℚ) : ℚ := 5 * k^2 - 2 * k + p
noncomputable def g (k q : ℚ) : ℚ := 4 * k^2 + q * k - 6

theorem find_p_plus_q (p q : ℚ) (h : ∀ k : ℚ, f k p * g k q = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) :
  p + q = -3 :=
sorry

end find_p_plus_q_l814_81466


namespace john_reads_days_per_week_l814_81493

-- Define the conditions
def john_reads_books_per_day := 4
def total_books_read := 48
def total_weeks := 6

-- Theorem statement
theorem john_reads_days_per_week :
  (total_books_read / john_reads_books_per_day) / total_weeks = 2 :=
by
  sorry

end john_reads_days_per_week_l814_81493


namespace cooking_time_per_side_l814_81479

-- Defining the problem conditions
def total_guests : ℕ := 30
def guests_wanting_2_burgers : ℕ := total_guests / 2
def guests_wanting_1_burger : ℕ := total_guests / 2
def burgers_per_guest_2 : ℕ := 2
def burgers_per_guest_1 : ℕ := 1
def total_burgers : ℕ := guests_wanting_2_burgers * burgers_per_guest_2 + guests_wanting_1_burger * burgers_per_guest_1
def burgers_per_batch : ℕ := 5
def total_batches : ℕ := total_burgers / burgers_per_batch
def total_cooking_time : ℕ := 72
def time_per_batch : ℕ := total_cooking_time / total_batches
def sides_per_burger : ℕ := 2

-- the theorem to prove the desired cooking time per side
theorem cooking_time_per_side : (time_per_batch / sides_per_burger) = 4 := by {
    -- Here we would enter the proof steps, but this is omitted as per the instructions.
    sorry
}

end cooking_time_per_side_l814_81479


namespace royal_children_l814_81457

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l814_81457


namespace force_on_dam_l814_81409

noncomputable def calculate_force (ρ g a b h : ℝ) :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem force_on_dam :
  let ρ := 1000
  let g := 10
  let a := 6.0
  let b := 9.6
  let h := 4.0
  calculate_force ρ g a b h = 576000 :=
by sorry

end force_on_dam_l814_81409


namespace increasing_or_decreasing_subseq_l814_81438

theorem increasing_or_decreasing_subseq (a : Fin (m * n + 1) → ℝ) :
  ∃ (s : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (s i) ≤ a (s j)) ∨
  ∃ (t : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (t i) ≥ a (t j)) :=
sorry

end increasing_or_decreasing_subseq_l814_81438


namespace siblings_of_kevin_l814_81490

-- Define traits of each child
structure Child where
  eye_color : String
  hair_color : String

def Oliver : Child := ⟨"Green", "Red"⟩
def Kevin : Child := ⟨"Grey", "Brown"⟩
def Lily : Child := ⟨"Grey", "Red"⟩
def Emma : Child := ⟨"Green", "Brown"⟩
def Noah : Child := ⟨"Green", "Red"⟩
def Mia : Child := ⟨"Green", "Brown"⟩

-- Define the condition that siblings must share at least one trait
def share_at_least_one_trait (c1 c2 : Child) : Prop :=
  c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color

-- Prove that Emma and Mia are Kevin's siblings
theorem siblings_of_kevin : share_at_least_one_trait Kevin Emma ∧ share_at_least_one_trait Kevin Mia ∧ share_at_least_one_trait Emma Mia :=
  sorry

end siblings_of_kevin_l814_81490


namespace problem1_problem2_l814_81471

noncomputable def arcSin (x : ℝ) : ℝ := Real.arcsin x

theorem problem1 :
  (S : ℝ) = 3 * Real.pi + 2 * Real.sqrt 2 - 6 * arcSin (Real.sqrt (2 / 3)) :=
by
  sorry

theorem problem2 :
  (S : ℝ) = 3 * arcSin (Real.sqrt (2 / 3)) - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l814_81471


namespace required_amount_of_water_l814_81472

/-- 
Given:
- A solution of 12 ounces with 60% alcohol,
- A desired final concentration of 40% alcohol,

Prove:
- The required amount of water to add is 6 ounces.
-/
theorem required_amount_of_water 
    (original_volume : ℚ)
    (initial_concentration : ℚ)
    (desired_concentration : ℚ)
    (final_volume : ℚ)
    (amount_of_water : ℚ)
    (h1 : original_volume = 12)
    (h2 : initial_concentration = 0.6)
    (h3 : desired_concentration = 0.4)
    (h4 : final_volume = original_volume + amount_of_water)
    (h5 : amount_of_alcohol = original_volume * initial_concentration)
    (h6 : desired_amount_of_alcohol = final_volume * desired_concentration)
    (h7 : amount_of_alcohol = desired_amount_of_alcohol) : 
  amount_of_water = 6 := 
sorry

end required_amount_of_water_l814_81472


namespace popsicle_sticks_ratio_l814_81469

/-- Sam, Sid, and Steve brought popsicle sticks for their group activity in their Art class. Sid has twice as many popsicle sticks as Steve. If Steve has 12 popsicle sticks and they can use 108 popsicle sticks for their Art class activity, prove that the ratio of the number of popsicle sticks Sam has to the number Sid has is 3:1. -/
theorem popsicle_sticks_ratio (Sid Sam Steve : ℕ) 
    (h1 : Sid = 2 * Steve) 
    (h2 : Steve = 12) 
    (h3 : Sam + Sid + Steve = 108) : 
    Sam / Sid = 3 :=
by 
    -- Proof steps go here
    sorry

end popsicle_sticks_ratio_l814_81469


namespace faster_train_speed_l814_81416

theorem faster_train_speed
  (length_per_train : ℝ)
  (speed_slower_train : ℝ)
  (passing_time_secs : ℝ)
  (speed_faster_train : ℝ) :
  length_per_train = 80 / 1000 →
  speed_slower_train = 36 →
  passing_time_secs = 36 →
  speed_faster_train = 52 :=
by
  intro h_length_per_train h_speed_slower_train h_passing_time_secs
  -- Skipped steps would go here
  sorry

end faster_train_speed_l814_81416


namespace largest_mersenne_prime_less_than_500_l814_81488

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end largest_mersenne_prime_less_than_500_l814_81488


namespace find_const_s_l814_81437

noncomputable def g (x : ℝ) (a b c d : ℝ) := (x + 2 * a) * (x + 2 * b) * (x + 2 * c) * (x + 2 * d)

theorem find_const_s (a b c d : ℝ) (p q r s : ℝ) (h1 : 1 + p + q + r + s = 4041)
  (h2 : g 1 a b c d = 1 + p + q + r + s) :
  s = 3584 := 
sorry

end find_const_s_l814_81437


namespace next_perfect_square_l814_81414

theorem next_perfect_square (x : ℤ) (h : ∃ k : ℤ, x = k^2) : ∃ z : ℤ, z = x + 2 * Int.sqrt x + 1 :=
by
  sorry

end next_perfect_square_l814_81414


namespace product_increases_exactly_13_times_by_subtracting_3_l814_81480

theorem product_increases_exactly_13_times_by_subtracting_3 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    13 * (n1 * n2 * n3 * n4 * n5 * n6 * n7) =
      ((n1 - 3) * (n2 - 3) * (n3 - 3) * (n4 - 3) * (n5 - 3) * (n6 - 3) * (n7 - 3)) :=
sorry

end product_increases_exactly_13_times_by_subtracting_3_l814_81480


namespace count_multiples_5_or_7_not_35_l814_81474

def count_multiples_5 (n : ℕ) : ℕ := n / 5
def count_multiples_7 (n : ℕ) : ℕ := n / 7
def count_multiples_35 (n : ℕ) : ℕ := n / 35
def inclusion_exclusion (a b c : ℕ) : ℕ := a + b - c

theorem count_multiples_5_or_7_not_35 : 
  inclusion_exclusion (count_multiples_5 3000) (count_multiples_7 3000) (count_multiples_35 3000) = 943 :=
by
  sorry

end count_multiples_5_or_7_not_35_l814_81474


namespace sam_bought_9_cans_l814_81484

-- Definitions based on conditions
def spent_amount_dollars := 20 - 5.50
def spent_amount_cents := 1450 -- to avoid floating point precision issues we equate to given value in cents
def coupon_discount_cents := 5 * 25
def total_cost_no_discount := spent_amount_cents + coupon_discount_cents
def cost_per_can := 175

-- Main statement to prove
theorem sam_bought_9_cans : total_cost_no_discount / cost_per_can = 9 :=
by
  sorry -- Proof goes here

end sam_bought_9_cans_l814_81484


namespace three_numbers_less_or_equal_than_3_l814_81482

theorem three_numbers_less_or_equal_than_3 : 
  let a := 0.8
  let b := 0.5
  let c := 0.9
  (a ≤ 3) ∧ (b ≤ 3) ∧ (c ≤ 3) → 
  3 = 3 :=
by
  intros h
  sorry

end three_numbers_less_or_equal_than_3_l814_81482


namespace darryl_parts_cost_l814_81440

-- Define the conditions
def patent_cost : ℕ := 4500
def machine_price : ℕ := 180
def break_even_units : ℕ := 45
def total_revenue := break_even_units * machine_price

-- Define the theorem using the conditions
theorem darryl_parts_cost :
  ∃ (parts_cost : ℕ), parts_cost = total_revenue - patent_cost ∧ parts_cost = 3600 := by
  sorry

end darryl_parts_cost_l814_81440


namespace pigeon_percentage_l814_81460

-- Define the conditions
variables (total_birds : ℕ)
variables (geese swans herons ducks pigeons : ℕ)
variables (h1 : geese = total_birds * 20 / 100)
variables (h2 : swans = total_birds * 30 / 100)
variables (h3 : herons = total_birds * 15 / 100)
variables (h4 : ducks = total_birds * 25 / 100)
variables (h5 : pigeons = total_birds * 10 / 100)

-- Define the target problem
theorem pigeon_percentage (h_total : total_birds = 100) :
  (pigeons * 100 / (total_birds - swans)) = 14 :=
by sorry

end pigeon_percentage_l814_81460


namespace original_number_is_9_l814_81464

theorem original_number_is_9 (x : ℤ) (h : 10 * x = x + 81) : x = 9 :=
sorry

end original_number_is_9_l814_81464


namespace initial_mean_of_observations_l814_81428

theorem initial_mean_of_observations (M : ℚ) (h : 50 * M + 11 = 50 * 36.5) : M = 36.28 := 
by
  sorry

end initial_mean_of_observations_l814_81428


namespace playground_width_l814_81496

open Nat

theorem playground_width (garden_width playground_length perimeter_garden : ℕ) (garden_area_eq_playground_area : Bool) :
  garden_width = 8 →
  playground_length = 16 →
  perimeter_garden = 64 →
  garden_area_eq_playground_area →
  ∃ (W : ℕ), W = 12 :=
by
  intros h_t1 h_t2 h_t3 h_t4
  sorry

end playground_width_l814_81496


namespace box_volume_of_pyramid_l814_81403

/-- A theorem to prove the volume of the smallest cube-shaped box that can house the given rectangular pyramid. -/
theorem box_volume_of_pyramid :
  (∀ (h l w : ℕ), h = 15 ∧ l = 8 ∧ w = 12 → (∀ (v : ℕ), v = (max h (max l w)) ^ 3 → v = 3375)) :=
by
  intros h l w h_condition v v_def
  sorry

end box_volume_of_pyramid_l814_81403


namespace complex_equilateral_triangle_expression_l814_81415

noncomputable def omega : ℂ :=
  Complex.exp (Complex.I * 2 * Real.pi / 3)

def is_root_of_quadratic (z : ℂ) (a b : ℂ) : Prop :=
  z^2 + a * z + b = 0

theorem complex_equilateral_triangle_expression (z1 z2 a b : ℂ) (h1 : is_root_of_quadratic z1 a b) 
  (h2 : is_root_of_quadratic z2 a b) (h3 : z2 = omega * z1) : a^2 / b = 1 := by
  sorry

end complex_equilateral_triangle_expression_l814_81415


namespace intersection_of_M_and_N_l814_81462

theorem intersection_of_M_and_N (x : ℝ) :
  {x | x > 1} ∩ {x | x^2 - 2 * x < 0} = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l814_81462


namespace sugar_merchant_profit_l814_81443

theorem sugar_merchant_profit 
    (total_sugar : ℕ)
    (sold_at_18 : ℕ)
    (remain_sugar : ℕ)
    (whole_profit : ℕ)
    (profit_18 : ℕ)
    (rem_profit_percent : ℕ) :
    total_sugar = 1000 →
    sold_at_18 = 600 →
    remain_sugar = total_sugar - sold_at_18 →
    whole_profit = 14 →
    profit_18 = 18 →
    (600 * profit_18 / 100) + (remain_sugar * rem_profit_percent / 100) = 
    (total_sugar * whole_profit / 100) →
    rem_profit_percent = 80 :=
by
    sorry

end sugar_merchant_profit_l814_81443


namespace johns_sixth_quiz_score_l814_81404

theorem johns_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (mean : ℕ) (n : ℕ) :
  s1 = 86 ∧ s2 = 91 ∧ s3 = 83 ∧ s4 = 88 ∧ s5 = 97 ∧ mean = 90 ∧ n = 6 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / n = mean ∧ s6 = 95 :=
by
  intro h
  obtain ⟨hs1, hs2, hs3, hs4, hs5, hmean, hn⟩ := h
  have htotal : s1 + s2 + s3 + s4 + s5 + 95 = 540 := by sorry
  have hmean_eq : (s1 + s2 + s3 + s4 + s5 + 95) / n = mean := by sorry
  exact ⟨95, hmean_eq, rfl⟩

end johns_sixth_quiz_score_l814_81404


namespace log_expression_in_terms_of_a_l814_81454

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

variable (a : ℝ) (h : a = log3 2)

theorem log_expression_in_terms_of_a : log3 8 - 2 * log3 6 = a - 2 :=
by
  sorry

end log_expression_in_terms_of_a_l814_81454


namespace quadratic_to_vertex_form_l814_81452

theorem quadratic_to_vertex_form:
  ∀ (x : ℝ), (x^2 - 4 * x + 3 = (x - 2)^2 - 1) :=
by
  sorry

end quadratic_to_vertex_form_l814_81452


namespace rectangle_width_l814_81468

theorem rectangle_width (L W : ℕ)
  (h1 : W = L + 3)
  (h2 : 2 * L + 2 * W = 54) :
  W = 15 :=
by
  sorry

end rectangle_width_l814_81468


namespace apple_cost_price_l814_81486

theorem apple_cost_price (SP : ℝ) (loss_ratio : ℝ) (CP : ℝ) (h1 : SP = 18) (h2 : loss_ratio = 1/6) (h3 : SP = CP - loss_ratio * CP) : CP = 21.6 :=
by
  sorry

end apple_cost_price_l814_81486


namespace average_percentage_increase_is_correct_l814_81473

def initial_prices : List ℝ := [300, 450, 600]
def price_increases : List ℝ := [0.10, 0.15, 0.20]

noncomputable def total_original_price : ℝ :=
  initial_prices.sum

noncomputable def total_new_price : ℝ :=
  (List.zipWith (λ p i => p * (1 + i)) initial_prices price_increases).sum

noncomputable def total_price_increase : ℝ :=
  total_new_price - total_original_price

noncomputable def average_percentage_increase : ℝ :=
  (total_price_increase / total_original_price) * 100

theorem average_percentage_increase_is_correct :
  average_percentage_increase = 16.11 := by
  sorry

end average_percentage_increase_is_correct_l814_81473


namespace number_of_pumps_l814_81433

theorem number_of_pumps (P : ℕ) : 
  (P * 8 * 2 = 8 * 6) → P = 3 :=
by
  intro h
  sorry

end number_of_pumps_l814_81433


namespace expand_binomials_l814_81441

variable (x y : ℝ)

theorem expand_binomials : 
  (3 * x - 2) * (2 * x + 4 * y + 1) = 6 * x^2 + 12 * x * y - x - 8 * y - 2 :=
by
  sorry

end expand_binomials_l814_81441


namespace employed_males_population_percentage_l814_81449

-- Define the conditions of the problem
variables (P : Type) (population : ℝ) (employed_population : ℝ) (employed_females : ℝ)

-- Assume total population is 100
def total_population : ℝ := 100

-- 70 percent of the population are employed
def employed_population_percentage : ℝ := total_population * 0.70

-- 70 percent of the employed people are females
def employed_females_percentage : ℝ := employed_population_percentage * 0.70

-- 21 percent of the population are employed males
def employed_males_percentage : ℝ := 21

-- Main statement to be proven
theorem employed_males_population_percentage :
  employed_males_percentage = ((employed_population_percentage - employed_females_percentage) / total_population) * 100 :=
sorry

end employed_males_population_percentage_l814_81449


namespace cannot_bisect_abs_function_l814_81406

theorem cannot_bisect_abs_function 
  (f : ℝ → ℝ)
  (hf1 : ∀ x, f x = |x|) :
  ¬ (∃ a b, a < b ∧ f a * f b < 0) :=
by
  sorry

end cannot_bisect_abs_function_l814_81406


namespace division_problem_l814_81448

theorem division_problem (n : ℕ) (h : n / 6 = 209) : n = 1254 := 
sorry

end division_problem_l814_81448


namespace triangle_side_length_l814_81436

theorem triangle_side_length (y z : ℝ) (cos_Y_minus_Z : ℝ) (h_y : y = 7) (h_z : z = 6) (h_cos : cos_Y_minus_Z = 17 / 18) : 
  ∃ x : ℝ, x = Real.sqrt 65 :=
by
  sorry

end triangle_side_length_l814_81436


namespace product_of_geometric_sequence_l814_81461

theorem product_of_geometric_sequence (x y z : ℝ) 
  (h_seq : ∃ r, x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) : 
  1 * x * y * z * 4 = 32 :=
by
  sorry

end product_of_geometric_sequence_l814_81461


namespace number_of_happy_configurations_is_odd_l814_81411

def S (m n : ℕ) := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 2 * m ∧ 1 ≤ p.2 ∧ p.2 ≤ 2 * n}

def happy_configurations (m n : ℕ) : ℕ := 
  sorry -- definition of the number of happy configurations is abstracted for this statement.

theorem number_of_happy_configurations_is_odd (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  happy_configurations m n % 2 = 1 := 
sorry

end number_of_happy_configurations_is_odd_l814_81411


namespace frog_jumps_further_l814_81430

-- Definitions according to conditions
def grasshopper_jump : ℕ := 36
def frog_jump : ℕ := 53

-- Theorem: The frog jumped 17 inches farther than the grasshopper
theorem frog_jumps_further (g_jump f_jump : ℕ) (h1 : g_jump = grasshopper_jump) (h2 : f_jump = frog_jump) :
  f_jump - g_jump = 17 :=
by
  -- Proof is skipped in this statement
  sorry

end frog_jumps_further_l814_81430


namespace no_rational_root_l814_81413

theorem no_rational_root (x : ℚ) : 3 * x^4 - 2 * x^3 - 8 * x^2 + x + 1 ≠ 0 := 
by
  sorry

end no_rational_root_l814_81413


namespace parallelepiped_face_areas_l814_81465

theorem parallelepiped_face_areas
    (h₁ : ℝ := 2)  -- height corresponding to face area x
    (h₂ : ℝ := 3)  -- height corresponding to face area y
    (h₃ : ℝ := 4)  -- height corresponding to face area z
    (total_surface_area : ℝ := 36) : 
    ∃ (x y z : ℝ), 
    2 * x + 2 * y + 2 * z = total_surface_area ∧
    (∃ V : ℝ, V = h₁ * x ∧ V = h₂ * y ∧ V = h₃ * z) ∧
    x = 108 / 13 ∧ y = 72 / 13 ∧ z = 54 / 13 := 
by 
  sorry

end parallelepiped_face_areas_l814_81465


namespace solve_cryptarithm_l814_81476

-- Declare non-computable constants for the letters
variables {A B C : ℕ}

-- Conditions from the problem
-- Different letters represent different digits
axiom diff_digits : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- A ≠ 0
axiom A_nonzero : A ≠ 0

-- Given cryptarithm equation
axiom cryptarithm_eq : 100 * C + 10 * B + A + 100 * A + 10 * A + A = 100 * B + A

-- The proof to show the correct values
theorem solve_cryptarithm : A = 5 ∧ B = 9 ∧ C = 3 :=
sorry

end solve_cryptarithm_l814_81476


namespace train_length_l814_81435

theorem train_length (V L : ℝ) (h₁ : V = L / 18) (h₂ : V = (L + 200) / 30) : L = 300 :=
by
  sorry

end train_length_l814_81435


namespace gravitational_field_height_depth_equality_l814_81483

theorem gravitational_field_height_depth_equality
  (R G ρ : ℝ) (hR : R > 0) :
  ∃ x : ℝ, x = R * ((-1 + Real.sqrt 5) / 2) ∧
  (G * ρ * ((4 / 3) * Real.pi * R^3) / (R + x)^2 = G * ρ * ((4 / 3) * Real.pi * (R - x)^3) / (R - x)^2) :=
by
  sorry

end gravitational_field_height_depth_equality_l814_81483


namespace apples_in_baskets_l814_81450

theorem apples_in_baskets (total_apples : ℕ) (first_basket : ℕ) (increase : ℕ) (baskets : ℕ) :
  total_apples = 495 ∧ first_basket = 25 ∧ increase = 2 ∧
  (total_apples = (baskets / 2) * (2 * first_basket + (baskets - 1) * increase)) -> baskets = 13 :=
by sorry

end apples_in_baskets_l814_81450


namespace peach_tree_average_production_l814_81426

-- Definitions derived from the conditions
def num_apple_trees : ℕ := 30
def kg_per_apple_tree : ℕ := 150
def num_peach_trees : ℕ := 45
def total_mass_fruit : ℕ := 7425

-- Main Statement to be proven
theorem peach_tree_average_production : 
  (total_mass_fruit - (num_apple_trees * kg_per_apple_tree)) = (num_peach_trees * 65) :=
by
  sorry

end peach_tree_average_production_l814_81426


namespace necessary_but_not_sufficient_condition_l814_81487

variable {a : ℕ → ℤ}

noncomputable def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
∀ (m n k : ℕ), a m * a k = a n * a (m + k - n)

noncomputable def is_root_of_quadratic (x y : ℤ) : Prop :=
x^2 + 3*x + 1 = 0 ∧ y^2 + 3*y + 1 = 0

theorem necessary_but_not_sufficient_condition 
  (a : ℕ → ℤ)
  (hgeo : is_geometric_sequence a)
  (hroots : is_root_of_quadratic (a 4) (a 12)) :
  a 8 = -1 ↔ (∃ x y : ℤ, is_root_of_quadratic x y ∧ x + y = -3 ∧ x * y = 1) :=
sorry

end necessary_but_not_sufficient_condition_l814_81487


namespace sqrt_diff_eq_neg_sixteen_l814_81489

theorem sqrt_diff_eq_neg_sixteen : 
  (Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2)) = -16 := 
  sorry

end sqrt_diff_eq_neg_sixteen_l814_81489


namespace petrov_vasechkin_boards_l814_81491

theorem petrov_vasechkin_boards:
  ∃ n : ℕ, 
  (∃ x y : ℕ, 2 * x + 3 * y = 87 ∧ x + y = n) ∧ 
  (∃ u v : ℕ, 3 * u + 5 * v = 94 ∧ u + v = n) ∧ 
  n = 30 := 
sorry

end petrov_vasechkin_boards_l814_81491


namespace find_two_digit_number_t_l814_81498

theorem find_two_digit_number_t (t : ℕ) (ht1 : 10 ≤ t) (ht2 : t ≤ 99) (ht3 : 13 * t % 100 = 52) : t = 12 := 
sorry

end find_two_digit_number_t_l814_81498


namespace div_relation_l814_81456

variable {a b c : ℚ}

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 2/5) : c / a = 5/6 := by
  sorry

end div_relation_l814_81456


namespace fraction_compare_l814_81420

theorem fraction_compare (a b c d e : ℚ) : 
  a = 0.3333333 → 
  b = 1 / (3 * 10^6) →
  ∃ x : ℚ, 
  x = 1 / 3 ∧ 
  (x > a + d ∧ 
   x = a + b ∧
   d = b ∧
   d = -1 / (3 * 10^6)) := 
  sorry

end fraction_compare_l814_81420


namespace not_sunny_prob_l814_81412

theorem not_sunny_prob (P_sunny : ℚ) (h : P_sunny = 5/7) : 1 - P_sunny = 2/7 :=
by sorry

end not_sunny_prob_l814_81412


namespace part1_part2_l814_81421

noncomputable def a_n (n : ℕ) : ℕ :=
  2^(n - 1)

noncomputable def b_n (n : ℕ) : ℕ :=
  2 * n

noncomputable def S_n (n : ℕ) : ℕ :=
  n^2 + n

theorem part1 (n : ℕ) : 
  S_n n = n^2 + n := 
sorry

noncomputable def C_n (n : ℕ) : ℚ :=
  (n^2 + n) / 2^(n - 1)

theorem part2 (n : ℕ) (k : ℕ) (k_gt_0 : 0 < k) : 
  (∀ n, C_n n ≤ C_n k) ↔ (k = 2 ∨ k = 3) :=
sorry

end part1_part2_l814_81421


namespace problem_statement_l814_81494

noncomputable def decimalPartSqrtFive : ℝ := Real.sqrt 5 - 2
def integerPartSqrtThirteen : ℕ := 3

theorem problem_statement :
  decimalPartSqrtFive + integerPartSqrtThirteen - Real.sqrt 5 = 1 :=
by
  sorry

end problem_statement_l814_81494


namespace y1_less_than_y2_l814_81499

noncomputable def y1 : ℝ := 2 * (-5) + 1
noncomputable def y2 : ℝ := 2 * 3 + 1

theorem y1_less_than_y2 : y1 < y2 := by
  sorry

end y1_less_than_y2_l814_81499


namespace find_a20_l814_81455

variable {a : ℕ → ℤ}
variable {d : ℤ}
variable {a_1 : ℤ}

def isArithmeticSeq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def formsGeomSeq (a1 a3 a4 : ℤ) : Prop :=
  (a3 - a1)^2 = a1 * (a4 - a1)

theorem find_a20 (h1 : isArithmeticSeq a (-2))
                 (h2 : formsGeomSeq a_1 (a_1 + 2*(-2)) (a_1 + 3*(-2)))
                 (ha1 : a_1 = 8) :
  a 20 = -30 :=
by
  sorry

end find_a20_l814_81455


namespace intersection_complement_equivalence_l814_81444

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equivalence :
  ((U \ M) ∩ N) = {3} := by
  sorry

end intersection_complement_equivalence_l814_81444


namespace find_a_l814_81447

def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

theorem find_a (a : ℝ) (h : f (f a) = f 9 + 1) : a = -1/4 := 
by 
  sorry

end find_a_l814_81447


namespace crayons_eaten_correct_l814_81442

variable (initial_crayons final_crayons : ℕ)

def crayonsEaten (initial_crayons final_crayons : ℕ) : ℕ :=
  initial_crayons - final_crayons

theorem crayons_eaten_correct : crayonsEaten 87 80 = 7 :=
  by
  sorry

end crayons_eaten_correct_l814_81442


namespace number_of_solutions_l814_81434

open Real

-- Define the main equation in terms of absolute values 
def equation (x : ℝ) : Prop := abs (x - abs (2 * x + 1)) = 3

-- Prove that there are exactly 2 distinct solutions to the equation
theorem number_of_solutions : 
  ∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂ :=
sorry

end number_of_solutions_l814_81434


namespace find_x_l814_81410

noncomputable def f (x : ℝ) : ℝ := 5 * x^3 - 3

theorem find_x (x : ℝ) : (f⁻¹ (-2) = x) → x = -43 := by
  sorry

end find_x_l814_81410


namespace combined_average_score_l814_81446

theorem combined_average_score (M A : ℝ) (m a : ℝ)
  (hM : M = 78) (hA : A = 85) (h_ratio : m = 2 * a / 3) :
  (78 * (2 * a / 3) + 85 * a) / ((2 * a / 3) + a) = 82 := by
  sorry

end combined_average_score_l814_81446


namespace boards_tested_l814_81432

-- Define the initial conditions and problem
def total_thumbtacks : ℕ := 450
def thumbtacks_remaining_each_can : ℕ := 30
def initial_thumbtacks_each_can := total_thumbtacks / 3
def thumbtacks_used_each_can := initial_thumbtacks_each_can - thumbtacks_remaining_each_can
def total_thumbtacks_used := thumbtacks_used_each_can * 3
def thumbtacks_per_board := 3

-- Define the proposition to prove 
theorem boards_tested (B : ℕ) : 
  (B = total_thumbtacks_used / thumbtacks_per_board) → B = 120 :=
by
  -- Proof skipped with sorry
  sorry

end boards_tested_l814_81432


namespace original_length_of_wood_l814_81422

theorem original_length_of_wood (s cl ol : ℝ) (h1 : s = 2.3) (h2 : cl = 6.6) (h3 : ol = cl + s) : 
  ol = 8.9 := 
by 
  sorry

end original_length_of_wood_l814_81422


namespace wage_of_one_man_l814_81477

/-- Proof that the wage of one man is Rs. 24 given the conditions. -/
theorem wage_of_one_man (M W_w B_w : ℕ) (H1 : 120 = 5 * M + W_w * 5 + B_w * 8) 
  (H2 : 5 * M = W_w * 5) (H3 : W_w * 5 = B_w * 8) : M = 24 :=
by
  sorry

end wage_of_one_man_l814_81477


namespace difference_of_squares_example_l814_81485

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 123) (h2 : b = 23) : a^2 - b^2 = 14600 :=
by
  rw [h1, h2]
  sorry

end difference_of_squares_example_l814_81485


namespace third_term_is_18_l814_81475

-- Define the first term and the common ratio
def a_1 : ℕ := 2
def q : ℕ := 3

-- Define the function for the nth term of an arithmetic-geometric sequence
def a_n (n : ℕ) : ℕ :=
  a_1 * q^(n-1)

-- Prove that the third term is 18
theorem third_term_is_18 : a_n 3 = 18 := by
  sorry

end third_term_is_18_l814_81475


namespace minimum_n_is_835_l814_81463

def problem_statement : Prop :=
  ∀ (S : Finset ℕ), S.card = 835 → (∀ (T : Finset ℕ), T ⊆ S → T.card = 4 →
    ∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + 2 * b + 3 * c = d)

theorem minimum_n_is_835 : problem_statement :=
sorry

end minimum_n_is_835_l814_81463


namespace smallest_integer_in_set_l814_81470

def avg_seven_consecutive_integers (n : ℤ) : ℤ :=
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7

theorem smallest_integer_in_set : ∃ (n : ℤ), n = 0 ∧ (n + 6 < 3 * avg_seven_consecutive_integers n) :=
by
  sorry

end smallest_integer_in_set_l814_81470


namespace michael_lap_time_l814_81401

theorem michael_lap_time :
  ∃ T : ℝ, (∀ D : ℝ, D = 45 → (9 * T = 10 * D) → T = 50) :=
by
  sorry

end michael_lap_time_l814_81401


namespace stratified_sampling_grade10_students_l814_81407

-- Definitions based on the given problem
def total_students := 900
def grade10_students := 300
def sample_size := 45

-- Calculation of the number of Grade 10 students in the sample
theorem stratified_sampling_grade10_students : (grade10_students * sample_size) / total_students = 15 := by
  sorry

end stratified_sampling_grade10_students_l814_81407


namespace articles_production_l814_81423

theorem articles_production (x y : ℕ) (e : ℝ) :
  (x * x * x * e / x = x^2 * e) → (y * (y + 2) * y * (e / x) = (e * y * (y^2 + 2 * y)) / x) :=
by 
  sorry

end articles_production_l814_81423


namespace amount_spent_on_shorts_l814_81467

def amount_spent_on_shirt := 12.14
def amount_spent_on_jacket := 7.43
def total_amount_spent_on_clothes := 33.56

theorem amount_spent_on_shorts : total_amount_spent_on_clothes - amount_spent_on_shirt - amount_spent_on_jacket = 13.99 :=
by
  sorry

end amount_spent_on_shorts_l814_81467


namespace three_digit_cubes_divisible_by_eight_l814_81478

theorem three_digit_cubes_divisible_by_eight :
  (∃ n1 n2 : ℕ, 100 ≤ n1 ∧ n1 < 1000 ∧ n2 < n1 ∧ 100 ≤ n2 ∧ n2 < 1000 ∧
  (∃ m1 m2 : ℕ, 2 ≤ m1 ∧ 2 ≤ m2 ∧ n1 = 8 * m1^3  ∧ n2 = 8 * m2^3)) :=
sorry

end three_digit_cubes_divisible_by_eight_l814_81478


namespace complex_number_solution_l814_81431

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : i * z = 1) : z = -i :=
by
  -- Mathematical proof will be here
  sorry

end complex_number_solution_l814_81431


namespace solve_quadratic_l814_81429

theorem solve_quadratic (x : ℝ) (h : x^2 - 6*x + 8 = 0) : x = 2 ∨ x = 4 :=
sorry

end solve_quadratic_l814_81429


namespace total_pens_bought_l814_81445

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 := 
sorry

end total_pens_bought_l814_81445


namespace base_10_uniqueness_l814_81458

theorem base_10_uniqueness : 
  (∀ a : ℕ, 12 = 3 * 4 ∧ 56 = 7 * 8 ↔ (a * b + (a + 1) = (a + 2) * (a + 3))) → b = 10 :=
sorry

end base_10_uniqueness_l814_81458


namespace negation_of_existential_square_inequality_l814_81495

theorem negation_of_existential_square_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_square_inequality_l814_81495


namespace evaluate_expression_l814_81492

-- Definitions for conditions
def x := (1 / 4 : ℚ)
def y := (1 / 2 : ℚ)
def z := (3 : ℚ)

-- Statement of the problem
theorem evaluate_expression : 
  4 * (x^3 * y^2 * z^2) = 9 / 64 :=
by
  sorry

end evaluate_expression_l814_81492


namespace total_employees_l814_81402

-- Defining the number of part-time and full-time employees
def p : ℕ := 2041
def f : ℕ := 63093

-- Statement that the total number of employees is the sum of part-time and full-time employees
theorem total_employees : p + f = 65134 :=
by
  -- Use Lean's built-in arithmetic to calculate the sum
  rfl

end total_employees_l814_81402


namespace total_money_collected_l814_81417

theorem total_money_collected (attendees : ℕ) (reserved_price unreserved_price : ℝ) (reserved_sold unreserved_sold : ℕ)
  (h_attendees : attendees = 1096)
  (h_reserved_price : reserved_price = 25.00)
  (h_unreserved_price : unreserved_price = 20.00)
  (h_reserved_sold : reserved_sold = 246)
  (h_unreserved_sold : unreserved_sold = 246) :
  (reserved_price * reserved_sold + unreserved_price * unreserved_sold) = 11070.00 :=
by
  sorry

end total_money_collected_l814_81417
