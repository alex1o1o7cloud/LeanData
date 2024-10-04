import Mathlib

namespace ratio_w_y_l132_132288

theorem ratio_w_y 
  (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
sorry

end ratio_w_y_l132_132288


namespace initial_men_in_garrison_l132_132987

variable (x : ℕ)

theorem initial_men_in_garrison (h1 : x * 65 = x * 50 + (x + 3000) * 20) : x = 2000 :=
  sorry

end initial_men_in_garrison_l132_132987


namespace evaluate_ceiling_sum_l132_132218

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l132_132218


namespace simple_interest_rate_l132_132807

variable (P : ℝ) (A : ℝ) (T : ℝ)

theorem simple_interest_rate (h1 : P = 35) (h2 : A = 36.4) (h3 : T = 1) :
  (A - P) / (P * T) = 0.04 :=
by
  sorry

end simple_interest_rate_l132_132807


namespace arithmetic_sequence_k_l132_132921

theorem arithmetic_sequence_k :
  ∀ (a : ℕ → ℤ) (d : ℤ) (k : ℕ),
  d ≠ 0 →
  (∀ n : ℕ, a n = a 0 + n * d) →
  a 0 = 0 →
  a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 →
  k = 22 :=
by
  intros a d k hdnz h_arith h_a1_zero h_ak_sum
  sorry

end arithmetic_sequence_k_l132_132921


namespace train_length_l132_132301

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_ms : ℝ) (distance_m : ℝ)
  (h1 : speed_km_hr = 90)
  (h2 : time_seconds = 9)
  (h3 : speed_ms = speed_km_hr * (1000 / 3600))
  (h4 : distance_m = speed_ms * time_seconds) :
  distance_m = 225 :=
by
  sorry

end train_length_l132_132301


namespace simplify_expression_l132_132816

theorem simplify_expression (a b : ℤ) : 
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b :=
by
  sorry

end simplify_expression_l132_132816


namespace ratio_of_squares_l132_132250

noncomputable def right_triangle : Type := sorry -- Placeholder for the right triangle type

variables (a b c : ℕ)

-- Given lengths of the triangle sides
def triangle_sides (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2

-- Define x and y based on the conditions in the problem
def side_length_square_x (x : ℝ) : Prop :=
  0 < x ∧ x < 5 ∧ x < 12

def side_length_square_y (y : ℝ) : Prop :=
  0 < y ∧ y < 13

-- The main theorem to prove
theorem ratio_of_squares (x y : ℝ) :
  ∀ a b c, triangle_sides a b c →
  side_length_square_x x →
  side_length_square_y y →
  x / y = 1 :=
sorry

end ratio_of_squares_l132_132250


namespace evaluate_expression_l132_132215

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l132_132215


namespace like_terms_exponents_l132_132776

theorem like_terms_exponents (n m : ℕ) (h1 : n + 2 = 3) (h2 : 2 * m - 1 = 3) : n = 1 ∧ m = 2 :=
by sorry

end like_terms_exponents_l132_132776


namespace selected_numbers_in_range_l132_132717

noncomputable def systematic_sampling (n_students selected_students interval_num start_num n : ℕ) : ℕ :=
  start_num + interval_num * (n - 1)

theorem selected_numbers_in_range (x : ℕ) :
  (500 = 500) ∧ (50 = 50) ∧ (10 = 500 / 50) ∧ (6 ∈ {y : ℕ | 1 ≤ y ∧ y ≤ 10}) ∧ (125 ≤ x ∧ x ≤ 140) → 
  (x = systematic_sampling 500 50 10 6 13 ∨ x = systematic_sampling 500 50 10 6 14) :=
by
  sorry

end selected_numbers_in_range_l132_132717


namespace bob_grade_is_35_l132_132113

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l132_132113


namespace min_value_expression_l132_132810

open Real

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_condition : a * b * c = 1) :
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 36 :=
by
  sorry

end min_value_expression_l132_132810


namespace sum_of_edges_of_geometric_progression_solid_l132_132864

theorem sum_of_edges_of_geometric_progression_solid
  (a : ℝ)
  (r : ℝ)
  (volume_eq : a^3 = 512)
  (surface_eq : 2 * (64 / r + 64 * r + 64) = 352)
  (r_value : r = 1.25 ∨ r = 0.8) :
  4 * (8 / r + 8 + 8 * r) = 97.6 := by
  sorry

end sum_of_edges_of_geometric_progression_solid_l132_132864


namespace number_of_multiples_840_in_range_l132_132091

theorem number_of_multiples_840_in_range :
  ∃ n, n = 1 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 2500 ∧ (840 ∣ x) → x = 1680 :=
by
  sorry

end number_of_multiples_840_in_range_l132_132091


namespace rahim_books_bought_l132_132138

theorem rahim_books_bought (x : ℕ) 
  (first_shop_cost second_shop_cost total_books : ℕ)
  (avg_price total_spent : ℕ)
  (h1 : first_shop_cost = 1500)
  (h2 : second_shop_cost = 340)
  (h3 : total_books = x + 60)
  (h4 : avg_price = 16)
  (h5 : total_spent = first_shop_cost + second_shop_cost)
  (h6 : avg_price = total_spent / total_books) :
  x = 55 :=
by
  sorry

end rahim_books_bought_l132_132138


namespace muffin_half_as_expensive_as_banana_l132_132819

-- Define Susie's expenditure in terms of muffin cost (m) and banana cost (b)
def susie_expenditure (m b : ℝ) : ℝ := 5 * m + 2 * b

-- Define Calvin's expenditure as three times Susie's expenditure
def calvin_expenditure_via_susie (m b : ℝ) : ℝ := 3 * (susie_expenditure m b)

-- Define Calvin's direct expenditure on muffins and bananas
def calvin_direct_expenditure (m b : ℝ) : ℝ := 3 * m + 12 * b

-- Formulate the theorem stating the relationship between muffin and banana costs
theorem muffin_half_as_expensive_as_banana (m b : ℝ) 
  (h₁ : susie_expenditure m b = 5 * m + 2 * b)
  (h₂ : calvin_expenditure_via_susie m b = calvin_direct_expenditure m b) : 
  m = (1/2) * b := 
by {
  -- These conditions automatically fulfill the given problem requirements.
  sorry
}

end muffin_half_as_expensive_as_banana_l132_132819


namespace each_interior_angle_of_regular_octagon_l132_132401

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l132_132401


namespace meaningful_fraction_l132_132826

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by
  sorry

end meaningful_fraction_l132_132826


namespace cos_double_angle_tan_sum_angles_l132_132899

variable (α β : ℝ)
variable (α_acute : 0 < α ∧ α < π / 2)
variable (β_acute : 0 < β ∧ β < π / 2)
variable (tan_alpha : Real.tan α = 4 / 3)
variable (sin_alpha_minus_beta : Real.sin (α - β) = - (Real.sqrt 5) / 5)

/- Prove that cos 2α = -7/25 given the conditions -/
theorem cos_double_angle :
  Real.cos (2 * α) = -7 / 25 :=
by
  sorry

/- Prove that tan (α + β) = -41/38 given the conditions -/
theorem tan_sum_angles :
  Real.tan (α + β) = -41 / 38 :=
by
  sorry

end cos_double_angle_tan_sum_angles_l132_132899


namespace greatest_value_a4_b4_l132_132494

theorem greatest_value_a4_b4
    (a b : Nat → ℝ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + a 1)
    (h_geom_seq : ∀ n, b (n + 1) = b n * b 1)
    (h_a1b1 : a 1 * b 1 = 20)
    (h_a2b2 : a 2 * b 2 = 19)
    (h_a3b3 : a 3 * b 3 = 14) :
    ∃ m : ℝ, a 4 * b 4 = 8 ∧ ∀ x, a 4 * b 4 ≤ x -> x = 8 := by
  sorry

end greatest_value_a4_b4_l132_132494


namespace angle_measure_l132_132589

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132589


namespace carsProducedInEurope_l132_132196

-- Definitions of the conditions
def carsProducedInNorthAmerica : ℕ := 3884
def totalCarsProduced : ℕ := 6755

-- Theorem statement
theorem carsProducedInEurope : ∃ (carsProducedInEurope : ℕ), totalCarsProduced = carsProducedInNorthAmerica + carsProducedInEurope ∧ carsProducedInEurope = 2871 := by
  sorry

end carsProducedInEurope_l132_132196


namespace arrange_plants_in_a_row_l132_132726

-- Definitions for the conditions
def basil_plants : ℕ := 5 -- Number of basil plants
def tomato_plants : ℕ := 4 -- Number of tomato plants

-- Theorem statement asserting the number of ways to arrange the plants
theorem arrange_plants_in_a_row : 
  let total_items := basil_plants + 1,
      ways_to_arrange_total_items := Nat.factorial total_items,
      ways_to_arrange_tomato_group := Nat.factorial tomato_plants in
  (ways_to_arrange_total_items * ways_to_arrange_tomato_group) = 17280 := 
by
  sorry

end arrange_plants_in_a_row_l132_132726


namespace find_different_mass_part_l132_132133

-- Definitions for the parts a1, a2, a3, a4 and their masses
variable {α : Type}
variables (a₁ a₂ a₃ a₄ : α)
variable [LinearOrder α]

-- Definition of the problem conditions
def different_mass_part (a₁ a₂ a₃ a₄ : α) : Prop :=
  (a₁ ≠ a₂ ∨ a₁ ≠ a₃ ∨ a₁ ≠ a₄ ∨ a₂ ≠ a₃ ∨ a₂ ≠ a₄ ∨ a₃ ≠ a₄)

-- Theorem statement assuming we can identify the differing part using two weighings on a pan balance
theorem find_different_mass_part (h : different_mass_part a₁ a₂ a₃ a₄) :
  ∃ (part : α), part = a₁ ∨ part = a₂ ∨ part = a₃ ∨ part = a₄ :=
sorry

end find_different_mass_part_l132_132133


namespace each_interior_angle_of_regular_octagon_l132_132402

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l132_132402


namespace age_difference_l132_132314

-- Define the present age of the son as a constant
def S : ℕ := 22

-- Define the equation given by the problem
noncomputable def age_relation (M : ℕ) : Prop :=
  M + 2 = 2 * (S + 2)

-- The theorem to prove the man is 24 years older than his son
theorem age_difference (M : ℕ) (h_rel : age_relation M) : M - S = 24 :=
by {
  sorry
}

end age_difference_l132_132314


namespace find_third_number_l132_132982

theorem find_third_number (x : ℕ) (h : 3 * 16 + 3 * 17 + 3 * x + 11 = 170) : x = 20 := by
  sorry

end find_third_number_l132_132982


namespace angle_complement_supplement_l132_132625

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132625


namespace total_age_of_wines_l132_132821

theorem total_age_of_wines (age_carlo_rosi : ℕ) (age_franzia : ℕ) (age_twin_valley : ℕ) 
    (h1 : age_carlo_rosi = 40) (h2 : age_franzia = 3 * age_carlo_rosi) (h3 : age_carlo_rosi = 4 * age_twin_valley) : 
    age_franzia + age_carlo_rosi + age_twin_valley = 170 := 
by
    sorry

end total_age_of_wines_l132_132821


namespace initial_bird_count_l132_132104

theorem initial_bird_count (B : ℕ) (h₁ : B + 7 = 12) : B = 5 :=
by
  sorry

end initial_bird_count_l132_132104


namespace complement_intersection_eq_l132_132081

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 2, 5}) (hB : B = {1, 3, 4})

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_eq_l132_132081


namespace average_greater_median_l132_132240

theorem average_greater_median :
  let h : ℝ := 120
  let s1 : ℝ := 4
  let s2 : ℝ := 4
  let s3 : ℝ := 5
  let s4 : ℝ := 7
  let s5 : ℝ := 9
  let median : ℝ := (s3 + s4) / 2
  let average : ℝ := (h + s1 + s2 + s3 + s4 + s5) / 6
  average - median = 18.8333 := by
    sorry

end average_greater_median_l132_132240


namespace jackson_entertainment_expense_l132_132489

noncomputable def total_spent_on_entertainment_computer_game_original_price : ℝ :=
  66 / 0.85

noncomputable def movie_ticket_price_with_tax : ℝ :=
  12 * 1.10

noncomputable def total_movie_tickets_cost : ℝ :=
  3 * movie_ticket_price_with_tax

noncomputable def total_snacks_and_transportation_cost : ℝ :=
  7 + 5

noncomputable def total_spent : ℝ :=
  66 + total_movie_tickets_cost + total_snacks_and_transportation_cost

theorem jackson_entertainment_expense :
  total_spent = 117.60 :=
by
  sorry

end jackson_entertainment_expense_l132_132489


namespace regular_octagon_interior_angle_l132_132416

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l132_132416


namespace four_digit_numbers_divisible_by_7_l132_132464

theorem four_digit_numbers_divisible_by_7 :
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_n := Nat.ceil (lower_bound / 7)
  let largest_n := upper_bound / 7
  ∃ n : ℕ, smallest_n = 143 ∧ largest_n = 1428 ∧ (largest_n - smallest_n + 1 = 1286) :=
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_n := Nat.ceil (lower_bound / 7)
  let largest_n := upper_bound / 7
  use smallest_n, largest_n
  have h1 : smallest_n = 143 := sorry
  have h2 : largest_n = 1428 := sorry
  have h3 : largest_n - smallest_n + 1 = 1286 := sorry
  exact ⟨h1, h2, h3⟩

end four_digit_numbers_divisible_by_7_l132_132464


namespace Danai_can_buy_more_decorations_l132_132879

theorem Danai_can_buy_more_decorations :
  let skulls := 12
  let broomsticks := 4
  let spiderwebs := 12
  let pumpkins := 24 -- 2 times the number of spiderwebs
  let cauldron := 1
  let planned_total := 83
  let budget_left := 10
  let current_decorations := skulls + broomsticks + spiderwebs + pumpkins + cauldron
  current_decorations = 53 → -- 12 + 4 + 12 + 24 + 1
  let additional_decorations_needed := planned_total - current_decorations
  additional_decorations_needed = 30 → -- 83 - 53
  (additional_decorations_needed - budget_left) = 20 → -- 30 - 10
  True := -- proving the statement
sorry

end Danai_can_buy_more_decorations_l132_132879


namespace max_value_of_h_l132_132341

noncomputable def f (x : ℝ) : ℝ := -x + 3
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := min (f x) (g x)

theorem max_value_of_h : ∃ x : ℝ, h x = 1 :=
by
  sorry

end max_value_of_h_l132_132341


namespace number_of_bonnies_l132_132013

theorem number_of_bonnies (B blueberries apples : ℝ) 
  (h1 : blueberries = 3 / 4 * B) 
  (h2 : apples = 3 * blueberries)
  (h3 : B + blueberries + apples = 240) : 
  B = 60 :=
by
  sorry

end number_of_bonnies_l132_132013


namespace angle_measure_is_60_l132_132641

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132641


namespace james_speed_downhill_l132_132802

theorem james_speed_downhill (T1 T2 v : ℝ) (h1 : T1 = 20 / v) (h2 : T2 = 12 / 3 + 1) (h3 : T1 = T2 - 1) : v = 5 :=
by
  -- Declare variables
  have hT2 : T2 = 5 := by linarith
  have hT1 : T1 = 4 := by linarith
  have hv : v = 20 / 4 := by sorry
  linarith

#exit

end james_speed_downhill_l132_132802


namespace bob_grade_is_35_l132_132111

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l132_132111


namespace angle_measure_l132_132670

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132670


namespace program_arrangements_l132_132832

/-- Given 5 programs, if A, B, and C appear in a specific order, then the number of different
    arrangements is 20. -/
theorem program_arrangements (A B C A_order : ℕ) : 
  (A + B + C + A_order = 5) → 
  (A_order = 3) → 
  (B = 1) → 
  (C = 1) → 
  (A = 1) → 
  (A * B * C * A_order = 1) :=
  by sorry

end program_arrangements_l132_132832


namespace find_d_l132_132263

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l132_132263


namespace orange_jellybeans_count_l132_132311

theorem orange_jellybeans_count (total blue purple red : Nat)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_red : red = 120) :
  ∃ orange : Nat, orange = total - (blue + purple + red) ∧ orange = 40 :=
by
  sorry

end orange_jellybeans_count_l132_132311


namespace bottles_per_person_l132_132160

theorem bottles_per_person
  (boxes : ℕ)
  (bottles_per_box : ℕ)
  (bottles_eaten : ℕ)
  (people : ℕ)
  (total_bottles : ℕ := boxes * bottles_per_box)
  (remaining_bottles : ℕ := total_bottles - bottles_eaten)
  (bottles_per_person : ℕ := remaining_bottles / people) :
  boxes = 7 → bottles_per_box = 9 → bottles_eaten = 7 → people = 8 → bottles_per_person = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end bottles_per_person_l132_132160


namespace slope_of_line_determined_by_solutions_l132_132841

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l132_132841


namespace molly_more_minutes_than_xanthia_l132_132184

-- Define the constants: reading speeds and book length
def xanthia_speed := 80  -- pages per hour
def molly_speed := 40    -- pages per hour
def book_length := 320   -- pages

-- Define the times taken to read the book in hours
def xanthia_time := book_length / xanthia_speed
def molly_time := book_length / molly_speed

-- Define the time difference in minutes
def time_difference_minutes := (molly_time - xanthia_time) * 60

theorem molly_more_minutes_than_xanthia : time_difference_minutes = 240 := 
by {
  -- Here the proof would go, but we'll leave it as a sorry for now.
  sorry
}

end molly_more_minutes_than_xanthia_l132_132184


namespace rate_of_Y_l132_132859

noncomputable def rate_X : ℝ := 2
noncomputable def time_to_cross : ℝ := 0.5

theorem rate_of_Y (rate_Y : ℝ) : rate_X * time_to_cross = 1 → rate_Y * time_to_cross = 1 → rate_Y = rate_X :=
by
    intros h_rate_X h_rate_Y
    sorry

end rate_of_Y_l132_132859


namespace dog_years_second_year_l132_132825

theorem dog_years_second_year (human_years : ℕ) :
  15 + human_years + 5 * 8 = 64 →
  human_years = 9 :=
by
  intro h
  sorry

end dog_years_second_year_l132_132825


namespace largest_number_l132_132853

theorem largest_number (P Q R S T : ℕ) 
  (hP_digits_prime : ∃ p1 p2, P = 10 * p1 + p2 ∧ Prime P ∧ Prime (p1 + p2))
  (hQ_multiple_of_5 : Q % 5 = 0)
  (hR_odd_non_prime : Odd R ∧ ¬ Prime R)
  (hS_prime_square : ∃ p, Prime p ∧ S = p * p)
  (hT_mean_prime : T = (P + Q) / 2 ∧ Prime T)
  (hP_range : 10 ≤ P ∧ P ≤ 99)
  (hQ_range : 2 ≤ Q ∧ Q ≤ 19)
  (hR_range : 2 ≤ R ∧ R ≤ 19)
  (hS_range : 2 ≤ S ∧ S ≤ 19)
  (hT_range : 2 ≤ T ∧ T ≤ 19) :
  max P (max Q (max R (max S T))) = Q := 
by 
  sorry

end largest_number_l132_132853


namespace tissue_magnification_l132_132723

theorem tissue_magnification (d_image d_actual : ℝ) (h_image : d_image = 0.3) (h_actual : d_actual = 0.0003) :
  (d_image / d_actual) = 1000 :=
by
  sorry

end tissue_magnification_l132_132723


namespace arrangement_ways_l132_132731

-- Defining the conditions
def num_basil_plants : Nat := 5
def num_tomato_plants : Nat := 4
def num_total_units : Nat := num_basil_plants + 1

-- Proof statement
theorem arrangement_ways : (num_total_units.factorial) * (num_tomato_plants.factorial) = 17280 := by
  sorry

end arrangement_ways_l132_132731


namespace total_female_officers_l132_132131

theorem total_female_officers
  (percent_female_on_duty : ℝ)
  (total_on_duty : ℝ)
  (half_of_total_on_duty : ℝ)
  (num_females_on_duty : ℝ) :
  percent_female_on_duty = 0.10 →
  total_on_duty = 200 →
  half_of_total_on_duty = total_on_duty / 2 →
  num_females_on_duty = half_of_total_on_duty →
  num_females_on_duty = percent_female_on_duty * (1000 : ℝ) :=
by
  intros h1 h2 h3 h4
  sorry

end total_female_officers_l132_132131


namespace inequality_am_gm_l132_132342

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l132_132342


namespace age_difference_l132_132315

-- Define the present age of the son as a constant
def S : ℕ := 22

-- Define the equation given by the problem
noncomputable def age_relation (M : ℕ) : Prop :=
  M + 2 = 2 * (S + 2)

-- The theorem to prove the man is 24 years older than his son
theorem age_difference (M : ℕ) (h_rel : age_relation M) : M - S = 24 :=
by {
  sorry
}

end age_difference_l132_132315


namespace angle_measure_l132_132576

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132576


namespace principal_amount_l132_132316

theorem principal_amount (P : ℕ) (R : ℕ) (T : ℕ) (SI : ℕ) 
  (h1 : R = 12)
  (h2 : T = 10)
  (h3 : SI = 1500) 
  (h4 : SI = (P * R * T) / 100) : P = 1250 :=
by sorry

end principal_amount_l132_132316


namespace angle_solution_l132_132651

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132651


namespace parabola_axis_of_symmetry_is_x_eq_1_l132_132062

theorem parabola_axis_of_symmetry_is_x_eq_1 :
  ∀ x : ℝ, ∀ y : ℝ, y = -2 * (x - 1)^2 + 3 → (∀ c : ℝ, c = 1 → ∃ x1 x2 : ℝ, x1 = c ∧ x2 = c) := 
by
  sorry

end parabola_axis_of_symmetry_is_x_eq_1_l132_132062


namespace Mildred_heavier_than_Carol_l132_132502

-- Definition of weights for Mildred and Carol
def weight_Mildred : ℕ := 59
def weight_Carol : ℕ := 9

-- Definition of how much heavier Mildred is than Carol
def weight_difference : ℕ := weight_Mildred - weight_Carol

-- The theorem stating the difference in weight
theorem Mildred_heavier_than_Carol : weight_difference = 50 := 
by 
  -- Just state the theorem without providing the actual steps (proof skipped)
  sorry

end Mildred_heavier_than_Carol_l132_132502


namespace probability_three_heads_in_seven_tosses_l132_132032

theorem probability_three_heads_in_seven_tosses :
  (Nat.choose 7 3 : ℝ) / (2 ^ 7 : ℝ) = 35 / 128 :=
by
  sorry

end probability_three_heads_in_seven_tosses_l132_132032


namespace eval_expression_l132_132216

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l132_132216


namespace Darcy_remaining_clothes_l132_132746

/--
Darcy initially has 20 shirts and 8 pairs of shorts.
He folds 12 of the shirts and 5 of the pairs of shorts.
We want to prove that the total number of remaining pieces of clothing Darcy has to fold is 11.
-/
theorem Darcy_remaining_clothes
  (initial_shirts : Nat)
  (initial_shorts : Nat)
  (folded_shirts : Nat)
  (folded_shorts : Nat)
  (remaining_shirts : Nat)
  (remaining_shorts : Nat)
  (total_remaining : Nat) :
  initial_shirts = 20 → initial_shorts = 8 →
  folded_shirts = 12 → folded_shorts = 5 →
  remaining_shirts = initial_shirts - folded_shirts →
  remaining_shorts = initial_shorts - folded_shorts →
  total_remaining = remaining_shirts + remaining_shorts →
  total_remaining = 11 := by
  sorry

end Darcy_remaining_clothes_l132_132746


namespace man_is_older_by_l132_132313

theorem man_is_older_by :
  ∀ (M S : ℕ), S = 22 → (M + 2) = 2 * (S + 2) → (M - S) = 24 :=
by
  intros M S h1 h2
  sorry

end man_is_older_by_l132_132313


namespace positive_difference_balances_l132_132493

noncomputable def laura_balance (L_0 : ℝ) (L_r : ℝ) (L_n : ℕ) (t : ℕ) : ℝ :=
  L_0 * (1 + L_r / L_n) ^ (L_n * t)

noncomputable def mark_balance (M_0 : ℝ) (M_r : ℝ) (t : ℕ) : ℝ :=
  M_0 * (1 + M_r * t)

theorem positive_difference_balances :
  let L_0 := 10000
  let L_r := 0.04
  let L_n := 2
  let t := 20
  let M_0 := 10000
  let M_r := 0.06
  abs ((laura_balance L_0 L_r L_n t) - (mark_balance M_0 M_r t)) = 80.40 :=
by
  sorry

end positive_difference_balances_l132_132493


namespace find_initial_men_l132_132253

noncomputable def initial_men_planned (M : ℕ) : Prop :=
  let initial_days := 10
  let additional_days := 20
  let total_days := initial_days + additional_days
  let men_sent := 25
  let initial_work := M * initial_days
  let remaining_men := M - men_sent
  let remaining_work := remaining_men * total_days
  initial_work = remaining_work 

theorem find_initial_men :
  ∃ M : ℕ, initial_men_planned M ∧ M = 38 :=
by
  have h : initial_men_planned 38 :=
    by
      sorry
  exact ⟨38, h, rfl⟩

end find_initial_men_l132_132253


namespace correct_operation_l132_132844

theorem correct_operation (x : ℝ) : (x^2) * (x^4) = x^6 :=
  sorry

end correct_operation_l132_132844


namespace regular_octagon_interior_angle_l132_132374

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l132_132374


namespace interior_angle_regular_octagon_l132_132412

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l132_132412


namespace total_screens_sold_l132_132998

variable (J F M : ℕ)
variable (feb_eq_fourth_of_march : F = M / 4)
variable (feb_eq_double_of_jan : F = 2 * J)
variable (march_sales : M = 8800)

theorem total_screens_sold (J F M : ℕ)
  (feb_eq_fourth_of_march : F = M / 4)
  (feb_eq_double_of_jan : F = 2 * J)
  (march_sales : M = 8800) :
  J + F + M = 12100 :=
by
  sorry

end total_screens_sold_l132_132998


namespace num_pos_four_digit_integers_l132_132774

theorem num_pos_four_digit_integers : 
  ∃ (n : ℕ), n = (Nat.factorial 4) / ((Nat.factorial 3) * (Nat.factorial 1)) ∧ n = 4 := 
by
  sorry

end num_pos_four_digit_integers_l132_132774


namespace P_inter_Q_empty_l132_132496

def P := {y : ℝ | ∃ x : ℝ, y = x^2}
def Q := {(x, y) : ℝ × ℝ | y = x^2}

theorem P_inter_Q_empty : P ∩ Q = ∅ :=
by sorry

end P_inter_Q_empty_l132_132496


namespace mutually_exclusive_complementary_event_l132_132862

-- Definitions of events
def hitting_target_at_least_once (shots: ℕ) : Prop := shots > 0
def not_hitting_target_at_all (shots: ℕ) : Prop := shots = 0

-- The statement to prove
theorem mutually_exclusive_complementary_event : 
  ∀ (shots: ℕ), (not_hitting_target_at_all shots ↔ ¬ hitting_target_at_least_once shots) :=
by 
  sorry

end mutually_exclusive_complementary_event_l132_132862


namespace average_weight_l132_132794

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l132_132794


namespace gdp_scientific_notation_l132_132483

theorem gdp_scientific_notation (gdp : ℝ) (h : gdp = 338.8 * 10^9) : gdp = 3.388 * 10^10 :=
by sorry

end gdp_scientific_notation_l132_132483


namespace angle_supplement_complement_l132_132605

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132605


namespace travelers_on_liner_l132_132162

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l132_132162


namespace angle_supplement_complement_l132_132607

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132607


namespace solve_for_y_l132_132075

theorem solve_for_y (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 :=
  sorry

end solve_for_y_l132_132075


namespace li_ming_estimated_weight_is_correct_l132_132230

-- Define the regression equation as a function
def regression_equation (x : ℝ) : ℝ := 0.7 * x - 52

-- Define the height of Li Ming
def li_ming_height : ℝ := 180

-- The estimated weight according to the regression equation
def estimated_weight : ℝ := regression_equation li_ming_height

-- Theorem statement: Given the height, the weight should be 74
theorem li_ming_estimated_weight_is_correct : estimated_weight = 74 :=
by
  sorry

end li_ming_estimated_weight_is_correct_l132_132230


namespace largest_of_five_l132_132975

def a : ℝ := 0.994
def b : ℝ := 0.9399
def c : ℝ := 0.933
def d : ℝ := 0.9940
def e : ℝ := 0.9309

theorem largest_of_five : (a > b ∧ a > c ∧ a ≥ d ∧ a > e) := by
  -- We add sorry here to skip the proof
  sorry

end largest_of_five_l132_132975


namespace coin_flip_probability_l132_132188

theorem coin_flip_probability (P : ℕ → ℕ → ℚ) (n : ℕ) :
  (∀ k, P k 0 = 1/2) →
  (∀ k, P k 1 = 1/2) →
  (∀ k m, P k m = 1/2) →
  n = 3 →
  P 0 0 * P 1 1 * P 2 1 = 1/8 :=
by
  intros h0 h1 h_indep hn
  sorry

end coin_flip_probability_l132_132188


namespace remaining_pieces_to_fold_l132_132741

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end remaining_pieces_to_fold_l132_132741


namespace regular_octagon_angle_l132_132452

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l132_132452


namespace curve_is_line_l132_132227

-- Let theta be the angle in polar coordinates
variable {θ : ℝ}

-- Define the condition given in the problem
def condition : Prop := θ = π / 4

-- Prove that the curve defined by this condition is a line
theorem curve_is_line (h : condition) : ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b :=
by
  sorry

end curve_is_line_l132_132227


namespace angle_complement_supplement_l132_132626

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132626


namespace negation_of_existence_l132_132285

theorem negation_of_existence (T : Type) (triangle : T → Prop) (sum_interior_angles : T → ℝ) :
  (¬ ∃ t : T, sum_interior_angles t ≠ 180) ↔ (∀ t : T, sum_interior_angles t = 180) :=
by 
  sorry

end negation_of_existence_l132_132285


namespace angle_supplement_complement_l132_132657

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132657


namespace sequence_2018_value_l132_132904

theorem sequence_2018_value :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2018 = -3 :=
sorry

end sequence_2018_value_l132_132904


namespace regular_octagon_interior_angle_l132_132378

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l132_132378


namespace volume_of_circumscribed_sphere_l132_132701

theorem volume_of_circumscribed_sphere (vol_cube : ℝ) (h : vol_cube = 8) :
  ∃ (vol_sphere : ℝ), vol_sphere = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_circumscribed_sphere_l132_132701


namespace angle_measure_is_60_l132_132634

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132634


namespace marbles_per_box_l132_132291

-- Define the total number of marbles
def total_marbles : Nat := 18

-- Define the number of boxes
def number_of_boxes : Nat := 3

-- Prove there are 6 marbles in each box
theorem marbles_per_box : total_marbles / number_of_boxes = 6 := by
  sorry

end marbles_per_box_l132_132291


namespace range_of_a_l132_132307

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3 * x + 1) ≤ a) → a ≥ 1/5 :=
by
  intro h
  sorry

end range_of_a_l132_132307


namespace points_symmetric_about_x_axis_l132_132960

def point := ℝ × ℝ

def symmetric_x_axis (A B : point) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem points_symmetric_about_x_axis : symmetric_x_axis (-1, 3) (-1, -3) :=
by
  sorry

end points_symmetric_about_x_axis_l132_132960


namespace total_hours_worked_l132_132720

theorem total_hours_worked (Amber_hours : ℕ) (h_Amber : Amber_hours = 12) 
  (Armand_hours : ℕ) (h_Armand : Armand_hours = Amber_hours / 3)
  (Ella_hours : ℕ) (h_Ella : Ella_hours = Amber_hours * 2) : 
  Amber_hours + Armand_hours + Ella_hours = 40 :=
by
  rw [h_Amber, h_Armand, h_Ella]
  norm_num
  sorry

end total_hours_worked_l132_132720


namespace total_distance_correct_l132_132118

def jonathan_distance : ℝ := 7.5

def mercedes_distance : ℝ := 2 * jonathan_distance

def davonte_distance : ℝ := mercedes_distance + 2

def total_distance : ℝ := mercedes_distance + davonte_distance

theorem total_distance_correct : total_distance = 32 := by
  rw [total_distance, mercedes_distance, davonte_distance]
  norm_num
  sorry

end total_distance_correct_l132_132118


namespace expression_in_terms_of_p_and_q_l132_132908

theorem expression_in_terms_of_p_and_q (x : ℝ) :
  let p := (1 - Real.cos x) * (1 + Real.sin x)
  let q := (1 + Real.cos x) * (1 - Real.sin x)
  (Real.cos x ^ 2 - Real.cos x ^ 4 - Real.sin (2 * x) + 2) = p * q - (p + q) :=
by
  sorry

end expression_in_terms_of_p_and_q_l132_132908


namespace angle_measure_is_60_l132_132640

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132640


namespace interior_angle_regular_octagon_l132_132428

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l132_132428


namespace inequality_proof_l132_132352

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l132_132352


namespace batsman_average_increase_l132_132195

theorem batsman_average_increase 
  (A : ℕ)
  (h1 : ∀ n ≤ 11, (1 / (n : ℝ)) * (A * n + 60) = 38) 
  (h2 : 1 / 12 * (A * 11 + 60) = 38)
  (h3 : ∀ n ≤ 12, (A * n : ℝ) ≤ (A * (n + 1) : ℝ)) :
  38 - A = 2 := 
sorry

end batsman_average_increase_l132_132195


namespace angle_supplement_complement_l132_132609

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132609


namespace total_bill_correct_l132_132870

def first_family_adults := 2
def first_family_children := 3
def second_family_adults := 4
def second_family_children := 2
def third_family_adults := 3
def third_family_children := 4

def adult_meal_cost := 8
def child_meal_cost := 5
def drink_cost_per_person := 2

def calculate_total_cost 
  (adults1 : ℕ) (children1 : ℕ) 
  (adults2 : ℕ) (children2 : ℕ) 
  (adults3 : ℕ) (children3 : ℕ)
  (adult_cost : ℕ) (child_cost : ℕ)
  (drink_cost : ℕ) : ℕ := 
  let meal_cost1 := (adults1 * adult_cost) + (children1 * child_cost)
  let meal_cost2 := (adults2 * adult_cost) + (children2 * child_cost)
  let meal_cost3 := (adults3 * adult_cost) + (children3 * child_cost)
  let drink_cost1 := (adults1 + children1) * drink_cost
  let drink_cost2 := (adults2 + children2) * drink_cost
  let drink_cost3 := (adults3 + children3) * drink_cost
  meal_cost1 + drink_cost1 + meal_cost2 + drink_cost2 + meal_cost3 + drink_cost3
   
theorem total_bill_correct :
  calculate_total_cost
    first_family_adults first_family_children
    second_family_adults second_family_children
    third_family_adults third_family_children
    adult_meal_cost child_meal_cost drink_cost_per_person = 153 :=
  sorry

end total_bill_correct_l132_132870


namespace hermia_elected_probability_l132_132268

def probability_hermia_elected (n : ℕ) (h1 : Odd n) (h2 : n > 0) : ℝ :=
  (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1))

theorem hermia_elected_probability (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  probability_hermia_elected n h1 h2 = (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1)) :=
sorry

end hermia_elected_probability_l132_132268


namespace regular_octagon_interior_angle_l132_132458

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l132_132458


namespace Bennett_sales_l132_132996

-- Define the variables for the number of screens sold in each month.
variables (J F M : ℕ)

-- State the given conditions.
theorem Bennett_sales (h1: F = 2 * J) (h2: F = M / 4) (h3: M = 8800) :
  J + F + M = 12100 := by
sorry

end Bennett_sales_l132_132996


namespace seventy_three_days_after_monday_is_thursday_l132_132735

def day_of_week : Nat → String
| 0 => "Monday"
| 1 => "Tuesday"
| 2 => "Wednesday"
| 3 => "Thursday"
| 4 => "Friday"
| 5 => "Saturday"
| _ => "Sunday"

theorem seventy_three_days_after_monday_is_thursday :
  day_of_week (73 % 7) = "Thursday" :=
by
  sorry

end seventy_three_days_after_monday_is_thursday_l132_132735


namespace total_distance_correct_l132_132117

def jonathan_distance : ℝ := 7.5

def mercedes_distance : ℝ := 2 * jonathan_distance

def davonte_distance : ℝ := mercedes_distance + 2

def total_distance : ℝ := mercedes_distance + davonte_distance

theorem total_distance_correct : total_distance = 32 := by
  rw [total_distance, mercedes_distance, davonte_distance]
  norm_num
  sorry

end total_distance_correct_l132_132117


namespace oliver_total_earnings_l132_132935

/-- Rates for different types of laundry items -/
def rate_regular : ℝ := 3
def rate_delicate : ℝ := 4
def rate_bulky : ℝ := 5

/-- Quantity of laundry items washed over three days -/
def quantity_day1_regular : ℝ := 7
def quantity_day1_delicate : ℝ := 4
def quantity_day1_bulky : ℝ := 2

def quantity_day2_regular : ℝ := 10
def quantity_day2_delicate : ℝ := 6
def quantity_day2_bulky : ℝ := 3

def quantity_day3_regular : ℝ := 20
def quantity_day3_delicate : ℝ := 4
def quantity_day3_bulky : ℝ := 0

/-- Discount on delicate clothes for the third day -/
def discount : ℝ := 0.2

/-- The expected earnings for each day and total -/
def earnings_day1 : ℝ :=
  rate_regular * quantity_day1_regular +
  rate_delicate * quantity_day1_delicate +
  rate_bulky * quantity_day1_bulky

def earnings_day2 : ℝ :=
  rate_regular * quantity_day2_regular +
  rate_delicate * quantity_day2_delicate +
  rate_bulky * quantity_day2_bulky

def earnings_day3 : ℝ :=
  rate_regular * quantity_day3_regular +
  (rate_delicate * quantity_day3_delicate * (1 - discount)) +
  rate_bulky * quantity_day3_bulky

def total_earnings : ℝ := earnings_day1 + earnings_day2 + earnings_day3

theorem oliver_total_earnings : total_earnings = 188.80 := by
  sorry

end oliver_total_earnings_l132_132935


namespace interior_angle_regular_octagon_l132_132429

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l132_132429


namespace find_d_l132_132264

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l132_132264


namespace sum_even_integers_602_to_700_l132_132962

-- Definitions based on the conditions and the problem statement
def sum_first_50_even_integers := 2550
def n_even_602_700 := 50
def first_term_602_to_700 := 602
def last_term_602_to_700 := 700

-- Theorem statement
theorem sum_even_integers_602_to_700 : 
  sum_first_50_even_integers = 2550 → 
  n_even_602_700 = 50 →
  (n_even_602_700 / 2) * (first_term_602_to_700 + last_term_602_to_700) = 32550 :=
by
  sorry

end sum_even_integers_602_to_700_l132_132962


namespace find_m_n_l132_132777

-- Given definition of like terms
def like_terms (a b : ℝ) (n1 n2 m1 m2 : ℤ) : Prop :=
  n1 = n2 ∧ m1 = m2

-- Variables
variables {x y : ℝ} {m n : ℤ}

-- Definitions based on problem conditions
def expr1 : ℝ := 2 * x^(n + 2) * y^3
def expr2 : ℝ := -3 * x^3 * y^(2 * m - 1)

-- Proof problem
theorem find_m_n (h : like_terms expr1 expr2 (n + 2) 3 3 (2 * m - 1)) : m = 2 ∧ n = 1 :=
  sorry

end find_m_n_l132_132777


namespace min_value_x_2y_l132_132363

theorem min_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y + 2 * x * y = 8) : x + 2 * y ≥ 4 :=
sorry

end min_value_x_2y_l132_132363


namespace geometric_then_sum_geometric_l132_132071

variable {a b c d : ℝ}

def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def forms_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

theorem geometric_then_sum_geometric (h : geometric_sequence a b c d) :
  forms_geometric_sequence (a + b) (b + c) (c + d) :=
sorry

end geometric_then_sum_geometric_l132_132071


namespace vector_computation_equiv_l132_132330

variables (u v w : ℤ × ℤ)

def vector_expr (u v w : ℤ × ℤ) :=
  2 • u + 4 • v - 3 • w

theorem vector_computation_equiv :
  u = (3, -5) →
  v = (-1, 6) →
  w = (2, -4) →
  vector_expr u v w = (-4, 26) :=
by
  intros hu hv hw
  rw [hu, hv, hw]
  dsimp [vector_expr]
  -- The actual proof goes here, but we use 'sorry' to skip it.
  sorry

end vector_computation_equiv_l132_132330


namespace andrew_purchased_mangoes_l132_132725

variable (m : ℕ)

def cost_of_grapes := 8 * 70
def cost_of_mangoes (m : ℕ) := 55 * m
def total_cost (m : ℕ) := cost_of_grapes + cost_of_mangoes m

theorem andrew_purchased_mangoes :
  total_cost m = 1055 → m = 9 := by
  intros h_total_cost
  sorry

end andrew_purchased_mangoes_l132_132725


namespace sumNats_l132_132562

-- Define the set of natural numbers between 29 and 31 inclusive
def NatRange : List ℕ := [29, 30, 31]

-- Define the condition that checks the elements in the range
def isValidNumbers (n : ℕ) : Prop := n ≤ 31 ∧ n > 28

-- Check if all numbers in NatRange are valid
def allValidNumbers : Prop := ∀ n, n ∈ NatRange → isValidNumbers n

-- Define the sum function for the list
def sumList (lst : List ℕ) : ℕ := lst.foldr (.+.) 0

-- The main theorem
theorem sumNats : (allValidNumbers → (sumList NatRange) = 90) :=
by
  sorry

end sumNats_l132_132562


namespace two_digit_number_representation_l132_132042

theorem two_digit_number_representation (a b : ℕ) (ha : a < 10) (hb : b < 10) : 10 * b + a = d :=
  sorry

end two_digit_number_representation_l132_132042


namespace sum_of_first_n_primes_eq_41_l132_132057

theorem sum_of_first_n_primes_eq_41 : 
  ∃ (n : ℕ) (primes : List ℕ), 
    primes = [2, 3, 5, 7, 11, 13] ∧ primes.sum = 41 ∧ primes.length = n := 
by 
  sorry

end sum_of_first_n_primes_eq_41_l132_132057


namespace plane_equation_l132_132736

theorem plane_equation (x y z : ℝ)
  (h₁ : ∃ t : ℝ, x = 2 * t + 1 ∧ y = -3 * t ∧ z = 3 - t)
  (h₂ : ∃ (t₁ t₂ : ℝ), 4 * t₁ + 5 * t₂ - 3 = 0 ∧ 2 * t₁ + t₂ + 2 * t₂ = 0) : 
  2*x - y + 7*z - 23 = 0 :=
sorry

end plane_equation_l132_132736


namespace regular_octagon_angle_l132_132450

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l132_132450


namespace lateral_surface_area_of_pyramid_l132_132543

theorem lateral_surface_area_of_pyramid
  (sin_alpha : ℝ)
  (A_section : ℝ)
  (h1 : sin_alpha = 15 / 17)
  (h2 : A_section = 3 * Real.sqrt 34) :
  ∃ A_lateral : ℝ, A_lateral = 68 :=
sorry

end lateral_surface_area_of_pyramid_l132_132543


namespace bob_grade_is_35_l132_132108

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l132_132108


namespace truck_travel_distance_l132_132866

theorem truck_travel_distance (b t : ℝ) (h1 : t > 0) :
  (300 * (b / 4) / t) / 3 = (25 * b) / t :=
by
  sorry

end truck_travel_distance_l132_132866


namespace three_digit_factorions_l132_132860

def is_factorion (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  let (a, b, c) := digits
  n = Nat.factorial a + Nat.factorial b + Nat.factorial c

theorem three_digit_factorions : ∀ n : ℕ, (100 ≤ n ∧ n < 1000) → is_factorion n → n = 145 :=
by
  sorry

end three_digit_factorions_l132_132860


namespace sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l132_132941

open Real

theorem sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z ≥ x * y + y * z + z * x) :
  sqrt (x * y * z) ≥ sqrt x + sqrt y + sqrt z :=
by
  sorry

end sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l132_132941


namespace common_ratio_value_l132_132152

theorem common_ratio_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) → (x + y + z = 0 ∨ x + y + z ≠ 0) → ((x + y) / z = -1 ∨ (x + y) / z = 2) :=
by
  sorry

end common_ratio_value_l132_132152


namespace price_of_basic_computer_l132_132026

-- Definitions for the prices
variables (C_b P M K C_e : ℝ)

-- Conditions
axiom h1 : C_b + P + M + K = 2500
axiom h2 : C_e + P + M + K = 3100
axiom h3 : P = (3100 / 6)
axiom h4 : M = (3100 / 5)
axiom h5 : K = (3100 / 8)
axiom h6 : C_e = C_b + 600

-- Theorem stating the price of the basic computer
theorem price_of_basic_computer : C_b = 975.83 :=
by {
  sorry
}

end price_of_basic_computer_l132_132026


namespace regular_octagon_interior_angle_l132_132372

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l132_132372


namespace solve_cubic_inequality_l132_132224

theorem solve_cubic_inequality :
  { x : ℝ | x^3 + x^2 - 7 * x + 6 < 0 } = { x : ℝ | -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 } :=
by
  sorry

end solve_cubic_inequality_l132_132224


namespace decimal_multiplication_l132_132364

theorem decimal_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by sorry

end decimal_multiplication_l132_132364


namespace common_tangent_l132_132473

-- Define the functions f and g
def f (x : ℝ) : ℝ := (1 / (2 * Real.exp 1)) * x^2
def g (a x : ℝ) : ℝ := a * Real.log x

-- Derivatives of the functions
def f' (x : ℝ) : ℝ := x / (Real.exp 1)
def g' (a x : ℝ) : ℝ := a / x

-- Statement of the problem
theorem common_tangent (a s : ℝ) (hs : 0 < s) (ha : a > 0) :
  (f' s = g' a s) ∧ (f s = g a s) → a = 1 :=
by
  sorry

end common_tangent_l132_132473


namespace angle_measure_l132_132614

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132614


namespace max_value_f_l132_132880

open Real

/-- Determine the maximum value of the function f(x) = 1 / (1 - x * (1 - x)). -/
theorem max_value_f (x : ℝ) : 
  ∃ y, y = (1 / (1 - x * (1 - x))) ∧ y ≤ 4/3 ∧ ∀ z, z = (1 / (1 - x * (1 - x))) → z ≤ 4/3 :=
by
  sorry

end max_value_f_l132_132880


namespace complex_equality_l132_132895

theorem complex_equality (a b : ℝ) (h : (⟨0, 1⟩ : ℂ) ^ 3 = ⟨a, -b⟩) : a + b = 1 :=
by
  sorry

end complex_equality_l132_132895


namespace regular_octagon_interior_angle_l132_132388

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l132_132388


namespace students_passed_correct_l132_132158

-- Define the number of students in ninth grade.
def students_total : ℕ := 180

-- Define the number of students who bombed their finals.
def students_bombed : ℕ := students_total / 4

-- Define the number of students remaining after removing those who bombed.
def students_remaining_after_bombed : ℕ := students_total - students_bombed

-- Define the number of students who didn't show up to take the test.
def students_didnt_show : ℕ := students_remaining_after_bombed / 3

-- Define the number of students remaining after removing those who didn't show up.
def students_remaining_after_no_show : ℕ := students_remaining_after_bombed - students_didnt_show

-- Define the number of students who got less than a D.
def students_less_than_d : ℕ := 20

-- Define the number of students who passed.
def students_passed : ℕ := students_remaining_after_no_show - students_less_than_d

-- Statement to prove the number of students who passed is 70.
theorem students_passed_correct : students_passed = 70 := by
  -- Proof will be inserted here.
  sorry

end students_passed_correct_l132_132158


namespace ken_change_l132_132044

theorem ken_change (cost_per_pound : ℕ) (quantity : ℕ) (amount_paid : ℕ) (total_cost : ℕ) (change : ℕ) 
(h1 : cost_per_pound = 7)
(h2 : quantity = 2)
(h3 : amount_paid = 20)
(h4 : total_cost = cost_per_pound * quantity)
(h5 : change = amount_paid - total_cost) : change = 6 :=
by 
  sorry

end ken_change_l132_132044


namespace arrangements_of_doctors_and_nurses_l132_132814

theorem arrangements_of_doctors_and_nurses (docs nurses schools : ℕ) 
  (h_docs : docs = 3) (h_nurses : nurses = 4) (h_schools : schools = 3) 
  (h_min_one_doc : True) (h_min_one_nurse : True) : 
  ∃ (arrangements : ℕ), arrangements = 216 := 
by
  have docs_arrangements : ℕ := Nat.factorial docs / Nat.factorial (docs - schools)
  have nurses_pairing : ℕ := Nat.choose nurses 2
  have nurses_arrangements : ℕ := docs_arrangements
  have total_arrangements : ℕ := docs_arrangements * nurses_pairing * nurses_arrangements
  have h : docs_arrangements = 6 := by sorry
  have h' : nurses_pairing = 6 := by sorry
  have h'' : nurses_arrangements = 6 := by sorry
  exists total_arrangements
  show total_arrangements = 216 by
    rw [h, h', h'']
    simp [total_arrangements]
    sorry

end arrangements_of_doctors_and_nurses_l132_132814


namespace regular_octagon_interior_angle_l132_132383

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l132_132383


namespace determine_n_l132_132049

open Function

noncomputable def coeff_3 (n : ℕ) : ℕ :=
  2^(n-2) * Nat.choose n 2

noncomputable def coeff_4 (n : ℕ) : ℕ :=
  2^(n-3) * Nat.choose n 3

theorem determine_n (n : ℕ) (b3_eq_2b4 : coeff_3 n = 2 * coeff_4 n) : n = 5 :=
  sorry

end determine_n_l132_132049


namespace parabola_directrix_p_l132_132238

/-- Given a parabola with equation y^2 = 2px and directrix x = -2, prove that p = 4 -/
theorem parabola_directrix_p (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x) ∧ (∀ x : ℝ, x = -2 → True) → p = 4 :=
by
  sorry

end parabola_directrix_p_l132_132238


namespace slope_of_line_determined_by_solutions_l132_132842

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l132_132842


namespace decimal_to_base13_185_l132_132048

theorem decimal_to_base13_185 : 
  ∀ n : ℕ, n = 185 → 
      ∃ a b c : ℕ, a * 13^2 + b * 13 + c = n ∧ 0 ≤ a ∧ a < 13 ∧ 0 ≤ b ∧ b < 13 ∧ 0 ≤ c ∧ c < 13 ∧ (a, b, c) = (1, 1, 3) := 
by
  intros n hn
  use 1, 1, 3
  sorry

end decimal_to_base13_185_l132_132048


namespace travelers_on_liner_l132_132167

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l132_132167


namespace angle_supplement_complement_l132_132603

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132603


namespace statement_B_statement_C_statement_D_l132_132755

variables (a b : ℝ)

-- Condition: a > 0
axiom a_pos : a > 0

-- Condition: e^a + ln b = 1
axiom eq1 : Real.exp a + Real.log b = 1

-- Statement B: a + ln b < 0
theorem statement_B : a + Real.log b < 0 :=
  sorry

-- Statement C: e^a + b > 2
theorem statement_C : Real.exp a + b > 2 :=
  sorry

-- Statement D: a + b > 1
theorem statement_D : a + b > 1 :=
  sorry

end statement_B_statement_C_statement_D_l132_132755


namespace triangle_area_ab_l132_132471

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (hline : ∀ x y : ℝ, 2 * a * x + 3 * b * y = 12) (harea : (1/2) * (6 / a) * (4 / b) = 9) : 
    a * b = 4 / 3 :=
by 
  sorry

end triangle_area_ab_l132_132471


namespace each_interior_angle_of_regular_octagon_l132_132405

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l132_132405


namespace dice_roll_probability_l132_132712

theorem dice_roll_probability : 
  ∃ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ (m - n > 0) ∧ 
  ( (15 : ℚ) / 36 = (5 : ℚ) / 12 ) :=
by {
  sorry
}

end dice_roll_probability_l132_132712


namespace one_add_i_cubed_eq_one_sub_i_l132_132018

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
sorry

end one_add_i_cubed_eq_one_sub_i_l132_132018


namespace fewerCansCollected_l132_132695

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end fewerCansCollected_l132_132695


namespace angle_supplement_complement_l132_132663

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132663


namespace angle_measure_l132_132569

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132569


namespace angle_measure_l132_132568

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132568


namespace greatest_remainder_when_dividing_by_10_l132_132084

theorem greatest_remainder_when_dividing_by_10 (x : ℕ) : 
  ∃ r : ℕ, r < 10 ∧ r = x % 10 ∧ r = 9 :=
by
  sorry

end greatest_remainder_when_dividing_by_10_l132_132084


namespace total_balloons_l132_132356

theorem total_balloons (F S M : ℕ) (hF : F = 5) (hS : S = 6) (hM : M = 7) : F + S + M = 18 :=
by 
  sorry

end total_balloons_l132_132356


namespace first_number_in_proportion_is_60_l132_132472

theorem first_number_in_proportion_is_60 : 
  ∀ (x : ℝ), (x / 6 = 2 / 0.19999999999999998) → x = 60 :=
by
  intros x hx
  sorry

end first_number_in_proportion_is_60_l132_132472


namespace increase_80_by_50_percent_l132_132703

theorem increase_80_by_50_percent : 
  let original_number := 80
  let percentage_increase := 0.5
  let increase := original_number * percentage_increase
  let final_number := original_number + increase
  final_number = 120 := 
by 
  sorry

end increase_80_by_50_percent_l132_132703


namespace work_days_together_l132_132978

-- Conditions
variable {W : ℝ} (h_a_alone : ∀ (W : ℝ), W / a_work_time = W / 16)
variable {a_work_time : ℝ} (h_work_time_a : a_work_time = 16)

-- Question translated to proof problem
theorem work_days_together (D : ℝ) :
  (10 * (W / D) + 12 * (W / 16) = W) → D = 40 :=
by
  intros h
  have eq1 : 10 * (W / D) + 12 * (W / 16) = W := h
  sorry

end work_days_together_l132_132978


namespace find_constants_l132_132121

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 2)

theorem find_constants (a : ℝ) (x : ℝ) (h : x ≠ -2) :
  f a (f a x) = x ∧ a = -4 :=
by
  sorry

end find_constants_l132_132121


namespace no_convolution_square_distrib_l132_132808

open MeasureTheory

variables {P : Measure ℝ} {p : ℝ → ℝ}
variables (c d : ℝ) (hx : 0 < c) (hy : c < d)

noncomputable def valid_density_function (p : ℝ → ℝ) : Prop :=
  ∀ x, (-1 ≤ x ∧ x ≤ 1) → c < p x ∧ p x < d

theorem no_convolution_square_distrib
  (symm : ∀ s, P s = P (-s))
  (abs_cont : ∀ s, Real.measure_theory.is_absolutely_continuous P volume)
  (bounded_density : ∀ x, ¬ (-1 ≤ x ∧ x ≤ 1) → p x = 0)
  (density_bound : valid_density_function p c d):
  ¬ ∃ Q : Measure ℝ, convolution Q Q = P :=
sorry

end no_convolution_square_distrib_l132_132808


namespace platform_length_is_correct_l132_132199

def speed_kmph : ℝ := 72
def seconds_to_cross_platform : ℝ := 26
def train_length_m : ℝ := 270.0416

noncomputable def length_of_platform : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * seconds_to_cross_platform
  total_distance - train_length_m

theorem platform_length_is_correct : 
  length_of_platform = 249.9584 := 
by
  sorry

end platform_length_is_correct_l132_132199


namespace train_speed_kmph_l132_132322

def length_of_train : ℝ := 120
def time_to_cross_bridge : ℝ := 17.39860811135109
def length_of_bridge : ℝ := 170

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60 := 
by
  sorry

end train_speed_kmph_l132_132322


namespace total_pizzas_served_l132_132318

-- Define the conditions
def pizzas_lunch : Nat := 9
def pizzas_dinner : Nat := 6

-- Define the theorem to prove
theorem total_pizzas_served : pizzas_lunch + pizzas_dinner = 15 := by
  sorry

end total_pizzas_served_l132_132318


namespace binary_to_decimal_l132_132739

theorem binary_to_decimal :
  1 * 2^8 + 0 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 379 :=
by
  sorry

end binary_to_decimal_l132_132739


namespace travelers_on_liner_l132_132161

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l132_132161


namespace angle_complement_supplement_l132_132627

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132627


namespace part_one_part_two_l132_132068

def f (a x : ℝ) : ℝ := abs (x - a ^ 2) + abs (x + 2 * a + 3)

theorem part_one (a x : ℝ) : f a x ≥ 2 :=
by 
  sorry

noncomputable def f_neg_three_over_two (a : ℝ) : ℝ := f a (-3/2)

theorem part_two (a : ℝ) (h : f_neg_three_over_two a < 3) : -1 < a ∧ a < 0 :=
by 
  sorry

end part_one_part_two_l132_132068


namespace simplify_and_evaluate_l132_132141

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate_l132_132141


namespace smallest_y_exists_l132_132153

theorem smallest_y_exists (M : ℤ) (y : ℕ) (h : 2520 * y = M ^ 3) : y = 3675 :=
by
  have h_factorization : 2520 = 2^3 * 3^2 * 5 * 7 := sorry
  sorry

end smallest_y_exists_l132_132153


namespace angle_supplement_complement_l132_132587

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132587


namespace range_of_a_l132_132245

noncomputable def f (x : ℝ) : ℝ := 6 / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 + a * x - 6 > 0) ↔ 5 ≤ a :=
by
  sorry

end range_of_a_l132_132245


namespace interior_angle_regular_octagon_l132_132435

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l132_132435


namespace wrapping_paper_needed_l132_132874

-- Define the conditions as variables in Lean
def wrapping_paper_first := 3.5
def wrapping_paper_second := (2 / 3) * wrapping_paper_first
def wrapping_paper_third := wrapping_paper_second + 0.5 * wrapping_paper_second
def wrapping_paper_fourth := wrapping_paper_first + wrapping_paper_second
def wrapping_paper_fifth := wrapping_paper_third - 0.25 * wrapping_paper_third

-- Define the total wrapping paper needed
def total_wrapping_paper := wrapping_paper_first + wrapping_paper_second + wrapping_paper_third + wrapping_paper_fourth + wrapping_paper_fifth

-- Statement to prove the final equivalence
theorem wrapping_paper_needed : 
  total_wrapping_paper = 17.79 := 
sorry  -- Proof is omitted

end wrapping_paper_needed_l132_132874


namespace combined_population_l132_132127

-- Defining the conditions
def population_New_England : ℕ := 2100000

def population_New_York (p_NE : ℕ) : ℕ := (2 / 3 : ℚ) * p_NE

-- The theorem to be proven
theorem combined_population (p_NE : ℕ) (h1 : p_NE = population_New_England) : 
  population_New_York p_NE + p_NE = 3500000 :=
by
  sorry

end combined_population_l132_132127


namespace regular_octagon_interior_angle_l132_132459

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l132_132459


namespace machine_parts_probabilities_l132_132175

-- Define the yield rates for the two machines
def yield_rate_A : ℝ := 0.8
def yield_rate_B : ℝ := 0.9

-- Define the probabilities of defectiveness for each machine
def defective_probability_A := 1 - yield_rate_A
def defective_probability_B := 1 - yield_rate_B

theorem machine_parts_probabilities :
  (defective_probability_A * defective_probability_B = 0.02) ∧
  (((yield_rate_A * defective_probability_B) + (defective_probability_A * yield_rate_B)) = 0.26) ∧
  (defective_probability_A * defective_probability_B + (1 - (defective_probability_A * defective_probability_B)) = 1) :=
by
  sorry

end machine_parts_probabilities_l132_132175


namespace angle_supplement_complement_l132_132606

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132606


namespace projectile_first_reaches_70_feet_l132_132148

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ, t = 7/4 ∧ 0 < t ∧ ∀ s : ℝ, s < t → -16 * s^2 + 80 * s < 70 :=
by 
  sorry

end projectile_first_reaches_70_feet_l132_132148


namespace walt_age_l132_132003

theorem walt_age (T W : ℕ) 
  (h1 : T = 3 * W)
  (h2 : T + 12 = 2 * (W + 12)) : 
  W = 12 :=
by
  sorry

end walt_age_l132_132003


namespace inning_is_31_l132_132984

noncomputable def inning_number (s: ℕ) (i: ℕ) (a: ℕ) : ℕ := s - a + i

theorem inning_is_31
  (batsman_runs: ℕ)
  (increase_average: ℕ)
  (final_average: ℕ) 
  (n: ℕ) 
  (h1: batsman_runs = 92)
  (h2: increase_average = 3)
  (h3: final_average = 44)
  (h4: 44 * n - 92 = 41 * n): 
  inning_number 44 1 3 = 31 := 
by 
  sorry

end inning_is_31_l132_132984


namespace regular_octagon_interior_angle_l132_132375

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l132_132375


namespace average_weight_of_three_l132_132799

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l132_132799


namespace regular_octagon_angle_l132_132453

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l132_132453


namespace solve_for_alpha_l132_132481

variables (α β γ δ : ℝ)

theorem solve_for_alpha (h : α + β + γ + δ = 360) : α = 360 - β - γ - δ :=
by sorry

end solve_for_alpha_l132_132481


namespace quarters_range_difference_l132_132946

theorem quarters_range_difference (n d q : ℕ) (h1 : n + d + q = 150) (h2 : 5 * n + 10 * d + 25 * q = 2000) :
  let max_quarters := 0
  let min_quarters := 62
  (max_quarters - min_quarters) = 62 :=
by
  let max_quarters := 0
  let min_quarters := 62
  sorry

end quarters_range_difference_l132_132946


namespace distance_of_route_l132_132047

theorem distance_of_route (Vq : ℝ) (Vy : ℝ) (D : ℝ) (h1 : Vy = 1.5 * Vq) (h2 : D = Vq * 2) (h3 : D = Vy * 1.3333333333333333) : D = 1.5 :=
by
  sorry

end distance_of_route_l132_132047


namespace correct_statement_is_D_l132_132299

/-
Given the following statements and their conditions:
A: Conducting a comprehensive survey is not an accurate approach to understand the sleep situation of middle school students in Changsha.
B: The mode of the dataset \(-1\), \(2\), \(5\), \(5\), \(7\), \(7\), \(4\) is not \(7\) only, because both \(5\) and \(7\) are modes.
C: A probability of precipitation of \(90\%\) does not guarantee it will rain tomorrow.
D: If two datasets, A and B, have the same mean, and the variances \(s_{A}^{2} = 0.3\) and \(s_{B}^{2} = 0.02\), then set B with a lower variance \(s_{B}^{2}\) is more stable.

Prove that the correct statement based on these conditions is D.
-/
theorem correct_statement_is_D
  (dataset_A dataset_B : Type)
  (mean_A mean_B : ℝ)
  (sA2 sB2 : ℝ)
  (h_same_mean: mean_A = mean_B)
  (h_variances: sA2 = 0.3 ∧ sB2 = 0.02)
  (h_stability: sA2 > sB2) :
  (if sA2 = 0.3 ∧ sB2 = 0.02 ∧ sA2 > sB2 then "D" else "not D") = "D" := by
  sorry

end correct_statement_is_D_l132_132299


namespace regular_octagon_interior_angle_l132_132418

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l132_132418


namespace angle_measure_l132_132618

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132618


namespace regular_octagon_interior_angle_l132_132399

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l132_132399


namespace solve_equation_l132_132520

theorem solve_equation:
  ∀ x y z : ℝ, x^2 + 5 * y^2 + 5 * z^2 - 4 * x * z - 2 * y - 4 * y * z + 1 = 0 → 
    x = 4 ∧ y = 1 ∧ z = 2 :=
by
  intros x y z h
  sorry

end solve_equation_l132_132520


namespace points_in_rectangle_distance_l132_132478

/-- In a 3x4 rectangle, if 4 points are randomly located, 
    then the distance between at least two of them is at most 25/8. -/
theorem points_in_rectangle_distance (a b : ℝ) (h₁ : a = 3) (h₂ : b = 4)
  {points : Fin 4 → ℝ × ℝ}
  (h₃ : ∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ a)
  (h₄ : ∀ i, 0 ≤ (points i).2 ∧ (points i).2 ≤ b) :
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 25 / 8 := 
by
  sorry

end points_in_rectangle_distance_l132_132478


namespace prism_surface_area_l132_132786

theorem prism_surface_area (P : ℝ) (h : ℝ) (S : ℝ) (s: ℝ) 
  (hP : P = 4)
  (hh : h = 2) 
  (hs : s = 1) 
  (h_surf_top : S = s * s) 
  (h_lat : S = 8) : 
  S = 10 := 
sorry

end prism_surface_area_l132_132786


namespace initial_books_count_l132_132170

theorem initial_books_count (x : ℕ) (h : x + 10 = 48) : x = 38 := 
by
  sorry

end initial_books_count_l132_132170


namespace inequality_proof_l132_132348

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l132_132348


namespace area_of_pentagon_correct_l132_132513

noncomputable def area_of_pentagon : ℝ :=
  let AB := 5
  let BC := 3
  let BD := 3
  let AC := Real.sqrt (AB^2 - BC^2)
  let AD := Real.sqrt (AB^2 - BD^2)
  let EC := 1
  let FD := 2
  let AE := AC - EC
  let AF := AD - FD
  let sin_alpha := BC / AB
  let cos_alpha := AC / AB
  let sin_2alpha := 2 * sin_alpha * cos_alpha
  let area_ABC := 0.5 * AB * BC
  let area_AEF := 0.5 * AE * AF * sin_2alpha
  2 * area_ABC - area_AEF

theorem area_of_pentagon_correct :
  area_of_pentagon = 9.12 := sorry

end area_of_pentagon_correct_l132_132513


namespace angle_solution_l132_132643

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132643


namespace philips_painting_total_l132_132937

def total_paintings_after_days (daily_paintings : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + daily_paintings * days

theorem philips_painting_total (daily_paintings initial_paintings days : ℕ) 
  (h1 : daily_paintings = 2) (h2 : initial_paintings = 20) (h3 : days = 30) : 
  total_paintings_after_days daily_paintings initial_paintings days = 80 := 
by
  sorry

end philips_painting_total_l132_132937


namespace angle_measure_l132_132686

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132686


namespace fraction_simplification_l132_132206

-- Define the numerator and denominator based on given conditions
def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512

-- Lean theorem that encapsulates the problem
theorem fraction_simplification : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_simplification_l132_132206


namespace inequality_solution_equality_condition_l132_132140

theorem inequality_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d :=
sorry

end inequality_solution_equality_condition_l132_132140


namespace geometric_sequence_relation_l132_132829

variables {a : ℕ → ℝ} {q : ℝ}
variables {m n p : ℕ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def are_in_geometric_sequence (a : ℕ → ℝ) (m n p : ℕ) : Prop :=
  a n ^ 2 = a m * a p

-- Theorem
theorem geometric_sequence_relation (h_geom : is_geometric_sequence a q) (h_order : are_in_geometric_sequence a m n p) (hq_ne_one : q ≠ 1) :
  2 * n = m + p :=
sorry

end geometric_sequence_relation_l132_132829


namespace correct_arrangements_count_l132_132968

def valid_arrangements_count : Nat :=
  let houses := ['O', 'R', 'B', 'Y', 'G']
  let arrangements := houses.permutations
  let valid_arr := arrangements.filter (fun a =>
    let o_idx := a.indexOf 'O'
    let r_idx := a.indexOf 'R'
    let b_idx := a.indexOf 'B'
    let y_idx := a.indexOf 'Y'
    let constraints_met :=
      o_idx < r_idx ∧       -- O before R
      b_idx < y_idx ∧       -- B before Y
      (b_idx + 1 != y_idx) ∧ -- B not next to Y
      (r_idx + 1 != b_idx) ∧ -- R not next to B
      (b_idx + 1 != r_idx)   -- symmetrical R not next to B

    constraints_met)
  valid_arr.length

theorem correct_arrangements_count : valid_arrangements_count = 5 :=
  by
    -- To be filled with proof steps.
    sorry

end correct_arrangements_count_l132_132968


namespace bob_grade_is_35_l132_132107

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l132_132107


namespace cuberoot_condition_l132_132468

/-- If \(\sqrt[3]{x-1}=3\), then \((x-1)^2 = 729\). -/
theorem cuberoot_condition (x : ℝ) (h : (x - 1)^(1/3) = 3) : (x - 1)^2 = 729 := 
  sorry

end cuberoot_condition_l132_132468


namespace min_cos_beta_l132_132760

open Real

theorem min_cos_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin (2 * α + β) = (3 / 2) * sin β) :
  cos β = sqrt 5 / 3 := 
sorry

end min_cos_beta_l132_132760


namespace solution_set_of_inequality_l132_132831

theorem solution_set_of_inequality :
  { x : ℝ | (x - 1) / x ≥ 2 } = { x : ℝ | -1 ≤ x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l132_132831


namespace perpendicular_lines_sum_l132_132765

theorem perpendicular_lines_sum (a b : ℝ) :
  (∃ (x y : ℝ), 2 * x - 5 * y + b = 0 ∧ a * x + 4 * y - 2 = 0 ∧ x = 1 ∧ y = -2) ∧
  (-a / 4) * (2 / 5) = -1 →
  a + b = -2 :=
by
  sorry

end perpendicular_lines_sum_l132_132765


namespace product_simplification_l132_132893

variables {a b c : ℝ}

theorem product_simplification (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ac) * ((ab)⁻¹ + (bc)⁻¹ + (ac)⁻¹)) = 
  ((ab + bc + ac)^2) / (abc) := 
sorry

end product_simplification_l132_132893


namespace translation_correct_l132_132246

-- Define the first line l1
def l1 (x : ℝ) : ℝ := 2 * x - 2

-- Define the second line l2
def l2 (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem translation_correct :
  ∀ x : ℝ, l2 x = l1 x + 2 :=
by
  intro x
  unfold l1 l2
  sorry

end translation_correct_l132_132246


namespace regular_octagon_interior_angle_deg_l132_132443

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l132_132443


namespace simplify_fraction_expr_l132_132279

theorem simplify_fraction_expr (a : ℝ) (h : a ≠ 1) : (a / (a - 1) + 1 / (1 - a)) = 1 := by
  sorry

end simplify_fraction_expr_l132_132279


namespace stockings_total_cost_l132_132510

-- Define a function to compute the total cost given the conditions
def total_cost (n_grandchildren n_children : Nat) 
               (stocking_price discount monogram_cost : Nat) : Nat :=
  let total_stockings := n_grandchildren + n_children
  let discounted_price := stocking_price - (stocking_price * discount / 100)
  let total_stockings_cost := discounted_price * total_stockings
  let total_monogram_cost := monogram_cost * total_stockings
  total_stockings_cost + total_monogram_cost

-- Prove that the total cost calculation is correct given the conditions
theorem stockings_total_cost :
  total_cost 5 4 20 10 5 = 207 :=
by
  -- A placeholder for the proof
  sorry

end stockings_total_cost_l132_132510


namespace max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l132_132565

theorem max_value_of_quadratic (r : ℝ) : -7 * r ^ 2 + 50 * r - 20 ≤ 5 :=
by sorry

theorem exists_r_for_max_value_of_quadratic : ∃ r : ℝ, -7 * r ^ 2 + 50 * r - 20 = 5 :=
by sorry

end max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l132_132565


namespace martins_spending_l132_132125

-- Define the conditions:
def dailyBerryConsumption : ℚ := 1 / 2
def costPerCup : ℚ := 2
def days : ℕ := 30

-- Define the main theorem:
theorem martins_spending : (dailyBerryConsumption * days * costPerCup) = 30 := by
  -- This is where the proof would go.
  sorry

end martins_spending_l132_132125


namespace angle_measure_l132_132668

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132668


namespace first_sequence_correct_second_sequence_correct_l132_132228

theorem first_sequence_correct (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 = 12) (h2 : a2 = a1 + 4) (h3 : a3 = a2 + 4) (h4 : a4 = a3 + 4) (h5 : a5 = a4 + 4) :
  a4 = 24 ∧ a5 = 28 :=
by sorry

theorem second_sequence_correct (b1 b2 b3 b4 b5 : ℕ) (h1 : b1 = 2) (h2 : b2 = b1 * 2) (h3 : b3 = b2 * 2) (h4 : b4 = b3 * 2) (h5 : b5 = b4 * 2) :
  b4 = 16 ∧ b5 = 32 :=
by sorry

end first_sequence_correct_second_sequence_correct_l132_132228


namespace interior_angle_regular_octagon_l132_132431

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l132_132431


namespace angle_solution_l132_132652

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132652


namespace angle_measure_is_60_l132_132632

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132632


namespace circle_properties_intercept_length_l132_132070

theorem circle_properties (a r : ℝ) (h1 : a^2 + 16 = r^2) (h2 : (6 - a)^2 + 16 = r^2) (h3 : r > 0) :
  a = 3 ∧ r = 5 :=
by
  sorry

theorem intercept_length (m : ℝ) (h : |24 + m| / 5 = 3) :
  m = -4 ∨ m = -44 :=
by
  sorry

end circle_properties_intercept_length_l132_132070


namespace how_many_oranges_put_back_l132_132871

variables (A O x : ℕ)

-- Conditions: prices and initial selection.
def price_apple (A : ℕ) : ℕ := 40 * A
def price_orange (O : ℕ) : ℕ := 60 * O
def total_fruit := 20
def average_price_initial : ℕ := 56 -- Average price in cents

-- Conditions: equation from initial average price.
def total_initial_cost := total_fruit * average_price_initial
axiom initial_cost_eq : price_apple A + price_orange O = total_initial_cost
axiom total_fruit_eq : A + O = total_fruit

-- New conditions: desired average price and number of fruits
def average_price_new : ℕ := 52 -- Average price in cents
axiom new_cost_eq : price_apple A + price_orange (O - x) = (total_fruit - x) * average_price_new

-- The statement to be proven
theorem how_many_oranges_put_back : 40 * A + 60 * (O - 10) = (total_fruit - 10) * 52 → x = 10 :=
sorry

end how_many_oranges_put_back_l132_132871


namespace regular_octagon_interior_angle_l132_132456

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l132_132456


namespace regular_octagon_interior_angle_l132_132457

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l132_132457


namespace lateral_surface_area_l132_132544

open Real

-- The sine of the dihedral angle at the lateral edge of a regular quadrilateral pyramid
def sin_dihedral_angle : ℝ := 15 / 17

-- The area of the pyramid's diagonal section
def area_diagonal_section : ℝ := 3 * sqrt 34

-- The statement that we need to prove
theorem lateral_surface_area (sin_dihedral_angle = 15 / 17) (area_diagonal_section = 3 * sqrt 34) : 
  lateral_surface_area = 68 :=
sorry

end lateral_surface_area_l132_132544


namespace find_abc_sum_l132_132362

theorem find_abc_sum (a b c : ℤ) (h1 : a - 2 * b = 4) (h2 : a * b + c^2 - 1 = 0) :
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
  sorry

end find_abc_sum_l132_132362


namespace triangle_angle_B_l132_132925

theorem triangle_angle_B (a b A B : ℝ) (h1 : a * Real.cos B = 3 * b * Real.cos A) (h2 : B = A - Real.pi / 6) : 
  B = Real.pi / 6 := by
  sorry

end triangle_angle_B_l132_132925


namespace perfect_square_trinomial_l132_132100

theorem perfect_square_trinomial (m : ℝ) :
  (∃ (a : ℝ), (x^2 + mx + 1) = (x + a)^2) ↔ (m = 2 ∨ m = -2) := sorry

end perfect_square_trinomial_l132_132100


namespace line_slope_is_negative_three_halves_l132_132296

theorem line_slope_is_negative_three_halves : 
  ∀ (x y : ℝ), (4 * y = -6 * x + 12) → (∀ x y, y = -((3/2) * x) + 3) :=
begin
  sorry
end

end line_slope_is_negative_three_halves_l132_132296


namespace speed_of_stream_l132_132849

theorem speed_of_stream (D v : ℝ) (h1 : ∀ D, D / (54 - v) = 2 * (D / (54 + v))) : v = 18 := 
sorry

end speed_of_stream_l132_132849


namespace jack_total_dollars_l132_132801

-- Constants
def initial_dollars : ℝ := 45
def euro_amount : ℝ := 36
def yen_amount : ℝ := 1350
def ruble_amount : ℝ := 1500
def euro_to_dollar : ℝ := 2
def yen_to_dollar : ℝ := 0.009
def ruble_to_dollar : ℝ := 0.013
def transaction_fee_rate : ℝ := 0.01
def spending_rate : ℝ := 0.1

-- Convert each foreign currency to dollars
def euros_to_dollars : ℝ := euro_amount * euro_to_dollar
def yen_to_dollars : ℝ := yen_amount * yen_to_dollar
def rubles_to_dollars : ℝ := ruble_amount * ruble_to_dollar

-- Calculate transaction fees for each currency conversion
def euros_fee : ℝ := euros_to_dollars * transaction_fee_rate
def yen_fee : ℝ := yen_to_dollars * transaction_fee_rate
def rubles_fee : ℝ := rubles_to_dollars * transaction_fee_rate

-- Subtract transaction fees from the converted amounts
def euros_after_fee : ℝ := euros_to_dollars - euros_fee
def yen_after_fee : ℝ := yen_to_dollars - yen_fee
def rubles_after_fee : ℝ := rubles_to_dollars - rubles_fee

-- Calculate total dollars after conversion and fees
def total_dollars_before_spending : ℝ := initial_dollars + euros_after_fee + yen_after_fee + rubles_after_fee

-- Calculate 10% expenditure
def spending_amount : ℝ := total_dollars_before_spending * spending_rate

-- Calculate final amount after spending
def final_amount : ℝ := total_dollars_before_spending - spending_amount

theorem jack_total_dollars : final_amount = 132.85 := by
  sorry

end jack_total_dollars_l132_132801


namespace pizzas_served_during_lunch_l132_132715

theorem pizzas_served_during_lunch {total_pizzas dinner_pizzas lunch_pizzas: ℕ} 
(h_total: total_pizzas = 15) (h_dinner: dinner_pizzas = 6) (h_eq: total_pizzas = dinner_pizzas + lunch_pizzas) : 
lunch_pizzas = 9 := by
  sorry

end pizzas_served_during_lunch_l132_132715


namespace sum_three_numbers_l132_132963

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_three_numbers_l132_132963


namespace weight_difference_l132_132282

-- Defining the weights of the individuals
variables (a b c d e : ℝ)

-- Given conditions as hypotheses
def conditions :=
  (a = 75) ∧
  ((a + b + c) / 3 = 84) ∧
  ((a + b + c + d) / 4 = 80) ∧
  ((b + c + d + e) / 4 = 79)

-- Theorem statement to prove the desired result
theorem weight_difference (h : conditions a b c d e) : e - d = 3 :=
by
  sorry

end weight_difference_l132_132282


namespace third_measurement_multiple_of_one_l132_132002

-- Define the lengths in meters
def length1_meter : ℕ := 6
def length2_meter : ℕ := 5

-- Convert lengths to centimeters
def length1_cm := length1_meter * 100
def length2_cm := length2_meter * 100

-- Define that the greatest common divisor (gcd) of lengths in cm is 100 cm
def gcd_length : ℕ := Nat.gcd length1_cm length2_cm

-- Given that the gcd is 100 cm
theorem third_measurement_multiple_of_one
  (h1 : gcd_length = 100) :
  ∃ n : ℕ, n = 1 :=
sorry

end third_measurement_multiple_of_one_l132_132002


namespace angle_measure_l132_132592

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132592


namespace common_chord_l132_132824

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- The common chord is the line where both circle equations are satisfied
theorem common_chord (x y : ℝ) : circle1 x y ∧ circle2 x y → x - 2*y + 5 = 0 :=
sorry

end common_chord_l132_132824


namespace quadratic_inequality_iff_abs_a_le_2_l132_132058

theorem quadratic_inequality_iff_abs_a_le_2 (a : ℝ) :
  (|a| ≤ 2) ↔ (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) :=
sorry

end quadratic_inequality_iff_abs_a_le_2_l132_132058


namespace find_width_of_room_l132_132490

variable (length : ℕ) (total_carpet_owned : ℕ) (additional_carpet_needed : ℕ)
variable (total_area : ℕ) (width : ℕ)

theorem find_width_of_room
  (h1 : length = 11) 
  (h2 : total_carpet_owned = 16) 
  (h3 : additional_carpet_needed = 149)
  (h4 : total_area = total_carpet_owned + additional_carpet_needed) 
  (h5 : total_area = length * width) :
  width = 15 := by
    sorry

end find_width_of_room_l132_132490


namespace angle_supplement_complement_l132_132661

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132661


namespace count_distinct_digits_l132_132036

theorem count_distinct_digits (n : ℕ) (h1 : ∃ (n : ℕ), n^3 = 125) : 
  n = 5 :=
by
  sorry

end count_distinct_digits_l132_132036


namespace find_theta_even_fn_l132_132764

noncomputable def f (x θ : ℝ) := Real.sin (x + θ) + Real.cos (x + θ)

theorem find_theta_even_fn (θ : ℝ) (hθ: 0 ≤ θ ∧ θ ≤ π / 2) 
  (h: ∀ x : ℝ, f x θ = f (-x) θ) : θ = π / 4 :=
by sorry

end find_theta_even_fn_l132_132764


namespace all_stones_weigh_the_same_l132_132012

theorem all_stones_weigh_the_same (x : Fin 13 → ℕ)
  (h : ∀ (i : Fin 13), ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧
    i ∉ A ∧ i ∉ B ∧ ∀ (j k : Fin 13), j ∈ A → k ∈ B → x j = x k): 
  ∀ i j : Fin 13, x i = x j := 
sorry

end all_stones_weigh_the_same_l132_132012


namespace travelers_on_liner_l132_132168

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l132_132168


namespace problem_3_div_27_l132_132065

theorem problem_3_div_27 (a b : ℕ) (h : 2^a = 8^(b + 1)) : 3^a / 27^b = 27 := by
  -- proof goes here
  sorry

end problem_3_div_27_l132_132065


namespace find_general_formula_sum_b_n_less_than_two_l132_132069

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def S_n (n : ℕ) : ℚ := (n^2 + n) / 2

noncomputable def b_n (n : ℕ) : ℚ := 1 / S_n n

theorem find_general_formula (n : ℕ) : b_n n = 2 / (n^2 + n) := by 
  sorry

theorem sum_b_n_less_than_two (n : ℕ) :
  Finset.sum (Finset.range n) (λ k => b_n (k + 1)) < 2 :=
by 
  sorry

end find_general_formula_sum_b_n_less_than_two_l132_132069


namespace arrangement_ways_l132_132730

-- Defining the conditions
def num_basil_plants : Nat := 5
def num_tomato_plants : Nat := 4
def num_total_units : Nat := num_basil_plants + 1

-- Proof statement
theorem arrangement_ways : (num_total_units.factorial) * (num_tomato_plants.factorial) = 17280 := by
  sorry

end arrangement_ways_l132_132730


namespace bob_grade_is_35_l132_132109

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l132_132109


namespace remaining_pieces_l132_132743

variables (sh_s : Nat) (sh_f : Nat) (sh_r : Nat)
variables (sh_pairs_s : Nat) (sh_pairs_f : Nat) (sh_pairs_r : Nat)
variables (total : Nat)

def conditions :=
  sh_s = 20 ∧
  sh_f = 12 ∧
  sh_r = 20 - 12 ∧
  sh_pairs_s = 8 ∧
  sh_pairs_f = 5 ∧
  sh_pairs_r = 8 - 5 ∧
  total = sh_r + sh_pairs_r

theorem remaining_pieces : conditions → total = 11 :=
by intro h; cases h with _ h'; cases h' with _ h''; cases h'' with _ h''';
   cases h''' with _ h''''; cases h'''' with _ h''''' ; cases h''''' with _ _ ;
   sorry

end remaining_pieces_l132_132743


namespace find_larger_integer_l132_132850

-- Defining the problem statement with the given conditions
theorem find_larger_integer (x : ℕ) (h : (x + 6) * 2 = 4 * x) : 4 * x = 24 :=
sorry

end find_larger_integer_l132_132850


namespace prob_part1_prob_part2_l132_132936

-- Define the probability that Person A hits the target
def pA : ℚ := 2 / 3

-- Define the probability that Person B hits the target
def pB : ℚ := 3 / 4

-- Define the number of shots
def nShotsA : ℕ := 3
def nShotsB : ℕ := 2

-- The problem posed to Person A
def probA_miss_at_least_once : ℚ := 1 - (pA ^ nShotsA)

-- The problem posed to Person A (exactly twice in 2 shots)
def probA_hits_exactly_twice : ℚ := pA ^ 2

-- The problem posed to Person B (exactly once in 2 shots)
def probB_hits_exactly_once : ℚ :=
  2 * (pB * (1 - pB))

-- The combined probability for Part 2
def combined_prob : ℚ := probA_hits_exactly_twice * probB_hits_exactly_once

theorem prob_part1 :
  probA_miss_at_least_once = 19 / 27 := by
  sorry

theorem prob_part2 :
  combined_prob = 1 / 6 := by
  sorry

end prob_part1_prob_part2_l132_132936


namespace find_angle_QPR_l132_132790

-- Define the angles and line segment
variables (R S Q T P : Type) 
variables (line_RT : R ≠ S)
variables (x : ℝ) 
variables (angle_PTQ : ℝ := 62)
variables (angle_RPS : ℝ := 34)

-- Hypothesis that PQ = PT, making triangle PQT isosceles
axiom eq_PQ_PT : ℝ

-- Conditions
axiom lie_on_RT : ∀ {R S Q T : Type}, R ≠ S 
axiom angle_PTQ_eq : angle_PTQ = 62
axiom angle_RPS_eq : angle_RPS = 34

-- Hypothesis that defines the problem structure
theorem find_angle_QPR : x = 11 := by
sorry

end find_angle_QPR_l132_132790


namespace sum_in_range_l132_132733

open Real

def mix1 := 3 + 3/8
def mix2 := 4 + 2/5
def mix3 := 6 + 1/11
def mixed_sum := mix1 + mix2 + mix3

theorem sum_in_range : mixed_sum > 13 ∧ mixed_sum < 14 :=
by
  -- Since we are just providing the statement, we leave the proof as a placeholder.
  sorry

end sum_in_range_l132_132733


namespace measure_angle_A_l132_132869

-- Angles A and B are supplementary
def supplementary (A B : ℝ) : Prop :=
  A + B = 180

-- Definition of the problem conditions
def problem_conditions (A B : ℝ) : Prop :=
  supplementary A B ∧ A = 4 * B

-- The measure of angle A
def measure_of_A := 144

-- The statement to prove
theorem measure_angle_A (A B : ℝ) :
  problem_conditions A B → A = measure_of_A := 
by
  sorry

end measure_angle_A_l132_132869


namespace arrangement_count_l132_132728

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end arrangement_count_l132_132728


namespace cos_660_degrees_is_one_half_l132_132193

noncomputable def cos_660_eq_one_half : Prop :=
  (Real.cos (660 * Real.pi / 180) = 1 / 2)

theorem cos_660_degrees_is_one_half : cos_660_eq_one_half :=
by
  sorry

end cos_660_degrees_is_one_half_l132_132193


namespace regular_octagon_interior_angle_deg_l132_132444

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l132_132444


namespace slope_of_line_determined_by_solutions_l132_132839

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l132_132839


namespace total_games_across_leagues_l132_132550

-- Defining the conditions for the leagues
def leagueA_teams := 20
def leagueB_teams := 25
def leagueC_teams := 30

-- Function to calculate the number of games in a round-robin tournament
def number_of_games (n : ℕ) := n * (n - 1) / 2

-- Proposition to prove total games across all leagues
theorem total_games_across_leagues :
  number_of_games leagueA_teams + number_of_games leagueB_teams + number_of_games leagueC_teams = 925 := by
  sorry

end total_games_across_leagues_l132_132550


namespace angle_complement_supplement_l132_132628

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132628


namespace radio_advertiser_savings_l132_132548

def total_store_price : ℚ := 299.99
def ad_payment : ℚ := 55.98
def payments_count : ℚ := 5
def shipping_handling : ℚ := 12.99

def total_ad_price : ℚ := payments_count * ad_payment + shipping_handling

def savings_in_dollars : ℚ := total_store_price - total_ad_price
def savings_in_cents : ℚ := savings_in_dollars * 100

theorem radio_advertiser_savings :
  savings_in_cents = 710 := by
  sorry

end radio_advertiser_savings_l132_132548


namespace angle_supplement_complement_l132_132580

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132580


namespace sum_of_fourth_powers_l132_132467

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 18.5 :=
sorry

end sum_of_fourth_powers_l132_132467


namespace hyperbola_parabola_foci_l132_132916

-- Definition of the hyperbola
def hyperbola (k : ℝ) (x y : ℝ) : Prop := y^2 / 5 - x^2 / k = 1

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Condition that both curves have the same foci
def same_foci (focus : ℝ) (x y : ℝ) : Prop := focus = 3 ∧ (parabola x y → ((0, focus) : ℝ×ℝ) = (0, 3)) ∧ (∃ k : ℝ, hyperbola k x y ∧ ((0, focus) : ℝ×ℝ) = (0, 3))

theorem hyperbola_parabola_foci (k : ℝ) (x y : ℝ) : same_foci 3 x y → k = -4 := 
by {
  sorry
}

end hyperbola_parabola_foci_l132_132916


namespace remaining_pieces_to_fold_l132_132742

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end remaining_pieces_to_fold_l132_132742


namespace number_of_ways_to_represent_5030_l132_132495

theorem number_of_ways_to_represent_5030 :
  let even := {x : ℕ | x % 2 = 0}
  let in_range := {x : ℕ | x ≤ 98}
  let valid_b := even ∩ in_range
  ∃ (M : ℕ), M = 150 ∧ ∀ (b3 b2 b1 b0 : ℕ), 
    b3 ∈ valid_b ∧ b2 ∈ valid_b ∧ b1 ∈ valid_b ∧ b0 ∈ valid_b →
    5030 = b3 * 10 ^ 3 + b2 * 10 ^ 2 + b1 * 10 + b0 → 
    M = 150 :=
  sorry

end number_of_ways_to_represent_5030_l132_132495


namespace average_speed_is_70_kmh_l132_132545

-- Define the given conditions
def distance1 : ℕ := 90
def distance2 : ℕ := 50
def time1 : ℕ := 1
def time2 : ℕ := 1

-- We need to prove that the average speed of the car is 70 km/h
theorem average_speed_is_70_kmh :
    ((distance1 + distance2) / (time1 + time2)) = 70 := 
by 
    -- This is the proof placeholder
    sorry

end average_speed_is_70_kmh_l132_132545


namespace find_d_l132_132258

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l132_132258


namespace students_passed_finals_l132_132156

def total_students := 180
def students_bombed := 1 / 4 * total_students
def remaining_students_after_bombed := total_students - students_bombed
def students_didnt_show := 1 / 3 * remaining_students_after_bombed
def students_failed_less_than_D := 20

theorem students_passed_finals : 
  total_students - students_bombed - students_didnt_show - students_failed_less_than_D = 70 := 
by 
  -- calculation to derive 70
  sorry

end students_passed_finals_l132_132156


namespace angle_supplement_complement_l132_132608

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132608


namespace total_distance_is_correct_l132_132115

def Jonathan_d : Real := 7.5

def Mercedes_d (J : Real) : Real := 2 * J

def Davonte_d (M : Real) : Real := M + 2

theorem total_distance_is_correct : 
  let J := Jonathan_d
  let M := Mercedes_d J
  let D := Davonte_d M
  M + D = 32 :=
by
  sorry

end total_distance_is_correct_l132_132115


namespace probability_xyz_72_l132_132688

noncomputable def dice_probability : ℚ :=
  let outcomes : Finset (ℕ × ℕ × ℕ) := 
    {(a, b, c) | a ∈ {1, 2, 3, 4, 5, 6}, b ∈ {1, 2, 3, 4, 5, 6}, c ∈ {1, 2, 3, 4, 5, 6}}.toFinset
  let favorable_outcomes := 
    outcomes.filter (λ (abc : ℕ × ℕ × ℕ), abc.1 * abc.2 * abc.3 = 72)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_xyz_72 : dice_probability = 1/24 :=
  sorry

end probability_xyz_72_l132_132688


namespace regular_octagon_interior_angle_eq_135_l132_132392

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l132_132392


namespace angle_measure_l132_132685

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132685


namespace total_hours_worked_l132_132722

theorem total_hours_worked (amber_hours : ℕ) (armand_hours : ℕ) (ella_hours : ℕ) 
(h_amber : amber_hours = 12) 
(h_armand : armand_hours = (1 / 3) * amber_hours) 
(h_ella : ella_hours = 2 * amber_hours) :
amber_hours + armand_hours + ella_hours = 40 :=
sorry

end total_hours_worked_l132_132722


namespace geo_seq_a12_equal_96_l132_132924

def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_a12_equal_96
  (a : ℕ → ℝ) (q : ℝ)
  (h0 : 1 < q)
  (h1 : is_geometric a q)
  (h2 : a 3 * a 7 = 72)
  (h3 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end geo_seq_a12_equal_96_l132_132924


namespace monday_dressing_time_l132_132132

theorem monday_dressing_time 
  (Tuesday_time Wednesday_time Thursday_time Friday_time Old_average_time : ℕ)
  (H_tuesday : Tuesday_time = 4)
  (H_wednesday : Wednesday_time = 3)
  (H_thursday : Thursday_time = 4)
  (H_friday : Friday_time = 2)
  (H_average : Old_average_time = 3) :
  ∃ Monday_time : ℕ, Monday_time = 2 :=
by
  let Total_time_5_days := Old_average_time * 5
  let Total_time := 4 + 3 + 4 + 2
  let Monday_time := Total_time_5_days - Total_time
  exact ⟨Monday_time, sorry⟩

end monday_dressing_time_l132_132132


namespace botanical_garden_path_length_l132_132858

theorem botanical_garden_path_length
  (scale : ℝ)
  (path_length_map : ℝ)
  (path_length_real : ℝ)
  (h_scale : scale = 500)
  (h_path_length_map : path_length_map = 6.5)
  (h_path_length_real : path_length_real = path_length_map * scale) :
  path_length_real = 3250 :=
by
  sorry

end botanical_garden_path_length_l132_132858


namespace jaya_amitabh_number_of_digits_l132_132254

-- Definitions
def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def digit_sum (n1 n2 : ℕ) : ℕ :=
  let (d1, d2) := (n1 % 10, n1 / 10)
  let (d3, d4) := (n2 % 10, n2 / 10)
  d1 + d2 + d3 + d4
def append_ages (j a : ℕ) : ℕ := 1000 * (j / 10) + 100 * (j % 10) + 10 * (a / 10) + (a % 10)
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Main theorem
theorem jaya_amitabh_number_of_digits 
  (j a : ℕ) 
  (hj : is_two_digit_number j)
  (ha : is_two_digit_number a)
  (h_sum : digit_sum j a = 7)
  (h_square : is_perfect_square (append_ages j a)) : 
  ∃ n : ℕ, String.length (toString (append_ages j a)) = 4 :=
by
  sorry

end jaya_amitabh_number_of_digits_l132_132254


namespace sufficient_but_not_necessary_condition_l132_132192

theorem sufficient_but_not_necessary_condition 
  (a b : ℝ) (h : a > b ∧ b > 0) : (a^2 > b^2) ∧ (¬ ∀ (a' b' : ℝ), a'^2 > b'^2 → a' > b' ∧ b' > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l132_132192


namespace steve_needs_28_feet_of_wood_l132_132529

theorem steve_needs_28_feet_of_wood :
  (6 * 4) + (2 * 2) = 28 := by
  sorry

end steve_needs_28_feet_of_wood_l132_132529


namespace grid_solution_l132_132560

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l132_132560


namespace least_possible_value_z_minus_x_l132_132189

theorem least_possible_value_z_minus_x (x y z : ℤ) (h1 : Even x) (h2 : Odd y) (h3 : Odd z) (h4 : x < y) (h5 : y < z) (h6 : y - x > 5) : z - x = 9 := 
sorry

end least_possible_value_z_minus_x_l132_132189


namespace regular_octagon_interior_angle_deg_l132_132447

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l132_132447


namespace find_cost_price_l132_132848

variable (CP SP1 SP2 : ℝ)

theorem find_cost_price
    (h1 : SP1 = CP * 0.92)
    (h2 : SP2 = CP * 1.04)
    (h3 : SP2 = SP1 + 140) :
    CP = 1166.67 :=
by
  -- Proof would be filled here
  sorry

end find_cost_price_l132_132848


namespace regular_octagon_interior_angle_l132_132384

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l132_132384


namespace exists_divisible_by_2021_l132_132932

def concat_numbers (n m : ℕ) : ℕ :=
  -- function to concatenate numbers from n to m
  sorry

theorem exists_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concat_numbers n m :=
by
  sorry

end exists_divisible_by_2021_l132_132932


namespace line_equation_M_l132_132903

theorem line_equation_M (x y : ℝ) :
  (∃ (m c : ℝ), y = m * x + c ∧ m = -5/4 ∧ c = -3)
  ∧ (∃ (slope intercept : ℝ), slope = 2 * (-5/4) ∧ intercept = (1/2) * -3 ∧ (y - 2 = slope * (x + 4)))
  → ∃ (a b : ℝ), y = a * x + b ∧ a = -5/2 ∧ b = -8 :=
by
  sorry

end line_equation_M_l132_132903


namespace sum_of_fourth_powers_l132_132466

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 18.5 :=
sorry

end sum_of_fourth_powers_l132_132466


namespace initial_members_count_l132_132537

theorem initial_members_count (n : ℕ) (W : ℕ)
  (h1 : W = n * 48)
  (h2 : W + 171 = (n + 2) * 51) : 
  n = 23 :=
by sorry

end initial_members_count_l132_132537


namespace regular_octagon_interior_angle_l132_132382

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l132_132382


namespace exponential_function_decreasing_l132_132284

theorem exponential_function_decreasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1) → ¬ (∀ x : ℝ, x > 0 → a ^ x > 0) :=
by
  sorry

end exponential_function_decreasing_l132_132284


namespace player_winning_strategy_l132_132756

-- Define the game conditions
def Sn (n : ℕ) : Type := Equiv.Perm (Fin n)

def game_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ G : Set (Sn n), ∃ x : Sn n, x ∈ G → G ≠ (Set.univ : Set (Sn n)))

-- Statement of the proof problem
theorem player_winning_strategy (n : ℕ) (hn : n > 1) : 
  ((n = 2 ∨ n = 3) → (∃ strategyA : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyA x x)) ∧ 
  ((n ≥ 4 ∧ n % 2 = 1) → (∃ strategyB : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyB x x)) :=
by
  sorry

end player_winning_strategy_l132_132756


namespace determine_grid_numbers_l132_132558

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l132_132558


namespace num_four_digit_integers_divisible_by_7_l132_132463

theorem num_four_digit_integers_divisible_by_7 :
  ∃ n : ℕ, n = 1286 ∧ ∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999) → (k % 7 = 0 ↔ ∃ m : ℕ, k = m * 7) :=
by {
  sorry
}

end num_four_digit_integers_divisible_by_7_l132_132463


namespace min_value_of_expression_l132_132093

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + (1 / (a - 1)) ≥ 3 :=
by sorry

end min_value_of_expression_l132_132093


namespace interior_angle_regular_octagon_l132_132433

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l132_132433


namespace intersecting_line_circle_condition_l132_132785

theorem intersecting_line_circle_condition {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x / a + y / b = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) ≥ 1 :=
sorry

end intersecting_line_circle_condition_l132_132785


namespace angle_measure_l132_132598

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132598


namespace appropriate_sampling_method_l132_132479

def total_families := 500
def high_income_families := 125
def middle_income_families := 280
def low_income_families := 95
def sample_size := 100
def influenced_by_income := True

theorem appropriate_sampling_method
  (htotal : total_families = 500)
  (hhigh : high_income_families = 125)
  (hmiddle : middle_income_families = 280)
  (hlow : low_income_families = 95)
  (hsample : sample_size = 100)
  (hinfluence : influenced_by_income = True) :
  ∃ method, method = "Stratified sampling method" :=
sorry

end appropriate_sampling_method_l132_132479


namespace pencil_lead_loss_l132_132969

theorem pencil_lead_loss (L r : ℝ) (h : r = L * 1/10):
  ((9/10 * r^3) * (2/3)) / (r^3) = 3/5 := 
by
  sorry

end pencil_lead_loss_l132_132969


namespace Lakers_win_in_7_games_l132_132955

-- Variables for probabilities given in the problem
variable (p_Lakers_win : ℚ := 1 / 4) -- Lakers' probability of winning a single game
variable (p_Celtics_win : ℚ := 3 / 4) -- Celtics' probability of winning a single game

-- Probabilities and combinations
def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_Lakers_win_game7 : ℚ :=
  let first_6_games := binom 6 3 * (p_Lakers_win ^ 3) * (p_Celtics_win ^ 3)
  let seventh_game := p_Lakers_win
  first_6_games * seventh_game

theorem Lakers_win_in_7_games : probability_Lakers_win_game7 = 540 / 16384 := by
  sorry

end Lakers_win_in_7_games_l132_132955


namespace regular_octagon_interior_angle_l132_132437

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l132_132437


namespace angle_supplement_complement_l132_132579

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132579


namespace TylerWeightDifference_l132_132176

-- Define the problem conditions
def PeterWeight : ℕ := 65
def SamWeight : ℕ := 105
def TylerWeight := 2 * PeterWeight

-- State the theorem
theorem TylerWeightDifference : (TylerWeight - SamWeight = 25) :=
by
  -- proof goes here
  sorry

end TylerWeightDifference_l132_132176


namespace eccentricity_hyperbola_l132_132001

theorem eccentricity_hyperbola : 
  let a2 := 4
  let b2 := 5
  let e := Real.sqrt (1 + (b2 / a2))
  e = 3 / 2 := by
    apply sorry

end eccentricity_hyperbola_l132_132001


namespace each_interior_angle_of_regular_octagon_l132_132404

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l132_132404


namespace minimum_value_of_expression_l132_132256

variable (a b c : ℝ)

noncomputable def expression (a b c : ℝ) := (a + b) / c + (a + c) / b + (b + c) / a

theorem minimum_value_of_expression (hp1 : 0 < a) (hp2 : 0 < b) (hp3 : 0 < c) (h1 : a = 2 * b) (h2 : a = 2 * c) :
  expression a b c = 9.25 := 
sorry

end minimum_value_of_expression_l132_132256


namespace angle_supplement_complement_l132_132599

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132599


namespace convex_ngon_can_be_divided_l132_132942

open Convex

theorem convex_ngon_can_be_divided (n : ℕ) (h : n ≥ 6) :
  ∃ (S : finset (fin 5 → ℝ)) (S' : finset (fin 5 → ℝ)),
    (∀ s ∈ S, Convex ℝ (finset.image id s)) ∧ 
    (∀ s' ∈ S', Convex ℝ (finset.image id s')) ∧
    (Convex ℝ (⋃₀ (S ∪ S'))) ∧
    (finset.card S + finset.card S' = n) :=
by
  sorry

end convex_ngon_can_be_divided_l132_132942


namespace min_value_frac_sq_l132_132339

theorem min_value_frac_sq (x : ℝ) (h : x > 12) : (x^2 / (x - 12)) >= 48 :=
by
  sorry

end min_value_frac_sq_l132_132339


namespace find_the_number_l132_132983

theorem find_the_number :
  ∃ X : ℝ, (66.2 = (6.620000000000001 / 100) * X) ∧ X = 1000 :=
by
  sorry

end find_the_number_l132_132983


namespace kids_played_on_tuesday_l132_132119

-- Define the total number of kids Julia played with
def total_kids : ℕ := 18

-- Define the number of kids Julia played with on Monday
def monday_kids : ℕ := 4

-- Define the number of kids Julia played with on Tuesday
def tuesday_kids : ℕ := total_kids - monday_kids

-- The proof goal:
theorem kids_played_on_tuesday : tuesday_kids = 14 :=
by sorry

end kids_played_on_tuesday_l132_132119


namespace average_weight_of_three_l132_132798

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l132_132798


namespace remaining_pieces_l132_132744

variables (sh_s : Nat) (sh_f : Nat) (sh_r : Nat)
variables (sh_pairs_s : Nat) (sh_pairs_f : Nat) (sh_pairs_r : Nat)
variables (total : Nat)

def conditions :=
  sh_s = 20 ∧
  sh_f = 12 ∧
  sh_r = 20 - 12 ∧
  sh_pairs_s = 8 ∧
  sh_pairs_f = 5 ∧
  sh_pairs_r = 8 - 5 ∧
  total = sh_r + sh_pairs_r

theorem remaining_pieces : conditions → total = 11 :=
by intro h; cases h with _ h'; cases h' with _ h''; cases h'' with _ h''';
   cases h''' with _ h''''; cases h'''' with _ h''''' ; cases h''''' with _ _ ;
   sorry

end remaining_pieces_l132_132744


namespace selina_sold_shirts_l132_132949

/-- Selina's selling problem -/
theorem selina_sold_shirts :
  let pants_price := 5
  let shorts_price := 3
  let shirts_price := 4
  let num_pants := 3
  let num_shorts := 5
  let remaining_money := 30 + (2 * 10)
  let money_from_pants := num_pants * pants_price
  let money_from_shorts := num_shorts * shorts_price
  let total_money_from_pants_and_shorts := money_from_pants + money_from_shorts
  let total_money_from_shirts := remaining_money - total_money_from_pants_and_shorts
  let num_shirts := total_money_from_shirts / shirts_price
  num_shirts = 5 := by
{
  sorry
}

end selina_sold_shirts_l132_132949


namespace fewerCansCollected_l132_132696

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end fewerCansCollected_l132_132696


namespace total_caffeine_is_correct_l132_132804

def first_drink_caffeine := 250 -- milligrams
def first_drink_size := 12 -- ounces

def second_drink_caffeine_per_ounce := (first_drink_caffeine / first_drink_size) * 3
def second_drink_size := 8 -- ounces
def second_drink_caffeine := second_drink_caffeine_per_ounce * second_drink_size

def third_drink_concentration := 18 -- milligrams per milliliter
def third_drink_size := 150 -- milliliters
def third_drink_caffeine := third_drink_concentration * third_drink_size

def caffeine_pill_caffeine := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine

def total_caffeine_consumed := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine + caffeine_pill_caffeine

theorem total_caffeine_is_correct : total_caffeine_consumed = 6900 :=
by
  sorry

end total_caffeine_is_correct_l132_132804


namespace problem_equivalence_l132_132811

variable {x y z w : ℝ}

theorem problem_equivalence (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end problem_equivalence_l132_132811


namespace regular_octagon_interior_angle_l132_132413

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l132_132413


namespace angle_measure_l132_132571

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132571


namespace interior_angle_regular_octagon_l132_132427

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l132_132427


namespace stockings_total_cost_l132_132511

-- Define a function to compute the total cost given the conditions
def total_cost (n_grandchildren n_children : Nat) 
               (stocking_price discount monogram_cost : Nat) : Nat :=
  let total_stockings := n_grandchildren + n_children
  let discounted_price := stocking_price - (stocking_price * discount / 100)
  let total_stockings_cost := discounted_price * total_stockings
  let total_monogram_cost := monogram_cost * total_stockings
  total_stockings_cost + total_monogram_cost

-- Prove that the total cost calculation is correct given the conditions
theorem stockings_total_cost :
  total_cost 5 4 20 10 5 = 207 :=
by
  -- A placeholder for the proof
  sorry

end stockings_total_cost_l132_132511


namespace simplify_and_evaluate_expression_l132_132278

variable (x y : ℚ)

theorem simplify_and_evaluate_expression :
    x = 2 / 15 → y = 3 / 2 → 
    (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 :=
by 
  intros h1 h2
  subst h1
  subst h2
  sorry

end simplify_and_evaluate_expression_l132_132278


namespace interior_angle_regular_octagon_l132_132410

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l132_132410


namespace max_value_x_l132_132338

theorem max_value_x : ∃ x, x ^ 2 = 38 ∧ x = Real.sqrt 38 := by
  sorry

end max_value_x_l132_132338


namespace interior_angle_regular_octagon_l132_132423

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l132_132423


namespace angle_measure_l132_132666

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132666


namespace garden_ratio_l132_132037

theorem garden_ratio (L W : ℕ) (h1 : L = 50) (h2 : 2 * L + 2 * W = 150) : L / W = 2 :=
by
  sorry

end garden_ratio_l132_132037


namespace stratified_sampling_l132_132039

theorem stratified_sampling (teachers male_students female_students total_pop sample_female_students proportion_total n : ℕ)
    (h_teachers : teachers = 200)
    (h_male_students : male_students = 1200)
    (h_female_students : female_students = 1000)
    (h_total_pop : total_pop = teachers + male_students + female_students)
    (h_sample_female_students : sample_female_students = 80)
    (h_proportion_total : proportion_total = female_students / total_pop)
    (h_proportion_equation : sample_female_students = proportion_total * n) :
  n = 192 :=
by
  sorry

end stratified_sampling_l132_132039


namespace range_of_m_common_tangents_with_opposite_abscissas_l132_132122

section part1
variable {x : ℝ}

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def h (m : ℝ) (x : ℝ) := m * f x / Real.sin x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 0 Real.pi, h m x ≥ Real.sqrt 2) ↔ m ∈ Set.Ici (Real.sqrt 2 / Real.exp (Real.pi / 4)) := 
by
  sorry
end part1

section part2
variable {x : ℝ}

noncomputable def g (x : ℝ) := Real.log x
noncomputable def f_tangent_line_at (x₁ : ℝ) (x : ℝ) := Real.exp x₁ * x + (1 - x₁) * Real.exp x₁
noncomputable def g_tangent_line_at (x₂ : ℝ) (x : ℝ) := x / x₂ + Real.log x₂ - 1

theorem common_tangents_with_opposite_abscissas :
  ∃ x₁ x₂ : ℝ, (f_tangent_line_at x₁ = g_tangent_line_at (Real.exp (-x₁))) ∧ (x₁ = -x₂) :=
by
  sorry
end part2

end range_of_m_common_tangents_with_opposite_abscissas_l132_132122


namespace last_two_nonzero_digits_70_factorial_l132_132150

theorem last_two_nonzero_digits_70_factorial : 
  let N := 70
  (∀ N : ℕ, 0 < N → N % 2 ≠ 0 → N % 5 ≠ 0 → ∃ x : ℕ, x % 100 = N % (N + (N! / (2 ^ 16)))) →
  (N! / 10 ^ 16) % 100 = 68 :=
by
sorry

end last_two_nonzero_digits_70_factorial_l132_132150


namespace triangle_area_l132_132922

theorem triangle_area (a b : ℝ) (h1 : b = (24 / a)) (h2 : 3 * 4 + a * (12 / a) = 12) : b = 3 / 2 :=
by
  sorry

end triangle_area_l132_132922


namespace solve_for_b_l132_132779

def is_imaginary (z : ℂ) : Prop := z.re = 0

theorem solve_for_b (b : ℝ) (i_is_imag_unit : ∀ (z : ℂ), i * z = z * i):
  is_imaginary (i * (b * i + 1)) → b = 0 :=
by
  sorry

end solve_for_b_l132_132779


namespace inequality_proof_l132_132349

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l132_132349


namespace number_of_teams_l132_132920

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
sorry

end number_of_teams_l132_132920


namespace sum_of_a_and_b_l132_132758

theorem sum_of_a_and_b {a b : ℝ} (h : a^2 + b^2 + (a*b)^2 = 4*a*b - 1) : a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_and_b_l132_132758


namespace problem1_problem2_l132_132771

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 0)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vec_a x
  let b := vec_b x
  (a.1 * b.1 + a.2 * b.2) + (a.1 * a.1 + a.2 * a.2)

theorem problem1 (k : ℤ) (x : ℝ) :
  x ∈ Set.Icc (k * Real.pi + 3 * Real.pi / 8) (k * Real.pi + 7 * Real.pi / 8) →
  Monotone.decreasingOn f fun y => y ∈ Set.Icc (k * Real.pi + 3 * Real.pi / 8) (k * Real.pi + 7 * Real.pi / 8) :=
sorry

theorem problem2 (A B : ℝ) (ABC : {A, B, C : ℝ // A + B + C = Real.pi}) :
  f (A / 2) = 1 →
  0 < f B ∧ f B ≤ (Real.sqrt 2 + 1) / 2 :=
sorry

end problem1_problem2_l132_132771


namespace water_bill_payment_ratio_l132_132526

variables (electricity_bill gas_bill water_bill internet_bill amount_remaining : ℤ)
variables (paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment : ℤ)

-- Define the given conditions
def stephanie_budget := 
  electricity_bill = 60 ∧
  gas_bill = 40 ∧
  water_bill = 40 ∧
  internet_bill = 25 ∧
  amount_remaining = 30 ∧
  paid_gas_bill_payments = 3 ∧ -- three-quarters
  paid_internet_bill_payments = 4 ∧ -- four payments of $5
  additional_gas_payment = 5

-- Define the given problem as a theorem
theorem water_bill_payment_ratio 
  (h : stephanie_budget electricity_bill gas_bill water_bill internet_bill amount_remaining paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment) :
  ∃ (paid_water_bill : ℤ), paid_water_bill / water_bill = 1 / 2 :=
sorry

end water_bill_payment_ratio_l132_132526


namespace students_passed_finals_l132_132155

def total_students := 180
def students_bombed := 1 / 4 * total_students
def remaining_students_after_bombed := total_students - students_bombed
def students_didnt_show := 1 / 3 * remaining_students_after_bombed
def students_failed_less_than_D := 20

theorem students_passed_finals : 
  total_students - students_bombed - students_didnt_show - students_failed_less_than_D = 70 := 
by 
  -- calculation to derive 70
  sorry

end students_passed_finals_l132_132155


namespace find_d_l132_132257

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l132_132257


namespace interior_angle_regular_octagon_l132_132426

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l132_132426


namespace smaller_cuboid_length_l132_132089

theorem smaller_cuboid_length
  (width_sm : ℝ)
  (height_sm : ℝ)
  (length_lg : ℝ)
  (width_lg : ℝ)
  (height_lg : ℝ)
  (num_sm : ℝ)
  (h1 : width_sm = 2)
  (h2 : height_sm = 3)
  (h3 : length_lg = 18)
  (h4 : width_lg = 15)
  (h5 : height_lg = 2)
  (h6 : num_sm = 18) :
  ∃ (length_sm : ℝ), (108 * length_sm = 540) ∧ (length_sm = 5) :=
by
  -- proof logic will be here
  sorry

end smaller_cuboid_length_l132_132089


namespace total_earnings_l132_132847

theorem total_earnings (x y : ℝ) (h : 20 * x * y = 18 * x * y + 150) : 
  18 * x * y + 20 * x * y + 20 * x * y = 4350 :=
by sorry

end total_earnings_l132_132847


namespace max_of_three_diff_pos_int_with_mean_7_l132_132293

theorem max_of_three_diff_pos_int_with_mean_7 (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_mean : (a + b + c) / 3 = 7) :
  max a (max b c) = 18 := 
sorry

end max_of_three_diff_pos_int_with_mean_7_l132_132293


namespace angle_supplement_complement_l132_132664

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132664


namespace rectangle_perimeter_inequality_l132_132714

-- Define rectilinear perimeters
def perimeter (length : ℝ) (width : ℝ) : ℝ := 2 * (length + width)

-- Definitions for rectangles contained within each other
def rectangle_contained (len1 wid1 len2 wid2 : ℝ) : Prop :=
  len1 ≤ len2 ∧ wid1 ≤ wid2

-- Statement of the problem
theorem rectangle_perimeter_inequality (l1 w1 l2 w2 : ℝ) (h : rectangle_contained l1 w1 l2 w2) :
  perimeter l1 w1 ≤ perimeter l2 w2 :=
sorry

end rectangle_perimeter_inequality_l132_132714


namespace find_cost_price_l132_132958

/-- 
Given:
- SP = 1290 (selling price)
- LossP = 14.000000000000002 (loss percentage)
Prove that: CP = 1500 (cost price)
--/
theorem find_cost_price (SP : ℝ) (LossP : ℝ) (CP : ℝ) (h1 : SP = 1290) (h2 : LossP = 14.000000000000002) : CP = 1500 :=
sorry

end find_cost_price_l132_132958


namespace weeks_to_work_l132_132487

-- Definitions of conditions as per step a)
def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def brothers_total_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def job_pay_per_week : ℕ := 3
def total_ticket_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
def total_savings := isabelle_savings + brothers_total_savings
def remaining_amount := total_ticket_cost - total_savings

-- Theorem statement to match the question
theorem weeks_to_work : remaining_amount / job_pay_per_week = 10 := by
  -- Lean expects a proof here, replaced with sorry to skip it
  sorry

end weeks_to_work_l132_132487


namespace abs_sum_of_roots_l132_132059

theorem abs_sum_of_roots 
  (a b c m : ℤ) 
  (h1 : a + b + c = 0)
  (h2 : ab + bc + ca = -2023)
  : |a| + |b| + |c| = 102 := 
sorry

end abs_sum_of_roots_l132_132059


namespace georgina_teaches_2_phrases_per_week_l132_132064

theorem georgina_teaches_2_phrases_per_week
    (total_phrases : ℕ) 
    (initial_phrases : ℕ) 
    (days_owned : ℕ)
    (phrases_per_week : ℕ):
    total_phrases = 17 → 
    initial_phrases = 3 → 
    days_owned = 49 → 
    phrases_per_week = (total_phrases - initial_phrases) / (days_owned / 7) → 
    phrases_per_week = 2 := 
by
  intros h_total h_initial h_days h_calc
  rw [h_total, h_initial, h_days] at h_calc
  sorry  -- Proof to be filled

end georgina_teaches_2_phrases_per_week_l132_132064


namespace tangent_line_at_x1_f_nonnegative_iff_l132_132770

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x-1) * Real.log x - m * (x+1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + (x-1) / x - m

theorem tangent_line_at_x1 (m : ℝ) (h : m = 1) :
  ∀ x y : ℝ, f x 1 = y → (x = 1) → x + y + 1 = 0 :=
sorry

theorem f_nonnegative_iff (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end tangent_line_at_x1_f_nonnegative_iff_l132_132770


namespace angle_measure_l132_132674

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132674


namespace angle_measure_l132_132575

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132575


namespace parabolas_intersect_at_point_l132_132274

theorem parabolas_intersect_at_point :
  ∀ (p q : ℝ), p + q = 2019 → (1 : ℝ)^2 + (p : ℝ) * 1 + q = 2020 :=
by
  intros p q h
  sorry

end parabolas_intersect_at_point_l132_132274


namespace angle_supplement_complement_l132_132658

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132658


namespace geometric_series_sum_l132_132051

theorem geometric_series_sum : 
  ∑' n : ℕ, (5 / 3) * (-1 / 3) ^ n = (5 / 4) := by
  sorry

end geometric_series_sum_l132_132051


namespace convex_polygon_can_be_divided_l132_132943

-- Define convex polygon and the requirement for question
def convex_polygon (n : ℕ) := sorry -- To be defined, as defining convexity is complex and out of scope for this exercise.

-- Definition for being able to divide a polygon into convex pentagons
def can_be_divided_into_pentagons (P : Type) [convex_polygon P] : Prop := sorry -- Again, the detailed definition is complex

theorem convex_polygon_can_be_divided (n : ℕ) 
  (h₁ : n ≥ 6) 
  (P : Type) 
  [convex_polygon P] : 
  can_be_divided_into_pentagons P :=
sorry -- Proof goes here

end convex_polygon_can_be_divided_l132_132943


namespace angle_measure_l132_132610

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132610


namespace sum_of_solutions_eq_320_l132_132292

theorem sum_of_solutions_eq_320 :
  ∃ (S : Finset ℝ), 
  (∀ x ∈ S, 0 < x ∧ x < 180 ∧ (1 + (Real.sin x / Real.sin (4 * x)) = (Real.sin (3 * x) / Real.sin (2 * x)))) 
  ∧ S.sum id = 320 :=
by {
  sorry
}

end sum_of_solutions_eq_320_l132_132292


namespace question_1_question_2_l132_132235

open Real

noncomputable def f (x a : ℝ) := abs (x - a) + 3 * x

theorem question_1 :
  {x : ℝ | f x 1 > 3 * x + 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by 
  sorry
  
theorem question_2 (h : {x : ℝ | f x a ≤ 0} = {x : ℝ | x ≤ -1}) :
  a = 2 :=
by 
  sorry

end question_1_question_2_l132_132235


namespace quadratic_root_relationship_l132_132239

theorem quadratic_root_relationship
  (m1 m2 : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h_eq1 : m1 * x1^2 + (1 / 3) * x1 + 1 = 0)
  (h_eq2 : m1 * x2^2 + (1 / 3) * x2 + 1 = 0)
  (h_eq3 : m2 * x3^2 + (1 / 3) * x3 + 1 = 0)
  (h_eq4 : m2 * x4^2 + (1 / 3) * x4 + 1 = 0)
  (h_order : x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ x2 < 0) :
  m2 > m1 ∧ m1 > 0 :=
sorry

end quadratic_root_relationship_l132_132239


namespace angle_supplement_complement_l132_132660

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132660


namespace regular_octagon_interior_angle_l132_132440

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l132_132440


namespace mildred_heavier_than_carol_l132_132503

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem mildred_heavier_than_carol : mildred_weight - carol_weight = 50 := 
by
  sorry

end mildred_heavier_than_carol_l132_132503


namespace johns_final_amount_l132_132926

def initial_amount : ℝ := 45.7
def deposit_amount : ℝ := 18.6
def withdrawal_amount : ℝ := 20.5

theorem johns_final_amount : initial_amount + deposit_amount - withdrawal_amount = 43.8 :=
by
  sorry

end johns_final_amount_l132_132926


namespace centers_collinear_l132_132901

theorem centers_collinear (k : ℝ) (hk : k ≠ -1) :
    ∀ p : ℝ × ℝ, p = (-k, -2*k-5) → (2*p.1 - p.2 - 5 = 0) :=
by
  sorry

end centers_collinear_l132_132901


namespace inverse_f_1_l132_132902

noncomputable def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

theorem inverse_f_1 : ∃ x : ℝ, f x = 1 ∧ x = 2 := by
sorry

end inverse_f_1_l132_132902


namespace angle_measure_l132_132611

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132611


namespace find_triples_l132_132337

theorem find_triples (a b c : ℕ) :
  (∃ n : ℕ, 2^a + 2^b + 2^c + 3 = n^2) ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end find_triples_l132_132337


namespace sergeant_distance_travel_l132_132308

noncomputable def sergeant_distance (x k : ℝ) : ℝ :=
  let t₁ := 1 / (x * (k - 1))
  let t₂ := 1 / (x * (k + 1))
  let t := t₁ + t₂
  let d := k * 4 / 3
  d

theorem sergeant_distance_travel (x k : ℝ) (h1 : (4 * k) / (k^2 - 1) = 4 / 3) :
  sergeant_distance x k = 8 / 3 := by
  sorry

end sergeant_distance_travel_l132_132308


namespace odd_periodic_function_l132_132243

theorem odd_periodic_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 5) = f x)
  (h_f1 : f 1 = 1)
  (h_f2 : f 2 = 2) :
  f 3 - f 4 = -1 :=
sorry

end odd_periodic_function_l132_132243


namespace solve_inequalities_l132_132143

theorem solve_inequalities :
  {x : ℝ | -3 < x ∧ x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 2} =
  { x : ℝ | (5 / (x + 3) ≥ 1) ∧ (x^2 + x - 2 ≥ 0) } :=
sorry

end solve_inequalities_l132_132143


namespace angle_supplement_complement_l132_132583

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132583


namespace smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l132_132974

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def has_prime_factor_less_than (n k : ℕ) : Prop := ∃ p : ℕ, p < k ∧ is_prime p ∧ p ∣ n

theorem smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60 :
  ∃ m : ℕ, 
    m = 4091 ∧ 
    ¬is_prime m ∧ 
    ¬is_square m ∧ 
    ¬has_prime_factor_less_than m 60 ∧ 
    (∀ n : ℕ, ¬is_prime n ∧ ¬is_square n ∧ ¬has_prime_factor_less_than n 60 → 4091 ≤ n) :=
by
  sorry

end smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l132_132974


namespace find_Δ_l132_132233

-- Define the constants and conditions
variables (Δ p : ℕ)
axiom condition1 : Δ + p = 84
axiom condition2 : (Δ + p) + p = 153

-- State the theorem
theorem find_Δ : Δ = 15 :=
by
  sorry

end find_Δ_l132_132233


namespace boat_width_l132_132202

-- Definitions: river width, number of boats, and space between/banks
def river_width : ℝ := 42
def num_boats : ℕ := 8
def space_between : ℝ := 2

-- Prove the width of each boat given the conditions
theorem boat_width : 
  ∃ w : ℝ, 
    8 * w + 7 * space_between + 2 * space_between = river_width ∧
    w = 3 :=
by
  sorry

end boat_width_l132_132202


namespace restore_grid_l132_132553

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l132_132553


namespace time_after_10000_seconds_l132_132800

def time_add_seconds (h m s : Nat) (t : Nat) : (Nat × Nat × Nat) :=
  let total_seconds := h * 3600 + m * 60 + s + t
  let hours := (total_seconds / 3600) % 24
  let minutes := (total_seconds % 3600) / 60
  let seconds := (total_seconds % 3600) % 60
  (hours, minutes, seconds)

theorem time_after_10000_seconds :
  time_add_seconds 5 45 0 10000 = (8, 31, 40) :=
by
  sorry

end time_after_10000_seconds_l132_132800


namespace fraction_identity_l132_132757

theorem fraction_identity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 = b^2 + b * c) (h2 : b^2 = c^2 + a * c) : 
  (1 / c) = (1 / a) + (1 / b) :=
by 
  sorry

end fraction_identity_l132_132757


namespace average_marks_of_first_class_l132_132956

theorem average_marks_of_first_class (n1 n2 : ℕ) (avg2 avg_all : ℝ)
  (h_n1 : n1 = 25) (h_n2 : n2 = 40) (h_avg2 : avg2 = 65) (h_avg_all : avg_all = 59.23076923076923) :
  ∃ (A : ℝ), A = 50 :=
by 
  sorry

end average_marks_of_first_class_l132_132956


namespace map_area_l132_132043

def length : ℕ := 5
def width : ℕ := 2
def area_of_map (length width : ℕ) : ℕ := length * width

theorem map_area : area_of_map length width = 10 := by
  sorry

end map_area_l132_132043


namespace angle_measure_l132_132671

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132671


namespace angle_measure_l132_132574

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132574


namespace parabola_standard_eq_l132_132098

theorem parabola_standard_eq (p : ℝ) (x y : ℝ) :
  (∃ x y, 3 * x - 4 * y - 12 = 0) →
  ( (p = 6 ∧ x^2 = -12 * y ∧ y = -3) ∨ (p = 8 ∧ y^2 = 16 * x ∧ x = 4)) :=
sorry

end parabola_standard_eq_l132_132098


namespace line_through_points_l132_132540

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 8)) (h2 : (x2, y2) = (5, 2)) :
  ∃ m b : ℝ, (∀ x, y = m * x + b → (x, y) = (2,8) ∨ (x, y) = (5, 2)) ∧ (m + b = 10) :=
by
  sorry

end line_through_points_l132_132540


namespace boat_breadth_is_two_l132_132704

noncomputable def breadth_of_boat (L h m g ρ : ℝ) : ℝ :=
  let W := m * g
  let V := W / (ρ * g)
  V / (L * h)

theorem boat_breadth_is_two :
  breadth_of_boat 7 0.01 140 9.81 1000 = 2 := 
by
  unfold breadth_of_boat
  simp
  sorry

end boat_breadth_is_two_l132_132704


namespace smallest_number_of_coins_l132_132835

theorem smallest_number_of_coins (p n d q h: ℕ) (total: ℕ) 
  (coin_value: ℕ → ℕ)
  (h_p: coin_value 1 = 1) 
  (h_n: coin_value 5 = 5) 
  (h_d: coin_value 10 = 10) 
  (h_q: coin_value 25 = 25) 
  (h_h: coin_value 50 = 50)
  (total_def: total = p * (coin_value 1) + n * (coin_value 5) +
                     d * (coin_value 10) + q * (coin_value 25) + 
                     h * (coin_value 50))
  (h_total: total = 100): 
  p + n + d + q + h = 3 :=
by
  sorry

end smallest_number_of_coins_l132_132835


namespace kendra_packs_l132_132492

/-- Kendra has some packs of pens. Tony has 2 packs of pens. There are 3 pens in each pack. 
Kendra and Tony decide to keep two pens each and give the remaining pens to their friends 
one pen per friend. They give pens to 14 friends. Prove that Kendra has 4 packs of pens. --/
theorem kendra_packs : ∀ (kendra_pens tony_pens pens_per_pack pens_kept pens_given friends : ℕ),
  tony_pens = 2 →
  pens_per_pack = 3 →
  pens_kept = 2 →
  pens_given = 14 →
  tony_pens * pens_per_pack - pens_kept + kendra_pens - pens_kept = pens_given →
  kendra_pens / pens_per_pack = 4 :=
by
  intros kendra_pens tony_pens pens_per_pack pens_kept pens_given friends
  intro h1
  intro h2
  intro h3
  intro h4
  intro h5
  sorry

end kendra_packs_l132_132492


namespace integer_roots_of_polynomial_l132_132885

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = 2 ∨ x = -3 ∨ x = 4 := 
by 
  sorry

end integer_roots_of_polynomial_l132_132885


namespace probability_each_player_has_3_l132_132172

noncomputable def three_friends_game_probability : ℚ :=
  sorry

theorem probability_each_player_has_3 :
  three_friends_game_probability = 1 / 4 :=
  sorry

end probability_each_player_has_3_l132_132172


namespace angle_supplement_complement_l132_132654

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132654


namespace sequence_solution_l132_132484

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n : ℕ, n ∈ (Set.Icc 1 9) → 
    (n * a (n + 1) = (n + 1) * a n + 2)) ∧ a 10 = 38 :=
by
  sorry

end sequence_solution_l132_132484


namespace angle_solution_l132_132646

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132646


namespace average_book_width_l132_132805

-- Define the widths of the books as given in the problem conditions
def widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

-- Define the number of books from the problem conditions
def num_books : ℝ := 6

-- We prove that the average width of the books is equal to 4.75
theorem average_book_width : (widths.sum / num_books) = 4.75 :=
by
  sorry

end average_book_width_l132_132805


namespace wine_age_problem_l132_132822

theorem wine_age_problem
  (carlo_rosi : ℕ)
  (franzia : ℕ)
  (twin_valley : ℕ)
  (h1 : franzia = 3 * carlo_rosi)
  (h2 : carlo_rosi = 4 * twin_valley)
  (h3 : carlo_rosi = 40) :
  franzia + carlo_rosi + twin_valley = 170 :=
by
  sorry

end wine_age_problem_l132_132822


namespace ratio_of_square_areas_l132_132041

noncomputable def ratio_of_areas (s : ℝ) : ℝ := s^2 / (4 * s^2)

theorem ratio_of_square_areas (s : ℝ) (h : s ≠ 0) : ratio_of_areas s = 1 / 4 := 
by
  sorry

end ratio_of_square_areas_l132_132041


namespace combined_mean_score_l132_132273

-- Definitions based on the conditions
def mean_score_class1 : ℕ := 90
def mean_score_class2 : ℕ := 80
def ratio_students (n1 n2 : ℕ) : Prop := n1 / n2 = 2 / 3

-- Proof statement
theorem combined_mean_score (n1 n2 : ℕ) 
  (h1 : ratio_students n1 n2) 
  (h2 : mean_score_class1 = 90) 
  (h3 : mean_score_class2 = 80) : 
  ((mean_score_class1 * n1) + (mean_score_class2 * n2)) / (n1 + n2) = 84 := 
by
  sorry

end combined_mean_score_l132_132273


namespace regular_octagon_interior_angle_l132_132376

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l132_132376


namespace probability_greater_than_two_on_three_dice_l132_132480

theorem probability_greater_than_two_on_three_dice :
  (4 / 6 : ℚ) ^ 3 = (8 / 27 : ℚ) :=
by
  sorry

end probability_greater_than_two_on_three_dice_l132_132480


namespace isabelle_work_weeks_l132_132486

-- Define the costs and savings
def isabelle_ticket_cost := 20
def brother_ticket_cost := 10
def brothers_count := 2
def brothers_savings := 5
def isabelle_savings := 5
def weekly_earnings := 3

-- Calculate total required work weeks
theorem isabelle_work_weeks :
  let total_ticket_cost := isabelle_ticket_cost + brother_ticket_cost * brothers_count in
  let total_savings := isabelle_savings + brothers_savings in
  let required_savings := total_ticket_cost - total_savings in
  required_savings / weekly_earnings = 10 :=
by
  sorry

end isabelle_work_weeks_l132_132486


namespace worker_usual_time_l132_132304

theorem worker_usual_time (S T : ℝ) (D : ℝ) (h1 : D = S * T)
    (h2 : D = (3/4) * S * (T + 8)) : T = 24 :=
by
  sorry

end worker_usual_time_l132_132304


namespace angle_measure_l132_132595

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132595


namespace factorize_eq_l132_132222

theorem factorize_eq (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_eq_l132_132222


namespace stickers_on_first_day_l132_132507

theorem stickers_on_first_day (s e total : ℕ) (h1 : e = 22) (h2 : total = 61) (h3 : total = s + e) : s = 39 :=
by
  sorry

end stickers_on_first_day_l132_132507


namespace regular_octagon_interior_angle_l132_132373

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l132_132373


namespace line_through_origin_in_quadrants_l132_132539

theorem line_through_origin_in_quadrants (A B C : ℝ) :
  (-A * x - B * y + C = 0) ∧ (0 = 0) ∧ (exists x y, 0 < x * y) →
  (C = 0) ∧ (A * B < 0) :=
sorry

end line_through_origin_in_quadrants_l132_132539


namespace expression_value_l132_132782

theorem expression_value (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x ^ 4 - 6 * x ^ 3 - 2 * x ^ 2 + 18 * x + 23) / (x ^ 2 - 8 * x + 15) = 5 :=
by
  sorry

end expression_value_l132_132782


namespace speed_downstream_l132_132711

def speed_in_still_water := 12 -- man in still water
def speed_of_stream := 6  -- speed of stream
def speed_upstream := 6  -- rowing upstream

theorem speed_downstream : 
  speed_in_still_water + speed_of_stream = 18 := 
by 
  sorry

end speed_downstream_l132_132711


namespace angle_measure_l132_132669

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132669


namespace m_cubed_plus_m_inv_cubed_l132_132912

theorem m_cubed_plus_m_inv_cubed (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 1 = 971 :=
sorry

end m_cubed_plus_m_inv_cubed_l132_132912


namespace steve_needs_28_feet_of_wood_l132_132527

-- Define the required lengths
def lengths_4_feet : Nat := 6
def lengths_2_feet : Nat := 2

-- Define the wood length in feet for each type
def wood_length_4 : Nat := 4
def wood_length_2 : Nat := 2

-- Total feet of wood required
def total_wood : Nat := lengths_4_feet * wood_length_4 + lengths_2_feet * wood_length_2

-- The theorem to prove that the total amount of wood required is 28 feet
theorem steve_needs_28_feet_of_wood : total_wood = 28 :=
by
  sorry

end steve_needs_28_feet_of_wood_l132_132527


namespace find_C_l132_132035

theorem find_C (A B C : ℕ) :
  (8 + 5 + 6 + 3 + 2 + A + B) % 3 = 0 →
  (4 + 3 + 7 + 5 + A + B + C) % 3 = 0 →
  C = 2 :=
by
  intros h1 h2
  sorry

end find_C_l132_132035


namespace percentage_of_oysters_with_pearls_l132_132855

def jamie_collects_oysters (oysters_per_dive dives total_pearls : ℕ) : ℕ :=
  oysters_per_dive * dives

def percentage_with_pearls (total_pearls total_oysters : ℕ) : ℕ :=
  (total_pearls * 100) / total_oysters

theorem percentage_of_oysters_with_pearls :
  ∀ (oysters_per_dive dives total_pearls : ℕ),
  oysters_per_dive = 16 →
  dives = 14 →
  total_pearls = 56 →
  percentage_with_pearls total_pearls (jamie_collects_oysters oysters_per_dive dives total_pearls) = 25 :=
by
  intros
  sorry

end percentage_of_oysters_with_pearls_l132_132855


namespace sum_lent_is_1000_l132_132021

theorem sum_lent_is_1000
    (P : ℝ)
    (r : ℝ)
    (t : ℝ)
    (I : ℝ)
    (h1 : r = 5)
    (h2 : t = 5)
    (h3 : I = P - 750)
    (h4 : I = P * r * t / 100) :
  P = 1000 :=
by sorry

end sum_lent_is_1000_l132_132021


namespace interior_angle_regular_octagon_l132_132411

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l132_132411


namespace bob_grade_is_35_l132_132112

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l132_132112


namespace prove_a_range_l132_132359

-- Defining the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, a^2 * x^2 + a * x - 2 = 0
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The proposition to prove
theorem prove_a_range (a : ℝ) (hpq : ¬(p a ∨ q a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end prove_a_range_l132_132359


namespace difference_between_numbers_l132_132034

theorem difference_between_numbers : 
  ∃ (a : ℕ), a + 10 * a = 30000 → 9 * a = 24543 := 
by 
  sorry

end difference_between_numbers_l132_132034


namespace line_through_points_on_parabola_l132_132896

theorem line_through_points_on_parabola
  (p q : ℝ)
  (hpq : p^2 - 4 * q > 0) :
  ∃ (A B : ℝ × ℝ),
    (exists (x₁ x₂ : ℝ), x₁^2 + p * x₁ + q = 0 ∧ x₂^2 + p * x₂ + q = 0 ∧
                         A = (x₁, x₁^2 / 3) ∧ B = (x₂, x₂^2 / 3) ∧
                         (∀ x y, (x, y) = A ∨ (x, y) = B → px + 3 * y + q = 0)) :=
sorry

end line_through_points_on_parabola_l132_132896


namespace Darcy_remaining_clothes_l132_132745

/--
Darcy initially has 20 shirts and 8 pairs of shorts.
He folds 12 of the shirts and 5 of the pairs of shorts.
We want to prove that the total number of remaining pieces of clothing Darcy has to fold is 11.
-/
theorem Darcy_remaining_clothes
  (initial_shirts : Nat)
  (initial_shorts : Nat)
  (folded_shirts : Nat)
  (folded_shorts : Nat)
  (remaining_shirts : Nat)
  (remaining_shorts : Nat)
  (total_remaining : Nat) :
  initial_shirts = 20 → initial_shorts = 8 →
  folded_shirts = 12 → folded_shorts = 5 →
  remaining_shirts = initial_shirts - folded_shirts →
  remaining_shorts = initial_shorts - folded_shorts →
  total_remaining = remaining_shirts + remaining_shorts →
  total_remaining = 11 := by
  sorry

end Darcy_remaining_clothes_l132_132745


namespace calculate_adults_in_play_l132_132522

theorem calculate_adults_in_play :
  ∃ A : ℕ, (11 * A = 49 + 50) := sorry

end calculate_adults_in_play_l132_132522


namespace percentage_error_in_area_l132_132302

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := s * (1 + 0.03)
  let A := s * s
  let A' := s' * s'
  ((A' - A) / A) * 100 = 6.09 :=
by
  sorry

end percentage_error_in_area_l132_132302


namespace RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l132_132120

-- Definitions representing the conditions
def TotalCrates : ℕ := 50
def PriceGrapes : ℕ := 15
def PriceMangoes : ℕ := 20
def PricePassionFruits : ℕ := 25
def CratesGrapes : ℕ := 13
def CratesMangoes : ℕ := 20
def CratesPassionFruits : ℕ := TotalCrates - CratesGrapes - CratesMangoes

def RevenueWithoutDiscounts : ℕ :=
  (CratesGrapes * PriceGrapes) +
  (CratesMangoes * PriceMangoes) +
  (CratesPassionFruits * PricePassionFruits)

def DiscountGrapes : Float := if CratesGrapes > 10 then 0.10 else 0.0
def DiscountMangoes : Float := if CratesMangoes > 15 then 0.15 else 0.0
def DiscountPassionFruits : Float := if CratesPassionFruits > 5 then 0.20 else 0.0

def DiscountedPrice (price : ℕ) (discount : Float) : Float := 
  price.toFloat * (1.0 - discount)

def RevenueWithDiscounts : Float :=
  (CratesGrapes.toFloat * DiscountedPrice PriceGrapes DiscountGrapes) +
  (CratesMangoes.toFloat * DiscountedPrice PriceMangoes DiscountMangoes) +
  (CratesPassionFruits.toFloat * DiscountedPrice PricePassionFruits DiscountPassionFruits)

-- Proof problems
theorem RevenueWithoutDiscounts_is_1020 : RevenueWithoutDiscounts = 1020 := sorry
theorem RevenueWithDiscounts_is_855_5 : RevenueWithDiscounts = 855.5 := sorry
theorem Difference_is_164_5 : (RevenueWithoutDiscounts.toFloat - RevenueWithDiscounts) = 164.5 := sorry

end RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l132_132120


namespace angle_supplement_complement_l132_132659

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132659


namespace cruise_liner_travelers_l132_132164

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l132_132164


namespace ln_gt_ln_sufficient_for_x_gt_y_l132_132852

noncomputable def ln : ℝ → ℝ := sorry  -- Assuming ln is imported from Mathlib

-- Conditions
variable (x y : ℝ)
axiom ln_gt_ln_of_x_gt_y (hxy : x > y) (hx_pos : 0 < x) (hy_pos : 0 < y) : ln x > ln y

theorem ln_gt_ln_sufficient_for_x_gt_y (h : ln x > ln y) : x > y := sorry

end ln_gt_ln_sufficient_for_x_gt_y_l132_132852


namespace sample_size_stratified_sampling_l132_132706

theorem sample_size_stratified_sampling :
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  sample_size = 20 :=
by
  -- Definitions:
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  
  -- Proof:
  sorry

end sample_size_stratified_sampling_l132_132706


namespace regular_octagon_interior_angle_l132_132442

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l132_132442


namespace regular_octagon_interior_angle_eq_135_l132_132394

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l132_132394


namespace angle_supplement_complement_l132_132585

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132585


namespace find_integer_modulo_l132_132890

theorem find_integer_modulo : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [MOD 11] := by
  use 3
  sorry

end find_integer_modulo_l132_132890


namespace angle_supplement_complement_l132_132900

theorem angle_supplement_complement (a : ℝ) (h : 180 - a = 3 * (90 - a)) : a = 45 :=
by
  sorry

end angle_supplement_complement_l132_132900


namespace regular_octagon_interior_angle_l132_132438

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l132_132438


namespace ice_cream_total_sum_l132_132992

noncomputable def totalIceCream (friday saturday sunday monday tuesday : ℝ) : ℝ :=
  friday + saturday + sunday + monday + tuesday

theorem ice_cream_total_sum : 
  let friday := 3.25
  let saturday := 2.5
  let sunday := 1.75
  let monday := 0.5
  let tuesday := 2 * monday
  totalIceCream friday saturday sunday monday tuesday = 9 := by
    sorry

end ice_cream_total_sum_l132_132992


namespace pizza_cost_l132_132139

theorem pizza_cost
  (P T : ℕ)
  (hT : T = 1)
  (h_total : 3 * P + 4 * T + 5 = 39) :
  P = 10 :=
by
  sorry

end pizza_cost_l132_132139


namespace horner_eval_at_2_l132_132177

def poly (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_eval_at_2 : poly 2 = 373 := by
  sorry

end horner_eval_at_2_l132_132177


namespace regular_octagon_interior_angle_eq_135_l132_132390

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l132_132390


namespace hermia_elected_probability_l132_132266

-- Define the problem statement and conditions in Lean 4
noncomputable def probability_hermia_elected (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : ℚ :=
  if n = 1 then 1 else (2^n - 1) / (n * 2^(n-1))

-- Lean theorem statement
theorem hermia_elected_probability (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : 
  probability_hermia_elected n h_odd h_pos = (2^n - 1) / (n * 2^(n-1)) :=
by
  sorry

end hermia_elected_probability_l132_132266


namespace find_d_l132_132262

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l132_132262


namespace andy_correct_answer_l132_132485

-- Let y be the number Andy is using
def y : ℕ := 13  -- Derived from the conditions

-- Given condition based on Andy's incorrect operation
def condition : Prop := 4 * y + 5 = 57

-- Statement of the proof problem
theorem andy_correct_answer : condition → ((y + 5) * 4 = 72) := by
  intros h
  sorry

end andy_correct_answer_l132_132485


namespace least_positive_integer_solution_l132_132973

theorem least_positive_integer_solution :
  ∃ N : ℕ, N > 0 ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ (N % 8 = 7) ∧ (N % 9 = 8) ∧ (N % 10 = 9) ∧ (N % 11 = 10) ∧ N = 27719 :=
by
  -- the proof is omitted
  sorry

end least_positive_integer_solution_l132_132973


namespace parabola_y_intercepts_zero_l132_132085

-- Define the quadratic equation
def quadratic (a b c y: ℝ) : ℝ := a * y^2 + b * y + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Condition: equation of the parabola and discriminant calculation
def parabola_equation : Prop := 
  let a := 3
  let b := -4
  let c := 5
  discriminant a b c < 0

-- Statement to prove
theorem parabola_y_intercepts_zero : 
  (parabola_equation) → (∀ y : ℝ, quadratic 3 (-4) 5 y ≠ 0) :=
by
  intro h
  sorry

end parabola_y_intercepts_zero_l132_132085


namespace printer_a_time_l132_132136

theorem printer_a_time :
  ∀ (A B : ℕ), 
  B = A + 4 → 
  A + B = 12 → 
  (480 / A = 120) :=
by 
  intros A B hB hAB
  sorry

end printer_a_time_l132_132136


namespace cake_flour_amount_l132_132272

theorem cake_flour_amount (sugar_cups : ℕ) (flour_already_in : ℕ) (extra_flour_needed : ℕ) (total_flour : ℕ) 
  (h1 : sugar_cups = 7) 
  (h2 : flour_already_in = 2)
  (h3 : extra_flour_needed = 2)
  (h4 : total_flour = sugar_cups + extra_flour_needed) : 
  total_flour = 9 := 
sorry

end cake_flour_amount_l132_132272


namespace value_of_a4_l132_132232

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem value_of_a4 {a : ℕ → ℝ} {S : ℕ → ℝ} (h1 : arithmetic_sequence a)
  (h2 : sum_of_arithmetic_sequence S a) (h3 : S 7 = 28) :
  a 4 = 4 := 
  sorry

end value_of_a4_l132_132232


namespace fewest_colored_paper_l132_132255
   
   /-- Jungkook, Hoseok, and Seokjin shared colored paper. 
       Jungkook took 10 cards, Hoseok took 7, and Seokjin took 2 less than Jungkook. 
       Prove that Hoseok took the fewest pieces of colored paper. -/
   theorem fewest_colored_paper 
       (Jungkook Hoseok Seokjin : ℕ)
       (hj : Jungkook = 10)
       (hh : Hoseok = 7)
       (hs : Seokjin = Jungkook - 2) :
       Hoseok < Jungkook ∧ Hoseok < Seokjin :=
   by
     sorry
   
end fewest_colored_paper_l132_132255


namespace casper_entry_exit_ways_correct_l132_132028

-- Define the total number of windows
def num_windows : Nat := 8

-- Define the number of ways Casper can enter and exit through different windows
def casper_entry_exit_ways (num_windows : Nat) : Nat :=
  num_windows * (num_windows - 1)

-- Create a theorem to state the problem and its solution
theorem casper_entry_exit_ways_correct : casper_entry_exit_ways num_windows = 56 := by
  sorry

end casper_entry_exit_ways_correct_l132_132028


namespace function_properties_l132_132768

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem function_properties (a b : ℝ) (h : (a - 1) ^ 2 - 4 * b < 0) : 
  (∀ x : ℝ, f x a b > x) ∧ (∀ x : ℝ, f (f x a b) a b > x) ∧ (a + b > 0) :=
by
  sorry

end function_properties_l132_132768


namespace line_tangent_to_parabola_j_eq_98_l132_132210

theorem line_tangent_to_parabola_j_eq_98 (j : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 7 * y + j = 0 → x ≠ 0) →
  j = 98 :=
by
  sorry

end line_tangent_to_parabola_j_eq_98_l132_132210


namespace interior_angle_regular_octagon_l132_132419

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l132_132419


namespace largest_square_with_five_interior_lattice_points_l132_132713

theorem largest_square_with_five_interior_lattice_points :
  ∃ (s : ℝ), (∀ (x y : ℤ), 1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → ((⌊s⌋ - 1)^2 = 5) ∧ s^2 = 18 := sorry

end largest_square_with_five_interior_lattice_points_l132_132713


namespace joanie_loan_difference_l132_132491

theorem joanie_loan_difference:
  let P := 6000
  let r := 0.12
  let t := 4
  let n_quarterly := 4
  let n_annually := 1
  let A_quarterly := P * (1 + r / n_quarterly)^(n_quarterly * t)
  let A_annually := P * (1 + r / n_annually)^t
  A_quarterly - A_annually = 187.12 := sorry

end joanie_loan_difference_l132_132491


namespace difference_between_extrema_l132_132767

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x

theorem difference_between_extrema (a b : ℝ)
  (h1 : 3 * (2 : ℝ)^2 + 6 * a * (2 : ℝ) + 3 * b = 0)
  (h2 : 3 * (1 : ℝ)^2 + 6 * a * (1 : ℝ) + 3 * b = -3) :
  f 0 a b - f 2 a b = 4 :=
by
  sorry

end difference_between_extrema_l132_132767


namespace angle_solution_l132_132648

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132648


namespace solve_for_y_solve_for_x_l132_132366

variable (x y : ℝ)

theorem solve_for_y (h : 2 * x + 3 * y - 4 = 0) : y = (4 - 2 * x) / 3 := 
sorry

theorem solve_for_x (h : 2 * x + 3 * y - 4 = 0) : x = (4 - 3 * y) / 2 := 
sorry

end solve_for_y_solve_for_x_l132_132366


namespace initial_points_l132_132951

theorem initial_points (n : ℕ) (h : 16 * n - 15 = 225) : n = 15 :=
sorry

end initial_points_l132_132951


namespace angle_solution_l132_132650

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132650


namespace sqrt_neg3_squared_l132_132999

theorem sqrt_neg3_squared : Real.sqrt ((-3)^2) = 3 :=
by sorry

end sqrt_neg3_squared_l132_132999


namespace derivative_not_in_second_quadrant_l132_132096

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b * x + c
noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x - 4

-- Given condition: Axis of symmetry is x = 2
def axis_of_symmetry (b : ℝ) : Prop := b = -4

-- Additional condition: behavior of the derivative and quadrant check
def not_in_second_quadrant (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f' x < 0

-- The main theorem to be proved
theorem derivative_not_in_second_quadrant (b c : ℝ) (h : axis_of_symmetry b) :
  not_in_second_quadrant f_derivative :=
by {
  sorry
}

end derivative_not_in_second_quadrant_l132_132096


namespace geometric_sequence_sum_l132_132248

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ)
  (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 90 :=
sorry

end geometric_sequence_sum_l132_132248


namespace directrix_of_parabola_l132_132541

-- Define the variables and constants
variables (x y a : ℝ) (h₁ : x^2 = 4 * a * y) (h₂ : x = -2) (h₃ : y = 1)

theorem directrix_of_parabola (h : (-2)^2 = 4 * a * 1) : y = -1 := 
by
  -- Our proof will happen here, but we omit the details
  sorry

end directrix_of_parabola_l132_132541


namespace friends_attended_l132_132309

theorem friends_attended (total_guests bride_couples groom_couples : ℕ)
                         (bride_guests groom_guests family_guests friends : ℕ)
                         (h1 : total_guests = 300)
                         (h2 : bride_couples = 30)
                         (h3 : groom_couples = 30)
                         (h4 : bride_guests = bride_couples * 2)
                         (h5 : groom_guests = groom_couples * 2)
                         (h6 : family_guests = bride_guests + groom_guests)
                         (h7 : friends = total_guests - family_guests) :
  friends = 180 :=
by sorry

end friends_attended_l132_132309


namespace convert_base_5_to_decimal_l132_132878

-- Define the base-5 number 44 and its decimal equivalent
def base_5_number : ℕ := 4 * 5^1 + 4 * 5^0

-- Prove that the base-5 number 44 equals 24 in decimal
theorem convert_base_5_to_decimal : base_5_number = 24 := by
  sorry

end convert_base_5_to_decimal_l132_132878


namespace angle_measure_is_60_l132_132636

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132636


namespace division_of_powers_l132_132298

theorem division_of_powers (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := 
by sorry

end division_of_powers_l132_132298


namespace Mildred_heavier_than_Carol_l132_132501

-- Definition of weights for Mildred and Carol
def weight_Mildred : ℕ := 59
def weight_Carol : ℕ := 9

-- Definition of how much heavier Mildred is than Carol
def weight_difference : ℕ := weight_Mildred - weight_Carol

-- The theorem stating the difference in weight
theorem Mildred_heavier_than_Carol : weight_difference = 50 := 
by 
  -- Just state the theorem without providing the actual steps (proof skipped)
  sorry

end Mildred_heavier_than_Carol_l132_132501


namespace product_of_consecutive_nat_is_divisible_by_2_l132_132275

theorem product_of_consecutive_nat_is_divisible_by_2 (n : ℕ) : 2 ∣ n * (n + 1) :=
sorry

end product_of_consecutive_nat_is_divisible_by_2_l132_132275


namespace find_distance_between_posters_and_wall_l132_132977

-- Definitions for given conditions
def poster_width : ℝ := 29.05
def num_posters : ℕ := 8
def wall_width : ℝ := 394.4

-- The proof statement: find the distance 'd' between posters and ends
theorem find_distance_between_posters_and_wall :
  ∃ d : ℝ, (wall_width - num_posters * poster_width) / (num_posters + 1) = d ∧ d = 18 := 
by {
  -- The proof would involve showing that this specific d meets the constraints.
  sorry
}

end find_distance_between_posters_and_wall_l132_132977


namespace angle_supplement_complement_l132_132577

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132577


namespace find_d_l132_132259

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l132_132259


namespace part_i_solution_set_part_ii_minimum_value_l132_132078

-- Part (I)
theorem part_i_solution_set :
  (∀ (x : ℝ), 1 = 1 ∧ 2 = 2 → |x - 1| + |x + 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) :=
by { sorry }

-- Part (II)
theorem part_ii_minimum_value (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 2 * a * b) :
  |x - a| + |x + b| ≥ 9 / 2 :=
by { sorry }

end part_i_solution_set_part_ii_minimum_value_l132_132078


namespace angle_measure_l132_132612

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132612


namespace find_d_l132_132261

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l132_132261


namespace expression_is_odd_l132_132910

-- Define positive integers
def is_positive (n : ℕ) := n > 0

-- Define odd integer
def is_odd (n : ℕ) := n % 2 = 1

-- Define multiple of 3
def is_multiple_of_3 (n : ℕ) := ∃ k : ℕ, n = 3 * k

-- The Lean 4 statement to prove the problem
theorem expression_is_odd (a b c : ℕ)
  (ha : is_positive a) (hb : is_positive b) (hc : is_positive c)
  (h_odd_a : is_odd a) (h_odd_b : is_odd b) (h_mult_3_c : is_multiple_of_3 c) :
  is_odd (5^a + (b-1)^2 * c) :=
by
  sorry

end expression_is_odd_l132_132910


namespace positive_real_solution_eq_l132_132225

theorem positive_real_solution_eq :
  ∃ x : ℝ, 0 < x ∧ ( (1/4) * (5 * x^2 - 4) = (x^2 - 40 * x - 5) * (x^2 + 20 * x + 2) ) ∧ x = 20 + 10 * Real.sqrt 41 :=
by
  sorry

end positive_real_solution_eq_l132_132225


namespace inequality_proof_l132_132347

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l132_132347


namespace planes_touch_three_spheres_count_l132_132320

-- Declare the conditions as definitions
def square_side_length : ℝ := 10
def radii : Fin 4 → ℝ
| 0 => 1
| 1 => 2
| 2 => 4
| 3 => 3

-- The proof problem statement
theorem planes_touch_three_spheres_count :
    ∃ (planes_that_touch_three_spheres : ℕ) (planes_that_intersect_fourth_sphere : ℕ),
    planes_that_touch_three_spheres = 26 ∧ planes_that_intersect_fourth_sphere = 8 := 
by
  -- sorry skips the proof
  sorry

end planes_touch_three_spheres_count_l132_132320


namespace interior_angle_regular_octagon_l132_132432

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l132_132432


namespace angle_supplement_complement_l132_132602

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132602


namespace regular_octagon_angle_l132_132451

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l132_132451


namespace factorial_expression_equiv_l132_132326

theorem factorial_expression_equiv :
  6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := 
sorry

end factorial_expression_equiv_l132_132326


namespace Bennett_sales_l132_132995

-- Define the variables for the number of screens sold in each month.
variables (J F M : ℕ)

-- State the given conditions.
theorem Bennett_sales (h1: F = 2 * J) (h2: F = M / 4) (h3: M = 8800) :
  J + F + M = 12100 := by
sorry

end Bennett_sales_l132_132995


namespace angle_measure_l132_132591

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132591


namespace interior_angle_regular_octagon_l132_132408

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l132_132408


namespace train_length_l132_132834

theorem train_length (L : ℝ) 
  (equal_length : ∀ (A B : ℝ), A = B → L = A)
  (same_direction : ∀ (dir1 dir2 : ℤ), dir1 = 1 → dir2 = 1)
  (speed_faster : ℝ := 50) (speed_slower : ℝ := 36)
  (time_to_pass : ℝ := 36)
  (relative_speed := speed_faster - speed_slower)
  (relative_speed_km_per_sec := relative_speed / 3600)
  (distance_covered := relative_speed_km_per_sec * time_to_pass)
  (total_distance := distance_covered)
  (length_per_train := total_distance / 2)
  (length_in_meters := length_per_train * 1000): 
  L = 70 := 
by 
  sorry

end train_length_l132_132834


namespace culture_growth_l132_132857

/-- Define the initial conditions and growth rates of the bacterial culture -/
def initial_cells : ℕ := 5

def growth_rate1 : ℕ := 3
def growth_rate2 : ℕ := 2

def cycle_duration : ℕ := 3
def first_phase_duration : ℕ := 6
def second_phase_duration : ℕ := 6

def total_duration : ℕ := 12

/-- Define the hypothesis that calculates the number of cells at any point in time based on the given rules -/
theorem culture_growth : 
    (initial_cells * growth_rate1^ (first_phase_duration / cycle_duration) 
    * growth_rate2^ (second_phase_duration / cycle_duration)) = 180 := 
sorry

end culture_growth_l132_132857


namespace sum_of_solutions_eq_minus_2_l132_132547

-- Defining the equation and the goal
theorem sum_of_solutions_eq_minus_2 (x1 x2 : ℝ) (floor : ℝ → ℤ) (h1 : floor (3 * x1 + 1) = 2 * x1 - 1 / 2)
(h2 : floor (3 * x2 + 1) = 2 * x2 - 1 / 2) :
  x1 + x2 = -2 :=
sorry

end sum_of_solutions_eq_minus_2_l132_132547


namespace regular_octagon_interior_angle_l132_132395

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l132_132395


namespace principal_amount_l132_132787

theorem principal_amount (P R T SI : ℝ) (hR : R = 4) (hT : T = 5) (hSI : SI = P - 2240) 
    (h_formula : SI = (P * R * T) / 100) : P = 2800 :=
by 
  sorry

end principal_amount_l132_132787


namespace probability_problem_l132_132025

noncomputable def prob_at_least_two_less_than_ten : ℚ :=
  let prob_less_than_ten := (9 : ℚ) / 20
  let prob_not_less_than_ten := (11 : ℚ) / 20
  let prob := λ k, nat.choose 5 k * (prob_less_than_ten ^ k) * (prob_not_less_than_ten ^ (5 - k))
  prob 2 + prob 3 + prob 4 + prob 5

theorem probability_problem :
  prob_at_least_two_less_than_ten = 157439 / 20000 := sorry

end probability_problem_l132_132025


namespace gain_per_year_is_120_l132_132861

def principal := 6000
def rate_borrow := 4
def rate_lend := 6
def time := 2

def simple_interest (P R T : Nat) : Nat := P * R * T / 100

def interest_earned := simple_interest principal rate_lend time
def interest_paid := simple_interest principal rate_borrow time
def gain_in_2_years := interest_earned - interest_paid
def gain_per_year := gain_in_2_years / 2

theorem gain_per_year_is_120 : gain_per_year = 120 :=
by
  sorry

end gain_per_year_is_120_l132_132861


namespace initial_candy_bobby_l132_132873

-- Definitions given conditions
def initial_candy (x : ℕ) : Prop :=
  (x + 42 = 70)

-- Theorem statement
theorem initial_candy_bobby : ∃ x : ℕ, initial_candy x ∧ x = 28 :=
by {
  sorry
}

end initial_candy_bobby_l132_132873


namespace smallest_a_condition_l132_132497

theorem smallest_a_condition
  (a b : ℝ)
  (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_eq : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a = 15 :=
sorry

end smallest_a_condition_l132_132497


namespace range_of_a_l132_132915

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 1 / x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  is_increasing_on (λ x => x^2 + a * x + 1 / x) (Set.Ioi (1 / 2)) ↔ 3 ≤ a := 
by
  sorry

end range_of_a_l132_132915


namespace ball_hits_ground_l132_132149

theorem ball_hits_ground 
  (y : ℝ → ℝ) 
  (height_eq : ∀ t, y t = -3 * t^2 - 6 * t + 90) :
  ∃ t : ℝ, y t = 0 ∧ t = 5.00 :=
by
  sorry

end ball_hits_ground_l132_132149


namespace divide_into_three_groups_l132_132993

-- Define a delegate as a vertex in a graph
structure Symposium (V : Type) where
  acquainted : V → V → Prop
  acquainted_symm : ∀ {a b}, acquainted a b → acquainted b a
  acquainted_irrefl : ∀ {a}, ¬acquainted a a
  exists_acquainted : ∀ (a : V), ∃ (b : V), acquainted a b
  exists_third_delegates : ∀ (a b : V), ∃ (c : V), ¬(acquainted c a ∧ acquainted c b)

-- State the problem in lean
theorem divide_into_three_groups (V : Type) [finite V] (S : Symposium V) :
  ∃ (G1 G2 G3 : set V), 
  (∀ (v : V), v ∈ G1 ∨ v ∈ G2 ∨ v ∈ G3) ∧
  (∀ (v ∈ G1), ∃ (w ∈ G1), S.acquainted v w) ∧
  (∀ (v ∈ G2), ∃ (w ∈ G2), S.acquainted v w) ∧
  (∀ (v ∈ G3), ∃ (w ∈ G3), S.acquainted v w) := 
sorry

end divide_into_three_groups_l132_132993


namespace angle_complement_supplement_l132_132623

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132623


namespace select_representatives_l132_132333

theorem select_representatives : 
  (Nat.choose 5 2 * Nat.choose 4 2) + (Nat.choose 5 3 * Nat.choose 4 1) = 100 := 
by
  sorry

end select_representatives_l132_132333


namespace interior_angle_regular_octagon_l132_132425

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l132_132425


namespace scientific_notation_l132_132324

theorem scientific_notation :
  56.9 * 10^9 = 5.69 * 10^(10 - 1) :=
by
  sorry

end scientific_notation_l132_132324


namespace selina_sold_shirts_l132_132950

/-- Selina's selling problem -/
theorem selina_sold_shirts :
  let pants_price := 5
  let shorts_price := 3
  let shirts_price := 4
  let num_pants := 3
  let num_shorts := 5
  let remaining_money := 30 + (2 * 10)
  let money_from_pants := num_pants * pants_price
  let money_from_shorts := num_shorts * shorts_price
  let total_money_from_pants_and_shorts := money_from_pants + money_from_shorts
  let total_money_from_shirts := remaining_money - total_money_from_pants_and_shorts
  let num_shirts := total_money_from_shirts / shirts_price
  num_shirts = 5 := by
{
  sorry
}

end selina_sold_shirts_l132_132950


namespace regular_octagon_interior_angle_l132_132371

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l132_132371


namespace regular_octagon_interior_angle_eq_135_l132_132391

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l132_132391


namespace cube_sum_minus_triple_product_l132_132469

theorem cube_sum_minus_triple_product (x y z : ℝ) (h1 : x + y + z = 8) (h2 : xy + yz + zx = 20) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 32 :=
sorry

end cube_sum_minus_triple_product_l132_132469


namespace tina_final_balance_l132_132017

noncomputable def monthlyIncome : ℝ := 1000
noncomputable def juneBonusRate : ℝ := 0.1
noncomputable def investmentReturnRate : ℝ := 0.05
noncomputable def taxRate : ℝ := 0.1

-- Savings rates
noncomputable def juneSavingsRate : ℝ := 0.25
noncomputable def julySavingsRate : ℝ := 0.20
noncomputable def augustSavingsRate : ℝ := 0.30

-- Expenses
noncomputable def juneRent : ℝ := 200
noncomputable def juneGroceries : ℝ := 100
noncomputable def juneBookRate : ℝ := 0.05

noncomputable def julyRent : ℝ := 250
noncomputable def julyGroceries : ℝ := 150
noncomputable def julyShoesRate : ℝ := 0.15

noncomputable def augustRent : ℝ := 300
noncomputable def augustGroceries : ℝ := 175
noncomputable def augustMiscellaneousRate : ℝ := 0.1

theorem tina_final_balance :
  let juneIncome := monthlyIncome * (1 + juneBonusRate)
  let juneSavings := juneIncome * juneSavingsRate
  let juneExpenses := juneRent + juneGroceries + juneIncome * juneBookRate
  let juneRemaining := juneIncome - juneSavings - juneExpenses

  let julyIncome := monthlyIncome
  let julyInvestmentReturn := juneSavings * investmentReturnRate
  let julyTotalIncome := julyIncome + julyInvestmentReturn
  let julySavings := julyTotalIncome * julySavingsRate
  let julyExpenses := julyRent + julyGroceries + julyIncome * julyShoesRate
  let julyRemaining := julyTotalIncome - julySavings - julyExpenses

  let augustIncome := monthlyIncome
  let augustInvestmentReturn := julySavings * investmentReturnRate
  let augustTotalIncome := augustIncome + augustInvestmentReturn
  let augustSavings := augustTotalIncome * augustSavingsRate
  let augustExpenses := augustRent + augustGroceries + augustIncome * augustMiscellaneousRate
  let augustRemaining := augustTotalIncome - augustSavings - augustExpenses

  let totalInvestmentReturn := julyInvestmentReturn + augustInvestmentReturn
  let totalTaxOnInvestment := totalInvestmentReturn * taxRate

  let finalBalance := juneRemaining + julyRemaining + augustRemaining - totalTaxOnInvestment

  finalBalance = 860.7075 := by
  sorry

end tina_final_balance_l132_132017


namespace problem_l132_132532

variable (x y z w : ℚ)

theorem problem
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 :=
by sorry

end problem_l132_132532


namespace angle_measure_l132_132570

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132570


namespace angle_measure_is_60_l132_132639

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132639


namespace ab_is_zero_l132_132072

-- Define that a function is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given function f
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b - 2

-- The main theorem to prove
theorem ab_is_zero (a b : ℝ) (h_odd : is_odd (f a b)) : a * b = 0 := 
sorry

end ab_is_zero_l132_132072


namespace trees_in_one_row_l132_132927

theorem trees_in_one_row (total_revenue : ℕ) (price_per_apple : ℕ) (apples_per_tree : ℕ) (trees_per_row : ℕ)
  (revenue_condition : total_revenue = 30)
  (price_condition : price_per_apple = 1 / 2)
  (apples_condition : apples_per_tree = 5)
  (trees_condition : trees_per_row = 4) :
  trees_per_row = 4 := by
  sorry

end trees_in_one_row_l132_132927


namespace range_of_m_l132_132354

-- Definitions used to state conditions of the problem.
def fractional_equation (m x : ℝ) : Prop := (m / (2 * x - 1)) + 2 = 0
def positive_solution (x : ℝ) : Prop := x > 0

-- The Lean 4 theorem statement
theorem range_of_m (m x : ℝ) (h : fractional_equation m x) (hx : positive_solution x) : m < 2 ∧ m ≠ 0 :=
by
  sorry

end range_of_m_l132_132354


namespace fixed_fee_rental_l132_132813

theorem fixed_fee_rental (F C h : ℕ) (hC : C = F + 7 * h) (hC80 : C = 80) (hh9 : h = 9) : F = 17 :=
by
  sorry

end fixed_fee_rental_l132_132813


namespace quadratic_inequality_solution_l132_132008

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 10 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 5} :=
by
  sorry

end quadratic_inequality_solution_l132_132008


namespace quadratic_inequality_empty_solution_set_l132_132897

theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ := 
by sorry

end quadratic_inequality_empty_solution_set_l132_132897


namespace find_d_l132_132265

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l132_132265


namespace find_abc_pairs_l132_132750

theorem find_abc_pairs :
  ∀ (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧ (a-1)*(b-1)*(c-1) ∣ a*b*c - 1 → 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  -- Proof omitted
  sorry

end find_abc_pairs_l132_132750


namespace mildred_heavier_than_carol_l132_132504

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem mildred_heavier_than_carol : mildred_weight - carol_weight = 50 := 
by
  sorry

end mildred_heavier_than_carol_l132_132504


namespace smallest_multiple_of_6_and_15_l132_132056

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ b = 30 := 
by 
  use 30 
  sorry

end smallest_multiple_of_6_and_15_l132_132056


namespace most_balls_l132_132964

def soccerballs : ℕ := 50
def basketballs : ℕ := 26
def baseballs : ℕ := basketballs + 8

theorem most_balls :
  max (max soccerballs basketballs) baseballs = soccerballs := by
  sorry

end most_balls_l132_132964


namespace minimum_value_18_sqrt_3_minimum_value_at_x_3_l132_132752

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 12*x + 81 / x^3

theorem minimum_value_18_sqrt_3 (x : ℝ) (hx : x > 0) :
  f x ≥ 18 * Real.sqrt 3 :=
by
  sorry

theorem minimum_value_at_x_3 : f 3 = 18 * Real.sqrt 3 :=
by
  sorry

end minimum_value_18_sqrt_3_minimum_value_at_x_3_l132_132752


namespace complement_union_l132_132809

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {-1, 2}

def B : Set Int := {-1, 0, 1}

theorem complement_union :
  (U \ B) ∪ A = {-2, -1, 2} :=
by
  sorry

end complement_union_l132_132809


namespace value_of_m_has_positive_root_l132_132914

theorem value_of_m_has_positive_root (x m : ℝ) (hx : x ≠ 3) :
    ((x + 5) / (x - 3) = 2 - m / (3 - x)) → x > 0 → m = 8 := 
sorry

end value_of_m_has_positive_root_l132_132914


namespace bombardiers_shots_l132_132014

theorem bombardiers_shots (x y z : ℕ) :
  x + y = z + 26 →
  x + y + 38 = y + z →
  x + z = y + 24 →
  x = 25 ∧ y = 64 ∧ z = 63 := by
  sorry

end bombardiers_shots_l132_132014


namespace terry_spent_total_l132_132953

def total_amount_spent (monday_spent tuesday_spent wednesday_spent : ℕ) : ℕ := 
  monday_spent + tuesday_spent + wednesday_spent

theorem terry_spent_total 
  (monday_spent : ℕ)
  (hmonday : monday_spent = 6)
  (tuesday_spent : ℕ)
  (htuesday : tuesday_spent = 2 * monday_spent)
  (wednesday_spent : ℕ)
  (hwednesday : wednesday_spent = 2 * (monday_spent + tuesday_spent)) :
  total_amount_spent monday_spent tuesday_spent wednesday_spent = 54 :=
by
  sorry

end terry_spent_total_l132_132953


namespace horse_revolutions_l132_132709

theorem horse_revolutions (r1 r2 r3 : ℝ) (rev1 : ℕ) 
  (h1 : r1 = 30) (h2 : r2 = 15) (h3 : r3 = 10) (h4 : rev1 = 40) :
  (r2 / r1 = 1 / 2 ∧ 2 * rev1 = 80) ∧ (r3 / r1 = 1 / 3 ∧ 3 * rev1 = 120) :=
by
  sorry

end horse_revolutions_l132_132709


namespace clock_rings_in_a_day_l132_132063

-- Define the conditions
def rings_every_3_hours : ℕ := 3
def first_ring : ℕ := 1 -- This is 1 A.M. in our problem
def total_hours_in_day : ℕ := 24

-- Define the theorem
theorem clock_rings_in_a_day (n_rings : ℕ) : 
  (∀ n : ℕ, n_rings = total_hours_in_day / rings_every_3_hours + 1) :=
by
  -- use sorry to skip the proof
  sorry

end clock_rings_in_a_day_l132_132063


namespace angle_measure_is_60_l132_132642

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132642


namespace six_digit_mod_27_l132_132144

theorem six_digit_mod_27 (X : ℕ) (hX : 100000 ≤ X ∧ X < 1000000) (Y : ℕ) (hY : ∃ a b : ℕ, 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ X = 1000 * a + b ∧ Y = 1000 * b + a) :
  X % 27 = Y % 27 := 
by
  sorry

end six_digit_mod_27_l132_132144


namespace competition_problem_l132_132788

theorem competition_problem (n : ℕ) (s : ℕ) (correct_first_12 : s = (12 * 13) / 2)
    (gain_708_if_last_12_correct : s + 708 = (n - 11) * (n + 12) / 2):
    n = 71 :=
by
  sorry

end competition_problem_l132_132788


namespace sum_series_l132_132327

theorem sum_series :
  3 * (List.sum (List.map (λ n => n - 1) (List.range' 2 14))) = 273 :=
by
  sorry

end sum_series_l132_132327


namespace regular_octagon_interior_angle_l132_132455

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l132_132455


namespace max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l132_132564

theorem max_value_of_quadratic (r : ℝ) : -7 * r ^ 2 + 50 * r - 20 ≤ 5 :=
by sorry

theorem exists_r_for_max_value_of_quadratic : ∃ r : ℝ, -7 * r ^ 2 + 50 * r - 20 = 5 :=
by sorry

end max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l132_132564


namespace problem1_problem2_l132_132067

theorem problem1 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : Real.exp x + Real.exp y > 2 * Real.exp 1 :=
by {
  sorry -- Proof goes here
}

theorem problem2 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : x * Real.log x + y * Real.log y > 0 :=
by {
  sorry -- Proof goes here
}

end problem1_problem2_l132_132067


namespace find_BC_length_l132_132074

noncomputable def area_triangle (A B C : ℝ) : ℝ :=
  1/2 * A * B * C

theorem find_BC_length (A B C : ℝ) (angleA : ℝ)
  (h1 : area_triangle 5 A (Real.sin (π / 6)) = 5 * Real.sqrt 3)
  (h2 : B = 5)
  (h3 : angleA = π / 6) :
  C = Real.sqrt 13 :=
by
  sorry

end find_BC_length_l132_132074


namespace angle_measure_l132_132677

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132677


namespace operation_result_l132_132747

def operation (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem operation_result : operation 3 (-1) = 0 :=
by
  sorry

end operation_result_l132_132747


namespace students_playing_both_l132_132101

theorem students_playing_both (T F L N B : ℕ)
  (hT : T = 39)
  (hF : F = 26)
  (hL : L = 20)
  (hN : N = 10)
  (hTotal : (F + L - B) + N = T) :
  B = 17 :=
by
  sorry

end students_playing_both_l132_132101


namespace three_pow_m_plus_2n_l132_132357

theorem three_pow_m_plus_2n (m n : ℕ) (h1 : 3^m = 5) (h2 : 9^n = 10) : 3^(m + 2 * n) = 50 :=
by
  sorry

end three_pow_m_plus_2n_l132_132357


namespace book_arrangement_count_l132_132465

theorem book_arrangement_count :
  let total_books := 6
  let identical_science_books := 3
  let unique_other_books := total_books - identical_science_books
  (total_books! / (identical_science_books! * unique_other_books!)) = 120 := by
  sorry

end book_arrangement_count_l132_132465


namespace height_difference_percentage_l132_132303

theorem height_difference_percentage (q p : ℝ) (h : p = 0.6 * q) : (q - p) / p * 100 = 66.67 := 
by
  sorry

end height_difference_percentage_l132_132303


namespace pencils_more_than_200_on_saturday_l132_132965

theorem pencils_more_than_200_on_saturday 
    (p : ℕ → ℕ) 
    (h_start : p 1 = 3)
    (h_next_day : ∀ n, p (n + 1) = (p n + 2) * 2) 
    : p 6 > 200 :=
by
  -- Proof steps can be filled in here.
  sorry

end pencils_more_than_200_on_saturday_l132_132965


namespace polynomial_evaluation_l132_132534

-- Define the polynomial p(x) and the condition p(x) - p'(x) = x^2 + 2x + 1
variable (p : ℝ → ℝ)
variable (hp : ∀ x, p x - (deriv p x) = x^2 + 2 * x + 1)

-- Statement to prove p(5) = 50 given the conditions
theorem polynomial_evaluation : p 5 = 50 := 
sorry

end polynomial_evaluation_l132_132534


namespace set_C_cannot_form_triangle_l132_132868

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given conditions
def set_A := (3, 6, 8)
def set_B := (3, 8, 9)
def set_C := (3, 6, 9)
def set_D := (6, 8, 9)

theorem set_C_cannot_form_triangle : ¬ is_triangle 3 6 9 :=
by
  -- Proof is omitted
  sorry

end set_C_cannot_form_triangle_l132_132868


namespace angle_measure_is_60_l132_132638

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132638


namespace travelers_on_liner_l132_132169

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l132_132169


namespace number_of_sophomores_l132_132708

theorem number_of_sophomores (n x : ℕ) (freshmen seniors selected freshmen_selected : ℕ)
  (h_freshmen : freshmen = 450)
  (h_seniors : seniors = 250)
  (h_selected : selected = 60)
  (h_freshmen_selected : freshmen_selected = 27)
  (h_eq : selected / (freshmen + seniors + x) = freshmen_selected / freshmen) :
  x = 300 := by
  sorry

end number_of_sophomores_l132_132708


namespace greatest_integer_difference_l132_132470

theorem greatest_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  ∃ d : ℤ, d = y - x ∧ ∀ z, 4 < z ∧ z < 8 ∧ 8 < y ∧ y < 12 → (y - z ≤ d) :=
sorry

end greatest_integer_difference_l132_132470


namespace cone_height_ratio_l132_132201

theorem cone_height_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rolls_19_times : 19 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  h / r = 6 * Real.sqrt 10 :=
by
  -- problem setup and mathematical manipulations
  sorry

end cone_height_ratio_l132_132201


namespace smallest_brownie_pan_size_l132_132369

theorem smallest_brownie_pan_size :
  ∃ s : ℕ, (s - 2) ^ 2 = 4 * s - 4 ∧ ∀ t : ℕ, (t - 2) ^ 2 = 4 * t - 4 → s <= t :=
by
  sorry

end smallest_brownie_pan_size_l132_132369


namespace largest_constant_C_l132_132751

theorem largest_constant_C :
  ∃ C, C = 2 / Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z) := sorry

end largest_constant_C_l132_132751


namespace proof_firstExpr_proof_secondExpr_l132_132046

noncomputable def firstExpr : ℝ :=
  Real.logb 2 (Real.sqrt (7 / 48)) + Real.logb 2 12 - (1 / 2) * Real.logb 2 42 - 1

theorem proof_firstExpr :
  firstExpr = -3 / 2 :=
by
  sorry

noncomputable def secondExpr : ℝ :=
  (Real.logb 10 2) ^ 2 + Real.logb 10 (2 * Real.logb 10 50 + Real.logb 10 25)

theorem proof_secondExpr :
  secondExpr = 0.0906 + Real.logb 10 5.004 :=
by
  sorry

end proof_firstExpr_proof_secondExpr_l132_132046


namespace pyramid_surface_area_l132_132147

noncomputable def total_surface_area (a : ℝ) : ℝ :=
  a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7) / 2

theorem pyramid_surface_area (a : ℝ) :
  let hexagon_base_area := 3 * a^2 * Real.sqrt 3 / 2
  let triangle_area_1 := a^2 / 2
  let triangle_area_2 := a^2
  let triangle_area_3 := a^2 * Real.sqrt 7 / 4
  let lateral_area := 2 * (triangle_area_1 + triangle_area_2 + triangle_area_3)
  total_surface_area a = hexagon_base_area + lateral_area := 
sorry

end pyramid_surface_area_l132_132147


namespace probability_is_correct_l132_132251

noncomputable def probability_total_more_than_7 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 15
  favorable_outcomes / total_outcomes

theorem probability_is_correct :
  probability_total_more_than_7 = 5 / 12 :=
by
  sorry

end probability_is_correct_l132_132251


namespace regular_octagon_interior_angle_l132_132414

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l132_132414


namespace angle_value_l132_132923

theorem angle_value (x : ℝ) (h₁ : (90 : ℝ) = 44 + x) : x = 46 :=
by
  sorry

end angle_value_l132_132923


namespace regular_octagon_interior_angle_eq_135_l132_132389

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l132_132389


namespace largest_prime_value_of_quadratic_expression_l132_132563

theorem largest_prime_value_of_quadratic_expression : 
  ∃ n : ℕ, n > 0 ∧ Prime (n^2 - 12 * n + 27) ∧ ∀ m : ℕ, m > 0 → Prime (m^2 - 12 * m + 27) → (n^2 - 12 * n + 27) ≥ (m^2 - 12 * m + 27) := 
by
  sorry


end largest_prime_value_of_quadratic_expression_l132_132563


namespace cycling_speed_l132_132959

-- Definitions based on given conditions.
def ratio_L_B : ℕ := 1
def ratio_B_L : ℕ := 2
def area_of_park : ℕ := 20000
def time_in_minutes : ℕ := 6

-- The question translated to Lean 4 statement.
theorem cycling_speed (L B : ℕ) (h1 : ratio_L_B * B = ratio_B_L * L)
  (h2 : L * B = area_of_park)
  (h3 : B = 2 * L) :
  (2 * L + 2 * B) / (time_in_minutes / 60) = 6000 := by
  sorry

end cycling_speed_l132_132959


namespace angle_complement_supplement_l132_132621

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132621


namespace solve_equation_l132_132521

/-- 
  Given the equation:
    ∀ x, (x = 2 ∨ (3 < x ∧ x < 4)) ↔ (⌊(1/x) * ⌊x⌋^2⌋ = 2),
  where ⌊u⌋ represents the greatest integer less than or equal to u.
-/
theorem solve_equation (x : ℝ) : (x = 2 ∨ (3 < x ∧ x < 4)) ↔ ⌊(1/x) * ⌊x⌋^2⌋ = 2 := 
sorry

end solve_equation_l132_132521


namespace bus_speed_excluding_stoppages_l132_132052

theorem bus_speed_excluding_stoppages 
  (v_s : ℕ) -- Speed including stoppages in kmph
  (stop_duration_minutes : ℕ) -- Duration of stoppages in minutes per hour
  (stop_duration_fraction : ℚ := stop_duration_minutes / 60) -- Fraction of hour stopped
  (moving_fraction : ℚ := 1 - stop_duration_fraction) -- Fraction of hour moving
  (distance_per_hour : ℚ := v_s) -- Distance traveled per hour including stoppages
  (v : ℚ) -- Speed excluding stoppages
  
  (h1 : v_s = 50)
  (h2 : stop_duration_minutes = 10)
  
  -- Equation representing the total distance equals the distance traveled moving
  (h3 : v * moving_fraction = distance_per_hour)
: v = 60 := sorry

end bus_speed_excluding_stoppages_l132_132052


namespace third_number_eq_l132_132011

theorem third_number_eq :
  ∃ x : ℝ, (0.625 * 0.0729 * x) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ x = 2.33075 := 
by
  sorry

end third_number_eq_l132_132011


namespace candy_bar_sales_l132_132815

def max_sales : ℕ := 24
def seth_sales (max_sales : ℕ) : ℕ := 3 * max_sales + 6
def emma_sales (seth_sales : ℕ) : ℕ := seth_sales / 2 + 5
def total_sales (seth_sales emma_sales : ℕ) : ℕ := seth_sales + emma_sales

theorem candy_bar_sales : total_sales (seth_sales max_sales) (emma_sales (seth_sales max_sales)) = 122 := by
  sorry

end candy_bar_sales_l132_132815


namespace same_percentage_loss_as_profit_l132_132006

theorem same_percentage_loss_as_profit (CP SP L : ℝ) (h_prof : SP = 1720)
  (h_loss : L = CP - (14.67 / 100) * CP)
  (h_25_prof : 1.25 * CP = 1875) :
  L = 1280 := 
  sorry

end same_percentage_loss_as_profit_l132_132006


namespace bob_grade_is_35_l132_132110

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l132_132110


namespace find_point_P_l132_132082

/-- 
Given two points A and B, find the coordinates of point P that lies on the line AB
and satisfies that the distance from A to P is half the vector from A to B.
-/
theorem find_point_P 
  (A B : ℝ × ℝ) 
  (hA : A = (3, -4)) 
  (hB : B = (-9, 2)) 
  (P : ℝ × ℝ) 
  (hP : P.1 - A.1 = (1/2) * (B.1 - A.1) ∧ P.2 - A.2 = (1/2) * (B.2 - A.2)) : 
  P = (-3, -1) := 
sorry

end find_point_P_l132_132082


namespace perimeter_pentagon_l132_132294

noncomputable def AB : ℝ := 1
noncomputable def BC : ℝ := Real.sqrt 2
noncomputable def CD : ℝ := Real.sqrt 3
noncomputable def DE : ℝ := 2

noncomputable def AC : ℝ := Real.sqrt (AB^2 + BC^2)
noncomputable def AD : ℝ := Real.sqrt (AC^2 + CD^2)
noncomputable def AE : ℝ := Real.sqrt (AD^2 + DE^2)

theorem perimeter_pentagon (ABCDE : List ℝ) (H : ABCDE = [AB, BC, CD, DE, AE]) :
  List.sum ABCDE = 3 + Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 10 :=
by
  sorry -- Proof skipped as instructed

end perimeter_pentagon_l132_132294


namespace restore_grid_values_l132_132555

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l132_132555


namespace angle_supplement_complement_l132_132586

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132586


namespace tangent_line_at_A_l132_132784

def f (x : ℝ) : ℝ := x ^ (1 / 2)

def tangent_line_equation (x y: ℝ) : Prop :=
  4 * x - 4 * y + 1 = 0

theorem tangent_line_at_A :
  tangent_line_equation (1/4) (f (1/4)) :=
by
  sorry

end tangent_line_at_A_l132_132784


namespace angle_measure_l132_132613

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132613


namespace total_waiting_time_l132_132128

def t1 : ℕ := 20
def t2 : ℕ := 4 * t1 + 14
def T : ℕ := t1 + t2

theorem total_waiting_time : T = 114 :=
by {
  -- Preliminary calculations and justification would go here
  sorry
}

end total_waiting_time_l132_132128


namespace base10_to_base4_156_eq_2130_l132_132971

def base10ToBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 4) ((n % 4) :: acc)
    loop n []

theorem base10_to_base4_156_eq_2130 :
  base10ToBase4 156 = [2, 1, 3, 0] := sorry

end base10_to_base4_156_eq_2130_l132_132971


namespace wine_age_problem_l132_132823

theorem wine_age_problem
  (carlo_rosi : ℕ)
  (franzia : ℕ)
  (twin_valley : ℕ)
  (h1 : franzia = 3 * carlo_rosi)
  (h2 : carlo_rosi = 4 * twin_valley)
  (h3 : carlo_rosi = 40) :
  franzia + carlo_rosi + twin_valley = 170 :=
by
  sorry

end wine_age_problem_l132_132823


namespace inequality_am_gm_l132_132345

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l132_132345


namespace function_property_l132_132783

theorem function_property 
  (f : ℝ → ℝ) 
  (hf : ∀ x, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / (x^2)) 
  : 
  (f (1 / 2) = 15) ∧
  (∀ x, x ≠ 1 → f (x) = 4 / (x - 1)^2 - 1) ∧
  (∀ x, x ≠ 0 → x ≠ 1 → f (1 / x) = 4 * x^2 / (x - 1)^2 - 1) :=
by {
  sorry
}

end function_property_l132_132783


namespace ceiling_sum_evaluation_l132_132217

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l132_132217


namespace eq_of_frac_sub_l132_132328

theorem eq_of_frac_sub (x : ℝ) (hx : x ≠ 1) : 
  (2 / (x^2 - 1) - 1 / (x - 1)) = - (1 / (x + 1)) := 
by sorry

end eq_of_frac_sub_l132_132328


namespace simplify_expression_l132_132817

theorem simplify_expression (x y : ℝ) : ((3 * x + 22) + (150 * y + 22)) = (3 * x + 150 * y + 44) :=
by
  sorry

end simplify_expression_l132_132817


namespace ellipse_standard_equation_l132_132546

theorem ellipse_standard_equation (a c : ℝ) (h1 : a^2 = 13) (h2 : c^2 = 12) :
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ 
    ((∀ x y : ℝ, (x^2 / 13 + y^2 = 1)) ∨ (∀ x y : ℝ, (x^2 + y^2 / 13 = 1)))) :=
by
  sorry

end ellipse_standard_equation_l132_132546


namespace tangent_line_inv_g_at_0_l132_132236

noncomputable def g (x : ℝ) := Real.log x

theorem tangent_line_inv_g_at_0 
  (h₁ : ∀ x, g x = Real.log x) 
  (h₂ : ∀ x, x > 0): 
  ∃ m b, (∀ x y, y = g⁻¹ x → y - m * x = b) ∧ 
         (m = 1) ∧ 
         (b = 1) ∧ 
         (∀ x y, x - y + 1 = 0) := 
by
  sorry

end tangent_line_inv_g_at_0_l132_132236


namespace inequality_proof_l132_132351

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l132_132351


namespace calc_square_uncovered_area_l132_132477

theorem calc_square_uncovered_area :
  ∀ (side_length : ℕ) (circle_diameter : ℝ) (num_circles : ℕ),
    side_length = 16 →
    circle_diameter = (16 / 3) →
    num_circles = 9 →
    (side_length ^ 2) - num_circles * (Real.pi * (circle_diameter / 2) ^ 2) = 256 - 64 * Real.pi :=
by
  intros side_length circle_diameter num_circles h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end calc_square_uncovered_area_l132_132477


namespace angle_supplement_complement_l132_132600

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132600


namespace clock_angle_at_7_15_l132_132461

theorem clock_angle_at_7_15 :
  let hour_hand := 210 + 30 / 4,
      minute_hand := 90 in
  |hour_hand - minute_hand| = 127.5 := 
by
  let hour_hand := 210 + 30 / 4
  let minute_hand := 90
  have : hour_hand = 217.5
  have : minute_hand = 90
  have : |hour_hand - minute_hand| = |217.5 - 90|
  have : |127.5| = 127.5
  sorry   -- proof omitted

end clock_angle_at_7_15_l132_132461


namespace inequality_proof_l132_132346

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l132_132346


namespace angle_measure_is_60_l132_132637

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132637


namespace gcd_is_13_eval_at_neg1_l132_132854

-- Define the GCD problem
def gcd_117_182 : ℕ := gcd 117 182

-- Define the polynomial evaluation problem
def f (x : ℝ) : ℝ := 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

-- Formalize the statements to be proved
theorem gcd_is_13 : gcd_117_182 = 13 := 
by sorry

theorem eval_at_neg1 : f (-1) = 12 := 
by sorry

end gcd_is_13_eval_at_neg1_l132_132854


namespace square_side_length_s2_l132_132277

theorem square_side_length_s2 (s1 s2 s3 : ℕ)
  (h1 : s1 + s2 + s3 = 3322)
  (h2 : s1 - s2 + s3 = 2020) :
  s2 = 651 :=
by sorry

end square_side_length_s2_l132_132277


namespace candy_division_l132_132355

def pieces_per_bag (total_candies : ℕ) (bags : ℕ) : ℕ :=
total_candies / bags

theorem candy_division : pieces_per_bag 42 2 = 21 :=
by
  sorry

end candy_division_l132_132355


namespace largest_integer_l132_132911

def bin_op (n : ℤ) : ℤ := n - 5 * n

theorem largest_integer (n : ℤ) (h : 0 < n) (h' : bin_op n < 18) : n = 4 := sorry

end largest_integer_l132_132911


namespace find_u_values_l132_132073

namespace MathProof

variable (u v : ℝ)
variable (h1 : u ≠ 0) (h2 : v ≠ 0)
variable (h3 : u + 1/v = 8) (h4 : v + 1/u = 16/3)

theorem find_u_values : u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 :=
by {
  sorry
}

end MathProof

end find_u_values_l132_132073


namespace first_three_digits_of_x_are_571_l132_132561

noncomputable def x : ℝ := (10^2003 + 1)^(11/7)

theorem first_three_digits_of_x_are_571 : 
  ∃ d₁ d₂ d₃ : ℕ, 
  (d₁, d₂, d₃) = (5, 7, 1) ∧ 
  ∃ k : ℤ, 
  (x - k : ℝ) * 1000 = d₁ * 100 + d₂ * 10 + d₃ := 
by
  sorry

end first_three_digits_of_x_are_571_l132_132561


namespace total_students_in_line_l132_132300

-- Define the conditions
def students_in_front : Nat := 15
def students_behind : Nat := 12

-- Define the statement to prove: total number of students in line is 28
theorem total_students_in_line : students_in_front + 1 + students_behind = 28 := 
by 
  -- Placeholder for the proof
  sorry

end total_students_in_line_l132_132300


namespace inequality_am_gm_l132_132343

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l132_132343


namespace angle_measure_l132_132594

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132594


namespace no_solutions_then_a_eq_zero_l132_132488

theorem no_solutions_then_a_eq_zero (a b : ℝ) :
  (∀ x y : ℝ, ¬ (y^2 = x^2 + a * x + b ∧ x^2 = y^2 + a * y + b)) → a = 0 :=
by
  sorry

end no_solutions_then_a_eq_zero_l132_132488


namespace average_sale_over_six_months_l132_132310

theorem average_sale_over_six_months : 
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  average_sale = 3500 :=
by
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  show average_sale = 3500
  sorry

end average_sale_over_six_months_l132_132310


namespace travelers_on_liner_l132_132163

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l132_132163


namespace angle_measure_l132_132684

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132684


namespace interior_angle_regular_octagon_l132_132422

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l132_132422


namespace directrix_of_parabola_l132_132368

-- Define the parabola and the line conditions
def parabola (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def focus_line (x y : ℝ) := 2 * x + 3 * y - 8 = 0

-- Theorem stating that the directrix of the parabola is x = -4
theorem directrix_of_parabola (p : ℝ) (hx : ∃ x, ∃ y, focus_line x y) (hp : parabola p) :
  ∃ k : ℝ, k = 4 → ∀ x y : ℝ, (-x) = -4 :=
by
  sorry

end directrix_of_parabola_l132_132368


namespace angle_measure_is_60_l132_132635

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132635


namespace angle_complement_supplement_l132_132630

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132630


namespace meaningful_expression_range_l132_132913

theorem meaningful_expression_range (x : ℝ) : 
  (x - 1 ≥ 0) ∧ (x ≠ 3) ↔ (x ≥ 1 ∧ x ≠ 3) := 
by
  sorry

end meaningful_expression_range_l132_132913


namespace regular_octagon_interior_angle_l132_132400

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l132_132400


namespace right_triangle_AB_is_approximately_8point3_l132_132789

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem right_triangle_AB_is_approximately_8point3 :
  ∀ (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (BC AB : ℝ),
  angle_A = 40 ∧ angle_B = 90 ∧ BC = 7 →
  AB = 7 / tan_deg 40 →
  abs (AB - 8.3) < 0.1 :=
by
  intros A B C angle_A angle_B BC AB h_cond h_AB
  sorry

end right_triangle_AB_is_approximately_8point3_l132_132789


namespace happy_dictionary_problem_l132_132919

def smallest_positive_integer : ℕ := 1
def largest_negative_integer : ℤ := -1
def smallest_abs_rational : ℚ := 0

theorem happy_dictionary_problem : 
  smallest_positive_integer - largest_negative_integer + smallest_abs_rational = 2 := 
by
  sorry

end happy_dictionary_problem_l132_132919


namespace largest_y_value_l132_132836

theorem largest_y_value (y : ℝ) (h : 3*y^2 + 18*y - 90 = y*(y + 17)) : y ≤ 3 :=
by
  sorry

end largest_y_value_l132_132836


namespace kendall_tau_correct_l132_132981

-- Base Lean setup and list of dependencies might go here

structure TestScores :=
  (A : List ℚ)
  (B : List ℚ)

-- Constants from the problem
def scores : TestScores :=
  { A := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
  , B := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70] }

-- Function to calculate the Kendall rank correlation coefficient
noncomputable def kendall_tau (scores : TestScores) : ℚ :=
  -- the method of calculating Kendall tau could be very complex
  -- hence we assume the correct coefficient directly for the example
  0.51

-- The proof problem
theorem kendall_tau_correct : kendall_tau scores = 0.51 :=
by
  sorry

end kendall_tau_correct_l132_132981


namespace prob_greater_than_2_l132_132287

noncomputable section

open Probability

-- Let's define the random variable ξ following the normal distribution N(0, σ^2).
def ξ (σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := MeasureTheory.ProbabilityMeasure.ofReal le_real_of_is_finite measure_space volume

-- Given conditions in the problem
variable {σ : ℝ}
axiom h₁ : MeasureTheory.Measure.map (λ x : ℝ, x) (MeasureTheory.ProbabilityMeasure.toMeasure (ξ σ)) = 
                MeasureTheory.ProbabilityMeasure.toMeasure (MeasureTheory.ProbabilityMeasure.normal 0 σ)
axiom h₂ : ∫ x in -2..2, MeasureTheory.density (MeasureTheory.ProbabilityMeasure.toMeasure (ξ σ)) (λ x, 1) = 0.6

-- The question to prove
theorem prob_greater_than_2 : 
  ∫ x in 2..∞, MeasureTheory.density (MeasureTheory.ProbabilityMeasure.toMeasure (ξ σ)) (λ x, 1) = 0.2 :=
sorry

end prob_greater_than_2_l132_132287


namespace point_outside_circle_l132_132099

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a*x + b*y = 1 ∧ x^2 + y^2 = 1)) : a^2 + b^2 ≥ 1 :=
sorry

end point_outside_circle_l132_132099


namespace dima_is_mistaken_l132_132050

theorem dima_is_mistaken :
  (∃ n : Nat, n > 0 ∧ ∀ n, 3 * n = 4 * n) → False :=
by
  intros h
  obtain ⟨n, hn1, hn2⟩ := h
  have hn := (hn2 n)
  linarith

end dima_is_mistaken_l132_132050


namespace number_of_friends_shared_with_l132_132515

-- Conditions and given data
def doughnuts_samuel : ℕ := 2 * 12
def doughnuts_cathy : ℕ := 3 * 12
def total_doughnuts : ℕ := doughnuts_samuel + doughnuts_cathy
def each_person_doughnuts : ℕ := 6
def total_people := total_doughnuts / each_person_doughnuts
def samuel_and_cathy : ℕ := 2

-- Statement to prove - Number of friends they shared with
theorem number_of_friends_shared_with : (total_people - samuel_and_cathy) = 8 := by
  sorry

end number_of_friends_shared_with_l132_132515


namespace tangent_line_at_P_no_zero_points_sum_of_zero_points_l132_132361

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

/-- Given that f(x) = ln(x) - 2x, prove that the tangent line at point P(1, -2) has the equation x + y + 1 = 0. -/
theorem tangent_line_at_P (a : ℝ) (h : a = 2) : ∀ x y : ℝ, x + y + 1 = 0 :=
sorry

/-- Show that for f(x) = ln(x) - ax, the function f(x) has no zero points if a > 1/e. -/
theorem no_zero_points (a : ℝ) (h : a > 1 / Real.exp 1) : ¬∃ x : ℝ, f x a = 0 :=
sorry

/-- For f(x) = ln(x) - ax and x1 ≠ x2 such that f(x1) = f(x2) = 0, prove that x1 + x2 > 2 / a. -/
theorem sum_of_zero_points (a x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) : x₁ + x₂ > 2 / a :=
sorry

end tangent_line_at_P_no_zero_points_sum_of_zero_points_l132_132361


namespace selina_sells_5_shirts_l132_132947

theorem selina_sells_5_shirts
    (pants_price shorts_price shirts_price : ℕ)
    (pants_sold shorts_sold shirts_bought remaining_money : ℕ)
    (total_earnings : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  pants_sold = 3 →
  shorts_sold = 5 →
  shirts_bought = 2 →
  remaining_money = 30 →
  total_earnings = remaining_money + shirts_bought * 10 →
  total_earnings = 50 →
  total_earnings = pants_sold * pants_price + shorts_sold * shorts_price + 20 →
  20 / shirts_price = 5 :=
by
  sorry

end selina_sells_5_shirts_l132_132947


namespace column_heights_achievable_l132_132335

open Int

noncomputable def number_of_column_heights (n : ℕ) (h₁ h₂ h₃ : ℕ) : ℕ :=
  let min_height := n * h₁
  let max_height := n * h₃
  max_height - min_height + 1

theorem column_heights_achievable :
  number_of_column_heights 80 3 8 15 = 961 := by
  -- Proof goes here.
  sorry

end column_heights_achievable_l132_132335


namespace angle_solution_l132_132647

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132647


namespace operation_example_result_l132_132778

def myOperation (A B : ℕ) : ℕ := (A^2 + B^2) / 3

theorem operation_example_result : myOperation (myOperation 6 3) 9 = 102 := by
  sorry

end operation_example_result_l132_132778


namespace angle_measure_is_60_l132_132633

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l132_132633


namespace angle_measure_l132_132681

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132681


namespace angle_measure_l132_132679

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132679


namespace vegetables_sold_ratio_l132_132033

def totalMassInstalled (carrots zucchini broccoli : ℕ) : ℕ := carrots + zucchini + broccoli

def massSold (soldMass : ℕ) : ℕ := soldMass

def vegetablesSoldRatio (carrots zucchini broccoli soldMass : ℕ) : ℚ :=
  soldMass / (carrots + zucchini + broccoli)

theorem vegetables_sold_ratio
  (carrots zucchini broccoli soldMass : ℕ)
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8)
  (h_soldMass : soldMass = 18) :
  vegetablesSoldRatio carrots zucchini broccoli soldMass = 1 / 2 := by
  sorry

end vegetables_sold_ratio_l132_132033


namespace Q_value_ratio_l132_132331

noncomputable def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

noncomputable def roots : Fin 2009 → ℂ := sorry -- Define distinct roots s1, s2, ..., s2009

noncomputable def Q (z : ℂ) : ℂ := sorry -- Define the polynomial Q of degree 2009

theorem Q_value_ratio :
  (∀ j : Fin 2009, Q (roots j + 2 / roots j) = 0) →
  (Q (2) / Q (-2) = 361 / 400) :=
sorry

end Q_value_ratio_l132_132331


namespace students_passed_correct_l132_132157

-- Define the number of students in ninth grade.
def students_total : ℕ := 180

-- Define the number of students who bombed their finals.
def students_bombed : ℕ := students_total / 4

-- Define the number of students remaining after removing those who bombed.
def students_remaining_after_bombed : ℕ := students_total - students_bombed

-- Define the number of students who didn't show up to take the test.
def students_didnt_show : ℕ := students_remaining_after_bombed / 3

-- Define the number of students remaining after removing those who didn't show up.
def students_remaining_after_no_show : ℕ := students_remaining_after_bombed - students_didnt_show

-- Define the number of students who got less than a D.
def students_less_than_d : ℕ := 20

-- Define the number of students who passed.
def students_passed : ℕ := students_remaining_after_no_show - students_less_than_d

-- Statement to prove the number of students who passed is 70.
theorem students_passed_correct : students_passed = 70 := by
  -- Proof will be inserted here.
  sorry

end students_passed_correct_l132_132157


namespace chloe_pawn_loss_l132_132523

theorem chloe_pawn_loss (sophia_lost : ℕ) (total_left : ℕ) (total_initial : ℕ) (each_start : ℕ) (sophia_initial : ℕ) :
  sophia_lost = 5 → total_left = 10 → each_start = 8 → total_initial = 16 → sophia_initial = 8 →
  ∃ (chloe_lost : ℕ), chloe_lost = 1 :=
by
  sorry

end chloe_pawn_loss_l132_132523


namespace find_smallest_number_l132_132129

variable (x : ℕ)

def second_number := 2 * x
def third_number := 4 * second_number x
def average := (x + second_number x + third_number x) / 3

theorem find_smallest_number (h : average x = 165) : x = 45 := by
  sorry

end find_smallest_number_l132_132129


namespace find_a_l132_132077

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = 3 * x^(a-2) - 2) (h_cond : f 2 = 4) : a = 3 :=
by
  sorry

end find_a_l132_132077


namespace angle_solution_l132_132653

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132653


namespace y_intercepts_count_l132_132087

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count_l132_132087


namespace sqrt_fraction_value_l132_132843

theorem sqrt_fraction_value (a b c d : Nat) (h : a = 2 ∧ b = 0 ∧ c = 2 ∧ d = 3) : 
  Real.sqrt (2023 / (a + b + c + d)) = 17 := by
  sorry

end sqrt_fraction_value_l132_132843


namespace slope_of_line_determined_by_solutions_l132_132840

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l132_132840


namespace proposition_not_hold_for_4_l132_132705

variable (P : ℕ → Prop)

axiom induction_step (k : ℕ) (hk : k > 0) : P k → P (k + 1)
axiom base_case : ¬ P 5

theorem proposition_not_hold_for_4 : ¬ P 4 :=
sorry

end proposition_not_hold_for_4_l132_132705


namespace subcommittees_with_at_least_one_teacher_l132_132542

-- Define the total number of members and the count of teachers
def total_members : ℕ := 12
def teacher_count : ℕ := 5
def subcommittee_size : ℕ := 5

-- Define binomial coefficient calculation
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Problem statement: number of five-person subcommittees with at least one teacher
theorem subcommittees_with_at_least_one_teacher :
  binom total_members subcommittee_size - binom (total_members - teacher_count) subcommittee_size = 771 := by
  sorry

end subcommittees_with_at_least_one_teacher_l132_132542


namespace smallest_solution_x4_minus_40x2_plus_400_eq_zero_l132_132182

theorem smallest_solution_x4_minus_40x2_plus_400_eq_zero :
  ∃ x : ℝ, (x^4 - 40 * x^2 + 400 = 0) ∧ (∀ y : ℝ, (y^4 - 40 * y^2 + 400 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_40x2_plus_400_eq_zero_l132_132182


namespace fewer_cans_collected_today_than_yesterday_l132_132694

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end fewer_cans_collected_today_than_yesterday_l132_132694


namespace angle_measure_l132_132619

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132619


namespace angle_supplement_complement_l132_132655

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132655


namespace angle_measure_l132_132675

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132675


namespace edward_toy_cars_l132_132211

def initial_amount : ℝ := 17.80
def cost_per_car : ℝ := 0.95
def cost_of_race_track : ℝ := 6.00
def remaining_amount : ℝ := 8.00

theorem edward_toy_cars : ∃ (n : ℕ), initial_amount - remaining_amount = n * cost_per_car + cost_of_race_track ∧ n = 4 := by
  sorry

end edward_toy_cars_l132_132211


namespace find_grid_values_l132_132554

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l132_132554


namespace average_weight_l132_132795

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l132_132795


namespace correct_statement_about_parabola_l132_132061

theorem correct_statement_about_parabola (x : ℝ) : 
  let y := -2 * (x - 1)^2 + 3 in
  (∀ x, y = -2 * (x - 1)^2 + 3 → ∃ S : Prop, S = "The axis of symmetry is the line x = 1") :=

sorry

end correct_statement_about_parabola_l132_132061


namespace solutions_count_l132_132475

noncomputable def number_of_solutions (x y z : ℚ) : ℕ :=
if (x^2 - y * z = 1) ∧ (y^2 - x * z = 1) ∧ (z^2 - x * y = 1)
then 6
else 0

theorem solutions_count : number_of_solutions x y z = 6 :=
sorry

end solutions_count_l132_132475


namespace remainder_of_13_plus_x_mod_29_l132_132812

theorem remainder_of_13_plus_x_mod_29
  (x : ℕ)
  (hx : 8 * x ≡ 1 [MOD 29])
  (hp : 0 < x) : 
  (13 + x) % 29 = 18 :=
sorry

end remainder_of_13_plus_x_mod_29_l132_132812


namespace relationship_among_f_values_l132_132762

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0)

theorem relationship_among_f_values (h₀ : 0 < 2) (h₁ : 2 < 3) :
  f 0 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end relationship_among_f_values_l132_132762


namespace initial_weight_of_load_l132_132321

variable (W : ℝ)
variable (h : 0.8 * 0.9 * W = 36000)

theorem initial_weight_of_load :
  W = 50000 :=
by
  sorry

end initial_weight_of_load_l132_132321


namespace triangle_area_is_rational_l132_132990

-- Definition of the area of a triangle given vertices with integer coordinates
def triangle_area (x1 x2 x3 y1 y2 y3 : ℤ) : ℚ :=
0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The theorem stating that the area of a triangle formed by points with integer coordinates is rational
theorem triangle_area_is_rational (x1 x2 x3 y1 y2 y3 : ℤ) :
  ∃ (area : ℚ), area = triangle_area x1 x2 x3 y1 y2 y3 :=
by
  sorry

end triangle_area_is_rational_l132_132990


namespace series_sum_199_l132_132732

noncomputable def seriesSum : ℕ → ℤ
| 0       => 1
| (n + 1) => seriesSum n + (-1)^(n + 1) * (n + 2)

theorem series_sum_199 : seriesSum 199 = 100 := 
by
  sorry

end series_sum_199_l132_132732


namespace angle_measure_l132_132597

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132597


namespace garden_breadth_l132_132917

theorem garden_breadth (P L B : ℕ) (h₁ : P = 950) (h₂ : L = 375) (h₃ : P = 2 * (L + B)) : B = 100 := by
  sorry

end garden_breadth_l132_132917


namespace plates_are_multiple_of_eleven_l132_132208

theorem plates_are_multiple_of_eleven
    (P : ℕ)    -- Number of plates
    (S : ℕ := 33)    -- Number of spoons
    (g : ℕ := 11)    -- Greatest number of groups
    (hS : S % g = 0)    -- Condition: All spoons can be divided into these groups evenly
    (hP : ∀ (k : ℕ), P = k * g) : ∃ x : ℕ, P = 11 * x :=
by
  sorry

end plates_are_multiple_of_eleven_l132_132208


namespace circle_center_radius_l132_132283

theorem circle_center_radius : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (2, 0) ∧ radius = 2 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ (x - 2)^2 + y^2 = 4 :=
by
  sorry

end circle_center_radius_l132_132283


namespace expression_evaluation_l132_132519

variable (x y : ℤ)

theorem expression_evaluation (h₁ : x = -1) (h₂ : y = 1) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 2 :=
by
  rw [h₁, h₂]
  have h₃ : (-1 + 1) * (-1 - 1) - (4 * (-1)^3 * 1 - 8 * (-1) * 1^3) / (2 * (-1) * 1) = (-2) - (-10 / -2) := by sorry
  have h₄ : (-2) - 5 = 2 := by sorry
  sorry

end expression_evaluation_l132_132519


namespace sumata_family_miles_driven_per_day_l132_132280

theorem sumata_family_miles_driven_per_day :
  let total_miles := 1837.5
  let number_of_days := 13.5
  let miles_per_day := total_miles / number_of_days
  (miles_per_day : Real) = 136.1111 :=
by
  sorry

end sumata_family_miles_driven_per_day_l132_132280


namespace hermia_elected_probability_l132_132267

noncomputable def probability_h ispected_president (n : ℕ) (h : n % 2 = 1) : ℚ :=
  (2^n - 1) / (n * 2^(n-1))

theorem hermia_elected_probability (n : ℕ) (h : n % 2 = 1) :
  let P := probability_h ispected_president n h in 
  hermia_elected_probability = P := 
  sorry

end hermia_elected_probability_l132_132267


namespace evaluate_expression_l132_132882

theorem evaluate_expression : (1 - 1 / (1 - 1 / (1 + 2))) = (-1 / 2) :=
by sorry

end evaluate_expression_l132_132882


namespace sequence_remainder_prime_l132_132929

theorem sequence_remainder_prime (p : ℕ) (hp : Nat.Prime p) (x : ℕ → ℕ)
  (h1 : ∀ i, 0 ≤ i ∧ i < p → x i = i)
  (h2 : ∀ n, n ≥ p → x n = x (n-1) + x (n-p)) :
  (x (p^3) % p) = p - 1 :=
sorry

end sequence_remainder_prime_l132_132929


namespace angle_measure_l132_132588

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132588


namespace ratio_AB_CD_lengths_AB_CD_l132_132851

-- Given conditions as definitions
def ABD_triangle (A B D : Point) : Prop := true  -- In quadrilateral ABCD, a diagonal BD is drawn
def BCD_triangle (B C D : Point) : Prop := true  -- Circles are inscribed in triangles ABD and BCD
def Line_through_B_center_AM_M (A B D M : Point) (AM MD : ℚ) : Prop :=
  (AM = 8/5) ∧ (MD = 12/5)
def Line_through_D_center_BN_N (B C D N : Point) (BN NC : ℚ) : Prop :=
  (BN = 30/11) ∧ (NC = 25/11)

-- Mathematically equivalent proof problems
theorem ratio_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB / CD = 4 / 5 :=
by
  sorry

theorem lengths_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB + CD = 9 ∧
  AB - CD = -1 :=
by 
  sorry

end ratio_AB_CD_lengths_AB_CD_l132_132851


namespace probability_of_xyz_72_l132_132687

noncomputable def probability_product_is_72 : ℚ :=
  let dice := {1, 2, 3, 4, 5, 6}
  let outcomes := {e : ℕ × ℕ × ℕ | e.1 ∈ dice ∧ e.2.1 ∈ dice ∧ e.2.2 ∈ dice}
  let favourable_outcomes := {e : ℕ × ℕ × ℕ | e ∈ outcomes ∧ e.1 * e.2.1 * e.2.2 = 72}
  (favourable_outcomes.to_finset.card : ℚ) / outcomes.to_finset.card

theorem probability_of_xyz_72 :
  probability_product_is_72 = 1 / 24 :=
sorry

end probability_of_xyz_72_l132_132687


namespace david_more_pushups_l132_132332

theorem david_more_pushups (d z : ℕ) (h1 : d = 51) (h2 : d + z = 53) : d - z = 49 := by
  sorry

end david_more_pushups_l132_132332


namespace stockings_total_cost_l132_132509

-- Defining the conditions
def total_stockings : ℕ := 9
def original_price_per_stocking : ℝ := 20
def discount_rate : ℝ := 0.10
def monogramming_cost_per_stocking : ℝ := 5

-- Calculate the total cost of stockings
theorem stockings_total_cost :
  total_stockings * ((original_price_per_stocking * (1 - discount_rate)) + monogramming_cost_per_stocking) = 207 := 
by
  sorry

end stockings_total_cost_l132_132509


namespace number_of_terms_l132_132365

variable {α : Type} [LinearOrderedField α]

def sum_of_arithmetic_sequence (a₁ aₙ d : α) (n : ℕ) : α :=
  n * (a₁ + aₙ) / 2

theorem number_of_terms (a₁ aₙ : α) (d : α) (n : ℕ)
  (h₀ : 4 * (2 * a₁ + 3 * d) / 2 = 21)
  (h₁ : 4 * (2 * aₙ - 3 * d) / 2 = 67)
  (h₂ : sum_of_arithmetic_sequence a₁ aₙ d n = 286) :
  n = 26 :=
sorry

end number_of_terms_l132_132365


namespace regular_octagon_interior_angle_l132_132417

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l132_132417


namespace q_sufficient_but_not_necessary_for_p_l132_132737

variable (x : ℝ)

def p : Prop := (x - 2) ^ 2 ≤ 1
def q : Prop := 2 / (x - 1) ≥ 1

theorem q_sufficient_but_not_necessary_for_p : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) := 
by
  sorry

end q_sufficient_but_not_necessary_for_p_l132_132737


namespace max_f_value_inequality_m_n_l132_132234

section
variable (x : ℝ)

def f (x : ℝ) := abs (x - 1) - 2 * abs (x + 1)

theorem max_f_value : ∃ k, (∀ x : ℝ, f x ≤ k) ∧ (∃ x₀ : ℝ, f x₀ = k) ∧ k = 2 := 
by sorry

theorem inequality_m_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 1 / m + 1 / (2 * n) = 2) :
  m + 2 * n ≥ 2 :=
by sorry

end

end max_f_value_inequality_m_n_l132_132234


namespace parabola_intersection_radius_sqr_l132_132005

theorem parabola_intersection_radius_sqr {x y : ℝ} :
  (y = (x - 2)^2) →
  (x - 3 = (y + 2)^2) →
  ∃ r, r^2 = 9 / 2 :=
by
  intros h1 h2
  sorry

end parabola_intersection_radius_sqr_l132_132005


namespace people_on_train_after_third_stop_l132_132833

variable (initial_people : ℕ) (off_1 boarded_1 off_2 boarded_2 off_3 boarded_3 : ℕ)

def people_after_first_stop (initial : ℕ) (off_1 boarded_1 : ℕ) : ℕ :=
  initial - off_1 + boarded_1

def people_after_second_stop (first_stop : ℕ) (off_2 boarded_2 : ℕ) : ℕ :=
  first_stop - off_2 + boarded_2

def people_after_third_stop (second_stop : ℕ) (off_3 boarded_3 : ℕ) : ℕ :=
  second_stop - off_3 + boarded_3

theorem people_on_train_after_third_stop :
  people_after_third_stop (people_after_second_stop (people_after_first_stop initial_people off_1 boarded_1) off_2 boarded_2) off_3 boarded_3 = 42 :=
  by
    have initial_people := 48
    have off_1 := 12
    have boarded_1 := 7
    have off_2 := 15
    have boarded_2 := 9
    have off_3 := 6
    have boarded_3 := 11
    sorry

end people_on_train_after_third_stop_l132_132833


namespace exists_visible_point_l132_132200

open Nat -- to use natural numbers and their operations

def is_visible (x y : ℤ) : Prop :=
  Int.gcd x y = 1

theorem exists_visible_point (n : ℕ) (hn : n > 0) :
  ∃ a b : ℤ, is_visible a b ∧
  ∀ (P : ℤ × ℤ), (P ≠ (a, b) → (Int.sqrt ((P.fst - a) * (P.fst - a) + (P.snd - b) * (P.snd - b)) > n)) :=
sorry

end exists_visible_point_l132_132200


namespace angle_measure_l132_132566

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132566


namespace medium_supermarkets_in_sample_l132_132031

-- Definitions of the conditions
def total_supermarkets : ℕ := 200 + 400 + 1400
def prop_medium_supermarkets : ℚ := 400 / total_supermarkets
def sample_size : ℕ := 100

-- Problem: Prove that the number of medium-sized supermarkets in the sample is 20.
theorem medium_supermarkets_in_sample : 
  (sample_size * prop_medium_supermarkets) = 20 :=
by
  sorry

end medium_supermarkets_in_sample_l132_132031


namespace regular_octagon_interior_angle_l132_132381

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l132_132381


namespace last_year_ticket_cost_l132_132806

theorem last_year_ticket_cost (this_year_cost : ℝ) (increase_percentage : ℝ) (last_year_cost : ℝ) :
  this_year_cost = last_year_cost * (1 + increase_percentage) ↔ last_year_cost = 85 :=
by
  let this_year_cost := 102
  let increase_percentage := 0.20
  sorry

end last_year_ticket_cost_l132_132806


namespace steve_needs_28_feet_of_wood_l132_132528

-- Define the required lengths
def lengths_4_feet : Nat := 6
def lengths_2_feet : Nat := 2

-- Define the wood length in feet for each type
def wood_length_4 : Nat := 4
def wood_length_2 : Nat := 2

-- Total feet of wood required
def total_wood : Nat := lengths_4_feet * wood_length_4 + lengths_2_feet * wood_length_2

-- The theorem to prove that the total amount of wood required is 28 feet
theorem steve_needs_28_feet_of_wood : total_wood = 28 :=
by
  sorry

end steve_needs_28_feet_of_wood_l132_132528


namespace greening_task_equation_l132_132023

variable (x : ℝ)

theorem greening_task_equation (h1 : 600000 = 600 * 1000)
    (h2 : ∀ a b : ℝ, a * 1.25 = b -> b = a * (1 + 25 / 100)) :
  (60 * (1 + 25 / 100)) / x - 60 / x = 30 := by
  sorry

end greening_task_equation_l132_132023


namespace liz_car_percentage_sale_l132_132123

theorem liz_car_percentage_sale (P : ℝ) (h1 : 30000 = P - 2500) (h2 : 26000 = P * (80 / 100)) : 80 = 80 :=
by 
  sorry

end liz_car_percentage_sale_l132_132123


namespace combined_flock_size_after_5_years_l132_132883

noncomputable def initial_flock_size : ℕ := 100
noncomputable def ducks_killed_per_year : ℕ := 20
noncomputable def ducks_born_per_year : ℕ := 30
noncomputable def years_passed : ℕ := 5
noncomputable def other_flock_size : ℕ := 150

theorem combined_flock_size_after_5_years
  (init_size : ℕ := initial_flock_size)
  (killed_per_year : ℕ := ducks_killed_per_year)
  (born_per_year : ℕ := ducks_born_per_year)
  (years : ℕ := years_passed)
  (other_size : ℕ := other_flock_size) :
  init_size + (years * (born_per_year - killed_per_year)) + other_size = 300 := by
  -- The formal proof would go here.
  sorry

end combined_flock_size_after_5_years_l132_132883


namespace angle_measure_l132_132590

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132590


namespace number_of_multiples_of_15_l132_132090

theorem number_of_multiples_of_15 (a b : ℕ) (h₁ : a = 15) (h₂ : b = 305) : 
  ∃ n : ℕ, n = 20 ∧ ∀ k, (1 ≤ k ∧ k ≤ n) → (15 * k) ≥ a ∧ (15 * k) ≤ b := by
  sorry

end number_of_multiples_of_15_l132_132090


namespace worker_net_salary_change_l132_132991

theorem worker_net_salary_change (S : ℝ) :
  let final_salary := S * 1.15 * 0.90 * 1.20 * 0.95
  let net_change := final_salary - S
  net_change = 0.0355 * S := by
  -- Proof goes here
  sorry

end worker_net_salary_change_l132_132991


namespace find_single_digit_A_l132_132710

theorem find_single_digit_A (A : ℕ) (h1 : A < 10) (h2 : (11 * A)^2 = 5929) : A = 7 := 
sorry

end find_single_digit_A_l132_132710


namespace restaurant_pizzas_l132_132319

theorem restaurant_pizzas (lunch dinner total : ℕ) (h_lunch : lunch = 9) (h_dinner : dinner = 6) 
  (h_total : total = lunch + dinner) : total = 15 :=
by
  rw [h_lunch, h_dinner, h_total]
  norm_num

end restaurant_pizzas_l132_132319


namespace integer_solutions_of_quadratic_eq_l132_132229

theorem integer_solutions_of_quadratic_eq (b : ℤ) :
  ∃ p q : ℤ, (p+9) * (q+9) = 81 ∧ p + q = -b ∧ p * q = 9*b :=
sorry

end integer_solutions_of_quadratic_eq_l132_132229


namespace ab_max_min_sum_l132_132772

-- Define the conditions
variables {a b : ℝ}
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 4 * b = 4

-- Problem (1)
theorem ab_max : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → a * b ≤ 1 :=
by sorry

-- Problem (2)
theorem min_sum : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → (1 / a) + (4 / b) ≥ 25 / 4 :=
by sorry

end ab_max_min_sum_l132_132772


namespace inequality_proof_l132_132353

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l132_132353


namespace average_weight_of_three_l132_132797

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l132_132797


namespace min_value_of_x_l132_132702

-- Definitions for the conditions given in the problem
def men := 4
def women (x : ℕ) := x
def min_x := 594

-- Definition of the probability p
def C (n k : ℕ) : ℕ := sorry -- Define the binomial coefficient properly

def probability (x : ℕ) : ℚ :=
  (2 * (C (x+1) 2) + (x + 1)) /
  (C (x + 1) 3 + 3 * (C (x + 1) 2) + (x + 1))

-- The theorem statement to prove
theorem min_value_of_x (x : ℕ) : probability x ≤ 1 / 100 →  x = min_x := 
by
  sorry

end min_value_of_x_l132_132702


namespace regular_octagon_angle_l132_132454

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l132_132454


namespace cost_per_tshirt_l132_132734
-- Import necessary libraries

-- Define the given conditions
def t_shirts : ℕ := 20
def total_cost : ℝ := 199

-- Define the target proof statement
theorem cost_per_tshirt : (total_cost / t_shirts) = 9.95 := 
sorry

end cost_per_tshirt_l132_132734


namespace find_number_l132_132185

def sum := 555 + 445
def difference := 555 - 445
def quotient := 2 * difference
def remainder := 30
def N : ℕ := 220030

theorem find_number (N : ℕ) : 
  N = sum * quotient + remainder :=
  by
    sorry

end find_number_l132_132185


namespace cycling_journey_l132_132994

theorem cycling_journey :
  ∃ y : ℚ, 0 < y ∧ y <= 12 ∧ (15 * y + 10 * (12 - y) = 150) ∧ y = 6 :=
by
  sorry

end cycling_journey_l132_132994


namespace ineq_five_times_x_minus_six_gt_one_l132_132884

variable {x : ℝ}

theorem ineq_five_times_x_minus_six_gt_one (x : ℝ) : 5 * x - 6 > 1 :=
sorry

end ineq_five_times_x_minus_six_gt_one_l132_132884


namespace checkerboard_corners_sum_l132_132506

theorem checkerboard_corners_sum : 
  let N : ℕ := 9 
  let corners := [1, 9, 73, 81]
  (corners.sum = 164) := by
  sorry

end checkerboard_corners_sum_l132_132506


namespace largest_integer_satisfying_l132_132209

theorem largest_integer_satisfying (x : ℤ) : 
  (∃ x, (2/7 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < 3/4) → x = 4 := 
by 
  sorry

end largest_integer_satisfying_l132_132209


namespace set_intersection_l132_132360

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x^2 < 4

theorem set_intersection : {x | A x} ∩ {x | B x} = {x | 0 < x ∧ x < 2} := by
  sorry

end set_intersection_l132_132360


namespace find_digits_l132_132888

/-- 
  Find distinct digits A, B, C, and D such that 9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B).
 -/
theorem find_digits
  (A B C D : ℕ)
  (hA : A ≠ B) (hA : A ≠ C) (hA : A ≠ D)
  (hB : B ≠ C) (hB : B ≠ D)
  (hC : C ≠ D)
  (hNonZeroB : B ≠ 0) :
  9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B) ↔ (A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7) := by
  sorry

end find_digits_l132_132888


namespace ratio_a_d_l132_132187

theorem ratio_a_d (a b c d : ℕ) 
  (hab : a * 4 = b * 3) 
  (hbc : b * 9 = c * 7) 
  (hcd : c * 7 = d * 5) : 
  a * 12 = d :=
sorry

end ratio_a_d_l132_132187


namespace sports_day_results_l132_132198

-- Conditions and questions
variables (a b c : ℕ)
variables (class1_score class2_score class3_score class4_score : ℕ)

-- Conditions given in the problem
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom a_gt_b_gt_c : a > b ∧ b > c
axiom no_ties : (class1_score ≠ class2_score) ∧ (class2_score ≠ class3_score) ∧ (class3_score ≠ class4_score) ∧ (class1_score ≠ class3_score) ∧ (class1_score ≠ class4_score) ∧ (class2_score ≠ class4_score)
axiom class_scores : class1_score + class2_score + class3_score + class4_score = 40

-- To prove
theorem sports_day_results : a + b + c = 8 ∧ a = 5 :=
by
  sorry

end sports_day_results_l132_132198


namespace budget_percentage_for_genetically_modified_organisms_l132_132197

theorem budget_percentage_for_genetically_modified_organisms
  (microphotonics : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (industrial_lubricants : ℝ)
  (astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 15 →
  industrial_lubricants = 8 →
  astrophysics_degrees = 72 →
  (72 / 360) * 100 = 20 →
  100 - (14 + 24 + 15 + 8 + 20) = 19 :=
  sorry

end budget_percentage_for_genetically_modified_organisms_l132_132197


namespace inequality_one_inequality_two_l132_132270

noncomputable def primes : Set ℕ := {p | Nat.prime p}

theorem inequality_one (n : ℕ) (h : 3 ≤ n) : 
  (∑ p in primes.filter (λ p, p ≤ n), (1 : ℝ) / p) ≥ Real.log (Real.log n) + O(1) :=
sorry

theorem inequality_two (n k : ℕ) (h1 : 3 ≤ n) (h2 : 0 < k) : 
  (∑ p in primes.filter (λ p, p ≤ n), (1 : ℝ) / p) ≤ (Real.ofNat k ! * Real.ofNat k * Real.log n)^(1 / Real.ofNat k) :=
sorry

end inequality_one_inequality_two_l132_132270


namespace percentage_of_mixture_X_is_13_333_l132_132516

variable (X Y : ℝ) (P : ℝ)

-- Conditions
def mixture_X_contains_40_percent_ryegrass : Prop := X = 0.40
def mixture_Y_contains_25_percent_ryegrass : Prop := Y = 0.25
def final_mixture_contains_27_percent_ryegrass : Prop := 0.4 * P + 0.25 * (100 - P) = 27

-- The goal
theorem percentage_of_mixture_X_is_13_333
    (h1 : mixture_X_contains_40_percent_ryegrass X)
    (h2 : mixture_Y_contains_25_percent_ryegrass Y)
    (h3 : final_mixture_contains_27_percent_ryegrass P) :
  P = 200 / 15 := by
  sorry

end percentage_of_mixture_X_is_13_333_l132_132516


namespace tom_age_is_19_l132_132173

-- Define the ages of Carla, Tom, Dave, and Emily
variable (C : ℕ) -- Carla's age

-- Conditions
def tom_age := 2 * C - 1
def dave_age := C + 3
def emily_age := C / 2

-- Sum of their ages equating to 48
def total_age := C + tom_age C + dave_age C + emily_age C

-- Theorem to be proven
theorem tom_age_is_19 (h : total_age C = 48) : tom_age C = 19 := 
by {
  sorry
}

end tom_age_is_19_l132_132173


namespace parabola_y_intercepts_zero_l132_132086

-- Define the quadratic equation
def quadratic (a b c y: ℝ) : ℝ := a * y^2 + b * y + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Condition: equation of the parabola and discriminant calculation
def parabola_equation : Prop := 
  let a := 3
  let b := -4
  let c := 5
  discriminant a b c < 0

-- Statement to prove
theorem parabola_y_intercepts_zero : 
  (parabola_equation) → (∀ y : ℝ, quadratic 3 (-4) 5 y ≠ 0) :=
by
  intro h
  sorry

end parabola_y_intercepts_zero_l132_132086


namespace boys_in_choir_l132_132159

theorem boys_in_choir
  (h1 : 20 + 2 * 20 + 16 + b = 88)
  : b = 12 :=
by
  sorry

end boys_in_choir_l132_132159


namespace correct_statements_l132_132766

variables {n : ℕ}
noncomputable def S (n : ℕ) : ℝ := (n + 1) / n
noncomputable def T (n : ℕ) : ℝ := (n + 1)
noncomputable def a (n : ℕ) : ℝ := if n = 1 then 2 else (-(1:ℝ)) / (n * (n - 1))

theorem correct_statements (n : ℕ) (hn : n ≠ 0) :
  (S n + T n = S n * T n) ∧ (a 1 = 2) ∧ (∀ n, ∃ d, ∀ m, T (n + m) - T n = m * d) ∧ (S n = (n + 1) / n) :=
by
  sorry

end correct_statements_l132_132766


namespace scientific_notation_000073_l132_132538

theorem scientific_notation_000073 : 0.000073 = 7.3 * 10^(-5) := by
  sorry

end scientific_notation_000073_l132_132538


namespace necessarily_negative_l132_132276

theorem necessarily_negative (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : -2 < b ∧ b < 0) (h3 : 0 < c ∧ c < 1) : b + c < 0 :=
sorry

end necessarily_negative_l132_132276


namespace factorize_eq_l132_132221

theorem factorize_eq (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_eq_l132_132221


namespace stockings_total_cost_l132_132508

-- Defining the conditions
def total_stockings : ℕ := 9
def original_price_per_stocking : ℝ := 20
def discount_rate : ℝ := 0.10
def monogramming_cost_per_stocking : ℝ := 5

-- Calculate the total cost of stockings
theorem stockings_total_cost :
  total_stockings * ((original_price_per_stocking * (1 - discount_rate)) + monogramming_cost_per_stocking) = 207 := 
by
  sorry

end stockings_total_cost_l132_132508


namespace arrange_plants_in_a_row_l132_132727

-- Definitions for the conditions
def basil_plants : ℕ := 5 -- Number of basil plants
def tomato_plants : ℕ := 4 -- Number of tomato plants

-- Theorem statement asserting the number of ways to arrange the plants
theorem arrange_plants_in_a_row : 
  let total_items := basil_plants + 1,
      ways_to_arrange_total_items := Nat.factorial total_items,
      ways_to_arrange_tomato_group := Nat.factorial tomato_plants in
  (ways_to_arrange_total_items * ways_to_arrange_tomato_group) = 17280 := 
by
  sorry

end arrange_plants_in_a_row_l132_132727


namespace restore_grid_l132_132556

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l132_132556


namespace cafeteria_green_apples_l132_132961

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

end cafeteria_green_apples_l132_132961


namespace find_radius_yz_l132_132103

-- Define the setup for the centers of the circles and their radii
def circle_with_center (c : Type*) (radius : ℝ) : Prop := sorry
def tangent_to (c₁ c₂ : Type*) : Prop := sorry

-- Given conditions
variable (O X Y Z : Type*)
variable (r : ℝ)
variable (Xe_radius : circle_with_center X 1)
variable (O_radius : circle_with_center O 2)
variable (XtangentO : tangent_to X O)
variable (YtangentO : tangent_to Y O)
variable (YtangentX : tangent_to Y X)
variable (YtangentZ : tangent_to Y Z)
variable (ZtangentO : tangent_to Z O)
variable (ZtangentX : tangent_to Z X)
variable (ZtangentY : tangent_to Z Y)

-- The theorem to prove
theorem find_radius_yz :
  r = 8 / 9 := sorry

end find_radius_yz_l132_132103


namespace vector_addition_dot_product_l132_132083

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

theorem vector_addition :
  let c := (1, 2) + (3, 1)
  c = (4, 3) := by
  sorry

theorem dot_product :
  let d := (1 * 3 + 2 * 1)
  d = 5 := by
  sorry

end vector_addition_dot_product_l132_132083


namespace students_not_making_cut_l132_132022

theorem students_not_making_cut
  (girls boys called_back : ℕ) 
  (h1 : girls = 39) 
  (h2 : boys = 4) 
  (h3 : called_back = 26) :
  (girls + boys) - called_back = 17 := 
by sorry

end students_not_making_cut_l132_132022


namespace angle_of_inclination_l132_132180

theorem angle_of_inclination (x y : ℝ) (θ : ℝ) :
  (x - y - 1 = 0) → θ = 45 :=
by
  sorry

end angle_of_inclination_l132_132180


namespace bounded_sequence_l132_132830

theorem bounded_sequence (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 2)
  (h_rec : ∀ n : ℕ, a (n + 2) = (a (n + 1) + a n) / Nat.gcd (a n) (a (n + 1))) :
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M := 
sorry

end bounded_sequence_l132_132830


namespace interior_angle_regular_octagon_l132_132436

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l132_132436


namespace base_4_representation_156_l132_132970

theorem base_4_representation_156 :
  ∃ b3 b2 b1 b0 : ℕ,
    156 = b3 * 4^3 + b2 * 4^2 + b1 * 4^1 + b0 * 4^0 ∧
    b3 = 2 ∧ b2 = 1 ∧ b1 = 3 ∧ b0 = 0 :=
by
  have h1 : 156 = 2 * 4^3 + 28 := by norm_num
  have h2 : 28 = 1 * 4^2 + 12 := by norm_num
  have h3 : 12 = 3 * 4^1 + 0 := by norm_num
  refine ⟨2, 1, 3, 0, _, rfl, rfl, rfl, rfl⟩
  rw [h1, h2, h3]
  norm_num

end base_4_representation_156_l132_132970


namespace no_valid_k_exists_l132_132325

theorem no_valid_k_exists {k : ℕ} : ¬(∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p + q = 41 ∧ p * q = k) :=
by
  sorry

end no_valid_k_exists_l132_132325


namespace prob1_prob2_l132_132237

-- Define the polynomial function
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Problem 1: Prove |b| ≤ 1, given conditions
theorem prob1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : |b| ≤ 1 :=
sorry

-- Problem 2: Find a = 2, given conditions
theorem prob2 (a b c : ℝ) 
  (h1 : polynomial a b c 0 = -1) 
  (h2 : polynomial a b c 1 = 1) 
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : 
  a = 2 :=
sorry

end prob1_prob2_l132_132237


namespace eval_g_l132_132877

def g (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + x + 1

theorem eval_g : 3 * g 2 + 2 * g (-2) = -9 := 
by {
  sorry
}

end eval_g_l132_132877


namespace history_paper_pages_l132_132818

theorem history_paper_pages (p d : ℕ) (h1 : p = 11) (h2 : d = 3) : p * d = 33 :=
by
  sorry

end history_paper_pages_l132_132818


namespace angle_measure_l132_132572

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132572


namespace regular_octagon_interior_angle_l132_132380

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l132_132380


namespace three_gorges_scientific_notation_l132_132535

theorem three_gorges_scientific_notation :
  ∃a n : ℝ, (1 ≤ |a| ∧ |a| < 10) ∧ (798.5 * 10^1 = a * 10^n) ∧ a = 7.985 ∧ n = 2 :=
by
  sorry

end three_gorges_scientific_notation_l132_132535


namespace tan_addition_identity_l132_132549

theorem tan_addition_identity 
  (tan_30 : Real := Real.tan (Real.pi / 6))
  (tan_15 : Real := 2 - Real.sqrt 3) : 
  tan_15 + tan_30 + tan_15 * tan_30 = 1 := 
by
  have h1 : tan_30 = Real.sqrt 3 / 3 := sorry
  have h2 : tan_15 = 2 - Real.sqrt 3 := sorry
  sorry

end tan_addition_identity_l132_132549


namespace exists_unique_decomposition_l132_132079

theorem exists_unique_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ),
    (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n, z n ≥ z (n-1)) ∧
    (∀ n, y n * (z n - z (n-1)) = 0) ∧
    z 0 = 0 :=
sorry

end exists_unique_decomposition_l132_132079


namespace trapezium_area_l132_132054

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 285 :=
by {
  sorry
}

end trapezium_area_l132_132054


namespace angle_supplement_complement_l132_132578

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132578


namespace angle_solution_l132_132644

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132644


namespace bob_grade_is_35_l132_132106

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l132_132106


namespace angle_complement_supplement_l132_132624

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132624


namespace magnitude_fourth_power_l132_132055

open Complex

noncomputable def complex_magnitude_example : ℂ := 4 + 3 * Real.sqrt 3 * Complex.I

theorem magnitude_fourth_power :
  ‖complex_magnitude_example ^ 4‖ = 1849 := by
  sorry

end magnitude_fourth_power_l132_132055


namespace probability_of_xyz_eq_72_l132_132690

open ProbabilityTheory Finset

def dice_values : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_of_xyz_eq_72 :
  (∑ x in dice_values, ∑ y in dice_values, ∑ z in dice_values, 
   if x * y * z = 72 then 1 else 0) / (dice_values.card ^ 3) = 1 / 36 :=
by
  sorry -- Proof omitted

end probability_of_xyz_eq_72_l132_132690


namespace regular_octagon_interior_angle_deg_l132_132446

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l132_132446


namespace geometric_sequence_properties_l132_132249

-- Define the first term and common ratio
def first_term : ℕ := 12
def common_ratio : ℚ := 1/2

-- Define the formula for the n-th term of the geometric sequence
def nth_term (a : ℕ) (r : ℚ) (n : ℕ) := a * r^(n-1)

-- The 8th term in the sequence
def term_8 := nth_term first_term common_ratio 8

-- Half of the 8th term
def half_term_8 := (1/2) * term_8

-- Prove that the 8th term is 3/32 and half of the 8th term is 3/64
theorem geometric_sequence_properties : 
  (term_8 = (3/32)) ∧ (half_term_8 = (3/64)) := 
by 
  sorry

end geometric_sequence_properties_l132_132249


namespace flour_ratio_correct_l132_132933

-- Definitions based on conditions
def initial_sugar : ℕ := 13
def initial_flour : ℕ := 25
def initial_baking_soda : ℕ := 35
def initial_cocoa_powder : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa_powder : ℕ := 15

-- Calculate remaining ingredients
def remaining_flour : ℕ := initial_flour - added_flour
def remaining_sugar : ℕ := initial_sugar - added_sugar
def remaining_cocoa_powder : ℕ := initial_cocoa_powder - added_cocoa_powder

-- Calculate ratio
def total_remaining_sugar_and_cocoa : ℕ := remaining_sugar + remaining_cocoa_powder
def flour_to_sugar_cocoa_ratio : ℕ × ℕ := (remaining_flour, total_remaining_sugar_and_cocoa)

-- Proposition stating the desired ratio
theorem flour_ratio_correct : flour_to_sugar_cocoa_ratio = (17, 46) := by
  sorry

end flour_ratio_correct_l132_132933


namespace area_of_PQ_square_l132_132482

theorem area_of_PQ_square (a b c : ℕ)
  (h1 : a^2 = 144)
  (h2 : b^2 = 169)
  (h3 : a^2 + c^2 = b^2) :
  c^2 = 25 :=
by
  sorry

end area_of_PQ_square_l132_132482


namespace vans_capacity_l132_132773

def students : ℕ := 33
def adults : ℕ := 9
def vans : ℕ := 6

def total_people : ℕ := students + adults
def people_per_van : ℕ := total_people / vans

theorem vans_capacity : people_per_van = 7 := by
  sorry

end vans_capacity_l132_132773


namespace probability_of_getting_a_prize_l132_132697

theorem probability_of_getting_a_prize {prizes blanks : ℕ} (h_prizes : prizes = 10) (h_blanks : blanks = 25) :
  (prizes / (prizes + blanks) : ℚ) = 2 / 7 :=
by
  sorry

end probability_of_getting_a_prize_l132_132697


namespace parabola_focus_coordinates_l132_132889

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 4 * x^2) : (0, 1/16) = (0, 1/16) :=
by
  sorry

end parabola_focus_coordinates_l132_132889


namespace sum_of_digits_l132_132095

theorem sum_of_digits (x y z w : ℕ) 
  (hxz : z + x = 10) 
  (hyz : y + z = 9) 
  (hxw : x + w = 9) 
  (hx_ne_hy : x ≠ y)
  (hx_ne_hz : x ≠ z)
  (hx_ne_hw : x ≠ w)
  (hy_ne_hz : y ≠ z)
  (hy_ne_hw : y ≠ w)
  (hz_ne_hw : z ≠ w) :
  x + y + z + w = 19 := by
  sorry

end sum_of_digits_l132_132095


namespace solution_to_eq_l132_132151

def eq1 (x y z t : ℕ) : Prop := x * y - x * z + y * t = 182
def cond_numbers (n : ℕ) : Prop := n = 12 ∨ n = 14 ∨ n = 37 ∨ n = 65

theorem solution_to_eq 
  (x y z t : ℕ) 
  (hx : cond_numbers x) 
  (hy : cond_numbers y) 
  (hz : cond_numbers z) 
  (ht : cond_numbers t) 
  (h : eq1 x y z t) : 
  (x = 12 ∧ y = 37 ∧ z = 65 ∧ t = 14) ∨ 
  (x = 37 ∧ y = 12 ∧ z = 14 ∧ t = 65) := 
sorry

end solution_to_eq_l132_132151


namespace regular_octagon_interior_angle_l132_132379

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l132_132379


namespace interest_difference_l132_132305

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := 
  P * (1 + r)^t - P

theorem interest_difference : 
  simple_interest 500 0.20 2 - (500 * (1 + 0.20)^2 - 500) = 20 := by
  sorry

end interest_difference_l132_132305


namespace definite_integral_cos_exp_l132_132881

open Real

theorem definite_integral_cos_exp :
  ∫ x in -π..0, (cos x + exp x) = 1 - (1 / exp π) :=
by
  sorry

end definite_integral_cos_exp_l132_132881


namespace exists_consecutive_non_primes_l132_132517

theorem exists_consecutive_non_primes (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ i : ℕ, i < k → ¬Nat.Prime (n + i) := 
sorry

end exists_consecutive_non_primes_l132_132517


namespace sum_of_integers_l132_132007

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 103) 
                        (h2 : Nat.gcd a b = 1) 
                        (h3 : a < 20) 
                        (h4 : b < 20) : 
                        a + b = 19 :=
  by sorry

end sum_of_integers_l132_132007


namespace value_of_composed_operations_l132_132892

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_composed_operations : op2 (op1 15) = -15 :=
by
  sorry

end value_of_composed_operations_l132_132892


namespace angle_measure_l132_132620

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132620


namespace regular_octagon_interior_angle_l132_132377

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l132_132377


namespace compute_expression_l132_132876

theorem compute_expression : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end compute_expression_l132_132876


namespace maximum_m_value_l132_132306

variable {a b c : ℝ}

noncomputable def maximum_m : ℝ := 9/8

theorem maximum_m_value 
  (h1 : (a - b)^2 + (b - c)^2 + (c - a)^2 ≥ maximum_m * a^2)
  (h2 : b^2 - 4 * a * c ≥ 0) : 
  maximum_m = 9 / 8 :=
sorry

end maximum_m_value_l132_132306


namespace ellipse_closer_to_circle_l132_132898

variables (a : ℝ)

-- Conditions: 1 < a < 2 + sqrt 5
def in_range_a (a : ℝ) : Prop := 1 < a ∧ a < 2 + Real.sqrt 5

-- Ellipse eccentricity should decrease as 'a' increases for the given range 1 < a < 2 + sqrt 5
theorem ellipse_closer_to_circle (h_range : in_range_a a) :
    ∃ b : ℝ, b = Real.sqrt (1 - (a^2 - 1) / (4 * a)) ∧ ∀ a', (1 < a' ∧ a' < 2 + Real.sqrt 5 ∧ a < a') → b > Real.sqrt (1 - (a'^2 - 1) / (4 * a')) := 
sorry

end ellipse_closer_to_circle_l132_132898


namespace not_possible_consecutive_results_l132_132130

theorem not_possible_consecutive_results 
  (dot_counts : ℕ → ℕ)
  (h_identical_conditions : ∀ (i : ℕ), dot_counts i = 1 ∨ dot_counts i = 2 ∨ dot_counts i = 3) 
  (h_correct_dot_distribution : ∀ (i j : ℕ), (i ≠ j → dot_counts i ≠ dot_counts j))
  : ¬ (∃ (consecutive : ℕ → ℕ), 
        (∀ (k : ℕ), k < 6 → consecutive k = dot_counts (4 * k) + dot_counts (4 * k + 1) 
                         + dot_counts (4 * k + 2) + dot_counts (4 * k + 3))
        ∧ (∀ (k : ℕ), k < 5 → consecutive (k + 1) = consecutive k + 1)) := sorry

end not_possible_consecutive_results_l132_132130


namespace speed_ratio_l132_132203

theorem speed_ratio (v_A v_B : ℝ) (h : 71 / v_B = 142 / v_A) : v_A / v_B = 2 :=
by
  sorry

end speed_ratio_l132_132203


namespace no_intersection_l132_132498

-- Definitions of the sets M1 and M2 based on parameters A, B, C and integer x
def M1 (A B : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = x^2 + A * x + B}
def M2 (C : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = 2 * x^2 + 2 * x + C}

-- The statement of the theorem
theorem no_intersection (A B : ℤ) : ∃ C : ℤ, M1 A B ∩ M2 C = ∅ :=
sorry

end no_intersection_l132_132498


namespace find_integer_k_l132_132886

theorem find_integer_k (k : ℤ) : (∃ k : ℤ, (k = 6) ∨ (k = 2) ∨ (k = 0) ∨ (k = -4)) ↔ (∃ k : ℤ, (2 * k^2 + k - 8) % (k - 1) = 0) :=
by
  sorry

end find_integer_k_l132_132886


namespace calculate_expression_l132_132205

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 :=
by
  sorry

end calculate_expression_l132_132205


namespace proposition_R_is_converse_negation_of_P_l132_132244

variables (x y : ℝ)

def P : Prop := x + y = 0 → x = -y
def Q : Prop := ¬(x + y = 0) → x ≠ -y
def R : Prop := x ≠ -y → ¬(x + y = 0)

theorem proposition_R_is_converse_negation_of_P : R x y ↔ ¬P x y :=
by sorry

end proposition_R_is_converse_negation_of_P_l132_132244


namespace consecutive_nums_sum_as_product_l132_132698

theorem consecutive_nums_sum_as_product {n : ℕ} (h : 100 < n) :
  ∃ (a b c : ℕ), (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (2 ≤ a) ∧ (2 ≤ b) ∧ (2 ≤ c) ∧ 
  ((n + (n+1) + (n+2) = a * b * c) ∨ ((n+1) + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end consecutive_nums_sum_as_product_l132_132698


namespace LCM_of_36_and_220_l132_132954

theorem LCM_of_36_and_220:
  let A := 36
  let B := 220
  let productAB := A * B
  let HCF := 4
  let LCM := (A * B) / HCF
  LCM = 1980 := 
by
  sorry

end LCM_of_36_and_220_l132_132954


namespace salary_january_l132_132281

variable (J F M A May : ℝ)

theorem salary_january 
  (h1 : J + F + M + A = 32000) 
  (h2 : F + M + A + May = 33600) 
  (h3 : May = 6500) : 
  J = 4900 := 
by {
 sorry 
}

end salary_january_l132_132281


namespace parabola_vertex_correct_l132_132334

noncomputable def parabola_vertex (p q : ℝ) : ℝ × ℝ :=
  let a := -1
  let b := p
  let c := q
  let x_vertex := -b / (2 * a)
  let y_vertex := a * x_vertex^2 + b * x_vertex + c
  (x_vertex, y_vertex)

theorem parabola_vertex_correct (p q : ℝ) :
  (parabola_vertex 2 24 = (1, 25)) :=
  sorry

end parabola_vertex_correct_l132_132334


namespace bankers_gain_is_126_l132_132146

-- Define the given conditions
def present_worth : ℝ := 600
def interest_rate : ℝ := 0.10
def time_period : ℕ := 2

-- Define the formula for compound interest to find the amount due A
def amount_due (PW : ℝ) (R : ℝ) (T : ℕ) : ℝ := PW * (1 + R) ^ T

-- Define the banker's gain as the difference between the amount due and the present worth
def bankers_gain (A : ℝ) (PW : ℝ) : ℝ := A - PW

-- The theorem to prove that the banker's gain is Rs. 126 given the conditions
theorem bankers_gain_is_126 : bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 126 := by
  sorry

end bankers_gain_is_126_l132_132146


namespace fractions_of_120_equals_2_halves_l132_132179

theorem fractions_of_120_equals_2_halves :
  (1 / 6) * (1 / 4) * (1 / 5) * 120 = 2 / 2 := 
by
  sorry

end fractions_of_120_equals_2_halves_l132_132179


namespace find_set_of_x_l132_132092

noncomputable def exponential_inequality_solution (x : ℝ) : Prop :=
  1 < Real.exp x ∧ Real.exp x < 2

theorem find_set_of_x (x : ℝ) :
  exponential_inequality_solution x ↔ 0 < x ∧ x < Real.log 2 :=
by
  sorry

end find_set_of_x_l132_132092


namespace regular_octagon_interior_angle_l132_132385

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l132_132385


namespace inequality_gt_zero_l132_132944

theorem inequality_gt_zero (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 :=
  sorry

end inequality_gt_zero_l132_132944


namespace chessboard_colorings_l132_132525

-- Definitions based on conditions
def valid_chessboard_colorings_count : ℕ :=
  2 ^ 33

-- Theorem statement with the question, conditions, and the correct answer
theorem chessboard_colorings : 
  valid_chessboard_colorings_count = 2 ^ 33 := by
  sorry

end chessboard_colorings_l132_132525


namespace inequality_proof_l132_132350

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l132_132350


namespace chef_bought_almonds_l132_132856

theorem chef_bought_almonds (total_nuts pecans : ℝ)
  (h1 : total_nuts = 0.52) (h2 : pecans = 0.38) :
  total_nuts - pecans = 0.14 :=
by
  sorry

end chef_bought_almonds_l132_132856


namespace regular_octagon_interior_angle_l132_132396

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l132_132396


namespace more_newborn_elephants_than_baby_hippos_l132_132724

-- Define the given conditions
def initial_elephants := 20
def initial_hippos := 35
def female_frac := 5 / 7
def births_per_female_hippo := 5
def total_animals_after_birth := 315

-- Calculate the required values
def female_hippos := female_frac * initial_hippos
def baby_hippos := female_hippos * births_per_female_hippo
def total_animals_before_birth := initial_elephants + initial_hippos
def total_newborns := total_animals_after_birth - total_animals_before_birth
def newborn_elephants := total_newborns - baby_hippos

-- Define the proof statement
theorem more_newborn_elephants_than_baby_hippos :
  (newborn_elephants - baby_hippos) = 10 :=
by
  sorry

end more_newborn_elephants_than_baby_hippos_l132_132724


namespace simplify_and_evaluate_l132_132142

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate_l132_132142


namespace angle_measure_l132_132567

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132567


namespace factorize_expression_l132_132053

variable (b : ℝ)

theorem factorize_expression : 2 * b^3 - 4 * b^2 + 2 * b = 2 * b * (b - 1)^2 := by
  sorry

end factorize_expression_l132_132053


namespace negation_of_diagonals_equal_l132_132845

def Rectangle : Type := sorry -- Let's assume there exists a type Rectangle
def diagonals_equal (r : Rectangle) : Prop := sorry -- Assume a function that checks if diagonals are equal

theorem negation_of_diagonals_equal :
  ¬(∀ r : Rectangle, diagonals_equal r) ↔ ∃ r : Rectangle, ¬diagonals_equal r :=
by
  sorry

end negation_of_diagonals_equal_l132_132845


namespace radar_coverage_correct_l132_132894

noncomputable def radar_coverage (r : ℝ) (width : ℝ) : ℝ × ℝ :=
  let θ := Real.pi / 7
  let distance := 40 / Real.sin θ
  let area := 1440 * Real.pi / Real.tan θ
  (distance, area)

theorem radar_coverage_correct : radar_coverage 41 18 = 
  (40 / Real.sin (Real.pi / 7), 1440 * Real.pi / Real.tan (Real.pi / 7)) :=
by
  sorry

end radar_coverage_correct_l132_132894


namespace pipe_A_fill_time_l132_132552

theorem pipe_A_fill_time (x : ℝ) (h1 : ∀ t : ℝ, t = 45) (h2 : ∀ t : ℝ, t = 18) :
  (1/x + 1/45 = 1/18) → x = 30 :=
by {
  -- Proof is omitted
  sorry
}

end pipe_A_fill_time_l132_132552


namespace correct_average_calculation_l132_132536

theorem correct_average_calculation (n : ℕ) (incorrect_avg correct_num wrong_num : ℕ) (incorrect_avg_eq : incorrect_avg = 21) (n_eq : n = 10) (correct_num_eq : correct_num = 36) (wrong_num_eq : wrong_num = 26) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 22 := by
  sorry

end correct_average_calculation_l132_132536


namespace restore_grid_values_l132_132559

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l132_132559


namespace area_increase_of_square_garden_l132_132863

theorem area_increase_of_square_garden
  (length : ℝ) (width : ℝ)
  (h_length : length = 60)
  (h_width : width = 20) :
  let perimeter := 2 * (length + width)
  let side_length := perimeter / 4
  let initial_area := length * width
  let square_area := side_length ^ 2
  square_area - initial_area = 400 :=
by
  sorry

end area_increase_of_square_garden_l132_132863


namespace smallest_hot_dog_packages_l132_132875

theorem smallest_hot_dog_packages (d : ℕ) (b : ℕ) (hd : d = 10) (hb : b = 15) :
  ∃ n : ℕ, n * d = m * b ∧ n = 3 :=
by
  sorry

end smallest_hot_dog_packages_l132_132875


namespace sqrt_fraction_sum_l132_132045

theorem sqrt_fraction_sum : 
    Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30 := 
by
  sorry

end sqrt_fraction_sum_l132_132045


namespace angle_measure_l132_132683

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132683


namespace each_interior_angle_of_regular_octagon_l132_132406

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l132_132406


namespace quadratic_roots_proof_l132_132060

theorem quadratic_roots_proof (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c = 0 ↔ (x = 1 ∨ x = -2)) → (b = 1 ∧ c = -2) :=
by
  sorry

end quadratic_roots_proof_l132_132060


namespace evaluate_power_l132_132336

theorem evaluate_power :
  (64 : ℝ) = 2^6 →
  64^(3/4 : ℝ) = 16 * Real.sqrt 2 :=
by
  intro h₁
  rw [h₁]
  sorry

end evaluate_power_l132_132336


namespace other_girl_age_l132_132000

theorem other_girl_age (x : ℕ) (h1 : 13 + x = 27) : x = 14 := by
  sorry

end other_girl_age_l132_132000


namespace smallest_possible_N_l132_132707

theorem smallest_possible_N (table_size N : ℕ) (h_table_size : table_size = 72) :
  (∀ seating : Finset ℕ, (seating.card = N) → (seating ⊆ Finset.range table_size) →
    ∃ i ∈ Finset.range table_size, (seating = ∅ ∨ ∃ j, (j ∈ seating) ∧ (i = (j + 1) % table_size ∨ i = (j - 1) % table_size)))
  → N = 18 :=
by sorry

end smallest_possible_N_l132_132707


namespace vector_magnitude_sub_l132_132906

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (theta : ℝ) (h_theta : theta = Real.pi / 3)

/-- Given vectors a and b with magnitudes 2 and 3 respectively, and the angle between them is 60 degrees,
    we need to prove that the magnitude of the vector a - b is sqrt(7). -/
theorem vector_magnitude_sub : ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end vector_magnitude_sub_l132_132906


namespace proof_problem_l132_132220

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l132_132220


namespace angle_measure_l132_132678

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132678


namespace problem_1_problem_2_l132_132761

theorem problem_1 (α : ℝ) (hα : Real.tan α = 2) :
  Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

theorem problem_2 (α : ℝ) (hα : Real.tan α = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 :=
by
  sorry

end problem_1_problem_2_l132_132761


namespace correct_comparison_l132_132692

-- Definitions of conditions based on the problem 
def hormones_participate : Prop := false 
def enzymes_produced_by_living_cells : Prop := true 
def hormones_produced_by_endocrine : Prop := true 
def endocrine_can_produce_both : Prop := true 
def synthesize_enzymes_not_nec_hormones : Prop := true 
def not_all_proteins : Prop := true 

-- Statement of the equivalence between the correct answer and its proof
theorem correct_comparison :  (¬hormones_participate ∧ enzymes_produced_by_living_cells ∧ hormones_produced_by_endocrine ∧ endocrine_can_produce_both ∧ synthesize_enzymes_not_nec_hormones ∧ not_all_proteins) → (endocrine_can_produce_both) :=
by
  sorry

end correct_comparison_l132_132692


namespace angle_measure_l132_132665

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132665


namespace trapezium_area_l132_132226

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 315 ∧ h = 15 :=
by 
  -- The proof steps would go here
  sorry

end trapezium_area_l132_132226


namespace regular_octagon_interior_angle_l132_132397

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l132_132397


namespace consecutive_cubes_perfect_square_l132_132740

theorem consecutive_cubes_perfect_square :
  ∃ n k : ℕ, (n + 1)^3 - n^3 = k^2 ∧ 
             (∀ m l : ℕ, (m + 1)^3 - m^3 = l^2 → n ≤ m) :=
sorry

end consecutive_cubes_perfect_square_l132_132740


namespace decimal_to_base8_conversion_l132_132792

-- Define the base and the number in decimal.
def base : ℕ := 8
def decimal_number : ℕ := 127

-- Define the expected representation in base 8.
def expected_base8_representation : ℕ := 177

-- Theorem stating that conversion of 127 in base 10 to base 8 yields 177
theorem decimal_to_base8_conversion : Nat.ofDigits base (Nat.digits base decimal_number) = expected_base8_representation := 
by
  sorry

end decimal_to_base8_conversion_l132_132792


namespace slower_speed_is_35_l132_132976

-- Define the given conditions
def distance : ℝ := 70 -- distance is 70 km
def speed_on_time : ℝ := 40 -- on-time average speed is 40 km/hr
def delay : ℝ := 0.25 -- delay is 15 minutes or 0.25 hours

-- This is the statement we need to prove
theorem slower_speed_is_35 :
  ∃ slower_speed : ℝ, 
    slower_speed = distance / (distance / speed_on_time + delay) ∧ slower_speed = 35 :=
by
  sorry

end slower_speed_is_35_l132_132976


namespace grid_game_winner_l132_132979

theorem grid_game_winner {m n : ℕ} :
  (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") = (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") := by
  sorry

end grid_game_winner_l132_132979


namespace piglet_balloons_l132_132754

theorem piglet_balloons (n w o total_balloons: ℕ) (H1: w = 2 * n) (H2: o = 4 * n) (H3: n + w + o = total_balloons) (H4: total_balloons = 44) : n - (7 * n - total_balloons) = 2 :=
by
  sorry

end piglet_balloons_l132_132754


namespace right_triangle_area_hypotenuse_30_deg_l132_132038

theorem right_triangle_area_hypotenuse_30_deg
  (h : Real)
  (θ : Real)
  (A : Real)
  (H1 : θ = 30)
  (H2 : h = 12)
  : A = 18 * Real.sqrt 3 := by
  sorry

end right_triangle_area_hypotenuse_30_deg_l132_132038


namespace solve_frac_eq_l132_132738

theorem solve_frac_eq (x : ℝ) (h : 3 - 5 / x + 2 / (x^2) = 0) : 
  ∃ y : ℝ, (y = 3 / x ∧ (y = 9 / 2 ∨ y = 3)) :=
sorry

end solve_frac_eq_l132_132738


namespace tangent_intersection_x_l132_132985

theorem tangent_intersection_x :
  ∃ x : ℝ, 
    0 < x ∧ (∃ r1 r2 : ℝ, 
     (r1 = 3) ∧ 
     (r2 = 8) ∧ 
     (0, 0) = (0, 0) ∧ 
     (18, 0) = (18, 0) ∧
     (∀ t : ℝ, t > 0 → t = x / (18 - x) → t = r1 / r2) ∧ 
      x = 54 / 11) := 
sorry

end tangent_intersection_x_l132_132985


namespace both_selected_probability_l132_132966

-- Define the probabilities of selection for X and Y
def P_X := 1 / 7
def P_Y := 2 / 9

-- Statement to prove that the probability of both being selected is 2 / 63
theorem both_selected_probability :
  (P_X * P_Y) = (2 / 63) :=
by
  -- Proof skipped
  sorry

end both_selected_probability_l132_132966


namespace gcd_of_72_90_120_l132_132972

theorem gcd_of_72_90_120 : Nat.gcd (Nat.gcd 72 90) 120 = 6 := 
by 
  have h1 : 72 = 2^3 * 3^2 := by norm_num
  have h2 : 90 = 2 * 3^2 * 5 := by norm_num
  have h3 : 120 = 2^3 * 3 * 5 := by norm_num
  sorry

end gcd_of_72_90_120_l132_132972


namespace fourth_machine_works_for_12_hours_daily_l132_132918

noncomputable def hours_fourth_machine_works (m1_hours m1_production_rate: ℕ) (m2_hours m2_production_rate: ℕ) (price_per_kg: ℕ) (total_earning: ℕ) :=
  let m1_total_production := m1_hours * m1_production_rate
  let m1_total_output := 3 * m1_total_production
  let m1_revenue := m1_total_output * price_per_kg
  let remaining_revenue := total_earning - m1_revenue
  let m2_total_production := remaining_revenue / price_per_kg
  m2_total_production / m2_production_rate

theorem fourth_machine_works_for_12_hours_daily : hours_fourth_machine_works 23 2 (sorry) (sorry) 50 8100 = 12 := by
  sorry

end fourth_machine_works_for_12_hours_daily_l132_132918


namespace total_hours_worked_l132_132719

theorem total_hours_worked (Amber_hours : ℕ) (h_Amber : Amber_hours = 12) 
  (Armand_hours : ℕ) (h_Armand : Armand_hours = Amber_hours / 3)
  (Ella_hours : ℕ) (h_Ella : Ella_hours = Amber_hours * 2) : 
  Amber_hours + Armand_hours + Ella_hours = 40 :=
by
  rw [h_Amber, h_Armand, h_Ella]
  norm_num
  sorry

end total_hours_worked_l132_132719


namespace total_age_of_wines_l132_132820

theorem total_age_of_wines (age_carlo_rosi : ℕ) (age_franzia : ℕ) (age_twin_valley : ℕ) 
    (h1 : age_carlo_rosi = 40) (h2 : age_franzia = 3 * age_carlo_rosi) (h3 : age_carlo_rosi = 4 * age_twin_valley) : 
    age_franzia + age_carlo_rosi + age_twin_valley = 170 := 
by
    sorry

end total_age_of_wines_l132_132820


namespace collision_probability_l132_132154

def time := ℝ  -- representing time in hours

-- Define the time intervals as mentioned in the conditions
noncomputable def trainA_arrival_time : set time := {t | 9 ≤ t ∧ t ≤ 14.5}
noncomputable def trainB_arrival_time : set time := {t | 9.5 ≤ t ∧ t ≤ 12.5}
noncomputable def intersection_clear_time := 45 / 60 -- in hours

-- Define the event space
def is_collision (a b : time) : Prop :=
  abs (a - b) < intersection_clear_time

-- Define the probability function
noncomputable def uniform_prob (s : set time) : ℝ := sorry

-- Conditions:
-- 1. Train A arrives between 9:00 AM and 2:30 PM.
-- 2. Train B arrives between 9:30 AM and 12:30 PM.
-- 3. Each train takes 45 minutes to clear the intersection.
def prob_collision : ℝ :=
  (uniform_prob trainA_arrival_time) *
  (∫ a in trainA_arrival_time, ∫ b in trainB_arrival_time, indicator is_collision a b)

theorem collision_probability : prob_collision = 13 / 48 :=
  sorry

end collision_probability_l132_132154


namespace angle_supplement_complement_l132_132604

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132604


namespace angle_measure_l132_132673

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132673


namespace length_of_train_l132_132718

-- We state the problem as a theorem in Lean
theorem length_of_train (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ)
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 32)
  (h_train_speed_kmh : train_speed_kmh = 45) :
  ∃ (train_length : ℝ), train_length = 250 := 
by
  -- We assume the necessary conditions as given
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  have total_distance : ℝ := train_speed_ms * crossing_time
  have train_length : ℝ := total_distance - bridge_length
  -- Conclude the length of the train is 250
  use train_length
  -- The proof steps are skipped using 'sorry'
  sorry

end length_of_train_l132_132718


namespace arrangement_count_l132_132729

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end arrangement_count_l132_132729


namespace angle_solution_l132_132645

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132645


namespace find_a_if_parallel_l132_132905

-- Definitions of the vectors and the scalar a
def vector_m : ℝ × ℝ := (2, 1)
def vector_n (a : ℝ) : ℝ × ℝ := (4, a)

-- Condition for parallel vectors
def are_parallel (m n : ℝ × ℝ) : Prop :=
  m.1 / n.1 = m.2 / n.2

-- Lean 4 statement
theorem find_a_if_parallel (a : ℝ) (h : are_parallel vector_m (vector_n a)) : a = 2 :=
by
  sorry

end find_a_if_parallel_l132_132905


namespace total_distance_is_correct_l132_132116

def Jonathan_d : Real := 7.5

def Mercedes_d (J : Real) : Real := 2 * J

def Davonte_d (M : Real) : Real := M + 2

theorem total_distance_is_correct : 
  let J := Jonathan_d
  let M := Mercedes_d J
  let D := Davonte_d M
  M + D = 32 :=
by
  sorry

end total_distance_is_correct_l132_132116


namespace find_missing_fraction_l132_132010

def f1 := 1/3
def f2 := 1/2
def f3 := 1/5
def f4 := 1/4
def f5 := -9/20
def f6 := -9/20
def total_sum := 45/100
def missing_fraction := 1/15

theorem find_missing_fraction : f1 + f2 + f3 + f4 + f5 + f6 + missing_fraction = total_sum :=
by
  sorry

end find_missing_fraction_l132_132010


namespace substract_repeating_decimal_l132_132223

noncomputable def repeating_decimal : ℝ := 1 / 3

theorem substract_repeating_decimal (x : ℝ) (h : x = repeating_decimal) : 
  1 - x = 2 / 3 :=
by
  sorry

end substract_repeating_decimal_l132_132223


namespace regular_octagon_interior_angle_l132_132387

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l132_132387


namespace value_of_x_plus_4_l132_132094

theorem value_of_x_plus_4 (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 :=
by
  sorry

end value_of_x_plus_4_l132_132094


namespace second_discount_percentage_l132_132828

/-- 
  Given:
  - The listed price of Rs. 560.
  - The final sale price after successive discounts of 20% and another discount is Rs. 313.6.
  Prove:
  - The second discount percentage is 30%.
-/
theorem second_discount_percentage (list_price final_price : ℝ) (first_discount_percentage : ℝ) : 
  list_price = 560 → 
  final_price = 313.6 → 
  first_discount_percentage = 20 → 
  ∃ (second_discount_percentage : ℝ), second_discount_percentage = 30 :=
by
  sorry

end second_discount_percentage_l132_132828


namespace arithmetic_mean_16_24_40_32_l132_132019

theorem arithmetic_mean_16_24_40_32 : (16 + 24 + 40 + 32) / 4 = 28 :=
by
  sorry

end arithmetic_mean_16_24_40_32_l132_132019


namespace angle_supplement_complement_l132_132581

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132581


namespace steve_needs_28_feet_of_wood_l132_132530

theorem steve_needs_28_feet_of_wood :
  (6 * 4) + (2 * 2) = 28 := by
  sorry

end steve_needs_28_feet_of_wood_l132_132530


namespace area_percent_less_l132_132191

theorem area_percent_less 
  (r1 r2 : ℝ)
  (h : r1 / r2 = 3 / 10) 
  : 1 - (π * (r1:ℝ)^2 / (π * (r2:ℝ)^2)) = 0.91 := 
by 
  sorry

end area_percent_less_l132_132191


namespace inequality_holds_l132_132980

noncomputable def positive_real_numbers := { x : ℝ // 0 < x }

theorem inequality_holds (a b c : positive_real_numbers) (h : (a.val * b.val + b.val * c.val + c.val * a.val) = 1) :
    (a.val / b.val + b.val / c.val + c.val / a.val) ≥ (a.val^2 + b.val^2 + c.val^2 + 2) :=
by
  sorry

end inequality_holds_l132_132980


namespace greatest_perimeter_of_strips_l132_132134

theorem greatest_perimeter_of_strips :
  let base := 10
  let height := 12
  let half_base := base / 2
  let right_triangle_area := (base / 2 * height) / 2
  let number_of_pieces := 10
  let sub_area := right_triangle_area / (number_of_pieces / 2)
  let h1 := (2 * sub_area) / half_base
  let hypotenuse := Real.sqrt (h1^2 + (half_base / 2)^2)
  let perimeter := half_base + 2 * hypotenuse
  perimeter = 11.934 :=
by
  sorry

end greatest_perimeter_of_strips_l132_132134


namespace noncongruent_integer_tris_l132_132907

theorem noncongruent_integer_tris : 
  ∃ S : Finset (ℕ × ℕ × ℕ), S.card = 18 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ S → 
      (a + b > c ∧ a + b + c < 20 ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2) :=
sorry

end noncongruent_integer_tris_l132_132907


namespace max_mogs_l132_132872

theorem max_mogs : ∃ x y z : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 3 * x + 4 * y + 8 * z = 100 ∧ z = 10 :=
by
  sorry

end max_mogs_l132_132872


namespace tagged_fish_in_second_catch_l132_132247

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ := 3200) 
  (initial_tagged : ℕ := 80) 
  (second_catch : ℕ := 80) 
  (T : ℕ) 
  (h : (T : ℚ) / second_catch = initial_tagged / total_fish) :
  T = 2 :=
by 
  sorry

end tagged_fish_in_second_catch_l132_132247


namespace impossible_to_place_50_pieces_on_torus_grid_l132_132207

theorem impossible_to_place_50_pieces_on_torus_grid :
  ¬ (∃ (a b c x y z : ℕ),
    a + b + c = 50 ∧
    2 * a ≤ x ∧ x ≤ 2 * b ∧
    2 * b ≤ y ∧ y ≤ 2 * c ∧
    2 * c ≤ z ∧ z ≤ 2 * a) :=
by
  sorry

end impossible_to_place_50_pieces_on_torus_grid_l132_132207


namespace find_a_values_l132_132748

theorem find_a_values (a n : ℕ) (h1 : 7 * a * n - 3 * n = 2020) :
    a = 68 ∨ a = 289 := sorry

end find_a_values_l132_132748


namespace perimeter_of_square_C_l132_132524

theorem perimeter_of_square_C (a b : ℝ) 
  (hA : 4 * a = 16) 
  (hB : 4 * b = 32) : 
  4 * (a + b) = 48 := by
  sorry

end perimeter_of_square_C_l132_132524


namespace angle_measure_l132_132596

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132596


namespace regular_octagon_interior_angle_l132_132460

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l132_132460


namespace slope_of_line_l132_132295

theorem slope_of_line : ∀ (x y : ℝ), 4 * y = -6 * x + 12 → ∃ m b : ℝ, y = m * x + b ∧ m = -3 / 2 :=
by 
sorry

end slope_of_line_l132_132295


namespace circle_area_irrational_if_rational_diameter_l132_132474

noncomputable def pi : ℝ := Real.pi

theorem circle_area_irrational_if_rational_diameter (d : ℚ) :
  ¬ ∃ (A : ℝ), A = pi * (d / 2)^2 ∧ (∃ (q : ℚ), A = q) :=
by
  sorry

end circle_area_irrational_if_rational_diameter_l132_132474


namespace regular_octagon_angle_l132_132449

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l132_132449


namespace angle_complement_supplement_l132_132629

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132629


namespace find_n_l132_132887

theorem find_n (n : ℕ) (h : n ≥ 2) : 
  (∀ (i j : ℕ), 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔ ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by
  sorry

end find_n_l132_132887


namespace quadratic_points_order_l132_132231

theorem quadratic_points_order (y1 y2 y3 : ℝ) :
  (y1 = -2 * (1:ℝ) ^ 2 + 4) →
  (y2 = -2 * (2:ℝ) ^ 2 + 4) →
  (y3 = -2 * (-3:ℝ) ^ 2 + 4) →
  y1 > y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end quadratic_points_order_l132_132231


namespace coin_difference_is_eight_l132_132135

theorem coin_difference_is_eight :
  let min_coins := 2  -- two 25-cent coins
  let max_coins := 10 -- ten 5-cent coins
  max_coins - min_coins = 8 :=
by
  sorry

end coin_difference_is_eight_l132_132135


namespace tan_390_correct_l132_132700

-- We assume basic trigonometric functions and their properties
noncomputable def tan_390_equals_sqrt3_div3 : Prop :=
  Real.tan (390 * Real.pi / 180) = Real.sqrt 3 / 3

theorem tan_390_correct : tan_390_equals_sqrt3_div3 :=
  by
  -- Proof is omitted
  sorry

end tan_390_correct_l132_132700


namespace ozverin_concentration_after_5_times_l132_132551

noncomputable def ozverin_concentration (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ) : ℝ :=
  C₀ * (1 - v / V) ^ n

theorem ozverin_concentration_after_5_times :
  ∀ (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ), V = 0.5 → C₀ = 0.4 → v = 50 → n = 5 →
  ozverin_concentration V C₀ v n = 0.236196 :=
by
  intros V C₀ v n hV hC₀ hv hn
  rw [hV, hC₀, hv, hn]
  simp only [ozverin_concentration]
  norm_num
  sorry

end ozverin_concentration_after_5_times_l132_132551


namespace tangent_line_at_x1_f_nonnegative_iff_l132_132769

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x-1) * Real.log x - m * (x+1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + (x-1) / x - m

theorem tangent_line_at_x1 (m : ℝ) (h : m = 1) :
  ∀ x y : ℝ, f x 1 = y → (x = 1) → x + y + 1 = 0 :=
sorry

theorem f_nonnegative_iff (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end tangent_line_at_x1_f_nonnegative_iff_l132_132769


namespace minimum_value_2_l132_132780

noncomputable def minimum_value (x y : ℝ) : ℝ := 2 * x + 3 * y ^ 2

theorem minimum_value_2 (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2 * y = 1) : minimum_value x y = 2 :=
sorry

end minimum_value_2_l132_132780


namespace selina_sells_5_shirts_l132_132948

theorem selina_sells_5_shirts
    (pants_price shorts_price shirts_price : ℕ)
    (pants_sold shorts_sold shirts_bought remaining_money : ℕ)
    (total_earnings : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  pants_sold = 3 →
  shorts_sold = 5 →
  shirts_bought = 2 →
  remaining_money = 30 →
  total_earnings = remaining_money + shirts_bought * 10 →
  total_earnings = 50 →
  total_earnings = pants_sold * pants_price + shorts_sold * shorts_price + 20 →
  20 / shirts_price = 5 :=
by
  sorry

end selina_sells_5_shirts_l132_132948


namespace remaining_blocks_correct_l132_132945

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks used
def used_blocks : ℕ := 36

-- Define the remaining blocks equation
def remaining_blocks : ℕ := initial_blocks - used_blocks

-- Prove that the number of remaining blocks is 23
theorem remaining_blocks_correct : remaining_blocks = 23 := by
  sorry

end remaining_blocks_correct_l132_132945


namespace problem_solution_l132_132080

def seq (a : ℕ → ℝ) (a1 : a 1 = 0) (rec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : Prop :=
  a 6 = Real.sqrt 3

theorem problem_solution (a : ℕ → ℝ) (h1 : a 1 = 0) (hrec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : 
  seq a h1 hrec :=
by
  sorry

end problem_solution_l132_132080


namespace confidence_level_for_relationship_l132_132476

-- Define the problem conditions and the target question.
def chi_squared_value : ℝ := 8.654
def critical_value : ℝ := 6.635
def confidence_level : ℝ := 99

theorem confidence_level_for_relationship (h : chi_squared_value > critical_value) : confidence_level = 99 :=
sorry

end confidence_level_for_relationship_l132_132476


namespace div_remainder_l132_132988

theorem div_remainder (B x : ℕ) (h1 : B = 301) (h2 : B % 7 = 0) : x = 3 :=
  sorry

end div_remainder_l132_132988


namespace total_valid_votes_l132_132190

theorem total_valid_votes (V : ℝ)
  (h1 : ∃ c1 c2 : ℝ, c1 = 0.70 * V ∧ c2 = 0.30 * V)
  (h2 : ∀ c1 c2, c1 - c2 = 182) : V = 455 :=
sorry

end total_valid_votes_l132_132190


namespace geometric_sequence_problem_l132_132791

variable {a : ℕ → ℝ}

theorem geometric_sequence_problem (h1 : a 5 * a 7 = 2) (h2 : a 2 + a 10 = 3) : 
  (a 12 / a 4 = 1 / 2) ∨ (a 12 / a 4 = 2) := 
sorry

end geometric_sequence_problem_l132_132791


namespace find_d_l132_132260

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l132_132260


namespace slope_of_line_determined_by_solutions_l132_132838

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l132_132838


namespace angle_measure_l132_132615

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132615


namespace radius_I_l132_132015

noncomputable def radius_O1 : ℝ := 3
noncomputable def radius_O2 : ℝ := 3
noncomputable def radius_O3 : ℝ := 3

axiom O1_O2_tangent : ∀ (O1 O2 : ℝ), O1 + O2 = radius_O1 + radius_O2
axiom O2_O3_tangent : ∀ (O2 O3 : ℝ), O2 + O3 = radius_O2 + radius_O3
axiom O3_O1_tangent : ∀ (O3 O1 : ℝ), O3 + O1 = radius_O3 + radius_O1

axiom I_O1_tangent : ∀ (I O1 : ℝ), I + O1 = radius_O1 + I
axiom I_O2_tangent : ∀ (I O2 : ℝ), I + O2 = radius_O2 + I
axiom I_O3_tangent : ∀ (I O3 : ℝ), I + O3 = radius_O3 + I

theorem radius_I : ∀ (I : ℝ), I = radius_O1 :=
by
  sorry

end radius_I_l132_132015


namespace part1_part2_part3_max_part3_min_l132_132763

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom f_add (x y : ℝ) : f (x + y) = f x + f y
axiom f_neg (x : ℝ) : x > 0 → f x < 0
axiom f_one : f 1 = -2

-- Prove that f(0) = 0
theorem part1 : f 0 = 0 := sorry

-- Prove that f(x) is an odd function
theorem part2 : ∀ x : ℝ, f (-x) = -f x := sorry

-- Prove the maximum and minimum values of f(x) on [-3,3]
theorem part3_max : f (-3) = 6 := sorry
theorem part3_min : f 3 = -6 := sorry

end part1_part2_part3_max_part3_min_l132_132763


namespace smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l132_132462

noncomputable def smaller_angle_at_715 : ℝ :=
  let hour_position := 7 * 30 + 30 / 4
  let minute_position := 15 * (360 / 60)
  let angle_between := abs (hour_position - minute_position)
  if angle_between > 180 then 360 - angle_between else angle_between

theorem smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m :
  smaller_angle_at_715 = 127.5 := 
sorry

end smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l132_132462


namespace exists_N_binary_representation_l132_132928

theorem exists_N_binary_representation (n p : ℕ) (h_composite : ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0) (h_proper_divisor : p > 0 ∧ p < n ∧ n % p = 0) :
  ∃ N : ℕ, ((1 + 2^p + 2^(n-p)) * N) % 2^n = 1 % 2^n :=
by
  sorry

end exists_N_binary_representation_l132_132928


namespace probability_of_getting_all_books_l132_132967

open Classical

def total_possible_scenarios : ℕ := 8
def favorable_scenarios : ℕ := 2

theorem probability_of_getting_all_books :
  (favorable_scenarios : ℚ) / total_possible_scenarios = 1 / 4 := 
  sorry

end probability_of_getting_all_books_l132_132967


namespace number_of_divisors_of_n_l132_132358

theorem number_of_divisors_of_n :
  let n : ℕ := (7^3) * (11^2) * (13^4)
  ∃ d : ℕ, d = 60 ∧ ∀ m : ℕ, m ∣ n ↔ ∃ l₁ l₂ l₃ : ℕ, l₁ ≤ 3 ∧ l₂ ≤ 2 ∧ l₃ ≤ 4 ∧ m = 7^l₁ * 11^l₂ * 13^l₃ := 
by
  sorry

end number_of_divisors_of_n_l132_132358


namespace angle_complement_supplement_l132_132631

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132631


namespace divisibility_condition_l132_132024

theorem divisibility_condition
  (a p q : ℕ) (hpq : p ≤ q) (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) :
  (p ∣ a^p ∨ p ∣ a^q) → (p ∣ a^p ∧ p ∣ a^q) :=
by
  sorry

end divisibility_condition_l132_132024


namespace evaluate_expression_l132_132214

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l132_132214


namespace find_initial_quantities_l132_132289

/-- 
Given:
- x + y = 92
- (2/5) * x + (1/4) * y = 26

Prove:
- x = 20
- y = 72
-/
theorem find_initial_quantities (x y : ℝ) (h1 : x + y = 92) (h2 : (2/5) * x + (1/4) * y = 26) :
  x = 20 ∧ y = 72 :=
sorry

end find_initial_quantities_l132_132289


namespace interior_angle_regular_octagon_l132_132409

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l132_132409


namespace angle_supplement_complement_l132_132601

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l132_132601


namespace emily_strawberry_harvest_l132_132213

-- Define the dimensions of the garden
def garden_length : ℕ := 10
def garden_width : ℕ := 7

-- Define the planting density
def plants_per_sqft : ℕ := 3

-- Define the yield per plant
def strawberries_per_plant : ℕ := 12

-- Define the expected number of strawberries
def expected_strawberries : ℕ := 2520

-- Theorem statement to prove the total number of strawberries
theorem emily_strawberry_harvest :
  garden_length * garden_width * plants_per_sqft * strawberries_per_plant = expected_strawberries :=
by
  -- Proof goes here (for now, we use sorry to indicate the proof is omitted)
  sorry

end emily_strawberry_harvest_l132_132213


namespace regular_octagon_interior_angle_eq_135_l132_132393

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l132_132393


namespace angle_supplement_complement_l132_132582

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132582


namespace combined_population_of_New_England_and_New_York_l132_132126

noncomputable def population_of_New_England : ℕ := 2100000

noncomputable def population_of_New_York := (2/3 : ℚ) * population_of_New_England

theorem combined_population_of_New_England_and_New_York :
  population_of_New_England + population_of_New_York = 3500000 :=
by sorry

end combined_population_of_New_England_and_New_York_l132_132126


namespace paintings_after_30_days_l132_132939

theorem paintings_after_30_days (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ)
    (h1 : paintings_per_day = 2)
    (h2 : initial_paintings = 20)
    (h3 : days = 30) :
    initial_paintings + paintings_per_day * days = 80 := by
  sorry

end paintings_after_30_days_l132_132939


namespace xy_equation_solution_l132_132242

theorem xy_equation_solution (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 11980 / 121 :=
by
  sorry

end xy_equation_solution_l132_132242


namespace interior_angle_regular_octagon_l132_132424

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l132_132424


namespace weng_total_earnings_l132_132178

noncomputable def weng_earnings_usd : ℝ :=
  let usd_per_hr_job1 : ℝ := 12
  let eur_per_hr_job2 : ℝ := 13
  let gbp_per_hr_job3 : ℝ := 9
  let hr_job1 : ℝ := 2 + 15 / 60
  let hr_job2 : ℝ := 1 + 40 / 60
  let hr_job3 : ℝ := 3 + 10 / 60
  let usd_to_eur : ℝ := 0.85
  let usd_to_gbp : ℝ := 0.76
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.32
  let earnings_job1 : ℝ := usd_per_hr_job1 * hr_job1
  let earnings_job2_eur : ℝ := eur_per_hr_job2 * hr_job2
  let earnings_job2_usd : ℝ := earnings_job2_eur * eur_to_usd
  let earnings_job3_gbp : ℝ := gbp_per_hr_job3 * hr_job3
  let earnings_job3_usd : ℝ := earnings_job3_gbp * gbp_to_usd
  earnings_job1 + earnings_job2_usd + earnings_job3_usd

theorem weng_total_earnings : weng_earnings_usd = 90.19 :=
by
  sorry

end weng_total_earnings_l132_132178


namespace scientific_notation_l132_132323

theorem scientific_notation :
  56.9 * 10^9 = 5.69 * 10^(10 - 1) :=
by
  sorry

end scientific_notation_l132_132323


namespace quadratic_function_choice_l132_132204

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Define the given equations as functions
def f_A (x : ℝ) : ℝ := 3 * x
def f_B (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def f_C (x : ℝ) : ℝ := (x - 1)^2
def f_D (x : ℝ) : ℝ := 2

-- State the Lean theorem statement
theorem quadratic_function_choice : is_quadratic f_C := sorry

end quadratic_function_choice_l132_132204


namespace number_multiplied_by_any_integer_results_in_itself_l132_132297

theorem number_multiplied_by_any_integer_results_in_itself (N : ℤ) (h : ∀ (x : ℤ), N * x = N) : N = 0 :=
  sorry

end number_multiplied_by_any_integer_results_in_itself_l132_132297


namespace gcd_102_238_l132_132827

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  -- Given conditions as part of proof structure
  have h1 : 238 = 102 * 2 + 34 := by rfl
  have h2 : 102 = 34 * 3 := by rfl
  sorry

end gcd_102_238_l132_132827


namespace slope_of_line_determined_by_solutions_l132_132837

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l132_132837


namespace angle_complement_supplement_l132_132622

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l132_132622


namespace coprime_exponents_iff_l132_132286

theorem coprime_exponents_iff (p q : ℕ) : 
  Nat.gcd (2^p - 1) (2^q - 1) = 1 ↔ Nat.gcd p q = 1 :=
by 
  sorry

end coprime_exponents_iff_l132_132286


namespace percent_notebooks_staplers_clips_l132_132145

def percent_not_special (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) : ℝ :=
  100 - (n + s + c)

theorem percent_notebooks_staplers_clips (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) :
  percent_not_special n s c h_n h_s h_c = 25 :=
by
  unfold percent_not_special
  rw [h_n, h_s, h_c]
  norm_num

end percent_notebooks_staplers_clips_l132_132145


namespace sally_seashells_l132_132016

theorem sally_seashells (T S: ℕ) (hT : T = 37) (h_total : T + S = 50) : S = 13 := by
  -- Skip the proof
  sorry

end sally_seashells_l132_132016


namespace circle_equation_l132_132340

theorem circle_equation 
  (P : ℝ × ℝ)
  (h1 : ∀ a : ℝ, (1 - a) * 2 + (P.snd) + 2 * a - 1 = 0)
  (h2 : P = (2, -1)) :
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end circle_equation_l132_132340


namespace remainder_when_divided_by_95_l132_132020

theorem remainder_when_divided_by_95 (x : ℤ) (h1 : x % 19 = 12) :
  x % 95 = 12 := 
sorry

end remainder_when_divided_by_95_l132_132020


namespace problem_1_problem_2_l132_132749

theorem problem_1 (p x : ℝ) (h1 : |p| ≤ 2) (h2 : x^2 + p*x + 1 > 2*x + p) : x < -1 ∨ x > 3 :=
sorry

theorem problem_2 (p x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) (h3 : x^2 + p*x + 1 > 2*x + p) : p > -1 :=
sorry

end problem_1_problem_2_l132_132749


namespace correct_grid_l132_132557

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l132_132557


namespace philips_painting_total_l132_132938

def total_paintings_after_days (daily_paintings : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + daily_paintings * days

theorem philips_painting_total (daily_paintings initial_paintings days : ℕ) 
  (h1 : daily_paintings = 2) (h2 : initial_paintings = 20) (h3 : days = 30) : 
  total_paintings_after_days daily_paintings initial_paintings days = 80 := 
by
  sorry

end philips_painting_total_l132_132938


namespace multiplication_is_valid_l132_132531

-- Define that the three-digit number n = 306
def three_digit_number := 306

-- The multiplication by 1995 should result in the defined product
def valid_multiplication (n : ℕ) := 1995 * n

theorem multiplication_is_valid : valid_multiplication three_digit_number = 1995 * 306 := by
  -- Since we only need the statement, we use sorry here
  sorry

end multiplication_is_valid_l132_132531


namespace donations_received_l132_132865

def profit : Nat := 960
def half_profit: Nat := profit / 2
def goal: Nat := 610
def extra: Nat := 180
def total_needed: Nat := goal + extra
def donations: Nat := total_needed - half_profit

theorem donations_received :
  donations = 310 := by
  -- Proof omitted
  sorry

end donations_received_l132_132865


namespace angle_measure_l132_132573

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l132_132573


namespace angle_solution_l132_132649

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l132_132649


namespace odd_three_mn_l132_132533

theorem odd_three_mn (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) : (3 * m * n) % 2 = 1 :=
sorry

end odd_three_mn_l132_132533


namespace rope_touching_tower_length_l132_132986

-- Definitions according to conditions
def tower_radius : ℝ := 10
def rope_length : ℝ := 30
def attachment_height : ℝ := 6
def horizontal_distance : ℝ := 6

-- The goal is to prove the length of the rope touching the tower
theorem rope_touching_tower_length :
  (let R := tower_radius in
   let L := rope_length in
   let h := attachment_height in
   let x := horizontal_distance in
   let effective_horizontal_length := R + x in
   let y := sqrt (effective_horizontal_length^2 - h^2) in
   let y_half := y / 2 in
   let θ := 2 * real.arccos (R / y_half) in
   R * θ) ≈ 12.28 :=
by
  let R := tower_radius
  let L := rope_length
  let h := attachment_height
  let x := horizontal_distance
  let effective_horizontal_length := R + x
  let y := real.sqrt (effective_horizontal_length^2 - h^2)
  let y_half := y / 2
  let θ := 2 * real.arccos (R / y_half)
  have hR : R = tower_radius := rfl
  have hL : L = rope_length := rfl
  have hh : h = attachment_height := rfl
  have hx : x = horizontal_distance := rfl
  have hex : effective_horizontal_length = R + x := rfl
  have hy : y = real.sqrt (effective_horizontal_length^2 - h^2) := rfl
  have hyh : y_half = y / 2 := rfl
  have hθ : θ = 2 * real.arccos (R / y_half) := rfl
  have := calc
    _ = tower_radius * θ : by rw [hR, hθ]
    _ = 10 * θ : by sorry -- Continue proving the actual computation
  exact sorry -- Skip the proof

end rope_touching_tower_length_l132_132986


namespace martin_berry_expenditure_l132_132124

theorem martin_berry_expenditure : 
  (let daily_consumption := 1 / 2
       berry_price := 2
       days := 30 in
   daily_consumption * days * berry_price = 30) :=
by
  sorry

end martin_berry_expenditure_l132_132124


namespace jill_water_jars_l132_132114

theorem jill_water_jars (x : ℕ) (h : x * (1 / 4 + 1 / 2 + 1) = 28) : 3 * x = 48 :=
by
  sorry

end jill_water_jars_l132_132114


namespace angle_measure_l132_132680

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132680


namespace compare_exponent_inequality_l132_132759

theorem compare_exponent_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 :=
sorry

end compare_exponent_inequality_l132_132759


namespace bus_travel_time_l132_132027

theorem bus_travel_time (D1 D2: ℝ) (T: ℝ) (h1: D1 + D2 = 250) (h2: D1 >= 0) (h3: D2 >= 0) :
  T = D1 / 40 + D2 / 60 ↔ D1 + D2 = 250 := 
by
  sorry

end bus_travel_time_l132_132027


namespace gumballs_in_packages_l132_132370

theorem gumballs_in_packages (total_gumballs : ℕ) (gumballs_per_package : ℕ) (h1 : total_gumballs = 20) (h2 : gumballs_per_package = 5) :
  total_gumballs / gumballs_per_package = 4 :=
by {
  sorry
}

end gumballs_in_packages_l132_132370


namespace Matt_overall_profit_l132_132934

def initialValue : ℕ := 8 * 6

def valueGivenAwayTrade1 : ℕ := 2 * 6
def valueReceivedTrade1 : ℕ := 3 * 2 + 9

def valueGivenAwayTrade2 : ℕ := 2 + 6
def valueReceivedTrade2 : ℕ := 2 * 5 + 8

def valueGivenAwayTrade3 : ℕ := 5 + 9
def valueReceivedTrade3 : ℕ := 3 * 3 + 10 + 1

def valueGivenAwayTrade4 : ℕ := 2 * 3 + 8
def valueReceivedTrade4 : ℕ := 2 * 7 + 4

def overallProfit : ℕ :=
  (valueReceivedTrade1 - valueGivenAwayTrade1) +
  (valueReceivedTrade2 - valueGivenAwayTrade2) +
  (valueReceivedTrade3 - valueGivenAwayTrade3) +
  (valueReceivedTrade4 - valueGivenAwayTrade4)

theorem Matt_overall_profit : overallProfit = 23 :=
by
  unfold overallProfit valueReceivedTrade1 valueGivenAwayTrade1 valueReceivedTrade2 valueGivenAwayTrade2 valueReceivedTrade3 valueGivenAwayTrade3 valueReceivedTrade4 valueGivenAwayTrade4
  linarith

end Matt_overall_profit_l132_132934


namespace total_hours_worked_l132_132721

theorem total_hours_worked (amber_hours : ℕ) (armand_hours : ℕ) (ella_hours : ℕ) 
(h_amber : amber_hours = 12) 
(h_armand : armand_hours = (1 / 3) * amber_hours) 
(h_ella : ella_hours = 2 * amber_hours) :
amber_hours + armand_hours + ella_hours = 40 :=
sorry

end total_hours_worked_l132_132721


namespace angle_supplement_complement_l132_132662

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132662


namespace find_rate_of_current_l132_132716

-- Parameters and definitions
variables (r w : Real)

-- Conditions of the problem
def original_journey := 3 * r^2 - 23 * w^2 = 0
def modified_journey := 6 * r^2 - 2 * w^2 + 40 * w = 0

-- Main theorem to prove
theorem find_rate_of_current (h1 : original_journey r w) (h2 : modified_journey r w) :
  w = 10 / 11 :=
sorry

end find_rate_of_current_l132_132716


namespace regular_octagon_interior_angle_l132_132386

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l132_132386


namespace angle_measure_l132_132616

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132616


namespace prob_xyz_eq_72_l132_132689

-- Define the set of possible outcomes for a standard six-sided die
def dice_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a predicate that checks if three dice rolls multiply to 72
def is_valid_combination (x y z : ℕ) : Prop := (x * y * z = 72)

-- Define the event space for three dice rolls
def event_space : Finset (ℕ × ℕ × ℕ) := Finset.product dice_outcomes (Finset.product dice_outcomes dice_outcomes)

-- Define the probability of an event
def probability {α : Type*} [Fintype α] (s : Finset α) (event : α → Prop) : ℚ :=
  (s.filter event).card.to_rat / s.card.to_rat

-- State the theorem
theorem prob_xyz_eq_72 : probability event_space (λ t, is_valid_combination t.1 t.2.1 t.2.2) = (7 / 216) := 
by { sorry }

end prob_xyz_eq_72_l132_132689


namespace event_with_highest_probability_l132_132691

-- Define the set of outcomes for a fair die
def outcomes := {1, 2, 3, 4, 5, 6}

-- Define the events
def is_odd (n : ℕ) := n ∈ {1, 3, 5}
def is_multiple_of_3 (n : ℕ) := n ∈ {3, 6}
def is_greater_than_5 (n : ℕ) := n ∈ {6}
def is_less_than_5 (n : ℕ) := n ∈ {1, 2, 3, 4}

-- Define probabilities
def probability (event : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (outcomes.card : ℚ)

-- Probabilities of the events
def probability_of_odd := probability {1, 3, 5}
def probability_of_multiple_of_3 := probability {3, 6}
def probability_of_greater_than_5 := probability {6}
def probability_of_less_than_5 := probability {1, 2, 3, 4}

-- The statement to prove
theorem event_with_highest_probability :
  max (max probability_of_odd probability_of_multiple_of_3)
      (max probability_of_greater_than_5 probability_of_less_than_5) =
  probability_of_less_than_5 :=
begin
  sorry
end

end event_with_highest_probability_l132_132691


namespace angle_measure_l132_132672

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132672


namespace y_intercepts_count_l132_132088

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count_l132_132088


namespace regular_octagon_interior_angle_l132_132439

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l132_132439


namespace interior_angle_regular_octagon_l132_132407

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l132_132407


namespace find_circle_center_value_x_plus_y_l132_132753

theorem find_circle_center_value_x_plus_y : 
  ∀ (x y : ℝ), (x^2 + y^2 = 4 * x - 6 * y + 9) → 
    x + y = -1 :=
by
  intros x y h
  sorry

end find_circle_center_value_x_plus_y_l132_132753


namespace uncommon_card_cost_l132_132174

/--
Tom's deck contains 19 rare cards, 11 uncommon cards, and 30 common cards.
Each rare card costs $1.
Each common card costs $0.25.
The total cost of the deck is $32.
Prove that the cost of each uncommon card is $0.50.
-/
theorem uncommon_card_cost (x : ℝ): 
  let rare_count := 19
  let uncommon_count := 11
  let common_count := 30
  let rare_cost := 1
  let common_cost := 0.25
  let total_cost := 32
  (rare_count * rare_cost) + (common_count * common_cost) + (uncommon_count * x) = total_cost 
  → x = 0.5 :=
by
  sorry

end uncommon_card_cost_l132_132174


namespace cans_per_bag_l132_132514

theorem cans_per_bag (bags_on_Saturday bags_on_Sunday total_cans : ℕ) (h_saturday : bags_on_Saturday = 3) (h_sunday : bags_on_Sunday = 4) (h_total : total_cans = 63) :
  (total_cans / (bags_on_Saturday + bags_on_Sunday) = 9) :=
by {
  sorry
}

end cans_per_bag_l132_132514


namespace John_pays_more_than_Jane_l132_132803

theorem John_pays_more_than_Jane : 
  let original_price := 24.00000000000002
  let discount_rate := 0.10
  let tip_rate := 0.15
  let discount := discount_rate * original_price
  let discounted_price := original_price - discount
  let john_tip := tip_rate * original_price
  let jane_tip := tip_rate * discounted_price
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.3600000000000003 :=
by
  sorry

end John_pays_more_than_Jane_l132_132803


namespace total_surface_area_of_resulting_structure_l132_132194

-- Definitions for the conditions
def bigCube := 12 * 12 * 12
def smallCube := 2 * 2 * 2
def totalSmallCubes := 64
def removedCubes := 7
def remainingCubes := totalSmallCubes - removedCubes
def surfaceAreaPerSmallCube := 24
def extraExposedSurfaceArea := 6
def effectiveSurfaceAreaPerSmallCube := surfaceAreaPerSmallCube + extraExposedSurfaceArea

-- Definition and the main statement of the proof problem.
def totalSurfaceArea := remainingCubes * effectiveSurfaceAreaPerSmallCube

theorem total_surface_area_of_resulting_structure : totalSurfaceArea = 1710 :=
by
  sorry

end total_surface_area_of_resulting_structure_l132_132194


namespace smallest_initial_number_l132_132499

theorem smallest_initial_number (N : ℕ) (h₁ : N ≤ 999) (h₂ : 27 * N - 240 ≥ 1000) : N = 46 :=
by {
    sorry
}

end smallest_initial_number_l132_132499


namespace angle_measure_l132_132676

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132676


namespace smallest_four_digit_divisible_by_9_l132_132181

theorem smallest_four_digit_divisible_by_9 
    (n : ℕ) 
    (h1 : 1000 ≤ n ∧ n < 10000) 
    (h2 : n % 9 = 0)
    (h3 : n % 10 % 2 = 1)
    (h4 : (n / 1000) % 2 = 1)
    (h5 : (n / 10) % 10 % 2 = 0)
    (h6 : (n / 100) % 10 % 2 = 0) :
  n = 3609 :=
sorry

end smallest_four_digit_divisible_by_9_l132_132181


namespace ap_square_sequel_l132_132793

theorem ap_square_sequel {a b c : ℝ} (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                     (h2 : 2 * (b / (c + a)) = (a / (b + c)) + (c / (a + b))) :
  (a^2 + c^2 = 2 * b^2) :=
by
  sorry

end ap_square_sequel_l132_132793


namespace cans_per_person_on_second_day_l132_132329

theorem cans_per_person_on_second_day :
  ∀ (initial_stock : ℕ) (people_first_day : ℕ) (cans_taken_first_day : ℕ)
    (restock_first_day : ℕ) (people_second_day : ℕ)
    (restock_second_day : ℕ) (total_cans_given : ℕ) (cans_per_person_second_day : ℚ),
    cans_taken_first_day = 1 →
    initial_stock = 2000 →
    people_first_day = 500 →
    restock_first_day = 1500 →
    people_second_day = 1000 →
    restock_second_day = 3000 →
    total_cans_given = 2500 →
    cans_per_person_second_day = total_cans_given / people_second_day →
    cans_per_person_second_day = 2.5 := by
  sorry

end cans_per_person_on_second_day_l132_132329


namespace cruise_liner_travelers_l132_132165

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l132_132165


namespace interior_angle_regular_octagon_l132_132420

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l132_132420


namespace angle_measure_l132_132617

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132617


namespace employees_count_l132_132957

-- Let E be the number of employees excluding the manager
def E (employees : ℕ) : ℕ := employees

-- Let T be the total salary of employees excluding the manager
def T (employees : ℕ) : ℕ := employees * 1500

-- Conditions given in the problem
def average_salary (employees : ℕ) : ℕ := T employees / E employees
def new_average_salary (employees : ℕ) : ℕ := (T employees + 22500) / (E employees + 1)

theorem employees_count : (average_salary employees = 1500) ∧ (new_average_salary employees = 2500) ∧ (manager_salary = 22500) → (E employees = 20) :=
  by sorry

end employees_count_l132_132957


namespace regular_octagon_interior_angle_l132_132441

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l132_132441


namespace func_g_neither_even_nor_odd_l132_132252

noncomputable def func_g (x : ℝ) : ℝ := (⌈x⌉ : ℝ) - (1 / 3)

theorem func_g_neither_even_nor_odd :
  (¬ ∀ x, func_g (-x) = func_g x) ∧ (¬ ∀ x, func_g (-x) = -func_g x) :=
by
  sorry

end func_g_neither_even_nor_odd_l132_132252


namespace angle_supplement_complement_l132_132584

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l132_132584


namespace paintings_after_30_days_l132_132940

theorem paintings_after_30_days (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ)
    (h1 : paintings_per_day = 2)
    (h2 : initial_paintings = 20)
    (h3 : days = 30) :
    initial_paintings + paintings_per_day * days = 80 := by
  sorry

end paintings_after_30_days_l132_132940


namespace strawberries_harvest_l132_132183

theorem strawberries_harvest (length : ℕ) (width : ℕ) 
  (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) 
  (area := length * width) (total_plants := plants_per_sqft * area) 
  (total_strawberries := strawberries_per_plant * total_plants) :
  length = 10 → width = 9 →
  plants_per_sqft = 5 → strawberries_per_plant = 8 →
  total_strawberries = 3600 := by
  sorry

end strawberries_harvest_l132_132183


namespace interior_angle_regular_octagon_l132_132434

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l132_132434


namespace range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l132_132269

-- Define the propositions
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5 * x - 6 ≥ 0

-- Question 1: Prove that for a = 1 and p ∧ q is true, the range of x is [2, 3)
theorem range_of_x_when_a_eq_1_p_and_q : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 ≤ x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬p is a sufficient but not necessary condition for ¬q, 
-- then the range of a is (1, 2)
theorem range_of_a_when_not_p_sufficient_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬p a x → ¬q x) ∧ (∃ x : ℝ, ¬(¬p a x → ¬q x)) → 1 < a ∧ a < 2 := 
by sorry

end range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l132_132269


namespace fewer_cans_collected_today_than_yesterday_l132_132693

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end fewer_cans_collected_today_than_yesterday_l132_132693


namespace sum_ps_at_10_l132_132930

def S : Finset (Vector (Fin 10) (Fin 2)) := 
  Finset.univ

def ps (s : Vector (Fin 10) (Fin 2)) : Polynomial (Fin 2) :=
  Polynomial.ofFinset (s.toFinset)

theorem sum_ps_at_10 : (∑ s in S, ps s.eval 10) = 512 := sorry

end sum_ps_at_10_l132_132930


namespace interior_angle_regular_octagon_l132_132421

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l132_132421


namespace lisa_hotdog_record_l132_132271

theorem lisa_hotdog_record
  (hotdogs_eaten : ℕ)
  (eaten_in_first_half : ℕ)
  (rate_per_minute : ℕ)
  (time_in_minutes : ℕ)
  (first_half_duration : ℕ)
  (remaining_time : ℕ) :
  eaten_in_first_half = 20 →
  rate_per_minute = 11 →
  first_half_duration = 5 →
  remaining_time = 5 →
  time_in_minutes = first_half_duration + remaining_time →
  hotdogs_eaten = eaten_in_first_half + rate_per_minute * remaining_time →
  hotdogs_eaten = 75 := by
  intros
  sorry

end lisa_hotdog_record_l132_132271


namespace industrial_lubricants_percentage_l132_132029

theorem industrial_lubricants_percentage :
  let a := 12   -- percentage for microphotonics
  let b := 24   -- percentage for home electronics
  let c := 15   -- percentage for food additives
  let d := 29   -- percentage for genetically modified microorganisms
  let angle_basic_astrophysics := 43.2 -- degrees for basic astrophysics
  let total_angle := 360              -- total degrees in a circle
  let total_budget := 100             -- total budget in percentage
  let e := (angle_basic_astrophysics / total_angle) * total_budget -- percentage for basic astrophysics
  a + b + c + d + e = 92 → total_budget - (a + b + c + d + e) = 8 :=
by
  intros
  sorry

end industrial_lubricants_percentage_l132_132029


namespace sufficient_but_not_necessary_condition_l132_132367

theorem sufficient_but_not_necessary_condition (b c : ℝ) :
  (∃ x0 : ℝ, (x0^2 + b * x0 + c) < 0) ↔ (c < 0) ∨ true :=
sorry

end sufficient_but_not_necessary_condition_l132_132367


namespace find_other_endpoint_of_diameter_l132_132030

theorem find_other_endpoint_of_diameter 
    (center endpoint : ℝ × ℝ) 
    (h_center : center = (5, -2)) 
    (h_endpoint : endpoint = (2, 3))
    : (center.1 + (center.1 - endpoint.1), center.2 + (center.2 - endpoint.2)) = (8, -7) := 
by
  sorry

end find_other_endpoint_of_diameter_l132_132030


namespace bob_grade_is_35_l132_132105

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l132_132105


namespace stadium_length_in_yards_l132_132004

theorem stadium_length_in_yards (length_in_feet : ℕ) (conversion_factor : ℕ) : ℕ :=
    length_in_feet / conversion_factor

example : stadium_length_in_yards 240 3 = 80 :=
by sorry

end stadium_length_in_yards_l132_132004


namespace angle_measure_l132_132682

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l132_132682


namespace split_payment_l132_132512

noncomputable def Rahul_work_per_day := (1 : ℝ) / 3
noncomputable def Rajesh_work_per_day := (1 : ℝ) / 2
noncomputable def Ritesh_work_per_day := (1 : ℝ) / 4

noncomputable def total_work_per_day := Rahul_work_per_day + Rajesh_work_per_day + Ritesh_work_per_day

noncomputable def Rahul_proportion := Rahul_work_per_day / total_work_per_day
noncomputable def Rajesh_proportion := Rajesh_work_per_day / total_work_per_day
noncomputable def Ritesh_proportion := Ritesh_work_per_day / total_work_per_day

noncomputable def total_payment := 510

noncomputable def Rahul_share := Rahul_proportion * total_payment
noncomputable def Rajesh_share := Rajesh_proportion * total_payment
noncomputable def Ritesh_share := Ritesh_proportion * total_payment

theorem split_payment :
  Rahul_share + Rajesh_share + Ritesh_share = total_payment :=
by
  sorry

end split_payment_l132_132512


namespace angle_supplement_complement_l132_132656

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l132_132656


namespace angle_measure_l132_132593

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l132_132593


namespace total_screens_sold_l132_132997

variable (J F M : ℕ)
variable (feb_eq_fourth_of_march : F = M / 4)
variable (feb_eq_double_of_jan : F = 2 * J)
variable (march_sales : M = 8800)

theorem total_screens_sold (J F M : ℕ)
  (feb_eq_fourth_of_march : F = M / 4)
  (feb_eq_double_of_jan : F = 2 * J)
  (march_sales : M = 8800) :
  J + F + M = 12100 :=
by
  sorry

end total_screens_sold_l132_132997


namespace ceiling_sum_l132_132219

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l132_132219


namespace find_speed_of_boat_l132_132009

noncomputable def speed_of_boat_in_still_water 
  (v : ℝ) 
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) : Prop :=
  v = 42

theorem find_speed_of_boat 
  (v : ℝ)
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) 
  (h1 : v + current_speed = distance / (time_in_minutes / 60)) : 
  speed_of_boat_in_still_water v :=
by
  sorry

end find_speed_of_boat_l132_132009


namespace walking_ring_width_l132_132867

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) :
  r₁ - r₂ = 10 :=
by
  sorry

end walking_ring_width_l132_132867


namespace value_of_m_l132_132909

theorem value_of_m (a a1 a2 a3 a4 a5 a6 m : ℝ) (x : ℝ)
  (h1 : (1 + m * x)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  (m = 1 ∨ m = -3) :=
sorry

end value_of_m_l132_132909


namespace ratio_areas_l132_132040

-- Define the perimeter P
variable (P : ℝ) (hP : P > 0)

-- Define the side lengths
noncomputable def side_length_square := P / 4
noncomputable def side_length_triangle := P / 3

-- Define the radius of the circumscribed circle for the square
noncomputable def radius_square := (P * Real.sqrt 2) / 8
-- Define the area of the circumscribed circle for the square
noncomputable def area_circle_square := Real.pi * (radius_square P)^2

-- Define the radius of the circumscribed circle for the equilateral triangle
noncomputable def radius_triangle := (P * Real.sqrt 3) / 9 
-- Define the area of the circumscribed circle for the equilateral triangle
noncomputable def area_circle_triangle := Real.pi * (radius_triangle P)^2

-- Prove the ratio of the areas is 27/32
theorem ratio_areas (P : ℝ) (hP : P > 0) : 
  (area_circle_square P / area_circle_triangle P) = (27 / 32) := by
  sorry

end ratio_areas_l132_132040


namespace each_interior_angle_of_regular_octagon_l132_132403

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l132_132403


namespace center_circle_sum_l132_132891

theorem center_circle_sum (x y : ℝ) (h : x^2 + y^2 = 4 * x + 10 * y - 12) : x + y = 7 := 
sorry

end center_circle_sum_l132_132891


namespace proof_expr_l132_132066

theorem proof_expr (a b c : ℤ) (h1 : a - b = 3) (h2 : b - c = 2) : (a - c)^2 + 3 * a + 1 - 3 * c = 41 := by {
  sorry
}

end proof_expr_l132_132066


namespace regular_octagon_interior_angle_l132_132398

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l132_132398


namespace value_of_sum_of_squares_l132_132775

theorem value_of_sum_of_squares (x y : ℝ) (h₁ : (x + y)^2 = 25) (h₂ : x * y = -6) : x^2 + y^2 = 37 :=
by
  sorry

end value_of_sum_of_squares_l132_132775


namespace regular_octagon_interior_angle_deg_l132_132445

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l132_132445


namespace correct_money_calculation_l132_132699

structure BootSale :=
(initial_money : ℕ)
(price_per_boot : ℕ)
(total_taken : ℕ)
(total_returned : ℕ)
(money_spent : ℕ)
(remaining_money_to_return : ℕ)

theorem correct_money_calculation (bs : BootSale) :
  bs.initial_money = 25 →
  bs.price_per_boot = 12 →
  bs.total_taken = 25 →
  bs.total_returned = 5 →
  bs.money_spent = 3 →
  bs.remaining_money_to_return = 2 →
  bs.total_taken - bs.total_returned + bs.money_spent = 23 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end correct_money_calculation_l132_132699


namespace average_weight_l132_132796

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l132_132796


namespace cylinder_radius_l132_132989

theorem cylinder_radius
  (r₁ r₂ : ℝ)
  (rounds₁ rounds₂ : ℕ)
  (H₁ : r₁ = 14)
  (H₂ : rounds₁ = 70)
  (H₃ : rounds₂ = 49)
  (L₁ : rounds₁ * 2 * Real.pi * r₁ = rounds₂ * 2 * Real.pi * r₂) :
  r₂ = 20 := 
sorry

end cylinder_radius_l132_132989


namespace square_area_EFGH_l132_132102

theorem square_area_EFGH (AB BP : ℝ) (h1 : AB = Real.sqrt 72) (h2 : BP = 2) (x : ℝ)
  (h3 : AB + BP = 2 * x + 2) : x^2 = 18 :=
by
  sorry

end square_area_EFGH_l132_132102


namespace ratio_pentagon_side_length_to_rectangle_width_l132_132317

def pentagon_side_length (p : ℕ) (n : ℕ) := p / n
def rectangle_width (p : ℕ) (ratio : ℕ) := p / (2 * (1 + ratio))

theorem ratio_pentagon_side_length_to_rectangle_width :
  pentagon_side_length 60 5 / rectangle_width 80 3 = (6 : ℚ) / 5 :=
by {
  sorry
}

end ratio_pentagon_side_length_to_rectangle_width_l132_132317


namespace remainder_abc_l132_132781

theorem remainder_abc (a b c : ℕ) 
  (h₀ : a < 9) (h₁ : b < 9) (h₂ : c < 9)
  (h₃ : (a + 3 * b + 2 * c) % 9 = 0)
  (h₄ : (2 * a + 2 * b + 3 * c) % 9 = 3)
  (h₅ : (3 * a + b + 2 * c) % 9 = 6) : 
  (a * b * c) % 9 = 0 := by
  sorry

end remainder_abc_l132_132781


namespace interior_angle_regular_octagon_l132_132430

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l132_132430


namespace sum_of_distinct_integers_l132_132931

noncomputable def a : ℤ := 11
noncomputable def b : ℤ := 9
noncomputable def c : ℤ := 4
noncomputable def d : ℤ := 2
noncomputable def e : ℤ := 1

def condition : Prop := (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120
def distinct_integers : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem sum_of_distinct_integers (h1 : condition) (h2 : distinct_integers) : a + b + c + d + e = 27 :=
by
  sorry

end sum_of_distinct_integers_l132_132931


namespace luis_can_make_sum_multiple_of_4_l132_132500

noncomputable def sum_of_dice (dice: List ℕ) : ℕ :=
  dice.sum 

theorem luis_can_make_sum_multiple_of_4 (d1 d2 d3: ℕ) 
  (h1: 1 ≤ d1 ∧ d1 ≤ 6) 
  (h2: 1 ≤ d2 ∧ d2 ≤ 6) 
  (h3: 1 ≤ d3 ∧ d3 ≤ 6) : 
  ∃ (dice: List ℕ), dice.length = 3 ∧ 
  sum_of_dice dice % 4 = 0 := 
by
  sorry

end luis_can_make_sum_multiple_of_4_l132_132500


namespace red_ball_prob_gt_black_ball_prob_l132_132290

theorem red_ball_prob_gt_black_ball_prob (m : ℕ) (h : 8 > m) : m ≠ 10 :=
by
  sorry

end red_ball_prob_gt_black_ball_prob_l132_132290


namespace probability_tiles_A_B_l132_132171

def is_favorable_draw_A (n : ℕ) : Prop := n < 20
def is_favorable_draw_B (m : ℕ) : Prop := (m % 2 = 1) ∨ (m > 45)

def tiles_A := Finset.range 31 -- tiles 0 to 30 (numbers 1 to 30)
def tiles_B := Finset.range' 21 51 -- tiles 21 to 50

def count_A_favorable := (tiles_A.filter is_favorable_draw_A).card
def count_B_favorable := (tiles_B.filter is_favorable_draw_B).card
def total_A := tiles_A.card
def total_B := tiles_B.card

def prob_A := count_A_favorable / total_A
def prob_B := count_B_favorable / total_B

theorem probability_tiles_A_B:
  count_A_favorable = 19 ∧
  count_B_favorable = 19 ∧
  total_A = 30 ∧
  total_B = 30 →
  prob_A * prob_B = (361 : ℚ) / (900 : ℚ) := by
  sorry

end probability_tiles_A_B_l132_132171


namespace initial_money_amount_l132_132186

theorem initial_money_amount (M : ℝ)
  (h_clothes : M * (1 / 3) = c)
  (h_food : (M - c) * (1 / 5) = f)
  (h_travel : (M - c - f) * (1 / 4) = t)
  (h_remaining : M - c - f - t = 600) : M = 1500 := by
  sorry

end initial_money_amount_l132_132186


namespace first_book_length_l132_132846

-- Statement of the problem
theorem first_book_length
  (x : ℕ) -- Number of pages in the first book
  (total_pages : ℕ)
  (days_in_two_weeks : ℕ)
  (pages_per_day : ℕ)
  (second_book_pages : ℕ := 100) :
  pages_per_day = 20 ∧ days_in_two_weeks = 14 ∧ total_pages = 280 ∧ total_pages = pages_per_day * days_in_two_weeks ∧ total_pages = x + second_book_pages → x = 180 :=
by
  sorry

end first_book_length_l132_132846


namespace angle_measure_l132_132667

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l132_132667


namespace no_solution_for_inequality_l132_132952

theorem no_solution_for_inequality (x : ℝ) (h : |x| > 2) : ¬ (5 * x^2 + 6 * x + 8 < 0) := 
by
  sorry

end no_solution_for_inequality_l132_132952


namespace regular_octagon_interior_angle_deg_l132_132448

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l132_132448


namespace cruise_liner_travelers_l132_132166

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l132_132166


namespace regular_octagon_interior_angle_l132_132415

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l132_132415


namespace man_is_older_by_l132_132312

theorem man_is_older_by :
  ∀ (M S : ℕ), S = 22 → (M + 2) = 2 * (S + 2) → (M - S) = 24 :=
by
  intros M S h1 h2
  sorry

end man_is_older_by_l132_132312


namespace purely_imaginary_complex_l132_132097

theorem purely_imaginary_complex (a : ℝ) 
  (h₁ : a^2 + 2 * a - 3 = 0)
  (h₂ : a + 3 ≠ 0) : a = 1 := by
  sorry

end purely_imaginary_complex_l132_132097


namespace sum_of_roots_l132_132241

theorem sum_of_roots (x : ℝ) (h : (x + 3) * (x - 2) = 15) : x = -1 :=
sorry

end sum_of_roots_l132_132241


namespace find_a_for_extraneous_roots_find_a_for_no_solution_l132_132076

-- Define the original fractional equation
def eq_fraction (x a: ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Proposition for extraneous roots
theorem find_a_for_extraneous_roots (a: ℝ) (extraneous_roots : ∃ x : ℝ, (x - a) / (x - 2) - 5 / x = 1 ∧ (x = 0 ∨ x = 2)): a = 2 := by 
sorry

-- Proposition for no solution
theorem find_a_for_no_solution (a: ℝ) (no_solution : ∀ x : ℝ, (x - a) / (x - 2) - 5 / x ≠ 1): a = -3 ∨ a = 2 := by 
sorry

end find_a_for_extraneous_roots_find_a_for_no_solution_l132_132076


namespace cost_per_chair_l132_132212

theorem cost_per_chair (total_spent : ℕ) (chairs_bought : ℕ) (cost : ℕ) 
  (h1 : total_spent = 180) 
  (h2 : chairs_bought = 12) 
  (h3 : cost = total_spent / chairs_bought) : 
  cost = 15 :=
by
  -- Proof steps go here (skipped with sorry)
  sorry

end cost_per_chair_l132_132212


namespace two_mul_seven_pow_n_plus_one_divisible_by_three_l132_132137

-- Definition of natural numbers
variable (n : ℕ)

-- Statement of the problem in Lean
theorem two_mul_seven_pow_n_plus_one_divisible_by_three (n : ℕ) : 3 ∣ (2 * 7^n + 1) := 
sorry

end two_mul_seven_pow_n_plus_one_divisible_by_three_l132_132137


namespace inequality_am_gm_l132_132344

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l132_132344


namespace meet_time_opposite_directions_catch_up_time_same_direction_l132_132505

def length_of_track := 440
def speed_A := 5
def speed_B := 6

theorem meet_time_opposite_directions :
  (length_of_track / (speed_A + speed_B)) = 40 :=
by
  sorry

theorem catch_up_time_same_direction :
  (length_of_track / (speed_B - speed_A)) = 440 :=
by
  sorry

end meet_time_opposite_directions_catch_up_time_same_direction_l132_132505


namespace simplify_fraction_product_l132_132518

theorem simplify_fraction_product : 
  (270 / 24) * (7 / 210) * (6 / 4) = 4.5 :=
by
  sorry

end simplify_fraction_product_l132_132518
