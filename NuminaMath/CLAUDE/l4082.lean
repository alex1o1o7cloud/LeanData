import Mathlib

namespace inequality_not_always_true_l4082_408251

theorem inequality_not_always_true (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a * c < 0) :
  ∃ a b c, a < b ∧ b < c ∧ a * c < 0 ∧ c^2 / a ≥ b^2 / a :=
sorry

end inequality_not_always_true_l4082_408251


namespace vegetable_ghee_weight_l4082_408270

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3360

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 330

theorem vegetable_ghee_weight : 
  weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume + 
  weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume = total_weight := by
sorry

end vegetable_ghee_weight_l4082_408270


namespace teacher_number_game_l4082_408258

theorem teacher_number_game (x : ℝ) : 
  let max_result := 2 * (3 * (x + 1))
  let lisa_result := 2 * ((max_result / 2) - 1)
  lisa_result = 2 * x + 2 := by sorry

end teacher_number_game_l4082_408258


namespace savings_calculation_l4082_408209

def num_machines : ℕ := 25
def bearings_per_machine : ℕ := 45
def regular_price : ℚ := 125/100
def sale_price : ℚ := 80/100
def discount_first_20 : ℚ := 25/100
def discount_remaining : ℚ := 35/100
def first_batch : ℕ := 20

def total_bearings : ℕ := num_machines * bearings_per_machine

def regular_total_cost : ℚ := (total_bearings : ℚ) * regular_price

def sale_cost_before_discount : ℚ := (total_bearings : ℚ) * sale_price

def first_batch_bearings : ℕ := first_batch * bearings_per_machine
def remaining_bearings : ℕ := total_bearings - first_batch_bearings

def first_batch_cost : ℚ := (first_batch_bearings : ℚ) * sale_price * (1 - discount_first_20)
def remaining_cost : ℚ := (remaining_bearings : ℚ) * sale_price * (1 - discount_remaining)

def total_discounted_cost : ℚ := first_batch_cost + remaining_cost

theorem savings_calculation : 
  regular_total_cost - total_discounted_cost = 74925/100 :=
by sorry

end savings_calculation_l4082_408209


namespace max_ab_value_l4082_408206

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (2^x * 2^y) → x * y ≤ a * b) → 
  a * b = 1/4 := by
sorry

end max_ab_value_l4082_408206


namespace number_of_observations_l4082_408261

theorem number_of_observations (original_mean new_mean : ℝ) (correction : ℝ) :
  original_mean = 36 →
  correction = 1 →
  new_mean = 36.02 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * new_mean = (n : ℝ) * original_mean + correction :=
by
  sorry

end number_of_observations_l4082_408261


namespace shopping_trip_expenses_l4082_408296

theorem shopping_trip_expenses (total : ℝ) (food_percent : ℝ) :
  (total > 0) →
  (food_percent ≥ 0) →
  (food_percent ≤ 100) →
  (0.5 * total + food_percent / 100 * total + 0.4 * total = total) →
  (0.04 * 0.5 * total + 0.08 * 0.4 * total = 0.052 * total) →
  food_percent = 10 := by
  sorry

end shopping_trip_expenses_l4082_408296


namespace cylinder_surface_area_minimized_l4082_408246

/-- Theorem: For a cylinder with fixed volume, the surface area is minimized when the height is twice the radius. -/
theorem cylinder_surface_area_minimized (R H V : ℝ) (h_positive : R > 0 ∧ H > 0 ∧ V > 0) 
  (h_volume : π * R^2 * H = V / 2) :
  let A := 2 * π * R^2 + 2 * π * R * H
  ∀ R' H' : ℝ, R' > 0 → H' > 0 → π * R'^2 * H' = V / 2 → 
    2 * π * R'^2 + 2 * π * R' * H' ≥ A → H / R = 2 :=
by sorry

end cylinder_surface_area_minimized_l4082_408246


namespace table_count_l4082_408230

theorem table_count (num_books : ℕ) (h : num_books = 100000) :
  ∃ (num_tables : ℕ),
    (num_tables : ℚ) * (2 / 5 * num_tables) = num_books ∧
    num_tables = 500 := by
  sorry

end table_count_l4082_408230


namespace problem_solution_l4082_408253

theorem problem_solution : (2200 - 2023)^2 / 196 = 144 := by
  sorry

end problem_solution_l4082_408253


namespace max_volume_regular_pyramid_l4082_408201

/-- 
For a regular n-sided pyramid with surface area S, 
prove that the maximum volume V is given by the formula:
V = (√2 / 12) * (S^(3/2)) / √(n * tan(π/n))
-/
theorem max_volume_regular_pyramid (n : ℕ) (S : ℝ) (h₁ : n ≥ 3) (h₂ : S > 0) :
  ∃ V : ℝ, V = (Real.sqrt 2 / 12) * S^(3/2) / Real.sqrt (n * Real.tan (π / n)) ∧
    ∀ V' : ℝ, (∃ (Q h : ℝ), V' = (1/3) * Q * h ∧ 
      S = Q + n * Q / (2 * Real.cos (π / n))) → V' ≤ V := by
  sorry


end max_volume_regular_pyramid_l4082_408201


namespace specific_lamp_arrangement_probability_l4082_408274

/-- The probability of a specific lamp arrangement and state --/
def specific_arrangement_probability (total_lamps : ℕ) (purple_lamps : ℕ) (green_lamps : ℕ) (lamps_on : ℕ) : ℚ :=
  let total_arrangements := Nat.choose total_lamps purple_lamps * Nat.choose total_lamps lamps_on
  let specific_arrangements := Nat.choose (total_lamps - 2) (purple_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_on - 1)
  specific_arrangements / total_arrangements

/-- The main theorem statement --/
theorem specific_lamp_arrangement_probability :
  specific_arrangement_probability 8 4 4 4 = 4 / 49 := by
  sorry

end specific_lamp_arrangement_probability_l4082_408274


namespace remainder_sum_mod_35_l4082_408217

theorem remainder_sum_mod_35 (f y z : ℤ) 
  (hf : f % 5 = 3) 
  (hy : y % 5 = 4) 
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 := by
  sorry

end remainder_sum_mod_35_l4082_408217


namespace probability_at_least_one_woman_l4082_408235

theorem probability_at_least_one_woman (total : ℕ) (men women selected : ℕ) 
  (h_total : total = men + women)
  (h_men : men = 6)
  (h_women : women = 4)
  (h_selected : selected = 3) :
  1 - (Nat.choose men selected : ℚ) / (Nat.choose total selected) = 5/6 := by
  sorry

end probability_at_least_one_woman_l4082_408235


namespace equation_rewrite_product_l4082_408200

theorem equation_rewrite_product (a b x y : ℝ) (m' n' p' : ℤ) :
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 2)) →
  ((a^m'*x - a^n') * (a^p'*y - a^3) = a^5*b^5) →
  m' * n' * p' = 48 := by
  sorry

end equation_rewrite_product_l4082_408200


namespace area_ratio_concentric_spheres_specific_sphere_areas_l4082_408265

/-- Given two concentric spheres with radii R₁ and R₂, if a region on the smaller sphere
    has an area A₁, then the corresponding region on the larger sphere has an area A₂. -/
theorem area_ratio_concentric_spheres (R₁ R₂ A₁ A₂ : ℝ) 
    (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : A₁ > 0) :
  R₁ = 4 → R₂ = 6 → A₁ = 37 → A₂ = (R₂ / R₁)^2 * A₁ → A₂ = 83.25 := by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_sphere_areas :
  let R₁ : ℝ := 4
  let R₂ : ℝ := 6
  let A₁ : ℝ := 37
  let A₂ : ℝ := (R₂ / R₁)^2 * A₁
  A₂ = 83.25 := by
  sorry

end area_ratio_concentric_spheres_specific_sphere_areas_l4082_408265


namespace completing_square_equivalence_l4082_408232

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 4*x + 3 = 0 ↔ (x - 2)^2 = 1 := by
  sorry

end completing_square_equivalence_l4082_408232


namespace triangle_problem_l4082_408237

noncomputable section

def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 - 1

theorem triangle_problem (A B C a b c : ℝ) :
  c = Real.sqrt 3 →
  f C = 0 →
  Real.sin B = 2 * Real.sin A →
  0 < C →
  C < π →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  C = π/3 ∧ a = 1 ∧ b = 2 := by
  sorry

end

end triangle_problem_l4082_408237


namespace bookstore_shoe_store_sales_coincidence_l4082_408215

def is_multiple_of_5 (n : ℕ) : Prop := ∃ k, n = 5 * k

def shoe_store_sale_day (n : ℕ) : Prop := ∃ k, n = 3 + 6 * k

def july_day (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 31

theorem bookstore_shoe_store_sales_coincidence :
  (∃! d : ℕ, july_day d ∧ is_multiple_of_5 d ∧ shoe_store_sale_day d) := by
  sorry

end bookstore_shoe_store_sales_coincidence_l4082_408215


namespace selene_purchase_cost_l4082_408294

/-- Calculates the total cost of items after applying a discount -/
def total_cost_after_discount (camera_price : ℚ) (frame_price : ℚ) (camera_count : ℕ) (frame_count : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_before_discount := camera_price * camera_count + frame_price * frame_count
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

/-- Proves that Selene pays $551 for her purchase -/
theorem selene_purchase_cost :
  let camera_price : ℚ := 110
  let frame_price : ℚ := 120
  let camera_count : ℕ := 2
  let frame_count : ℕ := 3
  let discount_rate : ℚ := 5 / 100
  total_cost_after_discount camera_price frame_price camera_count frame_count discount_rate = 551 := by
  sorry


end selene_purchase_cost_l4082_408294


namespace root_cube_relation_l4082_408272

/-- The polynomial f(x) = x^3 + 2x^2 + 3x + 4 -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- The polynomial h(x) = x^3 + bx^2 + cx + d -/
def h (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- Theorem stating the relationship between f and h, and the values of b, c, and d -/
theorem root_cube_relation (b c d : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧
    (∀ x : ℝ, h x b c d = 0 ↔ ∃ r : ℝ, f r = 0 ∧ x = r^3)) →
  b = 6 ∧ c = -8 ∧ d = 16 := by
sorry

end root_cube_relation_l4082_408272


namespace lindsey_final_balance_l4082_408212

def september_savings : ℕ := 50
def october_savings : ℕ := 37
def november_savings : ℕ := 11
def mom_bonus_threshold : ℕ := 75
def mom_bonus : ℕ := 25
def video_game_cost : ℕ := 87

def total_savings : ℕ := september_savings + october_savings + november_savings

def final_balance : ℕ :=
  if total_savings > mom_bonus_threshold
  then total_savings + mom_bonus - video_game_cost
  else total_savings - video_game_cost

theorem lindsey_final_balance : final_balance = 36 := by
  sorry

end lindsey_final_balance_l4082_408212


namespace chess_tournament_participants_l4082_408202

theorem chess_tournament_participants (x : ℕ) : 
  (∃ y : ℕ, 2 * x * y + 16 = (x + 2) * (x + 1)) ↔ (x = 7 ∨ x = 14) :=
by sorry

end chess_tournament_participants_l4082_408202


namespace remainder_of_482157_div_6_l4082_408283

theorem remainder_of_482157_div_6 : 482157 % 6 = 3 := by
  sorry

end remainder_of_482157_div_6_l4082_408283


namespace constant_term_proof_l4082_408213

theorem constant_term_proof (x y z : ℤ) (k : ℤ) : 
  x = 20 → 
  4 * x + y + z = k → 
  2 * x - y - z = 40 → 
  3 * x + y - z = 20 → 
  k = 80 := by
sorry

end constant_term_proof_l4082_408213


namespace appetizer_cost_per_person_l4082_408287

def potato_chip_cost : ℝ := 1.00
def creme_fraiche_cost : ℝ := 5.00
def caviar_cost : ℝ := 73.00
def num_people : ℕ := 3
def num_potato_chip_bags : ℕ := 3

theorem appetizer_cost_per_person :
  (num_potato_chip_bags * potato_chip_cost + creme_fraiche_cost + caviar_cost) / num_people = 27.00 := by
  sorry

end appetizer_cost_per_person_l4082_408287


namespace same_remainder_divisor_l4082_408233

theorem same_remainder_divisor : 
  ∃ (d : ℕ), d > 1 ∧ 
  (1059 % d = 1417 % d) ∧ 
  (1059 % d = 2312 % d) ∧ 
  (1417 % d = 2312 % d) ∧
  (∀ (k : ℕ), k > d → 
    (1059 % k ≠ 1417 % k) ∨ 
    (1059 % k ≠ 2312 % k) ∨ 
    (1417 % k ≠ 2312 % k)) →
  d = 179 := by
sorry

end same_remainder_divisor_l4082_408233


namespace card_value_decrease_is_57_16_l4082_408269

/-- The percent decrease of a baseball card's value over four years -/
def card_value_decrease : ℝ :=
  let year1_decrease := 0.30
  let year2_decrease := 0.10
  let year3_decrease := 0.20
  let year4_decrease := 0.15
  let remaining_value := (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease) * (1 - year4_decrease)
  (1 - remaining_value) * 100

/-- Theorem stating that the total percent decrease of the card's value over four years is 57.16% -/
theorem card_value_decrease_is_57_16 : 
  ∃ ε > 0, |card_value_decrease - 57.16| < ε :=
by
  sorry

end card_value_decrease_is_57_16_l4082_408269


namespace consecutive_integers_sum_l4082_408266

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (7 * n + 21 = 2821) → (n + 6 = 406) := by
  sorry

end consecutive_integers_sum_l4082_408266


namespace system_solution_unique_l4082_408241

theorem system_solution_unique :
  ∃! (x y : ℝ), x^2 + y * Real.sqrt (x * y) = 336 ∧ y^2 + x * Real.sqrt (x * y) = 112 ∧ x = 18 ∧ y = 2 := by
  sorry

end system_solution_unique_l4082_408241


namespace distribution_theorem_l4082_408227

-- Define the total number of employees
def total_employees : ℕ := 8

-- Define the number of departments
def num_departments : ℕ := 2

-- Define the number of English translators
def num_translators : ℕ := 2

-- Define the function to calculate the number of distribution schemes
def distribution_schemes (n : ℕ) (k : ℕ) (t : ℕ) : ℕ := 
  (Nat.choose (n - t) ((n - t) / 2)) * 2

-- Theorem statement
theorem distribution_theorem : 
  distribution_schemes total_employees num_departments num_translators = 40 := by
  sorry

end distribution_theorem_l4082_408227


namespace base_8_to_16_digit_count_l4082_408225

theorem base_8_to_16_digit_count :
  ∀ n : ℕ,
  (1000 ≤ n ∧ n ≤ 7777) →  -- 4 digits in base 8
  (512 ≤ n ∧ n ≤ 4095) →   -- Equivalent range in decimal
  (0x200 ≤ n ∧ n ≤ 0xFFF)  -- 3 digits in base 16
  := by sorry

end base_8_to_16_digit_count_l4082_408225


namespace exists_n_composite_power_of_two_plus_fifteen_l4082_408240

theorem exists_n_composite_power_of_two_plus_fifteen :
  ∃ n : ℕ, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2^n + 15 = a * b :=
by sorry

end exists_n_composite_power_of_two_plus_fifteen_l4082_408240


namespace eggs_per_crate_l4082_408297

theorem eggs_per_crate (initial_crates : ℕ) (given_away : ℕ) (additional_crates : ℕ) (final_count : ℕ) :
  initial_crates = 6 →
  given_away = 2 →
  additional_crates = 5 →
  final_count = 270 →
  ∃ (eggs_per_crate : ℕ), eggs_per_crate = 30 ∧
    final_count = (initial_crates - given_away + additional_crates) * eggs_per_crate :=
by sorry

end eggs_per_crate_l4082_408297


namespace spade_or_king_probability_l4082_408238

/-- The probability of drawing a spade or a king from a standard deck of cards -/
theorem spade_or_king_probability (total_cards : ℕ) (spades : ℕ) (kings : ℕ) (overlap : ℕ) :
  total_cards = 52 →
  spades = 13 →
  kings = 4 →
  overlap = 1 →
  (spades + kings - overlap : ℚ) / total_cards = 4 / 13 := by
sorry

end spade_or_king_probability_l4082_408238


namespace D_72_l4082_408277

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where order matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) = 43 -/
theorem D_72 : D 72 = 43 := by sorry

end D_72_l4082_408277


namespace embankment_construction_time_l4082_408293

/-- Given that 60 workers take 3 days to build half of an embankment,
    prove that 45 workers would take 8 days to build the entire embankment,
    assuming all workers work at the same rate. -/
theorem embankment_construction_time
  (workers_60 : ℕ) (days_60 : ℕ) (half_embankment : ℚ)
  (workers_45 : ℕ) (days_45 : ℕ) (full_embankment : ℚ)
  (h1 : workers_60 = 60)
  (h2 : days_60 = 3)
  (h3 : half_embankment = 1/2)
  (h4 : workers_45 = 45)
  (h5 : days_45 = 8)
  (h6 : full_embankment = 1)
  (h7 : ∀ w d, w * d * half_embankment = workers_60 * days_60 * half_embankment →
               w * d * full_embankment = workers_45 * days_45 * full_embankment) :
  workers_45 * days_45 * full_embankment = workers_60 * days_60 * full_embankment :=
by sorry

end embankment_construction_time_l4082_408293


namespace sufficient_but_not_necessary_l4082_408226

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x < 1 → x^2 - 4*x + 3 > 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x + 3 > 0 ∧ x ≥ 1) := by
  sorry

end sufficient_but_not_necessary_l4082_408226


namespace vectors_theorem_l4082_408256

/-- Two non-collinear vectors in a plane -/
structure NonCollinearVectors (V : Type*) [AddCommGroup V] [Module ℝ V] where
  e₁ : V
  e₂ : V
  noncollinear : ¬ ∃ (r : ℝ), e₁ = r • e₂

/-- Definition of vectors AB, CB, and CD -/
def vectors_relation (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (ncv : NonCollinearVectors V) (k : ℝ) : Prop :=
  ∃ (A B C D : V),
    B - A = ncv.e₁ - k • ncv.e₂ ∧
    B - C = 2 • ncv.e₁ + ncv.e₂ ∧
    D - C = 3 • ncv.e₁ - ncv.e₂

/-- Collinearity of points A, B, and D -/
def collinear (V : Type*) [AddCommGroup V] [Module ℝ V] (A B D : V) : Prop :=
  ∃ (t : ℝ), D - A = t • (B - A)

/-- The main theorem -/
theorem vectors_theorem (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (ncv : NonCollinearVectors V) :
  ∀ k, vectors_relation V ncv k → 
  (∃ A B D, collinear V A B D) → 
  k = 2 := by
  sorry

end vectors_theorem_l4082_408256


namespace bicycle_ride_time_l4082_408218

/-- Proves the total time Hyeonil rode the bicycle given the conditions -/
theorem bicycle_ride_time (speed : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
  (h1 : speed = 4.25)
  (h2 : initial_time = 60)
  (h3 : additional_distance = 29.75) :
  initial_time + additional_distance / speed = 67 := by
  sorry

#check bicycle_ride_time

end bicycle_ride_time_l4082_408218


namespace smallest_number_proof_l4082_408262

theorem smallest_number_proof (x y z : ℝ) 
  (sum_xy : x + y = 23)
  (sum_xz : x + z = 31)
  (sum_yz : y + z = 11) :
  min x (min y z) = 21.5 := by
sorry

end smallest_number_proof_l4082_408262


namespace haley_recycling_cans_l4082_408224

theorem haley_recycling_cans : ∃ (c : ℕ), c = 9 ∧ c - 7 = 2 := by
  sorry

end haley_recycling_cans_l4082_408224


namespace initial_cards_l4082_408285

theorem initial_cards (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 4 → total = 13 → initial + added = total → initial = 9 := by
  sorry

end initial_cards_l4082_408285


namespace bales_stored_l4082_408254

theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 22)
  (h2 : final_bales = 89) :
  final_bales - initial_bales = 67 := by
  sorry

end bales_stored_l4082_408254


namespace bread_cost_l4082_408243

/-- The cost of the loaf of bread given the conditions of Ted's sandwich-making scenario --/
theorem bread_cost (sandwich_meat_cost meat_packs cheese_cost cheese_packs : ℕ → ℚ)
  (meat_coupon cheese_coupon : ℚ) (sandwich_price : ℚ) (sandwich_count : ℕ) :
  let total_meat_cost := meat_packs 2 * sandwich_meat_cost 1 - meat_coupon
  let total_cheese_cost := cheese_packs 2 * cheese_cost 1 - cheese_coupon
  let total_ingredient_cost := total_meat_cost + total_cheese_cost
  let total_revenue := sandwich_count * sandwich_price
  total_revenue - total_ingredient_cost = 4 :=
by sorry

#check bread_cost (λ _ => 5) (λ _ => 2) (λ _ => 4) (λ _ => 2) 1 1 2 10

end bread_cost_l4082_408243


namespace probability_at_least_one_female_l4082_408248

/-- The probability of selecting at least one female student when choosing 2 people
    from a group of 3 male and 2 female students is 0.7 -/
theorem probability_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (selected : ℕ) (h1 : total_students = male_students + female_students)
  (h2 : total_students = 5) (h3 : male_students = 3) (h4 : female_students = 2) (h5 : selected = 2) :
  (Nat.choose total_students selected - Nat.choose male_students selected : ℚ) /
  Nat.choose total_students selected = 7/10 := by
sorry

end probability_at_least_one_female_l4082_408248


namespace no_primes_in_factorial_range_l4082_408289

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k ∈ Finset.range n, ¬ Nat.Prime (n! - k) :=
by sorry

end no_primes_in_factorial_range_l4082_408289


namespace cistern_filling_time_l4082_408216

theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 8 →
  combined_fill_time = 40 / 3 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 5 := by
sorry

end cistern_filling_time_l4082_408216


namespace probability_ratio_l4082_408282

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing four slips with the same number -/
def p : ℚ := (distinct_numbers * (slips_per_number.choose drawn_slips)) / (total_slips.choose drawn_slips)

/-- The probability of drawing two pairs of different numbers -/
def q : ℚ := (distinct_numbers.choose 2 * (slips_per_number.choose 2) * (slips_per_number.choose 2)) / (total_slips.choose drawn_slips)

theorem probability_ratio :
  q / p = 90 := by sorry

end probability_ratio_l4082_408282


namespace remainder_divisibility_l4082_408268

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 17) → 
  (∃ m : ℤ, N = 13 * m + 4) :=
by sorry

end remainder_divisibility_l4082_408268


namespace quadratic_is_perfect_square_perfect_square_coefficient_l4082_408257

/-- A quadratic expression is a perfect square if and only if its discriminant is zero -/
theorem quadratic_is_perfect_square (a b c : ℝ) :
  (∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2) ↔ b^2 = 4 * a * c := by sorry

/-- The main theorem: If 6x^2 + cx + 16 is a perfect square, then c = 8√6 -/
theorem perfect_square_coefficient (c : ℝ) :
  (∃ p q : ℝ, ∀ x, 6 * x^2 + c * x + 16 = (p * x + q)^2) → c = 8 * Real.sqrt 6 := by sorry

end quadratic_is_perfect_square_perfect_square_coefficient_l4082_408257


namespace octal_7624_is_decimal_3988_l4082_408275

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem octal_7624_is_decimal_3988 : octal_to_decimal 7624 = 3988 := by
  sorry

end octal_7624_is_decimal_3988_l4082_408275


namespace cookie_ratio_proof_l4082_408247

def cookie_problem (initial cookies_to_friend cookies_eaten cookies_left : ℕ) : Prop :=
  let cookies_after_friend := initial - cookies_to_friend
  let cookies_to_family := cookies_after_friend - cookies_eaten - cookies_left
  (2 * cookies_to_family = cookies_after_friend)

theorem cookie_ratio_proof :
  cookie_problem 19 5 2 5 := by
  sorry

end cookie_ratio_proof_l4082_408247


namespace quadratic_two_distinct_roots_l4082_408284

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - x₁ - k^2 = 0 ∧ x₂^2 - x₂ - k^2 = 0 :=
by sorry

end quadratic_two_distinct_roots_l4082_408284


namespace jennas_profit_calculation_l4082_408255

/-- Calculates Jenna's total profit after taxes for her wholesale business --/
def jennas_profit (supplier_a_price supplier_b_price resell_price rent tax_rate worker_salary shipping_fee supplier_a_qty supplier_b_qty : ℚ) : ℚ :=
  let total_widgets := supplier_a_qty + supplier_b_qty
  let purchase_cost := supplier_a_price * supplier_a_qty + supplier_b_price * supplier_b_qty
  let shipping_cost := shipping_fee * total_widgets
  let worker_cost := 4 * worker_salary
  let total_expenses := purchase_cost + shipping_cost + rent + worker_cost
  let revenue := resell_price * total_widgets
  let profit_before_tax := revenue - total_expenses
  let tax := tax_rate * profit_before_tax
  profit_before_tax - tax

theorem jennas_profit_calculation :
  jennas_profit 3.5 4 8 10000 0.25 2500 0.25 3000 2000 = 187.5 := by
  sorry

end jennas_profit_calculation_l4082_408255


namespace incorrect_spelling_probability_incorrect_spelling_probability_is_59_60_l4082_408207

/-- The probability of spelling "theer" incorrectly -/
theorem incorrect_spelling_probability : ℚ :=
  let total_letters : ℕ := 5
  let repeated_letter : ℕ := 2
  let distinct_letters : ℕ := 3
  let total_arrangements : ℕ := (Nat.choose total_letters repeated_letter) * (Nat.factorial distinct_letters)
  let correct_arrangements : ℕ := 1
  (total_arrangements - correct_arrangements : ℚ) / total_arrangements

/-- Proof that the probability of spelling "theer" incorrectly is 59/60 -/
theorem incorrect_spelling_probability_is_59_60 : 
  incorrect_spelling_probability = 59 / 60 := by
  sorry

end incorrect_spelling_probability_incorrect_spelling_probability_is_59_60_l4082_408207


namespace percentage_not_sold_is_25_percent_l4082_408204

-- Define the initial stock and daily sales
def initial_stock : ℕ := 600
def monday_sales : ℕ := 25
def tuesday_sales : ℕ := 70
def wednesday_sales : ℕ := 100
def thursday_sales : ℕ := 110
def friday_sales : ℕ := 145

-- Define the total sales
def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

-- Define the number of bags not sold
def bags_not_sold : ℕ := initial_stock - total_sales

-- Theorem to prove
theorem percentage_not_sold_is_25_percent :
  (bags_not_sold : ℚ) / (initial_stock : ℚ) * 100 = 25 := by
  sorry


end percentage_not_sold_is_25_percent_l4082_408204


namespace john_gathered_20_l4082_408260

/-- Given the total number of milk bottles and the number Marcus gathered,
    calculate the number of milk bottles John gathered. -/
def john_bottles (total : ℕ) (marcus : ℕ) : ℕ :=
  total - marcus

/-- Theorem stating that given 45 total milk bottles and 25 gathered by Marcus,
    John gathered 20 milk bottles. -/
theorem john_gathered_20 :
  john_bottles 45 25 = 20 := by
  sorry

end john_gathered_20_l4082_408260


namespace tower_arrangements_l4082_408276

/-- The number of ways to arrange n distinct objects --/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ways to arrange k objects from n distinct objects --/
def permutation (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k objects from n objects with repetition allowed --/
def multisetPermutation (n : ℕ) (k : List ℕ) : ℕ := sorry

/-- The number of different towers that can be built --/
def numTowers : ℕ := sorry

theorem tower_arrangements :
  let red := 3
  let blue := 3
  let green := 4
  let towerHeight := 9
  numTowers = multisetPermutation towerHeight [red - 1, blue, green] +
              multisetPermutation towerHeight [red, blue - 1, green] +
              multisetPermutation towerHeight [red, blue, green - 1] :=
by sorry

end tower_arrangements_l4082_408276


namespace range_of_a_for_decreasing_f_l4082_408280

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y < 3 → f a x > f a y) → 
  (a ≥ 0 ∧ a ≤ 3/4) :=
sorry

end range_of_a_for_decreasing_f_l4082_408280


namespace triangle_side_b_value_l4082_408259

theorem triangle_side_b_value (A B C : ℝ) (a b c : ℝ) :
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 2 →
  (a / Real.sin A = b / Real.sin B) →
  b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_value_l4082_408259


namespace first_term_is_four_l4082_408299

/-- Geometric sequence with common ratio -2 and sum of first 5 terms equal to 44 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * (-2)) ∧ 
  (a 1 + a 2 + a 3 + a 4 + a 5 = 44)

/-- The first term of the geometric sequence is 4 -/
theorem first_term_is_four (a : ℕ → ℝ) (h : geometric_sequence a) : a 1 = 4 := by
  sorry

end first_term_is_four_l4082_408299


namespace string_length_for_circular_token_l4082_408271

theorem string_length_for_circular_token : 
  let area : ℝ := 616
  let pi_approx : ℝ := 22 / 7
  let extra_length : ℝ := 5
  let radius : ℝ := Real.sqrt (area * 7 / 22)
  let circumference : ℝ := 2 * pi_approx * radius
  circumference + extra_length = 93 := by sorry

end string_length_for_circular_token_l4082_408271


namespace sheila_hourly_wage_l4082_408222

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hours_tth : ℕ  -- Hours worked on Tuesday, Thursday
  days_mwf : ℕ   -- Number of days worked with hours_mwf
  days_tth : ℕ   -- Number of days worked with hours_tth
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_mwf * schedule.days_mwf + schedule.hours_tth * schedule.days_tth
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $11 --/
theorem sheila_hourly_wage :
  let schedule : WorkSchedule := {
    hours_mwf := 8,
    hours_tth := 6,
    days_mwf := 3,
    days_tth := 2,
    weekly_earnings := 396
  }
  hourly_wage schedule = 11 := by
  sorry

end sheila_hourly_wage_l4082_408222


namespace binomial_simplification_l4082_408252

/-- Given two binomials M and N in terms of x, prove that if 2(M) - 3(N) = 4x - 6 - 9x - 15,
    then N = 3x + 5 and the simplified expression P = -5x - 21 -/
theorem binomial_simplification (x : ℝ) (M N : ℝ → ℝ) :
  (∀ x, 2 * M x - 3 * N x = 4 * x - 6 - 9 * x - 15) →
  (∀ x, N x = 3 * x + 5) ∧
  (∀ x, 2 * M x - 3 * N x = -5 * x - 21) :=
by sorry

end binomial_simplification_l4082_408252


namespace tyler_cake_eggs_l4082_408223

/-- Represents the number of eggs needed for a cake --/
def eggs_for_cake (people : ℕ) : ℕ := 2 * (people / 4)

/-- Represents the number of additional eggs needed --/
def additional_eggs_needed (recipe_eggs : ℕ) (available_eggs : ℕ) : ℕ :=
  max (recipe_eggs - available_eggs) 0

theorem tyler_cake_eggs : 
  additional_eggs_needed (eggs_for_cake 8) 3 = 1 := by sorry

end tyler_cake_eggs_l4082_408223


namespace tan_order_l4082_408267

open Real

noncomputable def f (x : ℝ) := tan (x + π/4)

theorem tan_order : f 0 > f (-1) ∧ f (-1) > f 1 := by sorry

end tan_order_l4082_408267


namespace sum_of_squares_in_sequence_l4082_408229

/-- A sequence with the property that a_{2n-1} = a_{n-1}^2 + a_n^2 for all n -/
def phi_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (2*n - 1) = (a (n-1))^2 + (a n)^2

theorem sum_of_squares_in_sequence (a : ℕ → ℝ) (h : phi_sequence a) :
  ∀ n : ℕ, ∃ m : ℕ, a m = (a (n-1))^2 + (a n)^2 :=
sorry

end sum_of_squares_in_sequence_l4082_408229


namespace probability_of_selecting_letter_l4082_408290

theorem probability_of_selecting_letter (total_letters : ℕ) (unique_letters : ℕ) 
  (h1 : total_letters = 26) (h2 : unique_letters = 8) : 
  (unique_letters : ℚ) / total_letters = 4 / 13 := by
  sorry

end probability_of_selecting_letter_l4082_408290


namespace meeting_point_symmetry_l4082_408249

theorem meeting_point_symmetry 
  (d : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / (a + b) * (d - a / 2) - b / (a + b) * d = -2) →
  (a / (a + b) * (d - b / 2) - a / (a + b) * d = -2) :=
by sorry

end meeting_point_symmetry_l4082_408249


namespace sin_sum_of_complex_exponentials_l4082_408205

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I ∧
  Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I →
  Real.sin (γ + δ) = 21/65 := by
  sorry

end sin_sum_of_complex_exponentials_l4082_408205


namespace sum_always_positive_l4082_408219

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (h_incr : MonotonicIncreasing f)
  (h_odd : OddFunction f)
  (h_arith : ArithmeticSequence a)
  (h_a3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end sum_always_positive_l4082_408219


namespace domain_shift_l4082_408298

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc (-1) 1

-- Define the domain of f(x + 1)
def domain_f_shifted : Set ℝ := Set.Icc (-2) 0

-- Theorem statement
theorem domain_shift :
  (∀ x ∈ domain_f, f x ≠ 0) →
  (∀ y ∈ domain_f_shifted, f (y + 1) ≠ 0) :=
sorry

end domain_shift_l4082_408298


namespace cubic_integer_roots_l4082_408292

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of distinct integer roots of a cubic polynomial -/
def num_distinct_integer_roots (p : CubicPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of distinct integer roots -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  num_distinct_integer_roots p ∈ ({0, 1, 2, 3} : Set ℕ) :=
sorry

end cubic_integer_roots_l4082_408292


namespace right_triangle_height_l4082_408286

theorem right_triangle_height (a b c h : ℝ) : 
  (a = 12 ∧ b = 5 ∧ a^2 + b^2 = c^2) → 
  (h = 60/13 ∨ h = 5) := by
sorry

end right_triangle_height_l4082_408286


namespace consecutive_sum_transformation_l4082_408236

theorem consecutive_sum_transformation (S : ℤ) : 
  ∃ (a : ℤ), 
    (a + (a + 1) = S) → 
    (3 * (a + 5) + 3 * ((a + 1) + 5) = 3 * S + 30) := by
  sorry

end consecutive_sum_transformation_l4082_408236


namespace no_function_satisfying_condition_l4082_408221

open Real

-- Define the type for positive real numbers
def PositiveReal := {x : ℝ // x > 0}

-- State the theorem
theorem no_function_satisfying_condition :
  ¬ ∃ (f : PositiveReal → PositiveReal),
    ∀ (x y : PositiveReal),
      (f (⟨x.val + y.val, sorry⟩)).val ^ 2 ≥ (f x).val ^ 2 * (1 + y.val * (f x).val) :=
by sorry

end no_function_satisfying_condition_l4082_408221


namespace multiply_80641_and_9999_l4082_408263

theorem multiply_80641_and_9999 : 80641 * 9999 = 805589359 := by
  sorry

end multiply_80641_and_9999_l4082_408263


namespace possible_values_of_a_l4082_408234

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) →
  (a = 3 ∨ a = 7) := by
sorry

end possible_values_of_a_l4082_408234


namespace lake_superior_weighted_average_l4082_408281

/-- Represents the data for fish caught in a lake -/
structure LakeFishData where
  species : List String
  counts : List Nat
  weights : List Float

/-- Calculates the weighted average weight of fish in a lake -/
def weightedAverageWeight (data : LakeFishData) : Float :=
  let totalWeight := (List.zip data.counts data.weights).map (fun (c, w) => c.toFloat * w) |> List.sum
  let totalCount := data.counts.sum
  totalWeight / totalCount.toFloat

/-- The fish data for Lake Superior -/
def lakeSuperiorData : LakeFishData :=
  { species := ["Perch", "Northern Pike", "Whitefish"]
  , counts := [17, 15, 8]
  , weights := [2.5, 4.0, 3.5] }

/-- Theorem stating that the weighted average weight of fish in Lake Superior is 3.2625kg -/
theorem lake_superior_weighted_average :
  weightedAverageWeight lakeSuperiorData = 3.2625 := by
  sorry

end lake_superior_weighted_average_l4082_408281


namespace willie_stickers_l4082_408211

theorem willie_stickers (initial : ℝ) (received : ℝ) (total : ℝ) :
  initial = 278.5 →
  received = 43.8 →
  total = initial + received →
  total = 322.3 := by
sorry

end willie_stickers_l4082_408211


namespace negation_of_universal_statement_l4082_408264

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x - 2| < 3) ↔ (∃ x : ℝ, |x - 2| ≥ 3) := by sorry

end negation_of_universal_statement_l4082_408264


namespace word_problems_count_l4082_408208

theorem word_problems_count (total_questions : ℕ) (addition_subtraction_problems : ℕ) 
  (h1 : total_questions = 45)
  (h2 : addition_subtraction_problems = 28) :
  total_questions - addition_subtraction_problems = 17 := by
  sorry

end word_problems_count_l4082_408208


namespace greatest_integer_less_than_negative_fraction_l4082_408273

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end greatest_integer_less_than_negative_fraction_l4082_408273


namespace abs_one_plus_i_over_i_l4082_408295

def i : ℂ := Complex.I

theorem abs_one_plus_i_over_i : Complex.abs ((1 + i) / i) = Real.sqrt 2 := by sorry

end abs_one_plus_i_over_i_l4082_408295


namespace absolute_value_of_specific_integers_l4082_408244

theorem absolute_value_of_specific_integers :
  ∃ (a b c : ℤ),
    (∀ x : ℤ, x < 0 → x ≤ a) ∧
    (∀ x : ℤ, |x| ≥ |b|) ∧
    (∀ x : ℤ, x > 0 → c ≤ x) ∧
    |a + b - c| = 2 :=
by sorry

end absolute_value_of_specific_integers_l4082_408244


namespace cube_root_equation_solution_l4082_408239

theorem cube_root_equation_solution (x : ℝ) : 
  (15 * x + (15 * x + 8) ^ (1/3)) ^ (1/3) = 8 → x = 168/5 := by
  sorry

end cube_root_equation_solution_l4082_408239


namespace sandras_puppies_l4082_408291

theorem sandras_puppies (total_portions : ℕ) (num_days : ℕ) (feedings_per_day : ℕ) :
  total_portions = 105 →
  num_days = 5 →
  feedings_per_day = 3 →
  (total_portions / num_days) / feedings_per_day = 7 :=
by sorry

end sandras_puppies_l4082_408291


namespace checkerboard_chips_l4082_408210

/-- The total number of chips on an n × n checkerboard where each square (i, j) has |i - j| chips -/
def total_chips (n : ℕ) : ℕ := n * (n + 1) * (n - 1) / 3

/-- Theorem stating that if the total number of chips is 2660, then n = 20 -/
theorem checkerboard_chips (n : ℕ) : total_chips n = 2660 → n = 20 := by
  sorry

end checkerboard_chips_l4082_408210


namespace perpendicular_from_perpendicular_and_parallel_perpendicular_from_parallel_planes_l4082_408203

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_from_perpendicular_and_parallel
  (m n : Line) (α : Plane)
  (h1 : perpendicular_plane m α)
  (h2 : parallel_plane n α) :
  perpendicular m n :=
sorry

-- Theorem 2
theorem perpendicular_from_parallel_planes
  (m : Line) (α β γ : Plane)
  (h1 : parallel_planes α β)
  (h2 : parallel_planes β γ)
  (h3 : perpendicular_plane m α) :
  perpendicular_plane m γ :=
sorry

end perpendicular_from_perpendicular_and_parallel_perpendicular_from_parallel_planes_l4082_408203


namespace production_exceeds_target_in_2022_l4082_408279

def initial_production : ℕ := 20000
def annual_increase_rate : ℝ := 0.2
def target_production : ℕ := 60000
def start_year : ℕ := 2015

theorem production_exceeds_target_in_2022 :
  let production_after_n_years (n : ℕ) := initial_production * (1 + annual_increase_rate) ^ n
  ∀ y : ℕ, y < 2022 - start_year → production_after_n_years y ≤ target_production ∧
  production_after_n_years (2022 - start_year) > target_production :=
by sorry

end production_exceeds_target_in_2022_l4082_408279


namespace diophantine_equation_solution_l4082_408228

theorem diophantine_equation_solution (x y : ℤ) :
  7 * x - 3 * y = 2 ↔ ∃ k : ℤ, x = 3 * k + 2 ∧ y = 7 * k + 4 := by
  sorry

end diophantine_equation_solution_l4082_408228


namespace exam_students_count_l4082_408250

theorem exam_students_count : 
  ∀ (total : ℕ) (first_div second_div just_passed : ℝ),
    first_div = 0.25 * total →
    second_div = 0.54 * total →
    just_passed = total - first_div - second_div →
    just_passed = 63 →
    total = 300 := by
  sorry

end exam_students_count_l4082_408250


namespace not_identity_element_l4082_408288

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b + 1

-- Theorem stating that 1 is not an identity element for * in S
theorem not_identity_element :
  ¬ (∀ a ∈ S, (star 1 a = a ∧ star a 1 = a)) :=
by sorry

end not_identity_element_l4082_408288


namespace chord_distance_from_center_l4082_408231

/-- Given a circle where a chord intersects a diameter at an angle of 30° and
    divides it into segments of lengths a and b, the distance from the center
    of the circle to the chord is (1/4)|a - b|. -/
theorem chord_distance_from_center (a b : ℝ) :
  let chord_angle : ℝ := 30 * π / 180  -- 30° in radians
  let distance_to_chord : ℝ → ℝ → ℝ := λ x y => (1/4) * |x - y|
  chord_angle = 30 * π / 180 →
  distance_to_chord a b = (1/4) * |a - b| :=
by sorry

end chord_distance_from_center_l4082_408231


namespace discount_calculation_l4082_408214

/-- Given the original cost of plants and the amount actually spent, prove that the discount received is $399.00 -/
theorem discount_calculation (original_cost spent_amount : ℚ) 
  (h1 : original_cost = 467) 
  (h2 : spent_amount = 68) : 
  original_cost - spent_amount = 399 := by
  sorry

end discount_calculation_l4082_408214


namespace collinear_points_d_values_l4082_408242

-- Define the points
def point_a (a : ℝ) : ℝ × ℝ × ℝ := (1, 0, a)
def point_b (b : ℝ) : ℝ × ℝ × ℝ := (b, 1, 0)
def point_c (c : ℝ) : ℝ × ℝ × ℝ := (0, c, 1)
def point_d (d : ℝ) : ℝ × ℝ × ℝ := (4*d, 4*d, -2*d)

-- Define collinearity
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), q - p = t • (r - p)

theorem collinear_points_d_values (a b c d : ℝ) :
  collinear (point_a a) (point_b b) (point_c c) ∧
  collinear (point_a a) (point_b b) (point_d d) →
  d = 1 ∨ d = 1/4 :=
sorry

end collinear_points_d_values_l4082_408242


namespace quadratic_equation_solutions_l4082_408278

theorem quadratic_equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = -3 ∧ 
  (∀ x : ℝ, x^2 + 2*x + 1 = 4 ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end quadratic_equation_solutions_l4082_408278


namespace medal_distribution_proof_l4082_408245

/-- Represents the number of runners --/
def total_runners : ℕ := 10

/-- Represents the number of British runners --/
def british_runners : ℕ := 4

/-- Represents the number of medals --/
def medals : ℕ := 3

/-- Calculates the number of ways to award medals with at least one British runner winning --/
def ways_to_award_medals : ℕ := sorry

theorem medal_distribution_proof :
  ways_to_award_medals = 492 :=
by sorry

end medal_distribution_proof_l4082_408245


namespace valid_fractions_are_complete_l4082_408220

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_fraction_in_range (n d : ℕ) : Prop :=
  22 / 3 < n / d ∧ n / d < 15 / 2

def is_valid_fraction (n d : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit d ∧ is_fraction_in_range n d ∧ Nat.gcd n d = 1

def valid_fractions : Set (ℕ × ℕ) :=
  {(81, 11), (82, 11), (89, 12), (96, 13), (97, 13)}

theorem valid_fractions_are_complete :
  ∀ (n d : ℕ), is_valid_fraction n d ↔ (n, d) ∈ valid_fractions := by sorry

end valid_fractions_are_complete_l4082_408220
