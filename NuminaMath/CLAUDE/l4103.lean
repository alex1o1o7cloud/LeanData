import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_expression_l4103_410342

def is_distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def expression (a b c : ℕ) : ℚ :=
  1 / (a + 2010 / (b + 1 / c))

theorem max_value_of_expression :
  ∀ a b c : ℕ,
    is_distinct a b c →
    is_nonzero_digit a →
    is_nonzero_digit b →
    is_nonzero_digit c →
    expression a b c ≤ 1 / 203 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4103_410342


namespace NUMINAMATH_CALUDE_race_head_start_l4103_410370

theorem race_head_start (L : ℝ) (va vb : ℝ) (h : va = 20/13 * vb) :
  let H := (L - H) / vb + 0.6 * L / vb - L / va
  H = 19/20 * L := by
sorry

end NUMINAMATH_CALUDE_race_head_start_l4103_410370


namespace NUMINAMATH_CALUDE_solution_form_l4103_410316

/-- A continuous function satisfying the given integral equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  ∀ (x : ℝ) (n : ℕ), n ≠ 0 →
    (n : ℝ)^2 * ∫ t in x..(x + 1 / (n : ℝ)), f t = (n : ℝ) * f x + 1 / 2

/-- The theorem stating the form of functions satisfying the equation -/
theorem solution_form (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_solution_form_l4103_410316


namespace NUMINAMATH_CALUDE_right_triangle_area_l4103_410382

theorem right_triangle_area (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  (1/2 : ℝ) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4103_410382


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_l4103_410363

theorem stratified_sampling_third_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (h1 : total_students = 2400) 
  (h2 : sample_size = 120) 
  (h3 : first_year = 760) 
  (h4 : second_year = 840) : 
  (total_students - first_year - second_year) * sample_size / total_students = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_l4103_410363


namespace NUMINAMATH_CALUDE_cos_difference_l4103_410321

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1) 
  (h2 : Real.cos A + Real.cos B = 3/2) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_l4103_410321


namespace NUMINAMATH_CALUDE_f_properties_l4103_410365

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem f_properties :
  (∃ (x₀ : ℝ), IsLocalMax f x₀) ∧
  (¬ ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M) ∧
  (∀ (b : ℝ), (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b) →
    (0 < b ∧ b < 6 * Real.exp (-3))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4103_410365


namespace NUMINAMATH_CALUDE_red_other_side_probability_l4103_410374

structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

def box : Finset Card := sorry

axiom box_size : box.card = 8

axiom black_both_sides : (box.filter (fun c => !c.side1 ∧ !c.side2)).card = 4

axiom black_red_sides : (box.filter (fun c => (c.side1 ∧ !c.side2) ∨ (!c.side1 ∧ c.side2))).card = 2

axiom red_both_sides : (box.filter (fun c => c.side1 ∧ c.side2)).card = 2

def observe_red (c : Card) : Bool := c.side1 ∨ c.side2

def other_side_red (c : Card) : Bool := c.side1 ∧ c.side2

theorem red_other_side_probability :
  (box.filter (fun c => other_side_red c)).card / (box.filter (fun c => observe_red c)).card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_red_other_side_probability_l4103_410374


namespace NUMINAMATH_CALUDE_lindsey_remaining_money_l4103_410311

/-- Calculates Lindsey's remaining money after saving and spending --/
theorem lindsey_remaining_money 
  (september_savings : ℕ) 
  (october_savings : ℕ) 
  (november_savings : ℕ) 
  (mom_bonus_threshold : ℕ) 
  (mom_bonus : ℕ) 
  (video_game_cost : ℕ) : 
  (september_savings = 50 ∧ 
   october_savings = 37 ∧ 
   november_savings = 11 ∧ 
   mom_bonus_threshold = 75 ∧ 
   mom_bonus = 25 ∧ 
   video_game_cost = 87) → 
  (let total_savings := september_savings + october_savings + november_savings
   let total_with_bonus := total_savings + (if total_savings > mom_bonus_threshold then mom_bonus else 0)
   let remaining_money := total_with_bonus - video_game_cost
   remaining_money = 36) := by
sorry

end NUMINAMATH_CALUDE_lindsey_remaining_money_l4103_410311


namespace NUMINAMATH_CALUDE_max_of_4_2_neg5_l4103_410356

def find_max (a b c : Int) : Int :=
  let max1 := max a b
  max max1 c

theorem max_of_4_2_neg5 :
  find_max 4 2 (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_of_4_2_neg5_l4103_410356


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l4103_410324

theorem sin_sum_of_complex_exponentials
  (γ δ : ℝ)
  (h1 : Complex.exp (γ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I)
  (h2 : Complex.exp (δ * Complex.I) = -(5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = -(63 / 65) :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l4103_410324


namespace NUMINAMATH_CALUDE_min_omega_for_cosine_function_l4103_410353

theorem min_omega_for_cosine_function (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.cos (ω * x - π / 6)) →
  (ω > 0) →
  (∀ x, f x ≤ f (π / 4)) →
  (∀ ω' > 0, (∀ x, Real.cos (ω' * x - π / 6) ≤ Real.cos (ω' * π / 4 - π / 6)) → ω' ≥ 2 / 3) →
  ω = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_omega_for_cosine_function_l4103_410353


namespace NUMINAMATH_CALUDE_zoo_trip_remainder_is_24_l4103_410331

/-- Calculates the amount left for lunch and snacks after a zoo trip for two people -/
def zoo_trip_remainder (zoo_ticket_price : ℚ) (bus_fare_one_way : ℚ) (total_money : ℚ) : ℚ :=
  let zoo_cost := 2 * zoo_ticket_price
  let bus_cost := 2 * 2 * bus_fare_one_way
  total_money - (zoo_cost + bus_cost)

/-- Theorem: Given the specified prices and total money, the remainder for lunch and snacks is $24 -/
theorem zoo_trip_remainder_is_24 :
  zoo_trip_remainder 5 1.5 40 = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_remainder_is_24_l4103_410331


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l4103_410368

theorem parallel_vectors_tan_theta (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (a : Fin 2 → Real)
  (b : Fin 2 → Real)
  (h_a : a = ![1 - Real.sin θ, 1])
  (h_b : b = ![1/2, 1 + Real.sin θ])
  (h_parallel : ∃ (k : Real), a = k • b) :
  Real.tan θ = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l4103_410368


namespace NUMINAMATH_CALUDE_abc_value_l4103_410366

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24 * Real.rpow 3 (1/4))
  (hac : a * c = 40 * Real.rpow 3 (1/4))
  (hbc : b * c = 15 * Real.rpow 3 (1/4)) :
  a * b * c = 120 * Real.rpow 3 (3/8) := by
sorry

end NUMINAMATH_CALUDE_abc_value_l4103_410366


namespace NUMINAMATH_CALUDE_relay_team_selection_l4103_410359

/-- The number of athletes in the track and field team -/
def total_athletes : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of twins -/
def num_twins : ℕ := 2

/-- The size of the relay team -/
def team_size : ℕ := 7

/-- The number of ways to choose the relay team -/
def num_ways : ℕ := 3762

theorem relay_team_selection :
  (num_triplets * (num_twins * (Nat.choose (total_athletes - num_triplets - 1 - 1) (team_size - 1 - 1)) +
  1 * (Nat.choose (total_athletes - num_triplets - 2) (team_size - 1 - 2)))) = num_ways :=
sorry

end NUMINAMATH_CALUDE_relay_team_selection_l4103_410359


namespace NUMINAMATH_CALUDE_subcommittee_count_l4103_410328

theorem subcommittee_count (n m k : ℕ) (hn : n = 12) (hm : m = 5) (hk : k = 5) :
  Nat.choose n k - Nat.choose (n - m) k = 771 :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l4103_410328


namespace NUMINAMATH_CALUDE_simplify_w_squared_series_l4103_410306

theorem simplify_w_squared_series (w : ℝ) : 
  3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_w_squared_series_l4103_410306


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l4103_410318

theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 4 < 0 ↔ 1 < x ∧ x < 4) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l4103_410318


namespace NUMINAMATH_CALUDE_square_of_difference_negative_first_l4103_410320

theorem square_of_difference_negative_first (x y : ℝ) : (-x + y)^2 = x^2 - 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_negative_first_l4103_410320


namespace NUMINAMATH_CALUDE_cruise_ship_luxury_suites_l4103_410326

/-- The number of women in luxury suites on a cruise ship -/
def women_in_luxury_suites (total_passengers : ℕ) (percent_women : ℚ) (percent_women_in_luxury : ℚ) : ℕ :=
  ⌊(total_passengers : ℚ) * percent_women * percent_women_in_luxury⌋₊

/-- Theorem: Given a cruise ship with 300 passengers, where 70% are women and 15% of women are in luxury suites, 
    the number of women in luxury suites is 32. -/
theorem cruise_ship_luxury_suites : 
  women_in_luxury_suites 300 (70/100) (15/100) = 32 := by
  sorry

end NUMINAMATH_CALUDE_cruise_ship_luxury_suites_l4103_410326


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l4103_410317

theorem inequality_solution_sets (a b x : ℝ) :
  let f := fun x => b * x^2 - (3 * a * b - b) * x + 2 * a^2 * b - a * b
  (∀ x, b = 1 ∧ a > 1 → (f x < 0 ↔ a < x ∧ x < 2 * a - 1)) ∧
  (∀ x, b = a ∧ a ≤ 1 → 
    ((a = 0 ∨ a = 1) → ¬∃ x, f x < 0) ∧
    (0 < a ∧ a < 1 → (f x < 0 ↔ 2 * a - 1 < x ∧ x < a)) ∧
    (a < 0 → (f x < 0 ↔ x < 2 * a - 1 ∨ x > a))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l4103_410317


namespace NUMINAMATH_CALUDE_smallest_valid_tournament_l4103_410392

/-- A tournament is valid if for any two players, there exists a third player who beat both of them -/
def is_valid_tournament (k : ℕ) (tournament : Fin k → Fin k → Bool) : Prop :=
  k > 1 ∧
  (∀ i j, i ≠ j → tournament i j = !tournament j i) ∧
  (∀ i j, i ≠ j → ∃ m, m ≠ i ∧ m ≠ j ∧ tournament m i ∧ tournament m j)

/-- The smallest k for which a valid tournament exists is 7 -/
theorem smallest_valid_tournament : 
  (∃ k : ℕ, ∃ tournament : Fin k → Fin k → Bool, is_valid_tournament k tournament) ∧
  (∀ k : ℕ, k < 7 → ¬∃ tournament : Fin k → Fin k → Bool, is_valid_tournament k tournament) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_tournament_l4103_410392


namespace NUMINAMATH_CALUDE_parallel_vectors_theta_l4103_410341

theorem parallel_vectors_theta (θ : Real) 
  (h1 : θ > 0) (h2 : θ < Real.pi / 2)
  (a : Fin 2 → Real) (b : Fin 2 → Real)
  (ha : a = ![3/2, Real.sin θ])
  (hb : b = ![Real.cos θ, 1/3])
  (h_parallel : ∃ (k : Real), a = k • b) :
  θ = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_theta_l4103_410341


namespace NUMINAMATH_CALUDE_thousand_pow_seven_div_ten_pow_seventeen_l4103_410301

theorem thousand_pow_seven_div_ten_pow_seventeen :
  (1000 : ℕ)^7 / (10 : ℕ)^17 = (10 : ℕ)^4 := by
  sorry

end NUMINAMATH_CALUDE_thousand_pow_seven_div_ten_pow_seventeen_l4103_410301


namespace NUMINAMATH_CALUDE_red_notes_per_row_l4103_410345

theorem red_notes_per_row (total_rows : ℕ) (total_notes : ℕ) (extra_blue : ℕ) :
  total_rows = 5 →
  total_notes = 100 →
  extra_blue = 10 →
  ∃ (red_per_row : ℕ),
    red_per_row = 6 ∧
    total_notes = total_rows * red_per_row + 2 * (total_rows * red_per_row) + extra_blue :=
by sorry

end NUMINAMATH_CALUDE_red_notes_per_row_l4103_410345


namespace NUMINAMATH_CALUDE_set_inclusion_l4103_410386

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem set_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_set_inclusion_l4103_410386


namespace NUMINAMATH_CALUDE_similar_triangles_height_l4103_410327

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 15 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l4103_410327


namespace NUMINAMATH_CALUDE_digit_sum_s_99_l4103_410334

/-- s(n) is the number formed by concatenating the first n perfect squares -/
def s (n : ℕ) : ℕ := sorry

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: The digit sum of s(99) is 4 -/
theorem digit_sum_s_99 : digitSum (s 99) = 4 := by sorry

end NUMINAMATH_CALUDE_digit_sum_s_99_l4103_410334


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l4103_410310

theorem concentric_circles_radii_difference 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : π * R^2 = 3 * π * r^2) : 
  ∃ ε > 0, |R - r - 0.73 * r| < ε := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l4103_410310


namespace NUMINAMATH_CALUDE_probability_score_3_points_l4103_410389

/-- The probability of hitting target A -/
def prob_hit_A : ℚ := 3/4

/-- The probability of hitting target B -/
def prob_hit_B : ℚ := 2/3

/-- The score for hitting target A -/
def score_hit_A : ℤ := 1

/-- The score for missing target A -/
def score_miss_A : ℤ := -1

/-- The score for hitting target B -/
def score_hit_B : ℤ := 2

/-- The score for missing target B -/
def score_miss_B : ℤ := 0

/-- The number of shots at target B -/
def shots_B : ℕ := 2

theorem probability_score_3_points : 
  (prob_hit_A * shots_B * prob_hit_B * (1 - prob_hit_B) + 
   (1 - prob_hit_A) * prob_hit_B^shots_B) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_score_3_points_l4103_410389


namespace NUMINAMATH_CALUDE_remainder_inequality_l4103_410325

theorem remainder_inequality (p : ℕ) (hp : p ≥ 5) (hp_prime : Nat.Prime p) : 
  let R : ℕ → ℕ := λ k => k % p
  let S := {a : ℕ | a > 0 ∧ ∀ m ∈ Finset.range (p - 1), m + 1 + R (m * a) > a}
  let T := {p - 1} ∪ {a | ∃ s ∈ Finset.range p, a = (p - 1) / s}
  S = T := by sorry

end NUMINAMATH_CALUDE_remainder_inequality_l4103_410325


namespace NUMINAMATH_CALUDE_percentage_difference_l4103_410378

theorem percentage_difference (x y : ℝ) (h : x = 11 * y) :
  (x - y) / x * 100 = 10 / 11 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l4103_410378


namespace NUMINAMATH_CALUDE_marathon_completion_time_l4103_410360

/-- The time to complete a marathon given the distance and average pace -/
theorem marathon_completion_time 
  (distance : ℕ) 
  (avg_pace : ℕ) 
  (h1 : distance = 24)  -- marathon distance in miles
  (h2 : avg_pace = 9)   -- average pace in minutes per mile
  : distance * avg_pace = 216 := by
  sorry

end NUMINAMATH_CALUDE_marathon_completion_time_l4103_410360


namespace NUMINAMATH_CALUDE_square_difference_l4103_410346

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end NUMINAMATH_CALUDE_square_difference_l4103_410346


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l4103_410358

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l4103_410358


namespace NUMINAMATH_CALUDE_lowest_price_after_discounts_l4103_410333

/-- Calculates the lowest possible price of a product after applying regular and sale discounts -/
theorem lowest_price_after_discounts 
  (msrp : ℝ)
  (max_regular_discount : ℝ)
  (sale_discount : ℝ)
  (h1 : msrp = 40)
  (h2 : max_regular_discount = 0.3)
  (h3 : sale_discount = 0.2)
  : ∃ (lowest_price : ℝ), lowest_price = 22.4 :=
by
  sorry

#check lowest_price_after_discounts

end NUMINAMATH_CALUDE_lowest_price_after_discounts_l4103_410333


namespace NUMINAMATH_CALUDE_root_equation_implies_sum_l4103_410330

theorem root_equation_implies_sum (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) - b = 0 → a - b + 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_sum_l4103_410330


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l4103_410319

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) := by
sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l4103_410319


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l4103_410339

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to (tc + b), then t = -24/17 -/
theorem parallel_vectors_t_value (a b c : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (-3, 4))
  (h2 : b = (-1, 5))
  (h3 : c = (2, 3))
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 - c.1, a.2 - c.2) = k • (t * c.1 + b.1, t * c.2 + b.2)) :
  t = -24/17 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l4103_410339


namespace NUMINAMATH_CALUDE_train_speed_l4103_410332

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 441) (h2 : time = 21) :
  length / time = 21 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4103_410332


namespace NUMINAMATH_CALUDE_journey_final_distance_l4103_410304

-- Define the directions
inductive Direction
| NorthEast
| SouthEast
| SouthWest
| NorthWest

-- Define a leg of the journey
structure Leg where
  distance : ℝ
  direction : Direction

-- Define the journey
def journey : List Leg := [
  { distance := 5, direction := Direction.NorthEast },
  { distance := 15, direction := Direction.SouthEast },
  { distance := 25, direction := Direction.SouthWest },
  { distance := 35, direction := Direction.NorthWest },
  { distance := 20, direction := Direction.NorthEast }
]

-- Function to calculate the final distance
def finalDistance (j : List Leg) : ℝ := sorry

-- Theorem stating that the final distance is 20 miles
theorem journey_final_distance : finalDistance journey = 20 := by sorry

end NUMINAMATH_CALUDE_journey_final_distance_l4103_410304


namespace NUMINAMATH_CALUDE_fraction_equality_l4103_410349

theorem fraction_equality (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 1 / 2) :
  e / a = 8 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4103_410349


namespace NUMINAMATH_CALUDE_parallelogram_area_l4103_410302

/-- The area of a parallelogram with base 12 cm and height 48 cm is 576 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 12 → 
    height = 48 → 
    area = base * height → 
    area = 576 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l4103_410302


namespace NUMINAMATH_CALUDE_moon_weight_calculation_l4103_410391

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The weight of Mars in tons -/
def mars_weight : ℝ := 2 * moon_weight

/-- The percentage of iron in the moon's composition -/
def iron_percentage : ℝ := 0.5

/-- The percentage of carbon in the moon's composition -/
def carbon_percentage : ℝ := 0.2

/-- The percentage of other elements in the moon's composition -/
def other_percentage : ℝ := 1 - iron_percentage - carbon_percentage

/-- The weight of other elements on Mars in tons -/
def mars_other_elements : ℝ := 150

theorem moon_weight_calculation :
  moon_weight = mars_other_elements / other_percentage / 2 := by sorry

end NUMINAMATH_CALUDE_moon_weight_calculation_l4103_410391


namespace NUMINAMATH_CALUDE_min_cards_for_two_of_each_suit_l4103_410396

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (jokers : ℕ)

/-- Defines the minimum number of cards to draw to ensure at least n cards of each suit -/
def min_cards_to_draw (d : Deck) (n : ℕ) : ℕ :=
  d.suits * (d.cards_per_suit - n + 1) + d.jokers + n - 1

/-- Theorem: The minimum number of cards to draw to ensure at least 2 cards of each suit is 43 -/
theorem min_cards_for_two_of_each_suit (d : Deck) 
  (h1 : d.total_cards = 54)
  (h2 : d.suits = 4)
  (h3 : d.cards_per_suit = 13)
  (h4 : d.jokers = 2) :
  min_cards_to_draw d 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_for_two_of_each_suit_l4103_410396


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l4103_410384

theorem quadratic_inequality_empty_solution (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + a * x + 2 ≥ 0) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l4103_410384


namespace NUMINAMATH_CALUDE_expanded_ohara_triple_64_49_l4103_410381

/-- Definition of an Expanded O'Hara triple -/
def is_expanded_ohara_triple (a b x : ℕ) : Prop :=
  2 * (Real.sqrt a + Real.sqrt b) = x

/-- Theorem: If (64, 49, x) is an Expanded O'Hara triple, then x = 30 -/
theorem expanded_ohara_triple_64_49 (x : ℕ) :
  is_expanded_ohara_triple 64 49 x → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_expanded_ohara_triple_64_49_l4103_410381


namespace NUMINAMATH_CALUDE_exists_right_triangles_form_consecutive_l4103_410309

/-- A right-angled triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  right_angle : a^2 + b^2 = c^2

/-- A triangle with consecutive natural number side lengths -/
structure ConsecutiveTriangle where
  n : ℕ
  sides : Fin 3 → ℕ
  consecutive : sides = fun i => 2*n + i.val - 1

theorem exists_right_triangles_form_consecutive (A : ℕ) :
  ∃ (t : ConsecutiveTriangle) (rt1 rt2 : RightTriangle),
    t.sides 0 = rt1.a + rt2.a ∧
    t.sides 1 = rt1.b ∧
    t.sides 1 = rt2.b ∧
    t.sides 2 = rt1.c + rt2.c ∧
    A = (t.sides 0 * t.sides 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangles_form_consecutive_l4103_410309


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l4103_410329

theorem complex_fraction_equals_neg_i : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l4103_410329


namespace NUMINAMATH_CALUDE_find_p_l4103_410335

/-- Given a system of equations with a known solution, prove the value of p. -/
theorem find_p (p q : ℝ) (h1 : p * 2 + q * (-4) = 8) (h2 : 3 * 2 - q * (-4) = 38) : p = 20 := by
  sorry

#check find_p

end NUMINAMATH_CALUDE_find_p_l4103_410335


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l4103_410385

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l4103_410385


namespace NUMINAMATH_CALUDE_odd_factors_360_l4103_410395

/-- The number of odd factors of 360 -/
def num_odd_factors_360 : ℕ := sorry

/-- Theorem: The number of odd factors of 360 is 6 -/
theorem odd_factors_360 : num_odd_factors_360 = 6 := by sorry

end NUMINAMATH_CALUDE_odd_factors_360_l4103_410395


namespace NUMINAMATH_CALUDE_pairings_of_six_items_l4103_410380

/-- The number of possible pairings between two sets of 6 distinct items -/
def num_pairings (n : ℕ) : ℕ := n * n

/-- Theorem: The number of possible pairings between two sets of 6 distinct items is 36 -/
theorem pairings_of_six_items :
  num_pairings 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pairings_of_six_items_l4103_410380


namespace NUMINAMATH_CALUDE_min_value_expression_l4103_410362

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  let M := Real.sqrt (1 + 2 * a^2) + 2 * Real.sqrt ((5/12)^2 + b^2)
  ∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 →
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) ≥ 5 * Real.sqrt 34 / 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4103_410362


namespace NUMINAMATH_CALUDE_annie_milkshakes_l4103_410347

/-- The number of milkshakes Annie bought -/
def milkshakes : ℕ := sorry

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 4

/-- The cost of a milkshake in dollars -/
def milkshake_cost : ℕ := 5

/-- The number of hamburgers Annie bought -/
def hamburgers_bought : ℕ := 8

/-- Annie's initial amount of money in dollars -/
def initial_money : ℕ := 132

/-- Annie's remaining money after purchases in dollars -/
def remaining_money : ℕ := 70

theorem annie_milkshakes :
  milkshakes = 6 ∧
  initial_money = remaining_money + hamburgers_bought * hamburger_cost + milkshakes * milkshake_cost :=
by sorry

end NUMINAMATH_CALUDE_annie_milkshakes_l4103_410347


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_15120_l4103_410387

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_15120 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ digit_product n = 15120 →
    n ≤ 98754 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_15120_l4103_410387


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l4103_410372

theorem angle_sum_in_circle (x : ℝ) : 
  (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l4103_410372


namespace NUMINAMATH_CALUDE_cube_volume_ratio_and_surface_area_l4103_410390

/-- Edge length of the smaller cube in inches -/
def small_cube_edge : ℝ := 4

/-- Edge length of the larger cube in feet -/
def large_cube_edge : ℝ := 2

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Volume of a cube given its edge length -/
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

/-- Surface area of a cube given its edge length -/
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge ^ 2

theorem cube_volume_ratio_and_surface_area :
  (cube_volume small_cube_edge) / (cube_volume (large_cube_edge * feet_to_inches)) = 1 / 216 ∧
  cube_surface_area small_cube_edge = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_and_surface_area_l4103_410390


namespace NUMINAMATH_CALUDE_time_to_mow_one_line_l4103_410313

def total_lines : ℕ := 40
def total_flowers : ℕ := 56
def time_per_flower : ℚ := 1/2
def total_gardening_time : ℕ := 108

theorem time_to_mow_one_line :
  (total_gardening_time - total_flowers * time_per_flower) / total_lines = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_to_mow_one_line_l4103_410313


namespace NUMINAMATH_CALUDE_bucket_capacity_ratio_l4103_410323

/-- Given two buckets P and Q, prove that their capacity ratio is 3:1 based on their filling times. -/
theorem bucket_capacity_ratio (P Q : ℝ) : 
  (P > 0) →  -- Bucket P has positive capacity
  (Q > 0) →  -- Bucket Q has positive capacity
  (60 * P = 45 * (P + Q)) →  -- Filling condition
  (P / Q = 3) :=  -- Ratio of capacities
by sorry

end NUMINAMATH_CALUDE_bucket_capacity_ratio_l4103_410323


namespace NUMINAMATH_CALUDE_equal_volume_equivalent_by_decomposition_l4103_410340

/-- A type representing geometric shapes (either rectangular parallelepipeds or prisms) -/
structure GeometricShape where
  volume : ℝ

/-- A type representing a decomposition of a geometric shape -/
structure Decomposition (α : Type) where
  parts : List α

/-- A function that checks if two decompositions are equivalent -/
def equivalent_decompositions {α : Type} (d1 d2 : Decomposition α) : Prop :=
  sorry

/-- A function that transforms one shape into another using a decomposition -/
def transform (s1 s2 : GeometricShape) (d : Decomposition GeometricShape) : Prop :=
  sorry

/-- The main theorem stating that equal-volume shapes are equivalent by decomposition -/
theorem equal_volume_equivalent_by_decomposition (s1 s2 : GeometricShape) :
  s1.volume = s2.volume →
  ∃ (d : Decomposition GeometricShape), transform s1 s2 d ∧ transform s2 s1 d :=
sorry

end NUMINAMATH_CALUDE_equal_volume_equivalent_by_decomposition_l4103_410340


namespace NUMINAMATH_CALUDE_odd_painted_faces_5x5x1_l4103_410364

/-- Represents a 3D grid of unit cubes -/
structure CubeGrid :=
  (length : Nat)
  (width : Nat)
  (height : Nat)

/-- Counts the number of cubes with an odd number of painted faces in a given grid -/
def countOddPaintedFaces (grid : CubeGrid) : Nat :=
  sorry

/-- The main theorem stating that a 5x5x1 grid has 9 cubes with an odd number of painted faces -/
theorem odd_painted_faces_5x5x1 :
  let grid := CubeGrid.mk 5 5 1
  countOddPaintedFaces grid = 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_painted_faces_5x5x1_l4103_410364


namespace NUMINAMATH_CALUDE_maria_carrots_l4103_410393

def total_carrots (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

theorem maria_carrots : total_carrots 48 11 15 = 52 := by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_l4103_410393


namespace NUMINAMATH_CALUDE_sad_probability_value_l4103_410307

/-- Represents a person in the company -/
inductive Person : Type
| boy : Fin 3 → Person
| girl : Fin 3 → Person

/-- Represents the love relation between people -/
def loves : Person → Person → Prop := sorry

/-- The sad circumstance where no one is loved by the one they love -/
def sad_circumstance (loves : Person → Person → Prop) : Prop :=
  ∀ p : Person, ∃ q : Person, loves p q ∧ ¬loves q p

/-- The total number of possible love arrangements -/
def total_arrangements : ℕ := 729

/-- The number of sad arrangements -/
def sad_arrangements : ℕ := 156

/-- The probability of the sad circumstance -/
def sad_probability : ℚ := sad_arrangements / total_arrangements

theorem sad_probability_value : sad_probability = 156 / 729 :=
sorry

end NUMINAMATH_CALUDE_sad_probability_value_l4103_410307


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l4103_410354

def set_A : Set ℝ := {x | x - 2 > 0}
def set_B : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem union_of_A_and_B :
  set_A ∪ set_B = Set.Ici (1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l4103_410354


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_relation_l4103_410348

theorem cubic_expansion_coefficient_relation :
  ∀ (a₀ a₁ a₂ a₃ : ℝ),
  (∀ x : ℝ, (2*x + Real.sqrt 3)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_relation_l4103_410348


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l4103_410300

theorem sqrt_x_plus_inverse (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l4103_410300


namespace NUMINAMATH_CALUDE_siblings_ages_sum_l4103_410388

theorem siblings_ages_sum (x y z : ℕ+) 
  (h1 : y = x + 1)
  (h2 : x * y * z = 96) :
  x + y + z = 15 := by
sorry

end NUMINAMATH_CALUDE_siblings_ages_sum_l4103_410388


namespace NUMINAMATH_CALUDE_eugene_model_house_l4103_410351

/-- The number of toothpicks Eugene uses for each card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in a deck -/
def cards_in_deck : ℕ := 52

/-- The number of cards Eugene didn't use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in each box -/
def toothpicks_per_box : ℕ := 450

/-- The number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ := 6

theorem eugene_model_house :
  (cards_in_deck - unused_cards) * toothpicks_per_card / toothpicks_per_box = boxes_used :=
sorry

end NUMINAMATH_CALUDE_eugene_model_house_l4103_410351


namespace NUMINAMATH_CALUDE_smallest_set_with_both_progressions_l4103_410312

/-- A sequence of integers forms a geometric progression of length 5 -/
def IsGeometricProgression (s : Finset ℤ) : Prop :=
  ∃ (a q : ℤ), q ≠ 0 ∧ s = {a, a*q, a*q^2, a*q^3, a*q^4}

/-- A sequence of integers forms an arithmetic progression of length 5 -/
def IsArithmeticProgression (s : Finset ℤ) : Prop :=
  ∃ (a d : ℤ), s = {a, a+d, a+2*d, a+3*d, a+4*d}

/-- The main theorem stating the smallest number of distinct integers -/
theorem smallest_set_with_both_progressions :
  ∀ (s : Finset ℤ), (∃ (s1 s2 : Finset ℤ), s1 ⊆ s ∧ s2 ⊆ s ∧ 
    IsGeometricProgression s1 ∧ IsArithmeticProgression s2) →
  s.card ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_set_with_both_progressions_l4103_410312


namespace NUMINAMATH_CALUDE_locus_of_concyclic_points_l4103_410337

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the special points of the triangle
def H : ℝ × ℝ := sorry  -- Orthocenter
def I : ℝ × ℝ := sorry  -- Incenter
def G : ℝ × ℝ := sorry  -- Centroid

-- Define points E and F that divide AB into three equal parts
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Define the angle at vertex C
def angle_C : ℝ := sorry

-- Define a predicate for points being concyclic
def are_concyclic (p q r s : ℝ × ℝ) : Prop := sorry

-- Define a predicate for a point being on a circular arc
def on_circular_arc (p center : ℝ × ℝ) (start_point end_point : ℝ × ℝ) (arc_angle : ℝ) : Prop := sorry

theorem locus_of_concyclic_points :
  are_concyclic A B H I →
  (angle_C = 60) ∧ 
  (∃ (center : ℝ × ℝ), on_circular_arc G center E F 120) :=
sorry

end NUMINAMATH_CALUDE_locus_of_concyclic_points_l4103_410337


namespace NUMINAMATH_CALUDE_max_cut_trees_100x100_l4103_410350

/-- Represents a square grid of trees -/
structure TreeGrid where
  size : ℕ
  trees : Fin size → Fin size → Bool

/-- Checks if a tree can be cut without making any other cut tree visible -/
def canCutTree (grid : TreeGrid) (x y : Fin grid.size) : Bool := sorry

/-- Counts the maximum number of trees that can be cut in a grid -/
def maxCutTrees (grid : TreeGrid) : ℕ := sorry

/-- Theorem: In a 100x100 grid, the maximum number of trees that can be cut
    while ensuring no stump is visible from any other stump is 2500 -/
theorem max_cut_trees_100x100 :
  ∀ (grid : TreeGrid), grid.size = 100 → maxCutTrees grid = 2500 := by sorry

end NUMINAMATH_CALUDE_max_cut_trees_100x100_l4103_410350


namespace NUMINAMATH_CALUDE_polynomial_properties_l4103_410338

/-- The polynomial -8x³y^(m+1) + xy² - 3/4x³ + 6y is a sixth-degree quadrinomial -/
def is_sixth_degree_quadrinomial (m : ℕ) : Prop :=
  3 + (m + 1) = 6

/-- The monomial 2/5πx^ny^(5-m) has the same degree as the polynomial -/
def monomial_same_degree (m n : ℕ) : Prop :=
  n + (5 - m) = 6

/-- The polynomial coefficients sum to -7/4 -/
def coefficients_sum : ℚ :=
  -8 + 1 + (-3/4) + 6

theorem polynomial_properties :
  ∃ (m n : ℕ),
    is_sixth_degree_quadrinomial m ∧
    monomial_same_degree m n ∧
    m = 2 ∧
    n = 3 ∧
    coefficients_sum = -7/4 := by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l4103_410338


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l4103_410383

def U : Finset ℕ := {1,2,3,4,5,6}
def M : Finset ℕ := {2,3,5}
def N : Finset ℕ := {4,5}

theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l4103_410383


namespace NUMINAMATH_CALUDE_wardrobe_probability_l4103_410375

def num_shirts : ℕ := 5
def num_shorts : ℕ := 6
def num_socks : ℕ := 7
def num_selected : ℕ := 4

def total_articles : ℕ := num_shirts + num_shorts + num_socks

theorem wardrobe_probability :
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 1 * Nat.choose num_socks 1) /
  (Nat.choose total_articles num_selected) = 7 / 51 :=
by sorry

end NUMINAMATH_CALUDE_wardrobe_probability_l4103_410375


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4103_410373

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_common_difference :
  ∀ (d : ℚ), 
    (arithmetic_sequence 1 d 0 = 1) →
    (sum_arithmetic_sequence 1 d 5 = 20) →
    d = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4103_410373


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_l4103_410397

theorem product_and_reciprocal_relation (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 16 → 1 / x = 3 * (1 / y) → |x - y| = (8 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_l4103_410397


namespace NUMINAMATH_CALUDE_inequality_proof_l4103_410343

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4103_410343


namespace NUMINAMATH_CALUDE_order_of_abc_l4103_410399

/-- Given a = 0.1e^(0.1), b = 1/9, and c = -ln(0.9), prove that c < a < b -/
theorem order_of_abc (a b c : ℝ) (ha : a = 0.1 * Real.exp 0.1) (hb : b = 1/9) (hc : c = -Real.log 0.9) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l4103_410399


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l4103_410377

open Set

theorem fraction_inequality_solution (x : ℝ) :
  (x - 5) / ((x - 3)^2) < 0 ↔ x ∈ Iio 3 ∪ Ioo 3 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l4103_410377


namespace NUMINAMATH_CALUDE_function_inequality_l4103_410376

open Set

-- Define the interval [a, b]
variable (a b : ℝ) (hab : a < b)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State that f and g are differentiable on [a, b]
variable (hf : DifferentiableOn ℝ f (Icc a b))
variable (hg : DifferentiableOn ℝ g (Icc a b))

-- State that f'(x) < g'(x) for all x in [a, b]
variable (h_deriv : ∀ x ∈ Icc a b, deriv f x < deriv g x)

-- State the theorem
theorem function_inequality (x : ℝ) (hx : a < x ∧ x < b) :
  f x + g a < g x + f a :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l4103_410376


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l4103_410352

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l4103_410352


namespace NUMINAMATH_CALUDE_tanner_savings_l4103_410369

theorem tanner_savings (september_savings : ℤ) : 
  september_savings + 48 + 25 - 49 = 41 → september_savings = 17 := by
  sorry

end NUMINAMATH_CALUDE_tanner_savings_l4103_410369


namespace NUMINAMATH_CALUDE_unique_plants_count_l4103_410314

/-- Represents a flower bed -/
structure FlowerBed where
  plants : ℕ

/-- Represents the overlap between two flower beds -/
structure Overlap where
  plants : ℕ

/-- Represents the overlap among three flower beds -/
structure TripleOverlap where
  plants : ℕ

/-- Calculates the total number of unique plants across three overlapping flower beds -/
def totalUniquePlants (a b c : FlowerBed) (ab ac bc : Overlap) (abc : TripleOverlap) : ℕ :=
  a.plants + b.plants + c.plants - ab.plants - ac.plants - bc.plants + abc.plants

/-- Theorem stating that the total number of unique plants across three specific overlapping flower beds is 1320 -/
theorem unique_plants_count :
  let a : FlowerBed := ⟨600⟩
  let b : FlowerBed := ⟨550⟩
  let c : FlowerBed := ⟨400⟩
  let ab : Overlap := ⟨60⟩
  let ac : Overlap := ⟨110⟩
  let bc : Overlap := ⟨90⟩
  let abc : TripleOverlap := ⟨30⟩
  totalUniquePlants a b c ab ac bc abc = 1320 := by
  sorry

end NUMINAMATH_CALUDE_unique_plants_count_l4103_410314


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_68_l4103_410355

-- Define the function f
def f (a : ℝ) : ℝ := a + 3

-- Define the function F
def F (a b : ℝ) : ℝ := b^2 + a

-- Theorem to prove
theorem F_of_4_f_of_5_equals_68 : F 4 (f 5) = 68 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_68_l4103_410355


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l4103_410394

theorem complete_square_quadratic (x : ℝ) : 
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l4103_410394


namespace NUMINAMATH_CALUDE_triangle_ratio_l4103_410379

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = Real.sqrt 13 →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l4103_410379


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l4103_410361

theorem probability_triangle_or_circle :
  let total_figures : ℕ := 10
  let triangle_count : ℕ := 4
  let circle_count : ℕ := 3
  let target_count : ℕ := triangle_count + circle_count
  (target_count : ℚ) / total_figures = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l4103_410361


namespace NUMINAMATH_CALUDE_josie_animal_count_l4103_410322

/-- The number of antelopes Josie counted -/
def num_antelopes : ℕ := 80

/-- The number of rabbits Josie counted -/
def num_rabbits : ℕ := num_antelopes + 34

/-- The number of hyenas Josie counted -/
def num_hyenas : ℕ := num_antelopes + num_rabbits - 42

/-- The number of wild dogs Josie counted -/
def num_wild_dogs : ℕ := num_hyenas + 50

/-- The number of leopards Josie counted -/
def num_leopards : ℕ := num_rabbits / 2

/-- The total number of animals Josie counted -/
def total_animals : ℕ := num_antelopes + num_rabbits + num_hyenas + num_wild_dogs + num_leopards

theorem josie_animal_count : total_animals = 605 := by
  sorry

end NUMINAMATH_CALUDE_josie_animal_count_l4103_410322


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l4103_410303

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r and R are real numbers representing radii
  (h_positive : r > 0) -- r is positive
  (h_ratio : π * R^2 = 4 * π * r^2) -- area ratio is 1:4
  : R - r = r := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l4103_410303


namespace NUMINAMATH_CALUDE_square_with_quarter_circles_area_l4103_410398

theorem square_with_quarter_circles_area (π : Real) : 
  let square_side : Real := 4
  let quarter_circle_radius : Real := square_side / 2
  let square_area : Real := square_side ^ 2
  let quarter_circle_area : Real := π * quarter_circle_radius ^ 2 / 4
  let total_quarter_circles_area : Real := 4 * quarter_circle_area
  square_area - total_quarter_circles_area = 16 - 4 * π := by sorry

end NUMINAMATH_CALUDE_square_with_quarter_circles_area_l4103_410398


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l4103_410315

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l4103_410315


namespace NUMINAMATH_CALUDE_eriks_mother_money_l4103_410371

/-- The amount of money Erik's mother gave him. -/
def money_from_mother : ℕ := sorry

/-- The number of loaves of bread Erik bought. -/
def bread_loaves : ℕ := 3

/-- The number of cartons of orange juice Erik bought. -/
def juice_cartons : ℕ := 3

/-- The cost of one loaf of bread in dollars. -/
def bread_cost : ℕ := 3

/-- The cost of one carton of orange juice in dollars. -/
def juice_cost : ℕ := 6

/-- The amount of money Erik has left in dollars. -/
def money_left : ℕ := 59

/-- Theorem stating that the amount of money Erik's mother gave him is $86. -/
theorem eriks_mother_money : money_from_mother = 86 := by sorry

end NUMINAMATH_CALUDE_eriks_mother_money_l4103_410371


namespace NUMINAMATH_CALUDE_geometric_mean_combined_sets_l4103_410357

theorem geometric_mean_combined_sets :
  ∀ (y₁ y₂ y₃ y₄ z₁ z₂ z₃ z₄ : ℝ),
    y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ y₄ > 0 ∧
    z₁ > 0 ∧ z₂ > 0 ∧ z₃ > 0 ∧ z₄ > 0 →
    (y₁ * y₂ * y₃ * y₄) ^ (1/4 : ℝ) = 2048 →
    (z₁ * z₂ * z₃ * z₄) ^ (1/4 : ℝ) = 8 →
    (y₁ * y₂ * y₃ * y₄ * z₁ * z₂ * z₃ * z₄) ^ (1/8 : ℝ) = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_combined_sets_l4103_410357


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1260_l4103_410305

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

theorem sum_of_extreme_prime_factors_of_1260 :
  ∃ (min max : ℕ),
    is_prime_factor min 1260 ∧
    is_prime_factor max 1260 ∧
    (∀ p, is_prime_factor p 1260 → min ≤ p) ∧
    (∀ p, is_prime_factor p 1260 → p ≤ max) ∧
    min + max = 9 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1260_l4103_410305


namespace NUMINAMATH_CALUDE_coins_missing_fraction_l4103_410344

theorem coins_missing_fraction (initial_coins : ℚ) : 
  initial_coins > 0 →
  let lost_coins := (1 / 3 : ℚ) * initial_coins
  let found_coins := (2 / 3 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = (1 / 9 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_coins_missing_fraction_l4103_410344


namespace NUMINAMATH_CALUDE_solution_set_of_equations_l4103_410336

theorem solution_set_of_equations (x y z : ℝ) : 
  (3 * (x^2 + y^2 + z^2) = 1 ∧ 
   x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x+y+z)^2) ↔ 
  ((x = 0 ∧ y = 0 ∧ z = Real.sqrt 3 / 3) ∨
   (x = 0 ∧ y = 0 ∧ z = -Real.sqrt 3 / 3) ∨
   (x = 0 ∧ y = Real.sqrt 3 / 3 ∧ z = 0) ∨
   (x = 0 ∧ y = -Real.sqrt 3 / 3 ∧ z = 0) ∨
   (x = Real.sqrt 3 / 3 ∧ y = 0 ∧ z = 0) ∨
   (x = -Real.sqrt 3 / 3 ∧ y = 0 ∧ z = 0) ∨
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨
   (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_equations_l4103_410336


namespace NUMINAMATH_CALUDE_probability_theorem_l4103_410308

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Finset (Fin 5))
  num_faces : faces.card = 12

/-- Three distinct vertices of a regular dodecahedron -/
def ThreeVertices (d : RegularDodecahedron) := Finset (Fin 3)

/-- The probability that a plane determined by three randomly chosen distinct
    vertices of a regular dodecahedron contains points inside the dodecahedron -/
def probability_plane_intersects_interior (d : RegularDodecahedron) : ℚ :=
  1 - 1 / 9.5

/-- Theorem stating the probability of a plane determined by three randomly chosen
    distinct vertices of a regular dodecahedron containing points inside the dodecahedron -/
theorem probability_theorem (d : RegularDodecahedron) :
  probability_plane_intersects_interior d = 1 - 1 / 9.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l4103_410308


namespace NUMINAMATH_CALUDE_triangle_properties_l4103_410367

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A ≠ π / 2)
  (h2 : 3 * Real.sin t.A * Real.cos t.B + (1/2) * t.b * Real.sin (2 * t.A) = 3 * Real.sin t.C) :
  (t.a = 3) ∧ 
  (t.A = 2 * π / 3 → 
    ∃ (max_perimeter : Real), max_perimeter = 3 + 2 * Real.sqrt 3 ∧
    ∀ (perimeter : Real), perimeter = t.a + t.b + t.c → perimeter ≤ max_perimeter) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4103_410367
