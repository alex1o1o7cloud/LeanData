import Mathlib

namespace NUMINAMATH_CALUDE_lucy_disproves_tom_l1733_173346

-- Define the visible sides of the cards
def visible_numbers : List ℕ := [2, 4, 5, 7]
def visible_letters : List Char := ['B', 'C', 'D']

-- Define primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define consonant
def is_consonant (c : Char) : Prop := c ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

-- Tom's claim
def toms_claim (n : ℕ) (c : Char) : Prop := is_prime n → is_consonant c

-- Lucy's action
def lucy_flips_5 : Prop := ∃ (c : Char), c ∉ visible_letters ∧ ¬(is_consonant c)

-- Theorem to prove
theorem lucy_disproves_tom : 
  (∀ n ∈ visible_numbers, is_prime n → n ≠ 5 → ∃ c ∈ visible_letters, toms_claim n c) →
  lucy_flips_5 →
  ¬(∀ n c, toms_claim n c) :=
by sorry

end NUMINAMATH_CALUDE_lucy_disproves_tom_l1733_173346


namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l1733_173392

/-- Represents an alcohol solution with a volume and concentration -/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two alcohol solutions -/
def AlcoholMixture (s1 s2 : AlcoholSolution) (finalConcentration : ℝ) :=
  s1.volume * s1.concentration + s2.volume * s2.concentration = 
    (s1.volume + s2.volume) * finalConcentration

theorem alcohol_mixture_concentration 
  (s1 s2 : AlcoholSolution) (finalConcentration : ℝ) :
  s1.volume = 75 →
  s2.volume = 125 →
  s2.concentration = 0.12 →
  finalConcentration = 0.15 →
  AlcoholMixture s1 s2 finalConcentration →
  s1.concentration = 0.20 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l1733_173392


namespace NUMINAMATH_CALUDE_stream_speed_l1733_173383

/-- Proves that the speed of a stream is 20 km/h given the conditions of the rowing problem -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) 
  (h1 : boat_speed = 60)
  (h2 : upstream_time = 2 * downstream_time)
  (h3 : upstream_time > 0)
  (h4 : downstream_time > 0) :
  let stream_speed := (boat_speed - boat_speed * downstream_time / upstream_time) / 2
  stream_speed = 20 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l1733_173383


namespace NUMINAMATH_CALUDE_number_difference_l1733_173361

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : |x - y| = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1733_173361


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l1733_173336

/-- Given a point A and a line l, find the equations of lines passing through A
    that are parallel and perpendicular to l. -/
theorem parallel_perpendicular_lines
  (A : ℝ × ℝ)
  (l : ℝ → ℝ → Prop)
  (h_A : A = (2, 2))
  (h_l : l = fun x y ↦ 3 * x + 4 * y - 20 = 0) :
  ∃ (l_parallel l_perpendicular : ℝ → ℝ → Prop),
    (∀ x y, l_parallel x y ↔ 3 * x + 4 * y - 14 = 0) ∧
    (∀ x y, l_perpendicular x y ↔ 4 * x - 3 * y - 2 = 0) ∧
    (∀ x y, l_parallel x y → l_parallel A.1 A.2) ∧
    (∀ x y, l_perpendicular x y → l_perpendicular A.1 A.2) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → (y₂ - y₁) * 4 = (x₂ - x₁) * 3) ∧
    (∀ x₁ y₁ x₂ y₂, l_parallel x₁ y₁ → l_parallel x₂ y₂ → (y₂ - y₁) * 4 = (x₂ - x₁) * 3) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l_perpendicular x₂ y₂ → (y₂ - y₁) * 3 = -(x₂ - x₁) * 4) :=
by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l1733_173336


namespace NUMINAMATH_CALUDE_f_has_root_iff_f_ln_b_gt_inv_b_l1733_173316

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1: f has a root iff 0 < a ≤ 1/e
theorem f_has_root_iff (a : ℝ) (h : a > 0) :
  (∃ x > 0, f a x = 0) ↔ a ≤ (Real.exp 1)⁻¹ :=
sorry

-- Theorem 2: When a ≥ 2/e and b > 1, f(ln b) > 1/b
theorem f_ln_b_gt_inv_b (a b : ℝ) (ha : a ≥ 2 / Real.exp 1) (hb : b > 1) :
  f a (Real.log b) > b⁻¹ :=
sorry

end NUMINAMATH_CALUDE_f_has_root_iff_f_ln_b_gt_inv_b_l1733_173316


namespace NUMINAMATH_CALUDE_total_chips_calculation_l1733_173354

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ) : ℕ :=
  viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla

/-- Theorem stating the total number of chips Viviana and Susana have together -/
theorem total_chips_calculation :
  ∀ (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ),
  viviana_chocolate = susana_chocolate + 5 →
  susana_vanilla = (3 * viviana_vanilla) / 4 →
  viviana_vanilla = 20 →
  susana_chocolate = 25 →
  total_chips viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla = 90 :=
by sorry

end NUMINAMATH_CALUDE_total_chips_calculation_l1733_173354


namespace NUMINAMATH_CALUDE_platform_length_l1733_173306

/-- Calculates the length of a platform given train specifications -/
theorem platform_length 
  (train_length : ℝ) 
  (time_tree : ℝ) 
  (time_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 210) : 
  ∃ platform_length : ℝ, platform_length = 900 ∧ 
  time_platform = (train_length + platform_length) / (train_length / time_tree) :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1733_173306


namespace NUMINAMATH_CALUDE_tangent_slope_and_extrema_l1733_173312

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem tangent_slope_and_extrema (a : ℝ) :
  (deriv (f a) 0 = 1) →
  (a = 1) ∧
  (∀ x ∈ Set.Icc 0 2, f 1 0 ≤ f 1 x) ∧
  (∀ x ∈ Set.Icc 0 2, f 1 x ≤ f 1 2) ∧
  (f 1 0 = 0) ∧
  (f 1 2 = 2 * Real.exp 2) :=
by sorry

end

end NUMINAMATH_CALUDE_tangent_slope_and_extrema_l1733_173312


namespace NUMINAMATH_CALUDE_consecutive_integers_equation_l1733_173343

theorem consecutive_integers_equation (x y z : ℤ) : 
  (y = x - 1) →
  (z = x - 2) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 11) →
  z = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_equation_l1733_173343


namespace NUMINAMATH_CALUDE_rex_to_total_ratio_l1733_173304

/-- Represents the number of Pokemon cards collected by each person -/
structure PokemonCards where
  nicole : ℕ
  cindy : ℕ
  rex : ℕ

/-- Represents the problem statement and conditions -/
def pokemon_card_problem (cards : PokemonCards) : Prop :=
  cards.nicole = 400 ∧
  cards.cindy = 2 * cards.nicole ∧
  cards.rex = 150 * 4 ∧
  cards.rex < cards.nicole + cards.cindy

/-- Theorem stating the ratio of Rex's cards to Nicole and Cindy's combined total -/
theorem rex_to_total_ratio (cards : PokemonCards) 
  (h : pokemon_card_problem cards) : 
  (cards.rex : ℚ) / (cards.nicole + cards.cindy : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rex_to_total_ratio_l1733_173304


namespace NUMINAMATH_CALUDE_fourth_person_height_l1733_173338

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ = h₁ + 2 →                 -- difference between 1st and 2nd
  h₃ = h₂ + 2 →                 -- difference between 2nd and 3rd
  h₄ = h₃ + 6 →                 -- difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 77  -- average height
  → h₄ = 83 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1733_173338


namespace NUMINAMATH_CALUDE_sum_of_7th_and_11th_terms_l1733_173324

/-- An arithmetic sequence {a_n} with the sum of its first 17 terms equal to 51 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), (∀ n, a n = a₁ + (n - 1) * d) ∧
  (a 1 + a 17) * 17 / 2 = 51

/-- Theorem: In an arithmetic sequence {a_n} where the sum of the first 17 terms is 51,
    the sum of the 7th and 11th terms is 6 -/
theorem sum_of_7th_and_11th_terms
  (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 7 + a 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_7th_and_11th_terms_l1733_173324


namespace NUMINAMATH_CALUDE_mobile_phone_purchase_price_l1733_173381

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The loss percentage on the refrigerator sale -/
def refrigerator_loss_percent : ℝ := 0.05

/-- The profit percentage on the mobile phone sale -/
def mobile_profit_percent : ℝ := 0.10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 50

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

theorem mobile_phone_purchase_price :
  ∃ (x : ℝ),
    x = mobile_price ∧
    refrigerator_price * (1 - refrigerator_loss_percent) +
    x * (1 + mobile_profit_percent) =
    refrigerator_price + x + overall_profit :=
by sorry

end NUMINAMATH_CALUDE_mobile_phone_purchase_price_l1733_173381


namespace NUMINAMATH_CALUDE_gamma_max_success_ratio_l1733_173389

theorem gamma_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (gamma_day1_score gamma_day1_total : ℕ)
  (gamma_day2_score gamma_day2_total : ℕ)
  (h1 : alpha_day1_score = 170)
  (h2 : alpha_day1_total = 280)
  (h3 : alpha_day2_score = 150)
  (h4 : alpha_day2_total = 220)
  (h5 : gamma_day1_total < alpha_day1_total)
  (h6 : gamma_day1_score > 0)
  (h7 : gamma_day2_score > 0)
  (h8 : (gamma_day1_score : ℚ) / gamma_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total)
  (h9 : (gamma_day2_score : ℚ) / gamma_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total)
  (h10 : gamma_day1_total + gamma_day2_total = 500)
  (h11 : (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 320 / 500) :
  (gamma_day1_score + gamma_day2_score : ℚ) / 500 ≤ 170 / 500 :=
by sorry

end NUMINAMATH_CALUDE_gamma_max_success_ratio_l1733_173389


namespace NUMINAMATH_CALUDE_households_with_car_l1733_173373

theorem households_with_car (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 14)
  (h4 : bike_only = 35) :
  ∃ (car : ℕ), car = 44 ∧ 
    car + bike_only + both + neither = total ∧
    car + bike_only + neither = total - both :=
by
  sorry

#check households_with_car

end NUMINAMATH_CALUDE_households_with_car_l1733_173373


namespace NUMINAMATH_CALUDE_shooter_probability_l1733_173394

theorem shooter_probability (p : ℝ) (h : p = 2/3) :
  1 - (1 - p)^3 = 26/27 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l1733_173394


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l1733_173314

theorem same_color_plate_probability :
  let total_plates : ℕ := 12
  let red_plates : ℕ := 7
  let blue_plates : ℕ := 5
  let selected_plates : ℕ := 3
  let total_combinations := Nat.choose total_plates selected_plates
  let red_combinations := Nat.choose red_plates selected_plates
  let blue_combinations := Nat.choose blue_plates selected_plates
  (red_combinations + blue_combinations : ℚ) / total_combinations = 9 / 44 :=
by sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l1733_173314


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1733_173315

theorem complex_arithmetic_equality : (-1 : ℚ)^2023 + (6 - 5/4) * 4/3 + 4 / (-2/3) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1733_173315


namespace NUMINAMATH_CALUDE_negation_existential_to_universal_l1733_173340

theorem negation_existential_to_universal :
  (¬ ∃ x : ℝ, 2 * x - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_to_universal_l1733_173340


namespace NUMINAMATH_CALUDE_allan_initial_balloons_l1733_173397

theorem allan_initial_balloons :
  ∀ (allan_initial jake_balloons allan_bought total : ℕ),
    jake_balloons = 5 →
    allan_bought = 2 →
    total = 10 →
    allan_initial + allan_bought + jake_balloons = total →
    allan_initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_allan_initial_balloons_l1733_173397


namespace NUMINAMATH_CALUDE_cookie_bags_count_l1733_173376

theorem cookie_bags_count (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_count_l1733_173376


namespace NUMINAMATH_CALUDE_range_of_a_l1733_173341

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + Real.exp (-x) - a

def range_subset (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x a ≥ 0

theorem range_of_a (a : ℝ) :
  (range_subset f a) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1733_173341


namespace NUMINAMATH_CALUDE_max_four_digit_divisible_by_36_11_l1733_173344

def digit_reverse (n : Nat) : Nat :=
  -- Implementation of digit reversal (not provided)
  sorry

theorem max_four_digit_divisible_by_36_11 :
  ∃ (m : Nat),
    1000 ≤ m ∧ m ≤ 9999 ∧
    m % 36 = 0 ∧
    (digit_reverse m) % 36 = 0 ∧
    m % 11 = 0 ∧
    ∀ (k : Nat), 1000 ≤ k ∧ k ≤ 9999 ∧
      k % 36 = 0 ∧ (digit_reverse k) % 36 = 0 ∧ k % 11 = 0 →
      k ≤ m ∧
    m = 9504 :=
by
  sorry

end NUMINAMATH_CALUDE_max_four_digit_divisible_by_36_11_l1733_173344


namespace NUMINAMATH_CALUDE_candy_expenditure_l1733_173378

theorem candy_expenditure (initial : ℕ) (oranges apples left : ℕ) 
  (h1 : initial = 95)
  (h2 : oranges = 14)
  (h3 : apples = 25)
  (h4 : left = 50) :
  initial - (oranges + apples) - left = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_expenditure_l1733_173378


namespace NUMINAMATH_CALUDE_equations_solutions_l1733_173393

-- Define the equations
def equation1 (x : ℝ) : Prop :=
  (x - 3) / (x - 2) + 1 = 3 / (2 - x)

def equation2 (x : ℝ) : Prop :=
  (x - 2) / (x + 2) - (x + 2) / (x - 2) = 16 / (x^2 - 4)

-- Theorem statement
theorem equations_solutions :
  equation1 1 ∧ equation2 (-4) :=
by sorry

end NUMINAMATH_CALUDE_equations_solutions_l1733_173393


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1733_173333

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1733_173333


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l1733_173331

/-- Calculates the remaining money after Amanda's purchases -/
def remaining_money (gift_amount : ℚ) (tape_price : ℚ) (num_tapes : ℕ) 
  (headphone_price : ℚ) (vinyl_price : ℚ) (poster_price : ℚ) 
  (tape_discount : ℚ) (headphone_tax : ℚ) (shipping_cost : ℚ) : ℚ :=
  let tape_total := tape_price * num_tapes * (1 - tape_discount)
  let headphone_total := headphone_price * (1 + headphone_tax)
  let total_cost := tape_total + headphone_total + vinyl_price + poster_price + shipping_cost
  gift_amount - total_cost

/-- Theorem stating that Amanda will have $16.75 left after her purchases -/
theorem amanda_remaining_money :
  remaining_money 200 15 3 55 35 45 0.1 0.05 5 = 16.75 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l1733_173331


namespace NUMINAMATH_CALUDE_rational_function_property_l1733_173321

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in the graph of a rational function -/
def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes in the graph of a rational function -/
def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes in the graph of a rational function -/
def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes in the graph of a rational function -/
def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

/-- The main theorem about the specific rational function -/
theorem rational_function_property : 
  let f : RationalFunction := {
    numerator := Polynomial.monomial 2 1 - Polynomial.monomial 1 5 + Polynomial.monomial 0 6,
    denominator := Polynomial.monomial 3 1 - Polynomial.monomial 2 3 + Polynomial.monomial 1 2
  }
  let p := count_holes f
  let q := count_vertical_asymptotes f
  let r := count_horizontal_asymptotes f
  let s := count_oblique_asymptotes f
  p + 2*q + 3*r + 4*s = 8 := by sorry

end NUMINAMATH_CALUDE_rational_function_property_l1733_173321


namespace NUMINAMATH_CALUDE_rational_sum_problem_l1733_173307

theorem rational_sum_problem (a b c d : ℚ) 
  (h1 : b + c + d = -1)
  (h2 : a + c + d = -3)
  (h3 : a + b + d = 2)
  (h4 : a + b + c = 17) :
  a = 6 ∧ b = 8 ∧ c = 3 ∧ d = -12 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_problem_l1733_173307


namespace NUMINAMATH_CALUDE_grass_seed_min_cost_l1733_173308

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat

/-- Finds the minimum cost to purchase grass seed given constraints -/
def minCostGrassSeed (bags : List GrassSeedBag) (minWeight maxWeight : Nat) : Rat :=
  sorry

/-- The problem statement -/
theorem grass_seed_min_cost :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 1382/100 },
    { weight := 10, price := 2043/100 },
    { weight := 25, price := 3225/100 }
  ]
  let minWeight : Nat := 65
  let maxWeight : Nat := 80
  minCostGrassSeed bags minWeight maxWeight = 9875/100 := by
  sorry

end NUMINAMATH_CALUDE_grass_seed_min_cost_l1733_173308


namespace NUMINAMATH_CALUDE_equal_sum_parallel_segments_l1733_173384

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ) (BC : ℝ) (CA : ℝ)

/-- Points on the sides of the triangle -/
structure Points (t : Triangle) :=
  (D : ℝ) (E : ℝ) (F : ℝ) (G : ℝ)
  (h₁ : 0 ≤ D ∧ D ≤ E ∧ E ≤ t.AB)
  (h₂ : 0 ≤ F ∧ F ≤ G ∧ G ≤ t.CA)

/-- Perimeter of triangle ADF -/
def perim_ADF (t : Triangle) (p : Points t) : ℝ :=
  p.D + (p.F - p.D) + p.F

/-- Perimeter of trapezoid DEFG -/
def perim_DEFG (t : Triangle) (p : Points t) : ℝ :=
  (p.E - p.D) + (p.G - p.F) + p.G + (p.F - p.D)

/-- Perimeter of trapezoid EBCG -/
def perim_EBCG (t : Triangle) (p : Points t) : ℝ :=
  (t.AB - p.E) + t.BC + (t.CA - p.G) + (p.G - p.F)

theorem equal_sum_parallel_segments (t : Triangle) (p : Points t) 
    (h_sides : t.AB = 2 ∧ t.BC = 3 ∧ t.CA = 4)
    (h_parallel : (p.E - p.D) / t.BC = (p.G - p.F) / t.BC)
    (h_perims : perim_ADF t p = perim_DEFG t p ∧ perim_DEFG t p = perim_EBCG t p) :
    (p.E - p.D) + (p.G - p.F) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_parallel_segments_l1733_173384


namespace NUMINAMATH_CALUDE_complex_equation_proof_l1733_173395

theorem complex_equation_proof (x : ℂ) (h : x - 1/x = 3*I) : x^12 - 1/x^12 = 103682 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l1733_173395


namespace NUMINAMATH_CALUDE_linear_function_properties_l1733_173319

def LinearFunction (m c : ℝ) : ℝ → ℝ := fun x ↦ m * x + c

theorem linear_function_properties (f : ℝ → ℝ) (m c : ℝ) 
  (h1 : ∃ k : ℝ, ∀ x, f x + 2 = 3 * k * x)
  (h2 : f 1 = 4)
  (h3 : f = LinearFunction m c) :
  (f = LinearFunction 6 (-2)) ∧ 
  (∀ a b : ℝ, f (-1) = a ∧ f 2 = b → a < b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1733_173319


namespace NUMINAMATH_CALUDE_prob_sum_seven_l1733_173337

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The set of all possible outcomes when throwing two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes where the sum is 7 -/
def sum_seven : Finset (ℕ × ℕ) :=
  all_outcomes.filter (λ p => p.1 + p.2 + 2 = 7)

/-- The probability of getting a sum of 7 when throwing two fair dice -/
theorem prob_sum_seven :
  (sum_seven.card : ℚ) / all_outcomes.card = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_prob_sum_seven_l1733_173337


namespace NUMINAMATH_CALUDE_square_area_increase_l1733_173325

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.3 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.69 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l1733_173325


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l1733_173310

-- Define the logarithm base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem cos_double_angle_special_case (x : ℝ) (h : lg (Real.cos x) = -1/2) : 
  Real.cos (2 * x) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l1733_173310


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1733_173301

theorem triangle_angle_proof (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  c = 45 →                 -- one angle is 45°
  b = 2 * a →              -- ratio of other two angles is 2:1
  a = 45 :=                -- prove that the smaller angle is also 45°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1733_173301


namespace NUMINAMATH_CALUDE_marathon_finishers_l1733_173349

/-- Proves that 563 people finished the marathon given the conditions --/
theorem marathon_finishers :
  ∀ (finished : ℕ),
  (finished + (finished + 124) = 1250) →
  finished = 563 := by
  sorry

end NUMINAMATH_CALUDE_marathon_finishers_l1733_173349


namespace NUMINAMATH_CALUDE_new_players_count_new_players_proof_l1733_173345

theorem new_players_count (returning_players : ℕ) (players_per_group : ℕ) (num_groups : ℕ) : ℕ :=
  let total_players := num_groups * players_per_group
  total_players - returning_players

theorem new_players_proof :
  new_players_count 6 6 9 = 48 := by
  sorry

end NUMINAMATH_CALUDE_new_players_count_new_players_proof_l1733_173345


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_equals_zero_l1733_173302

theorem sum_mod_thirteen_equals_zero :
  (7650 + 7651 + 7652 + 7653 + 7654) % 13 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_equals_zero_l1733_173302


namespace NUMINAMATH_CALUDE_ratio_problem_l1733_173369

theorem ratio_problem (a b : ℝ) (h1 : a ≠ b) (h2 : (a + b) / (a - b) = 3) : a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1733_173369


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_40_squared_l1733_173386

theorem largest_prime_divisor_of_17_squared_plus_40_squared :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 40^2) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (17^2 + 40^2) → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_40_squared_l1733_173386


namespace NUMINAMATH_CALUDE_correct_student_distribution_l1733_173328

/-- Ticket pricing structure -/
def ticket_price (n : ℕ) : ℕ :=
  if n ≤ 50 then 15
  else if n ≤ 100 then 12
  else 10

/-- Total number of students -/
def total_students : ℕ := 105

/-- Total amount paid -/
def total_paid : ℕ := 1401

/-- Number of students in Class (1) -/
def class_1_students : ℕ := 47

/-- Number of students in Class (2) -/
def class_2_students : ℕ := total_students - class_1_students

/-- Theorem: Given the ticket pricing structure and total amount paid, 
    the number of students in Class (1) is 47 and in Class (2) is 58 -/
theorem correct_student_distribution :
  class_1_students > 40 ∧ 
  class_1_students < 50 ∧
  class_2_students = 58 ∧
  class_1_students + class_2_students = total_students ∧
  ticket_price class_1_students * class_1_students + 
  ticket_price class_2_students * class_2_students = total_paid :=
by sorry

end NUMINAMATH_CALUDE_correct_student_distribution_l1733_173328


namespace NUMINAMATH_CALUDE_solution_relationship_l1733_173303

theorem solution_relationship (x y : ℝ) : 
  (2 * x + y = 7) → (x - y = 5) → (x + 2 * y = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_relationship_l1733_173303


namespace NUMINAMATH_CALUDE_dartboard_probability_l1733_173332

structure Dartboard :=
  (outer_radius : ℝ)
  (inner_radius : ℝ)
  (sections : ℕ)
  (inner_values : Fin 2 → ℕ)
  (outer_values : Fin 2 → ℕ)

def probability_of_score (db : Dartboard) (score : ℕ) (darts : ℕ) : ℚ :=
  sorry

theorem dartboard_probability (db : Dartboard) :
  db.outer_radius = 8 ∧
  db.inner_radius = 4 ∧
  db.sections = 4 ∧
  db.inner_values 0 = 3 ∧
  db.inner_values 1 = 4 ∧
  db.outer_values 0 = 2 ∧
  db.outer_values 1 = 5 →
  probability_of_score db 12 3 = 9 / 1024 :=
sorry

end NUMINAMATH_CALUDE_dartboard_probability_l1733_173332


namespace NUMINAMATH_CALUDE_square_diagonal_l1733_173356

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (h1 : A = 9/16) (h2 : A = s^2) (h3 : d = s * Real.sqrt 2) :
  d = 3/4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l1733_173356


namespace NUMINAMATH_CALUDE_point_not_on_graph_l1733_173375

/-- The function f(x) = x^2 / (x + 1) -/
def f (x : ℚ) : ℚ := x^2 / (x + 1)

/-- The point (-1/2, 1/6) -/
def point : ℚ × ℚ := (-1/2, 1/6)

/-- Theorem: The point (-1/2, 1/6) is not on the graph of f(x) = x^2 / (x + 1) -/
theorem point_not_on_graph : f point.1 ≠ point.2 := by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l1733_173375


namespace NUMINAMATH_CALUDE_ratio_problem_l1733_173311

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : 
  x / y = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1733_173311


namespace NUMINAMATH_CALUDE_rectangular_plot_longer_side_l1733_173374

theorem rectangular_plot_longer_side 
  (width : ℝ) 
  (pole_distance : ℝ) 
  (num_poles : ℕ) 
  (h1 : width = 30)
  (h2 : pole_distance = 5)
  (h3 : num_poles = 32) :
  let perimeter := pole_distance * (num_poles - 1 : ℝ)
  let length := (perimeter / 2) - width
  length = 47.5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_longer_side_l1733_173374


namespace NUMINAMATH_CALUDE_transformed_curve_equation_l1733_173309

/-- Given a curve y = (1/3)cos(2x) and a scaling transformation x' = 2x, y' = 3y,
    the transformed curve is y' = cos(x'). -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  y = (1/3) * Real.cos (2 * x) →
  x' = 2 * x →
  y' = 3 * y →
  y' = Real.cos x' := by
  sorry

end NUMINAMATH_CALUDE_transformed_curve_equation_l1733_173309


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1733_173330

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1733_173330


namespace NUMINAMATH_CALUDE_problem_solution_l1733_173398

/-- A function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := sorry

/-- The constant of proportionality -/
def k : ℝ := sorry

theorem problem_solution :
  (∀ x : ℝ, f x - 4 = k * (2 * x + 1)) →  -- y-4 is directly proportional to 2x+1
  (f (-1) = 6) →                          -- When x = -1, y = 6
  (∀ x : ℝ, f x = -4 * x + 2) ∧           -- Functional expression
  (f (3/2) = -4)                          -- When y = -4, x = 3/2
  := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1733_173398


namespace NUMINAMATH_CALUDE_single_draw_probability_triple_draw_probability_l1733_173385

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the outcome of drawing a single ball -/
def SingleDrawOutcome := BallColor

/-- Represents the outcome of drawing three balls -/
def TripleDrawOutcome := (BallColor × BallColor × BallColor)

/-- The total number of balls in the box -/
def totalBalls : Nat := 5 + 2

/-- The number of white balls in the box -/
def whiteBalls : Nat := 5

/-- The number of black balls in the box -/
def blackBalls : Nat := 2

/-- A function that simulates drawing a single ball -/
noncomputable def simulateSingleDraw : SingleDrawOutcome := sorry

/-- A function that simulates drawing three balls -/
noncomputable def simulateTripleDraw : TripleDrawOutcome := sorry

/-- Checks if a single draw outcome is favorable (white ball) -/
def isFavorableSingleDraw (outcome : SingleDrawOutcome) : Bool :=
  match outcome with
  | BallColor.White => true
  | BallColor.Black => false

/-- Checks if a triple draw outcome is favorable (all white balls) -/
def isFavorableTripleDraw (outcome : TripleDrawOutcome) : Bool :=
  match outcome with
  | (BallColor.White, BallColor.White, BallColor.White) => true
  | _ => false

/-- Theorem: The probability of drawing a white ball is equal to the ratio of favorable outcomes to total outcomes in a random simulation -/
theorem single_draw_probability (n : Nat) (m : Nat) :
  m ≤ n →
  (m : ℚ) / n = whiteBalls / totalBalls :=
sorry

/-- Theorem: The probability of drawing three white balls is equal to the ratio of favorable outcomes to total outcomes in a random simulation -/
theorem triple_draw_probability (n : Nat) (m : Nat) :
  m ≤ n →
  (m : ℚ) / n = (whiteBalls / totalBalls) * ((whiteBalls - 1) / (totalBalls - 1)) * ((whiteBalls - 2) / (totalBalls - 2)) :=
sorry

end NUMINAMATH_CALUDE_single_draw_probability_triple_draw_probability_l1733_173385


namespace NUMINAMATH_CALUDE_constant_product_l1733_173323

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}

-- Define the symmetry axis
def symmetry_axis : Set (ℝ × ℝ) :=
  {p | 2 * p.1 - 3 * p.2 + 6 = 0}

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define line m
def line_m : Set (ℝ × ℝ) :=
  {p | p.1 + 2 * p.2 + 2 = 0}

-- Define the theorem
theorem constant_product :
  ∀ l : Set (ℝ × ℝ),
  (P ∈ l) →
  (∃ A B : ℝ × ℝ,
    A ∈ circle_C ∧
    B ∈ line_m ∧
    A ∈ l ∧
    B ∈ l ∧
    (∃ C : ℝ × ℝ, C ∈ circle_C ∧ C ∈ l ∧ A ≠ C ∧ 
      A = ((C.1 + A.1) / 2, (C.2 + A.2) / 2)) →
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * 
     Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 6)) :=
sorry


end NUMINAMATH_CALUDE_constant_product_l1733_173323


namespace NUMINAMATH_CALUDE_louisa_travel_problem_l1733_173318

/-- Louisa's vacation travel problem -/
theorem louisa_travel_problem (first_day_distance : ℝ) (speed : ℝ) (time_difference : ℝ) 
  (h1 : first_day_distance = 100)
  (h2 : speed = 25)
  (h3 : time_difference = 3)
  : ∃ (second_day_distance : ℝ), second_day_distance = 175 := by
  sorry

end NUMINAMATH_CALUDE_louisa_travel_problem_l1733_173318


namespace NUMINAMATH_CALUDE_even_sum_probability_l1733_173351

-- Define the possible outcomes for each spinner
def X : Finset ℕ := {2, 5, 7}
def Y : Finset ℕ := {2, 4, 6}
def Z : Finset ℕ := {1, 2, 3, 4}

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define the probability of getting an even sum
def probEvenSum : ℚ := sorry

-- Theorem statement
theorem even_sum_probability :
  probEvenSum = 1/2 := by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l1733_173351


namespace NUMINAMATH_CALUDE_max_socks_is_eighteen_l1733_173305

/-- Represents the amount of yarn needed for different items -/
structure YarnAmount where
  sock : ℕ
  hat : ℕ
  sweater : ℕ

/-- Represents the two balls of yarn -/
structure YarnBalls where
  large : YarnAmount
  small : YarnAmount

/-- The given conditions for the yarn balls -/
def yarn_conditions : YarnBalls where
  large := { sock := 3, hat := 5, sweater := 1 }
  small := { sock := 0, hat := 2, sweater := 1/2 }

/-- The maximum number of socks that can be knitted -/
def max_socks : ℕ := 18

/-- Theorem stating that the maximum number of socks that can be knitted is 18 -/
theorem max_socks_is_eighteen (y : YarnBalls) (h : y = yarn_conditions) : 
  (∃ (n : ℕ), n ≤ max_socks ∧ 
    n * y.large.sock ≤ y.large.hat * y.large.sock + y.small.hat * y.large.sock ∧
    ∀ (m : ℕ), m * y.large.sock ≤ y.large.hat * y.large.sock + y.small.hat * y.large.sock → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_max_socks_is_eighteen_l1733_173305


namespace NUMINAMATH_CALUDE_problem_1_l1733_173391

theorem problem_1 (x : ℝ) (h : x^2 = 2) : (3*x)^2 - 4*(x^3)^2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1733_173391


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l1733_173363

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given A(a,1) and B(5,b) are symmetric with respect to the origin, prove a - b = -4 -/
theorem symmetric_points_difference (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a - b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l1733_173363


namespace NUMINAMATH_CALUDE_range_of_m_for_p_and_q_range_of_t_for_q_necessary_not_sufficient_for_s_l1733_173367

-- Define the propositions
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

def q (m : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / m^2 + y^2 / (2*m + 8) = 1 → 
  (∃ c : ℝ, c > 0 ∧ x^2 / (m^2 - c^2) + y^2 / m^2 = 1)

def s (m t : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / (m - t) + y^2 / (m - t - 1) = 1 → 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1)

-- Theorem statements
theorem range_of_m_for_p_and_q :
  ∀ m : ℝ, (p m ∧ q m) ↔ ((-4 < m ∧ m < -2) ∨ m > 4) :=
sorry

theorem range_of_t_for_q_necessary_not_sufficient_for_s :
  ∀ t : ℝ, (∀ m : ℝ, s m t → q m) ∧ (∃ m : ℝ, q m ∧ ¬s m t) ↔ 
  ((-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_p_and_q_range_of_t_for_q_necessary_not_sufficient_for_s_l1733_173367


namespace NUMINAMATH_CALUDE_solutions_of_absolute_value_equation_l1733_173347

theorem solutions_of_absolute_value_equation :
  {x : ℝ | |x - 2| + |x - 3| = 1} = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_solutions_of_absolute_value_equation_l1733_173347


namespace NUMINAMATH_CALUDE_count_divisible_by_11_eq_36_l1733_173355

/-- The number obtained by concatenating integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- The count of k values in [1, 200] for which a_k is divisible by 11 -/
def count_divisible_by_11 : ℕ := sorry

/-- Theorem stating that the count of k values in [1, 200] for which a_k is divisible by 11 is 36 -/
theorem count_divisible_by_11_eq_36 : count_divisible_by_11 = 36 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_eq_36_l1733_173355


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1733_173399

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1 ∨ 4 < x}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1733_173399


namespace NUMINAMATH_CALUDE_surface_area_combined_shape_l1733_173326

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of the modified shape -/
def surfaceAreaModifiedShape (original : CubeDimensions) (removed : CubeDimensions) : ℝ :=
  sorry

/-- Calculates the surface area of the combined shape -/
def surfaceAreaCombinedShape (original : CubeDimensions) (removed : CubeDimensions) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the combined shape is 38 cm² -/
theorem surface_area_combined_shape :
  let original := CubeDimensions.mk 2 2 2
  let removed := CubeDimensions.mk 1 1 1
  surfaceAreaCombinedShape original removed = 38 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_combined_shape_l1733_173326


namespace NUMINAMATH_CALUDE_min_value_of_y_l1733_173350

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 2) :
  ∀ y : ℝ, y = 4*a + b → y ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_y_l1733_173350


namespace NUMINAMATH_CALUDE_school_survey_l1733_173382

theorem school_survey (total_students : ℕ) (sample_size : ℕ) (girl_boy_diff : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girl_boy_diff = 20 →
  (sample_size - girl_boy_diff) / 2 / sample_size * total_students = 720 :=
by sorry

end NUMINAMATH_CALUDE_school_survey_l1733_173382


namespace NUMINAMATH_CALUDE_yoongi_has_smallest_points_l1733_173377

def jungkook_points : ℕ := 9
def yoongi_points : ℕ := 4
def yuna_points : ℕ := 5

theorem yoongi_has_smallest_points : 
  yoongi_points ≤ jungkook_points ∧ yoongi_points ≤ yuna_points :=
by sorry

end NUMINAMATH_CALUDE_yoongi_has_smallest_points_l1733_173377


namespace NUMINAMATH_CALUDE_beetles_eaten_per_day_l1733_173390

/-- The number of beetles eaten by one bird per day -/
def beetles_per_bird : ℕ := 12

/-- The number of birds eaten by one snake per day -/
def birds_per_snake : ℕ := 3

/-- The number of snakes eaten by one jaguar per day -/
def snakes_per_jaguar : ℕ := 5

/-- The number of jaguars in the forest -/
def jaguars_in_forest : ℕ := 6

/-- The total number of beetles eaten per day in the forest -/
def total_beetles_eaten : ℕ := 
  jaguars_in_forest * snakes_per_jaguar * birds_per_snake * beetles_per_bird

theorem beetles_eaten_per_day :
  total_beetles_eaten = 1080 := by
  sorry

end NUMINAMATH_CALUDE_beetles_eaten_per_day_l1733_173390


namespace NUMINAMATH_CALUDE_proportion_solution_l1733_173335

theorem proportion_solution (x y : ℝ) : 
  (0.60 : ℝ) / x = y / 4 ∧ x = 0.39999999999999997 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1733_173335


namespace NUMINAMATH_CALUDE_problem_solution_l1733_173357

theorem problem_solution :
  ∀ (a b m n : ℝ),
  (m = (a + 4) ^ (1 / (b - 1))) →
  (n = (3 * b - 1) ^ (1 / (a - 2))) →
  ((b - 1) = 2) →
  ((a - 2) = 3) →
  ((m - 2 * n) ^ (1 / 3) = -1) ∧
  (∀ (m' n' : ℝ),
    (m' = Real.sqrt (1 - a) + Real.sqrt (a - 1) + 1) →
    (n' = 25) →
    (Real.sqrt (3 * n' + 6 * m') = 9 ∨ Real.sqrt (3 * n' + 6 * m') = -9)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1733_173357


namespace NUMINAMATH_CALUDE_alternatingArrangements_4_3_l1733_173388

/-- The number of ways to arrange 4 men and 3 women in a row, such that no two men or two women are adjacent -/
def alternatingArrangements (numMen : Nat) (numWomen : Nat) : Nat :=
  Nat.factorial numMen * 
  (Nat.choose (numMen + 1) numWomen) * 
  Nat.factorial numWomen

/-- Theorem stating that the number of alternating arrangements of 4 men and 3 women is 1440 -/
theorem alternatingArrangements_4_3 : 
  alternatingArrangements 4 3 = 1440 := by
  sorry

#eval alternatingArrangements 4 3

end NUMINAMATH_CALUDE_alternatingArrangements_4_3_l1733_173388


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l1733_173368

/-- The equation of a line symmetric to y = 3x + 1 with respect to the y-axis -/
theorem symmetric_line_equation : 
  ∀ (x y : ℝ), (∃ (m n : ℝ), n = 3 * m + 1 ∧ x + m = 0 ∧ y = n) → y = -3 * x + 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l1733_173368


namespace NUMINAMATH_CALUDE_pet_store_count_l1733_173342

/-- Given the ratios of cats to dogs and dogs to birds, and the number of cats,
    prove the number of dogs and birds -/
theorem pet_store_count (cats : ℕ) (dogs : ℕ) (birds : ℕ) : 
  cats = 20 →                   -- There are 20 cats
  5 * cats = 4 * dogs →         -- Ratio of cats to dogs is 4:5
  7 * dogs = 3 * birds →        -- Ratio of dogs to birds is 3:7
  dogs = 25 ∧ birds = 56 :=     -- Prove dogs = 25 and birds = 56
by sorry

end NUMINAMATH_CALUDE_pet_store_count_l1733_173342


namespace NUMINAMATH_CALUDE_calculation_proof_expression_equivalence_l1733_173380

-- First part of the problem
theorem calculation_proof : 28 + 72 + (9 - 8) = 172 := by sorry

-- Second part of the problem
def original_expression : ℚ := 4600 / 23 - 19 * 10

def reordered_expression : ℚ := (4600 / 23) - (19 * 10)

theorem expression_equivalence : original_expression = reordered_expression := by sorry

end NUMINAMATH_CALUDE_calculation_proof_expression_equivalence_l1733_173380


namespace NUMINAMATH_CALUDE_oven_temperature_l1733_173360

theorem oven_temperature (required_temp increase_needed : ℕ) 
  (h1 : required_temp = 546)
  (h2 : increase_needed = 396) :
  required_temp - increase_needed = 150 := by
sorry

end NUMINAMATH_CALUDE_oven_temperature_l1733_173360


namespace NUMINAMATH_CALUDE_ball_probability_l1733_173366

theorem ball_probability (n m : ℕ) : 
  n > 0 ∧ m ≤ n ∧
  (1 - (m.choose 2 : ℚ) / (n.choose 2 : ℚ) = 3/5) ∧
  (6 * (m : ℚ) / (n : ℚ) = 4) →
  ((n - m - 1 : ℚ) / (n - 1 : ℚ) = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_ball_probability_l1733_173366


namespace NUMINAMATH_CALUDE_product_of_one_plus_greater_than_eight_l1733_173320

theorem product_of_one_plus_greater_than_eight
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_prod : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_greater_than_eight_l1733_173320


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1733_173329

/-- Hyperbola struct -/
structure Hyperbola where
  F₁ : ℝ × ℝ  -- First focus
  F₂ : ℝ × ℝ  -- Second focus
  e : ℝ        -- Eccentricity

/-- Point on the hyperbola -/
def Point : Type := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Length of the real axis of a hyperbola -/
def realAxisLength (h : Hyperbola) : ℝ := sorry

/-- Theorem statement -/
theorem hyperbola_real_axis_length 
  (C : Hyperbola) 
  (P : Point) 
  (h_eccentricity : C.e = Real.sqrt 5)
  (h_point_on_hyperbola : P ∈ {p : Point | distance p C.F₁ - distance p C.F₂ = realAxisLength C})
  (h_distance_ratio : 2 * distance P C.F₁ = 3 * distance P C.F₂)
  (h_triangle_area : triangleArea P C.F₁ C.F₂ = 2 * Real.sqrt 5) :
  realAxisLength C = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1733_173329


namespace NUMINAMATH_CALUDE_factorization_equality_l1733_173387

theorem factorization_equality (a : ℝ) : 2*a^2 + 4*a + 2 = 2*(a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1733_173387


namespace NUMINAMATH_CALUDE_exists_nonconvergent_sequence_l1733_173313

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: The sequence is increasing -/
def IsIncreasing (a : Sequence) : Prop :=
  ∀ n, a n < a (n + 1)

/-- Property: Each term is either the arithmetic mean or the geometric mean of its neighbors -/
def IsMeanOfNeighbors (a : Sequence) : Prop :=
  ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) * a (n + 1) = a n * a (n + 2))

/-- Property: The sequence is an arithmetic progression from a certain point -/
def EventuallyArithmetic (a : Sequence) : Prop :=
  ∃ N, ∀ n ≥ N, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Property: The sequence is a geometric progression from a certain point -/
def EventuallyGeometric (a : Sequence) : Prop :=
  ∃ N, ∀ n ≥ N, a (n + 2) * a n = a (n + 1) * a (n + 1)

/-- The main theorem -/
theorem exists_nonconvergent_sequence :
  ∃ (a : Sequence), IsIncreasing a ∧ IsMeanOfNeighbors a ∧
    ¬(EventuallyArithmetic a ∨ EventuallyGeometric a) :=
sorry

end NUMINAMATH_CALUDE_exists_nonconvergent_sequence_l1733_173313


namespace NUMINAMATH_CALUDE_odd_factors_of_450_l1733_173371

/-- The number of odd factors of a natural number n -/
def num_odd_factors (n : ℕ) : ℕ := sorry

/-- 450 has exactly 9 odd factors -/
theorem odd_factors_of_450 : num_odd_factors 450 = 9 := by sorry

end NUMINAMATH_CALUDE_odd_factors_of_450_l1733_173371


namespace NUMINAMATH_CALUDE_percentage_problem_l1733_173353

theorem percentage_problem (x p : ℝ) (h1 : 0.25 * x = (p/100) * 500 - 5) (h2 : x = 180) : p = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1733_173353


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1733_173334

/-- A coloring of a complete graph with 6 vertices using 5 colors -/
def GraphColoring : Type := Fin 6 → Fin 6 → Fin 5

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (c : GraphColoring) : Prop :=
  ∀ v : Fin 6, ∀ u w : Fin 6, u ≠ v → w ≠ v → u ≠ w → c v u ≠ c v w

/-- Theorem stating that a valid 5-coloring exists for a complete graph with 6 vertices -/
theorem exists_valid_coloring : ∃ c : GraphColoring, is_valid_coloring c := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1733_173334


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1733_173348

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 5 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 25 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1733_173348


namespace NUMINAMATH_CALUDE_triangle_area_formula_l1733_173322

variable (m₁ m₂ m₃ : ℝ)
variable (u u₁ u₂ u₃ t : ℝ)

def is_altitude (m : ℝ) : Prop := m > 0

theorem triangle_area_formula 
  (h₁ : is_altitude m₁)
  (h₂ : is_altitude m₂)
  (h₃ : is_altitude m₃)
  (hu : u = 1/2 * (1/m₁ + 1/m₂ + 1/m₃))
  (hu₁ : u₁ = u - 1/m₁)
  (hu₂ : u₂ = u - 1/m₂)
  (hu₃ : u₃ = u - 1/m₃)
  : t = 4 * Real.sqrt (u * u₁ * u₂ * u₃) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_formula_l1733_173322


namespace NUMINAMATH_CALUDE_polynomial_b_value_l1733_173300

theorem polynomial_b_value (A B : ℤ) : 
  let p := fun z : ℝ => z^4 - 9*z^3 + A*z^2 + B*z + 18
  (∃ r1 r2 r3 r4 : ℕ+, (p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ p r4 = 0) ∧ 
                       (r1 + r2 + r3 + r4 = 9)) →
  B = -20 := by
sorry

end NUMINAMATH_CALUDE_polynomial_b_value_l1733_173300


namespace NUMINAMATH_CALUDE_x_range_l1733_173364

-- Define the inequality condition
def inequality_condition (x m : ℝ) : Prop :=
  2 * x - 1 > m * (x^2 - 1)

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  |m| ≤ 2

-- Theorem statement
theorem x_range :
  (∀ x m : ℝ, m_range m → inequality_condition x m) →
  ∃ a b : ℝ, a = (Real.sqrt 7 - 1) / 2 ∧ b = (Real.sqrt 3 + 1) / 2 ∧
    ∀ x : ℝ, (∀ m : ℝ, m_range m → inequality_condition x m) → a < x ∧ x < b :=
sorry

end NUMINAMATH_CALUDE_x_range_l1733_173364


namespace NUMINAMATH_CALUDE_quartic_comparison_l1733_173339

noncomputable def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

def sum_of_zeros (f : ℝ → ℝ) : ℝ := 2 -- From Vieta's formula for quartic polynomial

def product_of_zeros (f : ℝ → ℝ) : ℝ := f 0

def sum_of_coefficients (f : ℝ → ℝ) : ℝ := 3 -- 1 - 2 + 3 - 4 + 5

theorem quartic_comparison :
  (sum_of_zeros Q)^2 ≤ Q (-1) ∧
  (sum_of_zeros Q)^2 ≤ product_of_zeros Q ∧
  (sum_of_zeros Q)^2 ≤ sum_of_coefficients Q :=
sorry

end NUMINAMATH_CALUDE_quartic_comparison_l1733_173339


namespace NUMINAMATH_CALUDE_h_k_equality_implies_m_value_l1733_173370

/-- The function h(x) = x^2 - 3x + m -/
def h (x m : ℝ) : ℝ := x^2 - 3*x + m

/-- The function k(x) = x^2 - 3x + 5m -/
def k (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

/-- Theorem stating that if 3h(5) = 2k(5), then m = 10/7 -/
theorem h_k_equality_implies_m_value :
  ∀ m : ℝ, 3 * (h 5 m) = 2 * (k 5 m) → m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_h_k_equality_implies_m_value_l1733_173370


namespace NUMINAMATH_CALUDE_quintic_integer_root_count_l1733_173362

/-- Represents a polynomial of degree 5 with integer coefficients -/
structure QuinticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The set of possible numbers of integer roots for a quintic polynomial with integer coefficients -/
def PossibleRootCounts : Set ℕ := {0, 1, 2, 3, 5}

/-- Counts the number of integer roots of a quintic polynomial, including multiplicity -/
def countIntegerRoots (p : QuinticPolynomial) : ℕ := sorry

/-- Theorem stating that the number of integer roots of a quintic polynomial with integer coefficients
    can only be 0, 1, 2, 3, or 5 -/
theorem quintic_integer_root_count (p : QuinticPolynomial) :
  countIntegerRoots p ∈ PossibleRootCounts := by sorry

end NUMINAMATH_CALUDE_quintic_integer_root_count_l1733_173362


namespace NUMINAMATH_CALUDE_coefficient_x4_in_product_l1733_173365

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - 4*x^2 + x - 1
def q (x : ℝ) : ℝ := 3*x^4 - 4*x^3 + 5*x^2 - 2*x + 6

theorem coefficient_x4_in_product :
  ∃ (a b c d e f g h i j : ℝ),
    p x * q x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + (-38)*x^4 + f*x^3 + g*x^2 + h*x + i :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_product_l1733_173365


namespace NUMINAMATH_CALUDE_lisa_spoon_count_l1733_173317

theorem lisa_spoon_count :
  let num_children : ℕ := 6
  let baby_spoons_per_child : ℕ := 4
  let decorative_spoons : ℕ := 4
  let large_spoons : ℕ := 20
  let dessert_spoons : ℕ := 10
  let teaspoons : ℕ := 25
  
  let total_baby_spoons := num_children * baby_spoons_per_child
  let total_special_spoons := total_baby_spoons + decorative_spoons
  let total_new_spoons := large_spoons + dessert_spoons + teaspoons
  let total_spoons := total_special_spoons + total_new_spoons

  total_spoons = 83 := by
sorry

end NUMINAMATH_CALUDE_lisa_spoon_count_l1733_173317


namespace NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l1733_173352

theorem rectangle_to_square_area_ratio (a : ℝ) (a_pos : 0 < a) : 
  let square_side := a
  let square_diagonal := a * Real.sqrt 2
  let rectangle_length := square_diagonal
  let rectangle_width := square_side
  let square_area := square_side ^ 2
  let rectangle_area := rectangle_length * rectangle_width
  rectangle_area / square_area = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l1733_173352


namespace NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l1733_173372

theorem sixth_root_of_24414062515625 : 
  (24414062515625 : ℝ) ^ (1/6 : ℝ) = 51 := by sorry

end NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l1733_173372


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_reciprocal_l1733_173359

theorem cubic_root_sum_squares_reciprocal (α β γ : ℂ) : 
  α^3 - 6*α^2 + 11*α - 6 = 0 →
  β^3 - 6*β^2 + 11*β - 6 = 0 →
  γ^3 - 6*γ^2 + 11*γ - 6 = 0 →
  α ≠ β → β ≠ γ → γ ≠ α →
  1/α^2 + 1/β^2 + 1/γ^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_reciprocal_l1733_173359


namespace NUMINAMATH_CALUDE_inequality_proof_l1733_173379

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1733_173379


namespace NUMINAMATH_CALUDE_treasure_in_fourth_bag_l1733_173396

/-- Given four bags A, B, C, and D, prove that D is the heaviest bag. -/
theorem treasure_in_fourth_bag (A B C D : ℝ) 
  (h1 : A + B < C)
  (h2 : A + C = D)
  (h3 : A + D > B + C) :
  D > A ∧ D > B ∧ D > C := by
  sorry

end NUMINAMATH_CALUDE_treasure_in_fourth_bag_l1733_173396


namespace NUMINAMATH_CALUDE_square_fence_perimeter_36_posts_l1733_173327

/-- Calculates the perimeter of a square fence given the number of posts, post width, and gap between posts. -/
def square_fence_perimeter (total_posts : ℕ) (post_width_inches : ℕ) (gap_feet : ℕ) : ℕ :=
  let posts_per_side : ℕ := (total_posts - 4) / 4 + 1
  let side_length : ℕ := (posts_per_side - 1) * gap_feet
  4 * side_length

/-- Theorem stating that a square fence with 36 posts, 6-inch wide posts, and 6-foot gaps has a perimeter of 192 feet. -/
theorem square_fence_perimeter_36_posts :
  square_fence_perimeter 36 6 6 = 192 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_36_posts_l1733_173327


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1733_173358

theorem rectangle_diagonal (l w : ℝ) (h1 : l + w = 15) (h2 : l = 2 * w) :
  Real.sqrt (l^2 + w^2) = Real.sqrt 125 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1733_173358
