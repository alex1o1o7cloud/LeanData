import Mathlib

namespace NUMINAMATH_CALUDE_P_symmetric_l1696_169659

/-- Definition of the polynomial sequence P_m -/
def P : ℕ → (ℚ → ℚ → ℚ → ℚ)
| 0 => λ _ _ _ => 1
| (m + 1) => λ x y z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

/-- Statement that P_m is symmetric for all m -/
theorem P_symmetric (m : ℕ) (x y z : ℚ) :
  P m x y z = P m y x z ∧
  P m x y z = P m x z y ∧
  P m x y z = P m y z x ∧
  P m x y z = P m z x y ∧
  P m x y z = P m z y x :=
by sorry

end NUMINAMATH_CALUDE_P_symmetric_l1696_169659


namespace NUMINAMATH_CALUDE_cd_length_possibilities_l1696_169676

/-- Represents a tetrahedron ABCD inscribed in a cylinder --/
structure InscribedTetrahedron where
  /-- Length of edge AB --/
  ab : ℝ
  /-- Length of edges AC and CB --/
  ac_cb : ℝ
  /-- Length of edges AD and DB --/
  ad_db : ℝ
  /-- Assertion that the tetrahedron is inscribed in a cylinder with minimal radius --/
  inscribed_minimal : Bool
  /-- Assertion that all vertices lie on the lateral surface of the cylinder --/
  vertices_on_surface : Bool
  /-- Assertion that CD is parallel to the cylinder's axis --/
  cd_parallel_axis : Bool

/-- Theorem stating the possible lengths of CD in the inscribed tetrahedron --/
theorem cd_length_possibilities (t : InscribedTetrahedron) 
  (h1 : t.ab = 2)
  (h2 : t.ac_cb = 6)
  (h3 : t.ad_db = 7)
  (h4 : t.inscribed_minimal)
  (h5 : t.vertices_on_surface)
  (h6 : t.cd_parallel_axis) :
  ∃ (cd : ℝ), (cd = Real.sqrt 47 + Real.sqrt 34) ∨ (cd = |Real.sqrt 47 - Real.sqrt 34|) :=
sorry

end NUMINAMATH_CALUDE_cd_length_possibilities_l1696_169676


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1696_169655

/-- Given a geometric progression with the first three terms 2^(1/4), 2^(1/8), and 2^(1/16),
    the fourth term is 2^(1/32). -/
theorem fourth_term_of_geometric_progression (a₁ a₂ a₃ a₄ : ℝ) : 
  a₁ = 2^(1/4) → a₂ = 2^(1/8) → a₃ = 2^(1/16) → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) →
  a₄ = 2^(1/32) := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1696_169655


namespace NUMINAMATH_CALUDE_bruce_shopping_theorem_l1696_169651

/-- Calculates the remaining money after Bruce's shopping trip. -/
def remaining_money (initial_amount shirt_price num_shirts pants_price : ℕ) : ℕ :=
  initial_amount - (shirt_price * num_shirts + pants_price)

/-- Theorem stating that Bruce has $20 left after his shopping trip. -/
theorem bruce_shopping_theorem :
  remaining_money 71 5 5 26 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bruce_shopping_theorem_l1696_169651


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_12_l1696_169656

theorem tan_alpha_plus_pi_12 (α : Real) 
  (h : Real.sin α = 3 * Real.sin (α + π/6)) : 
  Real.tan (α + π/12) = 2 * Real.sqrt 3 - 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_12_l1696_169656


namespace NUMINAMATH_CALUDE_max_m_value_l1696_169679

def f (x : ℝ) := x^3 - 3*x^2

theorem max_m_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f x ∈ Set.Icc (-4) 0) →
  (∃ x ∈ Set.Icc (-1) m, f x = -4) →
  (∃ x ∈ Set.Icc (-1) m, f x = 0) →
  m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1696_169679


namespace NUMINAMATH_CALUDE_bass_strings_l1696_169626

theorem bass_strings (num_basses : ℕ) (num_guitars : ℕ) (num_8string_guitars : ℕ) 
  (guitar_strings : ℕ) (total_strings : ℕ) :
  num_basses = 3 →
  num_guitars = 2 * num_basses →
  guitar_strings = 6 →
  num_8string_guitars = num_guitars - 3 →
  total_strings = 72 →
  ∃ bass_strings : ℕ, 
    bass_strings * num_basses + guitar_strings * num_guitars + 8 * num_8string_guitars = total_strings ∧
    bass_strings = 4 :=
by sorry

end NUMINAMATH_CALUDE_bass_strings_l1696_169626


namespace NUMINAMATH_CALUDE_reflection_point_l1696_169657

/-- A function that passes through a given point when shifted -/
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

/-- Reflection of a function across the x-axis -/
def reflect_x (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => -f x

/-- A function passes through a point -/
def function_at_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

theorem reflection_point (f : ℝ → ℝ) :
  passes_through f 3 2 →
  function_at_point (reflect_x f) 4 (-2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_point_l1696_169657


namespace NUMINAMATH_CALUDE_camping_probability_l1696_169670

theorem camping_probability (p_rain p_tents_on_time : ℝ) : 
  p_rain = 1 / 2 →
  p_tents_on_time = 1 / 2 →
  (p_rain * (1 - p_tents_on_time)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_camping_probability_l1696_169670


namespace NUMINAMATH_CALUDE_james_weekly_pistachio_expense_l1696_169686

/-- Represents the cost of pistachios in dollars per can. -/
def cost_per_can : ℝ := 10

/-- Represents the amount of pistachios in ounces per can. -/
def ounces_per_can : ℝ := 5

/-- Represents the amount of pistachios James eats in ounces every 5 days. -/
def ounces_per_five_days : ℝ := 30

/-- Represents the number of days in a week. -/
def days_in_week : ℝ := 7

/-- Proves that James spends $84 per week on pistachios. -/
theorem james_weekly_pistachio_expense : 
  (cost_per_can / ounces_per_can) * (ounces_per_five_days / 5) * days_in_week = 84 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_pistachio_expense_l1696_169686


namespace NUMINAMATH_CALUDE_longest_chord_of_circle_with_radius_five_l1696_169640

/-- A circle with a given radius. -/
structure Circle where
  radius : ℝ

/-- The longest chord of a circle is its diameter, which is twice the radius. -/
def longestChordLength (c : Circle) : ℝ := 2 * c.radius

theorem longest_chord_of_circle_with_radius_five :
  ∃ (c : Circle), c.radius = 5 ∧ longestChordLength c = 10 := by
  sorry

end NUMINAMATH_CALUDE_longest_chord_of_circle_with_radius_five_l1696_169640


namespace NUMINAMATH_CALUDE_range_of_c_l1696_169666

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^y < c^x

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + x + (1/2) * c > 0

theorem range_of_c (c : ℝ) (h_c : c > 0) 
  (h_or : p c ∨ q c) (h_not_and : ¬(p c ∧ q c)) : 
  c ∈ Set.Ioc 0 (1/2) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l1696_169666


namespace NUMINAMATH_CALUDE_arccos_cos_eq_two_thirds_x_l1696_169618

theorem arccos_cos_eq_two_thirds_x (x : Real) :
  0 ≤ x ∧ x ≤ (3 * Real.pi / 2) →
  (Real.arccos (Real.cos x) = 2 * x / 3) ↔ (x = 0 ∨ x = 6 * Real.pi / 5 ∨ x = 12 * Real.pi / 5) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_two_thirds_x_l1696_169618


namespace NUMINAMATH_CALUDE_additional_tank_capacity_l1696_169685

theorem additional_tank_capacity
  (existing_tanks : ℕ)
  (fish_per_existing_tank : ℕ)
  (additional_tanks : ℕ)
  (total_fish : ℕ)
  (h1 : existing_tanks = 3)
  (h2 : fish_per_existing_tank = 15)
  (h3 : additional_tanks = 3)
  (h4 : total_fish = 75) :
  (total_fish - existing_tanks * fish_per_existing_tank) / additional_tanks = 10 :=
by sorry

end NUMINAMATH_CALUDE_additional_tank_capacity_l1696_169685


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1696_169693

theorem election_winner_percentage (winner_votes loser_votes : ℕ) : 
  winner_votes = 750 →
  winner_votes - loser_votes = 500 →
  (winner_votes : ℚ) / (winner_votes + loser_votes : ℚ) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1696_169693


namespace NUMINAMATH_CALUDE_floor_abs_sum_abs_floor_l1696_169632

theorem floor_abs_sum_abs_floor : ⌊|(-5.7:ℝ)|⌋ + |⌊(-5.7:ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_abs_floor_l1696_169632


namespace NUMINAMATH_CALUDE_cad_to_jpy_exchange_l1696_169644

/-- The exchange rate from Canadian dollars (CAD) to Japanese yen (JPY) -/
def exchange_rate (cad : ℚ) (jpy : ℚ) : Prop :=
  5000 / 60 = jpy / cad

/-- The rounded exchange rate for 1 CAD in JPY -/
def rounded_rate (rate : ℚ) : ℕ :=
  (rate + 1/2).floor.toNat

theorem cad_to_jpy_exchange :
  ∃ (rate : ℚ), exchange_rate 1 rate ∧ rounded_rate rate = 83 := by
  sorry

end NUMINAMATH_CALUDE_cad_to_jpy_exchange_l1696_169644


namespace NUMINAMATH_CALUDE_sqrt_product_equals_27_l1696_169607

theorem sqrt_product_equals_27 (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (6 * x) * Real.sqrt (9 * x) = 27) : 
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_27_l1696_169607


namespace NUMINAMATH_CALUDE_range_of_a_l1696_169663

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1 / (x - 1) < 1
def q (x a : ℝ) : Prop := x^2 + (a - 1) * x - a > 0

-- Define the property that p is sufficient but not necessary for q
def p_sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, p_sufficient_not_necessary a ↔ -2 < a ∧ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1696_169663


namespace NUMINAMATH_CALUDE_sequence_existence_and_extension_l1696_169661

theorem sequence_existence_and_extension (m : ℕ) (hm : m ≥ 2) :
  (∃ x : ℕ → ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) ∧
  (∀ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
    ∃ y : ℤ → ℕ, (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
               (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2 * m → y i = x i)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_and_extension_l1696_169661


namespace NUMINAMATH_CALUDE_hotel_visit_permutations_l1696_169604

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

def constrained_permutations (n : ℕ) : ℕ :=
  number_of_permutations n / 4

theorem hotel_visit_permutations :
  constrained_permutations 5 = 30 := by sorry

end NUMINAMATH_CALUDE_hotel_visit_permutations_l1696_169604


namespace NUMINAMATH_CALUDE_complex_division_equality_l1696_169610

theorem complex_division_equality : (3 - I) / (2 + I) = 1 - I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l1696_169610


namespace NUMINAMATH_CALUDE_opposite_absolute_square_l1696_169609

theorem opposite_absolute_square (x y : ℝ) : 
  (|x - 2| = -(y + 7)^2 ∨ -(x - 2) = (y + 7)^2) → y^x = 49 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_square_l1696_169609


namespace NUMINAMATH_CALUDE_ashoks_marks_l1696_169681

theorem ashoks_marks (total_subjects : ℕ) (average_6_subjects : ℝ) (marks_6th_subject : ℝ) :
  total_subjects = 6 →
  average_6_subjects = 80 →
  marks_6th_subject = 110 →
  let total_marks := average_6_subjects * total_subjects
  let marks_5_subjects := total_marks - marks_6th_subject
  let average_5_subjects := marks_5_subjects / 5
  average_5_subjects = 74 := by
sorry

end NUMINAMATH_CALUDE_ashoks_marks_l1696_169681


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l1696_169643

/-- The time required for bacteria growth under specific conditions -/
theorem bacteria_growth_time (initial_count : ℕ) (final_count : ℕ) (growth_factor : ℕ) (growth_time : ℕ) (total_time : ℕ) : 
  initial_count = 200 →
  final_count = 145800 →
  growth_factor = 3 →
  growth_time = 3 →
  (initial_count * growth_factor ^ (total_time / growth_time) = final_count) →
  total_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l1696_169643


namespace NUMINAMATH_CALUDE_triangle_properties_l1696_169624

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h : area t = t.a^2 / 2) : 
  (Real.tan t.A = 2 * t.a^2 / (t.b^2 + t.c^2 - t.a^2)) ∧ 
  (∃ (x : ℝ), x = Real.sqrt 5 ∧ ∀ (y : ℝ), t.c / t.b + t.b / t.c ≤ x) ∧
  (∃ (m : ℝ), ∀ (x : ℝ), m ≤ t.b * t.c / t.a^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1696_169624


namespace NUMINAMATH_CALUDE_fourth_power_divisor_count_l1696_169667

theorem fourth_power_divisor_count (n : ℕ+) : ∃ d : ℕ, 
  (∀ k : ℕ, k ∣ n^4 ↔ k ≤ d) ∧ d % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_divisor_count_l1696_169667


namespace NUMINAMATH_CALUDE_pool_filling_time_l1696_169649

/-- Proves that filling a pool of 15,000 gallons with four hoses (two at 2 gal/min, two at 3 gal/min) takes 25 hours -/
theorem pool_filling_time : 
  let pool_volume : ℝ := 15000
  let hose_rate_1 : ℝ := 2
  let hose_rate_2 : ℝ := 3
  let num_hoses_1 : ℕ := 2
  let num_hoses_2 : ℕ := 2
  let total_rate : ℝ := hose_rate_1 * num_hoses_1 + hose_rate_2 * num_hoses_2
  let fill_time_minutes : ℝ := pool_volume / total_rate
  let fill_time_hours : ℝ := fill_time_minutes / 60
  fill_time_hours = 25 := by
sorry


end NUMINAMATH_CALUDE_pool_filling_time_l1696_169649


namespace NUMINAMATH_CALUDE_people_disliking_both_tv_and_games_l1696_169653

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 25 / 100
def both_dislike_percentage : ℚ := 15 / 100

theorem people_disliking_both_tv_and_games :
  ⌊(tv_dislike_percentage * total_surveyed : ℚ) * both_dislike_percentage⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_people_disliking_both_tv_and_games_l1696_169653


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l1696_169621

def total_cards : ℕ := 12
def cards_per_color : ℕ := 4
def num_colors : ℕ := 3
def num_numbers : ℕ := 4

def winning_pairs : ℕ := 
  (num_colors * (cards_per_color.choose 2)) + (num_numbers * (num_colors.choose 2))

def total_pairs : ℕ := total_cards.choose 2

theorem probability_of_winning_pair :
  (winning_pairs : ℚ) / total_pairs = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l1696_169621


namespace NUMINAMATH_CALUDE_cafeteria_choices_l1696_169605

theorem cafeteria_choices (num_dishes : ℕ) (num_students : ℕ) : 
  num_dishes = 5 → num_students = 3 → (num_dishes ^ num_students) = 125 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_choices_l1696_169605


namespace NUMINAMATH_CALUDE_cross_ratio_preserving_is_projective_l1696_169646

/-- A mapping between two lines -/
structure LineMapping (α : Type*) where
  to_fun : α → α

/-- The cross ratio of four points -/
def cross_ratio {α : Type*} [Field α] (x y z w : α) : α :=
  ((x - z) * (y - w)) / ((x - w) * (y - z))

/-- A mapping preserves cross ratio -/
def preserves_cross_ratio {α : Type*} [Field α] (f : LineMapping α) : Prop :=
  ∀ (x y z w : α), cross_ratio (f.to_fun x) (f.to_fun y) (f.to_fun z) (f.to_fun w) = cross_ratio x y z w

/-- Definition of a projective transformation -/
def is_projective {α : Type*} [Field α] (f : LineMapping α) : Prop :=
  ∃ (a b c d : α), (a * d - b * c ≠ 0) ∧
    (∀ x, f.to_fun x = (a * x + b) / (c * x + d))

/-- Main theorem: A cross-ratio preserving mapping is projective -/
theorem cross_ratio_preserving_is_projective {α : Type*} [Field α] (f : LineMapping α) :
  preserves_cross_ratio f → is_projective f :=
sorry

end NUMINAMATH_CALUDE_cross_ratio_preserving_is_projective_l1696_169646


namespace NUMINAMATH_CALUDE_agnes_flight_cost_l1696_169641

/-- Represents the cost structure for different transportation modes -/
structure TransportCost where
  busCostPerKm : ℝ
  airplaneCostPerKm : ℝ
  airplaneBookingFee : ℝ

/-- Represents the distances between cities -/
structure CityDistances where
  xToY : ℝ
  xToZ : ℝ

/-- Calculates the cost of an airplane trip -/
def airplaneTripCost (cost : TransportCost) (distance : ℝ) : ℝ :=
  cost.airplaneBookingFee + cost.airplaneCostPerKm * distance

theorem agnes_flight_cost (cost : TransportCost) (distances : CityDistances) :
  cost.busCostPerKm = 0.20 →
  cost.airplaneCostPerKm = 0.12 →
  cost.airplaneBookingFee = 120 →
  distances.xToY = 4500 →
  distances.xToZ = 4000 →
  airplaneTripCost cost distances.xToY = 660 := by
  sorry


end NUMINAMATH_CALUDE_agnes_flight_cost_l1696_169641


namespace NUMINAMATH_CALUDE_orange_harvest_orange_harvest_solution_l1696_169664

theorem orange_harvest (discarded : ℕ) (days : ℕ) (remaining : ℕ) : ℕ :=
  let harvested := (remaining + days * discarded) / days
  harvested

theorem orange_harvest_solution :
  orange_harvest 71 51 153 = 74 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_orange_harvest_solution_l1696_169664


namespace NUMINAMATH_CALUDE_percentage_only_cat_owners_l1696_169691

def total_students : ℕ := 500
def dog_owners : ℕ := 120
def cat_owners : ℕ := 80
def both_owners : ℕ := 40

def only_cat_owners : ℕ := cat_owners - both_owners

theorem percentage_only_cat_owners :
  (only_cat_owners : ℚ) / total_students * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_only_cat_owners_l1696_169691


namespace NUMINAMATH_CALUDE_sqrt_difference_power_l1696_169671

theorem sqrt_difference_power (A B : ℤ) : 
  ∃ A B : ℤ, (Real.sqrt 1969 - Real.sqrt 1968) ^ 1969 = A * Real.sqrt 1969 - B * Real.sqrt 1968 ∧ 
  1969 * A^2 - 1968 * B^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_power_l1696_169671


namespace NUMINAMATH_CALUDE_president_vp_selection_ways_l1696_169697

/-- Represents the composition of a club -/
structure ClubComposition where
  total_members : Nat
  boys : Nat
  girls : Nat
  senior_boys : Nat
  senior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_president_and_vp (club : ClubComposition) : Nat :=
  let boy_pres_girl_vp := club.senior_boys * club.girls
  let girl_pres_boy_vp := club.senior_girls * (club.boys - club.senior_boys)
  boy_pres_girl_vp + girl_pres_boy_vp

/-- Theorem stating the number of ways to choose a president and vice-president -/
theorem president_vp_selection_ways (club : ClubComposition) 
  (h1 : club.total_members = 24)
  (h2 : club.boys = 8)
  (h3 : club.girls = 16)
  (h4 : club.senior_boys = 2)
  (h5 : club.senior_girls = 2)
  (h6 : club.senior_boys + club.senior_girls = 4)
  (h7 : club.boys + club.girls = club.total_members) :
  choose_president_and_vp club = 44 := by
  sorry

end NUMINAMATH_CALUDE_president_vp_selection_ways_l1696_169697


namespace NUMINAMATH_CALUDE_lcd_of_fractions_l1696_169675

def fractions : List Nat := [3, 4, 5, 8, 9, 11]

theorem lcd_of_fractions : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 8) 9) 11 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_lcd_of_fractions_l1696_169675


namespace NUMINAMATH_CALUDE_coffee_mixture_cost_theorem_l1696_169674

/-- The cost of the more expensive coffee per pound -/
def expensive_coffee_cost : ℝ := 7.28

/-- The cost of the cheaper coffee per pound -/
def cheaper_coffee_cost : ℝ := 6.42

/-- The amount of cheaper coffee in pounds -/
def cheaper_coffee_amount : ℝ := 7

/-- The amount of expensive coffee in pounds -/
def expensive_coffee_amount : ℝ := 68.25

/-- The price of the mixture per pound -/
def mixture_price : ℝ := 7.20

/-- The total amount of coffee in the mixture -/
def total_coffee_amount : ℝ := cheaper_coffee_amount + expensive_coffee_amount

theorem coffee_mixture_cost_theorem :
  cheaper_coffee_amount * cheaper_coffee_cost +
  expensive_coffee_amount * expensive_coffee_cost =
  total_coffee_amount * mixture_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_cost_theorem_l1696_169674


namespace NUMINAMATH_CALUDE_race_head_start_l1696_169614

theorem race_head_start (vA vB L H : ℝ) : 
  vA = (15 / 13) * vB →
  (L - H) / vB = L / vA - 0.4 * L / vB →
  H = (8 / 15) * L :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l1696_169614


namespace NUMINAMATH_CALUDE_tshirt_sale_revenue_l1696_169658

/-- Calculates the total money made from selling t-shirts with a discount -/
theorem tshirt_sale_revenue (original_price discount : ℕ) (num_sold : ℕ) :
  original_price = 51 →
  discount = 8 →
  num_sold = 130 →
  (original_price - discount) * num_sold = 5590 :=
by sorry

end NUMINAMATH_CALUDE_tshirt_sale_revenue_l1696_169658


namespace NUMINAMATH_CALUDE_dice_sum_theorem_l1696_169600

def Die := Fin 6

def roll_sum (d1 d2 : Die) : ℕ := d1.val + d2.val + 2

def possible_sums : Set ℕ := {n | ∃ (d1 d2 : Die), roll_sum d1 d2 = n}

theorem dice_sum_theorem : possible_sums = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_theorem_l1696_169600


namespace NUMINAMATH_CALUDE_function_is_constant_l1696_169682

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def is_continuous (f : ℝ → ℝ) : Prop := Continuous f

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 * x - 2) ≤ f x ∧ f x ≤ f (2 * x - 1)

-- State the theorem
theorem function_is_constant
  (h_continuous : is_continuous f)
  (h_inequality : satisfies_inequality f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l1696_169682


namespace NUMINAMATH_CALUDE_sandy_painting_area_l1696_169630

/-- The area Sandy needs to paint on her bedroom wall -/
def area_to_paint (wall_height wall_length bookshelf_width bookshelf_height : ℝ) : ℝ :=
  wall_height * wall_length - bookshelf_width * bookshelf_height

/-- Theorem stating that Sandy needs to paint 135 square feet -/
theorem sandy_painting_area :
  area_to_paint 10 15 3 5 = 135 := by
  sorry

#eval area_to_paint 10 15 3 5

end NUMINAMATH_CALUDE_sandy_painting_area_l1696_169630


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1696_169635

theorem simplify_polynomial (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) =
  r^3 - 4 * r^2 + 2 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1696_169635


namespace NUMINAMATH_CALUDE_prob_second_draw_3_eq_11_48_l1696_169683

-- Define the boxes and their initial contents
def box1 : Finset ℕ := {1, 1, 2, 3}
def box2 : Finset ℕ := {1, 1, 3}
def box3 : Finset ℕ := {1, 1, 1, 2, 2}

-- Define the probability of drawing a ball from a box
def prob_draw (box : Finset ℕ) (label : ℕ) : ℚ :=
  (box.filter (λ x => x = label)).card / box.card

-- Define the probability of the second draw being 3
def prob_second_draw_3 : ℚ :=
  (prob_draw box1 1 * prob_draw (box1 ∪ {1}) 3) +
  (prob_draw box1 2 * prob_draw (box2 ∪ {2}) 3) +
  (prob_draw box1 3 * prob_draw (box3 ∪ {3}) 3)

-- Theorem statement
theorem prob_second_draw_3_eq_11_48 : prob_second_draw_3 = 11 / 48 := by
  sorry


end NUMINAMATH_CALUDE_prob_second_draw_3_eq_11_48_l1696_169683


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1696_169665

-- Define the sample space
def Ω : Type := Unit

-- Define the event of hitting the target on the first shot
def hit1 : Set Ω := sorry

-- Define the event of hitting the target on the second shot
def hit2 : Set Ω := sorry

-- Define the event of hitting the target at least once
def hitAtLeastOnce : Set Ω := hit1 ∪ hit2

-- Define the event of missing the target both times
def missBoth : Set Ω := (hit1 ∪ hit2)ᶜ

-- Theorem stating that hitAtLeastOnce and missBoth are mutually exclusive
theorem mutually_exclusive_events : 
  hitAtLeastOnce ∩ missBoth = ∅ ∧ hitAtLeastOnce ∪ missBoth = Set.univ :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1696_169665


namespace NUMINAMATH_CALUDE_cafe_problem_l1696_169696

/-- The number of local offices that ordered sandwiches -/
def num_offices : ℕ := 3

/-- The number of sandwiches ordered by each office -/
def sandwiches_per_office : ℕ := 10

/-- The number of sandwiches ordered by each customer in half of the group -/
def sandwiches_per_customer : ℕ := 4

/-- The total number of sandwiches made by the café -/
def total_sandwiches : ℕ := 54

/-- The number of customers in the group that arrived at the café -/
def num_customers : ℕ := 12

theorem cafe_problem :
  num_offices * sandwiches_per_office +
  (num_customers / 2) * sandwiches_per_customer =
  total_sandwiches :=
by sorry

end NUMINAMATH_CALUDE_cafe_problem_l1696_169696


namespace NUMINAMATH_CALUDE_smallest_nonzero_y_value_l1696_169608

theorem smallest_nonzero_y_value (y : ℝ) : 
  y > 0 ∧ Real.sqrt (6 * y + 3) = 3 * y + 1 → y ≥ Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonzero_y_value_l1696_169608


namespace NUMINAMATH_CALUDE_interior_angles_sum_l1696_169629

theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1440) →
  (180 * ((n + 3) - 2) = 1980) :=
by sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l1696_169629


namespace NUMINAMATH_CALUDE_hexagon_area_from_triangle_l1696_169627

-- Define the regular hexagon ABCDEF
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

-- Define points G, H, I
def G (h : RegularHexagon) : ℝ × ℝ := sorry
def H (h : RegularHexagon) : ℝ × ℝ := sorry
def I (h : RegularHexagon) : ℝ × ℝ := sorry

-- Define the area of a triangle
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the area of a hexagon
def hexagon_area (h : RegularHexagon) : ℝ := sorry

-- Theorem statement
theorem hexagon_area_from_triangle (h : RegularHexagon) :
  triangle_area (G h) (H h) (I h) = 100 → hexagon_area h = 600 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_from_triangle_l1696_169627


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1696_169673

theorem circle_line_intersection (m : ℝ) :
  (∃! (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    (p1.1^2 + p1.2^2 = m) ∧ (p2.1^2 + p2.2^2 = m) ∧ (p3.1^2 + p3.2^2 = m) ∧
    |p1.1 - p1.2 + Real.sqrt 2| / Real.sqrt 2 = 1 ∧
    |p2.1 - p2.2 + Real.sqrt 2| / Real.sqrt 2 = 1 ∧
    |p3.1 - p3.2 + Real.sqrt 2| / Real.sqrt 2 = 1) →
  m = 4 :=
by sorry


end NUMINAMATH_CALUDE_circle_line_intersection_l1696_169673


namespace NUMINAMATH_CALUDE_june_first_is_friday_l1696_169672

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with given properties -/
structure Month where
  days : Nat
  firstDay : DayOfWeek
  mondayCount : Nat
  thursdayCount : Nat

/-- Function to determine if a month satisfies the given conditions -/
def satisfiesConditions (m : Month) : Prop :=
  m.days = 30 ∧ m.mondayCount = 3 ∧ m.thursdayCount = 3

/-- Theorem stating that a month satisfying the conditions must start on a Friday -/
theorem june_first_is_friday (m : Month) :
  satisfiesConditions m → m.firstDay = DayOfWeek.Friday :=
by
  sorry


end NUMINAMATH_CALUDE_june_first_is_friday_l1696_169672


namespace NUMINAMATH_CALUDE_existence_of_special_point_set_l1696_169625

/-- A closed region bounded by a regular polygon -/
structure RegularPolygonRegion where
  vertices : Set (ℝ × ℝ)
  is_regular : Bool
  is_closed : Bool

/-- A set of points in the plane -/
def PointSet := Set (ℝ × ℝ)

/-- Predicate to check if a set of points can be covered by a region -/
def IsCovered (S : PointSet) (C : RegularPolygonRegion) : Prop := sorry

/-- Predicate to check if any n points from a set can be covered by a region -/
def AnyNPointsCovered (S : PointSet) (C : RegularPolygonRegion) (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem existence_of_special_point_set (C : RegularPolygonRegion) (n : ℕ) :
  ∃ (S : PointSet), AnyNPointsCovered S C n ∧ ¬IsCovered S C := by sorry

end NUMINAMATH_CALUDE_existence_of_special_point_set_l1696_169625


namespace NUMINAMATH_CALUDE_fraction_equality_l1696_169642

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - 3 * b^2

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a + 2 * b - 2 * a * b^2

-- Theorem statement
theorem fraction_equality :
  (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1696_169642


namespace NUMINAMATH_CALUDE_sin_two_theta_equals_three_fourths_l1696_169699

theorem sin_two_theta_equals_three_fourths (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2)
  (h2 : Real.sin (π * Real.cos θ) = Real.cos (π * Real.sin θ)) : 
  Real.sin (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_theta_equals_three_fourths_l1696_169699


namespace NUMINAMATH_CALUDE_problem_3_l1696_169612

theorem problem_3 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a^2 + b^2 = 6*a*b) :
  (a + b) / (a - b) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l1696_169612


namespace NUMINAMATH_CALUDE_cs_physics_overlap_l1696_169634

/-- Represents the fraction of students in one club who also attend another club -/
def club_overlap (club1 club2 : Type) : ℚ := sorry

theorem cs_physics_overlap :
  let m := club_overlap Mathematics Physics
  let c := club_overlap Mathematics ComputerScience
  let p := club_overlap Physics Mathematics
  let q := club_overlap Physics ComputerScience
  let r := club_overlap ComputerScience Mathematics
  m = 1/6 ∧ c = 1/8 ∧ p = 1/3 ∧ q = 1/5 ∧ r = 1/7 →
  club_overlap ComputerScience Physics = 4/35 :=
sorry

end NUMINAMATH_CALUDE_cs_physics_overlap_l1696_169634


namespace NUMINAMATH_CALUDE_inverse_proportion_l1696_169669

/-- Given that p and q are inversely proportional, prove that if p = 30 when q = 4, 
    then p = 240/11 when q = 5.5 -/
theorem inverse_proportion (p q : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, p * q = k) 
    (h1 : p = 30 ∧ q = 4) : 
    (p = 240/11 ∧ q = 5.5) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1696_169669


namespace NUMINAMATH_CALUDE_second_graders_count_l1696_169619

theorem second_graders_count (kindergartners : ℕ) (first_graders : ℕ) (total_students : ℕ) 
  (h1 : kindergartners = 34)
  (h2 : first_graders = 48)
  (h3 : total_students = 120) :
  total_students - (kindergartners + first_graders) = 38 := by
  sorry

end NUMINAMATH_CALUDE_second_graders_count_l1696_169619


namespace NUMINAMATH_CALUDE_max_correct_answers_l1696_169650

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_points = 5 →
  incorrect_points = -2 →
  total_score = 150 →
  ∃ (correct blank incorrect : ℕ),
    correct + blank + incorrect = total_questions ∧
    correct_points * correct + incorrect_points * incorrect = total_score ∧
    correct ≤ 38 ∧
    ∀ (c : ℕ), c > 38 →
      ¬(∃ (b i : ℕ), c + b + i = total_questions ∧
        correct_points * c + incorrect_points * i = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1696_169650


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l1696_169636

theorem smallest_n_for_integer_roots : ∃ (x y : ℤ),
  x^2 - 91*x + 2014 = 0 ∧ y^2 - 91*y + 2014 = 0 ∧
  (∀ (n : ℕ) (a b : ℤ), n < 91 → (a^2 - n*a + 2014 = 0 ∧ b^2 - n*b + 2014 = 0) → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l1696_169636


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1696_169606

theorem cubic_equation_solution :
  {x : ℝ | x^3 + 6*x^2 + 11*x + 6 = 12} = {-3, -2, -1} := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1696_169606


namespace NUMINAMATH_CALUDE_g_of_three_equals_five_l1696_169633

theorem g_of_three_equals_five (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = 2 * x + 3) :
  g 3 = 5 := by sorry

end NUMINAMATH_CALUDE_g_of_three_equals_five_l1696_169633


namespace NUMINAMATH_CALUDE_min_x_coord_midpoint_l1696_169690

/-- Given a segment AB of length 3 with endpoints on the parabola y^2 = x,
    the minimum x-coordinate of the midpoint M of AB is 5/4 -/
theorem min_x_coord_midpoint (A B M : ℝ × ℝ) :
  (A.2^2 = A.1) →  -- A is on the parabola y^2 = x
  (B.2^2 = B.1) →  -- B is on the parabola y^2 = x
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 9 →  -- AB has length 3
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  M.1 ≥ 5/4 :=
sorry

end NUMINAMATH_CALUDE_min_x_coord_midpoint_l1696_169690


namespace NUMINAMATH_CALUDE_adult_admission_price_l1696_169637

/-- Proves that the admission price for adults is 8 dollars given the specified conditions -/
theorem adult_admission_price
  (total_amount : ℕ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (children_price : ℕ)
  (h1 : total_amount = 201)
  (h2 : total_tickets = 33)
  (h3 : children_tickets = 21)
  (h4 : children_price = 5) :
  (total_amount - children_tickets * children_price) / (total_tickets - children_tickets) = 8 := by
  sorry


end NUMINAMATH_CALUDE_adult_admission_price_l1696_169637


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1696_169687

theorem sphere_surface_area (c : Real) (h : c = 2 * Real.pi) :
  ∃ (r : Real), 
    c = 2 * Real.pi * r ∧ 
    4 * Real.pi * r^2 = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1696_169687


namespace NUMINAMATH_CALUDE_children_events_count_l1696_169660

theorem children_events_count (cupcakes_per_event : ℝ) (total_cupcakes : ℕ) 
  (h1 : cupcakes_per_event = 96.0)
  (h2 : total_cupcakes = 768) :
  (total_cupcakes : ℝ) / cupcakes_per_event = 8 := by
  sorry

end NUMINAMATH_CALUDE_children_events_count_l1696_169660


namespace NUMINAMATH_CALUDE_password_probability_l1696_169695

/-- Represents the set of symbols used in the password -/
def SymbolSet : Finset Char := {'!', '@', '#', '$', '%'}

/-- Represents the set of favorable symbols -/
def FavorableSymbols : Finset Char := {'$', '%', '@'}

/-- Represents the set of two-digit numbers (00 to 99) -/
def TwoDigitNumbers : Finset Nat := Finset.range 100

/-- Represents the set of even two-digit numbers -/
def EvenTwoDigitNumbers : Finset Nat := TwoDigitNumbers.filter (fun n => n % 2 = 0)

/-- The probability of Alice's password meeting the specific criteria -/
theorem password_probability : 
  (EvenTwoDigitNumbers.card : ℚ) / TwoDigitNumbers.card * 
  (FavorableSymbols.card : ℚ) / SymbolSet.card * 
  (EvenTwoDigitNumbers.card : ℚ) / TwoDigitNumbers.card = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_password_probability_l1696_169695


namespace NUMINAMATH_CALUDE_problem_solution_l1696_169648

open Real

def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

def q : Prop := ∃ x₀ > 0, 8 * x₀ + 1 / (2 * x₀) ≤ 4

theorem problem_solution : (¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1696_169648


namespace NUMINAMATH_CALUDE_elise_remaining_money_l1696_169628

/-- Calculates the remaining money for Elise --/
def remaining_money (initial saved comic_book puzzle : ℕ) : ℕ :=
  initial + saved - (comic_book + puzzle)

/-- Theorem: Elise's remaining money is $1 --/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_elise_remaining_money_l1696_169628


namespace NUMINAMATH_CALUDE_alpha_beta_range_l1696_169602

-- Define the curve E
def curve_E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line l1
def line_l1 (k x y : ℝ) : Prop := y = k * (x + 2)

-- Define the intersection points A and B
def intersection_points (x1 y1 x2 y2 k : ℝ) : Prop :=
  curve_E x1 y1 ∧ curve_E x2 y2 ∧ 
  line_l1 k x1 y1 ∧ line_l1 k x2 y2 ∧
  x1 ≠ x2

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the relationship between α, β, and the points
def alpha_beta_relation (α β x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  α = (1 - x1) / (x3 - 1) ∧
  β = (1 - x2) / (x4 - 1) ∧
  curve_E x3 y3 ∧ curve_E x4 y4

-- Main theorem
theorem alpha_beta_range :
  ∀ (k x1 y1 x2 y2 x3 y3 x4 y4 α β : ℝ),
    0 < k^2 ∧ k^2 < 1/2 →
    intersection_points x1 y1 x2 y2 k →
    alpha_beta_relation α β x1 y1 x2 y2 x3 y3 x4 y4 →
    6 < α + β ∧ α + β < 10 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_range_l1696_169602


namespace NUMINAMATH_CALUDE_average_age_first_group_l1696_169613

theorem average_age_first_group (total_students : Nat) (avg_age_all : ℝ) 
  (first_group_size second_group_size : Nat) (avg_age_second_group : ℝ) 
  (age_last_student : ℝ) :
  total_students = 15 →
  avg_age_all = 15 →
  first_group_size = 7 →
  second_group_size = 7 →
  avg_age_second_group = 16 →
  age_last_student = 15 →
  (total_students * avg_age_all - second_group_size * avg_age_second_group - age_last_student) / first_group_size = 14 := by
sorry

end NUMINAMATH_CALUDE_average_age_first_group_l1696_169613


namespace NUMINAMATH_CALUDE_kabadi_players_count_l1696_169611

/-- Represents the number of players in different categories -/
structure PlayerCounts where
  total : ℕ
  khoKhoOnly : ℕ
  bothGames : ℕ

/-- Calculates the number of players who play kabadi -/
def kabadiPlayers (counts : PlayerCounts) : ℕ :=
  counts.total - counts.khoKhoOnly + counts.bothGames

/-- Theorem stating the number of kabadi players given the conditions -/
theorem kabadi_players_count (counts : PlayerCounts) 
  (h1 : counts.total = 30)
  (h2 : counts.khoKhoOnly = 20)
  (h3 : counts.bothGames = 5) :
  kabadiPlayers counts = 15 := by
  sorry


end NUMINAMATH_CALUDE_kabadi_players_count_l1696_169611


namespace NUMINAMATH_CALUDE_sally_bought_48_eggs_l1696_169680

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Sally bought -/
def dozens_bought : ℕ := 4

/-- Theorem: Sally bought 48 eggs -/
theorem sally_bought_48_eggs : dozens_bought * eggs_per_dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_48_eggs_l1696_169680


namespace NUMINAMATH_CALUDE_unique_solution_for_m_l1696_169601

theorem unique_solution_for_m :
  ∀ (x y m : ℚ),
  (2 * x + y = 3 * m) →
  (x - 4 * y = -2 * m) →
  (y + 2 * m = 1 + x) →
  m = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_m_l1696_169601


namespace NUMINAMATH_CALUDE_bird_ratio_l1696_169668

/-- Represents the number of birds caught by a cat during the day. -/
def birds_day : ℕ := 8

/-- Represents the total number of birds caught by a cat. -/
def birds_total : ℕ := 24

/-- Represents the number of birds caught by a cat at night. -/
def birds_night : ℕ := birds_total - birds_day

/-- The theorem states that the ratio of birds caught at night to birds caught during the day is 2:1. -/
theorem bird_ratio : birds_night / birds_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_ratio_l1696_169668


namespace NUMINAMATH_CALUDE_isabella_hair_length_l1696_169616

/-- Calculates the length of hair after a given time period. -/
def hair_length (initial_length : ℝ) (growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_length + growth_rate * months

/-- Theorem stating that Isabella's hair length after y months is 18 + xy -/
theorem isabella_hair_length (x y : ℝ) :
  hair_length 18 x y = 18 + x * y := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_length_l1696_169616


namespace NUMINAMATH_CALUDE_debbys_flour_amount_l1696_169662

/-- Calculates the final amount of flour Debby has -/
def final_flour_amount (initial : ℕ) (used : ℕ) (given : ℕ) (bought : ℕ) : ℕ :=
  initial - used - given + bought

/-- Proves that Debby's final amount of flour is 11 pounds -/
theorem debbys_flour_amount :
  final_flour_amount 12 3 2 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_debbys_flour_amount_l1696_169662


namespace NUMINAMATH_CALUDE_train_crossing_time_l1696_169603

/-- Time taken for a train to cross a man running in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 220 →
  train_speed = 80 * 1000 / 3600 →
  man_speed = 8 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1696_169603


namespace NUMINAMATH_CALUDE_fraction_relation_l1696_169639

theorem fraction_relation (x y z w : ℚ) 
  (h1 : x / y = 12)
  (h2 : z / y = 4)
  (h3 : z / w = 3 / 4) :
  w / x = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l1696_169639


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1696_169689

def is_valid_pair (A B : ℕ+) : Prop :=
  12 ∣ A ∧ 12 ∣ B ∧
  20 ∣ A ∧ 20 ∣ B ∧
  45 ∣ A ∧ 45 ∣ B ∧
  Nat.lcm A B = 4320

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ+ × ℕ+)), 
    (∀ p ∈ pairs, is_valid_pair p.1 p.2) ∧
    (∀ A B, is_valid_pair A B → (A, B) ∈ pairs) ∧
    pairs.card = 11 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1696_169689


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1696_169645

open Set

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1696_169645


namespace NUMINAMATH_CALUDE_hall_breadth_l1696_169694

/-- The breadth of a hall given its length, number of stones, and stone dimensions. -/
theorem hall_breadth (hall_length : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) : 
  hall_length = 36 →
  num_stones = 3600 →
  stone_length = 0.3 →
  stone_width = 0.5 →
  hall_length * (num_stones * stone_length * stone_width / hall_length) = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_breadth_l1696_169694


namespace NUMINAMATH_CALUDE_simplify_negative_fraction_power_l1696_169654

theorem simplify_negative_fraction_power : 
  ((-1 : ℝ) / 343) ^ (-(2 : ℝ) / 3) = 49 := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_fraction_power_l1696_169654


namespace NUMINAMATH_CALUDE_xy9z_divisible_by_132_l1696_169631

def is_form_xy9z (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ n = x * 1000 + y * 100 + 90 + z

def valid_numbers : Set ℕ := {3696, 4092, 6996, 7392}

theorem xy9z_divisible_by_132 :
  ∀ n : ℕ, is_form_xy9z n ∧ 132 ∣ n ↔ n ∈ valid_numbers := by sorry

end NUMINAMATH_CALUDE_xy9z_divisible_by_132_l1696_169631


namespace NUMINAMATH_CALUDE_winnie_lollipops_left_l1696_169647

/-- The number of lollipops Winnie has left after distributing them equally among her friends -/
def lollipops_left (cherry wintergreen grape shrimp friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp) % friends

theorem winnie_lollipops_left :
  lollipops_left 32 150 7 280 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_left_l1696_169647


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1696_169615

theorem solution_set_of_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1696_169615


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1696_169677

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ = k ∧ x₂^2 - x₂ = k) → k > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1696_169677


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l1696_169684

/-- The volume of a rectangular solid with given face areas and a dimension relation -/
theorem rectangular_solid_volume (a b c : ℝ) 
  (side_area : a * b = 15)
  (front_area : b * c = 10)
  (top_area : a * c = 6)
  (dimension_relation : b = 2 * a ∨ a = 2 * b ∨ c = 2 * a ∨ a = 2 * c ∨ c = 2 * b ∨ b = 2 * c) :
  a * b * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l1696_169684


namespace NUMINAMATH_CALUDE_equation_solution_l1696_169623

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1696_169623


namespace NUMINAMATH_CALUDE_archibald_apple_consumption_l1696_169652

def apples_per_day_first_two_weeks (x : ℝ) : Prop :=
  let first_two_weeks := 14 * x
  let next_three_weeks := 14 * x
  let last_two_weeks := 14 * 3
  let total_apples := 7 * 10
  first_two_weeks + next_three_weeks + last_two_weeks = total_apples

theorem archibald_apple_consumption : 
  ∃ x : ℝ, apples_per_day_first_two_weeks x ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_archibald_apple_consumption_l1696_169652


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l1696_169622

theorem nina_widget_purchase (initial_money : ℕ) (initial_widgets : ℕ) (price_reduction : ℕ) 
  (h1 : initial_money = 24)
  (h2 : initial_widgets = 6)
  (h3 : price_reduction = 1)
  : (initial_money / (initial_money / initial_widgets - price_reduction) : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_purchase_l1696_169622


namespace NUMINAMATH_CALUDE_mary_nickels_theorem_l1696_169692

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem mary_nickels_theorem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 7) 
  (h2 : final = 12) : 
  nickels_from_dad initial final = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_theorem_l1696_169692


namespace NUMINAMATH_CALUDE_smallest_value_between_zero_and_one_l1696_169698

theorem smallest_value_between_zero_and_one (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^2 < x ∧ x^2 < Real.sqrt x ∧ x^2 < 3*x ∧ x^2 < 1/x := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_between_zero_and_one_l1696_169698


namespace NUMINAMATH_CALUDE_system_solution_l1696_169638

theorem system_solution (u v : ℝ) : 
  u + v = 10 ∧ 3 * u - 2 * v = 5 → u = 5 ∧ v = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1696_169638


namespace NUMINAMATH_CALUDE_simplify_expression_l1696_169617

theorem simplify_expression (x : ℝ) : 3*x + 4*x - 2*x + 6*x - 3*x = 8*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1696_169617


namespace NUMINAMATH_CALUDE_circle_intersection_parallelogram_l1696_169688

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two circles intersect non-tangentially -/
def nonTangentialIntersection (c1 c2 : Circle) : Prop :=
  sorry

/-- Finds the intersection points of two circles -/
def circleIntersection (c1 c2 : Circle) : Set Point :=
  sorry

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (a b c d : Point) : Prop :=
  sorry

theorem circle_intersection_parallelogram 
  (k1 k2 k3 : Circle)
  (P : Point)
  (A B : Point)
  (D C : Point)
  (h1 : k1.radius = k2.radius ∧ k2.radius = k3.radius)
  (h2 : nonTangentialIntersection k1 k2 ∧ nonTangentialIntersection k2 k3 ∧ nonTangentialIntersection k3 k1)
  (h3 : P ∈ circleIntersection k1 k2 ∩ circleIntersection k2 k3 ∩ circleIntersection k3 k1)
  (h4 : A = k1.center)
  (h5 : B = k2.center)
  (h6 : D ∈ circleIntersection k1 k3 ∧ D ≠ P)
  (h7 : C ∈ circleIntersection k2 k3 ∧ C ≠ P)
  : isParallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_parallelogram_l1696_169688


namespace NUMINAMATH_CALUDE_winner_received_55_percent_l1696_169620

/-- Represents an election with two candidates -/
structure Election where
  winner_votes : ℕ
  margin : ℕ

/-- Calculates the percentage of votes received by the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / ((e.winner_votes + (e.winner_votes - e.margin)) : ℚ) * 100

/-- Theorem stating that in the given election scenario, the winner received 55% of the votes -/
theorem winner_received_55_percent (e : Election) 
  (h1 : e.winner_votes = 550) 
  (h2 : e.margin = 100) : 
  winner_percentage e = 55 := by
  sorry

#eval winner_percentage ⟨550, 100⟩

end NUMINAMATH_CALUDE_winner_received_55_percent_l1696_169620


namespace NUMINAMATH_CALUDE_sum_of_roots_and_coefficients_l1696_169678

theorem sum_of_roots_and_coefficients (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  c^2 + a*c + b = 0 →
  d^2 + a*d + b = 0 →
  a^2 + c*a + d = 0 →
  b^2 + c*b + d = 0 →
  a + b + c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_and_coefficients_l1696_169678
