import Mathlib

namespace NUMINAMATH_CALUDE_vegetable_baskets_weight_l3644_364469

def num_baskets : ℕ := 5
def standard_weight : ℕ := 50
def excess_deficiency : List ℤ := [3, -6, -4, 2, -1]

theorem vegetable_baskets_weight :
  (List.sum excess_deficiency = -6) ∧
  (num_baskets * standard_weight + List.sum excess_deficiency = 244) := by
sorry

end NUMINAMATH_CALUDE_vegetable_baskets_weight_l3644_364469


namespace NUMINAMATH_CALUDE_rap_song_requests_l3644_364400

/-- Represents the number of song requests for different genres --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests --/
theorem rap_song_requests (req : SongRequests) : req.rap = 2 :=
  by
  have h1 : req.total = 30 := by sorry
  have h2 : req.electropop = req.total / 2 := by sorry
  have h3 : req.dance = req.electropop / 3 := by sorry
  have h4 : req.rock = 5 := by sorry
  have h5 : req.oldies = req.rock - 3 := by sorry
  have h6 : req.dj_choice = req.oldies / 2 := by sorry
  have h7 : req.total = req.electropop + req.dance + req.rock + req.oldies + req.dj_choice + req.rap := by sorry
  sorry

end NUMINAMATH_CALUDE_rap_song_requests_l3644_364400


namespace NUMINAMATH_CALUDE_first_class_product_rate_l3644_364474

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    calculate the overall rate of first-class products. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (h_pass_rate : pass_rate = 0.95)
  (h_first_class_rate_among_qualified : first_class_rate_among_qualified = 0.2) :
  pass_rate * first_class_rate_among_qualified = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_first_class_product_rate_l3644_364474


namespace NUMINAMATH_CALUDE_advertisement_length_main_theorem_l3644_364475

/-- Proves that the advertisement length is 20 minutes given the movie theater conditions -/
theorem advertisement_length : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (movie_length : ℕ) (replays : ℕ) (operating_time : ℕ) (ad_length : ℕ) =>
    movie_length = 90 ∧ 
    replays = 6 ∧ 
    operating_time = 660 ∧
    movie_length * replays + ad_length * replays = operating_time →
    ad_length = 20

/-- The main theorem stating the advertisement length -/
theorem main_theorem : advertisement_length 90 6 660 20 := by
  sorry

end NUMINAMATH_CALUDE_advertisement_length_main_theorem_l3644_364475


namespace NUMINAMATH_CALUDE_ship_passengers_l3644_364416

theorem ship_passengers : 
  ∀ (P : ℕ), 
    (P / 4 : ℚ) + (P / 8 : ℚ) + (P / 12 : ℚ) + (P / 6 : ℚ) + 36 = P → 
    P = 96 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l3644_364416


namespace NUMINAMATH_CALUDE_misha_older_than_tanya_l3644_364444

/-- Represents a person's age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  inv : months < 12

/-- Compares two Ages -/
def Age.lt (a b : Age) : Prop :=
  a.years < b.years ∨ (a.years = b.years ∧ a.months < b.months)

/-- Adds months to an Age -/
def Age.addMonths (a : Age) (m : ℕ) : Age :=
  { years := a.years + (a.months + m) / 12,
    months := (a.months + m) % 12,
    inv := by sorry }

/-- Subtracts months from an Age -/
def Age.subMonths (a : Age) (m : ℕ) : Age :=
  { years := a.years - (m + 11) / 12,
    months := (a.months + 12 - (m % 12)) % 12,
    inv := by sorry }

theorem misha_older_than_tanya (tanya_past misha_future : Age) :
  tanya_past.addMonths 19 = tanya_past.addMonths 19 →
  misha_future.subMonths 16 = misha_future.subMonths 16 →
  tanya_past.years = 16 →
  misha_future.years = 19 →
  Age.lt (tanya_past.addMonths 19) (misha_future.subMonths 16) := by
  sorry

end NUMINAMATH_CALUDE_misha_older_than_tanya_l3644_364444


namespace NUMINAMATH_CALUDE_m_range_characterization_l3644_364415

/-- Proposition P: The equation x^2 + mx + 1 = 0 has two distinct negative roots -/
def P (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition Q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The range of values for m satisfying the given conditions -/
def m_range (m : ℝ) : Prop := m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3

theorem m_range_characterization :
  ∀ m : ℝ, ((P m ∨ Q m) ∧ ¬(P m ∧ Q m)) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l3644_364415


namespace NUMINAMATH_CALUDE_boat_animals_correct_number_of_dogs_l3644_364485

theorem boat_animals (sheep_initial : ℕ) (cows_initial : ℕ) (sheep_drowned : ℕ) (animals_survived : ℕ) : ℕ :=
  let cows_drowned := 2 * sheep_drowned
  let sheep_survived := sheep_initial - sheep_drowned
  let cows_survived := cows_initial - cows_drowned
  let dogs := animals_survived - sheep_survived - cows_survived
  dogs

theorem correct_number_of_dogs : 
  boat_animals 20 10 3 35 = 14 := by
  sorry

end NUMINAMATH_CALUDE_boat_animals_correct_number_of_dogs_l3644_364485


namespace NUMINAMATH_CALUDE_horizontal_chord_theorem_l3644_364463

-- Define the set of valid d values
def ValidD : Set ℝ := {d | ∃ n : ℕ+, d = 1 / n}

theorem horizontal_chord_theorem (f : ℝ → ℝ) (h_cont : Continuous f) (h_end : f 0 = f 1) :
  ∀ d : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x + d ∈ Set.Icc 0 1 ∧ f x = f (x + d)) ↔ d ∈ ValidD :=
sorry

end NUMINAMATH_CALUDE_horizontal_chord_theorem_l3644_364463


namespace NUMINAMATH_CALUDE_billy_caught_three_fish_l3644_364402

/-- Represents the number of fish caught by each family member and other relevant information --/
structure FishingTrip where
  ben_fish : ℕ
  judy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ
  filets_per_fish : ℕ

/-- Calculates the number of fish Billy caught given the fishing trip information --/
def billy_fish_count (trip : FishingTrip) : ℕ :=
  let total_kept := trip.total_filets / trip.filets_per_fish
  let total_caught := total_kept + trip.thrown_back
  total_caught - trip.ben_fish - trip.judy_fish - trip.jim_fish - trip.susie_fish

/-- Theorem stating that Billy caught 3 fish given the specific conditions of the fishing trip --/
theorem billy_caught_three_fish (trip : FishingTrip) 
  (h1 : trip.ben_fish = 4)
  (h2 : trip.judy_fish = 1)
  (h3 : trip.jim_fish = 2)
  (h4 : trip.susie_fish = 5)
  (h5 : trip.thrown_back = 3)
  (h6 : trip.total_filets = 24)
  (h7 : trip.filets_per_fish = 2) :
  billy_fish_count trip = 3 := by
  sorry

#eval billy_fish_count ⟨4, 1, 2, 5, 3, 24, 2⟩

end NUMINAMATH_CALUDE_billy_caught_three_fish_l3644_364402


namespace NUMINAMATH_CALUDE_rectangle_area_l3644_364489

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 49 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 147 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3644_364489


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3644_364408

/-- The problem statement translated to Lean 4 --/
theorem circle_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (x - a)^2 + (y - (a - 3))^2 = 1) ↔ 0 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l3644_364408


namespace NUMINAMATH_CALUDE_first_applicant_better_by_850_l3644_364455

/-- Represents an applicant for a job position -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  training_months : ℕ
  training_cost_per_month : ℕ
  hiring_bonus_percent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def net_gain (a : Applicant) : ℤ :=
  a.revenue - a.salary - (a.training_months * a.training_cost_per_month) - (a.salary * a.hiring_bonus_percent / 100)

theorem first_applicant_better_by_850 :
  let first := Applicant.mk 42000 93000 3 1200 0
  let second := Applicant.mk 45000 92000 0 0 1
  net_gain first - net_gain second = 850 := by sorry

end NUMINAMATH_CALUDE_first_applicant_better_by_850_l3644_364455


namespace NUMINAMATH_CALUDE_inequality_proof_l3644_364454

theorem inequality_proof (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_sum : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3644_364454


namespace NUMINAMATH_CALUDE_shaded_probability_is_half_l3644_364468

/-- Represents an isosceles triangle with base 2 cm and height 4 cm -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  is_isosceles : base = 2 ∧ height = 4

/-- Represents the division of the triangle into 4 regions -/
structure TriangleDivision where
  triangle : IsoscelesTriangle
  num_regions : ℕ
  is_divided : num_regions = 4

/-- Represents the shading of two opposite regions -/
structure ShadedRegions where
  division : TriangleDivision
  num_shaded : ℕ
  are_opposite : num_shaded = 2

/-- The probability of the spinner landing on a shaded region -/
def shaded_probability (shaded : ShadedRegions) : ℚ :=
  shaded.num_shaded / shaded.division.num_regions

/-- Theorem stating that the probability of landing on a shaded region is 1/2 -/
theorem shaded_probability_is_half (shaded : ShadedRegions) :
  shaded_probability shaded = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_probability_is_half_l3644_364468


namespace NUMINAMATH_CALUDE_range_of_p_l3644_364486

/-- The function p(x) = x^4 - 4x^2 + 4 -/
def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 4

/-- The theorem stating that the range of p(x) over [0, ∞) is [0, ∞) -/
theorem range_of_p :
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, x ≥ 0 ∧ p x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_l3644_364486


namespace NUMINAMATH_CALUDE_janet_time_saved_l3644_364431

/-- The number of minutes Janet spends looking for her keys daily -/
def looking_time : ℕ := 8

/-- The number of minutes Janet spends complaining after finding her keys daily -/
def complaining_time : ℕ := 3

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total time Janet saves in a week if she stops losing her keys -/
def time_saved : ℕ := (looking_time + complaining_time) * days_in_week

theorem janet_time_saved : time_saved = 77 := by
  sorry

end NUMINAMATH_CALUDE_janet_time_saved_l3644_364431


namespace NUMINAMATH_CALUDE_combined_capacity_is_forty_l3644_364436

/-- The combined capacity of two buses, each with 1/6 the capacity of a train that holds 120 people. -/
def combined_bus_capacity : ℕ :=
  let train_capacity : ℕ := 120
  let bus_capacity : ℕ := train_capacity / 6
  2 * bus_capacity

/-- Theorem stating that the combined capacity of the two buses is 40 people. -/
theorem combined_capacity_is_forty : combined_bus_capacity = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_capacity_is_forty_l3644_364436


namespace NUMINAMATH_CALUDE_decimal_value_changes_when_removing_zeros_l3644_364461

theorem decimal_value_changes_when_removing_zeros : 7.0800 ≠ 7.8 := by sorry

end NUMINAMATH_CALUDE_decimal_value_changes_when_removing_zeros_l3644_364461


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l3644_364480

/-- The number of sides in a convex polygon where the sum of all angles except two is 3420 degrees. -/
def polygon_sides : ℕ := 22

/-- The sum of interior angles of a polygon with n sides. -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The sum of all angles except two in the polygon. -/
def given_angle_sum : ℝ := 3420

theorem convex_polygon_sides :
  ∃ (missing_angles : ℝ), 
    missing_angles ≥ 0 ∧ 
    missing_angles < 360 ∧
    interior_angle_sum polygon_sides = given_angle_sum + missing_angles := by
  sorry

#check convex_polygon_sides

end NUMINAMATH_CALUDE_convex_polygon_sides_l3644_364480


namespace NUMINAMATH_CALUDE_fans_with_all_items_l3644_364481

def stadium_capacity : ℕ := 4500
def tshirt_interval : ℕ := 60
def hat_interval : ℕ := 45
def keychain_interval : ℕ := 75

theorem fans_with_all_items :
  (stadium_capacity / (Nat.lcm tshirt_interval (Nat.lcm hat_interval keychain_interval))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l3644_364481


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3644_364458

theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 6) (h2 : p 4 = 10) :
  ∃ (q r : ℝ → ℝ), (∀ x, p x = q x * ((x - 2) * (x - 4)) + r x) ∧
                    (∃ a b : ℝ, ∀ x, r x = a * x + b) ∧
                    (∀ x, r x = 2 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3644_364458


namespace NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l3644_364439

/-- Calculates the teacher's earnings for piano lessons over a given number of weeks -/
def teacher_earnings (rate_per_half_hour : ℕ) (lesson_duration_hours : ℕ) (weeks : ℕ) : ℕ :=
  rate_per_half_hour * 2 * lesson_duration_hours * weeks

/-- Proves that the teacher earns $100 in 5 weeks under the given conditions -/
theorem teacher_earnings_five_weeks :
  teacher_earnings 10 1 5 = 100 :=
by
  sorry

#eval teacher_earnings 10 1 5

end NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l3644_364439


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3644_364419

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem sufficient_not_necessary_condition
  (a₁ q : ℝ) :
  (a₁ < 0 ∧ 0 < q ∧ q < 1 →
    ∀ n : ℕ, n > 0 → geometric_sequence a₁ q (n + 1) > geometric_sequence a₁ q n) ∧
  (∃ a₁' q' : ℝ, (∀ n : ℕ, n > 0 → geometric_sequence a₁' q' (n + 1) > geometric_sequence a₁' q' n) ∧
    ¬(a₁' < 0 ∧ 0 < q' ∧ q' < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3644_364419


namespace NUMINAMATH_CALUDE_set_operations_l3644_364430

def U := Set ℝ
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem set_operations :
  (A ∩ B = {x : ℝ | 2 < x ∧ x < 3}) ∧
  (Set.compl B = {x : ℝ | x ≤ 1 ∨ x ≥ 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3644_364430


namespace NUMINAMATH_CALUDE_function_simplification_l3644_364453

/-- Given f(x) = (2x + 1)^5 - 5(2x + 1)^4 + 10(2x + 1)^3 - 10(2x + 1)^2 + 5(2x + 1) - 1,
    prove that f(x) = 32x^5 for all real x -/
theorem function_simplification (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1
  f x = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_l3644_364453


namespace NUMINAMATH_CALUDE_f_max_value_l3644_364409

/-- The quadratic function f(x) = -3x^2 + 6x + 4 --/
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 4

/-- The maximum value of f(x) over all real numbers x --/
def max_value : ℝ := 7

/-- Theorem stating that the maximum value of f(x) is 7 --/
theorem f_max_value : ∀ x : ℝ, f x ≤ max_value := by sorry

end NUMINAMATH_CALUDE_f_max_value_l3644_364409


namespace NUMINAMATH_CALUDE_sam_fish_count_l3644_364411

theorem sam_fish_count (harry joe sam : ℕ) 
  (harry_joe : harry = 4 * joe)
  (joe_sam : joe = 8 * sam)
  (harry_count : harry = 224) : 
  sam = 7 := by
sorry

end NUMINAMATH_CALUDE_sam_fish_count_l3644_364411


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3644_364421

theorem cone_lateral_surface_area 
  (r : ℝ) (h : ℝ) (l : ℝ) (A : ℝ) 
  (h_r : r = 2) 
  (h_h : h = Real.sqrt 5) 
  (h_l : l = Real.sqrt (r^2 + h^2)) 
  (h_A : A = π * r * l) : A = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3644_364421


namespace NUMINAMATH_CALUDE_part_to_third_ratio_l3644_364449

theorem part_to_third_ratio (N P : ℝ) (h1 : (1/4) * (1/3) * P = 14) (h2 : 0.40 * N = 168) :
  P / ((1/3) * N) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_third_ratio_l3644_364449


namespace NUMINAMATH_CALUDE_anns_shopping_cost_anns_shopping_proof_l3644_364425

theorem anns_shopping_cost (total_spent : ℕ) (shorts_quantity : ℕ) (shorts_price : ℕ) 
  (shoes_quantity : ℕ) (shoes_price : ℕ) (tops_quantity : ℕ) : ℕ :=
  let shorts_total := shorts_quantity * shorts_price
  let shoes_total := shoes_quantity * shoes_price
  let tops_total := total_spent - shorts_total - shoes_total
  tops_total / tops_quantity

theorem anns_shopping_proof :
  anns_shopping_cost 75 5 7 2 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_anns_shopping_cost_anns_shopping_proof_l3644_364425


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3644_364433

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), prove that the other endpoint is (-1, 5) -/
theorem other_endpoint_of_line_segment (x₂ y₂ : ℚ) : 
  (3 = (7 + x₂) / 2) ∧ (1 = (-3 + y₂) / 2) → (x₂ = -1 ∧ y₂ = 5) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3644_364433


namespace NUMINAMATH_CALUDE_paint_area_is_129_l3644_364465

/-- The area of the wall to be painted, given the dimensions of the wall, window, and door. -/
def areaToBePainted (wallHeight wallLength windowHeight windowLength doorHeight doorLength : ℕ) : ℕ :=
  wallHeight * wallLength - (windowHeight * windowLength + doorHeight * doorLength)

/-- Theorem stating that the area to be painted is 129 square feet. -/
theorem paint_area_is_129 :
  areaToBePainted 10 15 3 5 2 3 = 129 := by
  sorry

#eval areaToBePainted 10 15 3 5 2 3

end NUMINAMATH_CALUDE_paint_area_is_129_l3644_364465


namespace NUMINAMATH_CALUDE_spelling_bee_participants_l3644_364450

/-- In a competition, given a participant's ranking from best and worst, determine the total number of participants. -/
theorem spelling_bee_participants (n : ℕ) 
  (h_best : n = 75)  -- Priya is the 75th best
  (h_worst : n = 75) -- Priya is the 75th worst
  : (2 * n - 1 : ℕ) = 149 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_participants_l3644_364450


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l3644_364466

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i : i^2024 + i^2025 + i^2026 + i^2027 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l3644_364466


namespace NUMINAMATH_CALUDE_johns_height_l3644_364441

theorem johns_height (building_height : ℝ) (building_shadow : ℝ) (johns_shadow_inches : ℝ) :
  building_height = 60 →
  building_shadow = 20 →
  johns_shadow_inches = 18 →
  ∃ (johns_height : ℝ), johns_height = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_johns_height_l3644_364441


namespace NUMINAMATH_CALUDE_magician_earnings_l3644_364451

/-- Represents the sales and conditions of the magician's card deck business --/
structure MagicianSales where
  initialPrice : ℝ
  initialStock : ℕ
  finalStock : ℕ
  promotionPrice : ℝ
  initialExchangeRate : ℝ
  changedExchangeRate : ℝ
  foreignCustomersBulk : ℕ
  foreignCustomersSingle : ℕ
  domesticCustomers : ℕ

/-- Calculates the total earnings of the magician in dollars --/
def calculateEarnings (sales : MagicianSales) : ℝ :=
  sorry

/-- Theorem stating that the magician's earnings equal 11 dollars --/
theorem magician_earnings (sales : MagicianSales) 
  (h1 : sales.initialPrice = 2)
  (h2 : sales.initialStock = 5)
  (h3 : sales.finalStock = 3)
  (h4 : sales.promotionPrice = 3)
  (h5 : sales.initialExchangeRate = 1)
  (h6 : sales.changedExchangeRate = 1.5)
  (h7 : sales.foreignCustomersBulk = 2)
  (h8 : sales.foreignCustomersSingle = 1)
  (h9 : sales.domesticCustomers = 2) :
  calculateEarnings sales = 11 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l3644_364451


namespace NUMINAMATH_CALUDE_corn_syrup_amount_l3644_364473

/-- Represents the ratio of ingredients in a drink formulation -/
structure Ratio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- Represents a drink formulation -/
structure Formulation :=
  (ratio : Ratio)
  (water_amount : ℚ)

def standard_ratio : Ratio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

def sport_ratio (r : Ratio) : Ratio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

def sport_formulation : Formulation :=
  { ratio := sport_ratio standard_ratio,
    water_amount := 120 }

theorem corn_syrup_amount :
  (sport_formulation.ratio.corn_syrup / sport_formulation.ratio.water) *
    sport_formulation.water_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_corn_syrup_amount_l3644_364473


namespace NUMINAMATH_CALUDE_ohara_triple_49_64_l3644_364460

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ (Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x)

/-- Theorem: If (49, 64, x) is an O'Hara triple, then x = 15 -/
theorem ohara_triple_49_64 (x : ℕ) :
  is_ohara_triple 49 64 x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_64_l3644_364460


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l3644_364432

def A (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![a, 2, b; 3, 3, 4; c, 6, d]

def B (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![-4, e, -12; f, -14, g; 3, h, 5]

theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  (A a b c d) * (B e f g h) = 1 →
  a + b + c + d + e + f + g + h = 47 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l3644_364432


namespace NUMINAMATH_CALUDE_fourth_sample_number_l3644_364478

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun i => firstSample + i * (totalStudents / sampleSize)

theorem fourth_sample_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (h_total : totalStudents = 52)
  (h_sample : sampleSize = 4)
  (h_first : systematicSample totalStudents sampleSize 7 0 = 7)
  (h_second : systematicSample totalStudents sampleSize 7 1 = 33)
  (h_third : systematicSample totalStudents sampleSize 7 2 = 46) :
  systematicSample totalStudents sampleSize 7 3 = 20 :=
sorry

end NUMINAMATH_CALUDE_fourth_sample_number_l3644_364478


namespace NUMINAMATH_CALUDE_slope_range_l3644_364418

-- Define the line l passing through point P(-2, 2) with slope k
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y - 2 = k * (x + 2)}

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 + 12*x + 35 = 0}

-- Define the condition that circles with centers on l and radius 1 have no common points with C
def no_common_points (k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ line_l k →
    ∀ (a b : ℝ), (a - x)^2 + (b - y)^2 ≤ 1 →
      (a, b) ∉ circle_C

-- State the theorem
theorem slope_range :
  ∀ k : ℝ, no_common_points k →
    k < 0 ∨ k > 4/3 :=
sorry

end NUMINAMATH_CALUDE_slope_range_l3644_364418


namespace NUMINAMATH_CALUDE_spade_theorem_l3644_364405

-- Define the binary operation ◊
def spade (A B : ℚ) : ℚ := 4 * A + 3 * B - 2

-- Theorem statement
theorem spade_theorem (A : ℚ) : spade A 7 = 40 → A = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_theorem_l3644_364405


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_l3644_364497

theorem sum_of_powers_of_three : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_l3644_364497


namespace NUMINAMATH_CALUDE_line_relationships_l3644_364417

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationships
  (α β : Plane) (a b l : Line)
  (h1 : intersect α β = l)
  (h2 : subset a α)
  (h3 : subset b β)
  (h4 : ¬ perp α β)
  (h5 : ¬ perp_line a l)
  (h6 : ¬ perp_line b l) :
  (∃ (a' b' : Line), parallel a' b' ∧ a' = a ∧ b' = b) ∧
  (∃ (a'' b'' : Line), perp_line a'' b'' ∧ a'' = a ∧ b'' = b) :=
sorry

end NUMINAMATH_CALUDE_line_relationships_l3644_364417


namespace NUMINAMATH_CALUDE_convex_number_count_l3644_364492

/-- A function that checks if a three-digit number is convex -/
def isConvex (n : Nat) : Bool :=
  let a₁ := n / 100
  let a₂ := (n / 10) % 10
  let a₃ := n % 10
  100 ≤ n ∧ n < 1000 ∧ a₁ < a₂ ∧ a₃ < a₂

/-- The count of convex numbers -/
def convexCount : Nat :=
  (List.range 1000).filter isConvex |>.length

/-- Theorem stating that the count of convex numbers is 240 -/
theorem convex_number_count : convexCount = 240 := by
  sorry

end NUMINAMATH_CALUDE_convex_number_count_l3644_364492


namespace NUMINAMATH_CALUDE_simplify_expression_l3644_364413

theorem simplify_expression : (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + 63 = 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3644_364413


namespace NUMINAMATH_CALUDE_quadratic_equals_binomial_square_l3644_364464

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + d = (a*x + b)^2) → d = 900 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equals_binomial_square_l3644_364464


namespace NUMINAMATH_CALUDE_racing_track_width_l3644_364476

theorem racing_track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) : 
  r₁ - r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_racing_track_width_l3644_364476


namespace NUMINAMATH_CALUDE_xy_max_value_l3644_364435

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  x * y ≤ 1/8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x * y = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_xy_max_value_l3644_364435


namespace NUMINAMATH_CALUDE_dice_sum_probabilities_l3644_364456

/-- The probability of rolling a sum of 15 with 3 dice -/
def p_sum_15 : ℚ := 10 / 216

/-- The probability of rolling a sum of at least 15 with 3 dice -/
def p_sum_at_least_15 : ℚ := 20 / 216

/-- The minimum number of trials to roll a sum of 15 exactly once with probability > 1/2 -/
def min_trials_sum_15 : ℕ := 15

/-- The minimum number of trials to roll a sum of at least 15 exactly once with probability > 1/2 -/
def min_trials_sum_at_least_15 : ℕ := 8

theorem dice_sum_probabilities :
  (1 - (1 - p_sum_15) ^ min_trials_sum_15 > 1/2) ∧
  (∀ n : ℕ, n < min_trials_sum_15 → 1 - (1 - p_sum_15) ^ n ≤ 1/2) ∧
  (1 - (1 - p_sum_at_least_15) ^ min_trials_sum_at_least_15 > 1/2) ∧
  (∀ n : ℕ, n < min_trials_sum_at_least_15 → 1 - (1 - p_sum_at_least_15) ^ n ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_probabilities_l3644_364456


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3644_364467

/-- Represents the fare structure of a taxi service -/
structure TaxiFare where
  base_fare : ℝ
  per_mile_charge : ℝ

/-- Calculates the total fare for a given distance -/
def total_fare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.base_fare + tf.per_mile_charge * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.base_fare = 40)
  (h2 : total_fare tf 80 = 200) :
  total_fare tf 100 = 240 := by
sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3644_364467


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l3644_364457

theorem polynomial_root_implies_coefficients : 
  ∀ (p q : ℝ), 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + p * (2 - 3 * Complex.I : ℂ) ^ 2 - 5 * (2 - 3 * Complex.I : ℂ) + q = 0 →
  p = 1/2 ∧ q = 117/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l3644_364457


namespace NUMINAMATH_CALUDE_five_primes_in_valid_set_l3644_364448

/-- The set of digits to choose from -/
def digit_set : Finset Nat := {3, 5, 7, 8}

/-- Function to form a two-digit number from two digits -/
def form_number (tens units : Nat) : Nat := 10 * tens + units

/-- Predicate to check if a number is formed from two different digits in the set -/
def is_valid_number (n : Nat) : Prop :=
  ∃ (tens units : Nat), tens ∈ digit_set ∧ units ∈ digit_set ∧ tens ≠ units ∧ n = form_number tens units

/-- The set of all valid two-digit numbers formed from the digit set -/
def valid_numbers : Finset Nat := sorry

/-- The theorem stating that there are exactly 5 prime numbers in the valid set -/
theorem five_primes_in_valid_set : (valid_numbers.filter Nat.Prime).card = 5 := by sorry

end NUMINAMATH_CALUDE_five_primes_in_valid_set_l3644_364448


namespace NUMINAMATH_CALUDE_regular_octahedron_faces_regular_octahedron_has_eight_faces_l3644_364410

/-- A regular octahedron is a Platonic solid with equilateral triangular faces. -/
structure RegularOctahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces of a regular octahedron is 8. -/
theorem regular_octahedron_faces (o : RegularOctahedron) : Nat :=
  8

/-- Prove that a regular octahedron has 8 faces. -/
theorem regular_octahedron_has_eight_faces (o : RegularOctahedron) :
  regular_octahedron_faces o = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_octahedron_faces_regular_octahedron_has_eight_faces_l3644_364410


namespace NUMINAMATH_CALUDE_min_value_expression_l3644_364422

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 4) * (b^2 + 4*b + 4) * (c^2 + 4*c + 4) / (a * b * c) ≥ 729 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3644_364422


namespace NUMINAMATH_CALUDE_range_of_k_l3644_364471

def system_of_inequalities (x k : ℝ) : Prop :=
  x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+7)*x + 7*k < 0

def integer_solutions (k : ℝ) : Prop :=
  ∀ x : ℤ, system_of_inequalities (x : ℝ) k ↔ x = -3 ∨ x = -2

theorem range_of_k :
  ∀ k : ℝ, integer_solutions k → k ∈ Set.Ici (-3) ∩ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3644_364471


namespace NUMINAMATH_CALUDE_missing_angle_is_zero_l3644_364495

/-- Represents a polygon with a missing angle -/
structure PolygonWithMissingAngle where
  n : ℕ                     -- number of sides
  sum_without_missing : ℝ   -- sum of all angles except the missing one
  missing_angle : ℝ         -- the missing angle

/-- The theorem stating that the missing angle is 0° -/
theorem missing_angle_is_zero (p : PolygonWithMissingAngle) 
  (h1 : p.sum_without_missing = 3240)
  (h2 : p.sum_without_missing + p.missing_angle = 180 * (p.n - 2)) :
  p.missing_angle = 0 := by
sorry


end NUMINAMATH_CALUDE_missing_angle_is_zero_l3644_364495


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l3644_364484

theorem complex_equation_real_solution (a : ℝ) : 
  (((2 * a) / (1 + Complex.I) + 1 + Complex.I).im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l3644_364484


namespace NUMINAMATH_CALUDE_sequences_1992_values_l3644_364498

/-- Two sequences of integer numbers satisfying given conditions -/
def Sequences (a b : ℕ → ℤ) : Prop :=
  (a 0 = 0) ∧ (b 0 = 8) ∧
  (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n + 2) ∧
  (∀ n : ℕ, b (n + 2) = 2 * b (n + 1) - b n) ∧
  (∀ n : ℕ, ∃ k : ℤ, (a n)^2 + (b n)^2 = k^2)

/-- The theorem to be proved -/
theorem sequences_1992_values (a b : ℕ → ℤ) (h : Sequences a b) :
  ((a 1992 = 31872 ∧ b 1992 = 31880) ∨ (a 1992 = -31872 ∧ b 1992 = -31864)) :=
sorry

end NUMINAMATH_CALUDE_sequences_1992_values_l3644_364498


namespace NUMINAMATH_CALUDE_george_socks_problem_l3644_364420

theorem george_socks_problem (initial_socks : ℕ) (new_socks : ℕ) (final_socks : ℕ) 
  (h1 : initial_socks = 28)
  (h2 : new_socks = 36)
  (h3 : final_socks = 60) :
  initial_socks - (initial_socks + new_socks - final_socks) + new_socks = final_socks ∧ 
  initial_socks + new_socks - final_socks = 4 :=
by sorry

end NUMINAMATH_CALUDE_george_socks_problem_l3644_364420


namespace NUMINAMATH_CALUDE_carnival_tickets_l3644_364470

theorem carnival_tickets (num_games : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ) :
  num_games = 5 →
  found_tickets = 5 →
  ticket_value = 3 →
  total_value = 30 →
  ∃ (tickets_per_game : ℕ),
    (tickets_per_game * num_games + found_tickets) * ticket_value = total_value ∧
    tickets_per_game = 1 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l3644_364470


namespace NUMINAMATH_CALUDE_work_time_problem_l3644_364452

/-- The time taken to complete a work when multiple workers work together -/
def combined_work_time (work_rates : List ℚ) : ℚ :=
  1 / (work_rates.sum)

/-- The problem of finding the combined work time for A, B, and C -/
theorem work_time_problem :
  let a_rate : ℚ := 1 / 12
  let b_rate : ℚ := 1 / 24
  let c_rate : ℚ := 1 / 18
  combined_work_time [a_rate, b_rate, c_rate] = 72 / 13 := by
  sorry

#eval combined_work_time [1/12, 1/24, 1/18]

end NUMINAMATH_CALUDE_work_time_problem_l3644_364452


namespace NUMINAMATH_CALUDE_science_books_in_large_box_probability_l3644_364401

def total_textbooks : ℕ := 16
def science_textbooks : ℕ := 4
def box_capacities : List ℕ := [2, 4, 5, 5]

theorem science_books_in_large_box_probability :
  let total_ways := (total_textbooks.choose box_capacities[2]) * 
                    ((total_textbooks - box_capacities[2]).choose box_capacities[2]) * 
                    ((total_textbooks - box_capacities[2] - box_capacities[2]).choose box_capacities[1]) * 
                    1
  let favorable_ways := 2 * (box_capacities[2].choose science_textbooks) * 
                        (total_textbooks - science_textbooks).choose 1 * 
                        ((total_textbooks - box_capacities[2]).choose box_capacities[2]) * 
                        ((total_textbooks - box_capacities[2] - box_capacities[2]).choose box_capacities[1])
  (favorable_ways : ℚ) / total_ways = 5 / 182 := by
  sorry

end NUMINAMATH_CALUDE_science_books_in_large_box_probability_l3644_364401


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l3644_364493

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l3644_364493


namespace NUMINAMATH_CALUDE_student_committee_candidates_l3644_364445

theorem student_committee_candidates : 
  ∃ n : ℕ, 
    n > 0 ∧ 
    n * (n - 1) = 132 ∧ 
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_candidates_l3644_364445


namespace NUMINAMATH_CALUDE_simplify_expression_l3644_364423

theorem simplify_expression (a b c : ℝ) (h : b^2 = c^2) :
  -|b| - |a-b| + |a-c| - |b+c| = - |a-b| + |a-c| - |b+c| := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3644_364423


namespace NUMINAMATH_CALUDE_cricketer_score_percentage_l3644_364494

/-- Calculates the percentage of runs made by running between the wickets -/
def percentage_runs_by_running (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let runs_from_boundaries := boundaries * 4
  let runs_from_sixes := sixes * 6
  let runs_by_running := total_runs - (runs_from_boundaries + runs_from_sixes)
  (runs_by_running : ℚ) / total_runs * 100

/-- Proves that the percentage of runs made by running between the wickets is approximately 60.53% -/
theorem cricketer_score_percentage :
  let result := percentage_runs_by_running 152 12 2
  ∃ ε > 0, |result - 60.53| < ε :=
sorry

end NUMINAMATH_CALUDE_cricketer_score_percentage_l3644_364494


namespace NUMINAMATH_CALUDE_gender_related_to_reading_l3644_364488

-- Define the survey data
def survey_data : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![30, 20],
    ![40, 10]]

-- Define the total number of observations
def N : ℕ := 100

-- Define the formula for calculating k^2
def calculate_k_squared (data : Matrix (Fin 2) (Fin 2) ℕ) (total : ℕ) : ℚ :=
  let O11 := data 0 0
  let O12 := data 0 1
  let O21 := data 1 0
  let O22 := data 1 1
  (total * (O11 * O22 - O12 * O21)^2 : ℚ) / 
  ((O11 + O12) * (O21 + O22) * (O11 + O21) * (O12 + O22) : ℚ)

-- Define the critical values
def critical_value_005 : ℚ := 3841 / 1000
def critical_value_001 : ℚ := 6635 / 1000

-- State the theorem
theorem gender_related_to_reading :
  let k_squared := calculate_k_squared survey_data N
  k_squared > critical_value_005 ∧ k_squared < critical_value_001 := by
  sorry

end NUMINAMATH_CALUDE_gender_related_to_reading_l3644_364488


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3644_364491

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  initial_price = 150 →
  first_discount = 20 →
  final_price = 108 →
  ∃ (second_discount : ℝ),
    second_discount = 10 ∧
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3644_364491


namespace NUMINAMATH_CALUDE_power_of_three_expression_l3644_364403

theorem power_of_three_expression : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_l3644_364403


namespace NUMINAMATH_CALUDE_cuboid_base_area_l3644_364442

theorem cuboid_base_area (volume : ℝ) (height : ℝ) (base_area : ℝ) :
  volume = 144 →
  height = 8 →
  volume = base_area * height →
  base_area = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_cuboid_base_area_l3644_364442


namespace NUMINAMATH_CALUDE_parkway_elementary_boys_l3644_364429

theorem parkway_elementary_boys (total_students : ℕ) (soccer_players : ℕ) (girls_not_playing : ℕ)
  (h1 : total_students = 470)
  (h2 : soccer_players = 250)
  (h3 : girls_not_playing = 135)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑⌊(86 : ℚ) / 100 * soccer_players⌋) :
  total_students - (girls_not_playing + (soccer_players - ⌊(86 : ℚ) / 100 * soccer_players⌋)) = 300 :=
by sorry

end NUMINAMATH_CALUDE_parkway_elementary_boys_l3644_364429


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3644_364459

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n : ℕ) : ℕ := factorial (n - 1)

def adjacent_arrangements (n : ℕ) : ℕ := 2 * factorial (n - 2)

theorem seating_arrangements_with_restriction (total_people : ℕ) 
  (h1 : total_people = 8) :
  circular_permutations total_people - adjacent_arrangements total_people = 3600 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3644_364459


namespace NUMINAMATH_CALUDE_simplify_expression_l3644_364426

theorem simplify_expression (b : ℝ) : (2 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 720 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3644_364426


namespace NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l3644_364487

/-- Calculates the number of trips to the water fountain given the distance to the fountain and total distance walked. -/
def trips_to_fountain (distance_to_fountain : ℕ) (total_distance_walked : ℕ) : ℕ :=
  total_distance_walked / (2 * distance_to_fountain)

/-- Theorem stating that given a distance of 30 feet to the fountain and 120 feet walked, the number of trips is 2. -/
theorem mrs_hilt_fountain_trips :
  trips_to_fountain 30 120 = 2 := by
  sorry


end NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l3644_364487


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l3644_364462

theorem shoe_price_calculation (initial_price : ℝ) : 
  initial_price = 50 →
  let wednesday_price := initial_price * (1 + 0.15)
  let thursday_price := wednesday_price * (1 - 0.05)
  let monday_price := thursday_price * (1 - 0.20)
  monday_price = 43.70 := by sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l3644_364462


namespace NUMINAMATH_CALUDE_not_divisible_by_2006_l3644_364499

theorem not_divisible_by_2006 (k : ℤ) : ¬(2006 ∣ (k^2 + k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2006_l3644_364499


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3644_364440

/-- Given a triangle with exradii 2, 3, and 6 cm, the radius of the inscribed circle is 1 cm. -/
theorem inscribed_circle_radius (r₁ r₂ r₃ : ℝ) (hr₁ : r₁ = 2) (hr₂ : r₂ = 3) (hr₃ : r₃ = 6) :
  (1 / r₁ + 1 / r₂ + 1 / r₃)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3644_364440


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3644_364414

theorem inequality_holds_iff (x : ℝ) :
  (∀ y : ℝ, y^2 - (5^x - 1)*(y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3644_364414


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3644_364428

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3644_364428


namespace NUMINAMATH_CALUDE_tractor_trailer_unloading_l3644_364406

theorem tractor_trailer_unloading (initial_load : ℝ) : 
  initial_load = 50000 → 
  let remaining_after_first := initial_load - 0.1 * initial_load
  let final_load := remaining_after_first - 0.2 * remaining_after_first
  final_load = 36000 := by
sorry

end NUMINAMATH_CALUDE_tractor_trailer_unloading_l3644_364406


namespace NUMINAMATH_CALUDE_jerry_feathers_l3644_364404

theorem jerry_feathers (x : ℕ) : 
  let hawk_feathers : ℕ := 6
  let eagle_feathers : ℕ := x * hawk_feathers
  let total_feathers : ℕ := hawk_feathers + eagle_feathers
  let remaining_after_gift : ℕ := total_feathers - 10
  let sold_feathers : ℕ := remaining_after_gift / 2
  let final_feathers : ℕ := remaining_after_gift - sold_feathers
  (final_feathers = 49) → (x = 17) :=
by sorry

end NUMINAMATH_CALUDE_jerry_feathers_l3644_364404


namespace NUMINAMATH_CALUDE_factoring_expression_l3644_364490

theorem factoring_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) = (y + 2) * (5 * y + 9) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3644_364490


namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l3644_364483

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 15*x + 35

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! c : ℝ, g c = c ∧ c = -5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l3644_364483


namespace NUMINAMATH_CALUDE_longest_side_range_l3644_364479

-- Define an obtuse triangle
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_obtuse : ∃ angle, angle > π/2 ∧ angle < π

-- Theorem statement
theorem longest_side_range (triangle : ObtuseTriangle) 
  (ha : triangle.a = 1) 
  (hb : triangle.b = 2) : 
  (Real.sqrt 5 < triangle.c ∧ triangle.c < 3) ∨ triangle.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_range_l3644_364479


namespace NUMINAMATH_CALUDE_practicing_to_writing_ratio_l3644_364443

/-- Represents the time spent on different activities for a speech --/
structure SpeechTime where
  outlining : ℕ
  writing : ℕ
  practicing : ℕ
  total : ℕ

/-- Defines the conditions of Javier's speech preparation --/
def javierSpeechTime : SpeechTime where
  outlining := 30
  writing := 30 + 28
  practicing := 117 - (30 + 58)
  total := 117

/-- Theorem stating the ratio of practicing to writing time --/
theorem practicing_to_writing_ratio :
  (javierSpeechTime.practicing : ℚ) / javierSpeechTime.writing = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_practicing_to_writing_ratio_l3644_364443


namespace NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l3644_364434

theorem gcd_seven_eight_factorial : 
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l3644_364434


namespace NUMINAMATH_CALUDE_marble_ratio_is_two_to_one_l3644_364438

/-- The ratio of Mary's blue marbles to Dan's blue marbles -/
def marble_ratio (dans_marbles marys_marbles : ℕ) : ℚ :=
  marys_marbles / dans_marbles

/-- Proof that the ratio of Mary's blue marbles to Dan's blue marbles is 2:1 -/
theorem marble_ratio_is_two_to_one :
  marble_ratio 5 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_is_two_to_one_l3644_364438


namespace NUMINAMATH_CALUDE_f_monotone_increasing_range_l3644_364407

/-- The function f(x) defined on the interval [0,1] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a - 1) * x + 3

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*a*x - (2*a - 1)

/-- Theorem stating the range of a for which f(x) is monotonically increasing on [0,1] -/
theorem f_monotone_increasing_range :
  {a : ℝ | ∀ x ∈ Set.Icc 0 1, f_derivative a x ≥ 0} = Set.Iic (1/2) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_range_l3644_364407


namespace NUMINAMATH_CALUDE_nail_color_percentage_difference_l3644_364477

theorem nail_color_percentage_difference (total nails : ℕ) (purple blue : ℕ) :
  total = 20 →
  purple = 6 →
  blue = 8 →
  let striped := total - purple - blue
  let blue_percentage := (blue : ℚ) / (total : ℚ) * 100
  let striped_percentage := (striped : ℚ) / (total : ℚ) * 100
  blue_percentage - striped_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_nail_color_percentage_difference_l3644_364477


namespace NUMINAMATH_CALUDE_max_value_expression_l3644_364446

theorem max_value_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  (x^2 - 2*x*y + y^2) * (x^2 - 2*x*z + z^2) * (y^2 - 2*y*z + z^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3644_364446


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_real_l3644_364496

theorem sqrt_2x_minus_1_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 1) ↔ x ≥ 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_real_l3644_364496


namespace NUMINAMATH_CALUDE_inverse_proportional_cube_root_l3644_364427

theorem inverse_proportional_cube_root (x y : ℝ) (k : ℝ) : 
  (x ^ 2 * y ^ (1/3) = k) →  -- x² and ³√y are inversely proportional
  (3 ^ 2 * 216 ^ (1/3) = k) →  -- x = 3 when y = 216
  (x * y = 54) →  -- xy = 54
  y = 18 * 4 ^ (1/3) :=  -- y = 18 ³√4
by sorry

end NUMINAMATH_CALUDE_inverse_proportional_cube_root_l3644_364427


namespace NUMINAMATH_CALUDE_graph_shift_l3644_364482

/-- Given a function f and real numbers a and b, 
    the graph of y = f(x - a) + b is obtained by shifting 
    the graph of y = f(x) a units right and b units up. -/
theorem graph_shift (f : ℝ → ℝ) (a b : ℝ) :
  ∀ x y, y = f (x - a) + b ↔ y - b = f (x - a) :=
by sorry

end NUMINAMATH_CALUDE_graph_shift_l3644_364482


namespace NUMINAMATH_CALUDE_polynomial_expansions_l3644_364447

theorem polynomial_expansions (x y : ℝ) : 
  ((x - 3) * (x^2 + 4) = x^3 - 3*x^2 + 4*x - 12) ∧ 
  ((3*x^2 - y) * (x + 2*y) = 3*x^3 + 6*y*x^2 - x*y - 2*y^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansions_l3644_364447


namespace NUMINAMATH_CALUDE_roots_equation_sum_l3644_364412

theorem roots_equation_sum (a b : ℝ) : 
  a^2 - 6*a + 8 = 0 → b^2 - 6*b + 8 = 0 → a^4 + b^4 + a^3*b + a*b^3 = 432 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l3644_364412


namespace NUMINAMATH_CALUDE_donkey_mule_bags_l3644_364437

theorem donkey_mule_bags (x y : ℕ) (hx : x > 0) (hy : y > 0) : 
  (y + 1 = 2 * (x - 1) ∧ y - 1 = x + 1) ↔ 
  (∃ (d m : ℕ), d = x ∧ m = y ∧ 
    (m + 1 = 2 * (d - 1)) ∧ 
    (m - 1 = d + 1)) :=
by sorry

end NUMINAMATH_CALUDE_donkey_mule_bags_l3644_364437


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l3644_364472

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def is_arithmetic_seq (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_equivalence
  (a b c : ℕ → ℝ)
  (h1 : ∀ n : ℕ, b n = a n - a (n + 2))
  (h2 : ∀ n : ℕ, c n = a n + 2 * a (n + 1) + 3 * a (n + 2)) :
  is_arithmetic_seq a ↔ is_arithmetic_seq c ∧ (∀ n : ℕ, b n ≤ b (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l3644_364472


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3644_364424

/-- Given a parallelogram with area 288 square centimeters and height 16 cm, 
    prove that its base length is 18 cm. -/
theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 288) 
  (h2 : height = 16) : 
  area / height = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3644_364424
