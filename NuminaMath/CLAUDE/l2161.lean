import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_x4_is_10_l2161_216183

/-- The coefficient of x^4 in the expansion of (x^2 + 1/x)^5 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 5 2)

/-- Theorem: The coefficient of x^4 in the expansion of (x^2 + 1/x)^5 is 10 -/
theorem coefficient_x4_is_10 : coefficient_x4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_10_l2161_216183


namespace NUMINAMATH_CALUDE_simon_change_theorem_l2161_216137

/-- Calculates the discounted price for a flower purchase -/
def discountedPrice (quantity : ℕ) (price : ℚ) (discount : ℚ) : ℚ :=
  (quantity : ℚ) * price * (1 - discount)

/-- Calculates the total price after tax -/
def totalPriceAfterTax (prices : List ℚ) (taxRate : ℚ) : ℚ :=
  let subtotal := prices.sum
  subtotal * (1 + taxRate)

theorem simon_change_theorem (pansyPrice petuniasPrice lilyPrice orchidPrice : ℚ)
    (pansyDiscount hydrangeaDiscount petuniaDiscount lilyDiscount orchidDiscount : ℚ)
    (hydrangeaPrice : ℚ) (taxRate : ℚ) :
    pansyPrice = 2.5 →
    petuniasPrice = 1 →
    lilyPrice = 5 →
    orchidPrice = 7.5 →
    hydrangeaPrice = 12.5 →
    pansyDiscount = 0.1 →
    hydrangeaDiscount = 0.15 →
    petuniaDiscount = 0.2 →
    lilyDiscount = 0.12 →
    orchidDiscount = 0.08 →
    taxRate = 0.06 →
    let pansies := discountedPrice 5 pansyPrice pansyDiscount
    let hydrangea := discountedPrice 1 hydrangeaPrice hydrangeaDiscount
    let petunias := discountedPrice 5 petuniasPrice petuniaDiscount
    let lilies := discountedPrice 3 lilyPrice lilyDiscount
    let orchids := discountedPrice 2 orchidPrice orchidDiscount
    let total := totalPriceAfterTax [pansies, hydrangea, petunias, lilies, orchids] taxRate
    100 - total = 43.95 := by sorry

end NUMINAMATH_CALUDE_simon_change_theorem_l2161_216137


namespace NUMINAMATH_CALUDE_duplicate_page_number_l2161_216138

theorem duplicate_page_number (n : ℕ) (p : ℕ) : 
  (n ≥ 70) →
  (n ≤ 71) →
  (n * (n + 1)) / 2 + p = 2550 →
  p = 80 := by
sorry

end NUMINAMATH_CALUDE_duplicate_page_number_l2161_216138


namespace NUMINAMATH_CALUDE_car_distance_in_13_hours_l2161_216189

/-- Represents the driving characteristics of the car -/
structure Car where
  speed : ℕ            -- Speed in miles per hour
  drive_time : ℕ       -- Continuous driving time in hours
  cool_time : ℕ        -- Cooling time in hours

/-- Calculates the total distance a car can travel in a given time -/
def total_distance (c : Car) (total_time : ℕ) : ℕ :=
  sorry

/-- Theorem stating the total distance the car can travel in 13 hours -/
theorem car_distance_in_13_hours (c : Car) 
  (h1 : c.speed = 8)
  (h2 : c.drive_time = 5)
  (h3 : c.cool_time = 1) :
  total_distance c 13 = 88 :=
sorry

end NUMINAMATH_CALUDE_car_distance_in_13_hours_l2161_216189


namespace NUMINAMATH_CALUDE_circle_center_sum_l2161_216121

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 14*y - 11 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h + 14*k - 11)) →
  h + k = -4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2161_216121


namespace NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l2161_216139

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_factorials_perfect_square (n : ℕ) :
  isPerfectSquare (sumOfFactorials n) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l2161_216139


namespace NUMINAMATH_CALUDE_elephant_entry_rate_utopia_park_elephant_rate_l2161_216129

/-- Calculates the rate at which new elephants entered Utopia National Park --/
theorem elephant_entry_rate (initial_elephants : ℕ) (exodus_rate : ℕ) (exodus_duration : ℕ) 
  (entry_duration : ℕ) (final_elephants : ℕ) : ℕ :=
  let elephants_left := exodus_rate * exodus_duration
  let elephants_after_exodus := initial_elephants - elephants_left
  let new_elephants := final_elephants - elephants_after_exodus
  new_elephants / entry_duration

/-- Proves that the rate of new elephants entering the park is 1500 per hour --/
theorem utopia_park_elephant_rate : 
  elephant_entry_rate 30000 2880 4 7 28980 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_elephant_entry_rate_utopia_park_elephant_rate_l2161_216129


namespace NUMINAMATH_CALUDE_union_equals_set_implies_m_equals_one_l2161_216114

theorem union_equals_set_implies_m_equals_one :
  let A : Set ℝ := {-1, 2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  ∀ m : ℝ, (A ∪ B = A) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_union_equals_set_implies_m_equals_one_l2161_216114


namespace NUMINAMATH_CALUDE_least_fourth_integer_l2161_216128

theorem least_fourth_integer (a b c d : ℕ+) : 
  (a + b + c + d : ℚ) / 4 = 18 →
  a = 3 * b →
  b = c - 2 →
  (c : ℚ) = 1.5 * d →
  d ≥ 10 ∧ ∀ x : ℕ+, x < 10 → 
    ¬∃ a' b' c' : ℕ+, (a' + b' + c' + x : ℚ) / 4 = 18 ∧
                      a' = 3 * b' ∧
                      b' = c' - 2 ∧
                      (c' : ℚ) = 1.5 * x := by
  sorry

#check least_fourth_integer

end NUMINAMATH_CALUDE_least_fourth_integer_l2161_216128


namespace NUMINAMATH_CALUDE_min_sum_squares_l2161_216154

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 7, 9}

theorem min_sum_squares (p q r s t u v w x : Int) 
  (hp : p ∈ S) (hq : q ∈ S) (hr : r ∈ S) (hs : s ∈ S) 
  (ht : t ∈ S) (hu : u ∈ S) (hv : v ∈ S) (hw : w ∈ S) (hx : x ∈ S)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ p ≠ x ∧
               q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧ q ≠ x ∧
               r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ r ≠ x ∧
               s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ s ≠ x ∧
               t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ t ≠ x ∧
               u ≠ v ∧ u ≠ w ∧ u ≠ x ∧
               v ≠ w ∧ v ≠ x ∧
               w ≠ x) :
  (p + q + r + s)^2 + (t + u + v + w + x)^2 ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2161_216154


namespace NUMINAMATH_CALUDE_complex_fraction_ratio_l2161_216177

theorem complex_fraction_ratio (x : ℝ) : x = 200 → x / 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_ratio_l2161_216177


namespace NUMINAMATH_CALUDE_reaction_weight_equality_l2161_216119

/-- Atomic weight of Calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Bromine in g/mol -/
def Br_weight : ℝ := 79.904

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Molecular weight of CaBr2 in g/mol -/
def CaBr2_weight : ℝ := Ca_weight + 2 * Br_weight

/-- Molecular weight of H2O in g/mol -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- Molecular weight of Ca(OH)2 in g/mol -/
def CaOH2_weight : ℝ := Ca_weight + 2 * (O_weight + H_weight)

/-- Molecular weight of HBr in g/mol -/
def HBr_weight : ℝ := H_weight + Br_weight

/-- Theorem stating that the molecular weight of reactants equals the molecular weight of products
    and is equal to 235.918 g/mol -/
theorem reaction_weight_equality :
  CaBr2_weight + 2 * H2O_weight = CaOH2_weight + 2 * HBr_weight ∧
  CaBr2_weight + 2 * H2O_weight = 235.918 :=
by sorry

end NUMINAMATH_CALUDE_reaction_weight_equality_l2161_216119


namespace NUMINAMATH_CALUDE_paintball_cost_per_box_l2161_216110

/-- Calculates the cost per box of paintballs -/
def cost_per_box (plays_per_month : ℕ) (boxes_per_play : ℕ) (total_monthly_cost : ℕ) : ℚ :=
  total_monthly_cost / (plays_per_month * boxes_per_play)

/-- Theorem: Given the problem conditions, the cost per box of paintballs is $25 -/
theorem paintball_cost_per_box :
  let plays_per_month : ℕ := 3
  let boxes_per_play : ℕ := 3
  let total_monthly_cost : ℕ := 225
  cost_per_box plays_per_month boxes_per_play total_monthly_cost = 25 := by
  sorry


end NUMINAMATH_CALUDE_paintball_cost_per_box_l2161_216110


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2161_216141

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a → a 2 * a 6 = 4 → (a 4 = 2 ∨ a 4 = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2161_216141


namespace NUMINAMATH_CALUDE_track_length_is_300_l2161_216187

-- Define the track length
def track_length : ℝ := sorry

-- Define Brenda's distance to first meeting
def brenda_first_meeting : ℝ := 120

-- Define Sally's additional distance to second meeting
def sally_additional : ℝ := 180

-- Theorem statement
theorem track_length_is_300 :
  -- Conditions
  (brenda_first_meeting + (track_length - brenda_first_meeting) = track_length) ∧
  (brenda_first_meeting + brenda_first_meeting = 
   track_length - brenda_first_meeting + sally_additional) →
  -- Conclusion
  track_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_track_length_is_300_l2161_216187


namespace NUMINAMATH_CALUDE_percent_students_with_cats_l2161_216117

/-- Given a school with 500 students where 75 students own cats,
    prove that 15% of the students own cats. -/
theorem percent_students_with_cats :
  let total_students : ℕ := 500
  let cat_owners : ℕ := 75
  (cat_owners : ℚ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percent_students_with_cats_l2161_216117


namespace NUMINAMATH_CALUDE_no_nonneg_integer_solution_l2161_216158

theorem no_nonneg_integer_solution :
  ¬ ∃ y : ℕ, Real.sqrt ((y - 2)^2 + 4^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_nonneg_integer_solution_l2161_216158


namespace NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l2161_216172

def particle_position (t : ℕ) : ℕ × ℕ :=
  let n := (Nat.sqrt (t / 4) : ℕ)
  let remaining_time := t - 4 * n^2
  let side_length := 2 * n + 1
  if remaining_time ≤ side_length then
    (remaining_time, 0)
  else if remaining_time ≤ 2 * side_length then
    (side_length, remaining_time - side_length)
  else if remaining_time ≤ 3 * side_length then
    (3 * side_length - remaining_time, side_length)
  else
    (0, 4 * side_length - remaining_time)

theorem particle_position_after_2023_minutes :
  particle_position 2023 = (87, 0) := by
  sorry

end NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l2161_216172


namespace NUMINAMATH_CALUDE_max_triangle_area_in_three_squares_l2161_216198

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in the plane -/
structure Square where
  center : Point
  side : ℝ

/-- Definition of a unit square -/
def isUnitSquare (s : Square) : Prop := s.side = 1

/-- Definition of a point being contained in a square -/
def isContainedIn (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : Point) : ℝ :=
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

/-- The main theorem -/
theorem max_triangle_area_in_three_squares 
  (s₁ s₂ s₃ : Square) 
  (h₁ : isUnitSquare s₁) 
  (h₂ : isUnitSquare s₂) 
  (h₃ : isUnitSquare s₃) 
  (X : Point) 
  (hX₁ : isContainedIn X s₁) 
  (hX₂ : isContainedIn X s₂) 
  (hX₃ : isContainedIn X s₃) 
  (A B C : Point) 
  (hA : isContainedIn A s₁ ∨ isContainedIn A s₂ ∨ isContainedIn A s₃)
  (hB : isContainedIn B s₁ ∨ isContainedIn B s₂ ∨ isContainedIn B s₃)
  (hC : isContainedIn C s₁ ∨ isContainedIn C s₂ ∨ isContainedIn C s₃) :
  triangleArea A B C ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_in_three_squares_l2161_216198


namespace NUMINAMATH_CALUDE_min_coins_needed_l2161_216148

/-- The cost of the sneakers in cents -/
def cost : ℕ := 4550

/-- The amount Chloe already has in cents (four $10 bills and ten quarters) -/
def existing_funds : ℕ := 4250

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The minimum number of additional coins (dimes and nickels) needed -/
def min_coins : ℕ := 30

theorem min_coins_needed :
  ∀ (d n : ℕ),
    d * dime_value + n * nickel_value + existing_funds ≥ cost →
    d + n ≥ min_coins :=
sorry

end NUMINAMATH_CALUDE_min_coins_needed_l2161_216148


namespace NUMINAMATH_CALUDE_fraction_as_power_series_l2161_216140

theorem fraction_as_power_series :
  ∃ (a : ℕ → ℚ), (9 : ℚ) / 10 = (5 : ℚ) / 6 + ∑' n, a n / (6 ^ (n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_as_power_series_l2161_216140


namespace NUMINAMATH_CALUDE_more_subsets_gt_2009_l2161_216192

def M : Finset ℕ := {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}

def product (s : Finset ℕ) : ℕ := s.prod id

def subsets_gt_2009 : Finset (Finset ℕ) :=
  M.powerset.filter (λ s => s.card = 4 ∧ product s > 2009)

def subsets_lt_2009 : Finset (Finset ℕ) :=
  M.powerset.filter (λ s => s.card = 4 ∧ product s < 2009)

theorem more_subsets_gt_2009 : subsets_gt_2009.card > subsets_lt_2009.card := by
  sorry

end NUMINAMATH_CALUDE_more_subsets_gt_2009_l2161_216192


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l2161_216150

def day1_income : ℕ := 300
def day2_income : ℕ := 150
def day3_income : ℕ := 750
def day4_income : ℕ := 400
def day5_income : ℕ := 500
def num_days : ℕ := 5

theorem cab_driver_average_income :
  (day1_income + day2_income + day3_income + day4_income + day5_income) / num_days = 420 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l2161_216150


namespace NUMINAMATH_CALUDE_chewbacca_gum_packs_l2161_216104

theorem chewbacca_gum_packs (y : ℚ) : 
  (∃ (orange_packs apple_packs : ℕ), 
    orange_packs * y + (25 : ℚ) % y = 25 ∧ 
    apple_packs * y + (35 : ℚ) % y = 35 ∧
    (25 - 2 * y) / 35 = 25 / (35 + 4 * y)) → 
  y = 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_chewbacca_gum_packs_l2161_216104


namespace NUMINAMATH_CALUDE_reporter_earnings_l2161_216196

/-- Reporter's earnings calculation --/
theorem reporter_earnings
  (words_per_minute : ℕ)
  (pay_per_word : ℚ)
  (pay_per_article : ℕ)
  (num_articles : ℕ)
  (hours_available : ℕ)
  (h1 : words_per_minute = 10)
  (h2 : pay_per_word = 1/10)
  (h3 : pay_per_article = 60)
  (h4 : num_articles = 3)
  (h5 : hours_available = 4)
  : (((hours_available * 60 * words_per_minute) * pay_per_word + num_articles * pay_per_article) / hours_available : ℚ) = 105 :=
by sorry

end NUMINAMATH_CALUDE_reporter_earnings_l2161_216196


namespace NUMINAMATH_CALUDE_puzzle_border_pieces_l2161_216174

theorem puzzle_border_pieces (total_pieces : ℕ) (trevor_pieces : ℕ) (missing_pieces : ℕ) : 
  total_pieces = 500 → 
  trevor_pieces = 105 → 
  missing_pieces = 5 → 
  (total_pieces - missing_pieces - trevor_pieces - 3 * trevor_pieces) = 75 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_border_pieces_l2161_216174


namespace NUMINAMATH_CALUDE_quadratic_roots_proof_l2161_216118

theorem quadratic_roots_proof (x₁ x₂ : ℝ) : x₁ = -1 ∧ x₂ = 6 →
  (x₁^2 - 5*x₁ - 6 = 0) ∧ (x₂^2 - 5*x₂ - 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_proof_l2161_216118


namespace NUMINAMATH_CALUDE_base8_divisibility_by_13_l2161_216156

/-- Converts a base-8 number of the form 3dd7₈ to base 10 --/
def base8_to_base10 (d : ℕ) : ℕ := 3 * 512 + d * 64 + d * 8 + 7

/-- Checks if a natural number is divisible by 13 --/
def divisible_by_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

/-- A base-8 digit is between 0 and 7 inclusive --/
def is_base8_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 7

theorem base8_divisibility_by_13 (d : ℕ) (h : is_base8_digit d) : 
  divisible_by_13 (base8_to_base10 d) ↔ (d = 1 ∨ d = 2) :=
sorry

end NUMINAMATH_CALUDE_base8_divisibility_by_13_l2161_216156


namespace NUMINAMATH_CALUDE_prime_odd_sum_l2161_216197

theorem prime_odd_sum (x y : ℕ) : 
  Nat.Prime x → 
  Odd y → 
  x^2 + y = 2009 → 
  x + y = 2007 := by
sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l2161_216197


namespace NUMINAMATH_CALUDE_equation_solutions_l2161_216184

theorem equation_solutions :
  (∃ x : ℚ, 2*x + 1 = -2 - 3*x ∧ x = -3/5) ∧
  (∃ x : ℚ, x + (1-2*x)/3 = 2 - (x+2)/2 ∧ x = 4/5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2161_216184


namespace NUMINAMATH_CALUDE_counterexample_exists_l2161_216151

theorem counterexample_exists : ∃ n : ℕ, ¬ Nat.Prime n ∧ ¬ Nat.Prime (n - 4) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2161_216151


namespace NUMINAMATH_CALUDE_power_of_power_l2161_216190

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2161_216190


namespace NUMINAMATH_CALUDE_storks_count_l2161_216132

theorem storks_count (initial_birds : ℕ) (additional_birds : ℕ) (final_total : ℕ) : 
  initial_birds = 6 → additional_birds = 4 → final_total = 10 →
  final_total = initial_birds + additional_birds →
  0 = final_total - (initial_birds + additional_birds) :=
by sorry

end NUMINAMATH_CALUDE_storks_count_l2161_216132


namespace NUMINAMATH_CALUDE_walking_problem_l2161_216168

/-- The correct system of equations for the walking problem -/
theorem walking_problem (x y : ℝ) : 
  (∃ (total_distance : ℝ) (meeting_time : ℝ) (additional_time : ℝ),
    total_distance = 3 ∧ 
    meeting_time = 20/60 ∧ 
    additional_time = 10/60 ∧ 
    meeting_time * (x + y) = total_distance ∧
    (total_distance - (meeting_time + additional_time) * x) = 2 * (total_distance - (meeting_time + additional_time) * y)) ↔ 
  ((20/60 * x + 20/60 * y = 3) ∧ (3 - 30/60 * x = 2 * (3 - 30/60 * y))) :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l2161_216168


namespace NUMINAMATH_CALUDE_positive_number_equation_solution_l2161_216107

theorem positive_number_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt ((10 * x) / 3) = x ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equation_solution_l2161_216107


namespace NUMINAMATH_CALUDE_corrected_mean_l2161_216173

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) :
  n = 50 →
  incorrect_mean = 36 →
  incorrect_value = 21 →
  correct_value = 48 →
  (n : ℝ) * incorrect_mean - incorrect_value + correct_value = 36.54 * n :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l2161_216173


namespace NUMINAMATH_CALUDE_fourth_square_area_l2161_216163

-- Define the triangles and their properties
def triangle_PQR (PQ PR QR : ℝ) : Prop :=
  PQ^2 + PR^2 = QR^2 ∧ PQ = 5 ∧ PR = 7

def triangle_PRS (PR PS RS : ℝ) : Prop :=
  PR^2 + PS^2 = RS^2 ∧ PS = 8 ∧ PR = 7

-- Theorem statement
theorem fourth_square_area 
  (PQ PR QR PS RS : ℝ) 
  (h1 : triangle_PQR PQ PR QR) 
  (h2 : triangle_PRS PR PS RS) : 
  RS^2 = 113 := by
sorry

end NUMINAMATH_CALUDE_fourth_square_area_l2161_216163


namespace NUMINAMATH_CALUDE_number_difference_l2161_216182

theorem number_difference (a b : ℕ) : 
  a + b = 26832 → 
  b % 10 = 0 → 
  a = b / 10 + 4 → 
  b - a = 21938 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2161_216182


namespace NUMINAMATH_CALUDE_circle_coverage_theorem_l2161_216153

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of circles used to cover the main circle -/
structure CoverConfiguration where
  main_circle : Circle
  covering_circles : List Circle

/-- Checks if a point is covered by a circle -/
def is_point_covered (point : ℝ × ℝ) (circle : Circle) : Prop :=
  let (x, y) := point
  let (cx, cy) := circle.center
  (x - cx)^2 + (y - cy)^2 ≤ circle.radius^2

/-- Checks if all points in the main circle are covered by at least one of the covering circles -/
def is_circle_covered (config : CoverConfiguration) : Prop :=
  ∀ point, is_point_covered point config.main_circle →
    ∃ cover_circle ∈ config.covering_circles, is_point_covered point cover_circle

/-- The main theorem stating that a circle with diameter 81.9 can be covered by 5 circles of diameter 50 -/
theorem circle_coverage_theorem :
  ∃ config : CoverConfiguration,
    config.main_circle.radius = 81.9 / 2 ∧
    config.covering_circles.length = 5 ∧
    (∀ circle ∈ config.covering_circles, circle.radius = 50 / 2) ∧
    is_circle_covered config :=
  sorry


end NUMINAMATH_CALUDE_circle_coverage_theorem_l2161_216153


namespace NUMINAMATH_CALUDE_trevor_eggs_left_l2161_216164

/-- Represents the number of eggs laid by each chicken and the number of eggs dropped --/
structure EggCollection where
  gertrude : ℕ
  blanche : ℕ
  nancy : ℕ
  martha : ℕ
  dropped : ℕ

/-- Calculates the number of eggs Trevor has left --/
def eggsLeft (collection : EggCollection) : ℕ :=
  collection.gertrude + collection.blanche + collection.nancy + collection.martha - collection.dropped

/-- Theorem stating that Trevor has 9 eggs left --/
theorem trevor_eggs_left :
  ∃ (collection : EggCollection),
    collection.gertrude = 4 ∧
    collection.blanche = 3 ∧
    collection.nancy = 2 ∧
    collection.martha = 2 ∧
    collection.dropped = 2 ∧
    eggsLeft collection = 9 := by
  sorry

end NUMINAMATH_CALUDE_trevor_eggs_left_l2161_216164


namespace NUMINAMATH_CALUDE_complex_not_in_first_quadrant_l2161_216195

theorem complex_not_in_first_quadrant (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 + Complex.I)
  ¬ (z.re > 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_not_in_first_quadrant_l2161_216195


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2161_216127

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating the form of functions satisfying the functional equation -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2161_216127


namespace NUMINAMATH_CALUDE_inequality_solution_l2161_216105

def solution_set (a : ℝ) : Set ℝ :=
  if a > 1 then {x | x ≤ 1 ∨ x ≥ a}
  else if a = 1 then Set.univ
  else {x | x ≤ a ∨ x ≥ 1}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + 1)*x + a ≥ 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2161_216105


namespace NUMINAMATH_CALUDE_vacation_expense_sharing_l2161_216167

/-- The vacation expense sharing problem -/
theorem vacation_expense_sharing 
  (alex kim lee nina : ℝ)
  (h_alex : alex = 130)
  (h_kim : kim = 150)
  (h_lee : lee = 170)
  (h_nina : nina = 200)
  (h_total : alex + kim + lee + nina = 650)
  (h_equal_share : (alex + kim + lee + nina) / 4 = 162.5)
  (a k : ℝ)
  (h_a : a = 162.5 - alex)
  (h_k : k = 162.5 - kim) :
  a - k = 20 := by
sorry

end NUMINAMATH_CALUDE_vacation_expense_sharing_l2161_216167


namespace NUMINAMATH_CALUDE_contrapositive_sum_irrational_l2161_216145

/-- The contrapositive of "If a + b is irrational, then at least one of a or b is irrational" -/
theorem contrapositive_sum_irrational (a b : ℝ) :
  (¬(∃ q : ℚ, (a : ℝ) = q) ∨ ¬(∃ q : ℚ, (b : ℝ) = q) → ¬(∃ q : ℚ, (a + b : ℝ) = q)) ↔
  ((∃ q : ℚ, (a : ℝ) = q) ∧ (∃ q : ℚ, (b : ℝ) = q) → (∃ q : ℚ, (a + b : ℝ) = q)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_sum_irrational_l2161_216145


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_smallest_positive_integer_l2161_216175

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 648 ∣ (450 * x) → x ≥ 36 := by
  sorry

theorem thirty_six_satisfies : 648 ∣ (450 * 36) := by
  sorry

theorem smallest_positive_integer : 
  ∃ (x : ℕ), x > 0 ∧ 648 ∣ (450 * x) ∧ ∀ (y : ℕ), y > 0 ∧ 648 ∣ (450 * y) → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_smallest_positive_integer_l2161_216175


namespace NUMINAMATH_CALUDE_red_bellied_minnows_count_l2161_216194

/-- Represents the number of minnows in a pond with different belly colors. -/
structure MinnowPond where
  total : ℕ
  red_percent : ℚ
  green_percent : ℚ
  white_count : ℕ

/-- Theorem stating the number of red-bellied minnows in the pond. -/
theorem red_bellied_minnows_count (pond : MinnowPond)
  (h1 : pond.red_percent = 2/5)
  (h2 : pond.green_percent = 3/10)
  (h3 : pond.white_count = 15)
  (h4 : pond.total * (1 - pond.red_percent - pond.green_percent) = pond.white_count) :
  pond.total * pond.red_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_red_bellied_minnows_count_l2161_216194


namespace NUMINAMATH_CALUDE_shelter_animals_count_l2161_216161

/-- Calculates the total number of animals in the shelter after adoption and new arrivals --/
def total_animals_after_events (initial_cats initial_dogs initial_rabbits : ℕ) : ℕ :=
  let adopted_cats := initial_cats / 4
  let adopted_dogs := initial_dogs / 3
  let new_cats := 3 * adopted_cats
  let new_dogs := 2 * adopted_dogs
  let final_cats := initial_cats - adopted_cats + new_cats
  let final_dogs := initial_dogs - adopted_dogs + new_dogs
  let final_rabbits := 2 * initial_rabbits
  final_cats + final_dogs + final_rabbits

/-- Theorem stating that given the initial conditions, the total number of animals after events is 210 --/
theorem shelter_animals_count : total_animals_after_events 60 45 30 = 210 := by
  sorry

end NUMINAMATH_CALUDE_shelter_animals_count_l2161_216161


namespace NUMINAMATH_CALUDE_range_of_a_l2161_216181

/-- The function g(x) = ax + 2 where a > 0 -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, g a x₁ = f x₀) →
  a ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2161_216181


namespace NUMINAMATH_CALUDE_xy_system_solution_l2161_216131

theorem xy_system_solution (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 110) :
  x^2 + y^2 = 8044 / 169 := by
sorry

end NUMINAMATH_CALUDE_xy_system_solution_l2161_216131


namespace NUMINAMATH_CALUDE_angle_B_value_max_perimeter_max_perimeter_achievable_l2161_216179

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given condition (2a-c)cos B = b cos C -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

/-- Theorem stating that B = π/3 -/
theorem angle_B_value (t : Triangle) (h : triangle_condition t) : t.B = π / 3 :=
sorry

/-- Theorem stating that when b = 2, the maximum perimeter is 6 -/
theorem max_perimeter (t : Triangle) (h : triangle_condition t) (hb : t.b = 2) :
  t.a + t.b + t.c ≤ 6 :=
sorry

/-- Theorem stating that the maximum perimeter of 6 is achievable -/
theorem max_perimeter_achievable : ∃ t : Triangle, triangle_condition t ∧ t.b = 2 ∧ t.a + t.b + t.c = 6 :=
sorry

end NUMINAMATH_CALUDE_angle_B_value_max_perimeter_max_perimeter_achievable_l2161_216179


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2161_216111

theorem greatest_of_three_consecutive_integers (x : ℤ) :
  (x + (x + 1) + (x + 2) = 18) → (max x (max (x + 1) (x + 2)) = 7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2161_216111


namespace NUMINAMATH_CALUDE_population_trend_l2161_216185

theorem population_trend (P₀ k : ℝ) (h₁ : P₀ > 0) (h₂ : -1 < k) (h₃ : k < 0) :
  ∀ n : ℕ, P₀ * (1 + k) ^ (n + 1) < P₀ * (1 + k) ^ n :=
by sorry

end NUMINAMATH_CALUDE_population_trend_l2161_216185


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_no_false_main_theorem_l2161_216159

-- Proposition 1
theorem proposition_1 (k : ℝ) : k > 0 → ∃ x : ℝ, x^2 - 2*x - k = 0 := by sorry

-- Proposition 2
theorem proposition_2 (x y : ℝ) : x + y ≠ 8 → x ≠ 2 ∨ y ≠ 6 := by sorry

-- Proposition 3
theorem proposition_3_no_false : ¬∃ (P : Prop), P ↔ ¬(∀ x y : ℝ, x*y = 0 → x = 0 ∨ y = 0) := by sorry

-- Main theorem combining all propositions
theorem main_theorem : 
  (∀ k : ℝ, k > 0 → ∃ x : ℝ, x^2 - 2*x - k = 0) ∧ 
  (∀ x y : ℝ, x + y ≠ 8 → x ≠ 2 ∨ y ≠ 6) ∧ 
  ¬∃ (P : Prop), P ↔ ¬(∀ x y : ℝ, x*y = 0 → x = 0 ∨ y = 0) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_no_false_main_theorem_l2161_216159


namespace NUMINAMATH_CALUDE_power_function_property_l2161_216115

/-- A power function is a function of the form f(x) = x^n for some real number n. -/
def PowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

/-- A function lies in the first and third quadrants if it's positive for positive x
    and negative for negative x. -/
def LiesInFirstAndThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

theorem power_function_property
  (f : ℝ → ℝ)
  (h_power : PowerFunction f)
  (h_quadrants : LiesInFirstAndThirdQuadrants f)
  (h_inequality : f 3 < f 2) :
  f (-3) > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l2161_216115


namespace NUMINAMATH_CALUDE_divisor_problem_l2161_216125

theorem divisor_problem (n d : ℕ) (h1 : n % d = 3) (h2 : (n^2) % d = 3) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2161_216125


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l2161_216113

/-- The molecular weight of a compound given the total weight and number of moles -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 525)
  (h2 : num_moles = 3)
  (h3 : total_weight > 0)
  (h4 : num_moles > 0) :
  total_weight / num_moles = 175 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l2161_216113


namespace NUMINAMATH_CALUDE_pet_store_cats_l2161_216142

theorem pet_store_cats (initial_siamese : ℕ) (sold : ℕ) (remaining : ℕ) (initial_house : ℕ) : 
  initial_siamese = 13 → 
  sold = 10 → 
  remaining = 8 → 
  initial_siamese + initial_house - sold = remaining → 
  initial_house = 5 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_l2161_216142


namespace NUMINAMATH_CALUDE_EF_is_one_eighth_of_GH_l2161_216166

-- Define the line segment GH and points E and F on it
variable (G H E F : Real)

-- Define the condition that E and F lie on GH
axiom E_on_GH : G ≤ E ∧ E ≤ H
axiom F_on_GH : G ≤ F ∧ F ≤ H

-- Define the length ratios
axiom GE_ratio : E - G = 3 * (H - E)
axiom GF_ratio : F - G = 7 * (H - F)

-- State the theorem to be proved
theorem EF_is_one_eighth_of_GH : (F - E) = (1/8) * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_EF_is_one_eighth_of_GH_l2161_216166


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2161_216108

theorem sum_of_three_numbers : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2161_216108


namespace NUMINAMATH_CALUDE_number_of_ways_to_turn_off_lights_l2161_216199

/-- The number of streetlights --/
def total_lights : ℕ := 12

/-- The number of lights that can be turned off --/
def lights_off : ℕ := 3

/-- The number of positions where lights can be turned off --/
def eligible_positions : ℕ := total_lights - 2 - lights_off + 1

/-- Theorem stating the number of ways to turn off lights --/
theorem number_of_ways_to_turn_off_lights : 
  Nat.choose eligible_positions lights_off = 56 := by sorry

end NUMINAMATH_CALUDE_number_of_ways_to_turn_off_lights_l2161_216199


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2161_216109

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2161_216109


namespace NUMINAMATH_CALUDE_x_equals_one_l2161_216171

theorem x_equals_one :
  ∀ x : ℝ,
  ((x^31) / (5^31)) * ((x^16) / (4^16)) = 1 / (2 * (10^31)) →
  x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_l2161_216171


namespace NUMINAMATH_CALUDE_cubic_root_cube_relation_l2161_216147

/-- Given a cubic polynomial f(x) = x^3 + 2x^2 + 3x + 4, there exists another cubic polynomial 
g(x) = x^3 + bx^2 + cx + d such that the roots of g(x) are the cubes of the roots of f(x). -/
theorem cubic_root_cube_relation : 
  ∃ (b c d : ℝ), ∀ (r : ℂ), (r^3 + 2*r^2 + 3*r + 4 = 0) → 
    ((r^3)^3 + b*(r^3)^2 + c*(r^3) + d = 0) := by
  sorry


end NUMINAMATH_CALUDE_cubic_root_cube_relation_l2161_216147


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2161_216176

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Checks if a distribution is valid for the given number of balls and boxes -/
def is_valid_distribution (d : Distribution) (num_balls num_boxes : Nat) : Prop :=
  d.length ≤ num_boxes ∧ d.sum = num_balls ∧ d.all (· ≥ 0)

/-- Counts the number of distinct ways to distribute indistinguishable balls into indistinguishable boxes -/
def count_distributions (num_balls num_boxes : Nat) : Nat :=
  sorry

theorem six_balls_four_boxes :
  count_distributions 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2161_216176


namespace NUMINAMATH_CALUDE_painters_time_equation_l2161_216134

/-- The time it takes for two painters to paint a room together, given their individual rates and a lunch break -/
theorem painters_time_equation (doug_rate : ℚ) (dave_rate : ℚ) (t : ℚ) 
  (h_doug : doug_rate = 1 / 5)
  (h_dave : dave_rate = 1 / 7)
  (h_positive : t > 0) :
  (doug_rate + dave_rate) * (t - 1) = 1 ↔ t = 47 / 12 :=
by sorry

end NUMINAMATH_CALUDE_painters_time_equation_l2161_216134


namespace NUMINAMATH_CALUDE_second_month_sale_l2161_216186

/-- Represents the sales data for a grocer over six months -/
structure GrocerSales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Theorem: Given the sales for five months and the average sale,
    prove that the sale in the second month was 7000 -/
theorem second_month_sale
  (sales : GrocerSales)
  (h1 : sales.month1 = 6400)
  (h3 : sales.month3 = 6800)
  (h4 : sales.month4 = 7200)
  (h5 : sales.month5 = 6500)
  (h6 : sales.month6 = 5100)
  (avg : (sales.month1 + sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6) / 6 = 6500) :
  sales.month2 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l2161_216186


namespace NUMINAMATH_CALUDE_wrok_represents_5167_l2161_216130

/-- Represents a mapping from characters to digits -/
def CodeMapping : Type := Char → Nat

/-- The code "GREAT WORK" represents digits 0-8 respectively -/
def great_work_code (mapping : CodeMapping) : Prop :=
  mapping 'G' = 0 ∧
  mapping 'R' = 1 ∧
  mapping 'E' = 2 ∧
  mapping 'A' = 3 ∧
  mapping 'T' = 4 ∧
  mapping 'W' = 5 ∧
  mapping 'O' = 6 ∧
  mapping 'R' = 1 ∧
  mapping 'K' = 7

/-- The code word "WROK" represents a 4-digit number -/
def wrok_code (mapping : CodeMapping) : Nat :=
  mapping 'W' * 1000 + mapping 'R' * 100 + mapping 'O' * 10 + mapping 'K'

theorem wrok_represents_5167 (mapping : CodeMapping) :
  great_work_code mapping → wrok_code mapping = 5167 := by
  sorry

end NUMINAMATH_CALUDE_wrok_represents_5167_l2161_216130


namespace NUMINAMATH_CALUDE_animals_per_aquarium_is_46_l2161_216144

/-- The number of saltwater aquariums Tyler has -/
def num_saltwater_aquariums : ℕ := 22

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := 1012

/-- The number of animals in each saltwater aquarium -/
def animals_per_aquarium : ℕ := total_saltwater_animals / num_saltwater_aquariums

theorem animals_per_aquarium_is_46 : animals_per_aquarium = 46 := by
  sorry

end NUMINAMATH_CALUDE_animals_per_aquarium_is_46_l2161_216144


namespace NUMINAMATH_CALUDE_poster_count_l2161_216123

/-- The total number of posters made by Mario, Samantha, and Jonathan -/
def total_posters (mario_posters samantha_posters jonathan_posters : ℕ) : ℕ :=
  mario_posters + samantha_posters + jonathan_posters

/-- Theorem stating the total number of posters made by Mario, Samantha, and Jonathan -/
theorem poster_count : ∃ (mario_posters samantha_posters jonathan_posters : ℕ),
  mario_posters = 36 ∧
  samantha_posters = mario_posters + 45 ∧
  jonathan_posters = 2 * samantha_posters ∧
  total_posters mario_posters samantha_posters jonathan_posters = 279 :=
by
  sorry

end NUMINAMATH_CALUDE_poster_count_l2161_216123


namespace NUMINAMATH_CALUDE_equation_satisfied_l2161_216133

theorem equation_satisfied (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  3 * x - 4 * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l2161_216133


namespace NUMINAMATH_CALUDE_circle_bisection_theorem_l2161_216162

/-- A circle in the 2D plane. -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the 2D plane. -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Predicate to check if a line bisects a circle. -/
def bisects (l : Line) (c : Circle) : Prop :=
  ∃ (x y : ℝ), c.equation x y ∧ l.equation x y

/-- The main theorem stating that if a specific circle is bisected by a specific line, then a = 1. -/
theorem circle_bisection_theorem (a : ℝ) : 
  let c : Circle := { equation := fun x y => x^2 + y^2 + 2*x - 4*y = 0 }
  let l : Line := { equation := fun x y => 3*x + y + a = 0 }
  bisects l c → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_bisection_theorem_l2161_216162


namespace NUMINAMATH_CALUDE_inequality_range_l2161_216146

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ k ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2161_216146


namespace NUMINAMATH_CALUDE_max_area_at_midline_l2161_216149

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a line parallel to AC
def ParallelLine (t : Triangle) (M N : ℝ × ℝ) : Prop :=
  -- Add appropriate condition for parallel lines
  sorry

-- Define the rectangle MNPQ
structure Rectangle (t : Triangle) :=
  (M N P Q : ℝ × ℝ)
  (parallel : ParallelLine t M N)

-- Define the area of a rectangle
def area (r : Rectangle t) : ℝ :=
  sorry

-- Define the midline of a triangle
def Midline (t : Triangle) (M N : ℝ × ℝ) : Prop :=
  -- Add appropriate condition for midline
  sorry

-- Theorem statement
theorem max_area_at_midline (t : Triangle) :
  ∀ (r : Rectangle t), 
    Midline t r.M r.N → 
    ∀ (r' : Rectangle t), area r ≥ area r' :=
sorry

end NUMINAMATH_CALUDE_max_area_at_midline_l2161_216149


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2161_216157

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (m b : ℝ),
    (∀ (x y : ℝ), perpendicular_line x y ↔ y = m * x + b) ∧
    (perpendicular_line point.1 point.2) ∧
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2161_216157


namespace NUMINAMATH_CALUDE_multiply_add_distribute_l2161_216143

theorem multiply_add_distribute : 3.5 * 2.5 + 6.5 * 2.5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_distribute_l2161_216143


namespace NUMINAMATH_CALUDE_f_inequality_solution_l2161_216120

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the domain of f
def domain (x : ℝ) : Prop := -2 < x ∧ x < 2

-- Define the solution set
def solution_set (a : ℝ) : Prop := (-2 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)

-- State the theorem
theorem f_inequality_solution :
  ∀ a : ℝ, domain a → domain (a^2 - 2) →
  (f a + f (a^2 - 2) < 0 ↔ solution_set a) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_l2161_216120


namespace NUMINAMATH_CALUDE_land_plot_side_length_l2161_216169

theorem land_plot_side_length (area : ℝ) (h : area = Real.sqrt 1600) :
  Real.sqrt area = 40 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_side_length_l2161_216169


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l2161_216103

theorem unique_two_digit_multiple : ∃! t : ℕ, 
  10 ≤ t ∧ t < 100 ∧ (13 * t) % 100 = 42 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l2161_216103


namespace NUMINAMATH_CALUDE_sticker_count_l2161_216136

theorem sticker_count (stickers_per_page : ℕ) (total_pages : ℕ) : 
  stickers_per_page = 10 → total_pages = 22 → stickers_per_page * total_pages = 220 :=
by sorry

end NUMINAMATH_CALUDE_sticker_count_l2161_216136


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l2161_216188

/-- Calculates the time for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (tree_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1100) 
  (h2 : tree_passing_time = 110) 
  (h3 : platform_length = 700) : 
  (train_length + platform_length) / (train_length / tree_passing_time) = 180 :=
sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l2161_216188


namespace NUMINAMATH_CALUDE_yellow_flags_in_200_l2161_216126

/-- Represents the number of flags in one complete pattern -/
def pattern_length : ℕ := 9

/-- Represents the number of yellow flags in one complete pattern -/
def yellow_per_pattern : ℕ := 3

/-- Represents the total number of flags we're considering -/
def total_flags : ℕ := 200

/-- Calculates the number of yellow flags in the given sequence -/
def yellow_flags (n : ℕ) : ℕ :=
  (n / pattern_length) * yellow_per_pattern + min yellow_per_pattern (n % pattern_length)

theorem yellow_flags_in_200 : yellow_flags total_flags = 67 := by
  sorry

end NUMINAMATH_CALUDE_yellow_flags_in_200_l2161_216126


namespace NUMINAMATH_CALUDE_max_integer_k_no_real_roots_l2161_216106

theorem max_integer_k_no_real_roots (k : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k ≤ -2 ∧ ∀ m : ℤ, (∀ x : ℝ, x^2 - 2*x - m ≠ 0) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_no_real_roots_l2161_216106


namespace NUMINAMATH_CALUDE_irrational_sqrt_7_and_others_rational_l2161_216124

theorem irrational_sqrt_7_and_others_rational : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 7 = (a : ℝ) / (b : ℝ)) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (4 : ℝ) / 3 = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 4 = (a : ℝ) / (b : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_7_and_others_rational_l2161_216124


namespace NUMINAMATH_CALUDE_certain_term_is_12th_l2161_216160

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  -- Sum of a certain term and the 12th term is 20
  certain_term_sum : ∃ n : ℕ, a + (n - 1) * d + (a + 11 * d) = 20
  -- Sum of first 12 terms is 120
  sum_12_terms : 6 * (2 * a + 11 * d) = 120

/-- The certain term is the 12th term itself -/
theorem certain_term_is_12th (ap : ArithmeticProgression) : 
  ∃ n : ℕ, n = 12 ∧ a + (n - 1) * d + (a + 11 * d) = 20 := by
  sorry

#check certain_term_is_12th

end NUMINAMATH_CALUDE_certain_term_is_12th_l2161_216160


namespace NUMINAMATH_CALUDE_percent_of_number_l2161_216135

/-- 0.1 percent of 12,356 is equal to 12.356 -/
theorem percent_of_number : (0.1 / 100) * 12356 = 12.356 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_l2161_216135


namespace NUMINAMATH_CALUDE_tyrah_pencils_l2161_216102

theorem tyrah_pencils (sarah tim tyrah : ℕ) : 
  tyrah = 6 * sarah →
  tim = 8 * sarah →
  tim = 16 →
  tyrah = 12 :=
by sorry

end NUMINAMATH_CALUDE_tyrah_pencils_l2161_216102


namespace NUMINAMATH_CALUDE_find_m_l2161_216116

theorem find_m : ∃ m : ℚ, 
  (∀ x : ℚ, 4 * x + 2 * m = 5 * x + 1 ↔ 3 * x = 6 * x - 1) → 
  m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2161_216116


namespace NUMINAMATH_CALUDE_kath_siblings_count_l2161_216101

/-- The number of siblings Kath took to the movie -/
def num_siblings : ℕ := 2

/-- The number of friends Kath took to the movie -/
def num_friends : ℕ := 3

/-- The regular admission cost -/
def regular_cost : ℕ := 8

/-- The discount for movies before 6 P.M. -/
def discount : ℕ := 3

/-- The total amount Kath paid for all admissions -/
def total_paid : ℕ := 30

/-- The actual admission cost per person (after discount) -/
def actual_cost : ℕ := regular_cost - discount

theorem kath_siblings_count :
  num_siblings = (total_paid - (num_friends + 1) * actual_cost) / actual_cost :=
sorry

end NUMINAMATH_CALUDE_kath_siblings_count_l2161_216101


namespace NUMINAMATH_CALUDE_yulia_profit_is_44_l2161_216170

/-- Calculates Yulia's profit given her revenues and expenses -/
def yulia_profit (lemonade_revenue babysitting_revenue lemonade_expenses : ℕ) : ℕ :=
  (lemonade_revenue + babysitting_revenue) - lemonade_expenses

/-- Proves that Yulia's profit is $44 given the provided revenues and expenses -/
theorem yulia_profit_is_44 :
  yulia_profit 47 31 34 = 44 := by
  sorry

end NUMINAMATH_CALUDE_yulia_profit_is_44_l2161_216170


namespace NUMINAMATH_CALUDE_opposite_of_four_l2161_216165

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- The opposite of 4 is -4. -/
theorem opposite_of_four : opposite 4 = -4 := by sorry

end NUMINAMATH_CALUDE_opposite_of_four_l2161_216165


namespace NUMINAMATH_CALUDE_multiplier_problem_l2161_216191

theorem multiplier_problem (n : ℝ) (m : ℝ) : 
  n = 3 → 7 * n = m * n + 12 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l2161_216191


namespace NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2161_216112

/-- Represents a color (Red or Blue) -/
inductive Color
| Red
| Blue

/-- Represents a 4 x 7 chessboard coloring -/
def Coloring := Fin 4 → Fin 7 → Color

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  top_left : Fin 4 × Fin 7
  bottom_right : Fin 4 × Fin 7

/-- Check if a rectangle has all corners of the same color -/
def has_same_color_corners (c : Coloring) (r : Rectangle) : Prop :=
  let (t, l) := r.top_left
  let (b, r) := r.bottom_right
  c t l = c t r ∧ c t l = c b l ∧ c t l = c b r

/-- Main theorem: For any coloring of a 4 x 7 chessboard, 
    there exists a rectangle with four corners of the same color -/
theorem chessboard_coloring_theorem :
  ∀ (c : Coloring), ∃ (r : Rectangle), has_same_color_corners c r :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2161_216112


namespace NUMINAMATH_CALUDE_second_brother_tells_truth_l2161_216100

-- Define the type for card suits
inductive Suit
| Hearts
| Diamonds
| Clubs
| Spades

-- Define the type for brothers
inductive Brother
| First
| Second

-- Define the statements made by the brothers
def statement (b : Brother) : Prop :=
  match b with
  | Brother.First => ∀ (s1 s2 : Suit), s1 = s2
  | Brother.Second => ∃ (s1 s2 : Suit), s1 ≠ s2

-- Define the truth-telling property
def tellsTruth (b : Brother) : Prop := statement b

-- Theorem statement
theorem second_brother_tells_truth :
  (∃! (b : Brother), tellsTruth b) →
  (∀ (b1 b2 : Brother), b1 ≠ b2 → (tellsTruth b1 ↔ ¬tellsTruth b2)) →
  tellsTruth Brother.Second :=
by sorry

end NUMINAMATH_CALUDE_second_brother_tells_truth_l2161_216100


namespace NUMINAMATH_CALUDE_lcm_12_18_l2161_216178

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l2161_216178


namespace NUMINAMATH_CALUDE_sum_of_roots_l2161_216180

-- Define the quadratic equation
def quadratic_equation (x m n : ℝ) : Prop := 2 * x^2 + m * x + n = 0

-- State the theorem
theorem sum_of_roots (m n : ℝ) 
  (hm : quadratic_equation m m n) 
  (hn : quadratic_equation n m n) : 
  m + n = -m / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2161_216180


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2161_216193

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (3 * a) * (a / 2) * a - a^3 = 2 * a^2 → a^3 = 64 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2161_216193


namespace NUMINAMATH_CALUDE_driver_work_days_l2161_216152

/-- Represents the number of days driven from Monday to Wednesday -/
def days_mon_to_wed : ℕ := 3

/-- Represents the number of days driven from Thursday to Friday -/
def days_thu_to_fri : ℕ := 2

/-- Average driving hours per day -/
def avg_hours_per_day : ℕ := 2

/-- Average speed from Monday to Wednesday in km/h -/
def speed_mon_to_wed : ℕ := 12

/-- Average speed from Thursday to Friday in km/h -/
def speed_thu_to_fri : ℕ := 9

/-- Total distance traveled in km -/
def total_distance : ℕ := 108

theorem driver_work_days : 
  days_mon_to_wed * avg_hours_per_day * speed_mon_to_wed + 
  days_thu_to_fri * avg_hours_per_day * speed_thu_to_fri = total_distance ∧
  days_mon_to_wed + days_thu_to_fri = 5 :=
by sorry

end NUMINAMATH_CALUDE_driver_work_days_l2161_216152


namespace NUMINAMATH_CALUDE_derivative_of_ln_2_minus_3x_l2161_216122

open Real

theorem derivative_of_ln_2_minus_3x (x : ℝ) : 
  deriv (λ x => Real.log (2 - 3*x)) x = 3 / (3*x - 2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_ln_2_minus_3x_l2161_216122


namespace NUMINAMATH_CALUDE_triangle_area_rational_l2161_216155

theorem triangle_area_rational (x₁ x₂ x₃ y₁ y₂ y₃ : ℤ) :
  ∃ (q : ℚ), q = (1/2) * |((x₁ + (1/2 : ℚ)) * (y₂ - y₃) + 
                           (x₂ + (1/2 : ℚ)) * (y₃ - y₁) + 
                           (x₃ + (1/2 : ℚ)) * (y₁ - y₂))| := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l2161_216155
