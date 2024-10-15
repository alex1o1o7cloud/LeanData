import Mathlib

namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l3785_378555

theorem cubic_minus_linear_factorization (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l3785_378555


namespace NUMINAMATH_CALUDE_lg_expression_equals_one_l3785_378526

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_one :
  lg 2 * lg 5 + (lg 5)^2 + lg 2 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_expression_equals_one_l3785_378526


namespace NUMINAMATH_CALUDE_tooth_arrangements_l3785_378571

def word_length : Nat := 5
def repeated_letter_count : Nat := 2

theorem tooth_arrangements : 
  (word_length.factorial) / (repeated_letter_count.factorial * repeated_letter_count.factorial) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tooth_arrangements_l3785_378571


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l3785_378577

/-- Given a hyperbola y = k/x where the point (3, -2) lies on it, 
    prove that the point (-2, 3) also lies on the same hyperbola. -/
theorem point_on_hyperbola (k : ℝ) : 
  (∃ k, k = 3 * (-2) ∧ -2 = k / 3) → 
  (∃ k, k = (-2) * 3 ∧ 3 = k / (-2)) := by
  sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l3785_378577


namespace NUMINAMATH_CALUDE_f_is_odd_l3785_378517

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l3785_378517


namespace NUMINAMATH_CALUDE_expand_expression_l3785_378525

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3785_378525


namespace NUMINAMATH_CALUDE_egg_plant_theorem_l3785_378573

/-- Represents the egg processing plant scenario --/
structure EggPlant where
  accepted : ℕ
  rejected : ℕ
  total : ℕ
  accepted_to_rejected_ratio : ℚ

/-- The initial state of the egg plant --/
def initial_state : EggPlant := {
  accepted := 0,
  rejected := 0,
  total := 400,
  accepted_to_rejected_ratio := 0
}

/-- The state after additional eggs are accepted --/
def modified_state (initial : EggPlant) : EggPlant := {
  accepted := initial.accepted + 12,
  rejected := initial.rejected - 4,
  total := initial.total,
  accepted_to_rejected_ratio := 99/1
}

/-- The theorem to prove --/
theorem egg_plant_theorem (initial : EggPlant) : 
  initial.accepted = 392 ∧ 
  initial.rejected = 8 ∧
  initial.total = 400 ∧
  (initial.accepted : ℚ) / initial.rejected = (initial.accepted + 12 : ℚ) / (initial.rejected - 4) ∧
  (initial.accepted + 12 : ℚ) / (initial.rejected - 4) = 99/1 := by
  sorry

end NUMINAMATH_CALUDE_egg_plant_theorem_l3785_378573


namespace NUMINAMATH_CALUDE_wickets_before_last_match_is_85_l3785_378513

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 85 -/
theorem wickets_before_last_match_is_85 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 5)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 85 :=
by sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_is_85_l3785_378513


namespace NUMINAMATH_CALUDE_matrix_product_zero_l3785_378580

variable {R : Type*} [CommRing R]

def A (d e : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e],
    ![-d, 0, d],
    ![e, -d, 0]]

def B (d e : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![d * d, d * e, d * d],
    ![d * e, e * e, e * d],
    ![d * d, e * d, d * d]]

theorem matrix_product_zero (d e : R) (h1 : d = e) :
  A d e * B d e = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_zero_l3785_378580


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_negative_two_satisfies_inequality_smallest_integer_is_negative_two_l3785_378558

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * x^2 - 4 < 20) → x ≥ -2 :=
by
  sorry

theorem negative_two_satisfies_inequality :
  3 * (-2)^2 - 4 < 20 :=
by
  sorry

theorem smallest_integer_is_negative_two :
  ∃ x : ℤ, (∀ y : ℤ, (3 * y^2 - 4 < 20) → y ≥ x) ∧ (3 * x^2 - 4 < 20) ∧ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_negative_two_satisfies_inequality_smallest_integer_is_negative_two_l3785_378558


namespace NUMINAMATH_CALUDE_x_power_8000_minus_inverse_l3785_378569

theorem x_power_8000_minus_inverse (x : ℂ) : 
  x - 1/x = 2*Complex.I → x^8000 - 1/x^8000 = 0 := by sorry

end NUMINAMATH_CALUDE_x_power_8000_minus_inverse_l3785_378569


namespace NUMINAMATH_CALUDE_apple_grape_equivalence_l3785_378538

/-- Given that 3/4 of 12 apples are worth 9 grapes, 
    prove that 1/2 of 6 apples are worth 3 grapes -/
theorem apple_grape_equivalence : 
  (3/4 : ℚ) * 12 * (1 : ℚ) = 9 * (1 : ℚ) → 
  (1/2 : ℚ) * 6 * (1 : ℚ) = 3 * (1 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_apple_grape_equivalence_l3785_378538


namespace NUMINAMATH_CALUDE_first_super_lucky_year_l3785_378562

def is_valid_date (month day year : ℕ) : Prop :=
  1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31 ∧ year > 2000

def is_super_lucky_date (month day year : ℕ) : Prop :=
  is_valid_date month day year ∧ month * day = year % 100

def has_two_super_lucky_dates (year : ℕ) : Prop :=
  ∃ (m1 d1 m2 d2 : ℕ), 
    is_super_lucky_date m1 d1 year ∧ 
    is_super_lucky_date m2 d2 year ∧ 
    (m1 ≠ m2 ∨ d1 ≠ d2)

theorem first_super_lucky_year : 
  (∀ y, 2000 < y ∧ y < 2004 → ¬ has_two_super_lucky_dates y) ∧ 
  has_two_super_lucky_dates 2004 :=
sorry

end NUMINAMATH_CALUDE_first_super_lucky_year_l3785_378562


namespace NUMINAMATH_CALUDE_zoe_strawberry_count_l3785_378532

/-- The number of strawberries Zoe ate -/
def num_strawberries : ℕ := sorry

/-- The number of ounces of yogurt Zoe ate -/
def yogurt_ounces : ℕ := 6

/-- Calories per strawberry -/
def calories_per_strawberry : ℕ := 4

/-- Calories per ounce of yogurt -/
def calories_per_yogurt_ounce : ℕ := 17

/-- Total calories consumed -/
def total_calories : ℕ := 150

theorem zoe_strawberry_count :
  num_strawberries * calories_per_strawberry +
  yogurt_ounces * calories_per_yogurt_ounce = total_calories ∧
  num_strawberries = 12 := by sorry

end NUMINAMATH_CALUDE_zoe_strawberry_count_l3785_378532


namespace NUMINAMATH_CALUDE_bananas_removed_l3785_378575

theorem bananas_removed (original : ℕ) (remaining : ℕ) (removed : ℕ)
  (h1 : original = 46)
  (h2 : remaining = 41)
  (h3 : removed = original - remaining) :
  removed = 5 := by
  sorry

end NUMINAMATH_CALUDE_bananas_removed_l3785_378575


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3785_378566

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  n < 1000 ∧ 
  ∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3785_378566


namespace NUMINAMATH_CALUDE_not_equivalent_statement_and_converse_l3785_378551

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- Define the given lines and planes
variable (a b c : Line)
variable (α β : Plane)

-- State the theorem
theorem not_equivalent_statement_and_converse :
  (b ≠ a ∧ c ≠ a ∧ c ≠ b) →  -- three different lines
  (α ≠ β) →  -- two different planes
  (subset b α) →  -- b is a subset of α
  (¬ subset c α) →  -- c is not a subset of α
  ¬ (((perp b β → perpPlanes α β) ↔ (perpPlanes α β → perp b β))) :=
by sorry

end NUMINAMATH_CALUDE_not_equivalent_statement_and_converse_l3785_378551


namespace NUMINAMATH_CALUDE_percentage_problem_l3785_378559

theorem percentage_problem (x : ℝ) (h : 0.40 * x = 160) : 0.20 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3785_378559


namespace NUMINAMATH_CALUDE_expression_evaluation_l3785_378554

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 2
  (x + 2*y)^2 - x*(x + 4*y) + (1 - y)*(1 + y) = 7 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3785_378554


namespace NUMINAMATH_CALUDE_S_five_three_l3785_378534

def S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem S_five_three : S 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_S_five_three_l3785_378534


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3785_378561

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := 2 / (1 + i) + (1 - i)^2
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3785_378561


namespace NUMINAMATH_CALUDE_required_machines_eq_ten_l3785_378504

/-- The number of cell phones produced by 2 machines per minute -/
def phones_per_2machines : ℕ := 10

/-- The number of machines used in the given condition -/
def given_machines : ℕ := 2

/-- The desired number of cell phones to be produced per minute -/
def desired_phones : ℕ := 50

/-- Calculates the number of machines required to produce the desired number of phones per minute -/
def required_machines : ℕ := desired_phones * given_machines / phones_per_2machines

theorem required_machines_eq_ten : required_machines = 10 := by
  sorry

end NUMINAMATH_CALUDE_required_machines_eq_ten_l3785_378504


namespace NUMINAMATH_CALUDE_refrigerator_part_payment_l3785_378543

/-- Given a refrigerator purchase where a part payment of 25% has been made
    and $2625 remains to be paid (representing 75% of the total cost),
    prove that the part payment is equal to $875. -/
theorem refrigerator_part_payment
  (total_cost : ℝ)
  (part_payment_percentage : ℝ)
  (remaining_payment : ℝ)
  (remaining_percentage : ℝ)
  (h1 : part_payment_percentage = 0.25)
  (h2 : remaining_payment = 2625)
  (h3 : remaining_percentage = 0.75)
  (h4 : remaining_payment = remaining_percentage * total_cost) :
  part_payment_percentage * total_cost = 875 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_part_payment_l3785_378543


namespace NUMINAMATH_CALUDE_tan_y_plus_pi_third_l3785_378503

theorem tan_y_plus_pi_third (y : Real) (h : Real.tan y = -3) :
  Real.tan (y + π / 3) = -(5 * Real.sqrt 3 - 6) / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_y_plus_pi_third_l3785_378503


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3785_378531

/-- Given a line segment from (2,2) to (x,6) with length 5 and x > 0, prove x = 5 -/
theorem line_segment_endpoint (x : ℝ) 
  (h1 : (x - 2)^2 + (6 - 2)^2 = 5^2) 
  (h2 : x > 0) : 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3785_378531


namespace NUMINAMATH_CALUDE_negative_three_to_zero_power_l3785_378592

theorem negative_three_to_zero_power : (-3 : ℤ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_to_zero_power_l3785_378592


namespace NUMINAMATH_CALUDE_greatest_number_with_gcd_l3785_378557

theorem greatest_number_with_gcd (X : ℕ) : 
  X ≤ 840 ∧ 
  7 ∣ X ∧ 
  Nat.gcd X 91 = 7 ∧ 
  Nat.gcd X 840 = 7 →
  X = 840 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_gcd_l3785_378557


namespace NUMINAMATH_CALUDE_f_on_negative_interval_l3785_378530

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

theorem f_on_negative_interval 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : period_two f) 
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-1) 0, f x = 2 - x := by
sorry

end NUMINAMATH_CALUDE_f_on_negative_interval_l3785_378530


namespace NUMINAMATH_CALUDE_animals_left_after_sale_l3785_378542

/-- Calculates the number of animals left in a pet store after a sale --/
theorem animals_left_after_sale (siamese_cats house_cats dogs birds cats_sold dogs_sold birds_sold : ℕ) :
  siamese_cats = 25 →
  house_cats = 55 →
  dogs = 30 →
  birds = 20 →
  cats_sold = 45 →
  dogs_sold = 25 →
  birds_sold = 10 →
  (siamese_cats + house_cats - cats_sold) + (dogs - dogs_sold) + (birds - birds_sold) = 50 := by
sorry

end NUMINAMATH_CALUDE_animals_left_after_sale_l3785_378542


namespace NUMINAMATH_CALUDE_tangent_identity_l3785_378553

theorem tangent_identity (α β γ : Real) (h : α + β + γ = π/4) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_identity_l3785_378553


namespace NUMINAMATH_CALUDE_tian_ji_win_probability_l3785_378589

/-- Represents the tiers of horses -/
inductive Tier
| Top
| Middle
| Bottom

/-- Represents a horse with its owner and tier -/
structure Horse :=
  (owner : String)
  (tier : Tier)

/-- Determines if one horse is better than another -/
def isBetter (h1 h2 : Horse) : Prop := sorry

/-- The set of all horses in the competition -/
def allHorses : Finset Horse := sorry

/-- The set of Tian Ji's horses -/
def tianJiHorses : Finset Horse := sorry

/-- The set of King Qi's horses -/
def kingQiHorses : Finset Horse := sorry

/-- Axioms representing the given conditions -/
axiom horse_count : (tianJiHorses.card = 3) ∧ (kingQiHorses.card = 3)
axiom tian_ji_top_vs_qi_middle : 
  ∃ (ht hm : Horse), ht ∈ tianJiHorses ∧ hm ∈ kingQiHorses ∧ 
  ht.tier = Tier.Top ∧ hm.tier = Tier.Middle ∧ isBetter ht hm
axiom tian_ji_top_vs_qi_top : 
  ∃ (ht1 ht2 : Horse), ht1 ∈ tianJiHorses ∧ ht2 ∈ kingQiHorses ∧ 
  ht1.tier = Tier.Top ∧ ht2.tier = Tier.Top ∧ isBetter ht2 ht1
axiom tian_ji_middle_vs_qi_bottom : 
  ∃ (hm hb : Horse), hm ∈ tianJiHorses ∧ hb ∈ kingQiHorses ∧ 
  hm.tier = Tier.Middle ∧ hb.tier = Tier.Bottom ∧ isBetter hm hb
axiom tian_ji_middle_vs_qi_middle : 
  ∃ (hm1 hm2 : Horse), hm1 ∈ tianJiHorses ∧ hm2 ∈ kingQiHorses ∧ 
  hm1.tier = Tier.Middle ∧ hm2.tier = Tier.Middle ∧ isBetter hm2 hm1
axiom tian_ji_bottom_vs_qi_bottom : 
  ∃ (hb1 hb2 : Horse), hb1 ∈ tianJiHorses ∧ hb2 ∈ kingQiHorses ∧ 
  hb1.tier = Tier.Bottom ∧ hb2.tier = Tier.Bottom ∧ isBetter hb2 hb1

/-- The probability of Tian Ji's horse winning in a random matchup -/
def tianJiWinProbability : ℚ := sorry

/-- Main theorem: The probability of Tian Ji's horse winning is 1/3 -/
theorem tian_ji_win_probability : tianJiWinProbability = 1/3 := by sorry

end NUMINAMATH_CALUDE_tian_ji_win_probability_l3785_378589


namespace NUMINAMATH_CALUDE_find_m_l3785_378518

theorem find_m : ∃ m : ℝ, ∀ x : ℝ, (x + 2) * (x + 3) = x^2 + m*x + 6 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3785_378518


namespace NUMINAMATH_CALUDE_circle_a_properties_l3785_378550

/-- Circle A with center (m, 2/m) passing through origin -/
def CircleA (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - m)^2 + (p.2 - 2/m)^2 = m^2 + 4/m^2}

/-- Line l: 2x + y - 4 = 0 -/
def LineL : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0}

theorem circle_a_properties (m : ℝ) (hm : m > 0) :
  /- When m = 2, the circle equation is (x-2)² + (y-1)² = 5 -/
  (∀ p : ℝ × ℝ, p ∈ CircleA 2 ↔ (p.1 - 2)^2 + (p.2 - 1)^2 = 5) ∧
  /- The area of triangle OBC is constant and equal to 4 -/
  (∃ B C : ℝ × ℝ, B ∈ CircleA m ∧ C ∈ CircleA m ∧ B.2 = 0 ∧ C.1 = 0 ∧
    abs (B.1 * C.2) / 2 = 4) ∧
  /- If line l intersects circle A at P and Q where |OP| = |OQ|, then |PQ| = 4√30/5 -/
  (∃ P Q : ℝ × ℝ, P ∈ CircleA 2 ∧ Q ∈ CircleA 2 ∧ P ∈ LineL ∧ Q ∈ LineL ∧
    P.1^2 + P.2^2 = Q.1^2 + Q.2^2 →
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (4 * Real.sqrt 30 / 5)^2) :=
by sorry


end NUMINAMATH_CALUDE_circle_a_properties_l3785_378550


namespace NUMINAMATH_CALUDE_max_inverse_sum_l3785_378588

theorem max_inverse_sum (x y a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hax : a^x = 2) 
  (hby : b^y = 2) 
  (hab : 2*a + b = 8) : 
  (∀ w z : ℝ, a^w = 2 → b^z = 2 → 1/w + 1/z ≤ 3) ∧ 
  (∃ w z : ℝ, a^w = 2 ∧ b^z = 2 ∧ 1/w + 1/z = 3) :=
sorry

end NUMINAMATH_CALUDE_max_inverse_sum_l3785_378588


namespace NUMINAMATH_CALUDE_sequential_discount_equivalence_l3785_378519

/-- The equivalent single discount percentage for two sequential discounts -/
def equivalent_discount (first_discount second_discount : ℝ) : ℝ :=
  1 - (1 - first_discount) * (1 - second_discount)

/-- Theorem stating that a 15% discount followed by a 25% discount 
    is equivalent to a single 36.25% discount -/
theorem sequential_discount_equivalence : 
  equivalent_discount 0.15 0.25 = 0.3625 := by
  sorry

#eval equivalent_discount 0.15 0.25

end NUMINAMATH_CALUDE_sequential_discount_equivalence_l3785_378519


namespace NUMINAMATH_CALUDE_factorization_equality_l3785_378582

theorem factorization_equality (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3785_378582


namespace NUMINAMATH_CALUDE_ratio_equality_l3785_378527

theorem ratio_equality : (1722^2 - 1715^2) / (1729^2 - 1708^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3785_378527


namespace NUMINAMATH_CALUDE_square_side_length_l3785_378599

/-- Given six identical squares arranged to form a larger rectangle ABCD with an area of 3456,
    the side length of each square is 24. -/
theorem square_side_length (s : ℝ) : s > 0 → s * s * 6 = 3456 → s = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3785_378599


namespace NUMINAMATH_CALUDE_gcd_of_squares_l3785_378556

theorem gcd_of_squares : Nat.gcd (114^2 + 226^2 + 338^2) (113^2 + 225^2 + 339^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l3785_378556


namespace NUMINAMATH_CALUDE_tangent_circles_concyclic_points_l3785_378506

/-- Four circles are tangent consecutively if each circle is tangent to the next one in the sequence. -/
def ConsecutivelyTangentCircles (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) : Prop := sorry

/-- Four points are the tangent points of consecutively tangent circles if they are the points where each pair of consecutive circles touch. -/
def TangentPoints (A B C D : ℝ × ℝ) (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) : Prop := sorry

/-- Four points are concyclic if they lie on the same circle. -/
def Concyclic (A B C D : ℝ × ℝ) : Prop := sorry

/-- Theorem: If four circles are tangent to each other consecutively at four points, then these four points are concyclic. -/
theorem tangent_circles_concyclic_points
  (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) (A B C D : ℝ × ℝ) :
  ConsecutivelyTangentCircles Γ₁ Γ₂ Γ₃ Γ₄ →
  TangentPoints A B C D Γ₁ Γ₂ Γ₃ Γ₄ →
  Concyclic A B C D :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_concyclic_points_l3785_378506


namespace NUMINAMATH_CALUDE_dealers_dishonesty_percentage_l3785_378507

/-- The dealer's percentage of dishonesty in terms of weight -/
theorem dealers_dishonesty_percentage
  (standard_weight : ℝ)
  (dealer_weight : ℝ)
  (h1 : standard_weight = 16)
  (h2 : dealer_weight = 14.8) :
  (standard_weight - dealer_weight) / standard_weight * 100 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_dealers_dishonesty_percentage_l3785_378507


namespace NUMINAMATH_CALUDE_circumcircle_equation_l3785_378548

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (4, 2)

-- Define a predicate for points on the circumcircle of triangle ABP
def on_circumcircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem circumcircle_equation :
  ∀ A B : ℝ × ℝ,
  given_circle A.1 A.2 →
  given_circle B.1 B.2 →
  (∃ t : ℝ, A = (4 * t / (t^2 + 1), 2 * t^2 / (t^2 + 1))) →
  (∃ s : ℝ, B = (4 * s / (s^2 + 1), 2 * s^2 / (s^2 + 1))) →
  on_circumcircle A.1 A.2 ∧ on_circumcircle B.1 B.2 ∧ on_circumcircle point_P.1 point_P.2 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l3785_378548


namespace NUMINAMATH_CALUDE_shirt_purchase_problem_l3785_378515

theorem shirt_purchase_problem (shirt_price pants_price : ℝ) 
  (num_shirts : ℕ) (num_pants : ℕ) (total_cost refund : ℝ) :
  shirt_price ≠ pants_price →
  shirt_price = 45 →
  num_pants = 3 →
  total_cost = 120 →
  refund = 0.25 * total_cost →
  total_cost = num_shirts * shirt_price + num_pants * pants_price →
  total_cost - refund = num_shirts * shirt_price →
  num_shirts = 2 :=
by
  sorry

#check shirt_purchase_problem

end NUMINAMATH_CALUDE_shirt_purchase_problem_l3785_378515


namespace NUMINAMATH_CALUDE_f_monotonicity_f_two_zeros_l3785_378560

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 2)

theorem f_monotonicity :
  let f₁ := f 1
  (∀ x y, x < y → x < 0 → y < 0 → f₁ y < f₁ x) ∧
  (∀ x y, 0 < x → x < y → f₁ x < f₁ y) :=
sorry

theorem f_two_zeros (a : ℝ) :
  (∃ x y, x < y ∧ f a x = 0 ∧ f a y = 0) ↔ (Real.exp (-1) < a) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_f_two_zeros_l3785_378560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l3785_378529

/-- Prove that for an arithmetic sequence, R = 2n²d -/
theorem arithmetic_sequence_sum_relation 
  (a d n : ℝ) 
  (S₁ : ℝ := n / 2 * (2 * a + (n - 1) * d))
  (S₂ : ℝ := n * (2 * a + (2 * n - 1) * d))
  (S₃ : ℝ := 3 * n / 2 * (2 * a + (3 * n - 1) * d))
  (R : ℝ := S₃ - S₂ - S₁) :
  R = 2 * n^2 * d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l3785_378529


namespace NUMINAMATH_CALUDE_x_minus_y_equals_60_l3785_378523

theorem x_minus_y_equals_60 (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_60_l3785_378523


namespace NUMINAMATH_CALUDE_fruit_arrangement_theorem_l3785_378509

def number_of_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) (unique : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2 * Nat.factorial unique)

theorem fruit_arrangement_theorem :
  number_of_arrangements 7 4 2 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_theorem_l3785_378509


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3785_378579

theorem negative_fraction_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3785_378579


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1657_l3785_378581

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (a b c : ℕ) : ℕ := a * 13^2 + b * 13 + c

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (a b c : ℕ) : ℕ := a * 14^2 + b * 14 + c

/-- The value of digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equals_1657 :
  base13ToBase10 4 2 0 + base14ToBase10 4 C 3 = 1657 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1657_l3785_378581


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3785_378514

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3785_378514


namespace NUMINAMATH_CALUDE_expected_participants_l3785_378502

/-- The expected number of participants in a school clean-up event after three years,
    given an initial number of participants and an annual increase rate. -/
theorem expected_participants (initial : ℕ) (increase_rate : ℚ) :
  initial = 800 →
  increase_rate = 1/2 →
  (initial * (1 + increase_rate)^3 : ℚ) = 2700 := by
  sorry

end NUMINAMATH_CALUDE_expected_participants_l3785_378502


namespace NUMINAMATH_CALUDE_dish_washing_time_l3785_378549

theorem dish_washing_time (dawn_time andy_time : ℕ) : 
  andy_time = 2 * dawn_time + 6 →
  andy_time = 46 →
  dawn_time = 20 := by
sorry

end NUMINAMATH_CALUDE_dish_washing_time_l3785_378549


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3785_378585

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 3, 5}

-- Define set B
def B : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3785_378585


namespace NUMINAMATH_CALUDE_increasing_quadratic_condition_l3785_378508

/-- If f(x) = -x^2 + 2ax - 3 is increasing on (-∞, 4), then a < 4 -/
theorem increasing_quadratic_condition (a : ℝ) : 
  (∀ x < 4, Monotone (fun x => -x^2 + 2*a*x - 3)) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_condition_l3785_378508


namespace NUMINAMATH_CALUDE_imaginary_number_properties_l3785_378516

/-- An imaginary number is a complex number with a non-zero imaginary part -/
def IsImaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_number_properties (x y : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk x y) (h2 : IsImaginary z) : 
  x ∈ Set.univ ∧ y ≠ 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_number_properties_l3785_378516


namespace NUMINAMATH_CALUDE_cubic_polynomial_inequality_iff_coeff_conditions_l3785_378586

/-- A cubic polynomial -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a point -/
def eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The inequality condition for the polynomial -/
def satisfiesInequality (p : CubicPolynomial) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → eval p (x + y) ≥ eval p x + eval p y

/-- The conditions on the coefficients -/
def satisfiesCoeffConditions (p : CubicPolynomial) : Prop :=
  p.a > 0 ∧ p.d ≤ 0 ∧ 8 * p.b^3 ≥ 243 * p.a^2 * p.d

theorem cubic_polynomial_inequality_iff_coeff_conditions (p : CubicPolynomial) :
  satisfiesInequality p ↔ satisfiesCoeffConditions p := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_inequality_iff_coeff_conditions_l3785_378586


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3785_378578

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 + 2 > 3*x) ↔ (∀ x : ℝ, x^2 + 2 ≤ 3*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3785_378578


namespace NUMINAMATH_CALUDE_subtracted_value_l3785_378540

theorem subtracted_value (N : ℝ) (x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 4) / 10 = 5) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3785_378540


namespace NUMINAMATH_CALUDE_solutions_to_quadratic_equation_l3785_378598

theorem solutions_to_quadratic_equation :
  ∀ x : ℝ, 3 * x^2 = 27 ↔ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_solutions_to_quadratic_equation_l3785_378598


namespace NUMINAMATH_CALUDE_problem_solution_l3785_378567

theorem problem_solution (a b : ℝ) (h : a^2 + |b+1| = 0) : (a+b)^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3785_378567


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3785_378528

theorem simple_interest_rate_calculation (P : ℝ) (P_positive : P > 0) :
  let final_amount := (7 / 6) * P
  let time := 6
  let interest := final_amount - P
  let R := (interest / P / time) * 100
  R = 100 / 36 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3785_378528


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3785_378587

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 3 = d ∧ s^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3785_378587


namespace NUMINAMATH_CALUDE_reflection_over_x_axis_of_P_l3785_378541

/-- Reflects a point over the x-axis -/
def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point -/
def P : ℝ × ℝ := (2, -3)

theorem reflection_over_x_axis_of_P :
  reflect_over_x_axis P = (2, 3) := by sorry

end NUMINAMATH_CALUDE_reflection_over_x_axis_of_P_l3785_378541


namespace NUMINAMATH_CALUDE_percentage_of_difference_l3785_378501

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  P * (x - y) = 0.3 * (x + y) →
  y = (1/3) * x →
  P = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_difference_l3785_378501


namespace NUMINAMATH_CALUDE_exponent_equation_l3785_378593

theorem exponent_equation (a : ℝ) (m : ℝ) (h : a ≠ 0) : 
  a^(m + 1) * a^(2*m - 1) = a^9 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_l3785_378593


namespace NUMINAMATH_CALUDE_round_39_982_to_three_sig_figs_l3785_378535

/-- Rounds a number to a specified number of significant figures -/
def roundToSigFigs (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Checks if a real number has exactly n significant figures -/
def hasSigFigs (x : ℝ) (n : ℕ) : Prop := sorry

theorem round_39_982_to_three_sig_figs :
  let x := 39.982
  let result := roundToSigFigs x 3
  result = 40.0 ∧ hasSigFigs result 3 := by sorry

end NUMINAMATH_CALUDE_round_39_982_to_three_sig_figs_l3785_378535


namespace NUMINAMATH_CALUDE_minimal_m_value_l3785_378574

theorem minimal_m_value (n k : ℕ) (hn : n > k) (hk : k > 1) :
  let m := (10^n - 1) / (10^k - 1)
  (∀ n' k' : ℕ, n' > k' → k' > 1 → (10^n' - 1) / (10^k' - 1) ≥ m) →
  m = 101 := by
  sorry

end NUMINAMATH_CALUDE_minimal_m_value_l3785_378574


namespace NUMINAMATH_CALUDE_inequality_proof_l3785_378590

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3*a*b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3785_378590


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l3785_378563

theorem consecutive_even_sum (n : ℤ) : 
  (∃ m : ℤ, m = n + 2 ∧ (m^2 - n^2 = 84)) → n + (n + 2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l3785_378563


namespace NUMINAMATH_CALUDE_fraction_modification_l3785_378591

theorem fraction_modification (x : ℚ) : 
  x = 437 → (537 - x) / (463 + x) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_modification_l3785_378591


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3785_378597

theorem diophantine_equation_solutions :
  ∀ (a b : ℕ), (2017 : ℕ) ^ a = b ^ 6 - 32 * b + 1 ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3785_378597


namespace NUMINAMATH_CALUDE_max_abs_u_l3785_378594

/-- Given a complex number z with |z| = 1, prove that the maximum value of 
    |z^4 - z^3 - 3z^2i - z + 1| is 5 and occurs when z = -1 -/
theorem max_abs_u (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max : ℝ), max = 5 ∧
  Complex.abs (z^4 - z^3 - 3*z^2*Complex.I - z + 1) ≤ max ∧
  Complex.abs ((-1 : ℂ)^4 - (-1 : ℂ)^3 - 3*(-1 : ℂ)^2*Complex.I - (-1 : ℂ) + 1) = max :=
sorry

end NUMINAMATH_CALUDE_max_abs_u_l3785_378594


namespace NUMINAMATH_CALUDE_solve_average_height_l3785_378547

def average_height_problem (n : ℕ) (initial_average : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) : Prop :=
  let total_incorrect := n * initial_average
  let height_difference := incorrect_height - correct_height
  let total_correct := total_incorrect - height_difference
  let actual_average := total_correct / n
  actual_average = 174.25

theorem solve_average_height :
  average_height_problem 20 175 151 136 := by sorry

end NUMINAMATH_CALUDE_solve_average_height_l3785_378547


namespace NUMINAMATH_CALUDE_laptop_savings_l3785_378546

/-- The in-store price of the laptop in dollars -/
def in_store_price : ℚ := 299.99

/-- The cost of one payment in the radio offer in dollars -/
def radio_payment : ℚ := 55.98

/-- The number of payments in the radio offer -/
def num_payments : ℕ := 5

/-- The shipping and handling charge in dollars -/
def shipping_charge : ℚ := 12.99

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

theorem laptop_savings : 
  (in_store_price - (radio_payment * num_payments + shipping_charge)) * cents_per_dollar = 710 := by
  sorry

end NUMINAMATH_CALUDE_laptop_savings_l3785_378546


namespace NUMINAMATH_CALUDE_xyz_sum_l3785_378512

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l3785_378512


namespace NUMINAMATH_CALUDE_fundraiser_proof_l3785_378595

/-- The number of students asked to bring brownies -/
def num_brownie_students : ℕ := 30

/-- The number of brownies each student brings -/
def brownies_per_student : ℕ := 12

/-- The number of students asked to bring cookies -/
def num_cookie_students : ℕ := 20

/-- The number of cookies each student brings -/
def cookies_per_student : ℕ := 24

/-- The number of students asked to bring donuts -/
def num_donut_students : ℕ := 15

/-- The number of donuts each student brings -/
def donuts_per_student : ℕ := 12

/-- The price of each item in dollars -/
def price_per_item : ℕ := 2

/-- The total amount raised in dollars -/
def total_amount_raised : ℕ := 2040

theorem fundraiser_proof :
  num_brownie_students * brownies_per_student * price_per_item +
  num_cookie_students * cookies_per_student * price_per_item +
  num_donut_students * donuts_per_student * price_per_item =
  total_amount_raised :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_proof_l3785_378595


namespace NUMINAMATH_CALUDE_trig_expression_value_l3785_378552

theorem trig_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l3785_378552


namespace NUMINAMATH_CALUDE_girls_in_school_l3785_378524

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (boys girls : ℕ), boys + girls = sample_size ∧ girls = boys - 10) :
  ∃ (school_girls : ℕ), school_girls = 760 ∧ 
    school_girls * sample_size = total_students * 95 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l3785_378524


namespace NUMINAMATH_CALUDE_absolute_difference_of_xy_l3785_378564

theorem absolute_difference_of_xy (x y : ℝ) 
  (h1 : x * y = 6) 
  (h2 : x + y = 7) : 
  |x - y| = 5 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_xy_l3785_378564


namespace NUMINAMATH_CALUDE_floral_arrangement_daisies_percentage_l3785_378505

theorem floral_arrangement_daisies_percentage
  (total : ℝ)
  (yellow_flowers : ℝ)
  (blue_flowers : ℝ)
  (yellow_tulips : ℝ)
  (blue_tulips : ℝ)
  (yellow_daisies : ℝ)
  (blue_daisies : ℝ)
  (h1 : yellow_flowers = 7 / 10 * total)
  (h2 : blue_flowers = 3 / 10 * total)
  (h3 : yellow_tulips = 1 / 2 * yellow_flowers)
  (h4 : blue_daisies = 1 / 3 * blue_flowers)
  (h5 : yellow_flowers + blue_flowers = total)
  (h6 : yellow_tulips + blue_tulips + yellow_daisies + blue_daisies = total)
  : (yellow_daisies + blue_daisies) / total = 9 / 20 :=
by sorry

end NUMINAMATH_CALUDE_floral_arrangement_daisies_percentage_l3785_378505


namespace NUMINAMATH_CALUDE_closest_to_100_l3785_378565

def expression : ℝ := (2.1 * (50.2 + 0.08)) - 5

def options : List ℝ := [95, 100, 101, 105]

theorem closest_to_100 : 
  ∀ x ∈ options, |expression - 100| ≤ |expression - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_100_l3785_378565


namespace NUMINAMATH_CALUDE_initial_discount_percentage_l3785_378537

/-- Given a dress with original price d and an initial discount percentage x,
    prove that x = 65 when a staff member pays 0.14d after an additional 60% discount. -/
theorem initial_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  0.40 * (1 - x / 100) * d = 0.14 * d → x = 65 := by
  sorry

end NUMINAMATH_CALUDE_initial_discount_percentage_l3785_378537


namespace NUMINAMATH_CALUDE_equal_digging_time_l3785_378539

/-- Given the same number of people, if it takes a certain number of days to dig one volume,
    it will take the same number of days to dig an equal volume. -/
theorem equal_digging_time (people : ℕ) (depth1 length1 breadth1 depth2 length2 breadth2 : ℝ)
  (days : ℝ) (h1 : depth1 * length1 * breadth1 = depth2 * length2 * breadth2)
  (h2 : depth1 = 100) (h3 : length1 = 25) (h4 : breadth1 = 30)
  (h5 : depth2 = 75) (h6 : length2 = 20) (h7 : breadth2 = 50)
  (h8 : days = 12) :
  days = 12 := by
  sorry

#check equal_digging_time

end NUMINAMATH_CALUDE_equal_digging_time_l3785_378539


namespace NUMINAMATH_CALUDE_problem_solution_l3785_378536

theorem problem_solution (a b : ℝ) 
  (h1 : 2 + a = 5 - b) 
  (h2 : 5 + b = 8 + a) : 
  2 - a = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3785_378536


namespace NUMINAMATH_CALUDE_max_distance_for_specific_bicycle_l3785_378568

/-- Represents a bicycle with swappable tires -/
structure Bicycle where
  front_tire_life : ℝ
  rear_tire_life : ℝ

/-- Calculates the maximum distance a bicycle can travel with tire swapping -/
def max_distance (b : Bicycle) : ℝ :=
  sorry

/-- Theorem stating the maximum distance for a specific bicycle -/
theorem max_distance_for_specific_bicycle :
  let b : Bicycle := { front_tire_life := 5000, rear_tire_life := 3000 }
  max_distance b = 3750 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_specific_bicycle_l3785_378568


namespace NUMINAMATH_CALUDE_rectangle_area_error_l3785_378572

theorem rectangle_area_error (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let measured_length := 1.15 * L
  let measured_width := 1.20 * W
  let true_area := L * W
  let calculated_area := measured_length * measured_width
  (calculated_area - true_area) / true_area * 100 = 38 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_l3785_378572


namespace NUMINAMATH_CALUDE_population_equality_l3785_378596

/-- The number of years it takes for two villages' populations to be equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  18

theorem population_equality (x_initial y_initial x_decrease y_increase : ℕ)
  (h1 : x_initial = 78000)
  (h2 : x_decrease = 1200)
  (h3 : y_initial = 42000)
  (h4 : y_increase = 800) :
  x_initial - x_decrease * (years_until_equal_population x_initial x_decrease y_initial y_increase) =
  y_initial + y_increase * (years_until_equal_population x_initial x_decrease y_initial y_increase) :=
by sorry

end NUMINAMATH_CALUDE_population_equality_l3785_378596


namespace NUMINAMATH_CALUDE_parentheses_removal_l3785_378544

theorem parentheses_removal (a b : ℝ) : -(-a + b - 1) = a - b + 1 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l3785_378544


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3785_378533

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of f(x) ≥ 0 is [-1, 4] -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : Set.Icc (-1 : ℝ) 4 = {x | QuadraticFunction a b c x ≥ 0}) :
  QuadraticFunction a b c 2 > QuadraticFunction a b c 3 ∧ 
  QuadraticFunction a b c 3 > QuadraticFunction a b c (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3785_378533


namespace NUMINAMATH_CALUDE_carbonate_weight_proof_l3785_378584

/-- The molecular weight of the carbonate part in Al2(CO3)3 -/
def carbonate_weight (total_weight : ℝ) (al_weight : ℝ) : ℝ :=
  total_weight - 2 * al_weight

/-- Proof that the molecular weight of the carbonate part in Al2(CO3)3 is 180.04 g/mol -/
theorem carbonate_weight_proof (total_weight : ℝ) (al_weight : ℝ) 
  (h1 : total_weight = 234)
  (h2 : al_weight = 26.98) :
  carbonate_weight total_weight al_weight = 180.04 := by
  sorry

#eval carbonate_weight 234 26.98

end NUMINAMATH_CALUDE_carbonate_weight_proof_l3785_378584


namespace NUMINAMATH_CALUDE_luke_received_21_dollars_l3785_378570

/-- Calculates the amount of money Luke received from his mom -/
def money_from_mom (initial amount_spent final : ℕ) : ℕ :=
  final - (initial - amount_spent)

/-- Proves that Luke received 21 dollars from his mom -/
theorem luke_received_21_dollars :
  money_from_mom 48 11 58 = 21 := by
  sorry

end NUMINAMATH_CALUDE_luke_received_21_dollars_l3785_378570


namespace NUMINAMATH_CALUDE_expo_stamps_theorem_l3785_378510

theorem expo_stamps_theorem (total_cost : ℕ) (cost_4 cost_8 : ℕ) (difference : ℕ) :
  total_cost = 660 →
  cost_4 = 4 →
  cost_8 = 8 →
  difference = 30 →
  ∃ (stamps_4 stamps_8 : ℕ),
    stamps_8 = stamps_4 + difference ∧
    total_cost = cost_4 * stamps_4 + cost_8 * stamps_8 →
    stamps_4 + stamps_8 = 100 :=
by sorry

end NUMINAMATH_CALUDE_expo_stamps_theorem_l3785_378510


namespace NUMINAMATH_CALUDE_jean_calories_l3785_378545

/-- Calculates the total calories consumed based on pages written and calories per donut -/
def total_calories (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

/-- Proves that Jean eats 900 calories given the conditions -/
theorem jean_calories : total_calories 12 2 150 = 900 := by
  sorry

end NUMINAMATH_CALUDE_jean_calories_l3785_378545


namespace NUMINAMATH_CALUDE_image_of_one_three_l3785_378522

/-- A set of ordered pairs of real numbers -/
def RealPair : Type := ℝ × ℝ

/-- The mapping f: A → B -/
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

/-- Theorem: The image of (1, 3) under f is (-2, 4) -/
theorem image_of_one_three :
  f (1, 3) = (-2, 4) := by
  sorry

end NUMINAMATH_CALUDE_image_of_one_three_l3785_378522


namespace NUMINAMATH_CALUDE_hotdog_competition_ratio_l3785_378511

/-- Hotdog eating competition rates and ratios -/
theorem hotdog_competition_ratio :
  let first_rate := 10 -- hot dogs per minute
  let second_rate := 3 * first_rate
  let third_rate := 300 / 5 -- 300 hot dogs in 5 minutes
  third_rate / second_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_competition_ratio_l3785_378511


namespace NUMINAMATH_CALUDE_cube_difference_l3785_378583

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : 
  a^3 - b^3 = 108 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l3785_378583


namespace NUMINAMATH_CALUDE_age_difference_l3785_378500

/-- Given the ages of four people with specific relationships, prove that Jack's age is 5 years more than twice Shannen's age. -/
theorem age_difference (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age > 2 * shannen_age →
  beckett_age + olaf_age + shannen_age + jack_age = 71 →
  jack_age - 2 * shannen_age = 5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3785_378500


namespace NUMINAMATH_CALUDE_no_prime_sum_56_l3785_378520

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the property we want to prove
theorem no_prime_sum_56 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 56 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_56_l3785_378520


namespace NUMINAMATH_CALUDE_missed_solution_l3785_378576

theorem missed_solution (x : ℝ) : x * (x - 3) = x - 3 → (x = 1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_missed_solution_l3785_378576


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3785_378521

theorem trigonometric_identities (x : Real) (h : Real.tan x = 2) :
  (2/3 * Real.sin x^2 + 1/4 * Real.cos x^2 = 7/12) ∧
  (2 * Real.sin x^2 - Real.sin x * Real.cos x + Real.cos x^2 = 7/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3785_378521
