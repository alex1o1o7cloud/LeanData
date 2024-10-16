import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l3252_325236

theorem divisibility_by_twelve (a b c d : ℤ) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l3252_325236


namespace NUMINAMATH_CALUDE_turnover_equation_l3252_325233

/-- Represents the turnover equation for an online store over three months -/
theorem turnover_equation (x : ℝ) : 
  let july_turnover : ℝ := 16
  let august_turnover : ℝ := july_turnover * (1 + x)
  let september_turnover : ℝ := august_turnover * (1 + x)
  let total_turnover : ℝ := 120
  july_turnover + august_turnover + september_turnover = total_turnover :=
by
  sorry

#check turnover_equation

end NUMINAMATH_CALUDE_turnover_equation_l3252_325233


namespace NUMINAMATH_CALUDE_raja_savings_l3252_325290

def monthly_income : ℝ := 37500

def household_percentage : ℝ := 35
def clothes_percentage : ℝ := 20
def medicines_percentage : ℝ := 5

def total_expenditure_percentage : ℝ := household_percentage + clothes_percentage + medicines_percentage

def savings_percentage : ℝ := 100 - total_expenditure_percentage

theorem raja_savings : (savings_percentage / 100) * monthly_income = 15000 := by
  sorry

end NUMINAMATH_CALUDE_raja_savings_l3252_325290


namespace NUMINAMATH_CALUDE_janet_earnings_theorem_l3252_325218

/-- Calculates Janet's earnings per hour based on the number of posts checked and payment rates. -/
def janet_earnings_per_hour (text_posts image_posts video_posts : ℕ) 
  (text_rate image_rate video_rate : ℚ) : ℚ :=
  text_posts * text_rate + image_posts * image_rate + video_posts * video_rate

/-- Proves that Janet's earnings per hour equal $69.50 given the specified conditions. -/
theorem janet_earnings_theorem : 
  janet_earnings_per_hour 150 80 20 0.25 0.30 0.40 = 69.50 := by
  sorry

end NUMINAMATH_CALUDE_janet_earnings_theorem_l3252_325218


namespace NUMINAMATH_CALUDE_average_diesel_cost_approx_9_94_l3252_325280

/-- Represents the diesel purchase data for a single year -/
structure YearlyPurchase where
  litres : ℝ
  pricePerLitre : ℝ

/-- Calculates the total cost for a year including delivery fees and taxes -/
def yearlyTotalCost (purchase : YearlyPurchase) (deliveryFee : ℝ) (taxes : ℝ) : ℝ :=
  purchase.litres * purchase.pricePerLitre + deliveryFee + taxes

/-- Theorem: The average cost per litre of diesel over three years is approximately 9.94 -/
theorem average_diesel_cost_approx_9_94 
  (year1 : YearlyPurchase)
  (year2 : YearlyPurchase)
  (year3 : YearlyPurchase)
  (deliveryFee : ℝ)
  (taxes : ℝ)
  (h1 : year1.litres = 520 ∧ year1.pricePerLitre = 8.5)
  (h2 : year2.litres = 540 ∧ year2.pricePerLitre = 9)
  (h3 : year3.litres = 560 ∧ year3.pricePerLitre = 9.5)
  (h4 : deliveryFee = 200)
  (h5 : taxes = 300) :
  let totalCost := yearlyTotalCost year1 deliveryFee taxes + 
                   yearlyTotalCost year2 deliveryFee taxes + 
                   yearlyTotalCost year3 deliveryFee taxes
  let totalLitres := year1.litres + year2.litres + year3.litres
  let averageCost := totalCost / totalLitres
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |averageCost - 9.94| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_diesel_cost_approx_9_94_l3252_325280


namespace NUMINAMATH_CALUDE_product_of_square_roots_l3252_325215

theorem product_of_square_roots (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (7 * q^5) = 29 * q^4 * Real.sqrt 840 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l3252_325215


namespace NUMINAMATH_CALUDE_marble_probability_difference_l3252_325240

/-- The number of red marbles in the box -/
def red : ℕ := 500

/-- The number of black marbles in the box -/
def black : ℕ := 700

/-- The number of blue marbles in the box -/
def blue : ℕ := 800

/-- The total number of marbles in the box -/
def total : ℕ := red + black + blue

/-- The probability of drawing two marbles of the same color -/
noncomputable def Ps : ℚ := 
  (red * (red - 1) + black * (black - 1) + blue * (blue - 1)) / (total * (total - 1))

/-- The probability of drawing two marbles of different colors -/
noncomputable def Pd : ℚ := 
  (red * black + red * blue + black * blue) * 2 / (total * (total - 1))

/-- Theorem stating that the absolute difference between Ps and Pd is 31/100 -/
theorem marble_probability_difference : |Ps - Pd| = 31 / 100 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l3252_325240


namespace NUMINAMATH_CALUDE_blue_to_purple_ratio_l3252_325207

/-- Represents the number of beads of each color in a necklace. -/
structure BeadCounts where
  purple : ℕ
  blue : ℕ
  green : ℕ

/-- The properties of the necklace as described in the problem. -/
def necklace_properties (b : BeadCounts) : Prop :=
  b.purple = 7 ∧
  b.green = b.blue + 11 ∧
  b.purple + b.blue + b.green = 46

/-- The theorem stating the ratio of blue to purple beads is 2:1. -/
theorem blue_to_purple_ratio (b : BeadCounts) :
  necklace_properties b → b.blue = 2 * b.purple := by
  sorry

end NUMINAMATH_CALUDE_blue_to_purple_ratio_l3252_325207


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3252_325271

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3252_325271


namespace NUMINAMATH_CALUDE_construct_from_blocks_l3252_325251

/-- A building block consists of 7 unit cubes in a 2x2x2 shape with one corner unit cube missing. -/
structure BuildingBlock :=
  (size : Nat)
  (unit_cubes : Nat)

/-- Definition of our specific building block -/
def specific_block : BuildingBlock :=
  { size := 2,
    unit_cubes := 7 }

/-- A cube with one unit removed -/
structure CubeWithUnitRemoved :=
  (edge_length : Nat)
  (total_units : Nat)

/-- Function to check if a cube with a unit removed can be constructed from building blocks -/
def can_construct (c : CubeWithUnitRemoved) (b : BuildingBlock) : Prop :=
  ∃ (num_blocks : Nat), c.total_units = num_blocks * b.unit_cubes

/-- Main theorem -/
theorem construct_from_blocks (n : Nat) (h : n ≥ 2) :
  let c := CubeWithUnitRemoved.mk (2^n) ((2^n)^3 - 1)
  can_construct c specific_block :=
by sorry

end NUMINAMATH_CALUDE_construct_from_blocks_l3252_325251


namespace NUMINAMATH_CALUDE_totalDispatchPlansIs36_l3252_325235

/-- The number of people to choose from -/
def totalPeople : Nat := 5

/-- The number of tasks to be assigned -/
def totalTasks : Nat := 4

/-- The number of people who can only do certain tasks -/
def restrictedPeople : Nat := 2

/-- The number of tasks that restricted people can do -/
def restrictedTasks : Nat := 2

/-- The number of people who can do any task -/
def unrestrictedPeople : Nat := totalPeople - restrictedPeople

/-- Calculate the number of ways to select and arrange k items from n items -/
def arrangementNumber (n k : Nat) : Nat := sorry

/-- Calculate the number of ways to select k items from n items -/
def combinationNumber (n k : Nat) : Nat := sorry

/-- The total number of different dispatch plans -/
def totalDispatchPlans : Nat :=
  combinationNumber restrictedPeople 1 * combinationNumber restrictedTasks 1 * 
    arrangementNumber unrestrictedPeople 3 +
  arrangementNumber restrictedPeople 2 * arrangementNumber unrestrictedPeople 2

/-- Theorem stating that the total number of different dispatch plans is 36 -/
theorem totalDispatchPlansIs36 : totalDispatchPlans = 36 := by sorry

end NUMINAMATH_CALUDE_totalDispatchPlansIs36_l3252_325235


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l3252_325227

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 3033 ≥ 3032 :=
sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 3033 = 3032 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l3252_325227


namespace NUMINAMATH_CALUDE_peters_birdseed_calculation_l3252_325264

/-- The amount of birdseed needed for a week given the number of birds and their daily consumption --/
def birdseed_needed (parakeet_count : ℕ) (parrot_count : ℕ) (finch_count : ℕ) 
  (parakeet_consumption : ℕ) (parrot_consumption : ℕ) (days : ℕ) : ℕ :=
  let finch_consumption := parakeet_consumption / 2
  let daily_total := parakeet_count * parakeet_consumption + 
                     parrot_count * parrot_consumption + 
                     finch_count * finch_consumption
  daily_total * days

/-- Theorem stating that Peter needs to buy 266 grams of birdseed for a week --/
theorem peters_birdseed_calculation :
  birdseed_needed 3 2 4 2 14 7 = 266 := by
  sorry


end NUMINAMATH_CALUDE_peters_birdseed_calculation_l3252_325264


namespace NUMINAMATH_CALUDE_M_equals_m_plus_one_l3252_325291

/-- Given natural numbers n, m, h, and b, where n ≥ h(m+1) and h ≥ 1,
    M_{(n, n m, b)} represents a certain combinatorial property. -/
def M (n m h b : ℕ) : ℕ := sorry

/-- Theorem stating that M_{(n, n m, b)} = m + 1 under given conditions -/
theorem M_equals_m_plus_one (n m h b : ℕ) (h1 : n ≥ h * (m + 1)) (h2 : h ≥ 1) :
  M n m h b = m + 1 := by
  sorry

end NUMINAMATH_CALUDE_M_equals_m_plus_one_l3252_325291


namespace NUMINAMATH_CALUDE_mod_inverse_sum_eq_50_l3252_325267

theorem mod_inverse_sum_eq_50 : (2 * (5⁻¹ : ZMod 56) + 8 * (11⁻¹ : ZMod 56)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_eq_50_l3252_325267


namespace NUMINAMATH_CALUDE_davontesCollectionArea_l3252_325254

/-- Represents the dimensions of a painting -/
structure PaintingDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a painting given its dimensions -/
def paintingArea (d : PaintingDimensions) : ℝ :=
  d.length * d.width

/-- Represents Davonte's art collection -/
structure ArtCollection where
  squarePaintings : List PaintingDimensions
  smallPaintings : List PaintingDimensions
  largePainting : PaintingDimensions

/-- Calculates the total area of an art collection -/
def totalArea (collection : ArtCollection) : ℝ :=
  (collection.squarePaintings.map paintingArea).sum +
  (collection.smallPaintings.map paintingArea).sum +
  paintingArea collection.largePainting

/-- Davonte's art collection -/
def davontesCollection : ArtCollection :=
  { squarePaintings := List.replicate 3 { length := 6, width := 6 }
    smallPaintings := List.replicate 4 { length := 2, width := 3 }
    largePainting := { length := 10, width := 15 } }

theorem davontesCollectionArea :
  totalArea davontesCollection = 282 := by
  sorry

end NUMINAMATH_CALUDE_davontesCollectionArea_l3252_325254


namespace NUMINAMATH_CALUDE_birth_details_correct_l3252_325255

/-- Represents the ages and relationships of family members -/
structure Family where
  teresa_age : ℕ
  morio_age : ℕ
  morio_age_at_michiko_birth : ℕ
  kenji_michiko_age_diff : ℕ
  yuki_kenji_age_diff : ℕ
  years_after_adoption_to_anniversary : ℕ
  wedding_anniversary : ℕ

/-- Calculates Teresa's age when Michiko was born and the years into marriage -/
def birth_details (f : Family) : ℕ × ℕ :=
  let michiko_age := f.morio_age - f.morio_age_at_michiko_birth
  let kenji_age := michiko_age - f.kenji_michiko_age_diff
  let yuki_age := kenji_age - f.yuki_kenji_age_diff
  let years_married_at_adoption := yuki_age - f.years_after_adoption_to_anniversary - f.wedding_anniversary
  let years_into_marriage := michiko_age - years_married_at_adoption
  let teresa_age_at_birth := f.teresa_age - michiko_age
  (teresa_age_at_birth, years_into_marriage)

theorem birth_details_correct (f : Family) 
  (h1 : f.teresa_age = 59)
  (h2 : f.morio_age = 71)
  (h3 : f.morio_age_at_michiko_birth = 38)
  (h4 : f.kenji_michiko_age_diff = 4)
  (h5 : f.yuki_kenji_age_diff = 3)
  (h6 : f.years_after_adoption_to_anniversary = 3)
  (h7 : f.wedding_anniversary = 25) :
  birth_details f = (26, 8) := by sorry


end NUMINAMATH_CALUDE_birth_details_correct_l3252_325255


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3252_325221

/-- An arithmetic sequence with first term 2 and the property a_2 + a_4 = a_6 has common difference 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 2) 
  (h_sum_property : a 2 + a 4 = a 6) :
  ∀ n : ℕ, a (n + 1) - a n = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3252_325221


namespace NUMINAMATH_CALUDE_count_even_three_digit_numbers_less_than_700_l3252_325246

def valid_digits : List Nat := [1, 2, 3, 4, 5, 6]

def is_even (n : Nat) : Bool :=
  n % 2 = 0

def is_three_digit (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000

def count_valid_numbers : Nat :=
  (valid_digits.filter (· < 7)).length *
  valid_digits.length *
  (valid_digits.filter is_even).length

theorem count_even_three_digit_numbers_less_than_700 :
  count_valid_numbers = 108 := by
  sorry

end NUMINAMATH_CALUDE_count_even_three_digit_numbers_less_than_700_l3252_325246


namespace NUMINAMATH_CALUDE_grocery_value_proof_l3252_325250

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trips : ℕ := 40
def fixed_charge : ℝ := 1.5
def grocery_charge_rate : ℝ := 0.05

theorem grocery_value_proof (grocery_value : ℝ) : 
  car_cost - initial_savings = trips * fixed_charge + grocery_charge_rate * grocery_value →
  grocery_value = 800 := by
  sorry

end NUMINAMATH_CALUDE_grocery_value_proof_l3252_325250


namespace NUMINAMATH_CALUDE_g_constant_value_l3252_325282

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 5

-- Theorem statement
theorem g_constant_value (x : ℝ) : g (x + 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_constant_value_l3252_325282


namespace NUMINAMATH_CALUDE_probability_white_balls_l3252_325265

def total_balls : ℕ := 16
def white_balls : ℕ := 8
def black_balls : ℕ := 8
def drawn_balls : ℕ := 2

theorem probability_white_balls (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls = 16)
  (h3 : white_balls = 8)
  (h4 : black_balls = 8)
  (h5 : drawn_balls = 2) :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 7 / 30 ∧
  1 - (Nat.choose black_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 23 / 30 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_balls_l3252_325265


namespace NUMINAMATH_CALUDE_simplify_square_root_l3252_325299

theorem simplify_square_root (x y : ℝ) (h1 : x * y < 0) (h2 : -y / x^2 ≥ 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) :=
sorry

end NUMINAMATH_CALUDE_simplify_square_root_l3252_325299


namespace NUMINAMATH_CALUDE_bug_meeting_point_l3252_325212

/-- Triangle PQR with sides PQ, QR, PR, and point S where bugs meet -/
structure TrianglePQR where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  S : ℝ

/-- Theorem: In triangle PQR with given side lengths, if two bugs start at P
    and crawl in opposite directions at the same speed, meeting at S,
    then QS = 5 -/
theorem bug_meeting_point (t : TrianglePQR)
  (h1 : t.PQ = 7)
  (h2 : t.QR = 8)
  (h3 : t.PR = 9)
  (h4 : t.S ≤ t.QR) -- S is on QR
  (h5 : t.PQ + t.S = (t.PQ + t.QR + t.PR) / 2) -- Bugs meet halfway
  : t.S = 5 := by
  sorry


end NUMINAMATH_CALUDE_bug_meeting_point_l3252_325212


namespace NUMINAMATH_CALUDE_set_operations_l3252_325224

open Set

def A : Set ℝ := {x | 2 * x - 8 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 4}) ∧
  ((Aᶜ ∪ B) = {x : ℝ | 0 < x}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l3252_325224


namespace NUMINAMATH_CALUDE_smallest_m_plus_n_l3252_325241

/-- The function f(x) = arcsin(log_m(nx)) has a domain that is a closed interval of length 1/1007 --/
def domain_length (m n : ℕ) : ℚ :=
  (m^2 - 1 : ℚ) / (m * n)

/-- The theorem stating the smallest possible value of m + n --/
theorem smallest_m_plus_n :
  ∃ (m n : ℕ),
    m > 1 ∧
    domain_length m n = 1/1007 ∧
    ∀ (m' n' : ℕ), m' > 1 → domain_length m' n' = 1/1007 → m + n ≤ m' + n' ∧
    m + n = 19099 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_plus_n_l3252_325241


namespace NUMINAMATH_CALUDE_f_min_value_existence_l3252_325275

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 3 else (x - 3) * Real.exp x + Real.exp 2

theorem f_min_value_existence (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) →
  (∃ (a' : ℝ), 0 ≤ a' ∧ a' ≤ Real.sqrt 3 ∧
    ∀ (x : ℝ), f a' x ≥ f a' (2 : ℝ)) ∧
  (∀ (a' : ℝ), a' > Real.sqrt 3 →
    ¬∃ (m : ℝ), ∀ (x : ℝ), f a' x ≥ m) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_existence_l3252_325275


namespace NUMINAMATH_CALUDE_min_value_abc_l3252_325217

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 ∧ ∃ (a₀ b₀ c₀ : ℝ), 
    a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^4 * b₀^3 * c₀^2 = 1/10368 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l3252_325217


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3252_325210

/-- Given a line L1 with equation x - y + 2 = 0 and a point P (2, 1),
    the line L2 passing through P and parallel to L1 has equation x - y = 1 -/
theorem parallel_line_through_point (x y : ℝ) :
  (x - y + 2 = 0) →  -- L1: original line equation
  (∃ m b : ℝ, x - y = m * x + b) →  -- L2: general form of parallel line
  (2 - 1 = m * 2 + b) →  -- L2 passes through (2, 1)
  (x - y = 1) :=  -- L2: final equation
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3252_325210


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3252_325253

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

def B : Set ℝ := {x : ℝ | x ≥ 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3252_325253


namespace NUMINAMATH_CALUDE_complex_equation_implication_l3252_325244

theorem complex_equation_implication (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + b * i) * i = 1 + 2 * i →
  a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implication_l3252_325244


namespace NUMINAMATH_CALUDE_equation_solution_l3252_325269

theorem equation_solution : ∃ x : ℚ, (54 - x / 6 * 3 = 36) ∧ (x = 36) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3252_325269


namespace NUMINAMATH_CALUDE_product_constant_percentage_change_l3252_325204

theorem product_constant_percentage_change (x1 y1 x2 y2 : ℝ) :
  x1 * y1 = x2 * y2 ∧ 
  y2 = y1 * (1 - 44.44444444444444 / 100) →
  x2 = x1 * (1 + 80 / 100) :=
by sorry

end NUMINAMATH_CALUDE_product_constant_percentage_change_l3252_325204


namespace NUMINAMATH_CALUDE_intersection_equality_implies_m_values_l3252_325203

def A (m : ℝ) : Set ℝ := {3, m^2}
def B (m : ℝ) : Set ℝ := {-1, 3, 3*m-2}

theorem intersection_equality_implies_m_values (m : ℝ) :
  A m ∩ B m = A m → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_m_values_l3252_325203


namespace NUMINAMATH_CALUDE_average_reading_days_is_64_l3252_325237

/-- Represents the reading speed ratio between Emery and Serena for books -/
def book_speed_ratio : ℚ := 5

/-- Represents the reading speed ratio between Emery and Serena for articles -/
def article_speed_ratio : ℚ := 3

/-- Represents the number of days it takes Emery to read the book -/
def emery_book_days : ℕ := 20

/-- Represents the number of days it takes Emery to read the article -/
def emery_article_days : ℕ := 2

/-- Calculates the average number of days for Emery and Serena to read both the book and the article -/
def average_reading_days : ℚ := 
  let serena_book_days := emery_book_days * book_speed_ratio
  let serena_article_days := emery_article_days * article_speed_ratio
  let emery_total_days := emery_book_days + emery_article_days
  let serena_total_days := serena_book_days + serena_article_days
  (emery_total_days + serena_total_days) / 2

theorem average_reading_days_is_64 : average_reading_days = 64 := by
  sorry

end NUMINAMATH_CALUDE_average_reading_days_is_64_l3252_325237


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l3252_325272

def z : ℂ := (-8 + Complex.I) * Complex.I

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l3252_325272


namespace NUMINAMATH_CALUDE_sqrt_rational_sum_l3252_325228

theorem sqrt_rational_sum (a b r : ℚ) (h : Real.sqrt a + Real.sqrt b = r) :
  ∃ (c d : ℚ), Real.sqrt a = c ∧ Real.sqrt b = d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_rational_sum_l3252_325228


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3252_325245

/-- 
Given an infinite geometric series with first term a and common ratio r,
if the sum of the original series is 81 times the sum of the series
that results when the first four terms are removed, then r = 1/3.
-/
theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) →
  r = 1/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3252_325245


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3252_325206

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 2 = 8 →                                 -- Given condition
  a 5 = 64 →                                -- Given condition
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3252_325206


namespace NUMINAMATH_CALUDE_batsman_average_l3252_325219

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℝ) :
  total_innings = 12 →
  last_innings_score = 92 →
  average_increase = 2 →
  (((total_innings - 1 : ℝ) * ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) + last_innings_score) / total_innings) - 
  ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) = average_increase →
  (((total_innings - 1 : ℝ) * ((total_innings * (70 : ℝ) - last_innings_score : ℝ) / total_innings) + last_innings_score) / total_innings) = 70 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l3252_325219


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3252_325202

theorem arithmetic_calculations :
  ((-1) + (-6) - (-4) + 0 = -3) ∧
  (24 * (-1/4) / (-3/2) = 4) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3252_325202


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l3252_325296

theorem min_value_of_function (x : ℝ) (h : x > 0) : x^2 + 2/x ≥ 3 := by sorry

theorem min_value_achievable : ∃ x > 0, x^2 + 2/x = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l3252_325296


namespace NUMINAMATH_CALUDE_diagonal_exit_return_l3252_325208

/-- A path on a 10x10 grid that visits each cell exactly once -/
def HamiltonianPath : Type := Fin 100 → Fin 100

/-- A function that checks if two cells are adjacent -/
def isAdjacent (a b : Fin 100) : Prop := sorry

/-- A function that checks if a cell is on the main diagonal -/
def isOnDiagonal (a : Fin 100) : Prop := sorry

/-- The theorem stating that for any Hamiltonian path on a 10x10 grid, 
    there must be a point where the path leaves and immediately returns to the diagonal -/
theorem diagonal_exit_return (path : HamiltonianPath) : 
  ∃ (i : Fin 99), isOnDiagonal (path i) ∧ 
                  ¬isOnDiagonal (path (i + 1)) ∧ 
                  isOnDiagonal (path (i + 2)) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_exit_return_l3252_325208


namespace NUMINAMATH_CALUDE_expression_nonnegative_iff_l3252_325295

/-- The expression (x^2-4x+4)/(9-x^3) is nonnegative if and only if x ≤ 3 -/
theorem expression_nonnegative_iff (x : ℝ) : (x^2 - 4*x + 4) / (9 - x^3) ≥ 0 ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_iff_l3252_325295


namespace NUMINAMATH_CALUDE_susan_spending_l3252_325258

theorem susan_spending (initial_amount : ℝ) (h1 : initial_amount = 600) : 
  let after_clothes := initial_amount / 2
  let after_books := after_clothes / 2
  after_books = 150 := by
sorry

end NUMINAMATH_CALUDE_susan_spending_l3252_325258


namespace NUMINAMATH_CALUDE_outfit_combinations_l3252_325261

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) : shirts = 5 → pants = 3 → shirts * pants = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3252_325261


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3252_325276

/-- Given that x and y are inversely proportional, prove that y = -56.25 when x = -12,
    given that x = 3y when x + y = 60 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (∀ x' y', x' * y' = k) →  -- x and y are inversely proportional
  (∃ x₀ y₀, x₀ = 3 * y₀ ∧ x₀ + y₀ = 60) →  -- when their sum is 60, x is three times y
  (x = -12 → y = -56.25) :=  -- y = -56.25 when x = -12
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3252_325276


namespace NUMINAMATH_CALUDE_max_NPM_value_l3252_325229

theorem max_NPM_value : 
  ∀ M : ℕ, 
  1 ≤ M ∧ M ≤ 9 →
  let MM := 10 * M + M
  let NPM := MM * M
  100 ≤ NPM ∧ NPM < 1000 →
  (∀ N P : ℕ, NPM = 100 * N + 10 * P + M → N < 10 ∧ P < 10) →
  NPM ≤ 891 :=
by sorry

end NUMINAMATH_CALUDE_max_NPM_value_l3252_325229


namespace NUMINAMATH_CALUDE_streaming_service_subscriber_decrease_l3252_325232

/-- Proves the maximum percentage decrease in subscribers for a streaming service --/
theorem streaming_service_subscriber_decrease
  (initial_price : ℝ)
  (price_increase_percentage : ℝ)
  (h_initial_price : initial_price = 15)
  (h_price_increase : price_increase_percentage = 0.20) :
  let new_price := initial_price * (1 + price_increase_percentage)
  let max_decrease_percentage := 1 - (initial_price / new_price)
  ∃ (ε : ℝ), ε > 0 ∧ abs (max_decrease_percentage - (1/6)) < ε :=
by sorry

end NUMINAMATH_CALUDE_streaming_service_subscriber_decrease_l3252_325232


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3252_325266

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) : 
  let sphere_diameter := outer_cube_edge
  let inner_cube_edge := sphere_diameter / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3252_325266


namespace NUMINAMATH_CALUDE_min_weeks_to_sunday_rest_l3252_325234

/-- Represents the work schedule cycle in days -/
def work_cycle : ℕ := 10

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the initial offset in days (starting rest on Saturday) -/
def initial_offset : ℕ := 6

/-- 
Theorem: Given a work schedule of 8 days work followed by 2 days rest,
starting with rest on Saturday and Sunday, the minimum number of weeks
before resting on a Sunday again is 7.
-/
theorem min_weeks_to_sunday_rest : 
  ∃ (n : ℕ), n > 0 ∧ 
  (n * days_in_week + initial_offset) % work_cycle = work_cycle - 1 ∧
  ∀ (m : ℕ), m > 0 → m < n → 
  (m * days_in_week + initial_offset) % work_cycle ≠ work_cycle - 1 ∧
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_min_weeks_to_sunday_rest_l3252_325234


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l3252_325273

theorem largest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, (2^k : ℕ) = 128 ∧ (2^k : ℕ) ∣ (17^4 - 15^4) ∧
  ∀ m : ℕ, 2^m ∣ (17^4 - 15^4) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l3252_325273


namespace NUMINAMATH_CALUDE_trees_in_yard_l3252_325294

theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 434) (h2 : tree_distance = 14) :
  (yard_length / tree_distance) + 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l3252_325294


namespace NUMINAMATH_CALUDE_triangle_area_l3252_325225

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 2 →
  c = 2 * Real.sqrt 2 →
  C = π / 4 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3252_325225


namespace NUMINAMATH_CALUDE_heaviest_weight_proof_l3252_325209

/-- Represents a set of four weights in an increasing geometric progression -/
structure GeometricWeights (a q : ℝ) :=
  (a_pos : a > 0)
  (q_gt_one : q > 1)

/-- Proves that the heaviest weight is heavier than the sum of any two other weights -/
theorem heaviest_weight_proof {a q : ℝ} (gw : GeometricWeights a q) :
  a * q^3 > a + a * q ∧ 
  a * q^3 > a + a * q^2 ∧ 
  a * q^3 > a * q + a * q^2 :=
sorry

end NUMINAMATH_CALUDE_heaviest_weight_proof_l3252_325209


namespace NUMINAMATH_CALUDE_train_length_l3252_325211

/-- Given a train that crosses a platform in 39 seconds and a signal pole in 18 seconds,
    where the platform is 350 meters long, prove that the length of the train is 300 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 350) :
  let train_length := (platform_length * pole_crossing_time) / (platform_crossing_time - pole_crossing_time)
  train_length = 300 := by sorry

end NUMINAMATH_CALUDE_train_length_l3252_325211


namespace NUMINAMATH_CALUDE_power_equation_solution_l3252_325292

theorem power_equation_solution : ∃ y : ℕ, (2^10 + 2^10 + 2^10 + 2^10 : ℕ) = 4^y ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3252_325292


namespace NUMINAMATH_CALUDE_cricket_bat_price_l3252_325283

/-- Represents the cost and selling prices of an item -/
structure PriceData where
  cost_price_a : ℝ
  selling_price_b : ℝ
  selling_price_c : ℝ

/-- Theorem stating the relationship between the prices and profits -/
theorem cricket_bat_price (p : PriceData) 
  (profit_a : p.selling_price_b = 1.20 * p.cost_price_a)
  (profit_b : p.selling_price_c = 1.25 * p.selling_price_b)
  (final_price : p.selling_price_c = 222) :
  p.cost_price_a = 148 := by
  sorry

#check cricket_bat_price

end NUMINAMATH_CALUDE_cricket_bat_price_l3252_325283


namespace NUMINAMATH_CALUDE_arccos_cos_eq_x_div_3_l3252_325293

theorem arccos_cos_eq_x_div_3 (x : ℝ) :
  -Real.pi ≤ x ∧ x ≤ 2 * Real.pi →
  (Real.arccos (Real.cos x) = x / 3 ↔ x = 0 ∨ x = 3 * Real.pi / 2 ∨ x = -3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_x_div_3_l3252_325293


namespace NUMINAMATH_CALUDE_problem_solution_l3252_325220

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

theorem problem_solution :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k : ℝ, {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1} ⊆ A → k > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3252_325220


namespace NUMINAMATH_CALUDE_parabola_shift_l3252_325238

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the horizontal shift
def shift : ℝ := 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - shift) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l3252_325238


namespace NUMINAMATH_CALUDE_f_properties_l3252_325281

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - 0.5 * x^2

theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- Monotonicity property
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ ∧ f a x₁ < f a x₂) ∧
  (∃ (x₃ x₄ : ℝ), x₃ > 0 ∧ x₄ > 0 ∧ x₃ < x₄ ∧ f a x₃ > f a x₄) ∧
  -- Maximum value of b
  (∀ b : ℝ, (∀ x : ℝ, x > 0 → f a x ≥ -0.5 * x^2 + a * x + b) →
    b ≤ 0.5 * (1 + Real.log 2)) ∧
  (∃ b : ℝ, b = 0.5 * (1 + Real.log 2) ∧
    (∀ x : ℝ, x > 0 → f (0.5) x ≥ -0.5 * x^2 + 0.5 * x + b)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3252_325281


namespace NUMINAMATH_CALUDE_one_mile_equals_600_rods_l3252_325287

/-- Conversion factor from miles to furlongs -/
def mile_to_furlong : ℚ := 12

/-- Conversion factor from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

/-- Theorem stating that one mile is equal to 600 rods -/
theorem one_mile_equals_600_rods : rods_in_mile = 600 := by
  sorry

end NUMINAMATH_CALUDE_one_mile_equals_600_rods_l3252_325287


namespace NUMINAMATH_CALUDE_set_intersection_example_l3252_325260

theorem set_intersection_example : 
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {0, 2, 4}
  A ∩ B = {0, 2} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3252_325260


namespace NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l3252_325268

/-- Given a rectangle with length l and width w, and a triangle constructed on its diagonal
    such that the area of the triangle equals the area of the rectangle,
    the altitude of the triangle drawn to the diagonal is 2lw / √(l^2 + w^2). -/
theorem triangle_altitude_on_rectangle_diagonal 
  (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := (1 / 2) * diagonal * (2 * l * w / diagonal)
  triangle_area = rectangle_area →
  2 * l * w / diagonal = 2 * l * w / Real.sqrt (l^2 + w^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l3252_325268


namespace NUMINAMATH_CALUDE_fraction_problem_l3252_325223

theorem fraction_problem (n : ℝ) (h : n = 180) : ∃ f : ℝ, f * (1/3 * 1/5 * n) + 6 = 1/15 * n ∧ f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3252_325223


namespace NUMINAMATH_CALUDE_committee_selection_l3252_325248

theorem committee_selection (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 4 = 15) := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l3252_325248


namespace NUMINAMATH_CALUDE_josh_book_purchase_l3252_325298

/-- The number of books Josh bought -/
def num_books : ℕ := sorry

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each film in dollars -/
def film_cost : ℕ := 5

/-- The cost of each book in dollars -/
def book_cost : ℕ := 4

/-- The cost of each CD in dollars -/
def cd_cost : ℕ := 3

/-- The total amount Josh spent in dollars -/
def total_spent : ℕ := 79

theorem josh_book_purchase : 
  num_books * book_cost + num_films * film_cost + num_cds * cd_cost = total_spent ∧ 
  num_books = 4 := by sorry

end NUMINAMATH_CALUDE_josh_book_purchase_l3252_325298


namespace NUMINAMATH_CALUDE_parabola_tangent_inequality_l3252_325256

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = y

-- Define points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, -1)

-- Define the line AB
def line_AB (x y : ℝ) : Prop := y = 2*x - 1

-- Define the property of being tangent
def is_tangent (line : (ℝ → ℝ → Prop)) (curve : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), line x y ∧ curve x y ∧ 
  ∀ (x' y' : ℝ), x' ≠ x → y' ≠ y → line x' y' → ¬curve x' y'

-- State the theorem
theorem parabola_tangent_inequality :
  ∀ (P Q : ℝ × ℝ),
  parabola P.1 P.2 →
  parabola Q.1 Q.2 →
  line_AB P.1 P.2 →
  line_AB Q.1 Q.2 →
  is_tangent line_AB parabola →
  (P.1^2 + P.2^2).sqrt * (Q.1^2 + Q.2^2).sqrt > 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_inequality_l3252_325256


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3252_325279

/-- A quadratic function with coefficients a, b, and c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, quadratic a b c (x + 4.5) = quadratic a b c (4.5 - x)) →
  quadratic a b c (-9) = 1 →
  quadratic a b c 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3252_325279


namespace NUMINAMATH_CALUDE_min_xy_value_l3252_325262

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  ∃ (x₀ y₀ : ℕ+), (1 : ℚ) / x₀ + (1 : ℚ) / (3 * y₀) = (1 : ℚ) / 6 ∧
    x₀.val * y₀.val = 48 ∧
    ∀ (a b : ℕ+), (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 →
      x₀.val * y₀.val ≤ a.val * b.val :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3252_325262


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l3252_325259

theorem consecutive_numbers_sum (x : ℕ) :
  (∃ y : ℕ, 0 ≤ y ∧ y ≤ 9 ∧
    (List.sum (List.filter (λ i => i ≠ x + y) [x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9]) = 2002)) →
  x = 218 ∧ 
  List.filter (λ i => i ≠ 223) [x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9] = 
    [218, 219, 220, 221, 222, 224, 225, 226, 227] := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l3252_325259


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_l3252_325270

theorem sequence_is_arithmetic_progression 
  (a : ℕ → ℝ) 
  (h : ∀ m n : ℕ, |a m + a n - a (m + n)| ≤ 1 / (m + n : ℝ)) : 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_l3252_325270


namespace NUMINAMATH_CALUDE_circle_dot_product_bound_l3252_325214

theorem circle_dot_product_bound :
  ∀ (A : ℝ × ℝ),
  A.1^2 + (A.2 - 1)^2 = 1 →
  -2 ≤ (A.1 * 2 + A.2 * 0) ∧ (A.1 * 2 + A.2 * 0) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_dot_product_bound_l3252_325214


namespace NUMINAMATH_CALUDE_multiplication_fraction_equality_l3252_325243

theorem multiplication_fraction_equality : 12 * (1 / 8) * 32 = 48 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_equality_l3252_325243


namespace NUMINAMATH_CALUDE_min_m_for_solution_non_monotonic_range_l3252_325252

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x^2 - 1| + x

-- Part I
theorem min_m_for_solution (a : ℝ) (h : a = 2) :
  (∃ m : ℝ, ∀ x : ℝ, f a x - m ≤ 0) ↔ m ≥ -Real.sqrt 2 / 2 :=
sorry

-- Part II
theorem non_monotonic_range (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Icc (-3) 2 ∧ y ∈ Set.Icc (-3) 2 ∧ x < y ∧ f a x > f a y) ↔
  a < -1/6 ∨ a > 1/6 :=
sorry

end NUMINAMATH_CALUDE_min_m_for_solution_non_monotonic_range_l3252_325252


namespace NUMINAMATH_CALUDE_johnny_emily_meeting_distance_l3252_325242

-- Define the total distance
def total_distance : ℝ := 60

-- Define walking rates
def matthew_rate : ℝ := 3
def johnny_rate : ℝ := 4
def emily_rate : ℝ := 5

-- Define the time difference between Matthew's start and Johnny/Emily's start
def time_diff : ℝ := 1

-- Define the function to calculate the distance Johnny walked
def johnny_distance (t : ℝ) : ℝ := johnny_rate * t

-- Theorem statement
theorem johnny_emily_meeting_distance :
  ∃ t : ℝ, t > 0 ∧ 
    matthew_rate * (t + time_diff) + johnny_distance t + emily_rate * t = total_distance ∧
    johnny_distance t = 19 := by
  sorry

end NUMINAMATH_CALUDE_johnny_emily_meeting_distance_l3252_325242


namespace NUMINAMATH_CALUDE_china_population_in_scientific_notation_l3252_325226

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

/-- The population of China according to the sixth national census -/
def china_population : ℝ := 1370540000

/-- The scientific notation representation of China's population -/
def china_population_scientific : ℝ := 1.37054 * (10 : ℝ) ^ 9

theorem china_population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n china_population ∧
  china_population_scientific = a * (10 : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_china_population_in_scientific_notation_l3252_325226


namespace NUMINAMATH_CALUDE_C₁_C₂_intersections_l3252_325239

/-- The polar coordinate equation of curve C₁ -/
def C₁_polar (ρ θ : ℝ) : Prop := ρ - 2 * Real.cos θ = 0

/-- The rectangular coordinate equation of curve C₁ -/
def C₁_rect (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The equation of curve C₂ -/
def C₂ (x y m : ℝ) : Prop := 2*x - y - 2*m - 1 = 0

/-- The condition for C₁ and C₂ to have two distinct intersection points -/
def has_two_intersections (m : ℝ) : Prop :=
  (1 - Real.sqrt 5) / 2 < m ∧ m < (1 + Real.sqrt 5) / 2

theorem C₁_C₂_intersections (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    C₁_rect x₁ y₁ ∧ C₁_rect x₂ y₂ ∧
    C₂ x₁ y₁ m ∧ C₂ x₂ y₂ m) ↔
  has_two_intersections m :=
sorry

end NUMINAMATH_CALUDE_C₁_C₂_intersections_l3252_325239


namespace NUMINAMATH_CALUDE_additional_capacity_l3252_325263

/-- Represents the number of cars used by the swimming club -/
def num_cars : Nat := 2

/-- Represents the number of vans used by the swimming club -/
def num_vans : Nat := 3

/-- Represents the number of people in each car -/
def people_per_car : Nat := 5

/-- Represents the number of people in each van -/
def people_per_van : Nat := 3

/-- Represents the maximum capacity of each car -/
def max_car_capacity : Nat := 6

/-- Represents the maximum capacity of each van -/
def max_van_capacity : Nat := 8

/-- Theorem stating the number of additional people that could have ridden with the swim team -/
theorem additional_capacity : 
  (num_cars * max_car_capacity + num_vans * max_van_capacity) - 
  (num_cars * people_per_car + num_vans * people_per_van) = 17 := by
  sorry

end NUMINAMATH_CALUDE_additional_capacity_l3252_325263


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l3252_325213

/-- Given a quadratic function f(x) = x^2 + ax + b that intersects the x-axis at (1, 0) 
    and has its axis of symmetry at x = 2, prove that f(x) = x^2 - 4x + 3 -/
theorem quadratic_function_unique_form (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^2 + a*x + b)
    (h2 : f 1 = 0)
    (h3 : -a/2 = 2) : 
  ∀ x, f x = x^2 - 4*x + 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_unique_form_l3252_325213


namespace NUMINAMATH_CALUDE_compute_expression_l3252_325205

theorem compute_expression : 3 * 3^4 - 9^60 / 9^57 = -486 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l3252_325205


namespace NUMINAMATH_CALUDE_binomial_1493_1492_l3252_325230

theorem binomial_1493_1492 : Nat.choose 1493 1492 = 1493 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1493_1492_l3252_325230


namespace NUMINAMATH_CALUDE_root_difference_l3252_325285

theorem root_difference (r s : ℝ) : 
  (∃ x, (1984 * x)^2 - 1983 * 1985 * x - 1 = 0 ∧ r = x ∧ 
    ∀ y, ((1984 * y)^2 - 1983 * 1985 * y - 1 = 0 → y ≤ r)) →
  (∃ x, 1983 * x^2 - 1984 * x + 1 = 0 ∧ s = x ∧ 
    ∀ y, (1983 * y^2 - 1984 * y + 1 = 0 → s ≤ y)) →
  r - s = 1982 / 1983 := by
sorry

end NUMINAMATH_CALUDE_root_difference_l3252_325285


namespace NUMINAMATH_CALUDE_n_squared_minus_one_divisible_by_24_l3252_325231

theorem n_squared_minus_one_divisible_by_24 (n : ℤ) 
  (h1 : ¬ 2 ∣ n) (h2 : ¬ 3 ∣ n) : 24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_n_squared_minus_one_divisible_by_24_l3252_325231


namespace NUMINAMATH_CALUDE_wheel_probability_l3252_325247

theorem wheel_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_G = 1/8 → 
  p_D + p_E + p_F + p_G = 1 →
  p_F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l3252_325247


namespace NUMINAMATH_CALUDE_strawberry_milk_probability_l3252_325297

theorem strawberry_milk_probability :
  let n : ℕ := 6  -- Total number of days
  let k : ℕ := 5  -- Number of successful days
  let p : ℚ := 3/4  -- Probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 729/2048 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_probability_l3252_325297


namespace NUMINAMATH_CALUDE_parabola_segment_length_l3252_325201

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  hp : p < 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def onParabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Represents a line passing through the focus of the parabola at an angle with the x-axis -/
structure FocusLine where
  para : Parabola
  angle : ℝ

/-- Calculates the length of the segment AB formed by the intersection of the focus line with the parabola -/
noncomputable def segmentLength (para : Parabola) (fl : FocusLine) : ℝ :=
  sorry -- Actual calculation would go here

theorem parabola_segment_length 
  (para : Parabola) 
  (ptA : Point) 
  (fl : FocusLine) :
  onParabola para ptA → 
  ptA.x = -2 → 
  ptA.y = -4 → 
  fl.para = para → 
  fl.angle = π/3 → 
  segmentLength para fl = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_segment_length_l3252_325201


namespace NUMINAMATH_CALUDE_externally_tangent_circles_equation_l3252_325200

-- Define the radii and angle
variable (r r' φ : ℝ)

-- Define the conditions
variable (hr : r > 0)
variable (hr' : r' > 0)
variable (hφ : 0 < φ ∧ φ < π)

-- Define the externally tangent circles condition
variable (h_tangent : r + r' > 0)

-- Theorem statement
theorem externally_tangent_circles_equation :
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r') := by
  sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_equation_l3252_325200


namespace NUMINAMATH_CALUDE_cylinder_side_surface_diagonal_l3252_325278

/-- Given a cylinder with height 8 feet and base perimeter 6 feet,
    prove that the diagonal of the rectangular plate forming its side surface is 10 feet. -/
theorem cylinder_side_surface_diagonal (h : ℝ) (p : ℝ) (d : ℝ) :
  h = 8 →
  p = 6 →
  d = (h^2 + p^2)^(1/2) →
  d = 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_side_surface_diagonal_l3252_325278


namespace NUMINAMATH_CALUDE_brothers_ratio_l3252_325257

theorem brothers_ratio (aaron_brothers bennett_brothers : ℕ) 
  (h1 : aaron_brothers = 4) 
  (h2 : bennett_brothers = 6) : 
  (bennett_brothers : ℚ) / aaron_brothers = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ratio_l3252_325257


namespace NUMINAMATH_CALUDE_f_decreasing_l3252_325274

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin x else -Real.sin x

theorem f_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(Real.pi / 2) + 2 * k * Real.pi) ((Real.pi / 2) + 2 * k * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_l3252_325274


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3252_325222

theorem pasta_preference_ratio :
  ∀ (total students : ℕ) (pasta_types : ℕ) (spaghetti_pref manicotti_pref : ℕ),
    total = 800 →
    pasta_types = 5 →
    spaghetti_pref = 300 →
    manicotti_pref = 120 →
    (spaghetti_pref : ℚ) / manicotti_pref = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3252_325222


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3252_325249

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_relation : a 3 * a 9 = 2 * (a 5)^2)
  (h_a2 : a 2 = 2) :
  a 1 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3252_325249


namespace NUMINAMATH_CALUDE_number_with_remainder_36_mod_45_l3252_325289

theorem number_with_remainder_36_mod_45 (k : ℤ) :
  k % 45 = 36 → ∃ (n : ℕ), k = 45 * n + 36 :=
by sorry

end NUMINAMATH_CALUDE_number_with_remainder_36_mod_45_l3252_325289


namespace NUMINAMATH_CALUDE_train_length_l3252_325288

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 9 → ∃ length : ℝ, abs (length - 74.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3252_325288


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3252_325286

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 4 + a 7 = 45)
  (h_sum2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3252_325286


namespace NUMINAMATH_CALUDE_max_negative_integers_l3252_325216

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (w : ℕ), w ≤ 4 ∧
  ∀ (n : ℕ), (∃ (s : Finset (Fin 6)), s.card = n ∧
    (∀ i ∈ s, match i with
      | 0 => a < 0
      | 1 => b < 0
      | 2 => c < 0
      | 3 => d < 0
      | 4 => e < 0
      | 5 => f < 0
    )) → n ≤ w :=
by sorry

end NUMINAMATH_CALUDE_max_negative_integers_l3252_325216


namespace NUMINAMATH_CALUDE_mnp_nmp_difference_implies_mmp_nnp_difference_l3252_325277

/-- Represents a three-digit number in base 10 -/
def ThreeDigitNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem mnp_nmp_difference_implies_mmp_nnp_difference
  (m n p : ℕ)
  (h : ThreeDigitNumber m n p - ThreeDigitNumber n m p = 180) :
  ThreeDigitNumber m m p - ThreeDigitNumber n n p = 220 := by
  sorry

end NUMINAMATH_CALUDE_mnp_nmp_difference_implies_mmp_nnp_difference_l3252_325277


namespace NUMINAMATH_CALUDE_x_satisfies_recurrence_l3252_325284

/-- The number of n-digit numbers containing only digits 0, 1, 2, 
    where any two adjacent digits differ by at most 1. -/
def x (n : ℕ) : ℕ := sorry

/-- The property that x satisfies the recurrence relation for n ≥ 2 -/
def satisfies_recurrence (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → x (n + 1) = 2 * x n + x (n - 1)

/-- Theorem stating that our defined x satisfies the recurrence relation -/
theorem x_satisfies_recurrence : satisfies_recurrence x := by sorry

end NUMINAMATH_CALUDE_x_satisfies_recurrence_l3252_325284
