import Mathlib

namespace investment_sum_l925_92597

/-- Proves that if a sum P is invested at 15% p.a. for two years instead of 12% p.a. for two years, 
    and the difference in interest is Rs. 840, then P = Rs. 14,000. -/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 840) → P = 14000 := by
  sorry

end investment_sum_l925_92597


namespace toy_cost_price_l925_92519

/-- The cost price of one toy -/
def cost_price : ℕ := sorry

/-- The selling price of 18 toys -/
def selling_price : ℕ := 18900

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The number of toys whose cost price equals the gain -/
def gain_toys : ℕ := 3

theorem toy_cost_price : 
  (toys_sold + gain_toys) * cost_price = selling_price → 
  cost_price = 900 := by sorry

end toy_cost_price_l925_92519


namespace pizza_calories_l925_92555

theorem pizza_calories (total_slices : ℕ) (eaten_slices_1 : ℕ) (calories_1 : ℕ) 
  (eaten_slices_2 : ℕ) (calories_2 : ℕ) : 
  total_slices = 12 → 
  eaten_slices_1 = 3 →
  calories_1 = 300 →
  eaten_slices_2 = 4 →
  calories_2 = 400 →
  eaten_slices_1 * calories_1 + eaten_slices_2 * calories_2 = 2500 := by
  sorry

end pizza_calories_l925_92555


namespace max_value_expression_max_value_achievable_l925_92539

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (3 * x^2 + 2 - Real.sqrt (9 * x^4 + 4)) / x ≤ 12 / (5 + 3 * Real.sqrt 3) :=
sorry

theorem max_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (3 * x^2 + 2 - Real.sqrt (9 * x^4 + 4)) / x = 12 / (5 + 3 * Real.sqrt 3) :=
sorry

end max_value_expression_max_value_achievable_l925_92539


namespace claire_pets_l925_92584

theorem claire_pets (total_pets : ℕ) (total_males : ℕ) : 
  total_pets = 92 →
  total_males = 25 →
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (gerbils / 4 : ℚ) + (hamsters / 3 : ℚ) = total_males ∧
    gerbils = 68 := by
  sorry

end claire_pets_l925_92584


namespace land_development_break_even_l925_92575

/-- Calculates the break-even price per lot given the total acreage, price per acre, and number of lots. -/
def breakEvenPricePerLot (totalAcres : ℕ) (pricePerAcre : ℕ) (numberOfLots : ℕ) : ℕ :=
  (totalAcres * pricePerAcre) / numberOfLots

/-- Proves that for 4 acres at $1,863 per acre split into 9 lots, the break-even price is $828 per lot. -/
theorem land_development_break_even :
  breakEvenPricePerLot 4 1863 9 = 828 := by
  sorry

#eval breakEvenPricePerLot 4 1863 9

end land_development_break_even_l925_92575


namespace min_value_of_fraction_sum_l925_92592

theorem min_value_of_fraction_sum (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : (a - b) * (b - c) * (c - a) = -16) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ x y z : ℝ, 
    x > y → y > z → (x - y) * (y - z) * (z - x) = -16 → 
    1 / (x - y) + 1 / (y - z) - 1 / (z - x) ≥ m :=
by sorry

end min_value_of_fraction_sum_l925_92592


namespace x_equals_y_squared_plus_two_y_minus_one_l925_92501

theorem x_equals_y_squared_plus_two_y_minus_one (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y - 1) / (y^2 + 2*y - 2) → x = y^2 + 2*y - 1 := by
  sorry

end x_equals_y_squared_plus_two_y_minus_one_l925_92501


namespace emma_wrapping_time_l925_92540

/-- Represents the time (in hours) it takes for Emma to wrap presents individually -/
def emma_time : ℝ := 6

/-- Represents the time (in hours) it takes for Troy to wrap presents individually -/
def troy_time : ℝ := 8

/-- Represents the time (in hours) Emma and Troy work together -/
def together_time : ℝ := 2

/-- Represents the additional time (in hours) Emma works alone after Troy leaves -/
def emma_extra_time : ℝ := 2.5

theorem emma_wrapping_time :
  emma_time = 6 ∧
  (together_time * (1 / emma_time + 1 / troy_time) + emma_extra_time / emma_time = 1) :=
sorry

end emma_wrapping_time_l925_92540


namespace robie_chocolate_bags_l925_92515

/-- Calculates the final number of chocolate bags after transactions -/
def final_chocolate_bags (initial : ℕ) (given_away : ℕ) (additional : ℕ) : ℕ :=
  initial - given_away + additional

/-- Proves that Robie's final number of chocolate bags is 4 -/
theorem robie_chocolate_bags : 
  final_chocolate_bags 3 2 3 = 4 := by
  sorry

end robie_chocolate_bags_l925_92515


namespace molecular_weight_AlCl3_l925_92566

/-- The molecular weight of 4 moles of AlCl3 -/
theorem molecular_weight_AlCl3 (atomic_weight_Al atomic_weight_Cl : ℝ) 
  (h1 : atomic_weight_Al = 26.98)
  (h2 : atomic_weight_Cl = 35.45) : ℝ := by
  sorry

#check molecular_weight_AlCl3

end molecular_weight_AlCl3_l925_92566


namespace last_infected_on_fifth_exam_l925_92544

def total_mice : ℕ := 10
def infected_mice : ℕ := 3
def healthy_mice : ℕ := 7

-- The number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to arrange k items from n items
def arrange (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem last_infected_on_fifth_exam :
  choose infected_mice 2 * arrange 4 2 * choose healthy_mice 2 * arrange 2 2 = 1512 := by
  sorry

end last_infected_on_fifth_exam_l925_92544


namespace arithmetic_sequence_ratio_l925_92596

/-- Two arithmetic sequences and their sums -/
def arithmetic_sequences (a b : ℕ → ℝ) (A B : ℕ → ℝ) : Prop :=
  (∀ n, A n = (n * (a 1 + a n)) / 2) ∧
  (∀ n, B n = (n * (b 1 + b n)) / 2) ∧
  (∀ n, a (n + 1) - a n = a 2 - a 1) ∧
  (∀ n, b (n + 1) - b n = b 2 - b 1)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℝ) (A B : ℕ → ℝ) 
  (h : arithmetic_sequences a b A B) 
  (h_ratio : ∀ n : ℕ, A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, a n / b n = (4 * n - 3) / (6 * n - 2) := by
  sorry

end arithmetic_sequence_ratio_l925_92596


namespace meeting_time_calculation_l925_92560

/-- Two people moving towards each other -/
structure TwoPersonMovement where
  v₁ : ℝ  -- Speed of person 1
  v₂ : ℝ  -- Speed of person 2
  t₂ : ℝ  -- Waiting time after turning around

/-- The theorem statement -/
theorem meeting_time_calculation (m : TwoPersonMovement) 
  (h₁ : m.v₁ = 6)   -- Speed of person 1 is 6 m/s
  (h₂ : m.v₂ = 4)   -- Speed of person 2 is 4 m/s
  (h₃ : m.t₂ = 600) -- Waiting time is 10 minutes (600 seconds)
  : ∃ t₁ : ℝ, t₁ = 1200 ∧ (m.v₁ * t₁ + m.v₂ * t₁ = 2 * m.v₂ * t₁ + m.v₂ * m.t₂) := by
  sorry

end meeting_time_calculation_l925_92560


namespace f_max_value_l925_92535

def f (x : ℝ) := |x| - |x - 3|

theorem f_max_value :
  (∀ x, f x ≤ 3) ∧ (∃ x, f x = 3) := by sorry

end f_max_value_l925_92535


namespace oil_leak_total_l925_92547

theorem oil_leak_total (leaked_before_fixing leaked_while_fixing : ℕ) 
  (h1 : leaked_before_fixing = 2475)
  (h2 : leaked_while_fixing = 3731) :
  leaked_before_fixing + leaked_while_fixing = 6206 :=
by sorry

end oil_leak_total_l925_92547


namespace max_sum_at_15_l925_92579

/-- An arithmetic sequence with first term 29 and S_10 = S_20 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 29
  sum_equal : (Finset.range 10).sum a = (Finset.range 20).sum a

/-- Sum of the first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum seq.a

/-- The maximum value of S_n occurs when n = 15 -/
theorem max_sum_at_15 (seq : ArithmeticSequence) :
  ∀ n : ℕ, S seq n ≤ S seq 15 :=
sorry

end max_sum_at_15_l925_92579


namespace intersection_of_A_and_B_l925_92542

def set_A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {1, 3} := by sorry

end intersection_of_A_and_B_l925_92542


namespace intersection_point_l₁_l₂_l925_92500

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint (l₁ l₂ : ℝ → ℝ → Prop) where
  x : ℝ
  y : ℝ
  on_l₁ : l₁ x y
  on_l₂ : l₂ x y
  unique : ∀ x' y', l₁ x' y' → l₂ x' y' → x' = x ∧ y' = y

/-- Line l₁: 2x - y - 10 = 0 -/
def l₁ (x y : ℝ) : Prop := 2 * x - y - 10 = 0

/-- Line l₂: 3x + 4y - 4 = 0 -/
def l₂ (x y : ℝ) : Prop := 3 * x + 4 * y - 4 = 0

/-- The intersection point of l₁ and l₂ is (4, -2) -/
theorem intersection_point_l₁_l₂ : IntersectionPoint l₁ l₂ where
  x := 4
  y := -2
  on_l₁ := by sorry
  on_l₂ := by sorry
  unique := by sorry

end intersection_point_l₁_l₂_l925_92500


namespace function_symmetry_l925_92537

/-- Given a function f(x) = ax³ + bx + c*sin(x) - 2 where f(-2) = 8, prove that f(2) = -12 -/
theorem function_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + c * Real.sin x - 2
  f (-2) = 8 → f 2 = -12 := by
  sorry

end function_symmetry_l925_92537


namespace unique_solution_l925_92562

theorem unique_solution : ∃! x : ℝ, x > 12 ∧ (x - 6) / 12 = 5 / (x - 12) := by
  sorry

end unique_solution_l925_92562


namespace quartic_polynomial_unique_l925_92522

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℂ → ℂ := fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d

theorem quartic_polynomial_unique 
  (q : ℂ → ℂ) 
  (h_monic : q = fun x ↦ x^4 + (q 1 - 1)*x^3 + (q 2 - q 1 + 2)*x^2 + (q 3 - q 2 + q 1 - 3)*x + q 0)
  (h_real : ∀ x : ℝ, ∃ y : ℝ, q x = y)
  (h_root : q (2 + I) = 0)
  (h_value : q 0 = -120) :
  q = QuarticPolynomial 1 (-19) (-116) 120 := by
  sorry

end quartic_polynomial_unique_l925_92522


namespace min_value_parallel_lines_l925_92504

theorem min_value_parallel_lines (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  (∀ x y : ℝ, 2 * a + 3 * b ≥ 25) ∧ (∃ x y : ℝ, 2 * a + 3 * b = 25) := by
  sorry

end min_value_parallel_lines_l925_92504


namespace min_value_of_expression_l925_92585

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (2 : ℝ) * b - (b - 3) * a = 0) : 
  ∀ x y : ℝ, 2 * a + 3 * b ≥ 25 := by sorry

end min_value_of_expression_l925_92585


namespace root_product_theorem_l925_92548

-- Define the polynomial f(x) = x^6 + x^3 + 1
def f (x : ℂ) : ℂ := x^6 + x^3 + 1

-- Define the function g(x) = x^2 - 3
def g (x : ℂ) : ℂ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ) 
  (hroots : (X - x₁) * (X - x₂) * (X - x₃) * (X - x₄) * (X - x₅) * (X - x₆) = f X) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ * g x₆ = 757 := by
  sorry

end root_product_theorem_l925_92548


namespace distinct_sets_count_l925_92586

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {5, 6, 7}
def C : Finset ℕ := {8, 9}

def form_sets (X Y : Finset ℕ) : Finset (Finset ℕ) :=
  (X.product Y).image (λ (x, y) => {x, y})

theorem distinct_sets_count :
  (form_sets A B ∪ form_sets A C ∪ form_sets B C).card = 26 := by
  sorry

end distinct_sets_count_l925_92586


namespace f_composition_value_l925_92529

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^3
  else if 0 ≤ x ∧ x < Real.pi / 2 then -Real.sin x
  else 0  -- undefined for x ≥ π/2, but we need to cover all cases

theorem f_composition_value : f (f (Real.pi / 6)) = -1/4 := by
  sorry

end f_composition_value_l925_92529


namespace nh3_formation_l925_92533

-- Define the chemical reaction
structure Reaction where
  nh4no3 : ℕ
  naoh : ℕ
  nh3 : ℕ

-- Define the stoichiometric relationship
def stoichiometric (r : Reaction) : Prop :=
  r.nh4no3 = r.naoh ∧ r.nh3 = r.nh4no3

-- Theorem statement
theorem nh3_formation (r : Reaction) (h : stoichiometric r) :
  r.nh3 = r.nh4no3 := by
  sorry

end nh3_formation_l925_92533


namespace divisibility_property_l925_92536

theorem divisibility_property (q : ℕ) (h_prime : Nat.Prime q) (h_odd : Odd q) :
  ∃ k : ℤ, (q - 1 : ℤ) ^ (q - 2) + 1 = k * q :=
sorry

end divisibility_property_l925_92536


namespace range_of_a_for_subset_l925_92572

/-- The set A is defined as a circle centered at (2, 1) with radius 1 -/
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 ≤ 1}

/-- The set B is defined as a diamond shape centered at (1, 1) -/
def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2*|p.1 - 1| + |p.2 - 1| ≤ a}

/-- Theorem stating the range of a for which A is a subset of B -/
theorem range_of_a_for_subset (a : ℝ) : A ⊆ B a ↔ a ≥ 2 + Real.sqrt 5 := by sorry

end range_of_a_for_subset_l925_92572


namespace exists_nonperiodic_sequence_satisfying_property_l925_92552

/-- A sequence of natural numbers. -/
def Sequence := ℕ → ℕ

/-- A sequence satisfies the given property if for any k, there exists a t such that
    the sequence remains constant when we add multiples of t to k. -/
def SatisfiesProperty (a : Sequence) : Prop :=
  ∀ k, ∃ t, ∀ m, a k = a (k + m * t)

/-- A sequence is periodic if there exists a period T such that
    for all k, a(k) = a(k + T). -/
def IsPeriodic (a : Sequence) : Prop :=
  ∃ T, ∀ k, a k = a (k + T)

/-- There exists a sequence that satisfies the property but is not periodic. -/
theorem exists_nonperiodic_sequence_satisfying_property :
  ∃ a : Sequence, SatisfiesProperty a ∧ ¬IsPeriodic a := by
  sorry

end exists_nonperiodic_sequence_satisfying_property_l925_92552


namespace bridge_length_bridge_length_proof_l925_92545

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proves that the length of the bridge is 230 meters given the specified conditions. -/
theorem bridge_length_proof :
  bridge_length 145 45 30 = 230 := by
  sorry

end bridge_length_bridge_length_proof_l925_92545


namespace x_plus_y_equals_9_l925_92509

theorem x_plus_y_equals_9 (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := by
  sorry

end x_plus_y_equals_9_l925_92509


namespace symmetry_axes_intersection_l925_92528

-- Define a polygon as a set of points in 2D space
def Polygon := Set (ℝ × ℝ)

-- Define an axis of symmetry for a polygon
def IsAxisOfSymmetry (p : Polygon) (axis : Set (ℝ × ℝ)) : Prop := sorry

-- Define the center of mass for a set of points
def CenterOfMass (points : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define a property that a point lies on a line
def PointOnLine (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem symmetry_axes_intersection (p : Polygon) 
  (h_multiple_axes : ∃ (axis1 axis2 : Set (ℝ × ℝ)), axis1 ≠ axis2 ∧ IsAxisOfSymmetry p axis1 ∧ IsAxisOfSymmetry p axis2) :
  ∀ (axis : Set (ℝ × ℝ)), IsAxisOfSymmetry p axis → 
    PointOnLine (CenterOfMass p) axis :=
sorry

end symmetry_axes_intersection_l925_92528


namespace quadratic_b_value_l925_92595

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- Define the theorem
theorem quadratic_b_value :
  ∀ (b c y₁ y₂ : ℝ),
  (f b c 2 = y₁) →
  (f b c (-2) = y₂) →
  (y₁ - y₂ = 12) →
  b = 3 := by
sorry


end quadratic_b_value_l925_92595


namespace floor_neg_seven_fourths_l925_92558

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_neg_seven_fourths_l925_92558


namespace triangle_inequality_l925_92559

-- Define the points
variable (A B C P A₁ B₁ C₁ : ℝ × ℝ)

-- Define the equilateral triangle ABC
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define that P is inside triangle ABC
def is_inside_triangle (P A B C : ℝ × ℝ) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  P = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)

-- Define that A₁, B₁, C₁ are on the sides of triangle ABC
def on_side (X Y Z : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ Y = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)

-- Define the theorem
theorem triangle_inequality (A B C P A₁ B₁ C₁ : ℝ × ℝ) 
  (h1 : is_equilateral A B C)
  (h2 : is_inside_triangle P A B C)
  (h3 : on_side A₁ B C)
  (h4 : on_side B₁ C A)
  (h5 : on_side C₁ A B) :
  dist A₁ B₁ * dist B₁ C₁ * dist C₁ A₁ ≥ dist A₁ B * dist B₁ C * dist C₁ A :=
by sorry

end triangle_inequality_l925_92559


namespace goose_egg_count_l925_92523

-- Define the number of goose eggs laid at the pond
def total_eggs : ℕ := 2000

-- Define the fraction of eggs that hatched
def hatch_rate : ℚ := 2/3

-- Define the fraction of hatched geese that survived the first month
def first_month_survival_rate : ℚ := 3/4

-- Define the fraction of geese that survived the first month but did not survive the first year
def first_year_mortality_rate : ℚ := 3/5

-- Define the number of geese that survived the first year
def survived_first_year : ℕ := 100

-- Theorem statement
theorem goose_egg_count :
  (total_eggs : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = survived_first_year :=
sorry

end goose_egg_count_l925_92523


namespace sandwiches_left_for_others_l925_92538

def total_sandwiches : ℕ := 20
def sandwiches_for_coworker : ℕ := 4
def sandwiches_for_self : ℕ := 2 * sandwiches_for_coworker

theorem sandwiches_left_for_others : 
  total_sandwiches - sandwiches_for_coworker - sandwiches_for_self = 8 := by
  sorry

end sandwiches_left_for_others_l925_92538


namespace max_hot_dogs_proof_l925_92571

-- Define pack sizes and prices
structure PackInfo where
  size : Nat
  price : Rat

-- Define the problem parameters
def budget : Rat := 300
def packInfos : List PackInfo := [
  ⟨8, 155/100⟩,
  ⟨20, 305/100⟩,
  ⟨50, 745/100⟩,
  ⟨100, 1410/100⟩,
  ⟨250, 2295/100⟩
]
def discountThreshold : Nat := 10
def discountRate : Rat := 5/100
def maxPacksPerSize : Nat := 30
def minTotalPacks : Nat := 15

-- Define a function to calculate the total number of hot dogs
def totalHotDogs (purchases : List (PackInfo × Nat)) : Nat :=
  purchases.foldl (fun acc (pack, quantity) => acc + pack.size * quantity) 0

-- Define a function to calculate the total cost
def totalCost (purchases : List (PackInfo × Nat)) : Rat :=
  purchases.foldl (fun acc (pack, quantity) =>
    let basePrice := pack.price * quantity
    let discountedPrice := if quantity > discountThreshold then basePrice * (1 - discountRate) else basePrice
    acc + discountedPrice
  ) 0

-- Theorem statement
theorem max_hot_dogs_proof :
  ∃ (purchases : List (PackInfo × Nat)),
    totalHotDogs purchases = 3250 ∧
    totalCost purchases ≤ budget ∧
    purchases.all (fun (_, quantity) => quantity ≤ maxPacksPerSize) ∧
    purchases.foldl (fun acc (_, quantity) => acc + quantity) 0 ≥ minTotalPacks ∧
    (∀ (otherPurchases : List (PackInfo × Nat)),
      totalCost otherPurchases ≤ budget →
      purchases.all (fun (_, quantity) => quantity ≤ maxPacksPerSize) →
      purchases.foldl (fun acc (_, quantity) => acc + quantity) 0 ≥ minTotalPacks →
      totalHotDogs otherPurchases ≤ totalHotDogs purchases) :=
by
  sorry

end max_hot_dogs_proof_l925_92571


namespace speed_ratio_with_head_start_l925_92551

/-- The ratio of speeds in a race where one runner has a head start -/
theorem speed_ratio_with_head_start (vA vB : ℝ) (h : vA > 0 ∧ vB > 0) : 
  (120 / vA = 60 / vB) → vA / vB = 2 := by
  sorry

#check speed_ratio_with_head_start

end speed_ratio_with_head_start_l925_92551


namespace unknown_number_exists_l925_92520

theorem unknown_number_exists : ∃ x : ℝ, 
  (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + x + (0.06 : ℝ)^2 = 0.08999999999999998 ∧ 
  abs (x - 0.092625) < 0.000001 := by
  sorry

end unknown_number_exists_l925_92520


namespace cos_2013pi_l925_92527

theorem cos_2013pi : Real.cos (2013 * Real.pi) = -1 := by
  sorry

end cos_2013pi_l925_92527


namespace morning_pear_sales_l925_92563

/-- Represents the sale of pears by a salesman in a day. -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ
  total : ℝ

/-- Theorem stating the number of kilograms of pears sold in the morning. -/
theorem morning_pear_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning)
  (h2 : sales.total = 360)
  (h3 : sales.total = sales.morning + sales.afternoon) :
  sales.morning = 120 := by
  sorry

end morning_pear_sales_l925_92563


namespace cross_section_distance_theorem_l925_92506

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  -- Add any necessary fields here
  mk ::

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

/-- Theorem about the distance of cross-sections in a right hexagonal pyramid -/
theorem cross_section_distance_theorem 
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h : cs1.distance_from_apex < cs2.distance_from_apex)
  (area_h : cs1.area < cs2.area)
  (d : ℝ)
  (h_d : d = cs2.distance_from_apex - cs1.distance_from_apex) :
  cs2.distance_from_apex = d / (1 - Real.sqrt (cs1.area / cs2.area)) :=
by sorry

end cross_section_distance_theorem_l925_92506


namespace quadratic_roots_condition_l925_92587

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 - 2 * x + 1 = 0 ∧ (k - 1) * y^2 - 2 * y + 1 = 0) ↔
  (k ≤ 2 ∧ k ≠ 1) :=
by sorry

end quadratic_roots_condition_l925_92587


namespace cubic_root_ratio_l925_92550

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) → 
  c / d = 5 / 6 := by
sorry

end cubic_root_ratio_l925_92550


namespace malar_completion_time_l925_92582

/-- The number of days Malar takes to complete the task alone -/
def M : ℝ := 60

/-- The number of days Roja takes to complete the task alone -/
def R : ℝ := 84

/-- The number of days Malar and Roja take to complete the task together -/
def T : ℝ := 35

theorem malar_completion_time :
  (1 / M + 1 / R = 1 / T) → M = 60 := by
  sorry

end malar_completion_time_l925_92582


namespace v_domain_characterization_l925_92507

/-- The function v(x) = 1 / sqrt(x^2 - 4) -/
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x^2 - 4)

/-- The domain of v(x) -/
def domain_v : Set ℝ := {x | x < -2 ∨ x > 2}

theorem v_domain_characterization :
  ∀ x : ℝ, v x ∈ Set.univ ↔ x ∈ domain_v :=
by sorry

end v_domain_characterization_l925_92507


namespace student_rank_theorem_l925_92517

/-- Given a line of students, this function calculates a student's rank from the right
    based on their rank from the left and the total number of students. -/
def rankFromRight (totalStudents : ℕ) (rankFromLeft : ℕ) : ℕ :=
  totalStudents - rankFromLeft + 1

/-- Theorem stating that for a line of 10 students, 
    a student ranked 5th from the left is ranked 6th from the right. -/
theorem student_rank_theorem :
  rankFromRight 10 5 = 6 := by
  sorry

end student_rank_theorem_l925_92517


namespace radio_selling_price_l925_92589

/-- Calculates the selling price of a radio given the purchase price, overhead expenses, and profit percentage. -/
def calculate_selling_price (purchase_price : ℚ) (overhead : ℚ) (profit_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + overhead
  let profit := (profit_percent / 100) * total_cost
  total_cost + profit

/-- Theorem stating that the selling price of the radio is 300 given the specified conditions. -/
theorem radio_selling_price :
  calculate_selling_price 225 28 (18577075098814234 / 1000000000) = 300 := by
  sorry

end radio_selling_price_l925_92589


namespace number_count_proof_l925_92502

/-- Given a set of numbers with specific average properties, prove that the total count is 8 -/
theorem number_count_proof (n : ℕ) (S : ℝ) (S₅ : ℝ) (S₃ : ℝ) : 
  S / n = 20 →
  S₅ / 5 = 12 →
  S₃ / 3 = 33.333333333333336 →
  S = S₅ + S₃ →
  n = 8 := by
sorry

end number_count_proof_l925_92502


namespace math_paper_probability_l925_92594

theorem math_paper_probability (total_pages : ℕ) (math_pages : ℕ) (prob : ℚ) :
  total_pages = 12 →
  math_pages = 2 →
  prob = math_pages / total_pages →
  prob = 1 / 6 := by
  sorry

end math_paper_probability_l925_92594


namespace cube_sum_ge_mixed_product_l925_92590

theorem cube_sum_ge_mixed_product {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 := by
  sorry

end cube_sum_ge_mixed_product_l925_92590


namespace john_trees_chopped_l925_92554

/-- Represents the number of trees John chopped down -/
def num_trees : ℕ := 30

/-- Represents the number of planks that can be made from each tree -/
def planks_per_tree : ℕ := 25

/-- Represents the number of planks needed to make one table -/
def planks_per_table : ℕ := 15

/-- Represents the selling price of each table in dollars -/
def price_per_table : ℕ := 300

/-- Represents the total labor cost in dollars -/
def labor_cost : ℕ := 3000

/-- Represents the profit John made in dollars -/
def profit : ℕ := 12000

theorem john_trees_chopped :
  num_trees * planks_per_tree / planks_per_table * price_per_table - labor_cost = profit :=
sorry

end john_trees_chopped_l925_92554


namespace fraction_simplification_l925_92570

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a + b ≠ 0) :
  ((a + b)^2 * (a^3 - b^3)) / ((a^2 - b^2)^2) = (a^2 + a*b + b^2) / (a - b) ∧
  (6*a^2*b^2 - 3*a^3*b - 3*a*b^3) / (a*b^3 - a^3*b) = 3 * (a - b) / (a + b) :=
sorry

end fraction_simplification_l925_92570


namespace sum_three_consecutive_integers_divisible_by_three_l925_92518

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : a > 1) :
  ∃ k : ℤ, (a - 1 : ℤ) + a + (a + 1) = 3 * k :=
by sorry

end sum_three_consecutive_integers_divisible_by_three_l925_92518


namespace grinder_loss_percentage_l925_92549

theorem grinder_loss_percentage (grinder_cp mobile_cp total_profit mobile_profit_percent : ℝ)
  (h1 : grinder_cp = 15000)
  (h2 : mobile_cp = 10000)
  (h3 : total_profit = 400)
  (h4 : mobile_profit_percent = 10) :
  let mobile_sp := mobile_cp * (1 + mobile_profit_percent / 100)
  let total_sp := grinder_cp + mobile_cp + total_profit
  let grinder_sp := total_sp - mobile_sp
  let loss_amount := grinder_cp - grinder_sp
  loss_amount / grinder_cp * 100 = 4 := by sorry

end grinder_loss_percentage_l925_92549


namespace auditorium_seats_l925_92503

/-- Represents the number of seats in a row of an auditorium -/
def seats (x : ℕ) : ℕ := 2 * x + 18

theorem auditorium_seats :
  (seats 1 = 20) ∧
  (seats 19 = 56) ∧
  (seats 26 = 70) :=
by sorry

end auditorium_seats_l925_92503


namespace student_count_l925_92553

theorem student_count (W : ℝ) (N : ℕ) (h1 : N > 0) :
  W / N - 12 = (W - 72 + 12) / N → N = 5 := by
sorry

end student_count_l925_92553


namespace max_leap_years_in_period_l925_92569

/-- Represents the number of years in a period -/
def period : ℕ := 200

/-- Represents the frequency of leap years -/
def leap_year_frequency : ℕ := 5

/-- Calculates the maximum number of leap years in the given period -/
def max_leap_years : ℕ := period / leap_year_frequency

/-- Theorem stating that the maximum number of leap years in a 200-year period is 40,
    given that leap years occur every 5 years -/
theorem max_leap_years_in_period :
  max_leap_years = 40 :=
by sorry

end max_leap_years_in_period_l925_92569


namespace total_experienced_monthly_earnings_l925_92556

def total_sailors : ℕ := 30
def inexperienced_sailors : ℕ := 8
def group_a_sailors : ℕ := 12
def group_b_sailors : ℕ := 10
def inexperienced_hourly_wage : ℚ := 12
def group_a_wage_multiplier : ℚ := 4/3
def group_b_wage_multiplier : ℚ := 5/4
def group_a_weekly_hours : ℕ := 50
def group_b_weekly_hours : ℕ := 60
def weeks_per_month : ℕ := 4

def group_a_hourly_wage : ℚ := inexperienced_hourly_wage * group_a_wage_multiplier
def group_b_hourly_wage : ℚ := inexperienced_hourly_wage * group_b_wage_multiplier

def group_a_monthly_earnings : ℚ := group_a_hourly_wage * group_a_weekly_hours * weeks_per_month * group_a_sailors
def group_b_monthly_earnings : ℚ := group_b_hourly_wage * group_b_weekly_hours * weeks_per_month * group_b_sailors

theorem total_experienced_monthly_earnings :
  group_a_monthly_earnings + group_b_monthly_earnings = 74400 := by
  sorry

end total_experienced_monthly_earnings_l925_92556


namespace problem_1_l925_92561

theorem problem_1 (α : Real) (h : Real.sin α - 2 * Real.cos α = 0) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 := by
  sorry

end problem_1_l925_92561


namespace perception_permutations_l925_92598

def word_length : ℕ := 10
def repeated_letters : ℕ := 2

theorem perception_permutations :
  (word_length.factorial) / ((repeated_letters.factorial) * (repeated_letters.factorial)) = 907200 :=
by sorry

end perception_permutations_l925_92598


namespace quadratic_equation_condition_l925_92568

theorem quadratic_equation_condition (x : ℝ) :
  (x^2 + 2*x - 3 = 0 ↔ (x = -3 ∨ x = 1)) →
  (x = 1 → x^2 + 2*x - 3 = 0) ∧
  ¬(x^2 + 2*x - 3 = 0 → x = 1) :=
by sorry

end quadratic_equation_condition_l925_92568


namespace complex_equation_solution_l925_92521

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end complex_equation_solution_l925_92521


namespace monica_study_ratio_l925_92576

/-- Monica's study schedule problem -/
theorem monica_study_ratio : 
  ∀ (thursday_hours : ℝ),
  thursday_hours > 0 →
  2 + thursday_hours + (thursday_hours / 2) + (2 + thursday_hours + (thursday_hours / 2)) = 22 →
  thursday_hours / 2 = 3 / 1 := by
sorry

end monica_study_ratio_l925_92576


namespace jake_bitcoin_theorem_l925_92513

def jake_bitcoin_problem (initial_fortune : ℕ) (first_donation : ℕ) (final_amount : ℕ) : Prop :=
  let after_first_donation := initial_fortune - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  let final_donation := after_tripling - final_amount
  final_donation = 10

theorem jake_bitcoin_theorem :
  jake_bitcoin_problem 80 20 80 := by sorry

end jake_bitcoin_theorem_l925_92513


namespace intersection_of_A_and_B_l925_92593

def set_A : Set ℝ := {x | x^2 + 2*x = 0}
def set_B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {0} := by sorry

end intersection_of_A_and_B_l925_92593


namespace complete_square_sum_l925_92541

theorem complete_square_sum (x : ℝ) : 
  (x^2 - 10*x + 15 = 0) → 
  ∃ (d e : ℤ), ((x + d : ℝ)^2 = e) ∧ (d + e = 5) :=
by sorry

end complete_square_sum_l925_92541


namespace johns_weight_l925_92530

theorem johns_weight (john mark : ℝ) 
  (h1 : john + mark = 240)
  (h2 : john - mark = john / 3) : 
  john = 144 := by
sorry

end johns_weight_l925_92530


namespace tank_water_level_l925_92508

theorem tank_water_level (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) :
  tank_capacity = 72 →
  initial_fraction = 3 / 4 →
  added_water = 9 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7 / 8 := by
  sorry

end tank_water_level_l925_92508


namespace greatest_possible_k_l925_92565

theorem greatest_possible_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
by sorry

end greatest_possible_k_l925_92565


namespace scenario_contradiction_characteristics_l925_92588

/-- Represents a person's reaction to a statement --/
inductive Reaction
  | Cry
  | Laugh

/-- Represents a family member --/
inductive FamilyMember
  | Mother
  | Father

/-- Represents the characteristics of a contradiction --/
structure ContradictionCharacteristics where
  interpenetrating : Bool
  specific : Bool

/-- Given scenario where a child's "I love you" causes different reactions --/
def scenario : FamilyMember → Reaction
  | FamilyMember.Mother => Reaction.Cry
  | FamilyMember.Father => Reaction.Laugh

/-- Theorem stating that the contradiction in the scenario exhibits both 
    interpenetration of contradictory sides and specificity --/
theorem scenario_contradiction_characteristics :
  ∃ (c : ContradictionCharacteristics), 
    c.interpenetrating ∧ c.specific := by
  sorry

end scenario_contradiction_characteristics_l925_92588


namespace five_dice_not_same_l925_92514

theorem five_dice_not_same (n : ℕ) (h : n = 8) :
  (1 - (n : ℚ) / n^5) = 4095 / 4096 :=
sorry

end five_dice_not_same_l925_92514


namespace treasure_value_l925_92564

theorem treasure_value (fonzie_investment aunt_bee_investment lapis_investment lapis_share : ℚ)
  (h1 : fonzie_investment = 7000)
  (h2 : aunt_bee_investment = 8000)
  (h3 : lapis_investment = 9000)
  (h4 : lapis_share = 337500) :
  let total_investment := fonzie_investment + aunt_bee_investment + lapis_investment
  let lapis_proportion := lapis_investment / total_investment
  lapis_proportion * (lapis_share / lapis_proportion) = 1125000 := by
sorry

end treasure_value_l925_92564


namespace martha_black_butterflies_l925_92583

def butterfly_collection (total blue yellow black : ℕ) : Prop :=
  total = blue + yellow + black ∧ blue = 2 * yellow

theorem martha_black_butterflies :
  ∀ total blue yellow black : ℕ,
  butterfly_collection total blue yellow black →
  total = 19 →
  blue = 6 →
  black = 10 :=
by
  sorry

end martha_black_butterflies_l925_92583


namespace complement_of_A_in_U_l925_92512

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 6, 7} := by sorry

end complement_of_A_in_U_l925_92512


namespace range_of_linear_function_l925_92573

-- Define the function f on the closed interval [0, 1]
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

-- State the theorem
theorem range_of_linear_function
  (a b : ℝ)
  (h_a_neg : a < 0)
  : Set.range (fun x => f x a b) = Set.Icc (a + b) b := by
  sorry

end range_of_linear_function_l925_92573


namespace P_speed_is_8_l925_92534

/-- Represents the cycling speed of P in kmph -/
def P_speed : ℝ := 8

/-- J's walking speed in kmph -/
def J_speed : ℝ := 6

/-- Time (in hours) J walks before P starts -/
def time_before_P_starts : ℝ := 1.5

/-- Total time (in hours) from J's start to the point where J is 3 km behind P -/
def total_time : ℝ := 7.5

/-- Distance (in km) J is behind P at the end -/
def distance_behind : ℝ := 3

theorem P_speed_is_8 :
  P_speed = 8 :=
sorry

#check P_speed_is_8

end P_speed_is_8_l925_92534


namespace sqrt_sum_equals_2sqrt14_l925_92581

theorem sqrt_sum_equals_2sqrt14 :
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end sqrt_sum_equals_2sqrt14_l925_92581


namespace willy_stuffed_animals_l925_92511

def total_stuffed_animals (initial : ℕ) (mom_gift : ℕ) (dad_multiplier : ℕ) : ℕ :=
  let after_mom := initial + mom_gift
  after_mom + (dad_multiplier * after_mom)

theorem willy_stuffed_animals :
  total_stuffed_animals 10 2 3 = 48 := by
  sorry

end willy_stuffed_animals_l925_92511


namespace interesting_quartet_inequality_l925_92543

theorem interesting_quartet_inequality (p a b c : ℕ) : 
  Nat.Prime p → p % 2 = 1 →
  a ≠ b → b ≠ c → a ≠ c →
  (ab + 1) % p = 0 →
  (ac + 1) % p = 0 →
  (bc + 1) % p = 0 →
  (p : ℚ) + 2 ≤ (a + b + c : ℚ) / 3 := by
  sorry

end interesting_quartet_inequality_l925_92543


namespace number_operation_l925_92557

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 24) / 10 = 3 := by
  sorry

end number_operation_l925_92557


namespace six_eight_ten_pythagorean_l925_92578

/-- A function to check if three positive integers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ+) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that 6, 8, and 10 form a Pythagorean triple -/
theorem six_eight_ten_pythagorean :
  isPythagoreanTriple 6 8 10 := by
  sorry

#check six_eight_ten_pythagorean

end six_eight_ten_pythagorean_l925_92578


namespace F_properties_l925_92524

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * abs x

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x

noncomputable def F (x : ℝ) : ℝ :=
  if f x ≥ g x then g x else f x

theorem F_properties :
  (∃ (M : ℝ), M = 7 - 2 * Real.sqrt 7 ∧ ∀ (x : ℝ), F x ≤ M) ∧
  (¬ ∃ (m : ℝ), ∀ (x : ℝ), F x ≥ m) :=
sorry

end F_properties_l925_92524


namespace same_route_probability_l925_92574

theorem same_route_probability (num_routes : ℕ) (num_students : ℕ) : 
  num_routes = 3 → num_students = 2 → 
  (num_routes : ℝ) / (num_routes * num_routes : ℝ) = 1 / 3 := by
  sorry

end same_route_probability_l925_92574


namespace first_1500_even_integers_digit_count_l925_92505

/-- Count the number of digits in a positive integer -/
def countDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for all even numbers from 2 to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def n1500 : ℕ := 3000

theorem first_1500_even_integers_digit_count :
  sumDigitsEven n1500 = 5448 := by sorry

end first_1500_even_integers_digit_count_l925_92505


namespace distribute_five_among_three_l925_92531

/-- The number of ways to distribute n distinct objects among k distinct groups,
    with each group receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects among 3 distinct groups,
    with each group receiving at least one object, is 150 -/
theorem distribute_five_among_three :
  distribute 5 3 = 150 := by sorry

end distribute_five_among_three_l925_92531


namespace path_length_for_73_l925_92526

/-- The length of a path along squares constructed on subdivisions of a segment --/
def path_length (segment_length : ℝ) : ℝ :=
  3 * segment_length

theorem path_length_for_73 :
  path_length 73 = 219 := by
  sorry

end path_length_for_73_l925_92526


namespace equation_solution_l925_92580

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem equation_solution :
  ∃ (z : ℂ), (1 - i * z = -1 + i * z) ∧ (z = -i) :=
by sorry

end equation_solution_l925_92580


namespace planes_parallel_if_perpendicular_to_same_line_l925_92510

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l925_92510


namespace unique_solution_g_equals_g_inv_l925_92577

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
sorry

end unique_solution_g_equals_g_inv_l925_92577


namespace q_range_l925_92599

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  (∀ y : ℝ, (∃ x : ℝ, q x = y) → y ≥ 0) ∧ 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, q x = y) :=
sorry

end q_range_l925_92599


namespace denis_neighbors_l925_92532

-- Define the students
inductive Student : Type
| Anya : Student
| Borya : Student
| Vera : Student
| Gena : Student
| Denis : Student

-- Define the line as a function from position (1 to 5) to Student
def Line : Type := Fin 5 → Student

-- Define what it means for two students to be adjacent
def adjacent (s1 s2 : Student) (line : Line) : Prop :=
  ∃ i : Fin 4, (line i = s1 ∧ line (i.succ) = s2) ∨ (line i = s2 ∧ line (i.succ) = s1)

-- State the theorem
theorem denis_neighbors (line : Line) : 
  (line 0 = Student.Borya) →  -- Borya is at the beginning
  (adjacent Student.Vera Student.Anya line ∧ ¬adjacent Student.Vera Student.Gena line) →  -- Vera next to Anya but not Gena
  (¬adjacent Student.Anya Student.Borya line ∧ ¬adjacent Student.Anya Student.Gena line ∧ ¬adjacent Student.Borya Student.Gena line) →  -- Anya, Borya, Gena not adjacent
  (adjacent Student.Denis Student.Anya line ∧ adjacent Student.Denis Student.Gena line) :=  -- Denis is next to Anya and Gena
by sorry

end denis_neighbors_l925_92532


namespace min_quotient_three_digit_number_l925_92546

theorem min_quotient_three_digit_number : 
  ∀ a b c : ℕ, 
  1 ≤ a ∧ a ≤ 9 ∧ 
  1 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  (100 * a + 10 * b + c : ℚ) / (a + b + c) ≥ 10.5 :=
by sorry

end min_quotient_three_digit_number_l925_92546


namespace not_perfect_square_polynomial_l925_92591

theorem not_perfect_square_polynomial (n : ℕ) : ¬∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 = m^2 := by
  sorry

end not_perfect_square_polynomial_l925_92591


namespace multiplicative_inverse_480_mod_4799_l925_92525

theorem multiplicative_inverse_480_mod_4799 : 
  (∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 40 ∧ b = 399 ∧ c = 401) →
  (∃ (n : ℕ), n < 4799 ∧ (480 * n) % 4799 = 1) ∧
  (480 * 4789) % 4799 = 1 :=
by sorry

end multiplicative_inverse_480_mod_4799_l925_92525


namespace franklins_gathering_theorem_l925_92567

/-- Represents the number of handshakes in Franklin's gathering --/
def franklins_gathering_handshakes (num_couples : ℕ) : ℕ :=
  let num_men := num_couples
  let num_women := num_couples
  let handshakes_among_men := num_men * (num_men - 1 + num_women - 1) / 2
  let franklins_handshakes := num_women
  handshakes_among_men + franklins_handshakes

/-- Theorem stating that the number of handshakes in Franklin's gathering with 15 couples is 225 --/
theorem franklins_gathering_theorem :
  franklins_gathering_handshakes 15 = 225 := by
  sorry

#eval franklins_gathering_handshakes 15

end franklins_gathering_theorem_l925_92567


namespace x_value_l925_92516

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 15 ∧ x = 35 := by
  sorry

end x_value_l925_92516
