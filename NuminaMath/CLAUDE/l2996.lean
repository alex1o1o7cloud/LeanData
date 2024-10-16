import Mathlib

namespace NUMINAMATH_CALUDE_cubic_fraction_equals_four_l2996_299692

theorem cubic_fraction_equals_four (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + 2*c = 0) : (a^3 + b^3 - c^3) / (a*b*c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_four_l2996_299692


namespace NUMINAMATH_CALUDE_prove_first_ingot_weight_l2996_299676

theorem prove_first_ingot_weight (weights : Fin 11 → ℕ) 
  (h_distinct : Function.Injective weights)
  (h_range : ∀ i, weights i ∈ Finset.range 12 \ {0}) : 
  ∃ (a b c d e f : Fin 11), 
    weights a + weights b + weights c + weights d ≤ 11 ∧
    weights a + weights e + weights f ≤ 11 ∧
    weights a = 1 :=
sorry

end NUMINAMATH_CALUDE_prove_first_ingot_weight_l2996_299676


namespace NUMINAMATH_CALUDE_no_bounded_sequences_with_property_l2996_299619

theorem no_bounded_sequences_with_property :
  ¬ ∃ (a b : ℕ → ℝ),
    (∃ M : ℝ, ∀ n, |a n| ≤ M ∧ |b n| ≤ M) ∧
    (∀ n m : ℕ, m > n → |a m - a n| > 1 / Real.sqrt n ∨ |b m - b n| > 1 / Real.sqrt n) :=
sorry

end NUMINAMATH_CALUDE_no_bounded_sequences_with_property_l2996_299619


namespace NUMINAMATH_CALUDE_closest_multiple_l2996_299604

def target : ℕ := 2500
def divisor : ℕ := 18

-- Define a function to calculate the distance between two natural numbers
def distance (a b : ℕ) : ℕ := max a b - min a b

-- Define a function to check if a number is a multiple of another
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the theorem
theorem closest_multiple :
  ∀ n : ℕ, is_multiple n divisor →
    distance n target ≥ distance 2502 target :=
sorry

end NUMINAMATH_CALUDE_closest_multiple_l2996_299604


namespace NUMINAMATH_CALUDE_calculate_expression_l2996_299682

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2996_299682


namespace NUMINAMATH_CALUDE_unique_triple_l2996_299626

theorem unique_triple : ∃! (x y z : ℕ), 
  100 ≤ x ∧ x < y ∧ y < z ∧ z < 1000 ∧ 
  (y - x = z - y) ∧ 
  (y * y = x * (z + 1000)) ∧
  x = 160 ∧ y = 560 ∧ z = 960 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l2996_299626


namespace NUMINAMATH_CALUDE_class_size_l2996_299643

/-- Represents the number of students in a class with English and German courses -/
structure ClassEnrollment where
  total : ℕ
  english : ℕ
  german : ℕ
  both : ℕ
  onlyEnglish : ℕ

/-- Theorem stating the total number of students given the enrollment conditions -/
theorem class_size (c : ClassEnrollment)
  (h1 : c.both = 12)
  (h2 : c.german = 22)
  (h3 : c.onlyEnglish = 23)
  (h4 : c.total = c.english + c.german - c.both)
  (h5 : c.english = c.onlyEnglish + c.both) :
  c.total = 45 := by
  sorry

#check class_size

end NUMINAMATH_CALUDE_class_size_l2996_299643


namespace NUMINAMATH_CALUDE_maries_daily_rent_is_24_l2996_299654

/-- Represents Marie's bakery finances --/
structure BakeryFinances where
  cashRegisterCost : ℕ
  dailyBreadLoaves : ℕ
  breadPrice : ℕ
  dailyCakes : ℕ
  cakePrice : ℕ
  dailyElectricityCost : ℕ
  daysToPayCashRegister : ℕ

/-- Calculates the daily rent given the bakery finances --/
def calculateDailyRent (finances : BakeryFinances) : ℕ :=
  let dailyRevenue := finances.dailyBreadLoaves * finances.breadPrice + finances.dailyCakes * finances.cakePrice
  let dailyProfit := finances.cashRegisterCost / finances.daysToPayCashRegister
  dailyRevenue - dailyProfit - finances.dailyElectricityCost

/-- Theorem stating that Marie's daily rent is $24 --/
theorem maries_daily_rent_is_24 (finances : BakeryFinances)
    (h1 : finances.cashRegisterCost = 1040)
    (h2 : finances.dailyBreadLoaves = 40)
    (h3 : finances.breadPrice = 2)
    (h4 : finances.dailyCakes = 6)
    (h5 : finances.cakePrice = 12)
    (h6 : finances.dailyElectricityCost = 2)
    (h7 : finances.daysToPayCashRegister = 8) :
    calculateDailyRent finances = 24 := by
  sorry

end NUMINAMATH_CALUDE_maries_daily_rent_is_24_l2996_299654


namespace NUMINAMATH_CALUDE_retail_price_maximizes_profit_l2996_299694

/-- The profit function for a shopping mall selling items -/
def profit_function (p : ℝ) : ℝ := (8300 - 170*p - p^2)*(p - 20)

/-- The derivative of the profit function -/
def profit_derivative (p : ℝ) : ℝ := -3*p^2 - 300*p + 11700

theorem retail_price_maximizes_profit :
  ∃ (p : ℝ), p > 20 ∧ 
  ∀ (q : ℝ), q > 20 → profit_function p ≥ profit_function q ∧
  p = 30 := by
  sorry

#check retail_price_maximizes_profit

end NUMINAMATH_CALUDE_retail_price_maximizes_profit_l2996_299694


namespace NUMINAMATH_CALUDE_percentage_boys_soccer_l2996_299602

def total_students : ℕ := 420
def boys : ℕ := 312
def soccer_players : ℕ := 250
def girls_not_playing : ℕ := 53

theorem percentage_boys_soccer : 
  (boys - (total_students - soccer_players - girls_not_playing)) / soccer_players * 100 = 78 := by
  sorry

end NUMINAMATH_CALUDE_percentage_boys_soccer_l2996_299602


namespace NUMINAMATH_CALUDE_rectangle_burn_time_l2996_299665

/-- Represents a rectangle made of wooden toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  burnTime : Nat  -- Time for one toothpick to burn in seconds

/-- Calculates the time for the entire structure to burn -/
def burnTime (rect : ToothpickRectangle) : Nat :=
  let maxPath := rect.rows + rect.cols - 2  -- Longest path from corner to middle
  (maxPath * rect.burnTime) + (rect.burnTime / 2)

theorem rectangle_burn_time :
  let rect := ToothpickRectangle.mk 3 5 10
  burnTime rect = 65 := by
  sorry

#eval burnTime (ToothpickRectangle.mk 3 5 10)

end NUMINAMATH_CALUDE_rectangle_burn_time_l2996_299665


namespace NUMINAMATH_CALUDE_f_20_5_l2996_299649

/-- 
  f(n, m) represents the number of possible increasing arithmetic sequences 
  that can be formed by selecting m terms from the numbers 1, 2, 3, ..., n
-/
def f (n m : ℕ) : ℕ :=
  sorry

/-- Helper function to check if a sequence is valid -/
def is_valid_sequence (seq : List ℕ) (n : ℕ) : Prop :=
  sorry

theorem f_20_5 : f 20 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_f_20_5_l2996_299649


namespace NUMINAMATH_CALUDE_fishing_ratio_l2996_299637

/-- Given the conditions of the fishing problem, prove that the ratio of trout to bass is 1:4 -/
theorem fishing_ratio : 
  ∀ (trout bass bluegill : ℕ),
  bass = 32 →
  bluegill = 2 * bass →
  trout + bass + bluegill = 104 →
  trout.gcd bass = 8 →
  (trout / 8 : ℚ) = 1 ∧ (bass / 8 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fishing_ratio_l2996_299637


namespace NUMINAMATH_CALUDE_vertex_of_f_l2996_299664

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Theorem stating that the vertex of f is at (-1, 3)
theorem vertex_of_f :
  (∃ (a : ℝ), f a = 3 ∧ ∀ x, f x ≤ 3) ∧ f (-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_f_l2996_299664


namespace NUMINAMATH_CALUDE_scientific_notation_of_23766400_l2996_299683

theorem scientific_notation_of_23766400 :
  23766400 = 2.37664 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_23766400_l2996_299683


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2996_299613

def P : Set ℝ := {x | x^2 ≠ 1}

def M (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a : 
  ∀ a : ℝ, M a ⊆ P ↔ a ∈ ({1, -1, 0} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2996_299613


namespace NUMINAMATH_CALUDE_douglas_county_x_votes_l2996_299652

/-- The percentage of votes Douglas won in county X -/
def douglas_votes_x : ℝ := 74

/-- The ratio of voters in county X to county Y -/
def voter_ratio : ℝ := 2

/-- The percentage of total votes Douglas won in both counties -/
def douglas_total_percent : ℝ := 66

/-- The percentage of votes Douglas won in county Y -/
def douglas_votes_y : ℝ := 50.00000000000002

theorem douglas_county_x_votes :
  let total_votes := voter_ratio + 1
  let douglas_total_votes := douglas_total_percent / 100 * total_votes
  let douglas_y_votes := douglas_votes_y / 100
  douglas_votes_x / 100 * voter_ratio + douglas_y_votes = douglas_total_votes :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_x_votes_l2996_299652


namespace NUMINAMATH_CALUDE_number_difference_l2996_299698

theorem number_difference (x y : ℝ) : 
  x + y = 50 →
  3 * max x y - 5 * min x y = 10 →
  |x - y| = 15 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2996_299698


namespace NUMINAMATH_CALUDE_candy_bar_calculation_l2996_299617

theorem candy_bar_calculation :
  let f : ℕ := 12
  let b : ℕ := f + 6
  let j : ℕ := 10 * (f + b)
  (40 : ℚ) / 100 * (j ^ 2 : ℚ) = 36000 := by sorry

end NUMINAMATH_CALUDE_candy_bar_calculation_l2996_299617


namespace NUMINAMATH_CALUDE_arithmetic_sequence_part1_arithmetic_sequence_part2_l2996_299660

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  seq_def : ∀ n : ℕ, a n = a 1 + (n - 1) * d

/-- Part 1 of the problem -/
theorem arithmetic_sequence_part1 (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = -1) (h2 : seq.a 8 = 2) : 
  seq.a 1 = -5 ∧ seq.d = 1 := by
  sorry

/-- Part 2 of the problem -/
theorem arithmetic_sequence_part2 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 + seq.a 6 = 12) (h2 : seq.a 4 = 7) :
  seq.a 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_part1_arithmetic_sequence_part2_l2996_299660


namespace NUMINAMATH_CALUDE_average_age_increase_l2996_299679

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℚ) (teacher_age : ℕ) : 
  num_students = 25 →
  student_avg_age = 26 →
  teacher_age = 52 →
  (student_avg_age * num_students + teacher_age) / (num_students + 1) - student_avg_age = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l2996_299679


namespace NUMINAMATH_CALUDE_zoo_animals_legs_count_l2996_299632

theorem zoo_animals_legs_count : 
  ∀ (total_heads : ℕ) (rabbit_count : ℕ) (peacock_count : ℕ),
    total_heads = 60 →
    rabbit_count = 36 →
    peacock_count = total_heads - rabbit_count →
    4 * rabbit_count + 2 * peacock_count = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_legs_count_l2996_299632


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_l2996_299623

/-- The marginal function of f -/
def marginal (f : ℕ → ℝ) : ℕ → ℝ := fun x ↦ f (x + 1) - f x

/-- The revenue function -/
def R : ℕ → ℝ := fun x ↦ 300 * x - 2 * x^2

/-- The cost function -/
def C : ℕ → ℝ := fun x ↦ 50 * x + 300

/-- The profit function -/
def p : ℕ → ℝ := fun x ↦ R x - C x

/-- The marginal profit function -/
def Mp : ℕ → ℝ := marginal p

theorem profit_and_marginal_profit (x : ℕ) (h : 1 ≤ x ∧ x ≤ 100) :
  p x = -2 * x^2 + 250 * x - 300 ∧
  Mp x = 248 - 4 * x ∧
  (∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ p y = 7512 ∧ ∀ z : ℕ, 1 ≤ z ∧ z ≤ 100 → p z ≤ p y) ∧
  (Mp 1 = 244 ∧ ∀ z : ℕ, 1 < z ∧ z ≤ 100 → Mp z ≤ Mp 1) :=
by sorry

#check profit_and_marginal_profit

end NUMINAMATH_CALUDE_profit_and_marginal_profit_l2996_299623


namespace NUMINAMATH_CALUDE_triangle_folding_theorem_l2996_299624

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A folding method is a function that takes a triangle and produces a set of fold lines -/
def FoldingMethod := Triangle → Set (ℝ × ℝ → ℝ × ℝ)

/-- The result of applying a folding method to a triangle -/
structure FoldedObject where
  original : Triangle
  foldLines : Set (ℝ × ℝ → ℝ × ℝ)
  thickness : ℕ

/-- A folding method is valid if it produces a folded object with uniform thickness -/
def isValidFolding (method : FoldingMethod) : Prop :=
  ∀ t : Triangle, ∃ fo : FoldedObject, 
    fo.original = t ∧ 
    fo.foldLines = method t ∧ 
    fo.thickness = 2020

/-- The main theorem: there exists a valid folding method for any triangle -/
theorem triangle_folding_theorem : ∃ (method : FoldingMethod), isValidFolding method := by
  sorry

end NUMINAMATH_CALUDE_triangle_folding_theorem_l2996_299624


namespace NUMINAMATH_CALUDE_triangle_side_range_l2996_299685

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_side_range (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c)
  (h2 : t.b = Real.sqrt 3) :
  Real.sqrt 3 < 2 * t.a + t.c ∧ 2 * t.a + t.c ≤ 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_range_l2996_299685


namespace NUMINAMATH_CALUDE_abs_sqrt_mul_eq_three_l2996_299646

theorem abs_sqrt_mul_eq_three : |(-3 : ℤ)| + Real.sqrt 4 + (-2 : ℤ) * (1 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sqrt_mul_eq_three_l2996_299646


namespace NUMINAMATH_CALUDE_tablet_charge_time_proof_l2996_299673

/-- The time in minutes to fully charge a smartphone. -/
def smartphone_charge_time : ℕ := 26

/-- The time in minutes to fully charge a tablet. -/
def tablet_charge_time : ℕ := 53

/-- The total charging time for half a smartphone and a full tablet. -/
def total_charge_time : ℕ := 66

/-- Proves that the tablet charge time is correct given the conditions. -/
theorem tablet_charge_time_proof :
  smartphone_charge_time / 2 + tablet_charge_time = total_charge_time :=
by sorry

end NUMINAMATH_CALUDE_tablet_charge_time_proof_l2996_299673


namespace NUMINAMATH_CALUDE_square_area_difference_l2996_299644

theorem square_area_difference (small_side large_side : ℝ) 
  (h1 : small_side = 4)
  (h2 : large_side = 9)
  (h3 : small_side < large_side) : 
  large_side^2 - small_side^2 = 65 := by
sorry

end NUMINAMATH_CALUDE_square_area_difference_l2996_299644


namespace NUMINAMATH_CALUDE_smallest_c_for_g_range_contains_one_l2996_299661

/-- The function g(x) defined as x^2 - 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

/-- Theorem stating that 2 is the smallest value of c such that 1 is in the range of g(x) -/
theorem smallest_c_for_g_range_contains_one :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 1) ↔ c ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_g_range_contains_one_l2996_299661


namespace NUMINAMATH_CALUDE_expression_simplification_l2996_299615

theorem expression_simplification (x : ℝ) (hx : x ≠ 0) :
  (x - 2)^2 - x*(x - 1) + (x^3 - 4*x^2) / x^2 = -2*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2996_299615


namespace NUMINAMATH_CALUDE_max_dot_product_OM_OC_l2996_299693

/-- Given points in a 2D Cartesian coordinate system -/
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, -1)

/-- M is a moving point with x-coordinate between -2 and 2 -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | -2 ≤ p.1 ∧ p.1 ≤ 2}

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: The maximum value of OM · OC is 4 -/
theorem max_dot_product_OM_OC :
  ∃ (m : ℝ × ℝ), m ∈ M ∧ 
    ∀ (n : ℝ × ℝ), n ∈ M → 
      dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) ≥ 
      dot_product (n.1 - O.1, n.2 - O.2) (C.1 - O.1, C.2 - O.2) ∧
    dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) = 4 :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_OM_OC_l2996_299693


namespace NUMINAMATH_CALUDE_third_pedal_similar_l2996_299616

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Generates the pedal triangle of a point P with respect to a given triangle -/
def pedalTriangle (P : Point) (T : Triangle) : Triangle :=
  sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  sorry

/-- Theorem stating that the third pedal triangle is similar to the original triangle -/
theorem third_pedal_similar (P : Point) (H₀ : Triangle) :
  let H₁ := pedalTriangle P H₀
  let H₂ := pedalTriangle P H₁
  let H₃ := pedalTriangle P H₂
  areSimilar H₃ H₀ :=
by
  sorry

end NUMINAMATH_CALUDE_third_pedal_similar_l2996_299616


namespace NUMINAMATH_CALUDE_min_triangulation_l2996_299630

/-- A regular polygon with n sides, where n ≥ 5 -/
structure RegularPolygon where
  n : ℕ
  n_ge_5 : n ≥ 5

/-- A triangulation of a regular polygon -/
structure Triangulation (p : RegularPolygon) where
  num_triangles : ℕ
  is_valid : Bool  -- Represents the validity of the triangulation

/-- The number of acute triangles in a valid triangulation is at least n -/
def min_acute_triangles (p : RegularPolygon) (t : Triangulation p) : Prop :=
  t.is_valid → t.num_triangles ≥ p.n

/-- The number of obtuse triangles in a valid triangulation is at least n -/
def min_obtuse_triangles (p : RegularPolygon) (t : Triangulation p) : Prop :=
  t.is_valid → t.num_triangles ≥ p.n

/-- The main theorem: both acute and obtuse triangulations have a minimum of n triangles -/
theorem min_triangulation (p : RegularPolygon) :
  (∀ t : Triangulation p, min_acute_triangles p t) ∧
  (∀ t : Triangulation p, min_obtuse_triangles p t) :=
sorry

end NUMINAMATH_CALUDE_min_triangulation_l2996_299630


namespace NUMINAMATH_CALUDE_route_b_saves_six_hours_l2996_299687

-- Define the time for each route (one way)
def route_a_time : ℕ := 5
def route_b_time : ℕ := 2

-- Define the function to calculate round trip time
def round_trip_time (one_way_time : ℕ) : ℕ := 2 * one_way_time

-- Define the function to calculate time saved
def time_saved (longer_route : ℕ) (shorter_route : ℕ) : ℕ :=
  round_trip_time longer_route - round_trip_time shorter_route

-- Theorem statement
theorem route_b_saves_six_hours :
  time_saved route_a_time route_b_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_route_b_saves_six_hours_l2996_299687


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2996_299672

theorem least_number_with_remainder (n : ℕ) : n = 36731 ↔ 
  (∀ d ∈ ({16, 27, 34, 45, 144} : Set ℕ), n % d = 11) ∧ 
  (∀ m < n, ∃ d ∈ ({16, 27, 34, 45, 144} : Set ℕ), m % d ≠ 11) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2996_299672


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2996_299620

/-- Represents the number of employees in each job category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  midLevel : ℕ
  junior : ℕ

/-- Represents the number of sampled employees in each job category -/
structure SampleCount where
  senior : ℕ
  midLevel : ℕ
  junior : ℕ

/-- Checks if the sample counts are correct for stratified sampling -/
def isCorrectStratifiedSample (ec : EmployeeCount) (sc : SampleCount) (sampleSize : ℕ) : Prop :=
  sc.senior = ec.senior * sampleSize / ec.total ∧
  sc.midLevel = ec.midLevel * sampleSize / ec.total ∧
  sc.junior = ec.junior * sampleSize / ec.total

theorem correct_stratified_sample :
  let ec : EmployeeCount := ⟨450, 45, 135, 270⟩
  let sc : SampleCount := ⟨3, 9, 18⟩
  isCorrectStratifiedSample ec sc 30 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l2996_299620


namespace NUMINAMATH_CALUDE_right_triangle_leg_lengths_l2996_299678

theorem right_triangle_leg_lengths 
  (c : ℝ) 
  (α β : ℝ) 
  (h_right : α + β = π / 2) 
  (h_tan : 6 * Real.tan β = Real.tan α + 1) :
  ∃ (a b : ℝ), 
    a^2 + b^2 = c^2 ∧ 
    a = (2 * c * Real.sqrt 5) / 5 ∧ 
    b = (c * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_lengths_l2996_299678


namespace NUMINAMATH_CALUDE_complex_real_condition_l2996_299621

theorem complex_real_condition (z m : ℂ) : z = (1 + Complex.I) * (1 + m * Complex.I) ∧ z.im = 0 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2996_299621


namespace NUMINAMATH_CALUDE_max_correct_answers_l2996_299675

theorem max_correct_answers (total_questions : Nat) (correct_score : Int) (incorrect_score : Int)
  (john_score : Int) (min_attempted : Nat) :
  total_questions = 25 →
  correct_score = 4 →
  incorrect_score = -3 →
  john_score = 52 →
  min_attempted = 20 →
  ∃ (correct incorrect unanswered : Nat),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_score + incorrect * incorrect_score = john_score ∧
    correct + incorrect ≥ min_attempted ∧
    correct ≤ 17 ∧
    ∀ (c : Nat), c > 17 →
      ¬(∃ (i u : Nat), c + i + u = total_questions ∧
        c * correct_score + i * incorrect_score = john_score ∧
        c + i ≥ min_attempted) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2996_299675


namespace NUMINAMATH_CALUDE_solve_for_y_l2996_299631

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = 5) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2996_299631


namespace NUMINAMATH_CALUDE_marinara_stain_soaking_time_l2996_299659

theorem marinara_stain_soaking_time 
  (grass_stain_time : ℕ) 
  (num_grass_stains : ℕ) 
  (num_marinara_stains : ℕ) 
  (total_time : ℕ) 
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_time = 19) :
  total_time - (grass_stain_time * num_grass_stains) = 7 := by
  sorry

#check marinara_stain_soaking_time

end NUMINAMATH_CALUDE_marinara_stain_soaking_time_l2996_299659


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l2996_299657

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l2996_299657


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_m_range_l2996_299647

theorem point_in_third_quadrant_m_range (m : ℝ) : 
  (m - 4 < 0 ∧ 1 - 2*m < 0) → (1/2 < m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_m_range_l2996_299647


namespace NUMINAMATH_CALUDE_squirrel_acorns_l2996_299645

theorem squirrel_acorns :
  let num_squirrels : ℕ := 5
  let acorns_needed : ℕ := 130
  let acorns_to_collect : ℕ := 15
  let acorns_per_squirrel : ℕ := acorns_needed - acorns_to_collect
  num_squirrels * acorns_per_squirrel = 575 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l2996_299645


namespace NUMINAMATH_CALUDE_number_division_problem_l2996_299691

theorem number_division_problem (x : ℝ) : x / 3 = 50 + x / 4 ↔ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2996_299691


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2996_299686

theorem purely_imaginary_complex_number (a : ℝ) : 
  (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I).re = 0 ∧ (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I).im ≠ 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2996_299686


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l2996_299612

/-- Represents a cyclic quadrilateral ABCD with angles α, β, γ, and ω -/
structure CyclicQuadrilateral where
  α : ℝ
  β : ℝ
  γ : ℝ
  ω : ℝ
  sum_180 : α + β + γ + ω = 180

/-- Theorem: In a cyclic quadrilateral ABCD, if α = c, β = 43°, γ = 59°, and ω = d, then d = 42° -/
theorem cyclic_quadrilateral_angle (q : CyclicQuadrilateral) (h1 : q.α = 36) (h2 : q.β = 43) (h3 : q.γ = 59) : q.ω = 42 := by
  sorry

#check cyclic_quadrilateral_angle

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l2996_299612


namespace NUMINAMATH_CALUDE_at_least_one_is_one_l2996_299648

theorem at_least_one_is_one (a b c : ℝ) 
  (sum_eq : a + b + c = 1/a + 1/b + 1/c) 
  (product_eq : a * b * c = 1) : 
  a = 1 ∨ b = 1 ∨ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_is_one_l2996_299648


namespace NUMINAMATH_CALUDE_silverware_probability_l2996_299607

theorem silverware_probability (forks spoons knives : ℕ) 
  (h1 : forks = 4) (h2 : spoons = 8) (h3 : knives = 6) : 
  let total := forks + spoons + knives
  let ways_to_choose_3 := Nat.choose total 3
  let ways_to_choose_2_spoons := Nat.choose spoons 2
  let ways_to_choose_1_knife := Nat.choose knives 1
  let favorable_outcomes := ways_to_choose_2_spoons * ways_to_choose_1_knife
  (favorable_outcomes : ℚ) / ways_to_choose_3 = 7 / 34 := by
sorry

end NUMINAMATH_CALUDE_silverware_probability_l2996_299607


namespace NUMINAMATH_CALUDE_probability_of_nine_in_three_elevenths_l2996_299666

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry -- Implementation of decimal representation calculation

def count_digit (digit : ℕ) (representation : List ℕ) : ℕ :=
  sorry -- Implementation of digit counting in the representation

theorem probability_of_nine_in_three_elevenths :
  let representation := decimal_representation 3 11
  let count_nines := count_digit 9 representation
  let total_digits := representation.length
  count_nines = 0 ∧ total_digits > 0 → count_nines / total_digits = 0 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_nine_in_three_elevenths_l2996_299666


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2996_299677

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, -1/2 < x ∧ x ≤ 1 ∧ 2^x - a > Real.arccos x) ↔ 
  a < Real.sqrt 2 / 2 - 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2996_299677


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2996_299640

def f (x : ℝ) := x^3 - 3*x^2 - 9*x + 2

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 2 ∧ min = -25 ∧
  (∀ x ∈ Set.Icc 0 4, f x ≤ max ∧ f x ≥ min) ∧
  (∃ x₁ ∈ Set.Icc 0 4, f x₁ = max) ∧
  (∃ x₂ ∈ Set.Icc 0 4, f x₂ = min) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2996_299640


namespace NUMINAMATH_CALUDE_function_properties_l2996_299625

-- Define the function f(x) = -x^2 + mx - m
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - m

theorem function_properties (m : ℝ) :
  -- 1. If the maximum value of f(x) is 0, then m = 0 or m = 4
  (∃ (max : ℝ), (∀ (x : ℝ), f m x ≤ max) ∧ (max = 0)) →
  (m = 0 ∨ m = 4) ∧

  -- 2. If f(x) is monotonically decreasing on [-1, 0], then m ≤ -2
  (∀ (x y : ℝ), -1 ≤ x ∧ x < y ∧ y ≤ 0 → f m x > f m y) →
  (m ≤ -2) ∧

  -- 3. The range of f(x) on [2, 3] is exactly [2, 3] if and only if m = 6
  (∀ (y : ℝ), 2 ≤ y ∧ y ≤ 3 ↔ ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 3 ∧ f m x = y) ↔
  (m = 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2996_299625


namespace NUMINAMATH_CALUDE_probability_red_ball_two_fifths_l2996_299636

/-- Represents a bag of colored balls -/
structure BallBag where
  red : ℕ
  black : ℕ

/-- Calculates the probability of drawing a red ball from the bag -/
def probabilityRedBall (bag : BallBag) : ℚ :=
  bag.red / (bag.red + bag.black)

/-- Theorem: The probability of drawing a red ball from a bag with 2 red balls and 3 black balls is 2/5 -/
theorem probability_red_ball_two_fifths :
  let bag : BallBag := { red := 2, black := 3 }
  probabilityRedBall bag = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_two_fifths_l2996_299636


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2996_299680

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2996_299680


namespace NUMINAMATH_CALUDE_books_total_is_140_l2996_299668

/-- The number of books Beatrix has -/
def beatrix_books : ℕ := 30

/-- The number of books Alannah has -/
def alannah_books : ℕ := beatrix_books + 20

/-- The number of books Queen has -/
def queen_books : ℕ := alannah_books + alannah_books / 5

/-- The total number of books all three have together -/
def total_books : ℕ := beatrix_books + alannah_books + queen_books

theorem books_total_is_140 : total_books = 140 := by
  sorry

end NUMINAMATH_CALUDE_books_total_is_140_l2996_299668


namespace NUMINAMATH_CALUDE_function_lower_bound_l2996_299609

theorem function_lower_bound (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, |x - 1/a| + |x + a| ≥ 2 := by sorry

end NUMINAMATH_CALUDE_function_lower_bound_l2996_299609


namespace NUMINAMATH_CALUDE_cistern_fill_time_l2996_299628

/-- The time it takes to fill the cistern with both taps open -/
def both_taps_time : ℚ := 28 / 3

/-- The time it takes to empty the cistern with the second tap -/
def empty_time : ℚ := 7

/-- The time it takes to fill the cistern with the first tap -/
def fill_time : ℚ := 4

theorem cistern_fill_time :
  (1 / fill_time - 1 / empty_time) = 1 / both_taps_time :=
by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l2996_299628


namespace NUMINAMATH_CALUDE_will_remaining_money_l2996_299667

/-- Calculates the remaining money after shopping and refund --/
def remaining_money (initial_amount sweater_cost tshirt_cost shoes_cost refund_percentage : ℚ) : ℚ :=
  initial_amount - sweater_cost - tshirt_cost + (shoes_cost * refund_percentage / 100)

/-- Proves that Will has $81 left after shopping and refund --/
theorem will_remaining_money :
  remaining_money 74 9 11 30 90 = 81 := by
  sorry

end NUMINAMATH_CALUDE_will_remaining_money_l2996_299667


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2996_299614

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) (h : a ≠ 0) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property
  (a b c : ℝ) (h : a ≠ 0)
  (x₁ x₂ : ℝ) (hx : x₁ ≠ x₂)
  (hf : QuadraticFunction a b c h x₁ = QuadraticFunction a b c h x₂) :
  QuadraticFunction a b c h (x₁ + x₂) = c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2996_299614


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l2996_299699

theorem unique_solution_implies_a_equals_two (a : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l2996_299699


namespace NUMINAMATH_CALUDE_jane_crayons_l2996_299600

theorem jane_crayons (initial_crayons : ℕ) (eaten_crayons : ℕ) : 
  initial_crayons = 87 → eaten_crayons = 7 → initial_crayons - eaten_crayons = 80 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayons_l2996_299600


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2996_299696

theorem inequality_equivalence (x : ℝ) : 
  |((x^2 + 2*x - 3) / 4)| ≤ 3 ↔ -5 ≤ x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2996_299696


namespace NUMINAMATH_CALUDE_egg_price_calculation_l2996_299669

theorem egg_price_calculation (num_eggs : ℕ) (num_chickens : ℕ) (price_per_chicken : ℚ) (total_spent : ℚ) :
  num_eggs = 20 →
  num_chickens = 6 →
  price_per_chicken = 8 →
  total_spent = 88 →
  (total_spent - (num_chickens * price_per_chicken)) / num_eggs = 2 :=
by sorry

end NUMINAMATH_CALUDE_egg_price_calculation_l2996_299669


namespace NUMINAMATH_CALUDE_monkey_banana_distribution_l2996_299610

/-- Represents the number of bananas each monkey receives when dividing the total equally -/
def bananas_per_monkey (num_monkeys : ℕ) (num_piles_type1 num_piles_type2 : ℕ) 
  (hands_per_pile_type1 hands_per_pile_type2 : ℕ) 
  (bananas_per_hand_type1 bananas_per_hand_type2 : ℕ) : ℕ :=
  let total_bananas := 
    num_piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
    num_piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2
  total_bananas / num_monkeys

/-- Theorem stating that given the problem conditions, each monkey receives 99 bananas -/
theorem monkey_banana_distribution :
  bananas_per_monkey 12 6 4 9 12 14 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_monkey_banana_distribution_l2996_299610


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l2996_299629

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (9 / a + 1 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 9 / a₀ + 1 / b₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l2996_299629


namespace NUMINAMATH_CALUDE_nth_equation_l2996_299688

theorem nth_equation (n : ℕ) : 
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1) : ℚ) = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_l2996_299688


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2996_299689

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47,
    prove that the 8th term is 71. -/
theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℤ)  -- The arithmetic sequence
  (h1 : a 4 = 23)  -- The 4th term is 23
  (h2 : a 6 = 47)  -- The 6th term is 47
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- The sequence is arithmetic
  : a 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2996_299689


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2996_299642

def M : Set ℝ := {x | x^2 = 2}
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_values (a : ℝ) :
  N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2996_299642


namespace NUMINAMATH_CALUDE_two_possible_w_values_l2996_299695

theorem two_possible_w_values 
  (w : ℂ) 
  (h_exists : ∃ (u v : ℂ), u ≠ v ∧ ∀ (z : ℂ), (z - u) * (z - v) = (z - w * u) * (z - w * v)) : 
  w = 1 ∨ w = -1 :=
sorry

end NUMINAMATH_CALUDE_two_possible_w_values_l2996_299695


namespace NUMINAMATH_CALUDE_probability_two_successes_four_trials_l2996_299638

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of exactly k successes in n trials with probability p of success per trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The theorem stating the probability of 2 successes in 4 trials with 0.3 probability of success -/
theorem probability_two_successes_four_trials :
  binomialProbability 4 2 0.3 = 0.2646 := by sorry

end NUMINAMATH_CALUDE_probability_two_successes_four_trials_l2996_299638


namespace NUMINAMATH_CALUDE_number_properties_l2996_299608

theorem number_properties :
  (∃! x : ℤ, ¬(x > 0) ∧ ¬(x < 0) ∧ x = 0) ∧
  (∃ x : ℤ, x < 0 ∧ ∀ y : ℤ, y < 0 → y ≤ x ∧ x = -1) ∧
  (∃ x : ℤ, x > 0 ∧ ∀ y : ℤ, y > 0 → x ≤ y ∧ x = 1) ∧
  (∃! x : ℤ, ∀ y : ℤ, |x| ≤ |y| ∧ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_number_properties_l2996_299608


namespace NUMINAMATH_CALUDE_house_of_cards_height_l2996_299627

/-- Given a triangular-shaped house of cards with base 40 cm,
    prove that if the total area of three similar houses is 1200 cm²,
    then the height of each house is 20 cm. -/
theorem house_of_cards_height
  (base : ℝ)
  (total_area : ℝ)
  (num_houses : ℕ)
  (h_base : base = 40)
  (h_total_area : total_area = 1200)
  (h_num_houses : num_houses = 3) :
  let area := total_area / num_houses
  let height := 2 * area / base
  height = 20 := by
sorry

end NUMINAMATH_CALUDE_house_of_cards_height_l2996_299627


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l2996_299641

theorem max_value_quadratic_function (p : ℝ) (hp : p > 0) :
  (∃ x ∈ Set.Icc (0 : ℝ) (4 / p), -1 / (2 * p) * x^2 + x > 1) ↔ 2 < p ∧ p < 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l2996_299641


namespace NUMINAMATH_CALUDE_hex_9A3_to_base_4_l2996_299633

/-- Converts a single hexadecimal digit to its decimal representation -/
def hex_to_dec (h : Char) : ℕ :=
  match h with
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- Default case, though it should never be reached for valid hex digits

/-- Converts a hexadecimal number (as a string) to its decimal representation -/
def hex_to_dec_num (s : String) : ℕ :=
  s.foldl (fun acc d => 16 * acc + hex_to_dec d) 0

/-- Converts a natural number to its base 4 representation (as a list of digits) -/
def to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The main theorem: 9A3₁₆ is equal to 212203₄ -/
theorem hex_9A3_to_base_4 :
  to_base_4 (hex_to_dec_num "9A3") = [2, 1, 2, 2, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_hex_9A3_to_base_4_l2996_299633


namespace NUMINAMATH_CALUDE_eight_chairs_subsets_l2996_299681

/-- The number of subsets of n chairs arranged in a circle that contain at least k adjacent chairs. -/
def subsets_with_adjacent_chairs (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The main theorem: For 8 chairs arranged in a circle, there are 33 subsets containing at least 4 adjacent chairs. -/
theorem eight_chairs_subsets : subsets_with_adjacent_chairs 8 4 = 33 := by sorry

end NUMINAMATH_CALUDE_eight_chairs_subsets_l2996_299681


namespace NUMINAMATH_CALUDE_profit_comparison_l2996_299611

/-- The profit function for Product A before upgrade -/
def profit_A_before (raw_material : ℝ) : ℝ := 120000 * raw_material

/-- The profit function for Product A after upgrade -/
def profit_A_after (x : ℝ) : ℝ := 12 * (500 - x) * (1 + 0.005 * x)

/-- The profit function for Product B -/
def profit_B (x a : ℝ) : ℝ := 12 * (a - 0.013 * x) * x

theorem profit_comparison (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, 0 < x ∧ x ≤ 300 ∧ profit_A_after x ≥ profit_A_before 500) ∧
  (∀ x : ℝ, 0 < x → x ≤ 300 → profit_B x a ≤ profit_A_after x) →
  a ≤ 5.5 :=
sorry

end NUMINAMATH_CALUDE_profit_comparison_l2996_299611


namespace NUMINAMATH_CALUDE_grocery_spending_l2996_299658

theorem grocery_spending (X : ℚ) : 
  X > 0 → X - 3 - 2 - (1/3)*(X - 5) = 18 → X = 32 := by
  sorry

end NUMINAMATH_CALUDE_grocery_spending_l2996_299658


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2996_299622

theorem sufficient_not_necessary :
  (∀ x : ℝ, 1 + 3 / (x - 1) ≥ 0 → (x + 2) * (x - 1) ≥ 0) ∧
  (∃ x : ℝ, (x + 2) * (x - 1) ≥ 0 ∧ ¬(1 + 3 / (x - 1) ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2996_299622


namespace NUMINAMATH_CALUDE_characterization_of_m_l2996_299655

theorem characterization_of_m (m : ℕ+) : 
  (∃ p : ℕ, Prime p ∧ ∀ n : ℕ+, ¬(p ∣ n^n.val - m.val)) ↔ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_m_l2996_299655


namespace NUMINAMATH_CALUDE_counterexample_odd_composite_plus_two_prime_l2996_299663

theorem counterexample_odd_composite_plus_two_prime :
  ∃ n : ℕ, 
    Odd n ∧ 
    ¬ Prime n ∧ 
    n > 1 ∧ 
    ¬ Prime (n + 2) ∧
    n = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_counterexample_odd_composite_plus_two_prime_l2996_299663


namespace NUMINAMATH_CALUDE_number_ordering_l2996_299650

def a : ℕ := 62398
def b : ℕ := 63298
def c : ℕ := 62389
def d : ℕ := 63289

theorem number_ordering : b > d ∧ d > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_number_ordering_l2996_299650


namespace NUMINAMATH_CALUDE_parallelepiped_to_cube_l2996_299603

/-- A rectangular parallelepiped with edges 8, 8, and 27 has the same volume as a cube with side length 12 -/
theorem parallelepiped_to_cube : 
  let parallelepiped_volume := 8 * 8 * 27
  let cube_volume := 12 * 12 * 12
  parallelepiped_volume = cube_volume := by
  sorry

#eval 8 * 8 * 27
#eval 12 * 12 * 12

end NUMINAMATH_CALUDE_parallelepiped_to_cube_l2996_299603


namespace NUMINAMATH_CALUDE_shelbys_drive_l2996_299606

/-- Shelby's driving problem -/
theorem shelbys_drive (sunny_speed rainy_speed foggy_speed : ℚ)
  (total_distance total_time : ℚ) (sunny_time rainy_time foggy_time : ℚ) :
  sunny_speed = 35 →
  rainy_speed = 25 →
  foggy_speed = 15 →
  total_distance = 20 →
  total_time = 60 →
  sunny_time + rainy_time + foggy_time = total_time →
  sunny_speed * sunny_time / 60 + rainy_speed * rainy_time / 60 + foggy_speed * foggy_time / 60 = total_distance →
  foggy_time = 45 := by
  sorry

#check shelbys_drive

end NUMINAMATH_CALUDE_shelbys_drive_l2996_299606


namespace NUMINAMATH_CALUDE_arithmetic_arrangement_proof_l2996_299697

theorem arithmetic_arrangement_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∧
  ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_arrangement_proof_l2996_299697


namespace NUMINAMATH_CALUDE_cell_population_after_10_days_l2996_299605

/-- The number of cells in a colony after a given number of days, 
    where the initial population is 5 cells and the population triples every 3 days. -/
def cell_population (days : ℕ) : ℕ :=
  5 * 3^(days / 3)

/-- Theorem stating that the cell population after 10 days is 135 cells. -/
theorem cell_population_after_10_days : cell_population 10 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cell_population_after_10_days_l2996_299605


namespace NUMINAMATH_CALUDE_min_max_f_on_interval_l2996_299634

-- Define the function f(x) = x³ - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem min_max_f_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-3) 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-3) 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-3) 3, f x₂ = max) ∧
    min = -16 ∧ max = 16 := by
  sorry


end NUMINAMATH_CALUDE_min_max_f_on_interval_l2996_299634


namespace NUMINAMATH_CALUDE_pau_total_chicken_l2996_299653

/-- Calculates the total number of chicken pieces Pau eats given the initial orders and a second round of ordering. -/
theorem pau_total_chicken (kobe_order : ℝ) (pau_multiplier : ℝ) (pau_extra : ℝ) (shaq_extra_percent : ℝ) : 
  kobe_order = 5 →
  pau_multiplier = 2 →
  pau_extra = 2.5 →
  shaq_extra_percent = 0.5 →
  2 * (pau_multiplier * kobe_order + pau_extra) = 25 := by
  sorry

end NUMINAMATH_CALUDE_pau_total_chicken_l2996_299653


namespace NUMINAMATH_CALUDE_shopkeeper_bananas_l2996_299674

/-- The number of bananas bought by the shopkeeper -/
def num_bananas : ℕ := 448

/-- The number of oranges bought by the shopkeeper -/
def num_oranges : ℕ := 600

/-- The percentage of oranges that are rotten -/
def orange_rotten_percent : ℚ := 15 / 100

/-- The percentage of bananas that are rotten -/
def banana_rotten_percent : ℚ := 8 / 100

/-- The percentage of all fruits in good condition -/
def good_fruit_percent : ℚ := 878 / 1000

theorem shopkeeper_bananas :
  let total_fruits := num_oranges + num_bananas
  let rotten_oranges := (orange_rotten_percent * num_oranges : ℚ).floor
  let rotten_bananas := (banana_rotten_percent * num_bananas : ℚ).floor
  let good_fruits := (good_fruit_percent * total_fruits : ℚ).floor
  good_fruits = total_fruits - (rotten_oranges + rotten_bananas) :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_bananas_l2996_299674


namespace NUMINAMATH_CALUDE_simplify_expression_l2996_299656

theorem simplify_expression (x : ℝ) : 3*x + 5*x^2 + 2 - (9 - 4*x - 5*x^2) = 10*x^2 + 7*x - 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2996_299656


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l2996_299618

theorem binomial_coefficient_seven_three : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l2996_299618


namespace NUMINAMATH_CALUDE_hyperbola_through_C_l2996_299639

/-- Given a point A on the parabola y = x^2 and a point B such that OB is perpendicular to OA,
    prove that the point C formed by the rectangle AOBC lies on the hyperbola y = -2/x -/
theorem hyperbola_through_C (A B C : ℝ × ℝ) : 
  A.1 = -1/2 ∧ A.2 = 1/4 ∧                          -- A is (-1/2, 1/4)
  A.2 = A.1^2 ∧                                     -- A is on the parabola y = x^2
  B.1 = 2 ∧ B.2 = 4 ∧                               -- B is (2, 4)
  (B.2 - 0) / (B.1 - 0) = -(A.2 - 0) / (A.1 - 0) ∧  -- OB ⟂ OA
  C.1 = A.1 ∧ C.2 = B.2                             -- C forms rectangle AOBC
  →
  C.2 = -2 / C.1                                    -- C is on the hyperbola y = -2/x
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_through_C_l2996_299639


namespace NUMINAMATH_CALUDE_f_properties_l2996_299671

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + Real.sin (x + Real.pi / 2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∃ x, f x = 55 / 8) ∧
  (∃ x, f x = -9 / 8) ∧
  (∀ x, f x ≤ 55 / 8) ∧
  (∀ x, f x ≥ -9 / 8) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2996_299671


namespace NUMINAMATH_CALUDE_probability_letter_in_mathematics_l2996_299601

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_letter_in_mathematics :
  (unique_letters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_letter_in_mathematics_l2996_299601


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2996_299684

/-- The repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def target_fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2996_299684


namespace NUMINAMATH_CALUDE_power_three_124_mod_7_l2996_299662

theorem power_three_124_mod_7 : 3^124 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_three_124_mod_7_l2996_299662


namespace NUMINAMATH_CALUDE_alice_marbles_distinct_choices_l2996_299690

/-- Represents the colors of marbles --/
inductive Color
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents the marble collection --/
structure MarbleCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of distinct ways to choose 2 marbles --/
def distinctChoices (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that for Alice's marble collection, there are 9 distinct ways to choose 2 marbles --/
theorem alice_marbles_distinct_choices :
  let aliceCollection : MarbleCollection := ⟨3, 2, 1, 4⟩
  distinctChoices aliceCollection = 9 :=
sorry

end NUMINAMATH_CALUDE_alice_marbles_distinct_choices_l2996_299690


namespace NUMINAMATH_CALUDE_alice_favorite_number_l2996_299651

def is_favorite_number (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 70 ∧
  n % 7 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 10 + n % 10) % 4 = 0

theorem alice_favorite_number :
  ∀ n : ℕ, is_favorite_number n ↔ n = 35 := by
  sorry

end NUMINAMATH_CALUDE_alice_favorite_number_l2996_299651


namespace NUMINAMATH_CALUDE_sqrt_combinability_with_sqrt_3_l2996_299635

theorem sqrt_combinability_with_sqrt_3 :
  ∃! x : ℝ, (x = Real.sqrt 32 ∨ x = -Real.sqrt 27 ∨ x = Real.sqrt 12 ∨ x = Real.sqrt (1/3)) ∧
  (∃ y : ℝ, x = y ∧ y ≠ 0 ∧ ∀ a b : ℝ, (y = a * Real.sqrt 3 + b → a = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_combinability_with_sqrt_3_l2996_299635


namespace NUMINAMATH_CALUDE_A_final_value_l2996_299670

def update_A (initial_A : Int) : Int :=
  -initial_A + 5

theorem A_final_value (initial_A : Int) (h : initial_A = 15) :
  update_A initial_A = -10 := by
  sorry

end NUMINAMATH_CALUDE_A_final_value_l2996_299670
