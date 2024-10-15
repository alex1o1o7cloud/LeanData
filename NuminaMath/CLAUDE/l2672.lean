import Mathlib

namespace NUMINAMATH_CALUDE_parabola_equation_l2672_267246

/-- A parabola with the given properties has the equation y² = 4x -/
theorem parabola_equation (p : ℝ) (h₁ : p > 0) : 
  (∃ M : ℝ × ℝ, M.1 = 3 ∧ 
   ∃ F : ℝ × ℝ, F.1 = p/2 ∧ F.2 = 0 ∧ 
   (M.1 - F.1)^2 + (M.2 - F.2)^2 = (2*p)^2) →
  (∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 4*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2672_267246


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l2672_267225

theorem rectangle_area_error_percentage (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let actual_area := L * W
  let measured_area := 1.10 * L * 0.95 * W
  let error_percentage := (measured_area - actual_area) / actual_area * 100
  error_percentage = 4.5 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l2672_267225


namespace NUMINAMATH_CALUDE_online_store_problem_l2672_267265

/-- Represents the purchase and selling prices of products A and B -/
structure Prices where
  purchaseA : ℝ
  purchaseB : ℝ
  sellingA : ℝ
  sellingB : ℝ

/-- Represents the first purchase conditions -/
structure FirstPurchase where
  totalItems : ℕ
  totalCost : ℝ

/-- Represents the second purchase conditions -/
structure SecondPurchase where
  totalItems : ℕ
  maxCost : ℝ

/-- Represents the sales conditions for product B -/
structure BSales where
  initialSales : ℕ
  additionalSalesPerReduction : ℕ

/-- Main theorem stating the solutions to the problem -/
theorem online_store_problem 
  (prices : Prices)
  (firstPurchase : FirstPurchase)
  (secondPurchase : SecondPurchase)
  (bSales : BSales)
  (h1 : prices.purchaseA = 30)
  (h2 : prices.purchaseB = 25)
  (h3 : prices.sellingA = 45)
  (h4 : prices.sellingB = 37)
  (h5 : firstPurchase.totalItems = 30)
  (h6 : firstPurchase.totalCost = 850)
  (h7 : secondPurchase.totalItems = 80)
  (h8 : secondPurchase.maxCost = 2200)
  (h9 : bSales.initialSales = 4)
  (h10 : bSales.additionalSalesPerReduction = 2) :
  (∃ (x y : ℕ), x + y = firstPurchase.totalItems ∧ 
    prices.purchaseA * x + prices.purchaseB * y = firstPurchase.totalCost ∧ 
    x = 20 ∧ y = 10) ∧
  (∃ (m : ℕ), m ≤ secondPurchase.totalItems ∧ 
    prices.purchaseA * m + prices.purchaseB * (secondPurchase.totalItems - m) ≤ secondPurchase.maxCost ∧
    (prices.sellingA - prices.purchaseA) * m + (prices.sellingB - prices.purchaseB) * (secondPurchase.totalItems - m) = 2520 ∧
    m = 40) ∧
  (∃ (a₁ a₂ : ℝ), (12 - a₁) * (bSales.initialSales + 2 * a₁) = 90 ∧
    (12 - a₂) * (bSales.initialSales + 2 * a₂) = 90 ∧
    a₁ = 3 ∧ a₂ = 7) := by
  sorry

end NUMINAMATH_CALUDE_online_store_problem_l2672_267265


namespace NUMINAMATH_CALUDE_max_common_ratio_geometric_sequence_l2672_267286

/-- Given a geometric sequence {a_n} satisfying a_1(a_2 + a_3) = 6a_1 - 9, 
    the maximum value of the common ratio q is (-1 + √5) / 2 -/
theorem max_common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 * (a 2 + a 3) = 6 * a 1 - 9 →  -- given equation
  q ≤ (-1 + Real.sqrt 5) / 2 ∧
  ∃ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n * q) ∧ 
    a 1 * (a 2 + a 3) = 6 * a 1 - 9 ∧ 
    q = (-1 + Real.sqrt 5) / 2 := by
  sorry

#check max_common_ratio_geometric_sequence

end NUMINAMATH_CALUDE_max_common_ratio_geometric_sequence_l2672_267286


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l2672_267274

theorem inverse_proportion_relationship (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁)
  (h2 : y₂ = 2 / x₂)
  (h3 : x₁ > 0)
  (h4 : 0 > x₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l2672_267274


namespace NUMINAMATH_CALUDE_integral_sin4_cos4_3x_l2672_267232

theorem integral_sin4_cos4_3x (x : ℝ) : 
  ∫ x in (0 : ℝ)..(2 * Real.pi), (Real.sin (3 * x))^4 * (Real.cos (3 * x))^4 = (3 * Real.pi) / 64 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin4_cos4_3x_l2672_267232


namespace NUMINAMATH_CALUDE_talent_show_girls_l2672_267287

theorem talent_show_girls (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 34 → difference = 22 → girls = total - (total - difference) / 2 → girls = 28 := by
sorry

end NUMINAMATH_CALUDE_talent_show_girls_l2672_267287


namespace NUMINAMATH_CALUDE_higher_power_of_two_divisibility_l2672_267283

theorem higher_power_of_two_divisibility (n k : ℕ) : 
  ∃ i ∈ Finset.range k, ∀ j ∈ Finset.range k, j ≠ i → 
    (∃ m : ℕ, (n + i + 1) = 2^m * (2*l + 1) ∧ 
              ∀ p : ℕ, (n + j + 1) = 2^p * (2*q + 1) → m > p) :=
by sorry

end NUMINAMATH_CALUDE_higher_power_of_two_divisibility_l2672_267283


namespace NUMINAMATH_CALUDE_characterization_of_m_l2672_267280

theorem characterization_of_m (m : ℕ+) : 
  (∃ p : ℕ, Prime p ∧ ∀ n : ℕ+, ¬(p ∣ n^n.val - m.val)) ↔ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_m_l2672_267280


namespace NUMINAMATH_CALUDE_projection_composition_l2672_267275

open Matrix

/-- The matrix that projects a vector onto (4, 2) -/
def proj_matrix_1 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4/5, 2/5; 2/5, 1/5]

/-- The matrix that projects a vector onto (2, 1) -/
def proj_matrix_2 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4/5, 2/5; 2/5, 1/5]

/-- The theorem stating that the composition of the two projection matrices
    results in the same matrix -/
theorem projection_composition :
  proj_matrix_2 * proj_matrix_1 = !![4/5, 2/5; 2/5, 1/5] := by sorry

end NUMINAMATH_CALUDE_projection_composition_l2672_267275


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2672_267261

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → (3 ∣ a) ∨ (3 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2672_267261


namespace NUMINAMATH_CALUDE_unreachable_zero_l2672_267211

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of possible moves -/
inductive Move where
  | swap : Move
  | scale : Move
  | negate : Move
  | increment : Move
  | decrement : Move

/-- Apply a move to a point -/
def applyMove (p : Point) (m : Move) : Point :=
  match m with
  | Move.swap => ⟨p.y, p.x⟩
  | Move.scale => ⟨3 * p.x, -2 * p.y⟩
  | Move.negate => ⟨-2 * p.x, 3 * p.y⟩
  | Move.increment => ⟨p.x + 1, p.y + 4⟩
  | Move.decrement => ⟨p.x - 1, p.y - 4⟩

/-- The sum of coordinates modulo 5 -/
def sumMod5 (p : Point) : ℤ :=
  (p.x + p.y) % 5

/-- Theorem: It's impossible to reach (0, 0) from (0, 1) using the given moves -/
theorem unreachable_zero : 
  ∀ (moves : List Move), 
    let finalPoint := moves.foldl applyMove ⟨0, 1⟩
    sumMod5 finalPoint ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_unreachable_zero_l2672_267211


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l2672_267235

-- Define the circles and line
def C₁ (x y : ℝ) := (x + 3)^2 + y^2 = 4
def C₂ (x y : ℝ) := (x + 1)^2 + (y + 2)^2 = 4
def symmetry_line (x y : ℝ) := x - y + 1 = 0

-- Define points
def A : ℝ × ℝ := (0, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the theorem
theorem circle_and_line_problem :
  -- Given conditions
  (∀ x y : ℝ, C₁ x y ↔ C₂ (y - 1) (-x - 1)) →  -- Symmetry condition
  (∃ k : ℝ, ∀ x : ℝ, C₁ x (k * x + 3)) →      -- Line l intersects C₁
  -- Conclusion
  ((∀ x y : ℝ, C₁ x y ↔ (x + 3)^2 + y^2 = 4) ∧
   (∃ M N : ℝ × ℝ, 
     (C₁ M.1 M.2 ∧ C₁ N.1 N.2) ∧
     (M.2 = 2 * M.1 + 3 ∨ M.2 = 3 * M.1 + 3) ∧
     (N.2 = 2 * N.1 + 3 ∨ N.2 = 3 * N.1 + 3) ∧
     (M.1 * N.1 + M.2 * N.2 = 7/5))) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_line_problem_l2672_267235


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l2672_267251

theorem merry_go_round_revolutions 
  (distance_A : ℝ) 
  (distance_B : ℝ) 
  (revolutions_A : ℝ) 
  (h1 : distance_A = 36) 
  (h2 : distance_B = 12) 
  (h3 : revolutions_A = 40) 
  (h4 : distance_A * revolutions_A = distance_B * revolutions_B) : 
  revolutions_B = 120 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l2672_267251


namespace NUMINAMATH_CALUDE_equation_solutions_l2672_267223

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 6 ∧ 
    ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 3/4 ∧ 
    ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2672_267223


namespace NUMINAMATH_CALUDE_average_decrease_l2672_267210

theorem average_decrease (n : ℕ) (initial_avg : ℚ) (new_obs : ℚ) : 
  n = 6 → 
  initial_avg = 11 → 
  new_obs = 4 → 
  (n * initial_avg + new_obs) / (n + 1) = initial_avg - 1 := by
sorry

end NUMINAMATH_CALUDE_average_decrease_l2672_267210


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l2672_267255

theorem cos_sixty_degrees : Real.cos (60 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l2672_267255


namespace NUMINAMATH_CALUDE_total_fruits_grown_special_technique_watermelons_special_pineapples_l2672_267276

/-- Represents the fruit growing data for a person -/
structure FruitData where
  watermelons : ℕ
  pineapples : ℕ
  mangoes : ℕ
  organic_watermelons : ℕ
  hydroponic_watermelons : ℕ
  dry_season_pineapples : ℕ
  vertical_pineapples : ℕ

/-- The fruit growing data for Jason -/
def jason : FruitData := {
  watermelons := 37,
  pineapples := 56,
  mangoes := 0,
  organic_watermelons := 15,
  hydroponic_watermelons := 0,
  dry_season_pineapples := 23,
  vertical_pineapples := 0
}

/-- The fruit growing data for Mark -/
def mark : FruitData := {
  watermelons := 68,
  pineapples := 27,
  mangoes := 0,
  organic_watermelons := 0,
  hydroponic_watermelons := 21,
  dry_season_pineapples := 0,
  vertical_pineapples := 17
}

/-- The fruit growing data for Sandy -/
def sandy : FruitData := {
  watermelons := 11,
  pineapples := 14,
  mangoes := 42,
  organic_watermelons := 0,
  hydroponic_watermelons := 0,
  dry_season_pineapples := 0,
  vertical_pineapples := 0
}

/-- Calculate the total fruits for a person -/
def totalFruits (data : FruitData) : ℕ :=
  data.watermelons + data.pineapples + data.mangoes

/-- Theorem stating the total number of fruits grown by all three people -/
theorem total_fruits_grown :
  totalFruits jason + totalFruits mark + totalFruits sandy = 255 := by
  sorry

/-- Theorem stating the number of watermelons grown using special techniques -/
theorem special_technique_watermelons :
  jason.organic_watermelons + mark.hydroponic_watermelons = 36 := by
  sorry

/-- Theorem stating the number of pineapples grown in dry season or vertically -/
theorem special_pineapples :
  jason.dry_season_pineapples + mark.vertical_pineapples = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_grown_special_technique_watermelons_special_pineapples_l2672_267276


namespace NUMINAMATH_CALUDE_binary_1101101000_to_octal_1550_l2672_267269

def binary_to_decimal (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem binary_1101101000_to_octal_1550 :
  let binary : List Bool := [false, false, false, true, false, true, true, false, true, true]
  let octal : List Nat := [0, 5, 5, 1]
  decimal_to_octal (binary_to_decimal binary) = octal.reverse := by
  sorry

end NUMINAMATH_CALUDE_binary_1101101000_to_octal_1550_l2672_267269


namespace NUMINAMATH_CALUDE_nancy_carrot_nv_l2672_267208

/-- Calculates the total nutritional value of carrots based on given conditions -/
def total_nutritional_value (initial_carrots : ℕ) (kept_carrots : ℕ) (new_seeds : ℕ) 
  (growth_factor : ℕ) (base_nv : ℝ) (nv_per_cm : ℝ) (growth_cm : ℝ) : ℝ :=
  let new_carrots := new_seeds * growth_factor
  let total_carrots := initial_carrots - kept_carrots + new_carrots
  let good_carrots := total_carrots - (total_carrots / 3)
  let new_carrot_nv := new_carrots * (base_nv + nv_per_cm * growth_cm)
  let kept_carrot_nv := kept_carrots * base_nv
  new_carrot_nv + kept_carrot_nv

/-- Theorem stating that the total nutritional value of Nancy's carrots is 92 -/
theorem nancy_carrot_nv : 
  total_nutritional_value 12 2 5 3 1 0.5 12 = 92 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrot_nv_l2672_267208


namespace NUMINAMATH_CALUDE_equation_solution_l2672_267201

theorem equation_solution : ∃! x : ℝ, x ≥ 2 ∧ 
  Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 3 ∧ 
  x = 44.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2672_267201


namespace NUMINAMATH_CALUDE_johns_grocery_spend_l2672_267273

/-- Represents the cost of John's purchase at the grocery store. -/
def grocery_purchase (chip_price corn_chip_price : ℚ) (chip_quantity corn_chip_quantity : ℕ) : ℚ :=
  chip_price * chip_quantity + corn_chip_price * corn_chip_quantity

/-- Proves that John's total spend is $45 given the specified conditions. -/
theorem johns_grocery_spend :
  grocery_purchase 2 1.5 15 10 = 45 := by
sorry

end NUMINAMATH_CALUDE_johns_grocery_spend_l2672_267273


namespace NUMINAMATH_CALUDE_intersection_distance_and_max_value_l2672_267234

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ : ℝ) : Prop := ρ = 1

/-- Curve C₂ in parametric form -/
def C₂ (t x y : ℝ) : Prop := x = 1 + t ∧ y = 2 + t

/-- Point M on C₁ -/
def M (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem intersection_distance_and_max_value :
  ∃ (A B : ℝ × ℝ),
    (∀ ρ, C₁ ρ → (A.1^2 + A.2^2 = ρ^2 ∧ B.1^2 + B.2^2 = ρ^2)) ∧
    (∃ t₁ t₂, C₂ t₁ A.1 A.2 ∧ C₂ t₂ B.1 B.2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 ∧
    (∀ x y, M x y → (x + 1) * (y + 1) ≤ 3/2 + Real.sqrt 2) ∧
    (∃ x y, M x y ∧ (x + 1) * (y + 1) = 3/2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_and_max_value_l2672_267234


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l2672_267200

/-- The number of additional people needed to mow a lawn in a shorter time -/
def additional_people_needed (initial_people initial_time target_time : ℕ) : ℕ :=
  (initial_people * initial_time / target_time) - initial_people

/-- Proof that 24 additional people are needed to mow the lawn in 2 hours -/
theorem lawn_mowing_problem :
  additional_people_needed 8 8 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l2672_267200


namespace NUMINAMATH_CALUDE_hema_rahul_ratio_l2672_267272

-- Define variables for ages
variable (Raj Ravi Hema Rahul : ℚ)

-- Define the conditions
axiom raj_older : Raj = Ravi + 3
axiom hema_younger : Hema = Ravi - 2
axiom raj_triple : Raj = 3 * Rahul
axiom raj_twenty : Raj = 20
axiom raj_hema_ratio : Raj = Hema + (1/3) * Hema

-- Theorem to prove
theorem hema_rahul_ratio : Hema / Rahul = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_hema_rahul_ratio_l2672_267272


namespace NUMINAMATH_CALUDE_kolya_tolya_ages_l2672_267204

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (is_valid : tens < 10 ∧ ones < 10)

/-- Calculates the numeric value of an Age -/
def Age.value (a : Age) : Nat :=
  10 * a.tens + a.ones

/-- Reverses the digits of an Age -/
def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.is_valid.symm⟩

theorem kolya_tolya_ages :
  ∃ (kolya_age tolya_age : Age),
    -- Kolya is older than Tolya
    kolya_age.value > tolya_age.value ∧
    -- Both ages are less than 100
    kolya_age.value < 100 ∧ tolya_age.value < 100 ∧
    -- Reversing Kolya's age gives Tolya's age
    kolya_age.reverse = tolya_age ∧
    -- The difference of squares is a perfect square
    ∃ (k : Nat), (kolya_age.value ^ 2 - tolya_age.value ^ 2 = k ^ 2) ∧
    -- Kolya is 65 and Tolya is 56
    kolya_age.value = 65 ∧ tolya_age.value = 56 := by
  sorry

end NUMINAMATH_CALUDE_kolya_tolya_ages_l2672_267204


namespace NUMINAMATH_CALUDE_square_minus_self_sum_l2672_267257

theorem square_minus_self_sum : (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_sum_l2672_267257


namespace NUMINAMATH_CALUDE_fraction_problem_l2672_267288

theorem fraction_problem (x : ℝ) : 
  (x * 7000 - (1 / 1000) * 7000 = 700) ↔ (x = 0.101) :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l2672_267288


namespace NUMINAMATH_CALUDE_min_additional_coins_l2672_267231

/-- The number of friends Alex has -/
def num_friends : ℕ := 12

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 63

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of additional coins needed -/
def additional_coins_needed : ℕ := sum_first_n num_friends - initial_coins

theorem min_additional_coins :
  additional_coins_needed = 15 :=
sorry

end NUMINAMATH_CALUDE_min_additional_coins_l2672_267231


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2672_267227

theorem quadratic_roots_difference (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - m*x₁ + 8 = 0 ∧ 
   x₂^2 - m*x₂ + 8 = 0 ∧ 
   |x₁ - x₂| = Real.sqrt 84) →
  m ≤ 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2672_267227


namespace NUMINAMATH_CALUDE_andy_candy_canes_l2672_267218

/-- The number of candy canes Andy got from his parents -/
def parents_candy : ℕ := sorry

/-- The number of candy canes Andy got from teachers -/
def teachers_candy : ℕ := 3 * 4

/-- The ratio of candy canes to cavities -/
def candy_to_cavity_ratio : ℕ := 4

/-- The number of cavities Andy got -/
def cavities : ℕ := 16

/-- The fraction of additional candy canes Andy buys -/
def bought_candy_fraction : ℚ := 1 / 7

theorem andy_candy_canes :
  parents_candy = 44 ∧
  parents_candy + teachers_candy + (parents_candy + teachers_candy : ℚ) * bought_candy_fraction = cavities * candy_to_cavity_ratio := by sorry

end NUMINAMATH_CALUDE_andy_candy_canes_l2672_267218


namespace NUMINAMATH_CALUDE_smallest_common_factor_l2672_267291

theorem smallest_common_factor (n : ℕ) : 
  (∃ k : ℕ, k > 1 ∧ k ∣ (9*n - 2) ∧ k ∣ (7*n + 3)) → n ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l2672_267291


namespace NUMINAMATH_CALUDE_g_3_6_neg1_eq_one_seventh_l2672_267294

/-- The function g as defined in the problem -/
def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

/-- Theorem stating that g(3, 6, -1) = 1/7 -/
theorem g_3_6_neg1_eq_one_seventh : g 3 6 (-1) = 1/7 := by sorry

end NUMINAMATH_CALUDE_g_3_6_neg1_eq_one_seventh_l2672_267294


namespace NUMINAMATH_CALUDE_determinant_inequality_range_l2672_267254

theorem determinant_inequality_range (x : ℝ) : 
  (Matrix.det !![x + 3, x^2; 1, 4] < 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ioi 6) := by
  sorry

end NUMINAMATH_CALUDE_determinant_inequality_range_l2672_267254


namespace NUMINAMATH_CALUDE_second_sale_price_is_270_l2672_267259

/-- Represents the clock selling scenario in a shop --/
structure ClockSale where
  originalCost : ℝ
  firstSaleMarkup : ℝ
  buybackPercentage : ℝ
  secondSaleProfit : ℝ
  costDifference : ℝ

/-- Calculates the second selling price of the clock --/
def secondSellingPrice (sale : ClockSale) : ℝ :=
  let firstSalePrice := sale.originalCost * (1 + sale.firstSaleMarkup)
  let buybackPrice := firstSalePrice * sale.buybackPercentage
  buybackPrice * (1 + sale.secondSaleProfit)

/-- Theorem stating the second selling price is $270 given the conditions --/
theorem second_sale_price_is_270 (sale : ClockSale)
  (h1 : sale.firstSaleMarkup = 0.2)
  (h2 : sale.buybackPercentage = 0.5)
  (h3 : sale.secondSaleProfit = 0.8)
  (h4 : sale.originalCost - (sale.originalCost * (1 + sale.firstSaleMarkup) * sale.buybackPercentage) = sale.costDifference)
  (h5 : sale.costDifference = 100)
  : secondSellingPrice sale = 270 := by
  sorry

#eval secondSellingPrice {
  originalCost := 250,
  firstSaleMarkup := 0.2,
  buybackPercentage := 0.5,
  secondSaleProfit := 0.8,
  costDifference := 100
}

end NUMINAMATH_CALUDE_second_sale_price_is_270_l2672_267259


namespace NUMINAMATH_CALUDE_inequality_counterexample_l2672_267292

theorem inequality_counterexample :
  ∃ (a b c d : ℝ), a > b ∧ c > d ∧ a + d ≤ b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_counterexample_l2672_267292


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2672_267209

theorem perfect_square_condition (n : ℕ+) : 
  ∃ (m : ℕ), 2^n.val + 12^n.val + 2011^n.val = m^2 ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2672_267209


namespace NUMINAMATH_CALUDE_parabola_vertex_l2672_267219

/-- The parabola defined by y = -x^2 + 3 has its vertex at (0, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -x^2 + 3 → (0, 3) = (x, y) ∨ ∃ z, y < -z^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2672_267219


namespace NUMINAMATH_CALUDE_equation_solution_l2672_267230

theorem equation_solution : ∃ r : ℝ, (24 - 5 = 3 * r + 7) ∧ (r = 4) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2672_267230


namespace NUMINAMATH_CALUDE_simplify_expression_l2672_267281

theorem simplify_expression (x : ℝ) : 3*x + 5*x^2 + 2 - (9 - 4*x - 5*x^2) = 10*x^2 + 7*x - 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2672_267281


namespace NUMINAMATH_CALUDE_polygon_construction_possible_l2672_267240

/-- Represents a line segment with a fixed length -/
structure LineSegment where
  length : ℝ

/-- Represents a polygon constructed from line segments -/
structure Polygon where
  segments : List LineSegment
  isValid : Bool  -- Indicates if the polygon is valid (closed and non-self-intersecting)

/-- Calculates the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Checks if it's possible to construct a polygon with given area using given line segments -/
def canConstructPolygon (segments : List LineSegment) (targetArea : ℝ) : Prop :=
  ∃ (p : Polygon), p.segments = segments ∧ p.isValid ∧ calculateArea p = targetArea

theorem polygon_construction_possible :
  let segments := List.replicate 12 { length := 2 }
  canConstructPolygon segments 16 := by
  sorry

end NUMINAMATH_CALUDE_polygon_construction_possible_l2672_267240


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2672_267206

theorem complex_fraction_simplification :
  (7 + 16 * Complex.I) / (3 - 4 * Complex.I) = 6 - (38 / 7) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2672_267206


namespace NUMINAMATH_CALUDE_school_bus_seats_l2672_267262

theorem school_bus_seats (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 60) (h2 : num_buses = 6) (h3 : total_students % num_buses = 0) :
  total_students / num_buses = 10 := by
sorry

end NUMINAMATH_CALUDE_school_bus_seats_l2672_267262


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2672_267233

theorem book_arrangement_count :
  let math_books : ℕ := 4
  let english_books : ℕ := 5
  let science_books : ℕ := 2
  let subject_groups : ℕ := 3
  let total_arrangements : ℕ :=
    (Nat.factorial subject_groups) *
    (Nat.factorial math_books) *
    (Nat.factorial english_books) *
    (Nat.factorial science_books)
  total_arrangements = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2672_267233


namespace NUMINAMATH_CALUDE_f_f_eq_x_solutions_l2672_267217

def f (x : ℝ) : ℝ := x^2 - 4*x - 5

def solution_set : Set ℝ := {(5 + 3*Real.sqrt 5)/2, (5 - 3*Real.sqrt 5)/2, (3 + Real.sqrt 41)/2, (3 - Real.sqrt 41)/2}

theorem f_f_eq_x_solutions :
  ∀ x : ℝ, f (f x) = x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_f_f_eq_x_solutions_l2672_267217


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2672_267224

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 30th term of the arithmetic sequence with first term 3 and common difference 6 is 177 -/
theorem thirtieth_term_of_sequence : arithmetic_sequence 3 6 30 = 177 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2672_267224


namespace NUMINAMATH_CALUDE_maries_daily_rent_is_24_l2672_267279

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

end NUMINAMATH_CALUDE_maries_daily_rent_is_24_l2672_267279


namespace NUMINAMATH_CALUDE_cos_power_six_expansion_l2672_267285

theorem cos_power_six_expansion (b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ θ : ℝ, Real.cos θ ^ 6 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) +
    b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 = 131 / 512 :=
by sorry

end NUMINAMATH_CALUDE_cos_power_six_expansion_l2672_267285


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_no_negative_nine_l2672_267256

theorem greatest_integer_b_for_no_negative_nine : ∃ (b : ℤ), 
  (∀ x : ℝ, 3 * x^2 + b * x + 15 ≠ -9) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, 3 * x^2 + c * x + 15 = -9) ∧
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_no_negative_nine_l2672_267256


namespace NUMINAMATH_CALUDE_books_at_end_of_month_l2672_267289

/-- Given a special collection of books, calculate the number of books at the end of the month. -/
theorem books_at_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 75)
  (h2 : loaned_books = 40)
  (h3 : return_rate = 65 / 100) : 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 61 := by
  sorry

#check books_at_end_of_month

end NUMINAMATH_CALUDE_books_at_end_of_month_l2672_267289


namespace NUMINAMATH_CALUDE_average_attendance_theorem_l2672_267258

/-- Calculates the average daily attendance for a week given the attendance data --/
def averageDailyAttendance (
  mondayAttendance : ℕ)
  (tuesdayAttendance : ℕ)
  (wednesdayToFridayAttendance : ℕ)
  (saturdayAttendance : ℕ)
  (sundayAttendance : ℕ)
  (absenteesJoiningWednesday : ℕ)
  (tuesdayOnlyAttendees : ℕ) : ℚ :=
  let totalAttendance := 
    mondayAttendance + 
    tuesdayAttendance + 
    (wednesdayToFridayAttendance + absenteesJoiningWednesday) + 
    wednesdayToFridayAttendance * 2 + 
    saturdayAttendance + 
    sundayAttendance
  totalAttendance / 7

/-- Theorem stating that the average daily attendance is 78/7 given the specific attendance data --/
theorem average_attendance_theorem :
  averageDailyAttendance 10 15 10 8 12 3 2 = 78 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_attendance_theorem_l2672_267258


namespace NUMINAMATH_CALUDE_non_juniors_playing_sport_l2672_267266

theorem non_juniors_playing_sport (total_students : ℕ) 
  (juniors_play_percent : ℚ) (non_juniors_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) : ℕ :=
  
  -- Define the given conditions
  let total_students := 600
  let juniors_play_percent := 1/2
  let non_juniors_not_play_percent := 2/5
  let total_not_play_percent := 13/25

  -- Define the number of non-juniors who play a sport
  let non_juniors_play := 72

  -- Proof statement (not implemented)
  by sorry

end NUMINAMATH_CALUDE_non_juniors_playing_sport_l2672_267266


namespace NUMINAMATH_CALUDE_jack_reading_pages_l2672_267239

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 13

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 67

/-- The total number of pages Jack needs to read -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem jack_reading_pages : total_pages = 871 := by
  sorry

end NUMINAMATH_CALUDE_jack_reading_pages_l2672_267239


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l2672_267297

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle given three points -/
def area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Angle in degrees between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem triangle_angle_theorem (t : Triangle) :
  let O := circumcenter t
  angle t.B t.C t.A = 75 →
  area O t.A t.B + area O t.B t.C = Real.sqrt 3 * area O t.C t.A →
  angle t.B t.A t.C = 45 := by
    sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l2672_267297


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l2672_267215

/-- The horizontal length of a rectangle with perimeter 54 cm and horizontal length 3 cm longer than vertical length is 15 cm. -/
theorem rectangle_horizontal_length :
  ∀ (h v : ℝ), 
    (2 * h + 2 * v = 54) →  -- Perimeter is 54 cm
    (h = v + 3) →           -- Horizontal length is 3 cm longer than vertical length
    h = 15 := by            -- Horizontal length is 15 cm
  sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l2672_267215


namespace NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_l2672_267220

/-- Represents the initial speed of the car in km/h -/
def V : ℝ := 60

/-- Represents the distance from A to B in km -/
def distance : ℝ := 300

/-- Represents the increase in speed on the return journey in km/h -/
def speed_increase : ℝ := 16

/-- Represents the time after which the speed was increased on the return journey in hours -/
def time_before_increase : ℝ := 1.2

/-- Represents the time difference between the outward and return journeys in hours -/
def time_difference : ℝ := 0.8

/-- Theorem stating that the initial speed satisfies the given conditions -/
theorem initial_speed_satisfies_conditions :
  (distance / V - time_difference = 
   time_before_increase + (distance - V * time_before_increase) / (V + speed_increase)) := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_l2672_267220


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2672_267284

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x - 2*y)^5 = a*x^5 + a₁*x^4*y + a₂*x^3*y^2 + a₃*x^2*y^3 + a₄*x*y^4 + a₅*y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2672_267284


namespace NUMINAMATH_CALUDE_nested_square_root_value_l2672_267290

theorem nested_square_root_value :
  ∀ y : ℝ, y = Real.sqrt (3 + y) → y = (1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l2672_267290


namespace NUMINAMATH_CALUDE_total_highlighters_l2672_267243

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_pink : pink = 4)
  (h_yellow : yellow = 2)
  (h_blue : blue = 5) :
  pink + yellow + blue = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l2672_267243


namespace NUMINAMATH_CALUDE_puzzle_missing_pieces_l2672_267248

theorem puzzle_missing_pieces 
  (total_pieces : ℕ) 
  (border_pieces : ℕ) 
  (trevor_pieces : ℕ) 
  (joe_multiplier : ℕ) : 
  total_pieces = 500 →
  border_pieces = 75 →
  trevor_pieces = 105 →
  joe_multiplier = 3 →
  total_pieces - border_pieces - (trevor_pieces + joe_multiplier * trevor_pieces) = 5 :=
by
  sorry

#check puzzle_missing_pieces

end NUMINAMATH_CALUDE_puzzle_missing_pieces_l2672_267248


namespace NUMINAMATH_CALUDE_average_string_length_l2672_267253

theorem average_string_length : 
  let string_lengths : List ℝ := [1.5, 4.5, 6, 3]
  let n : ℕ := string_lengths.length
  let sum : ℝ := string_lengths.sum
  sum / n = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_string_length_l2672_267253


namespace NUMINAMATH_CALUDE_all_squares_similar_l2672_267270

/-- A square is a quadrilateral with all sides equal and all angles 90 degrees. -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Similarity of shapes means they have the same shape but not necessarily the same size. -/
def are_similar (s1 s2 : Square) : Prop :=
  ∃ k : ℝ, k > 0 ∧ s1.side = k * s2.side

/-- Any two squares are similar. -/
theorem all_squares_similar (s1 s2 : Square) : are_similar s1 s2 := by
  sorry

end NUMINAMATH_CALUDE_all_squares_similar_l2672_267270


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2672_267221

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2672_267221


namespace NUMINAMATH_CALUDE_ball_probabilities_l2672_267271

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def num_drawn : ℕ := 2

theorem ball_probabilities :
  (num_red * (num_red - 1) / (total_balls * (total_balls - 1)) = 3 / 10) ∧
  (1 - (num_white * (num_white - 1) / (total_balls * (total_balls - 1))) = 9 / 10) :=
sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2672_267271


namespace NUMINAMATH_CALUDE_negation_equivalence_l2672_267229

-- Define the curve
def is_curve (m : ℕ) (x y : ℝ) : Prop := x^2 / m + y^2 = 1

-- Define what it means for the curve to be an ellipse (this is a placeholder definition)
def is_ellipse (m : ℕ) : Prop := ∃ x y : ℝ, is_curve m x y

-- The theorem to prove
theorem negation_equivalence :
  (¬ ∃ m : ℕ, is_ellipse m) ↔ (∀ m : ℕ, ¬ is_ellipse m) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2672_267229


namespace NUMINAMATH_CALUDE_problem_statement_l2672_267293

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 3) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ y₀ / x₀ + 3 / y₀ = 4 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → y' / x' + 3 / y' ≥ 4) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ x₀ * y₀ = 9 / 8 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → x' * y' ≤ 9 / 8) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ x₀^2 + 4 * y₀^2 = 9 / 2 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → x'^2 + 4 * y'^2 ≥ 9 / 2) ∧
  ¬(∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → Real.sqrt x' + Real.sqrt (2 * y') ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2672_267293


namespace NUMINAMATH_CALUDE_skating_time_calculation_l2672_267238

/-- The number of days Gage skated for each duration -/
def days_per_duration : ℕ := 4

/-- The duration of skating in minutes for the first set of days -/
def duration1 : ℕ := 80

/-- The duration of skating in minutes for the second set of days -/
def duration2 : ℕ := 105

/-- The desired average skating time in minutes per day -/
def desired_average : ℕ := 100

/-- The total number of days, including the day to be calculated -/
def total_days : ℕ := 2 * days_per_duration + 1

/-- The required skating time on the last day to achieve the desired average -/
def required_time : ℕ := 160

theorem skating_time_calculation :
  (days_per_duration * duration1 + days_per_duration * duration2 + required_time) / total_days = desired_average := by
  sorry

end NUMINAMATH_CALUDE_skating_time_calculation_l2672_267238


namespace NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l2672_267203

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 20) :
  let s := Real.sqrt ((d^2) / 2)
  4 * s = 40 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l2672_267203


namespace NUMINAMATH_CALUDE_sum_of_digits_0_to_99_l2672_267228

/-- The sum of all digits of integers from 0 to 99 inclusive -/
def sum_of_digits : ℕ := 900

/-- Theorem stating that the sum of all digits of integers from 0 to 99 inclusive is 900 -/
theorem sum_of_digits_0_to_99 : sum_of_digits = 900 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_0_to_99_l2672_267228


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l2672_267277

/-- Given a square and two circles with specific relationships, 
    prove the circumference of the smaller circle -/
theorem smaller_circle_circumference 
  (square_area : ℝ) 
  (larger_radius smaller_radius : ℝ) 
  (h1 : square_area = 784)
  (h2 : square_area = (2 * larger_radius)^2)
  (h3 : larger_radius = (7/3) * smaller_radius) : 
  2 * Real.pi * smaller_radius = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l2672_267277


namespace NUMINAMATH_CALUDE_complement_A_when_a_5_union_A_B_when_a_2_l2672_267295

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}
def B : Set ℝ := {x | x < 0 ∨ x > 5}

-- Theorem 1: Complement of A when a = 5
theorem complement_A_when_a_5 : 
  (A 5)ᶜ = {x : ℝ | x < 4 ∨ x > 11} := by sorry

-- Theorem 2: Union of A and B when a = 2
theorem union_A_B_when_a_2 : 
  A 2 ∪ B = {x : ℝ | x < 0 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_when_a_5_union_A_B_when_a_2_l2672_267295


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2672_267245

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_problem (a : ℝ) : 
  (f_derivative a 1 * (2 - 1) + f a 1 = 7) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2672_267245


namespace NUMINAMATH_CALUDE_rectangle_area_l2672_267236

/-- A rectangle divided into four identical squares with a given perimeter has a specific area -/
theorem rectangle_area (perimeter : ℝ) (h_perimeter : perimeter = 160) :
  let side_length := perimeter / 10
  let length := 4 * side_length
  let width := side_length
  let area := length * width
  area = 1024 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2672_267236


namespace NUMINAMATH_CALUDE_catch_up_point_l2672_267296

/-- Represents a car traveling between two cities -/
structure Car where
  speed : ℝ
  startTime : ℝ
  arrivalTime : ℝ

/-- The problem setup -/
def travelProblem (distanceAB : ℝ) (carA carB : Car) : Prop :=
  distanceAB > 0 ∧
  carA.startTime = carB.startTime + 1 ∧
  carA.arrivalTime + 1 = carB.arrivalTime ∧
  distanceAB = carA.speed * (carA.arrivalTime - carA.startTime) ∧
  distanceAB = carB.speed * (carB.arrivalTime - carB.startTime)

/-- The theorem to be proved -/
theorem catch_up_point (distanceAB : ℝ) (carA carB : Car) 
  (h : travelProblem distanceAB carA carB) : 
  ∃ (t : ℝ), carA.speed * (t - carA.startTime) = carB.speed * (t - carB.startTime) ∧ 
              carA.speed * (t - carA.startTime) = distanceAB - 150 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_point_l2672_267296


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2672_267205

theorem arithmetic_calculation : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2672_267205


namespace NUMINAMATH_CALUDE_slope_of_line_l2672_267278

/-- The slope of a line passing through two points is 1 -/
theorem slope_of_line (M N : ℝ × ℝ) (h1 : M = (-Real.sqrt 3, Real.sqrt 2)) 
  (h2 : N = (-Real.sqrt 2, Real.sqrt 3)) : 
  (N.2 - M.2) / (N.1 - M.1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2672_267278


namespace NUMINAMATH_CALUDE_candidate_a_support_l2672_267252

/-- Represents the percentage of registered voters in each category -/
structure VoterDistribution :=
  (democrats : ℝ)
  (republicans : ℝ)
  (independents : ℝ)
  (undecided : ℝ)

/-- Represents the percentage of voters in each category supporting candidate A -/
structure SupportDistribution :=
  (democrats : ℝ)
  (republicans : ℝ)
  (independents : ℝ)
  (undecided : ℝ)

/-- Calculates the total percentage of registered voters supporting candidate A -/
def calculateTotalSupport (vd : VoterDistribution) (sd : SupportDistribution) : ℝ :=
  vd.democrats * sd.democrats +
  vd.republicans * sd.republicans +
  vd.independents * sd.independents +
  vd.undecided * sd.undecided

theorem candidate_a_support :
  let vd : VoterDistribution := {
    democrats := 0.45,
    republicans := 0.30,
    independents := 0.20,
    undecided := 0.05
  }
  let sd : SupportDistribution := {
    democrats := 0.75,
    republicans := 0.25,
    independents := 0.50,
    undecided := 0.50
  }
  calculateTotalSupport vd sd = 0.5375 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_support_l2672_267252


namespace NUMINAMATH_CALUDE_group_size_proof_l2672_267237

theorem group_size_proof (n : ℕ) (f m : ℕ) : 
  f = 8 → 
  m + f = n → 
  (n - f : ℚ) / n - (n - m : ℚ) / n = 36 / 100 → 
  n = 25 := by
sorry

end NUMINAMATH_CALUDE_group_size_proof_l2672_267237


namespace NUMINAMATH_CALUDE_parabola_minimum_point_l2672_267207

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points A, B, C
def A : ℝ × ℝ := (-1, -3)
def B : ℝ × ℝ := (4, 2)
def C : ℝ × ℝ := (0, 2)

-- Define the theorem
theorem parabola_minimum_point (a b c : ℝ) :
  ∃ (m n : ℝ),
    -- The parabola passes through points A, B, C
    parabola a b c A.1 = A.2 ∧
    parabola a b c B.1 = B.2 ∧
    parabola a b c C.1 = C.2 ∧
    -- P(m, n) is on the axis of symmetry
    m = -b / (2 * a) ∧
    -- P(m, n) minimizes PA + PC
    ∀ (x y : ℝ), x = m → parabola a b c x = y →
      (Real.sqrt ((x - A.1)^2 + (y - A.2)^2) +
       Real.sqrt ((x - C.1)^2 + (y - C.2)^2)) ≥
      (Real.sqrt ((m - A.1)^2 + (n - A.2)^2) +
       Real.sqrt ((m - C.1)^2 + (n - C.2)^2)) →
    -- The y-coordinate of P is 0
    n = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_minimum_point_l2672_267207


namespace NUMINAMATH_CALUDE_exists_line_with_perpendicular_chord_l2672_267298

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + 2*y^2 = 8

-- Define the line l
def l (x y m : ℝ) : Prop := y = x + m

-- Define the condition for A and B being on the ellipse C and line l
def on_ellipse_and_line (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ m ∧ l x₂ y₂ m

-- Define the condition for AB being perpendicular to OA and OB
def perpendicular_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem exists_line_with_perpendicular_chord :
  ∃ m : ℝ, m = 4 * Real.sqrt 3 / 3 ∨ m = -4 * Real.sqrt 3 / 3 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    on_ellipse_and_line x₁ y₁ x₂ y₂ m ∧
    perpendicular_chord x₁ y₁ x₂ y₂ :=
  sorry

end NUMINAMATH_CALUDE_exists_line_with_perpendicular_chord_l2672_267298


namespace NUMINAMATH_CALUDE_min_value_fraction_l2672_267249

theorem min_value_fraction (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ m : ℝ, m = -1 - Real.sqrt 2 ∧ ∀ z, z = (2*x*y)/(x+y+1) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2672_267249


namespace NUMINAMATH_CALUDE_inclination_angle_sqrt3x_plus_y_minus2_l2672_267264

/-- The inclination angle of a line given by the equation √3x + y - 2 = 0 is 120°. -/
theorem inclination_angle_sqrt3x_plus_y_minus2 :
  let line : ℝ → ℝ → Prop := λ x y ↦ Real.sqrt 3 * x + y - 2 = 0
  ∃ α : ℝ, α = 120 * (π / 180) ∧ 
    ∀ x y : ℝ, line x y → Real.tan α = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_sqrt3x_plus_y_minus2_l2672_267264


namespace NUMINAMATH_CALUDE_simplify_fraction_l2672_267212

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2672_267212


namespace NUMINAMATH_CALUDE_cube_root_8000_simplification_l2672_267244

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3) = 8000^(1/3) ∧
                a = 20 ∧ b = 1 ∧
                ∀ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3) = 8000^(1/3) → d ≥ b :=
by sorry

end NUMINAMATH_CALUDE_cube_root_8000_simplification_l2672_267244


namespace NUMINAMATH_CALUDE_symmetric_about_origin_l2672_267247

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g: ℝ → ℝ is even if g(-x) = g(x) for all x ∈ ℝ -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- A function v: ℝ → ℝ is symmetric about the origin if v(-x) = -v(x) for all x ∈ ℝ -/
def SymmetricAboutOrigin (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

/-- Main theorem: If f is odd and g is even, then v(x) = f(x)|g(x)| is symmetric about the origin -/
theorem symmetric_about_origin (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  SymmetricAboutOrigin (fun x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_about_origin_l2672_267247


namespace NUMINAMATH_CALUDE_ball_count_proof_l2672_267202

theorem ball_count_proof (a : ℕ) (h1 : a > 0) (h2 : 3 ≤ a) :
  (3 : ℝ) / a = 1/4 → a = 12 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l2672_267202


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2672_267282

/-- The set of solutions to the equation z + y² + x³ = xyz with x = gcd(y, z) -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {s | s.1 > 0 ∧ s.2.1 > 0 ∧ s.2.2 > 0 ∧
       s.2.2 + s.2.1^2 + s.1^3 = s.1 * s.2.1 * s.2.2 ∧
       s.1 = Nat.gcd s.2.1 s.2.2}

theorem solution_set_characterization :
  SolutionSet = {(1, 2, 5), (1, 3, 5), (2, 2, 4), (2, 6, 4)} := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2672_267282


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2672_267213

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the square perimeter condition
def square_perimeter_condition (a b : ℝ) : Prop := 
  ∃ (c : ℝ), a^2 = b^2 + c^2 ∧ 4 * a = 4 * Real.sqrt 2 ∧ b = c

-- Define the line l
def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the symmetric point D
def symmetric_point (m : ℝ) (x y : ℝ) : Prop := x = 0 ∧ y = -m

-- Define the condition for D being inside the circle with EF as diameter
def inside_circle_condition (m : ℝ) : Prop :=
  ∀ k : ℝ, (m * Real.sqrt (4 * k^2 + 1))^2 < 2 * (1 + k^2) * (2 * k^2 + 1 - m^2)

-- Main theorem
theorem ellipse_m_range :
  ∀ a b m : ℝ,
  a > b ∧ b > 0 ∧ m > 0 ∧
  square_perimeter_condition a b ∧
  inside_circle_condition m →
  0 < m ∧ m < Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2672_267213


namespace NUMINAMATH_CALUDE_plane_line_perpendicular_parallel_perpendicular_parallel_transitive_l2672_267267

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (linePerpendicular : Line → Plane → Prop)

-- Define distinct planes
variable (α β γ : Plane)
variable (distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Define a line
variable (l : Line)

theorem plane_line_perpendicular_parallel 
  (h1 : linePerpendicular l α) 
  (h2 : linePerpendicular l β) : 
  parallel α β := by sorry

theorem perpendicular_parallel_transitive 
  (h1 : perpendicular α γ) 
  (h2 : parallel β γ) : 
  perpendicular α β := by sorry

end NUMINAMATH_CALUDE_plane_line_perpendicular_parallel_perpendicular_parallel_transitive_l2672_267267


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2672_267260

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2672_267260


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2672_267250

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := fun x ↦ (x + 2) * (x - 3)
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2672_267250


namespace NUMINAMATH_CALUDE_badminton_players_count_l2672_267226

/-- A sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ
  tennis_le_total : tennis ≤ total
  both_le_tennis : both ≤ tennis
  neither_le_total : neither ≤ total

/-- The number of members who play badminton in the sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total - club.tennis + club.both - club.neither

/-- Theorem stating the number of badminton players in the specific club scenario -/
theorem badminton_players_count (club : SportsClub) 
  (h_total : club.total = 30)
  (h_tennis : club.tennis = 19)
  (h_both : club.both = 8)
  (h_neither : club.neither = 2) :
  badminton_players club = 17 := by
  sorry

end NUMINAMATH_CALUDE_badminton_players_count_l2672_267226


namespace NUMINAMATH_CALUDE_simplify_expression_l2672_267268

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2672_267268


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l2672_267242

def f (x : ℕ) : ℕ := 3 * x + 2

def iterate (n : ℕ) (f : ℕ → ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate n f x)

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ, (1988 : ℕ) ∣ (iterate 100 f m) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l2672_267242


namespace NUMINAMATH_CALUDE_division_to_ratio_l2672_267241

theorem division_to_ratio (a b : ℝ) (h : a / b = 0.4) : a / b = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_division_to_ratio_l2672_267241


namespace NUMINAMATH_CALUDE_integer_between_sqrt27_and_7_l2672_267214

theorem integer_between_sqrt27_and_7 (x : ℤ) :
  (Real.sqrt 27 < x) ∧ (x < 7) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt27_and_7_l2672_267214


namespace NUMINAMATH_CALUDE_linear_equation_properties_l2672_267216

-- Define the linear equation
def linear_equation (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_equation_properties :
  ∃ (k b : ℝ),
    (linear_equation k b (-3) = -9) ∧
    (linear_equation k b 0 = -3) ∧
    (k = 2 ∧ b = -3) ∧
    (∀ x, linear_equation k b x ≥ 0 → x ≥ 1.5) ∧
    (∀ x, -1 ≤ x ∧ x < 2 → -5 ≤ linear_equation k b x ∧ linear_equation k b x < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_equation_properties_l2672_267216


namespace NUMINAMATH_CALUDE_hash_difference_l2672_267222

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 4 2 - hash 2 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l2672_267222


namespace NUMINAMATH_CALUDE_remaining_money_proof_l2672_267263

def calculate_remaining_money (initial_amount apples_price milk_price oranges_price candy_price eggs_price apples_discount milk_discount : ℚ) : ℚ :=
  let discounted_apples_price := apples_price * (1 - apples_discount)
  let discounted_milk_price := milk_price * (1 - milk_discount)
  let total_spent := discounted_apples_price + discounted_milk_price + oranges_price + candy_price + eggs_price
  initial_amount - total_spent

theorem remaining_money_proof :
  calculate_remaining_money 95 25 8 14 6 12 (15/100) (10/100) = 6891/200 :=
by sorry

end NUMINAMATH_CALUDE_remaining_money_proof_l2672_267263


namespace NUMINAMATH_CALUDE_harmonic_sum_equals_one_third_l2672_267299

-- Define the harmonic number sequence
def H : ℕ → ℚ
  | 0 => 0
  | n + 1 => H n + 1 / (n + 1)

-- Define the summand of the series
def summand (n : ℕ) : ℚ := 1 / ((n + 2 : ℚ) * H (n + 1) * H (n + 2))

-- State the theorem
theorem harmonic_sum_equals_one_third :
  ∑' n, summand n = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_harmonic_sum_equals_one_third_l2672_267299
