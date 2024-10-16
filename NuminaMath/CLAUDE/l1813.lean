import Mathlib

namespace NUMINAMATH_CALUDE_pants_price_problem_l1813_181338

theorem pants_price_problem (total_cost belt_price pants_price : ℝ) : 
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = belt_price + pants_price →
  pants_price = 34.00 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_problem_l1813_181338


namespace NUMINAMATH_CALUDE_expression_value_l1813_181305

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 2 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1813_181305


namespace NUMINAMATH_CALUDE_log_sum_equality_l1813_181393

theorem log_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1813_181393


namespace NUMINAMATH_CALUDE_tart_base_flour_calculation_l1813_181351

theorem tart_base_flour_calculation (original_bases : ℕ) (original_flour : ℚ) 
  (new_bases : ℕ) (new_flour : ℚ) : 
  original_bases = 40 → 
  original_flour = 1/8 → 
  new_bases = 25 → 
  original_bases * original_flour = new_bases * new_flour → 
  new_flour = 1/5 := by
sorry

end NUMINAMATH_CALUDE_tart_base_flour_calculation_l1813_181351


namespace NUMINAMATH_CALUDE_paco_cookie_difference_l1813_181373

/-- The number of more salty cookies than sweet cookies eaten by Paco -/
def cookies_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) : ℕ :=
  eaten_salty - eaten_sweet

/-- Theorem stating that Paco ate 13 more salty cookies than sweet cookies -/
theorem paco_cookie_difference :
  cookies_difference 40 25 15 28 = 13 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookie_difference_l1813_181373


namespace NUMINAMATH_CALUDE_probability_product_greater_than_five_l1813_181302

def S : Finset ℕ := {1, 2, 3, 4, 5}

def pairs : Finset (ℕ × ℕ) := S.product S |>.filter (λ (a, b) => a < b)

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (λ (a, b) => a * b > 5)

theorem probability_product_greater_than_five :
  (valid_pairs.card : ℚ) / pairs.card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_product_greater_than_five_l1813_181302


namespace NUMINAMATH_CALUDE_log_product_l1813_181347

theorem log_product (x y : ℝ) (h1 : Real.log (x / 2) = 0.5) (h2 : Real.log (y / 5) = 0.1) :
  Real.log (x * y) = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_log_product_l1813_181347


namespace NUMINAMATH_CALUDE_usual_time_calculation_l1813_181343

/-- Proves that given a constant distance and the fact that at 60% of usual speed 
    it takes 35 minutes more, the usual time to cover the distance is 52.5 minutes. -/
theorem usual_time_calculation (distance : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) 
    (h2 : usual_time > 0)
    (h3 : distance = usual_speed * usual_time)
    (h4 : distance = (0.6 * usual_speed) * (usual_time + 35/60)) :
  usual_time = 52.5 / 60 := by
sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l1813_181343


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l1813_181355

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l1813_181355


namespace NUMINAMATH_CALUDE_max_m_F_theorem_l1813_181395

/-- The maximum value of m(F) for subsets F of {1, ..., 2n} with n elements -/
def max_m_F (n : ℕ) : ℕ :=
  if n = 2 ∨ n = 3 then 12
  else if n = 4 then 24
  else if n % 2 = 1 then 3 * (n + 1)
  else 3 * (n + 2)

/-- The theorem stating the maximum value of m(F) -/
theorem max_m_F_theorem (n : ℕ) (h : n ≥ 2) :
  ∀ (F : Finset ℕ),
    F ⊆ Finset.range (2 * n + 1) →
    F.card = n →
    (∀ (x y : ℕ), x ∈ F → y ∈ F → x ≠ y → Nat.lcm x y ≥ max_m_F n) :=
by sorry

end NUMINAMATH_CALUDE_max_m_F_theorem_l1813_181395


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l1813_181321

/-- Given ticket information, prove the cost of a children's ticket -/
theorem childrens_ticket_cost 
  (adult_ticket_cost : ℝ) 
  (total_tickets : ℕ) 
  (total_cost : ℝ) 
  (childrens_tickets : ℕ) 
  (h1 : adult_ticket_cost = 5.50)
  (h2 : total_tickets = 21)
  (h3 : total_cost = 83.50)
  (h4 : childrens_tickets = 16) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost * childrens_tickets + 
    adult_ticket_cost * (total_tickets - childrens_tickets) = total_cost ∧ 
    childrens_ticket_cost = 3.50 :=
by sorry

end NUMINAMATH_CALUDE_childrens_ticket_cost_l1813_181321


namespace NUMINAMATH_CALUDE_unique_solution_system_l1813_181364

theorem unique_solution_system (m : ℝ) : ∃! (x y : ℝ), 
  ((m + 1) * x - y - 3 * m = 0) ∧ (4 * x + (m - 1) * y + 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1813_181364


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l1813_181391

theorem abs_inequality_equivalence (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11)) :=
by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l1813_181391


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1813_181367

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that the sum of their y-coordinate and x-coordinate respectively is 5. -/
theorem symmetric_points_sum (a b : ℝ) : 
  (2 : ℝ) = b ∧ a = 3 → a + b = 5 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1813_181367


namespace NUMINAMATH_CALUDE_linda_max_servings_l1813_181375

/-- Represents the recipe and available ingredients for making smoothies -/
structure SmoothieIngredients where
  recipe_bananas : ℕ        -- Bananas needed for 4 servings
  recipe_yogurt : ℕ         -- Cups of yogurt needed for 4 servings
  recipe_honey : ℕ          -- Tablespoons of honey needed for 4 servings
  available_bananas : ℕ     -- Bananas Linda has
  available_yogurt : ℕ      -- Cups of yogurt Linda has
  available_honey : ℕ       -- Tablespoons of honey Linda has

/-- Calculates the maximum number of servings that can be made -/
def max_servings (ingredients : SmoothieIngredients) : ℕ :=
  min
    (ingredients.available_bananas * 4 / ingredients.recipe_bananas)
    (min
      (ingredients.available_yogurt * 4 / ingredients.recipe_yogurt)
      (ingredients.available_honey * 4 / ingredients.recipe_honey))

/-- Theorem stating the maximum number of servings Linda can make -/
theorem linda_max_servings :
  let ingredients := SmoothieIngredients.mk 3 2 1 10 9 4
  max_servings ingredients = 13 := by
  sorry


end NUMINAMATH_CALUDE_linda_max_servings_l1813_181375


namespace NUMINAMATH_CALUDE_three_over_x_equals_one_l1813_181398

theorem three_over_x_equals_one (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_over_x_equals_one_l1813_181398


namespace NUMINAMATH_CALUDE_correct_purchase_and_savings_l1813_181389

/-- Represents the purchase of notebooks by a school -/
structure NotebookPurchase where
  type1 : ℕ  -- number of notebooks of first type
  type2 : ℕ  -- number of notebooks of second type

/-- Calculates the total cost of notebooks without discount -/
def totalCost (purchase : NotebookPurchase) : ℕ :=
  3 * purchase.type1 + 2 * purchase.type2

/-- Calculates the discounted cost of notebooks -/
def discountedCost (purchase : NotebookPurchase) : ℚ :=
  3 * purchase.type1 * (8/10) + 2 * purchase.type2 * (9/10)

/-- Theorem stating the correct purchase and savings -/
theorem correct_purchase_and_savings :
  ∃ (purchase : NotebookPurchase),
    totalCost purchase = 460 ∧
    purchase.type1 = 2 * purchase.type2 + 20 ∧
    purchase.type1 = 120 ∧
    purchase.type2 = 50 ∧
    460 - discountedCost purchase = 82 := by
  sorry


end NUMINAMATH_CALUDE_correct_purchase_and_savings_l1813_181389


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l1813_181392

def f (x : ℝ) := x^3 - 3*x - 3

theorem f_has_root_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l1813_181392


namespace NUMINAMATH_CALUDE_cubic_equation_implies_square_l1813_181397

theorem cubic_equation_implies_square (y : ℝ) : 
  2 * y^3 + 3 * y^2 - 2 * y - 8 = 0 → (5 * y - 2)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_implies_square_l1813_181397


namespace NUMINAMATH_CALUDE_total_trip_time_l1813_181316

def driving_time : ℝ := 5

theorem total_trip_time :
  let traffic_time := 2 * driving_time
  driving_time + traffic_time = 15 := by sorry

end NUMINAMATH_CALUDE_total_trip_time_l1813_181316


namespace NUMINAMATH_CALUDE_simplify_expression_l1813_181390

theorem simplify_expression (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(1 + 3*x) - 5*(1 - x^2) = -5*x^2 + 17*x - 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1813_181390


namespace NUMINAMATH_CALUDE_first_group_size_is_three_l1813_181326

/-- The number of people in the first group -/
def first_group_size : ℕ := 3

/-- The amount of work completed by the first group in 3 days -/
def first_group_work : ℕ := 3

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 3

/-- The number of people in the second group -/
def second_group_size : ℕ := 5

/-- The amount of work completed by the second group in 3 days -/
def second_group_work : ℕ := 5

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 3

theorem first_group_size_is_three :
  first_group_size * first_group_work * second_group_days =
  second_group_size * second_group_work * first_group_days :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_three_l1813_181326


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_one_l1813_181344

theorem smallest_divisor_with_remainder_one (total_boxes : Nat) (h1 : total_boxes = 301) 
  (h2 : total_boxes % 7 = 0) : 
  (∃ x : Nat, x > 0 ∧ total_boxes % x = 1) ∧ 
  (∀ y : Nat, y > 0 ∧ y < 3 → total_boxes % y ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_one_l1813_181344


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1813_181308

def f (x : ℝ) : ℝ := x^3

theorem derivative_f_at_zero : 
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1813_181308


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1813_181341

/-- Definition of the bow tie operation -/
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 5 ⋈ x = 12, then x = 42 -/
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 5 x = 12 → x = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1813_181341


namespace NUMINAMATH_CALUDE_all_f_zero_l1813_181382

-- Define the type for infinite sequences of integers
def T := ℕ → ℤ

-- Define the sum of two sequences
def seqSum (x y : T) : T := λ n => x n + y n

-- Define the property of having exactly one 1 and all others 0
def hasOneOne (x : T) : Prop :=
  ∃ i, x i = 1 ∧ ∀ j, j ≠ i → x j = 0

-- Define the function f with its properties
def isValidF (f : T → ℤ) : Prop :=
  (∀ x, hasOneOne x → f x = 0) ∧
  (∀ x y, f (seqSum x y) = f x + f y)

-- The theorem to prove
theorem all_f_zero (f : T → ℤ) (hf : isValidF f) :
  ∀ x : T, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_f_zero_l1813_181382


namespace NUMINAMATH_CALUDE_product_of_roots_l1813_181369

theorem product_of_roots (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1008 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1813_181369


namespace NUMINAMATH_CALUDE_max_ab_squared_l1813_181313

theorem max_ab_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 6) / 9 ∧ ∀ x y : ℝ, 0 < x → 0 < y → x + y = 2 → x * y^2 ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_ab_squared_l1813_181313


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_seven_sixteenths_l1813_181311

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor where
  pattern_size : Nat
  corner_symmetry : Bool
  dark_tiles_in_quadrant : Nat

/-- Calculates the fraction of dark tiles on the floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  sorry

/-- Theorem stating that for a floor with the given properties, 
    the fraction of dark tiles is 7/16 -/
theorem dark_tile_fraction_is_seven_sixteenths 
  (floor : TiledFloor) 
  (h1 : floor.pattern_size = 8) 
  (h2 : floor.corner_symmetry = true) 
  (h3 : floor.dark_tiles_in_quadrant = 7) : 
  dark_tile_fraction floor = 7 / 16 :=
sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_seven_sixteenths_l1813_181311


namespace NUMINAMATH_CALUDE_max_label_in_sample_l1813_181330

/-- Systematic sampling function that returns the maximum label in the sample -/
def systematic_sample_max (total : ℕ) (sample_size : ℕ) (first_item : ℕ) : ℕ :=
  let interval := total / sample_size
  let position := (first_item % interval) + 1
  (sample_size - (sample_size - position)) * interval + first_item

/-- Theorem stating the maximum label in the systematic sample -/
theorem max_label_in_sample :
  systematic_sample_max 80 5 10 = 74 := by
  sorry

#eval systematic_sample_max 80 5 10

end NUMINAMATH_CALUDE_max_label_in_sample_l1813_181330


namespace NUMINAMATH_CALUDE_inequality_proof_l1813_181378

theorem inequality_proof (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  a * x^2 + b * y^2 ≥ (a * x + b * y)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1813_181378


namespace NUMINAMATH_CALUDE_min_value_theorem_l1813_181354

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (2*a + 3*b + 4*c) * ((a + b)⁻¹ + (b + c)⁻¹ + (c + a)⁻¹) ≥ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1813_181354


namespace NUMINAMATH_CALUDE_min_value_implies_a_l1813_181331

theorem min_value_implies_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x = Real.sin x ^ 2 - 2 * a * Real.sin x + 1) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ 1/2) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1/2) →
  a = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l1813_181331


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1813_181329

/-- Given two parallel vectors a = (2, 5) and b = (x, -2), prove that x = -4/5 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 5]
  let b : Fin 2 → ℝ := ![x, -2]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  x = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1813_181329


namespace NUMINAMATH_CALUDE_one_third_of_ten_y_minus_three_l1813_181385

theorem one_third_of_ten_y_minus_three (y : ℝ) : (1/3) * (10*y - 3) = (10*y)/3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_ten_y_minus_three_l1813_181385


namespace NUMINAMATH_CALUDE_saltwater_volume_proof_l1813_181319

/-- Proves that the initial volume of a saltwater solution is 100 gallons,
    given the conditions of evaporation and addition of salt and water. -/
theorem saltwater_volume_proof :
  ∀ x : ℝ,
  x > 0 →
  let initial_salt := 0.20 * x
  let after_evaporation := 0.75 * x
  let final_volume := after_evaporation + 15
  let final_salt := initial_salt + 10
  (final_salt / final_volume = 1/3) →
  x = 100 :=
by
  sorry

#check saltwater_volume_proof

end NUMINAMATH_CALUDE_saltwater_volume_proof_l1813_181319


namespace NUMINAMATH_CALUDE_jerry_cans_time_l1813_181320

def throw_away_cans (total_cans : ℕ) (cans_per_trip : ℕ) (drain_time : ℕ) (walk_time : ℕ) : ℕ :=
  let trips := (total_cans + cans_per_trip - 1) / cans_per_trip
  let drain_total := trips * drain_time
  let walk_total := trips * (2 * walk_time)
  drain_total + walk_total

theorem jerry_cans_time :
  throw_away_cans 35 3 30 10 = 600 := by
  sorry

end NUMINAMATH_CALUDE_jerry_cans_time_l1813_181320


namespace NUMINAMATH_CALUDE_problem_statement_l1813_181388

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a+2)*(b+2) = 18) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x+2)*(y+2) = 18 ∧ 3/(x+2) + 3/(y+2) < 3/(a+2) + 3/(b+2)) ∨
  (3/(a+2) + 3/(b+2) = Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x+2)*(y+2) = 18 → 2*x + y ≥ 6) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x+2)*(y+2) = 18 → (x+1)*y ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1813_181388


namespace NUMINAMATH_CALUDE_varya_used_discount_l1813_181335

/-- Represents the quantity of items purchased by each girl -/
structure Purchase where
  pens : ℕ
  pencils : ℕ
  notebooks : ℕ

/-- Given the purchases of three girls and the fact that they all paid equally,
    prove that the second girl (Varya) must have used a discount -/
theorem varya_used_discount (p k l : ℚ) (anya varya sasha : Purchase) 
    (h_positive : p > 0 ∧ k > 0 ∧ l > 0)
    (h_anya : anya = ⟨2, 7, 1⟩)
    (h_varya : varya = ⟨5, 6, 5⟩)
    (h_sasha : sasha = ⟨8, 4, 9⟩)
    (h_equal_payment : ∃ (x : ℚ), 
      x = p * anya.pens + k * anya.pencils + l * anya.notebooks ∧
      x = p * varya.pens + k * varya.pencils + l * varya.notebooks ∧
      x = p * sasha.pens + k * sasha.pencils + l * sasha.notebooks) :
  p * varya.pens + k * varya.pencils + l * varya.notebooks < 
  (p * anya.pens + k * anya.pencils + l * anya.notebooks + 
   p * sasha.pens + k * sasha.pencils + l * sasha.notebooks) / 2 := by
  sorry


end NUMINAMATH_CALUDE_varya_used_discount_l1813_181335


namespace NUMINAMATH_CALUDE_F_lower_bound_F_max_value_l1813_181349

/-- The condition that x and y satisfy -/
def satisfies_condition (x y : ℝ) : Prop := x^2 + x*y + y^2 = 1

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := x^3*y + x*y^3

/-- Theorem stating that F(x, y) ≥ -2 for any x and y satisfying the condition -/
theorem F_lower_bound {x y : ℝ} (h : satisfies_condition x y) : F x y ≥ -2 := by
  sorry

/-- Theorem stating that the maximum value of F(x, y) is 1/4 -/
theorem F_max_value : ∃ (x y : ℝ), satisfies_condition x y ∧ F x y = 1/4 ∧ ∀ (a b : ℝ), satisfies_condition a b → F a b ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_F_lower_bound_F_max_value_l1813_181349


namespace NUMINAMATH_CALUDE_appended_digit_problem_l1813_181307

theorem appended_digit_problem (x y : ℕ) : 
  x > 0 → y < 10 → (10 * x + y) - x^2 = 8 * x → 
  ((x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8)) := by sorry

end NUMINAMATH_CALUDE_appended_digit_problem_l1813_181307


namespace NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angles_has_10_sides_l1813_181318

/-- The number of sides of a regular polygon with interior angles measuring 144 degrees -/
def regular_polygon_sides : ℕ :=
  let interior_angle : ℚ := 144
  let n : ℕ := 10
  n

/-- Theorem stating that a regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_with_144_degree_angles_has_10_sides :
  let interior_angle : ℚ := 144
  (interior_angle = (180 * (regular_polygon_sides - 2) : ℚ) / regular_polygon_sides) ∧
  (regular_polygon_sides > 2) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angles_has_10_sides_l1813_181318


namespace NUMINAMATH_CALUDE_f_derivative_l1813_181323

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (2 * x)

theorem f_derivative :
  deriv f = λ x => (2 * x + 2 * x^2) * Real.exp (2 * x) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_l1813_181323


namespace NUMINAMATH_CALUDE_find_k_l1813_181384

-- Define the binary linear equation
def binary_linear_equation (x y t : ℝ) : Prop := 3 * x - 2 * y = t

-- Define the theorem
theorem find_k (m n : ℝ) (h1 : binary_linear_equation m n 5) 
  (h2 : binary_linear_equation (m + 2) (n - 2) k) : k = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l1813_181384


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l1813_181346

/-- Custom operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 - y

/-- Theorem stating that h ⊗ (h ⊗ h) = h for any real h -/
theorem otimes_self_otimes_self (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l1813_181346


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l1813_181309

theorem repeating_decimal_difference : 
  (4 : ℚ) / 11 - (7 : ℚ) / 20 = (3 : ℚ) / 220 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l1813_181309


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1813_181300

-- Define the inequality function
def f (m x : ℝ) : ℝ := m * x^2 + (2*m - 1) * x - 2

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  if m < -1/2 then Set.Ioo (-2) (1/m)
  else if m = -1/2 then ∅
  else if -1/2 < m ∧ m < 0 then Set.Ioo (1/m) (-2)
  else if m = 0 then Set.Ioi (-2)
  else Set.union (Set.Iio (-2)) (Set.Ioi (1/m))

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  {x : ℝ | f m x > 0} = solution_set m :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1813_181300


namespace NUMINAMATH_CALUDE_unanswered_questions_count_l1813_181381

/-- Represents the scoring system for AHSME competition --/
structure ScoringSystem where
  correct : Int
  incorrect : Int
  unanswered : Int

/-- Represents the AHSME competition --/
structure AHSMECompetition where
  new_scoring : ScoringSystem
  old_scoring : ScoringSystem
  total_questions : Nat
  new_score : Int
  old_score : Int

/-- Theorem stating that the number of unanswered questions is 9 --/
theorem unanswered_questions_count (comp : AHSMECompetition)
  (h_new_scoring : comp.new_scoring = { correct := 5, incorrect := 0, unanswered := 2 })
  (h_old_scoring : comp.old_scoring = { correct := 4, incorrect := -1, unanswered := 0 })
  (h_old_base : comp.old_score - 30 = 4 * (comp.new_score / 5) - (comp.total_questions - (comp.new_score / 5) - 9))
  (h_total : comp.total_questions = 30)
  (h_new_score : comp.new_score = 93)
  (h_old_score : comp.old_score = 84) :
  ∃ (correct incorrect : Nat), 
    correct + incorrect + 9 = comp.total_questions ∧
    5 * correct + 2 * 9 = comp.new_score ∧
    4 * correct - incorrect = comp.old_score - 30 :=
by sorry


end NUMINAMATH_CALUDE_unanswered_questions_count_l1813_181381


namespace NUMINAMATH_CALUDE_sports_club_overlap_l1813_181312

theorem sports_club_overlap (N B T BT Neither : ℕ) : 
  N = 35 →
  B = 15 →
  T = 18 →
  Neither = 5 →
  B + T - BT = N - Neither →
  BT = 3 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l1813_181312


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1813_181350

theorem quadratic_inequality_condition (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 0 ∧ ∃ m₀ > 0, ¬(∀ x : ℝ, x^2 - 2*x + m₀ > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1813_181350


namespace NUMINAMATH_CALUDE_darias_remaining_balance_l1813_181314

/-- Calculates the remaining amount owed on a credit card after an initial payment --/
def remaining_balance (saved : ℕ) (couch_price : ℕ) (table_price : ℕ) (lamp_price : ℕ) : ℕ :=
  (couch_price + table_price + lamp_price) - saved

/-- Theorem stating that Daria's remaining balance is $400 --/
theorem darias_remaining_balance :
  remaining_balance 500 750 100 50 = 400 := by
  sorry

end NUMINAMATH_CALUDE_darias_remaining_balance_l1813_181314


namespace NUMINAMATH_CALUDE_sphere_radius_proof_l1813_181328

theorem sphere_radius_proof (a b c : ℝ) : 
  (a + b + c = 40) →
  (2 * a * b + 2 * b * c + 2 * c * a = 512) →
  (∃ r : ℝ, r^2 = 130 ∧ r^2 * 4 = a^2 + b^2 + c^2) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_proof_l1813_181328


namespace NUMINAMATH_CALUDE_point_B_coordinates_l1813_181337

/-- Given points A and C, and the relation between vectors AB and BC, 
    prove that the coordinates of point B are (-2, 5/3) -/
theorem point_B_coordinates 
  (A B C : ℝ × ℝ) 
  (hA : A = (2, 3)) 
  (hC : C = (0, 1)) 
  (h_vec : B - A = -2 • (C - B)) : 
  B = (-2, 5/3) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l1813_181337


namespace NUMINAMATH_CALUDE_locus_of_centers_l1813_181365

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define external tangency to C₁
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define internal tangency to C₂
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 1)^2 + b^2 = (3 - r)^2

theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  4 * a^2 + 4 * b^2 - 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1813_181365


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1813_181379

/-- Given a line segment with one endpoint at (10, 4) and midpoint at (4, -8),
    the sum of the coordinates of the other endpoint is -22. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (4 = (x + 10) / 2) → 
    (-8 = (y + 4) / 2) → 
    x + y = -22 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1813_181379


namespace NUMINAMATH_CALUDE_sum_of_100th_row_general_row_sum_formula_l1813_181317

/-- Represents the sum of numbers in the nth row of the triangular array -/
def rowSum (n : ℕ) : ℕ :=
  2^n - 3 * (n - 1)

/-- The triangular array is defined with 0, 1, 2, 3, ... along the sides,
    and interior numbers are obtained by adding the two adjacent numbers
    in the previous row and adding 1 to each sum. -/
axiom array_definition : True

theorem sum_of_100th_row :
  rowSum 100 = 2^100 - 297 :=
by sorry

theorem general_row_sum_formula (n : ℕ) (h : n > 0) :
  rowSum n = 2^n - 3 * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_100th_row_general_row_sum_formula_l1813_181317


namespace NUMINAMATH_CALUDE_evaluate_expression_l1813_181325

theorem evaluate_expression (a : ℝ) (h : a = 2) : (5 * a^2 - 13 * a + 4) * (2 * a - 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1813_181325


namespace NUMINAMATH_CALUDE_cloth_sale_total_price_l1813_181310

-- Define the parameters of the problem
def quantity : ℕ := 80
def profit_per_meter : ℕ := 7
def cost_price_per_meter : ℕ := 118

-- Define the theorem
theorem cloth_sale_total_price :
  (quantity * (cost_price_per_meter + profit_per_meter)) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_total_price_l1813_181310


namespace NUMINAMATH_CALUDE_monotonic_cubic_range_l1813_181332

/-- A cubic function parameterized by b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 3*x - 5

/-- The derivative of f with respect to x -/
def f_deriv (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*b*x + 3

theorem monotonic_cubic_range (b : ℝ) :
  (∀ x : ℝ, Monotone (f b)) ↔ b ∈ Set.Icc (-3) 3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_range_l1813_181332


namespace NUMINAMATH_CALUDE_merry_saturday_boxes_l1813_181348

/-- The number of boxes Merry had on Sunday -/
def sunday_boxes : ℕ := 25

/-- The number of apples in each box -/
def apples_per_box : ℕ := 10

/-- The total number of apples sold on Saturday and Sunday -/
def total_apples_sold : ℕ := 720

/-- The number of boxes left after selling -/
def boxes_left : ℕ := 3

/-- The number of boxes Merry had on Saturday -/
def saturday_boxes : ℕ := 69

theorem merry_saturday_boxes :
  saturday_boxes = 69 :=
by sorry

end NUMINAMATH_CALUDE_merry_saturday_boxes_l1813_181348


namespace NUMINAMATH_CALUDE_friendly_integers_in_range_two_not_friendly_l1813_181353

def friendly (a : ℕ) : Prop :=
  ∃ m n : ℕ+, (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem friendly_integers_in_range :
  ∃ S : Finset ℕ, S.card ≥ 500 ∧ ∀ a ∈ S, a ∈ Finset.range 2013 ∧ friendly a :=
sorry

theorem two_not_friendly : ¬ friendly 2 :=
sorry

end NUMINAMATH_CALUDE_friendly_integers_in_range_two_not_friendly_l1813_181353


namespace NUMINAMATH_CALUDE_log_of_geometric_is_arithmetic_l1813_181340

theorem log_of_geometric_is_arithmetic (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_geom : b / a = c / b) : 
  Real.log b - Real.log a = Real.log c - Real.log b :=
sorry

end NUMINAMATH_CALUDE_log_of_geometric_is_arithmetic_l1813_181340


namespace NUMINAMATH_CALUDE_altitude_intersection_location_depends_on_shape_l1813_181352

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define the shape of a triangle
inductive TriangleShape
  | Acute
  | Right
  | Obtuse

-- Define the location of a point relative to a triangle
inductive PointLocation
  | Inside
  | OnVertex
  | Outside

-- Function to determine the shape of a triangle
def determineShape (t : Triangle) : TriangleShape :=
  sorry

-- Function to find the intersection point of altitudes
def altitudeIntersection (t : Triangle) : ℝ × ℝ :=
  sorry

-- Function to determine the location of a point relative to a triangle
def determinePointLocation (t : Triangle) (p : ℝ × ℝ) : PointLocation :=
  sorry

-- Theorem stating that the location of the altitude intersection depends on the triangle shape
theorem altitude_intersection_location_depends_on_shape (t : Triangle) :
  let shape := determineShape t
  let intersection := altitudeIntersection t
  let location := determinePointLocation t intersection
  (shape = TriangleShape.Acute → location = PointLocation.Inside) ∧
  (shape = TriangleShape.Right → location = PointLocation.OnVertex) ∧
  (shape = TriangleShape.Obtuse → location = PointLocation.Outside) :=
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_location_depends_on_shape_l1813_181352


namespace NUMINAMATH_CALUDE_equation_equivalence_product_l1813_181345

theorem equation_equivalence_product (a b x y : ℤ) (m n p q : ℕ) :
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
  ((a^m*x - a^n)*(a^p*y - a^q) = a^5*b^5) →
  m*n*p*q = 2 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_product_l1813_181345


namespace NUMINAMATH_CALUDE_gym_towels_l1813_181383

theorem gym_towels (first_hour : ℕ) (second_hour_increase : ℚ) 
  (third_hour_increase : ℚ) (fourth_hour_increase : ℚ) 
  (total_towels : ℕ) : 
  first_hour = 50 →
  second_hour_increase = 1/5 →
  third_hour_increase = 1/4 →
  fourth_hour_increase = 1/3 →
  total_towels = 285 →
  let second_hour := first_hour + (first_hour * second_hour_increase).floor
  let third_hour := second_hour + (second_hour * third_hour_increase).floor
  let fourth_hour := third_hour + (third_hour * fourth_hour_increase).floor
  first_hour + second_hour + third_hour + fourth_hour = total_towels :=
by sorry

end NUMINAMATH_CALUDE_gym_towels_l1813_181383


namespace NUMINAMATH_CALUDE_probability_diamond_or_ace_in_two_draws_l1813_181333

/-- The probability of at least one of two cards being a diamond or an ace
    when drawn with replacement from a modified deck. -/
theorem probability_diamond_or_ace_in_two_draws :
  let total_cards : ℕ := 54
  let diamond_cards : ℕ := 13
  let ace_cards : ℕ := 4
  let diamond_or_ace_cards : ℕ := diamond_cards + ace_cards
  let prob_not_diamond_or_ace : ℚ := (total_cards - diamond_or_ace_cards) / total_cards
  let prob_at_least_one_diamond_or_ace : ℚ := 1 - prob_not_diamond_or_ace ^ 2
  prob_at_least_one_diamond_or_ace = 368 / 729 :=
by sorry

end NUMINAMATH_CALUDE_probability_diamond_or_ace_in_two_draws_l1813_181333


namespace NUMINAMATH_CALUDE_fraction_equality_l1813_181377

theorem fraction_equality : (2523 - 2428)^2 / 121 = 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1813_181377


namespace NUMINAMATH_CALUDE_zongzi_problem_l1813_181366

-- Define the types of zongzi gift boxes
inductive ZongziType
| RedDate
| EggYolk

-- Define the price and quantity of a zongzi gift box
structure ZongziBox where
  type : ZongziType
  price : ℕ
  quantity : ℕ

-- Define the problem parameters
def total_boxes : ℕ := 8
def max_cost : ℕ := 300
def total_recipients : ℕ := 65

-- Define the conditions of the problem
axiom price_relation : ∀ (rd : ZongziBox) (ey : ZongziBox),
  rd.type = ZongziType.RedDate → ey.type = ZongziType.EggYolk →
  3 * rd.price = 4 * ey.price

axiom combined_cost : ∀ (rd : ZongziBox) (ey : ZongziBox),
  rd.type = ZongziType.RedDate → ey.type = ZongziType.EggYolk →
  rd.price + 2 * ey.price = 100

axiom red_date_quantity : ∀ (rd : ZongziBox),
  rd.type = ZongziType.RedDate → rd.quantity = 10

axiom egg_yolk_quantity : ∀ (ey : ZongziBox),
  ey.type = ZongziType.EggYolk → ey.quantity = 6

-- Define the theorem to be proved
theorem zongzi_problem :
  ∃ (rd : ZongziBox) (ey : ZongziBox) (rd_count ey_count : ℕ),
    rd.type = ZongziType.RedDate ∧
    ey.type = ZongziType.EggYolk ∧
    rd.price = 40 ∧
    ey.price = 30 ∧
    rd_count = 5 ∧
    ey_count = 3 ∧
    rd_count + ey_count = total_boxes ∧
    rd_count * rd.price + ey_count * ey.price < max_cost ∧
    rd_count * rd.quantity + ey_count * ey.quantity ≥ total_recipients :=
  sorry


end NUMINAMATH_CALUDE_zongzi_problem_l1813_181366


namespace NUMINAMATH_CALUDE_system_solution_l1813_181304

theorem system_solution :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0) →
  (3 * x^2 * y^2 + y^4 = 84) →
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1813_181304


namespace NUMINAMATH_CALUDE_cube_root_of_512_l1813_181324

theorem cube_root_of_512 : (512 : ℝ)^(1/3 : ℝ) = 8 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_512_l1813_181324


namespace NUMINAMATH_CALUDE_lamps_per_room_l1813_181356

/-- Given a hotel with 147 lamps and 21 rooms, prove that each room gets 7 lamps. -/
theorem lamps_per_room :
  let total_lamps : ℕ := 147
  let total_rooms : ℕ := 21
  let lamps_per_room : ℕ := total_lamps / total_rooms
  lamps_per_room = 7 := by sorry

end NUMINAMATH_CALUDE_lamps_per_room_l1813_181356


namespace NUMINAMATH_CALUDE_symmetric_angle_ratio_l1813_181336

/-- 
Given a point P(x,y) on the terminal side of an angle θ (excluding the origin), 
where the terminal side of θ is symmetric to the terminal side of a 480° angle 
with respect to the x-axis, prove that xy/(x^2 + y^2) = √3/4.
-/
theorem symmetric_angle_ratio (x y : ℝ) (h1 : x ≠ 0 ∨ y ≠ 0) 
  (h2 : y = Real.sqrt 3 * x) : 
  (x * y) / (x^2 + y^2) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_angle_ratio_l1813_181336


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l1813_181315

-- Define a function to count positive factors
def count_positive_factors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := sorry

theorem greatest_number_with_odd_factors :
  ∀ n : ℕ, n < 150 → count_positive_factors n % 2 = 1 → n ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l1813_181315


namespace NUMINAMATH_CALUDE_video_game_map_width_l1813_181359

/-- Represents the dimensions of a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.length * prism.width * prism.height

theorem video_game_map_width :
  ∀ (prism : RectangularPrism),
    volume prism = 50 →
    prism.length = 5 →
    prism.height = 2 →
    prism.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_map_width_l1813_181359


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l1813_181306

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l1813_181306


namespace NUMINAMATH_CALUDE_lens_focal_length_theorem_l1813_181361

/-- Represents a thin lens with a parallel beam of light falling normally on it. -/
structure ThinLens where
  focal_length : ℝ

/-- Represents a screen that can be placed at different distances from the lens. -/
structure Screen where
  distance : ℝ
  spot_diameter : ℝ

/-- Checks if the spot diameter remains constant when the screen is moved. -/
def constant_spot_diameter (lens : ThinLens) (screen1 screen2 : Screen) : Prop :=
  screen1.spot_diameter = screen2.spot_diameter

/-- Theorem stating the possible focal lengths of the lens given the problem conditions. -/
theorem lens_focal_length_theorem (lens : ThinLens) (screen1 screen2 : Screen) :
  screen1.distance = 80 →
  screen2.distance = 40 →
  constant_spot_diameter lens screen1 screen2 →
  lens.focal_length = 100 ∨ lens.focal_length = 60 :=
sorry

end NUMINAMATH_CALUDE_lens_focal_length_theorem_l1813_181361


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l1813_181396

/-- Proves that the slower speed is 8.4 km/hr given the conditions of the problem -/
theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : faster_speed = 14)
  (h3 : additional_distance = 20)
  : ∃ slower_speed : ℝ,
    slower_speed = 8.4 ∧
    (actual_distance / faster_speed = (actual_distance - additional_distance) / slower_speed) := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l1813_181396


namespace NUMINAMATH_CALUDE_spinach_amount_l1813_181380

/-- The initial amount of raw spinach in ounces -/
def initial_spinach : ℝ := 40

/-- The percentage of initial volume after cooking -/
def cooking_ratio : ℝ := 0.20

/-- The amount of cream cheese in ounces -/
def cream_cheese : ℝ := 6

/-- The amount of eggs in ounces -/
def eggs : ℝ := 4

/-- The total volume of the quiche in ounces -/
def total_volume : ℝ := 18

theorem spinach_amount :
  initial_spinach * cooking_ratio + cream_cheese + eggs = total_volume :=
by sorry

end NUMINAMATH_CALUDE_spinach_amount_l1813_181380


namespace NUMINAMATH_CALUDE_mans_rate_l1813_181374

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 6)
  (h2 : speed_against_stream = 3) :
  (speed_with_stream + speed_against_stream) / 2 = 4.5 := by
  sorry

#check mans_rate

end NUMINAMATH_CALUDE_mans_rate_l1813_181374


namespace NUMINAMATH_CALUDE_tuesday_temperature_l1813_181372

/-- Given the average temperatures for different sets of days and the temperature on Friday,
    prove the temperature on Tuesday. -/
theorem tuesday_temperature
  (avg_tues_wed_thurs : (t + w + th) / 3 = 52)
  (avg_wed_thurs_fri : (w + th + 53) / 3 = 54)
  (fri_temp : ℝ)
  (h_fri_temp : fri_temp = 53) :
  t = 47 :=
by sorry


end NUMINAMATH_CALUDE_tuesday_temperature_l1813_181372


namespace NUMINAMATH_CALUDE_on_time_departure_rate_theorem_l1813_181357

/-- The number of flights that departed late -/
def late_flights : ℕ := 1

/-- The number of initial on-time flights -/
def initial_on_time : ℕ := 3

/-- The number of additional on-time flights needed -/
def additional_on_time : ℕ := 4

/-- The total number of flights -/
def total_flights : ℕ := late_flights + initial_on_time + additional_on_time

/-- The target on-time departure rate as a real number between 0 and 1 -/
def target_rate : ℝ := 0.875

theorem on_time_departure_rate_theorem :
  (initial_on_time + additional_on_time : ℝ) / total_flights > target_rate :=
sorry

end NUMINAMATH_CALUDE_on_time_departure_rate_theorem_l1813_181357


namespace NUMINAMATH_CALUDE_congruence_in_range_l1813_181339

theorem congruence_in_range : 
  ∀ n : ℤ, 10 ≤ n ∧ n ≤ 20 ∧ n ≡ 12345 [ZMOD 7] → n = 11 ∨ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_congruence_in_range_l1813_181339


namespace NUMINAMATH_CALUDE_bf_equals_ce_l1813_181358

-- Define the triangle ABC
variable (A B C : Point)

-- Define D as the foot of the angle bisector from A
def D : Point := sorry

-- Define E as the intersection of circumcircle ABD with AC
def E : Point := sorry

-- Define F as the intersection of circumcircle ADC with AB
def F : Point := sorry

-- Theorem statement
theorem bf_equals_ce : BF = CE := by sorry

end NUMINAMATH_CALUDE_bf_equals_ce_l1813_181358


namespace NUMINAMATH_CALUDE_distance_to_office_l1813_181322

theorem distance_to_office : 
  ∀ (v : ℝ) (d : ℝ),
  (d = v * (1/2)) →  -- Distance in heavy traffic
  (d = (v + 20) * (1/5)) →  -- Distance without traffic
  d = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_office_l1813_181322


namespace NUMINAMATH_CALUDE_game_preparation_time_l1813_181386

/-- The time taken to prepare all games is 150 minutes, given that each game takes 10 minutes to prepare and Andrew prepared 15 games. -/
theorem game_preparation_time : 
  let time_per_game : ℕ := 10
  let total_games : ℕ := 15
  let total_time := time_per_game * total_games
  total_time = 150 := by sorry

end NUMINAMATH_CALUDE_game_preparation_time_l1813_181386


namespace NUMINAMATH_CALUDE_domino_arrangements_equals_binomial_coefficient_l1813_181303

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with width and height -/
structure Domino :=
  (width : ℕ)
  (height : ℕ)

/-- The number of distinct arrangements of dominoes on a grid -/
def distinct_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ :=
  sorry

/-- The binomial coefficient (n choose k) -/
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem domino_arrangements_equals_binomial_coefficient :
  let g : Grid := { width := 5, height := 3 }
  let d : Domino := { width := 2, height := 1 }
  let num_dominoes : ℕ := 3
  distinct_arrangements g d num_dominoes = binomial_coefficient 6 2 :=
by sorry

end NUMINAMATH_CALUDE_domino_arrangements_equals_binomial_coefficient_l1813_181303


namespace NUMINAMATH_CALUDE_complex_norm_product_l1813_181363

theorem complex_norm_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_product_l1813_181363


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_equation_l1813_181360

theorem quadratic_inequality_and_equation (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0) →
  0 ≤ a ∧ a ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_equation_l1813_181360


namespace NUMINAMATH_CALUDE_complex_number_theorem_l1813_181371

theorem complex_number_theorem (z : ℂ) :
  (z^2).im = 0 ∧ Complex.abs (z - Complex.I) = 1 → z = 0 ∨ z = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l1813_181371


namespace NUMINAMATH_CALUDE_smallest_divisible_by_9_11_13_l1813_181301

theorem smallest_divisible_by_9_11_13 : ∃ n : ℕ, n > 0 ∧ 
  9 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 9 ∣ m → 11 ∣ m → 13 ∣ m → n ≤ m :=
by
  use 1287
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_9_11_13_l1813_181301


namespace NUMINAMATH_CALUDE_comparison_proofs_l1813_181387

theorem comparison_proofs :
  (-5 < -2) ∧ (-1/3 > -1/2) ∧ (abs (-5) > 0) := by
  sorry

end NUMINAMATH_CALUDE_comparison_proofs_l1813_181387


namespace NUMINAMATH_CALUDE_fraction_of_rotten_berries_l1813_181376

theorem fraction_of_rotten_berries 
  (total_berries : ℕ) 
  (berries_to_sell : ℕ) 
  (h1 : total_berries = 60) 
  (h2 : berries_to_sell = 20) 
  (h3 : berries_to_sell * 2 ≤ total_berries) :
  (total_berries - berries_to_sell * 2 : ℚ) / total_berries = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_rotten_berries_l1813_181376


namespace NUMINAMATH_CALUDE_russian_dolls_discount_l1813_181327

theorem russian_dolls_discount (original_price : ℝ) (original_quantity : ℕ) (discount_rate : ℝ) :
  original_price = 4 →
  original_quantity = 15 →
  discount_rate = 0.2 →
  ⌊(original_price * original_quantity) / (original_price * (1 - discount_rate))⌋ = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_russian_dolls_discount_l1813_181327


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l1813_181334

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l1813_181334


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1813_181362

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 3rd, 5th, and 7th terms of an arithmetic sequence
    where the sum of the 2nd and 8th terms is 10. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 10) : 
  a 3 + a 5 + a 7 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1813_181362


namespace NUMINAMATH_CALUDE_wheel_radii_theorem_l1813_181394

/-- Given two wheels A and B with radii R and r respectively, 
    if the ratio of their rotational speeds is 4:5 and 
    the distance between their centers is 9, 
    then R = 2.5 and r = 2. -/
theorem wheel_radii_theorem (R r : ℝ) : 
  (4 : ℝ) / 5 = 1200 / 1500 →  -- ratio of rotational speeds
  2 * (R + r) = 9 →            -- distance between centers
  R = 2.5 ∧ r = 2 := by
  sorry


end NUMINAMATH_CALUDE_wheel_radii_theorem_l1813_181394


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_reciprocal_squares_l1813_181342

theorem cubic_roots_sum_of_reciprocal_squares :
  ∀ a b c : ℝ,
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_reciprocal_squares_l1813_181342


namespace NUMINAMATH_CALUDE_second_ball_red_probability_l1813_181399

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4

def prob_second_red_given_first_red : ℚ := 5/9

theorem second_ball_red_probability :
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) /
  ((red_balls : ℚ) / total_balls) = prob_second_red_given_first_red :=
by sorry

end NUMINAMATH_CALUDE_second_ball_red_probability_l1813_181399


namespace NUMINAMATH_CALUDE_coin_value_is_70_rupees_l1813_181368

/-- Calculates the total value in rupees given the number of coins and their values -/
def total_value_in_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let value_20_paise := coins_20_paise * 20
  let value_25_paise := coins_25_paise * 25
  let total_paise := value_20_paise + value_25_paise
  total_paise / 100

/-- Proves that the total value of the given coins is 70 rupees -/
theorem coin_value_is_70_rupees :
  total_value_in_rupees 324 220 = 70 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_is_70_rupees_l1813_181368


namespace NUMINAMATH_CALUDE_caravan_camel_count_l1813_181370

/-- Represents the number of camels in the caravan -/
def num_camels : ℕ := 6

/-- Represents the number of hens in the caravan -/
def num_hens : ℕ := 60

/-- Represents the number of goats in the caravan -/
def num_goats : ℕ := 35

/-- Represents the number of keepers in the caravan -/
def num_keepers : ℕ := 10

/-- Represents the difference between the total number of feet and heads -/
def feet_head_difference : ℕ := 193

theorem caravan_camel_count : 
  (2 * num_hens + 4 * num_goats + 4 * num_camels + 2 * num_keepers) = 
  (num_hens + num_goats + num_camels + num_keepers + feet_head_difference) := by
  sorry

end NUMINAMATH_CALUDE_caravan_camel_count_l1813_181370
