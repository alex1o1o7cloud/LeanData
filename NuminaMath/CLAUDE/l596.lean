import Mathlib

namespace NUMINAMATH_CALUDE_class_ratios_l596_59649

theorem class_ratios (male_students female_students : ℕ) 
  (h1 : male_students = 30) 
  (h2 : female_students = 24) : 
  (female_students : ℚ) / male_students = 4/5 ∧ 
  (male_students : ℚ) / (male_students + female_students) = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_class_ratios_l596_59649


namespace NUMINAMATH_CALUDE_root_product_equation_l596_59611

theorem root_product_equation (p q : ℝ) (α β γ δ : ℂ) : 
  (x^2 + p*x + 1 = (x - α) * (x - β)) → 
  (x^2 + q*x + 1 = (x - γ) * (x - δ)) → 
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equation_l596_59611


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l596_59635

theorem sufficient_but_not_necessary (p q : Prop) :
  -- Part 1: Sufficient condition
  ((p ∧ q) → ¬(¬p)) ∧
  -- Part 2: Not necessary condition
  ∃ (r : Prop), (¬(¬r) ∧ ¬(r ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l596_59635


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l596_59668

theorem smallest_k_for_64_power_gt_4_16 :
  ∀ k : ℕ, (64 : ℝ) ^ k > (4 : ℝ) ^ 16 ↔ k ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l596_59668


namespace NUMINAMATH_CALUDE_equation_solution_l596_59652

theorem equation_solution : ∃! x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l596_59652


namespace NUMINAMATH_CALUDE_tree_distance_l596_59638

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 10) (h2 : d = 100) :
  let tree_spacing := d / 5
  tree_spacing * (n - 1) = 180 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l596_59638


namespace NUMINAMATH_CALUDE_twenty_customers_without_fish_l596_59632

/-- Represents the fish market scenario -/
structure FishMarket where
  total_customers : ℕ
  num_tuna : ℕ
  tuna_weight : ℕ
  customer_request : ℕ

/-- Calculates the number of customers who will go home without fish -/
def customers_without_fish (market : FishMarket) : ℕ :=
  market.total_customers - (market.num_tuna * market.tuna_weight / market.customer_request)

/-- Theorem stating that in the given scenario, 20 customers will go home without fish -/
theorem twenty_customers_without_fish :
  let market : FishMarket := {
    total_customers := 100,
    num_tuna := 10,
    tuna_weight := 200,
    customer_request := 25
  }
  customers_without_fish market = 20 := by sorry

end NUMINAMATH_CALUDE_twenty_customers_without_fish_l596_59632


namespace NUMINAMATH_CALUDE_count_multiples_of_five_l596_59634

def d₁ (a : ℕ) : ℕ := a^2 + 2^a + a * 2^((a + 1)/2)
def d₂ (a : ℕ) : ℕ := a^2 + 2^a - a * 2^((a + 1)/2)

theorem count_multiples_of_five :
  ∃ (S : Finset ℕ), S.card = 101 ∧
    (∀ a ∈ S, 1 ≤ a ∧ a ≤ 251 ∧ (d₁ a * d₂ a) % 5 = 0) ∧
    (∀ a : ℕ, 1 ≤ a ∧ a ≤ 251 ∧ (d₁ a * d₂ a) % 5 = 0 → a ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_count_multiples_of_five_l596_59634


namespace NUMINAMATH_CALUDE_expression_value_l596_59678

theorem expression_value : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l596_59678


namespace NUMINAMATH_CALUDE_cubic_function_properties_l596_59657

/-- A cubic function with a maximum at x = -1 and a minimum at x = 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c x ≤ f a b c (-1)) ∧
  (f a b c (-1) = 7) ∧
  (f' a b (-1) = 0) ∧
  (f' a b 3 = 0) →
  a = -3 ∧ b = -9 ∧ c = 2 ∧ f a b c 3 = -25 := by
  sorry

#check cubic_function_properties

end NUMINAMATH_CALUDE_cubic_function_properties_l596_59657


namespace NUMINAMATH_CALUDE_ratio_percentage_difference_l596_59664

theorem ratio_percentage_difference (A B : ℝ) (hA : A > 0) (hB : B > 0) (h_ratio : A / B = 1/8 * 7) :
  (B - A) / A * 100 = 100/7 := by
sorry

end NUMINAMATH_CALUDE_ratio_percentage_difference_l596_59664


namespace NUMINAMATH_CALUDE_calculate_markup_l596_59604

/-- Calculate the markup for an article given its purchase price, overhead percentage, and required net profit. -/
theorem calculate_markup (purchase_price overhead_percentage net_profit : ℝ) : 
  purchase_price = 48 → 
  overhead_percentage = 0.20 → 
  net_profit = 12 → 
  let overhead_cost := purchase_price * overhead_percentage
  let total_cost := purchase_price + overhead_cost
  let selling_price := total_cost + net_profit
  let markup := selling_price - purchase_price
  markup = 21.60 := by
sorry


end NUMINAMATH_CALUDE_calculate_markup_l596_59604


namespace NUMINAMATH_CALUDE_tile_cutting_theorem_l596_59630

/-- Represents a rectangular tile -/
structure Tile where
  width : ℝ
  height : ℝ

/-- Represents the arrangement of tiles -/
structure TileArrangement where
  tiles : List Tile
  width : ℝ
  height : ℝ
  tileCount : ℕ

/-- Represents a part of a cut tile -/
structure TilePart where
  width : ℝ
  height : ℝ

theorem tile_cutting_theorem (arrangement : TileArrangement) 
  (h1 : arrangement.width < arrangement.height)
  (h2 : arrangement.tileCount > 0) :
  ∃ (squareParts rectangleParts : List TilePart),
    (∀ t ∈ arrangement.tiles, ∃ p1 p2, p1 ∈ squareParts ∧ p2 ∈ rectangleParts) ∧
    (∃ s, s > 0 ∧ (∀ p ∈ squareParts, p.width * p.height = s^2 / arrangement.tileCount)) ∧
    (∃ w h, w > 0 ∧ h > 0 ∧ w ≠ h ∧ 
      (∀ p ∈ rectangleParts, p.width * p.height = w * h / arrangement.tileCount)) :=
by sorry

end NUMINAMATH_CALUDE_tile_cutting_theorem_l596_59630


namespace NUMINAMATH_CALUDE_complete_square_equation_l596_59675

theorem complete_square_equation (x : ℝ) : 
  (x^2 - 8*x + 15 = 0) ↔ ((x - 4)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equation_l596_59675


namespace NUMINAMATH_CALUDE_max_value_implies_k_l596_59673

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

theorem max_value_implies_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4) →
  k = 3/8 ∨ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_k_l596_59673


namespace NUMINAMATH_CALUDE_fish_speed_problem_l596_59621

/-- Calculates the downstream speed of a fish given its upstream and still water speeds. -/
def fish_downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: A fish with an upstream speed of 35 kmph and a still water speed of 45 kmph 
    has a downstream speed of 55 kmph. -/
theorem fish_speed_problem :
  fish_downstream_speed 35 45 = 55 := by
  sorry

#eval fish_downstream_speed 35 45

end NUMINAMATH_CALUDE_fish_speed_problem_l596_59621


namespace NUMINAMATH_CALUDE_larry_dog_time_l596_59698

/-- The number of minutes Larry spends on his dog each day -/
def time_spent_on_dog (walking_playing_time : ℕ) (feeding_time : ℕ) : ℕ :=
  walking_playing_time * 2 + feeding_time

theorem larry_dog_time :
  let walking_playing_time : ℕ := 30 -- half an hour in minutes
  let feeding_time : ℕ := 12 -- a fifth of an hour in minutes
  time_spent_on_dog walking_playing_time feeding_time = 72 := by
sorry

end NUMINAMATH_CALUDE_larry_dog_time_l596_59698


namespace NUMINAMATH_CALUDE_students_in_canteen_l596_59616

theorem students_in_canteen (total : ℕ) (absent_fraction : ℚ) (classroom_fraction : ℚ) :
  total = 40 →
  absent_fraction = 1 / 10 →
  classroom_fraction = 3 / 4 →
  (total : ℚ) * (1 - absent_fraction) * (1 - classroom_fraction) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_canteen_l596_59616


namespace NUMINAMATH_CALUDE_martin_martina_ages_l596_59631

/-- Martin's age -/
def martin_age : ℕ := 33

/-- Martina's age -/
def martina_age : ℕ := 22

/-- The condition from Martin's statement -/
def martin_condition (x y : ℕ) : Prop :=
  x = 3 * (y - (x - y))

/-- The condition from Martina's statement -/
def martina_condition (x y : ℕ) : Prop :=
  x + (x + (x - y)) = 77

theorem martin_martina_ages :
  martin_condition martin_age martina_age ∧
  martina_condition martin_age martina_age :=
by sorry

end NUMINAMATH_CALUDE_martin_martina_ages_l596_59631


namespace NUMINAMATH_CALUDE_min_value_quadratic_fraction_l596_59656

theorem min_value_quadratic_fraction (x : ℝ) (h : x ≥ 0) :
  (9 * x^2 + 17 * x + 15) / (5 * (x + 2)) ≥ 18 * Real.sqrt 3 / 5 ∧
  ∃ y ≥ 0, (9 * y^2 + 17 * y + 15) / (5 * (y + 2)) = 18 * Real.sqrt 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_fraction_l596_59656


namespace NUMINAMATH_CALUDE_parallelogram_point_D_l596_59688

/-- A point in the complex plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- A parallelogram in the complex plane -/
structure Parallelogram where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint

/-- The given parallelogram ABCD -/
def givenParallelogram : Parallelogram where
  A := { re := 4, im := 1 }
  B := { re := 3, im := 4 }
  C := { re := 5, im := 2 }
  D := { re := 6, im := -1 }

theorem parallelogram_point_D (p : Parallelogram) (h : p = givenParallelogram) :
  p.D.re = 6 ∧ p.D.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_point_D_l596_59688


namespace NUMINAMATH_CALUDE_football_players_count_l596_59699

theorem football_players_count (total : ℕ) (basketball : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 22)
  (h2 : basketball = 13)
  (h3 : neither = 3)
  (h4 : both = 18) :
  total - neither - (basketball - both) = 19 :=
by sorry

end NUMINAMATH_CALUDE_football_players_count_l596_59699


namespace NUMINAMATH_CALUDE_sphere_volume_from_cylinder_and_cone_l596_59617

/-- The volume of a sphere given specific conditions involving a cylinder and cone -/
theorem sphere_volume_from_cylinder_and_cone 
  (r : ℝ) -- radius of the sphere
  (h : ℝ) -- height of the cylinder and cone
  (M : ℝ) -- volume of the cylinder
  (h_eq : h = 2 * r) -- height is twice the radius
  (M_eq : M = π * r^2 * h) -- volume formula for cylinder
  (V_cone : ℝ := (1/3) * π * r^2 * h) -- volume of the cone
  (C : ℝ) -- volume of the sphere
  (vol_eq : M - V_cone = C) -- combined volume equals sphere volume
  : C = (8/3) * π * r^3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cylinder_and_cone_l596_59617


namespace NUMINAMATH_CALUDE_liam_paid_more_than_ellen_l596_59679

-- Define the pizza characteristics
def total_slices : ℕ := 12
def plain_pizza_cost : ℚ := 12
def extra_cheese_cost : ℚ := 3
def extra_cheese_slices : ℕ := total_slices / 3

-- Define what Liam and Ellen ate
def liam_extra_cheese_slices : ℕ := extra_cheese_slices
def liam_plain_slices : ℕ := 4
def ellen_plain_slices : ℕ := total_slices - liam_extra_cheese_slices - liam_plain_slices

-- Calculate total pizza cost
def total_pizza_cost : ℚ := plain_pizza_cost + extra_cheese_cost

-- Calculate cost per slice
def cost_per_slice : ℚ := total_pizza_cost / total_slices

-- Calculate what Liam and Ellen paid
def liam_payment : ℚ := cost_per_slice * (liam_extra_cheese_slices + liam_plain_slices)
def ellen_payment : ℚ := (plain_pizza_cost / total_slices) * ellen_plain_slices

-- Theorem to prove
theorem liam_paid_more_than_ellen : liam_payment - ellen_payment = 6 := by
  sorry

end NUMINAMATH_CALUDE_liam_paid_more_than_ellen_l596_59679


namespace NUMINAMATH_CALUDE_committee_probability_l596_59665

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

def prob_at_least_one_boy_and_girl : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size) / Nat.choose total_members committee_size

theorem committee_probability :
  prob_at_least_one_boy_and_girl = 574287 / 593775 :=
sorry

end NUMINAMATH_CALUDE_committee_probability_l596_59665


namespace NUMINAMATH_CALUDE_quadratic_root_property_l596_59695

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 50 = 0 → a^3 - 51*a = 50 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l596_59695


namespace NUMINAMATH_CALUDE_quartic_root_sum_l596_59671

/-- Given a quartic equation px^4 + qx^3 + rx^2 + sx + t = 0 with roots 4, -3, and 0, 
    and p ≠ 0, prove that (q+r)/p = -13 -/
theorem quartic_root_sum (p q r s t : ℝ) (hp : p ≠ 0) 
  (h1 : p * 4^4 + q * 4^3 + r * 4^2 + s * 4 + t = 0)
  (h2 : p * (-3)^4 + q * (-3)^3 + r * (-3)^2 + s * (-3) + t = 0)
  (h3 : t = 0) : 
  (q + r) / p = -13 := by
  sorry

end NUMINAMATH_CALUDE_quartic_root_sum_l596_59671


namespace NUMINAMATH_CALUDE_largest_d_inequality_d_satisfies_inequality_d_is_largest_l596_59618

theorem largest_d_inequality (d : ℝ) : 
  (d > 0 ∧ 
   ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
   Real.sqrt (x^2 + y^2) + d * |x - y| ≤ Real.sqrt (2 * (x + y))) → 
  d ≤ 1 / Real.sqrt 2 :=
by sorry

theorem d_satisfies_inequality : 
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
  Real.sqrt (x^2 + y^2) + (1 / Real.sqrt 2) * |x - y| ≤ Real.sqrt (2 * (x + y)) :=
by sorry

theorem d_is_largest : 
  ∀ (d : ℝ), d > 1 / Real.sqrt 2 → 
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
  Real.sqrt (x^2 + y^2) + d * |x - y| > Real.sqrt (2 * (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_largest_d_inequality_d_satisfies_inequality_d_is_largest_l596_59618


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l596_59606

theorem weight_of_replaced_person
  (n : ℕ) 
  (avg_increase : ℝ)
  (new_person_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  new_person_weight = 95 →
  new_person_weight - n * avg_increase = 75 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l596_59606


namespace NUMINAMATH_CALUDE_complex_number_theorem_l596_59600

theorem complex_number_theorem (a : ℝ) (z : ℂ) (h1 : z = (a^2 - 1) + (a + 1) * I) 
  (h2 : z.re = 0) : (a + I^2016) / (1 + I) = 1 - I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l596_59600


namespace NUMINAMATH_CALUDE_not_always_parallel_lines_l596_59661

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_lines 
  (l m : Line) (α : Plane) 
  (h1 : parallel_plane_line α l) 
  (h2 : subset m α) : 
  ¬ (∀ l m α, parallel_plane_line α l → subset m α → parallel l m) :=
sorry

end NUMINAMATH_CALUDE_not_always_parallel_lines_l596_59661


namespace NUMINAMATH_CALUDE_thousandth_digit_is_zero_l596_59613

def factorial (n : ℕ) : ℕ := Nat.factorial n

def expression : ℚ := (factorial 13 * factorial 23 + factorial 15 * factorial 17) / 7

theorem thousandth_digit_is_zero :
  ∃ (n : ℕ), n ≥ 1000 ∧ (expression * 10^n).floor % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_thousandth_digit_is_zero_l596_59613


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l596_59683

theorem gcd_lcm_product (a b : ℕ) (ha : a = 100) (hb : b = 120) :
  (Nat.gcd a b) * (Nat.lcm a b) = 12000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l596_59683


namespace NUMINAMATH_CALUDE_cos_equality_problem_l596_59680

theorem cos_equality_problem (m : ℤ) : 
  0 ≤ m ∧ m ≤ 180 → (Real.cos (m * π / 180) = Real.cos (1234 * π / 180) ↔ m = 154) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l596_59680


namespace NUMINAMATH_CALUDE_ratio_simplification_l596_59607

theorem ratio_simplification (a b c : ℝ) (n m p : ℕ+) 
  (h : a^(n : ℕ) / c^(p : ℕ) = 3 / 7 ∧ b^(m : ℕ) / c^(p : ℕ) = 4 / 7) :
  (a^(n : ℕ) + b^(m : ℕ) + c^(p : ℕ)) / c^(p : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_simplification_l596_59607


namespace NUMINAMATH_CALUDE_blocks_per_box_l596_59669

theorem blocks_per_box (total_blocks : ℕ) (num_boxes : ℕ) (h1 : total_blocks = 12) (h2 : num_boxes = 2) :
  total_blocks / num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_box_l596_59669


namespace NUMINAMATH_CALUDE_simplify_and_sum_l596_59626

theorem simplify_and_sum (d : ℝ) (a b c : ℤ) (h : d ≠ 0) :
  (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d + b + c * d^2 →
  a + b + c = 53 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_l596_59626


namespace NUMINAMATH_CALUDE_base_seven_digits_of_2401_l596_59697

theorem base_seven_digits_of_2401 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 2401 ∧ 2401 < 7^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_2401_l596_59697


namespace NUMINAMATH_CALUDE_infinite_sum_equals_one_tenth_l596_59647

/-- The infinite sum of n^2 / (n^6 + 5) from n = 0 to infinity equals 1/10 -/
theorem infinite_sum_equals_one_tenth :
  (∑' n : ℕ, (n^2 : ℝ) / (n^6 + 5)) = 1/10 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_one_tenth_l596_59647


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l596_59640

def A : Set ℕ := {1, 3}
def B : Set ℕ := {0, 3}

theorem union_of_A_and_B : A ∪ B = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l596_59640


namespace NUMINAMATH_CALUDE_mias_test_score_l596_59615

theorem mias_test_score (total_students : ℕ) (initial_average : ℚ) (average_after_ethan : ℚ) (final_average : ℚ) :
  total_students = 20 →
  initial_average = 84 →
  average_after_ethan = 85 →
  final_average = 86 →
  (total_students * final_average - (total_students - 1) * average_after_ethan : ℚ) = 105 := by
  sorry

end NUMINAMATH_CALUDE_mias_test_score_l596_59615


namespace NUMINAMATH_CALUDE_selection_problem_l596_59667

theorem selection_problem (n_sergeants m_soldiers : ℕ) 
  (k_sergeants k_soldiers : ℕ) (factor : ℕ) :
  n_sergeants = 6 →
  m_soldiers = 60 →
  k_sergeants = 2 →
  k_soldiers = 20 →
  factor = 3 →
  (factor * Nat.choose n_sergeants k_sergeants * Nat.choose m_soldiers k_soldiers) = 
  (3 * Nat.choose 6 2 * Nat.choose 60 20) :=
by sorry

end NUMINAMATH_CALUDE_selection_problem_l596_59667


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l596_59650

/-- Represents the number of valid outfit choices given specific clothing items and constraints. -/
theorem valid_outfit_choices : 
  -- Define the number of shirts, pants, and their colors
  let num_shirts : ℕ := 6
  let num_pants : ℕ := 6
  let num_colors : ℕ := 6
  
  -- Define the number of hats
  let num_patterned_hats : ℕ := 6
  let num_solid_hats : ℕ := 6
  let total_hats : ℕ := num_patterned_hats + num_solid_hats
  
  -- Calculate total combinations
  let total_combinations : ℕ := num_shirts * num_pants * total_hats
  
  -- Calculate invalid combinations
  let same_color_combinations : ℕ := num_colors
  let pattern_mismatch_combinations : ℕ := num_patterned_hats * num_shirts * (num_pants - 1)
  
  -- Calculate valid combinations
  let valid_combinations : ℕ := total_combinations - same_color_combinations - pattern_mismatch_combinations
  
  -- Prove that the number of valid outfit choices is 246
  valid_combinations = 246 := by
    sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l596_59650


namespace NUMINAMATH_CALUDE_carmen_daniel_difference_l596_59646

/-- Calculates the difference in miles biked between two cyclists after a given time -/
def miles_difference (carmen_rate daniel_rate time : ℝ) : ℝ :=
  carmen_rate * time - daniel_rate * time

theorem carmen_daniel_difference :
  miles_difference 15 10 3 = 15 := by sorry

end NUMINAMATH_CALUDE_carmen_daniel_difference_l596_59646


namespace NUMINAMATH_CALUDE_min_width_proof_l596_59666

/-- The minimum width of a rectangular area satisfying the given conditions -/
def min_width : ℝ := 5

/-- The length of the rectangular area in terms of its width -/
def length (w : ℝ) : ℝ := w + 15

/-- The area of the rectangular enclosure -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 100 → w ≥ min_width) ∧
  (area min_width ≥ 100) ∧
  (min_width > 0) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l596_59666


namespace NUMINAMATH_CALUDE_two_digit_prime_sum_20180500_prime_l596_59639

theorem two_digit_prime_sum_20180500_prime (n : ℕ) : 
  (10 ≤ n ∧ n < 100) →  -- n is a two-digit number
  Nat.Prime n →         -- n is prime
  Nat.Prime (n + 20180500) → -- n + 20180500 is prime
  n = 61 := by
sorry

end NUMINAMATH_CALUDE_two_digit_prime_sum_20180500_prime_l596_59639


namespace NUMINAMATH_CALUDE_line_intersection_proof_l596_59689

theorem line_intersection_proof :
  ∃! (x y : ℚ), 5 * x - 3 * y = 17 ∧ 8 * x + 2 * y = 22 ∧ x = 50 / 17 ∧ y = -13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_proof_l596_59689


namespace NUMINAMATH_CALUDE_original_group_size_l596_59685

/-- Proves that the original number of men in a group is 36, given the conditions of the work completion times. -/
theorem original_group_size (total_work : ℝ) : 
  (∃ (original_group : ℕ), 
    (original_group : ℝ) / 12 * total_work = total_work ∧
    ((original_group - 6 : ℝ) / 14) * total_work = total_work) →
  ∃ (original_group : ℕ), original_group = 36 := by
sorry

end NUMINAMATH_CALUDE_original_group_size_l596_59685


namespace NUMINAMATH_CALUDE_problem_statement_l596_59662

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 11) :
  c + 1 / b = 5 / 19 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l596_59662


namespace NUMINAMATH_CALUDE_alyssa_pullups_l596_59670

/-- Represents the number of exercises done by a person -/
structure ExerciseCount where
  pushups : ℕ
  crunches : ℕ
  pullups : ℕ

/-- Zachary's exercise count -/
def zachary : ExerciseCount := ⟨44, 17, 23⟩

/-- David's exercise count relative to Zachary's -/
def david : ExerciseCount := ⟨zachary.pushups + 29, zachary.crunches - 13, zachary.pullups + 10⟩

/-- Alyssa's exercise count relative to Zachary's -/
def alyssa : ExerciseCount := ⟨zachary.pushups * 2, zachary.crunches / 2, zachary.pullups - 8⟩

theorem alyssa_pullups : alyssa.pullups = 15 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_pullups_l596_59670


namespace NUMINAMATH_CALUDE_strictly_decreasing_quadratic_function_l596_59690

/-- A function f(x) = kx² - 4x - 8 is strictly decreasing on [4, 16] iff k ∈ (-∞, 1/8] -/
theorem strictly_decreasing_quadratic_function (k : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 16 →
    k * x₂^2 - 4*x₂ - 8 < k * x₁^2 - 4*x₁ - 8) ↔
  k ≤ 1/8 :=
sorry

end NUMINAMATH_CALUDE_strictly_decreasing_quadratic_function_l596_59690


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l596_59610

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x + 2 * y = -1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l596_59610


namespace NUMINAMATH_CALUDE_multiplication_problem_l596_59676

theorem multiplication_problem (A B C D : Nat) : 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A < 10 → B < 10 → C < 10 → D < 10 →
  C * 10 + D = 25 →
  (A * 100 + B * 10 + A) * (C * 10 + D) = C * 1000 + D * 100 + C * 10 + 0 →
  A + B = 2 := by
sorry

end NUMINAMATH_CALUDE_multiplication_problem_l596_59676


namespace NUMINAMATH_CALUDE_leon_order_proof_l596_59642

def toy_organizer_sets : ℕ := 3
def toy_organizer_price : ℚ := 78
def gaming_chair_price : ℚ := 83
def delivery_fee_rate : ℚ := 0.05
def total_payment : ℚ := 420

def gaming_chairs_ordered : ℕ := 2

theorem leon_order_proof :
  ∃ (g : ℕ), 
    (toy_organizer_sets * toy_organizer_price + g * gaming_chair_price) * (1 + delivery_fee_rate) = total_payment ∧
    g = gaming_chairs_ordered :=
by sorry

end NUMINAMATH_CALUDE_leon_order_proof_l596_59642


namespace NUMINAMATH_CALUDE_cookies_theorem_l596_59687

/-- The number of cookies each guest had -/
def cookies_per_guest : ℕ := 2

/-- The number of guests -/
def number_of_guests : ℕ := 5

/-- The total number of cookies prepared -/
def total_cookies : ℕ := cookies_per_guest * number_of_guests

theorem cookies_theorem : total_cookies = 10 := by
  sorry

end NUMINAMATH_CALUDE_cookies_theorem_l596_59687


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l596_59623

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ n : ℕ, n = 990 ∧
    5 ∣ n ∧
    6 ∣ n ∧
    n < 1000 ∧
    ∀ m : ℕ, m < 1000 → 5 ∣ m → 6 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l596_59623


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l596_59660

/-- A function that checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- A function that converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ :=
  sorry

/-- A function that returns the number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ :=
  sorry

theorem smallest_dual_base_palindrome :
  let n := 51  -- 110011₂ in base 10
  (∀ m < n, numDigits m 2 = 6 → isPalindrome m 2 → 
    ∀ b > 2, ¬(numDigits (baseConvert m 2 b) b = 4 ∧ isPalindrome (baseConvert m 2 b) b)) ∧
  numDigits n 2 = 6 ∧
  isPalindrome n 2 ∧
  numDigits (baseConvert n 2 3) 3 = 4 ∧
  isPalindrome (baseConvert n 2 3) 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l596_59660


namespace NUMINAMATH_CALUDE_aluminum_mass_calculation_l596_59651

/-- Given two parts with equal volume but different densities, 
    calculate the mass of one part given the mass difference. -/
theorem aluminum_mass_calculation 
  (ρA ρM : ℝ) -- densities of aluminum and copper
  (Δm : ℝ) -- mass difference
  (h1 : ρA = 2700) -- density of aluminum
  (h2 : ρM = 8900) -- density of copper
  (h3 : Δm = 0.06) -- mass difference in kg
  (h4 : ρM > ρA) -- copper is denser than aluminum
  : ∃ (mA : ℝ), mA = (ρA * Δm) / (ρM - ρA) :=
by sorry

end NUMINAMATH_CALUDE_aluminum_mass_calculation_l596_59651


namespace NUMINAMATH_CALUDE_ornament_profit_theorem_l596_59643

-- Define the cost and selling prices
def costPriceA : ℝ := 2000
def costPriceB : ℝ := 1500
def sellingPriceA : ℝ := 2500
def sellingPriceB : ℝ := 1800

-- Define the total number of ornaments and maximum budget
def totalOrnaments : ℕ := 20
def maxBudget : ℝ := 36000

-- Define the profit function
def profitFunction (x : ℝ) : ℝ := 200 * x + 6000

-- Theorem statement
theorem ornament_profit_theorem :
  -- Condition 1: Cost price difference
  (costPriceA - costPriceB = 500) →
  -- Condition 2: Equal quantity purchased
  (40000 / costPriceA = 30000 / costPriceB) →
  -- Condition 3: Budget constraint
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ totalOrnaments → costPriceA * x + costPriceB * (totalOrnaments - x) ≤ maxBudget) →
  -- Conclusion 1: Correct profit function
  (∀ x : ℝ, profitFunction x = (sellingPriceA - costPriceA) * x + (sellingPriceB - costPriceB) * (totalOrnaments - x)) ∧
  -- Conclusion 2: Maximum profit
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ totalOrnaments ∧ 
    ∀ y : ℝ, 0 ≤ y ∧ y ≤ totalOrnaments → profitFunction x ≥ profitFunction y) ∧
  profitFunction 12 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_ornament_profit_theorem_l596_59643


namespace NUMINAMATH_CALUDE_add_point_four_five_to_fifty_seven_point_two_five_l596_59696

theorem add_point_four_five_to_fifty_seven_point_two_five :
  57.25 + 0.45 = 57.7 := by
  sorry

end NUMINAMATH_CALUDE_add_point_four_five_to_fifty_seven_point_two_five_l596_59696


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l596_59622

/-- Given that p and q vary inversely, prove that when q = 2.8 for p = 500, 
    then q = 1.12 when p = 1250 -/
theorem inverse_variation_problem (p q : ℝ) (h : p * q = 500 * 2.8) :
  p = 1250 → q = 1.12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l596_59622


namespace NUMINAMATH_CALUDE_zero_point_exists_l596_59633

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_point_exists : ∃ c ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_exists_l596_59633


namespace NUMINAMATH_CALUDE_linear_function_properties_l596_59691

def f (x : ℝ) := -2 * x + 4

theorem linear_function_properties :
  (∀ x y, x < y → f x > f y) ∧
  (∀ x y, x < 0 ∧ y < 0 → ¬(f x < 0 ∧ f y < 0)) ∧
  (f 0 ≠ 0 ∨ 4 ≠ 0) ∧
  (∀ x, f x - 4 = -2 * x) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l596_59691


namespace NUMINAMATH_CALUDE_budget_allocation_theorem_l596_59677

def budget_allocation (microphotonics home_electronics food_additives industrial_lubricants astrophysics_degrees : ℝ) : Prop :=
  let total_degrees : ℝ := 360
  let percent_per_degree : ℝ := 100 / total_degrees
  let astrophysics_percent : ℝ := astrophysics_degrees * percent_per_degree
  let known_percent : ℝ := microphotonics + home_electronics + food_additives + industrial_lubricants + astrophysics_percent
  let gmo_percent : ℝ := 100 - known_percent
  gmo_percent = 19

theorem budget_allocation_theorem :
  budget_allocation 14 24 15 8 72 :=
by sorry

end NUMINAMATH_CALUDE_budget_allocation_theorem_l596_59677


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l596_59601

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l596_59601


namespace NUMINAMATH_CALUDE_sandys_age_l596_59686

theorem sandys_age (sandy_age molly_age : ℕ) 
  (age_difference : molly_age = sandy_age + 14)
  (age_ratio : sandy_age * 9 = molly_age * 7) :
  sandy_age = 49 := by
sorry

end NUMINAMATH_CALUDE_sandys_age_l596_59686


namespace NUMINAMATH_CALUDE_quadratic_root_complex_l596_59659

theorem quadratic_root_complex (c d : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (3 - 4 * Complex.I : ℂ) ^ 2 + c * (3 - 4 * Complex.I : ℂ) + d = 0 →
  c = -6 ∧ d = 25 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_complex_l596_59659


namespace NUMINAMATH_CALUDE_distinct_committees_l596_59628

/-- The number of teams in the volleyball league -/
def numTeams : ℕ := 5

/-- The number of players in each team -/
def playersPerTeam : ℕ := 8

/-- The number of committee members selected from the host team -/
def hostCommitteeMembers : ℕ := 4

/-- The number of committee members selected from each non-host team -/
def nonHostCommitteeMembers : ℕ := 1

/-- The total number of distinct tournament committees over one complete rotation -/
def totalCommittees : ℕ := numTeams * (Nat.choose playersPerTeam hostCommitteeMembers) * (playersPerTeam ^ (numTeams - 1))

theorem distinct_committees :
  totalCommittees = 1433600 := by
  sorry

end NUMINAMATH_CALUDE_distinct_committees_l596_59628


namespace NUMINAMATH_CALUDE_coin_flip_probability_l596_59672

theorem coin_flip_probability (n : ℕ) (k : ℕ) (h : n = 4 ∧ k = 3) :
  (2 : ℚ) / (2^n : ℚ) = 1/8 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l596_59672


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l596_59663

theorem least_three_digit_multiple_of_eight : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  n % 8 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ m % 8 = 0 → n ≤ m) ∧ 
  n = 104 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l596_59663


namespace NUMINAMATH_CALUDE_range_of_a_l596_59682

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(B a ⊆ A)) ↔ (1/2 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l596_59682


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l596_59614

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
noncomputable def P (S : Finset Nat) : ℝ := (S.card : ℝ) / (Ω.card : ℝ)

-- State the theorem
theorem conditional_probability_B_given_A : P (A ∩ B) / P A = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l596_59614


namespace NUMINAMATH_CALUDE_inequality_implication_l596_59655

theorem inequality_implication (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x - Real.log y > y - Real.log x → x - y > 1 / x - 1 / y := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l596_59655


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_l596_59681

theorem seeds_per_flowerbed (total_seeds : ℕ) (num_flowerbeds : ℕ) 
  (h1 : total_seeds = 45)
  (h2 : num_flowerbeds = 9)
  (h3 : total_seeds % num_flowerbeds = 0) :
  total_seeds / num_flowerbeds = 5 := by
sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_l596_59681


namespace NUMINAMATH_CALUDE_lamp_cost_theorem_l596_59608

-- Define the prices of lamps
def price_A : ℝ := sorry
def price_B : ℝ := sorry

-- Define the total number of lamps
def total_lamps : ℕ := 200

-- Define the function for total cost
def total_cost (a : ℕ) : ℝ := sorry

-- Theorem statement
theorem lamp_cost_theorem :
  -- Conditions
  (3 * price_A + 5 * price_B = 50) ∧
  (price_A + 3 * price_B = 26) ∧
  (∀ a : ℕ, total_cost a = price_A * a + price_B * (total_lamps - a)) →
  -- Conclusions
  (price_A = 5 ∧ price_B = 7) ∧
  (∀ a : ℕ, total_cost a = -2 * a + 1400) ∧
  (total_cost 80 = 1240) := by
  sorry


end NUMINAMATH_CALUDE_lamp_cost_theorem_l596_59608


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l596_59694

theorem complex_number_quadrant : ∃ (z : ℂ), z = (3 + 4*I)*I ∧ (z.re < 0 ∧ z.im > 0) :=
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l596_59694


namespace NUMINAMATH_CALUDE_first_term_is_0_375_l596_59644

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first 40 terms is 600 -/
  sum_first_40 : (40 : ℝ) / 2 * (2 * a + 39 * d) = 600
  /-- The sum of the next 40 terms (terms 41 to 80) is 1800 -/
  sum_next_40 : (40 : ℝ) / 2 * (2 * (a + 40 * d) + 39 * d) = 1800

/-- The first term of the arithmetic sequence with the given properties is 0.375 -/
theorem first_term_is_0_375 (seq : ArithmeticSequence) : seq.a = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_0_375_l596_59644


namespace NUMINAMATH_CALUDE_airline_capacity_l596_59620

/-- Calculates the number of passengers an airline can accommodate daily --/
theorem airline_capacity
  (num_airplanes : ℕ)
  (rows_per_airplane : ℕ)
  (seats_per_row : ℕ)
  (flights_per_day : ℕ)
  (h1 : num_airplanes = 5)
  (h2 : rows_per_airplane = 20)
  (h3 : seats_per_row = 7)
  (h4 : flights_per_day = 2) :
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day = 7000 :=
by sorry

end NUMINAMATH_CALUDE_airline_capacity_l596_59620


namespace NUMINAMATH_CALUDE_average_problem_l596_59602

theorem average_problem (a b c X Y Z : ℝ) 
  (h1 : (a + b + c) / 3 = 5)
  (h2 : (X + Y + Z) / 3 = 7) :
  ((2*a + 3*X) + (2*b + 3*Y) + (2*c + 3*Z)) / 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l596_59602


namespace NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l596_59645

theorem abs_sum_equals_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l596_59645


namespace NUMINAMATH_CALUDE_min_horses_oxen_solution_l596_59653

theorem min_horses_oxen_solution :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    344 * x - 265 * y = 33 ∧
    ∀ (x' y' : ℕ), x' > 0 → y' > 0 → 344 * x' - 265 * y' = 33 → x' ≥ x ∧ y' ≥ y :=
by
  -- The proof would go here
  sorry

#check min_horses_oxen_solution

end NUMINAMATH_CALUDE_min_horses_oxen_solution_l596_59653


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l596_59692

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l596_59692


namespace NUMINAMATH_CALUDE_inequality_B_is_linear_l596_59612

-- Define a linear inequality in one variable
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x ↔ a * x > b

-- Define the specific inequality from option B
def inequality_B (x : ℝ) : Prop := x / 2 > 0

-- Theorem stating that inequality_B is a linear inequality in one variable
theorem inequality_B_is_linear : is_linear_inequality_one_var inequality_B :=
sorry

end NUMINAMATH_CALUDE_inequality_B_is_linear_l596_59612


namespace NUMINAMATH_CALUDE_calculation_proof_l596_59674

theorem calculation_proof : 
  (Real.sqrt 18) / 3 + |Real.sqrt 2 - 2| + 2023^0 - (-1)^1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l596_59674


namespace NUMINAMATH_CALUDE_sin_160_cos_10_plus_cos_20_sin_10_l596_59624

theorem sin_160_cos_10_plus_cos_20_sin_10 :
  Real.sin (160 * π / 180) * Real.cos (10 * π / 180) +
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_160_cos_10_plus_cos_20_sin_10_l596_59624


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l596_59609

/-- The x-intercept of a line is a point where the line crosses the x-axis (y = 0). -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  (c / a, 0)

/-- The line equation is represented as ax + by = c -/
def line_equation (a b c : ℚ) (x y : ℚ) : Prop :=
  a * x + b * y = c

theorem x_intercept_of_line :
  x_intercept 4 7 28 = (7, 0) ∧
  line_equation 4 7 28 (x_intercept 4 7 28).1 (x_intercept 4 7 28).2 :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l596_59609


namespace NUMINAMATH_CALUDE_eight_balls_distribution_l596_59658

/-- The number of ways to distribute n distinct balls into 3 boxes,
    where box i contains at least i balls -/
def distribution_count (n : ℕ) : ℕ := sorry

/-- Theorem stating that there are 2268 ways to distribute 8 distinct balls
    into 3 boxes numbered 1, 2, and 3, where each box i contains at least i balls -/
theorem eight_balls_distribution : distribution_count 8 = 2268 := by sorry

end NUMINAMATH_CALUDE_eight_balls_distribution_l596_59658


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l596_59629

theorem complex_magnitude_example : Complex.abs (-3 + (8/5)*Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l596_59629


namespace NUMINAMATH_CALUDE_problem_solution_l596_59637

theorem problem_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3) → d = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l596_59637


namespace NUMINAMATH_CALUDE_nagy_birth_and_death_l596_59603

def birth_year : ℕ := 1849
def death_year : ℕ := 1934
def grandchild_birth_year : ℕ := 1932
def num_grandchildren : ℕ := 24

theorem nagy_birth_and_death :
  (∃ (n : ℕ), birth_year = n^2) ∧
  (birth_year ≥ 1834 ∧ birth_year ≤ 1887) ∧
  (death_year - birth_year = 84) ∧
  (grandchild_birth_year - birth_year = 83) ∧
  (num_grandchildren = 24) :=
by sorry

end NUMINAMATH_CALUDE_nagy_birth_and_death_l596_59603


namespace NUMINAMATH_CALUDE_expansion_unique_solution_l596_59648

/-- The number of terms in the expansion of (a+b+c+d+1)^n that include all four variables
    a, b, c, and d, each to some positive power. -/
def num_terms (n : ℕ) : ℕ := Nat.choose n 4

/-- The proposition that n is the unique positive integer such that the expansion of (a+b+c+d+1)^n
    contains exactly 715 terms with all four variables a, b, c, and d each to some positive power. -/
def is_unique_solution (n : ℕ) : Prop :=
  n > 0 ∧ num_terms n = 715 ∧ ∀ m : ℕ, m ≠ n → num_terms m ≠ 715

theorem expansion_unique_solution :
  is_unique_solution 13 :=
sorry

end NUMINAMATH_CALUDE_expansion_unique_solution_l596_59648


namespace NUMINAMATH_CALUDE_polynomial_factorization_l596_59684

theorem polynomial_factorization (x y : ℝ) : x * y^2 - 36 * x = x * (y + 6) * (y - 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l596_59684


namespace NUMINAMATH_CALUDE_greatest_n_for_perfect_square_product_l596_59619

def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem greatest_n_for_perfect_square_product (n : ℕ) : 
  n ≤ 2010 →
  (∀ k : ℕ, k > n → k ≤ 2010 → ¬is_perfect_square ((sum_squares k) * (sum_squares (2*k) - sum_squares k))) →
  is_perfect_square ((sum_squares n) * (sum_squares (2*n) - sum_squares n)) →
  n = 1935 := by sorry

end NUMINAMATH_CALUDE_greatest_n_for_perfect_square_product_l596_59619


namespace NUMINAMATH_CALUDE_box_width_is_twenty_l596_59605

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the properties of cubes filling a box -/
structure CubeFill where
  box : BoxDimensions
  cubeCount : ℕ
  cubeSideLength : ℕ

/-- Theorem stating that a box with given dimensions filled with 56 cubes has a width of 20 inches -/
theorem box_width_is_twenty
  (box : BoxDimensions)
  (fill : CubeFill)
  (h1 : box.length = 35)
  (h2 : box.depth = 10)
  (h3 : fill.box = box)
  (h4 : fill.cubeCount = 56)
  (h5 : fill.cubeSideLength * fill.cubeSideLength * fill.cubeSideLength * fill.cubeCount = box.length * box.width * box.depth)
  (h6 : fill.cubeSideLength ∣ box.length ∧ fill.cubeSideLength ∣ box.width ∧ fill.cubeSideLength ∣ box.depth)
  : box.width = 20 := by
  sorry

#check box_width_is_twenty

end NUMINAMATH_CALUDE_box_width_is_twenty_l596_59605


namespace NUMINAMATH_CALUDE_triangle_altitude_and_area_l596_59654

/-- Triangle with sides a, b, c and altitude h from the vertex opposite side b --/
structure Triangle (a b c : ℝ) where
  h : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_altitude_and_area 
  (t : Triangle 11 13 16) : t.h = 168 / 13 ∧ (1 / 2 : ℝ) * 13 * t.h = 84 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_area_l596_59654


namespace NUMINAMATH_CALUDE_square_of_fraction_equals_4088484_l596_59641

theorem square_of_fraction_equals_4088484 :
  ((2023^2 - 2023) / 2023)^2 = 4088484 := by
  sorry

end NUMINAMATH_CALUDE_square_of_fraction_equals_4088484_l596_59641


namespace NUMINAMATH_CALUDE_paul_lives_on_fifth_story_l596_59625

/-- The number of stories in Paul's apartment building -/
def S : ℕ := sorry

/-- The number of trips Paul makes each day -/
def trips_per_day : ℕ := 3

/-- The height of each story in feet -/
def story_height : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total vertical distance Paul travels in a week in feet -/
def total_vertical_distance : ℕ := 2100

theorem paul_lives_on_fifth_story :
  S * story_height * trips_per_day * 2 * days_in_week = total_vertical_distance →
  S = 5 := by
  sorry

end NUMINAMATH_CALUDE_paul_lives_on_fifth_story_l596_59625


namespace NUMINAMATH_CALUDE_tree_height_after_four_years_l596_59627

/-- The height of a tree that doubles every year -/
def treeHeight (initialHeight : ℝ) (years : ℕ) : ℝ :=
  initialHeight * (2 ^ years)

theorem tree_height_after_four_years
  (h : treeHeight 1 7 = 64) :
  treeHeight 1 4 = 8 :=
sorry

end NUMINAMATH_CALUDE_tree_height_after_four_years_l596_59627


namespace NUMINAMATH_CALUDE_trig_identity_l596_59693

theorem trig_identity (θ : Real) 
  (h : (1 - Real.cos θ) / (4 + Real.sin θ ^ 2) = 1 / 2) : 
  (4 + Real.cos θ ^ 3) * (3 + Real.sin θ ^ 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l596_59693


namespace NUMINAMATH_CALUDE_number_of_arrangements_l596_59636

/-- The number of applicants --/
def num_applicants : ℕ := 5

/-- The number of positions to be filled --/
def num_positions : ℕ := 3

/-- The number of students to be selected --/
def num_selected : ℕ := 3

/-- Function to calculate the number of arrangements --/
def calculate_arrangements (n : ℕ) (r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

/-- Theorem stating the number of arrangements --/
theorem number_of_arrangements :
  calculate_arrangements num_applicants num_selected +
  calculate_arrangements (num_applicants - 1) num_selected = 16 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l596_59636
