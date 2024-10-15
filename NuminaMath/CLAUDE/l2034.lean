import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_formula_l2034_203473

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8/15)
  (h2 : x - y = 2/15) : 
  x^2 - y^2 = 16/225 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l2034_203473


namespace NUMINAMATH_CALUDE_henry_final_balance_l2034_203418

/-- Henry's money transactions --/
def henry_money_problem (initial_amount received_from_relatives found_in_card spent_on_game donated_to_charity : ℚ) : Prop :=
  let total_received := initial_amount + received_from_relatives + found_in_card
  let total_spent := spent_on_game + donated_to_charity
  let final_balance := total_received - total_spent
  final_balance = 21.75

/-- Theorem stating that Henry's final balance is $21.75 --/
theorem henry_final_balance :
  henry_money_problem 11.75 18.50 5.25 10.60 3.15 := by
  sorry


end NUMINAMATH_CALUDE_henry_final_balance_l2034_203418


namespace NUMINAMATH_CALUDE_pair_count_theorem_l2034_203417

def count_pairs (n : ℕ) : ℕ :=
  (n - 50) * (n - 51) / 2 + 1275

theorem pair_count_theorem :
  count_pairs 100 = 2500 :=
sorry

end NUMINAMATH_CALUDE_pair_count_theorem_l2034_203417


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l2034_203404

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  -1 / (x + 1) - (a + 1) * log (x + 1) + a * x + Real.exp 1 - 2

theorem f_has_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    x₁ > -1 ∧ x₂ > -1 ∧ x₃ > -1 ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, x > -1 → f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔
  a > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l2034_203404


namespace NUMINAMATH_CALUDE_valid_n_count_l2034_203476

-- Define the triangle sides as functions of n
def AB (n : ℕ) : ℕ := 3 * n + 6
def BC (n : ℕ) : ℕ := 2 * n + 15
def AC (n : ℕ) : ℕ := 2 * n + 5

-- Define the conditions for a valid triangle
def isValidTriangle (n : ℕ) : Prop :=
  AB n + BC n > AC n ∧
  AB n + AC n > BC n ∧
  BC n + AC n > AB n ∧
  BC n > AB n ∧
  AB n > AC n

-- Theorem stating that there are exactly 7 valid values for n
theorem valid_n_count :
  ∃! (s : Finset ℕ), s.card = 7 ∧ ∀ n, n ∈ s ↔ isValidTriangle n :=
sorry

end NUMINAMATH_CALUDE_valid_n_count_l2034_203476


namespace NUMINAMATH_CALUDE_divisible_by_nineteen_l2034_203494

theorem divisible_by_nineteen (n : ℕ) :
  ∃ k : ℤ, (5 : ℤ)^(2*n+1) * 2^(n+2) + 3^(n+2) * 2^(2*n+1) = 19 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nineteen_l2034_203494


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2034_203425

theorem fraction_evaluation :
  let x : ℚ := 2/3
  let y : ℚ := 8/10
  (6*x + 10*y) / (60*x*y) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2034_203425


namespace NUMINAMATH_CALUDE_daniel_has_five_dogs_l2034_203490

/-- The number of legs for a healthy horse -/
def horse_legs : ℕ := 4

/-- The number of legs for a healthy cat -/
def cat_legs : ℕ := 4

/-- The number of legs for a healthy turtle -/
def turtle_legs : ℕ := 4

/-- The number of legs for a healthy goat -/
def goat_legs : ℕ := 4

/-- The number of legs for a healthy dog -/
def dog_legs : ℕ := 4

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The total number of legs of all animals Daniel has -/
def total_legs : ℕ := 72

theorem daniel_has_five_dogs :
  ∃ (num_dogs : ℕ), 
    num_dogs * dog_legs + 
    num_horses * horse_legs + 
    num_cats * cat_legs + 
    num_turtles * turtle_legs + 
    num_goats * goat_legs = total_legs ∧ 
    num_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_daniel_has_five_dogs_l2034_203490


namespace NUMINAMATH_CALUDE_least_common_multiple_of_band_sets_l2034_203442

theorem least_common_multiple_of_band_sets : Nat.lcm (Nat.lcm 2 9) 14 = 126 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_band_sets_l2034_203442


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2034_203460

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n < 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = -6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2034_203460


namespace NUMINAMATH_CALUDE_heating_pad_cost_per_use_l2034_203465

/-- The cost per use of a heating pad -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * num_weeks)

/-- Theorem: The cost per use of a heating pad is $5 -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_heating_pad_cost_per_use_l2034_203465


namespace NUMINAMATH_CALUDE_max_min_difference_d_l2034_203413

theorem max_min_difference_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 20) :
  ∃ (d_min d_max : ℝ), 
    (∀ d', (∃ a' b' c', a' + b' + c' + d' = 3 ∧ a'^2 + b'^2 + c'^2 + d'^2 = 20) → d_min ≤ d' ∧ d' ≤ d_max) ∧
    d_max - d_min = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_d_l2034_203413


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l2034_203408

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l2034_203408


namespace NUMINAMATH_CALUDE_stock_sale_before_brokerage_l2034_203434

/-- Calculates the total amount before brokerage given the cash realized and brokerage rate -/
def totalBeforeBrokerage (cashRealized : ℚ) (brokerageRate : ℚ) : ℚ :=
  cashRealized / (1 - brokerageRate)

/-- Theorem stating that for a stock sale with cash realization of 106.25 
    after a 1/4% brokerage fee, the total amount before brokerage is approximately 106.515 -/
theorem stock_sale_before_brokerage :
  let cashRealized : ℚ := 106.25
  let brokerageRate : ℚ := 1 / 400
  let result := totalBeforeBrokerage cashRealized brokerageRate
  ⌊result * 1000⌋ / 1000 = 106515 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_before_brokerage_l2034_203434


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l2034_203406

theorem pet_store_siamese_cats :
  let initial_house_cats : ℝ := 5.0
  let added_cats : ℝ := 10.0
  let total_cats_after : ℕ := 28
  let initial_siamese_cats : ℝ := initial_house_cats + added_cats + total_cats_after - (initial_house_cats + added_cats)
  initial_siamese_cats = 13 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l2034_203406


namespace NUMINAMATH_CALUDE_cube_volume_after_removal_l2034_203486

/-- Theorem: Volume of a cube with edge sum 72 cm after removing a 1 cm cube corner -/
theorem cube_volume_after_removal (edge_sum : ℝ) (small_cube_edge : ℝ) : 
  edge_sum = 72 → small_cube_edge = 1 → 
  (edge_sum / 12)^3 - small_cube_edge^3 = 215 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_after_removal_l2034_203486


namespace NUMINAMATH_CALUDE_find_b_l2034_203467

theorem find_b (x₁ x₂ c : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : ∃ y, y^2 + 2*x₁*y + 2*x₂ = 0 ∧ y^2 + 2*x₂*y + 2*x₁ = 0)
  (h₃ : x₁^2 + 5*(1/10)*x₁ + c = 0)
  (h₄ : x₂^2 + 5*(1/10)*x₂ + c = 0) :
  ∃ b : ℝ, b = 1/10 ∧ x₁^2 + 5*b*x₁ + c = 0 ∧ x₂^2 + 5*b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_find_b_l2034_203467


namespace NUMINAMATH_CALUDE_cuboid_length_is_40_l2034_203463

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The length of a cuboid with surface area 2400, breadth 10, and height 16 is 40 -/
theorem cuboid_length_is_40 :
  ∃ l : ℝ, cuboidSurfaceArea l 10 16 = 2400 ∧ l = 40 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_length_is_40_l2034_203463


namespace NUMINAMATH_CALUDE_point_in_region_implies_a_range_l2034_203409

theorem point_in_region_implies_a_range (a : ℝ) :
  (1 : ℝ) + (1 : ℝ) + a < 0 → a < -2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_implies_a_range_l2034_203409


namespace NUMINAMATH_CALUDE_square_field_area_l2034_203471

/-- The area of a square field with side length 8 meters is 64 square meters. -/
theorem square_field_area : 
  ∀ (side_length area : ℝ), 
  side_length = 8 → 
  area = side_length ^ 2 → 
  area = 64 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l2034_203471


namespace NUMINAMATH_CALUDE_f_n_formula_l2034_203491

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => f x
  | m + 1 => deriv (f_n m) x

theorem f_n_formula (n : ℕ) (x : ℝ) :
  f_n (n + 1) x = ((-1)^(n + 1) * (x - (n + 1))) / Real.exp x :=
by sorry

end NUMINAMATH_CALUDE_f_n_formula_l2034_203491


namespace NUMINAMATH_CALUDE_altitude_B_correct_median_A_correct_circumcircle_correct_l2034_203462

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, -1)
def C : ℝ × ℝ := (-2, 1)

-- Define the altitude from B to BC
def altitude_B (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the median from A to AC
def median_A (x : ℝ) : Prop := x = -1

-- Define the circumcircle of triangle ABC
def circumcircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 1 = 0

-- Theorem statements
theorem altitude_B_correct :
  ∀ x y : ℝ, altitude_B x y ↔ (x - y + 1 = 0) :=
sorry

theorem median_A_correct :
  ∀ x : ℝ, median_A x ↔ (x = -1) :=
sorry

theorem circumcircle_correct :
  ∀ x y : ℝ, circumcircle x y ↔ (x^2 + y^2 + 2*x - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_altitude_B_correct_median_A_correct_circumcircle_correct_l2034_203462


namespace NUMINAMATH_CALUDE_jerome_classmates_l2034_203488

/-- Represents Jerome's contact list --/
structure ContactList where
  classmates : ℕ
  outOfSchoolFriends : ℕ
  familyMembers : ℕ
  total : ℕ

/-- The properties of Jerome's contact list --/
def jeromeContactList : ContactList → Prop
  | cl => cl.outOfSchoolFriends = cl.classmates / 2 ∧
          cl.familyMembers = 3 ∧
          cl.total = 33 ∧
          cl.total = cl.classmates + cl.outOfSchoolFriends + cl.familyMembers

/-- Theorem: Jerome has 20 classmates on his contact list --/
theorem jerome_classmates :
  ∀ cl : ContactList, jeromeContactList cl → cl.classmates = 20 := by
  sorry


end NUMINAMATH_CALUDE_jerome_classmates_l2034_203488


namespace NUMINAMATH_CALUDE_baker_revenue_difference_l2034_203458

/-- Baker's sales and pricing information --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculate the difference between daily average and today's revenue --/
def revenue_difference (sales : BakerSales) : ℕ :=
  let usual_revenue := sales.usual_pastries * sales.pastry_price + sales.usual_bread * sales.bread_price
  let today_revenue := sales.today_pastries * sales.pastry_price + sales.today_bread * sales.bread_price
  today_revenue - usual_revenue

/-- Theorem stating the revenue difference for the given sales information --/
theorem baker_revenue_difference :
  revenue_difference ⟨20, 10, 14, 25, 2, 4⟩ = 48 := by
  sorry

end NUMINAMATH_CALUDE_baker_revenue_difference_l2034_203458


namespace NUMINAMATH_CALUDE_apple_picking_l2034_203422

theorem apple_picking (minjae_apples father_apples : ℝ) 
  (h1 : minjae_apples = 2.6)
  (h2 : father_apples = 5.98) :
  minjae_apples + father_apples = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_l2034_203422


namespace NUMINAMATH_CALUDE_function_symmetry_l2034_203400

/-- The function f(x) = 3cos(2x + π/6) is symmetric about the point (π/6, 0) -/
theorem function_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = 3 * Real.cos (2 * x + π / 6)) :
  ∀ x, f (π / 3 - x) = f (π / 3 + x) :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_l2034_203400


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2034_203479

theorem expression_simplification_and_evaluation :
  ∀ a : ℚ, -3 < a → a ≤ 0 → a ≠ -1 → a ≠ 0 → a ≠ 1 →
  let original_expr := (a - (2*a - 1) / a) / (1/a - a)
  let simplified_expr := (1 - a) / (1 + a)
  original_expr = simplified_expr ∧
  (a = -2 → simplified_expr = -3) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2034_203479


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l2034_203457

/-- Represents a rectangular yard with flower beds and a trapezoidal lawn -/
structure YardWithFlowerBeds where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  trapezoid_height : ℝ
  num_flower_beds : ℕ

/-- The fraction of the yard occupied by flower beds -/
def flower_bed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  25 / 324

/-- Theorem stating the fraction of the yard occupied by flower beds -/
theorem flower_bed_fraction_is_correct (yard : YardWithFlowerBeds) 
    (h1 : yard.trapezoid_short_side = 26)
    (h2 : yard.trapezoid_long_side = 36)
    (h3 : yard.trapezoid_height = 6)
    (h4 : yard.num_flower_beds = 3) : 
  flower_bed_fraction yard = 25 / 324 := by
  sorry

#check flower_bed_fraction_is_correct

end NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l2034_203457


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2034_203449

open Set

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2034_203449


namespace NUMINAMATH_CALUDE_nine_chapters_equations_correct_l2034_203498

/-- Represents the scenario of cars and people as described in "The Nine Chapters on the Mathematical Art" problem --/
def nine_chapters_problem (x y : ℤ) : Prop :=
  (y = 2*x + 9) ∧ (y = 3*(x - 2))

/-- Theorem stating that the equations correctly represent the described scenario --/
theorem nine_chapters_equations_correct :
  ∀ x y : ℤ, 
    nine_chapters_problem x y →
    (y = 2*x + 9) ∧ 
    (y = 3*(x - 2)) ∧
    (x > 0) ∧ 
    (y > 0) := by
  sorry

end NUMINAMATH_CALUDE_nine_chapters_equations_correct_l2034_203498


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2034_203433

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2034_203433


namespace NUMINAMATH_CALUDE_smallest_multiple_of_nine_l2034_203487

theorem smallest_multiple_of_nine (x y : ℤ) 
  (hx : ∃ k : ℤ, x + 2 = 9 * k) 
  (hy : ∃ k : ℤ, y - 2 = 9 * k) : 
  (∃ n : ℕ, n > 0 ∧ ∃ k : ℤ, x^2 - x*y + y^2 + n = 9 * k) ∧ 
  (∀ m : ℕ, m > 0 → (∃ k : ℤ, x^2 - x*y + y^2 + m = 9 * k) → m ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_nine_l2034_203487


namespace NUMINAMATH_CALUDE_magic_square_solution_l2034_203414

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  sum : ℕ
  row_sums : a + b + c = sum ∧ d + e + f = sum ∧ g + h + i = sum
  col_sums : a + d + g = sum ∧ b + e + h = sum ∧ c + f + i = sum
  diag_sums : a + e + i = sum ∧ c + e + g = sum

/-- The theorem to be proved -/
theorem magic_square_solution (ms : MagicSquare) 
  (h1 : ms.b = 25)
  (h2 : ms.c = 103)
  (h3 : ms.d = 3) :
  ms.a = 214 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_solution_l2034_203414


namespace NUMINAMATH_CALUDE_two_intersection_points_l2034_203447

def quadratic_function (c : ℝ) (x : ℝ) : ℝ := 2*x^2 - 3*x - c

theorem two_intersection_points (c : ℝ) (h : c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_function c x₁ = 0 ∧ quadratic_function c x₂ = 0 ∧
  ∀ x : ℝ, quadratic_function c x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_two_intersection_points_l2034_203447


namespace NUMINAMATH_CALUDE_glycerin_percentage_after_dilution_l2034_203438

def initial_glycerin_percentage : ℝ := 0.9
def initial_volume : ℝ := 4
def added_water : ℝ := 0.8

theorem glycerin_percentage_after_dilution :
  let initial_glycerin := initial_glycerin_percentage * initial_volume
  let final_volume := initial_volume + added_water
  let final_glycerin_percentage := initial_glycerin / final_volume
  final_glycerin_percentage = 0.75 := by sorry

end NUMINAMATH_CALUDE_glycerin_percentage_after_dilution_l2034_203438


namespace NUMINAMATH_CALUDE_a_range_l2034_203407

/-- Set A defined as { x | a ≤ x ≤ a+3 } -/
def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 3 }

/-- Set B defined as { x | x < -1 or x > 5 } -/
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

/-- Theorem stating that if A ∪ B = B, then a is in (-∞, -4) ∪ (5, +∞) -/
theorem a_range (a : ℝ) : (A a ∪ B = B) → a < -4 ∨ a > 5 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2034_203407


namespace NUMINAMATH_CALUDE_police_force_ratio_l2034_203444

/-- Given a police force with female officers and officers on duty, prove the ratio of female officers to total officers on duty. -/
theorem police_force_ratio (total_female : ℕ) (total_on_duty : ℕ) (female_duty_percent : ℚ) : 
  total_female = 300 →
  total_on_duty = 240 →
  female_duty_percent = 2/5 →
  (female_duty_percent * total_female) / total_on_duty = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_police_force_ratio_l2034_203444


namespace NUMINAMATH_CALUDE_sum_of_integers_l2034_203440

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 193) 
  (h2 : x.val * y.val = 84) : 
  x.val + y.val = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2034_203440


namespace NUMINAMATH_CALUDE_average_speed_theorem_l2034_203439

/-- Proves that the average speed of a trip is 40 mph given specific conditions -/
theorem average_speed_theorem (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

#check average_speed_theorem

end NUMINAMATH_CALUDE_average_speed_theorem_l2034_203439


namespace NUMINAMATH_CALUDE_set_inclusion_implies_m_bound_l2034_203405

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def B : Set ℝ := {x | x^2 - 4*x < 0}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem set_inclusion_implies_m_bound (m : ℝ) :
  C m ⊆ (C m ∩ B) → m ≤ 5/2 := by
  sorry


end NUMINAMATH_CALUDE_set_inclusion_implies_m_bound_l2034_203405


namespace NUMINAMATH_CALUDE_cubic_equation_one_solution_l2034_203411

/-- The cubic equation in x with parameter b -/
def cubic_equation (x b : ℝ) : ℝ := x^3 - b*x^2 - 3*b*x + b^2 - 4

/-- The condition for the equation to have exactly one real solution -/
def has_one_real_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation x b = 0

theorem cubic_equation_one_solution :
  ∀ b : ℝ, has_one_real_solution b ↔ b > 3 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_solution_l2034_203411


namespace NUMINAMATH_CALUDE_congruence_solution_l2034_203424

theorem congruence_solution (x : ℤ) : 
  (10 * x + 3) % 18 = 7 % 18 → 
  ∃ (a m : ℕ), 
    0 < m ∧ 
    0 < a ∧ 
    a < m ∧
    x % m = a % m ∧
    a = 4 ∧ 
    m = 9 ∧
    a + m = 13 := by
  sorry

#check congruence_solution

end NUMINAMATH_CALUDE_congruence_solution_l2034_203424


namespace NUMINAMATH_CALUDE_cubic_factorization_l2034_203401

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x + 3)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2034_203401


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_pyramid_l2034_203435

/-- The surface area of a sphere given a right square pyramid inscribed in it -/
theorem sphere_surface_area_from_pyramid (h V : ℝ) (h_pos : h > 0) (V_pos : V > 0) :
  let s := Real.sqrt (3 * V / h)
  let r := Real.sqrt ((s^2 + 2 * h^2) / 4)
  h = 4 → V = 16 → 4 * Real.pi * r^2 = 24 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_pyramid_l2034_203435


namespace NUMINAMATH_CALUDE_two_b_values_for_two_integer_solutions_l2034_203472

theorem two_b_values_for_two_integer_solutions : 
  ∃! (s : Finset ℤ), 
    (∀ b ∈ s, ∃! (t : Finset ℤ), (∀ x ∈ t, x^2 + b*x + 5 ≤ 0) ∧ t.card = 2) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_b_values_for_two_integer_solutions_l2034_203472


namespace NUMINAMATH_CALUDE_black_stones_count_l2034_203432

theorem black_stones_count (total : Nat) (white : Nat) : 
  total = 48 → 
  (4 * white) % 37 = 26 → 
  (4 * white) / 37 = 2 → 
  total - white = 23 := by
sorry

end NUMINAMATH_CALUDE_black_stones_count_l2034_203432


namespace NUMINAMATH_CALUDE_car_worth_calculation_l2034_203481

/-- Brendan's earnings and expenses in June -/
structure BrendanFinances where
  total_earnings : ℕ  -- Total earnings in June
  remaining_money : ℕ  -- Remaining money at the end of June
  car_worth : ℕ  -- Worth of the used car

/-- The worth of the car is the difference between total earnings and remaining money -/
theorem car_worth_calculation (b : BrendanFinances) 
  (h1 : b.total_earnings = 5000)
  (h2 : b.remaining_money = 1000)
  (h3 : b.car_worth = b.total_earnings - b.remaining_money) :
  b.car_worth = 4000 := by
  sorry

#check car_worth_calculation

end NUMINAMATH_CALUDE_car_worth_calculation_l2034_203481


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l2034_203482

theorem tennis_tournament_matches (n : Nat) (byes : Nat) :
  n = 100 →
  byes = 28 →
  ∃ m : Nat, m = n - 1 ∧ m % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l2034_203482


namespace NUMINAMATH_CALUDE_division_problem_l2034_203452

theorem division_problem (N : ℕ) : 
  (N / 7 = 12 ∧ N % 7 = 4) → (N / 3 = 29) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2034_203452


namespace NUMINAMATH_CALUDE_total_dress_designs_l2034_203431

/-- The number of fabric colors available. -/
def num_colors : ℕ := 5

/-- The number of patterns available. -/
def num_patterns : ℕ := 4

/-- The number of sleeve designs available. -/
def num_sleeve_designs : ℕ := 3

/-- Each dress design requires exactly one color, one pattern, and one sleeve design. -/
theorem total_dress_designs :
  num_colors * num_patterns * num_sleeve_designs = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l2034_203431


namespace NUMINAMATH_CALUDE_kangaroo_jumps_odd_jumps_zero_four_jumps_two_l2034_203497

/-- Represents a regular octagon with vertices labeled from 0 to 7 -/
def Octagon := Fin 8

/-- Defines whether two vertices are adjacent in the octagon -/
def adjacent (v w : Octagon) : Prop :=
  (v.val + 1) % 8 = w.val ∨ (w.val + 1) % 8 = v.val

/-- Defines the number of ways a kangaroo can reach vertex E from A in n jumps -/
def num_ways (n : ℕ) : ℕ :=
  sorry -- Definition to be implemented

/-- Main theorem: Characterizes the number of ways to reach E from A in n jumps -/
theorem kangaroo_jumps (n : ℕ) :
  num_ways n = if n % 2 = 0
    then let m := n / 2
         (((2 : ℝ) + Real.sqrt 2) ^ (m - 1) - ((2 : ℝ) - Real.sqrt 2) ^ (m - 1)) / Real.sqrt 2
    else 0 :=
  sorry

/-- The number of ways to reach E from A in an odd number of jumps is 0 -/
theorem odd_jumps_zero (n : ℕ) (h : n % 2 = 1) :
  num_ways n = 0 :=
  sorry

/-- The number of ways to reach E from A in 4 jumps is 2 -/
theorem four_jumps_two :
  num_ways 4 = 2 :=
  sorry

end NUMINAMATH_CALUDE_kangaroo_jumps_odd_jumps_zero_four_jumps_two_l2034_203497


namespace NUMINAMATH_CALUDE_expected_value_8_sided_die_l2034_203430

def standard_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_8_sided_die :
  let outcomes := standard_8_sided_die
  let probability (n : ℕ) := (1 : ℚ) / 8
  let expected_value := (outcomes.sum (λ n => (n + 1 : ℚ) * probability n)) / outcomes.card
  expected_value = 9/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_8_sided_die_l2034_203430


namespace NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l2034_203402

theorem kyle_money_after_snowboarding (dave_money : ℕ) (kyle_initial_money : ℕ) : 
  dave_money = 46 →
  kyle_initial_money = 3 * dave_money - 12 →
  kyle_initial_money / 3 = kyle_initial_money - (kyle_initial_money / 3) →
  kyle_initial_money - (kyle_initial_money / 3) = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l2034_203402


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2034_203480

theorem quadratic_roots_sum_of_squares (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2 * x₁^2 + 4 * m * x₁ + m = 0 ∧
    2 * x₂^2 + 4 * m * x₂ + m = 0 ∧
    x₁^2 + x₂^2 = 3/16) →
  m = -1/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2034_203480


namespace NUMINAMATH_CALUDE_f_11_values_l2034_203416

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

axiom coprime_property {f : ℕ → ℕ} {a b : ℕ} (h : is_coprime a b) : 
  f (a * b) = f a * f b

axiom prime_property {f : ℕ → ℕ} {m k : ℕ} (hm : is_prime m) (hk : is_prime k) : 
  f (m + k - 3) = f m + f k - f 3

theorem f_11_values (f : ℕ → ℕ) 
  (h1 : ∀ a b : ℕ, is_coprime a b → f (a * b) = f a * f b)
  (h2 : ∀ m k : ℕ, is_prime m → is_prime k → f (m + k - 3) = f m + f k - f 3) :
  f 11 = 1 ∨ f 11 = 11 :=
sorry

end NUMINAMATH_CALUDE_f_11_values_l2034_203416


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2034_203415

theorem polynomial_multiplication (x : ℝ) :
  (3 * x^2 - 2 * x + 4) * (-4 * x^2 + 3 * x - 6) =
  -12 * x^4 + 17 * x^3 - 40 * x^2 + 24 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2034_203415


namespace NUMINAMATH_CALUDE_sin_15_deg_identity_l2034_203441

theorem sin_15_deg_identity : 1 - 2 * (Real.sin (15 * π / 180))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_deg_identity_l2034_203441


namespace NUMINAMATH_CALUDE_incenter_circles_theorem_l2034_203426

-- Define the basic geometric objects
variable (A B C I : Point)
variable (O₁ O₂ O₃ : Point)
variable (A' B' C' : Point)

-- Define the incenter
def is_incenter (I : Point) (A B C : Point) : Prop := sorry

-- Define circles passing through points
def circle_through (O : Point) (P Q : Point) : Prop := sorry

-- Define perpendicular intersection of circles
def perpendicular_intersection (O : Point) (I : Point) : Prop := sorry

-- Define the other intersection point of two circles
def other_intersection (O₁ O₂ : Point) (P : Point) : Point := sorry

-- Define the circumradius of a triangle
def circumradius (A B C : Point) : ℝ := sorry

-- Define the radius of a circle
def circle_radius (O : Point) : ℝ := sorry

-- State the theorem
theorem incenter_circles_theorem 
  (h_incenter : is_incenter I A B C)
  (h_O₁ : circle_through O₁ B C)
  (h_O₂ : circle_through O₂ A C)
  (h_O₃ : circle_through O₃ A B)
  (h_perp₁ : perpendicular_intersection O₁ I)
  (h_perp₂ : perpendicular_intersection O₂ I)
  (h_perp₃ : perpendicular_intersection O₃ I)
  (h_A' : A' = other_intersection O₂ O₃ A)
  (h_B' : B' = other_intersection O₁ O₃ B)
  (h_C' : C' = other_intersection O₁ O₂ C) :
  circumradius A' B' C' = (1/2) * circle_radius I := by sorry

end NUMINAMATH_CALUDE_incenter_circles_theorem_l2034_203426


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2034_203459

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 1 - 9
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2034_203459


namespace NUMINAMATH_CALUDE_max_reflections_is_18_l2034_203410

/-- The angle between the lines in degrees -/
def angle : ℝ := 5

/-- The maximum angle for perpendicular reflection in degrees -/
def max_angle : ℝ := 90

/-- The maximum number of reflections -/
def max_reflections : ℕ := 18

/-- Theorem stating that the maximum number of reflections is 18 -/
theorem max_reflections_is_18 :
  ∀ n : ℕ, n * angle ≤ max_angle → n ≤ max_reflections :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_is_18_l2034_203410


namespace NUMINAMATH_CALUDE_base_7_representation_and_properties_l2034_203453

def base_10_to_base_7 (n : ℕ) : List ℕ :=
  sorry

def count_even_digits (digits : List ℕ) : ℕ :=
  sorry

def sum_even_digits (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_and_properties :
  let base_7_repr := base_10_to_base_7 1250
  base_7_repr = [3, 4, 3, 4] ∧
  count_even_digits base_7_repr = 2 ∧
  ¬(sum_even_digits base_7_repr % 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_base_7_representation_and_properties_l2034_203453


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2034_203489

theorem reciprocal_problem (x : ℚ) (h : 7 * x = 3) : 150 * (1 / x) = 350 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2034_203489


namespace NUMINAMATH_CALUDE_bus_children_difference_solve_bus_problem_l2034_203403

theorem bus_children_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_children children_off children_on final_children =>
    initial_children - children_off + children_on = final_children →
    children_off - children_on = 24

theorem solve_bus_problem :
  bus_children_difference 36 68 (68 - 24) 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_children_difference_solve_bus_problem_l2034_203403


namespace NUMINAMATH_CALUDE_nonnegative_root_condition_l2034_203443

/-- A polynomial of degree 4 with coefficient q -/
def polynomial (q : ℝ) (x : ℝ) : ℝ := x^4 + q*x^3 + x^2 + q*x + 4

/-- The condition for the existence of a non-negative real root -/
def has_nonnegative_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ polynomial q x = 0

/-- The theorem stating the condition on q for the existence of a non-negative root -/
theorem nonnegative_root_condition (q : ℝ) : 
  has_nonnegative_root q ↔ q ≤ -2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_nonnegative_root_condition_l2034_203443


namespace NUMINAMATH_CALUDE_eighth_term_value_arithmetic_sequence_eighth_term_l2034_203468

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first six terms
  sum_first_six : ℚ
  -- The seventh term
  seventh_term : ℚ
  -- Property: The sum of the first six terms is 21
  sum_property : sum_first_six = 21
  -- Property: The seventh term is 8
  seventh_property : seventh_term = 8

/-- Theorem: The eighth term of the arithmetic sequence is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence) : ℚ :=
  65 / 7

/-- The main theorem: Given the conditions, the eighth term is 65/7 -/
theorem arithmetic_sequence_eighth_term (seq : ArithmeticSequence) :
  eighth_term_value seq = 65 / 7 := by
  sorry


end NUMINAMATH_CALUDE_eighth_term_value_arithmetic_sequence_eighth_term_l2034_203468


namespace NUMINAMATH_CALUDE_dvd_player_cost_l2034_203412

theorem dvd_player_cost (d m : ℝ) 
  (h1 : d / m = 9 / 2)
  (h2 : d = m + 63) :
  d = 81 := by
sorry

end NUMINAMATH_CALUDE_dvd_player_cost_l2034_203412


namespace NUMINAMATH_CALUDE_triangle_side_length_l2034_203478

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 55) (h2 : b = 20) (h3 : c = 30) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2034_203478


namespace NUMINAMATH_CALUDE_unique_solution_range_l2034_203466

theorem unique_solution_range (x a : ℝ) : 
  (∃! x, Real.log (4 * x^2 + 4 * a * x) - Real.log (4 * x - a + 1) = 0) ↔ 
  (1/5 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_range_l2034_203466


namespace NUMINAMATH_CALUDE_special_set_characterization_l2034_203485

/-- The set of integers n ≥ 1 such that 2^n - 1 has exactly n positive integer divisors -/
def special_set : Set ℕ+ :=
  {n | (Nat.card (Nat.divisors ((2:ℕ)^(n:ℕ) - 1))) = n}

/-- Theorem stating that the special set is equal to {1, 2, 4, 6, 8, 16, 32} -/
theorem special_set_characterization :
  special_set = {1, 2, 4, 6, 8, 16, 32} := by sorry

end NUMINAMATH_CALUDE_special_set_characterization_l2034_203485


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2034_203427

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the equations of its asymptotes are y = ± x/2, then b = 1 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∀ x : ℝ, (∃ y : ℝ, y = x / 2 ∨ y = -x / 2) → 
    x^2 / 4 - y^2 / b^2 = 1) →
  b = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2034_203427


namespace NUMINAMATH_CALUDE_inequality_implication_l2034_203451

theorem inequality_implication (a b : ℝ) (h : a > b) : 2*a - 1 > 2*b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2034_203451


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2034_203436

def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
def B (a : ℝ) : Set ℝ := {a - 3, a - 1, a + 1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-2} → a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2034_203436


namespace NUMINAMATH_CALUDE_oplus_problem_l2034_203445

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation ⊕
def oplus : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.one
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem oplus_problem :
  oplus (oplus Element.three Element.two) (oplus Element.four Element.one) = Element.three :=
by sorry

end NUMINAMATH_CALUDE_oplus_problem_l2034_203445


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2034_203450

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x > f x) : 
  {x : ℝ | f x / Real.exp x > f 1 / Real.exp 1} = Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2034_203450


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2034_203429

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 - m * y - 3 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2034_203429


namespace NUMINAMATH_CALUDE_only_f3_is_quadratic_l2034_203469

-- Define the concept of a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the given functions
def f1 (x : ℝ) : ℝ := 3 * x
def f2 (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def f3 (x : ℝ) : ℝ := (x - 1)^2
def f4 (x : ℝ) : ℝ := 2

-- State the theorem
theorem only_f3_is_quadratic :
  (¬ is_quadratic f1) ∧
  (¬ ∀ a b c, is_quadratic (f2 a b c)) ∧
  is_quadratic f3 ∧
  (¬ is_quadratic f4) :=
sorry

end NUMINAMATH_CALUDE_only_f3_is_quadratic_l2034_203469


namespace NUMINAMATH_CALUDE_siblings_age_sum_l2034_203496

/-- The age difference between each sibling -/
def age_gap : ℕ := 5

/-- The current age of the eldest sibling -/
def eldest_age : ℕ := 20

/-- The number of years into the future we're calculating -/
def years_ahead : ℕ := 10

/-- The total age of three siblings born 'age_gap' years apart, 
    where the eldest is currently 'eldest_age' years old, 
    after 'years_ahead' years -/
def total_age (age_gap eldest_age years_ahead : ℕ) : ℕ :=
  (eldest_age + years_ahead) + 
  (eldest_age - age_gap + years_ahead) + 
  (eldest_age - 2 * age_gap + years_ahead)

theorem siblings_age_sum : 
  total_age age_gap eldest_age years_ahead = 75 := by
  sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l2034_203496


namespace NUMINAMATH_CALUDE_machine_quality_comparison_l2034_203475

/-- Represents a machine producing products of different quality classes -/
structure Machine where
  first_class : ℕ
  second_class : ℕ

/-- Calculates the frequency of first-class products for a machine -/
def first_class_frequency (m : Machine) : ℚ :=
  m.first_class / (m.first_class + m.second_class)

/-- Calculates the K² statistic for comparing two machines -/
def k_squared (m1 m2 : Machine) : ℚ :=
  let n := m1.first_class + m1.second_class + m2.first_class + m2.second_class
  let a := m1.first_class
  let b := m1.second_class
  let c := m2.first_class
  let d := m2.second_class
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The theorem to be proved -/
theorem machine_quality_comparison (machine_a machine_b : Machine)
  (h_a : machine_a = ⟨150, 50⟩)
  (h_b : machine_b = ⟨120, 80⟩) :
  first_class_frequency machine_a = 3/4 ∧
  first_class_frequency machine_b = 3/5 ∧
  k_squared machine_a machine_b > 6635/1000 := by
  sorry

end NUMINAMATH_CALUDE_machine_quality_comparison_l2034_203475


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l2034_203456

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π/4) = 2) : 
  Real.tan x / Real.tan (2*x) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l2034_203456


namespace NUMINAMATH_CALUDE_trichotomy_of_reals_l2034_203464

theorem trichotomy_of_reals : ∀ a b : ℝ, (a > b ∨ a = b ∨ a < b) ∧ 
  (¬(a > b ∧ a = b) ∧ ¬(a > b ∧ a < b) ∧ ¬(a = b ∧ a < b)) := by
  sorry

end NUMINAMATH_CALUDE_trichotomy_of_reals_l2034_203464


namespace NUMINAMATH_CALUDE_shell_ratio_l2034_203477

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def shell_problem (sc : ShellCounts) : Prop :=
  sc.david = 15 ∧
  sc.mia = 4 * sc.david ∧
  sc.ava = sc.mia + 20 ∧
  sc.david + sc.mia + sc.ava + sc.alice = 195

/-- The theorem to prove -/
theorem shell_ratio (sc : ShellCounts) 
  (h : shell_problem sc) : sc.alice * 2 = sc.ava := by
  sorry

end NUMINAMATH_CALUDE_shell_ratio_l2034_203477


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_closed_interval_l2034_203492

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x ≥ 3, y = Real.log (x + 1) / Real.log (1/2)}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

-- State the theorem
theorem M_intersect_N_eq_closed_interval :
  M ∩ N = Set.Icc (-3) (-2) := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_closed_interval_l2034_203492


namespace NUMINAMATH_CALUDE_smallest_non_trivial_divisor_of_product_l2034_203455

def product_of_even_integers (n : ℕ) : ℕ :=
  (List.range ((n + 1) / 2)).foldl (λ acc i => acc * (2 * (i + 1))) 1

theorem smallest_non_trivial_divisor_of_product (n : ℕ) (h : n = 134) :
  ∃ (d : ℕ), d > 1 ∧ d ∣ product_of_even_integers n ∧
  ∀ (k : ℕ), 1 < k → k < d → ¬(k ∣ product_of_even_integers n) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_non_trivial_divisor_of_product_l2034_203455


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l2034_203446

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Theorem: 1729 in base 10 is equal to 5020 in base 7 -/
theorem base_conversion_1729_to_base7 :
  1729 = fromBase7 [5, 0, 2, 0] := by
  sorry

#eval fromBase7 [5, 0, 2, 0]  -- Should output 1729

end NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l2034_203446


namespace NUMINAMATH_CALUDE_green_balls_count_l2034_203437

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  white = 10 →
  yellow = 10 →
  red = 47 →
  purple = 3 →
  prob_not_red_purple = 1/2 →
  ∃ green : ℕ, green = 30 ∧ total = white + yellow + red + purple + green :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l2034_203437


namespace NUMINAMATH_CALUDE_exponent_division_l2034_203423

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end NUMINAMATH_CALUDE_exponent_division_l2034_203423


namespace NUMINAMATH_CALUDE_function_equality_l2034_203483

theorem function_equality : ∀ x : ℝ, x = 3 * x^3 := by sorry

end NUMINAMATH_CALUDE_function_equality_l2034_203483


namespace NUMINAMATH_CALUDE_min_value_of_a_l2034_203428

theorem min_value_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ (x₁ : ℝ) (x₂ : ℝ), x₁ > 0 → 1 ≤ x₂ → x₂ ≤ Real.exp 1 → 
    x₁ + a^2 / x₁ ≥ x₂ - Real.log x₂) → 
  a ≥ Real.sqrt (Real.exp 1 - 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2034_203428


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l2034_203448

theorem imaginary_part_of_i_minus_one :
  Complex.im (Complex.I - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l2034_203448


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2034_203420

def M : ℕ := 39 * 48 * 77 * 150

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 62 = sum_even_divisors M :=
sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2034_203420


namespace NUMINAMATH_CALUDE_catastrophic_network_properties_l2034_203470

/-- A catastrophic road network between 6 cities -/
structure CatastrophicNetwork :=
  (cities : Fin 6 → Type)
  (road : cities i → cities j → Prop)
  (no_return : ∀ (i j : Fin 6) (x : cities i) (y : cities j), road x y → ¬ ∃ path : cities j → cities i, True)

theorem catastrophic_network_properties (n : CatastrophicNetwork) :
  (∃ i : Fin 6, ∀ j : Fin 6, ¬ ∃ x : n.cities i, ∃ y : n.cities j, n.road x y) ∧
  (∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → ∃ x : n.cities i, ∃ y : n.cities j, n.road x y) ∧
  (∃ i j : Fin 6, ∀ k l : Fin 6, ∃ path : n.cities k → n.cities l, True) ∧
  (∃ f : Fin 6 → Fin 6, Function.Bijective f ∧ 
    ∀ i j : Fin 6, i ≠ j → (f i < f j ↔ ∃ x : n.cities i, ∃ y : n.cities j, n.road x y)) :=
sorry

#check catastrophic_network_properties

end NUMINAMATH_CALUDE_catastrophic_network_properties_l2034_203470


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2034_203495

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The problem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a)
    (h_a3 : a 3 = 6)
    (h_sum : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2034_203495


namespace NUMINAMATH_CALUDE_complex_subtraction_multiplication_l2034_203419

theorem complex_subtraction_multiplication (i : ℂ) :
  (7 - 3 * i) - 3 * (2 + 4 * i) = 1 - 15 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_multiplication_l2034_203419


namespace NUMINAMATH_CALUDE_line_symmetry_l2034_203499

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - y = 0

-- Define symmetry with respect to x-axis
def symmetric_wrt_x_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -f x

-- Define the proposed symmetric line
def proposed_symmetric_line (x y : ℝ) : Prop := 2 * x + y = 0

-- Theorem statement
theorem line_symmetry :
  ∃ (f g : ℝ → ℝ),
    (∀ x y, original_line x y ↔ y = f x) ∧
    (∀ x y, proposed_symmetric_line x y ↔ y = g x) ∧
    symmetric_wrt_x_axis f g :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2034_203499


namespace NUMINAMATH_CALUDE_trevor_coin_count_l2034_203421

theorem trevor_coin_count : 
  let total_coins : ℕ := 77
  let quarters : ℕ := 29
  let dimes : ℕ := total_coins - quarters
  total_coins - quarters = dimes :=
by sorry

end NUMINAMATH_CALUDE_trevor_coin_count_l2034_203421


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2034_203493

theorem trigonometric_identity (α : ℝ) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) :
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / 
  (1 + Real.sin (4 * α) + Real.cos (4 * α)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2034_203493


namespace NUMINAMATH_CALUDE_proposition_truth_l2034_203454

theorem proposition_truth (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : p ∨ q) : 
  (¬p) ∨ (¬q) := by
sorry

end NUMINAMATH_CALUDE_proposition_truth_l2034_203454


namespace NUMINAMATH_CALUDE_sum_of_first_n_naturals_l2034_203461

theorem sum_of_first_n_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_naturals_l2034_203461


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l2034_203474

theorem largest_prime_divisor_test (m : ℕ) : 
  700 ≤ m → m ≤ 750 → 
  (∀ p : ℕ, p.Prime → p ≤ 23 → m % p ≠ 0) → 
  m.Prime :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l2034_203474


namespace NUMINAMATH_CALUDE_intersection_equals_C_l2034_203484

-- Define the set of angles less than 90°
def A : Set ℝ := {α | α < 90}

-- Define the set of angles in the first quadrant
def B : Set ℝ := {α | ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90}

-- Define the set of angles α such that k · 360° < α < k · 360° + 90° for some integer k ≤ 0
def C : Set ℝ := {α | ∃ k : ℤ, k ≤ 0 ∧ k * 360 < α ∧ α < k * 360 + 90}

-- Theorem statement
theorem intersection_equals_C : A ∩ B = C := by sorry

end NUMINAMATH_CALUDE_intersection_equals_C_l2034_203484
