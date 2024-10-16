import Mathlib

namespace NUMINAMATH_CALUDE_cubic_sum_ge_product_sum_l2067_206781

theorem cubic_sum_ge_product_sum (u v : ℝ) (hu : 0 < u) (hv : 0 < v) :
  u^3 + v^3 ≥ u^2 * v + v^2 * u := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_ge_product_sum_l2067_206781


namespace NUMINAMATH_CALUDE_statement_a_statement_b_statements_a_and_b_correct_l2067_206739

-- Statement A
theorem statement_a (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a + c > b + c := by
  sorry

-- Statement B
theorem statement_b (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

-- Combined theorem for A and B
theorem statements_a_and_b_correct :
  (∀ (a b c : ℝ), a > b → c < 0 → a + c > b + c) ∧
  (∀ (a b : ℝ), a > b → b > 0 → (a + b) / 2 > Real.sqrt (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_statements_a_and_b_correct_l2067_206739


namespace NUMINAMATH_CALUDE_rectangle_area_l2067_206799

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 112 → l * b = 588 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2067_206799


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l2067_206750

theorem rectangular_box_dimensions (A B C : ℝ) : 
  A > 0 ∧ B > 0 ∧ C > 0 →
  A * B = 40 →
  A * C = 90 →
  B * C = 100 →
  A + B + C = 83 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l2067_206750


namespace NUMINAMATH_CALUDE_tv_horizontal_length_l2067_206752

/-- Represents a rectangular TV screen -/
structure TVScreen where
  horizontal : ℝ
  vertical : ℝ
  diagonal : ℝ

/-- The TV screen satisfies the 16:9 aspect ratio and has a 36-inch diagonal -/
def is_valid_tv_screen (tv : TVScreen) : Prop :=
  tv.horizontal / tv.vertical = 16 / 9 ∧ 
  tv.diagonal = 36 ∧
  tv.diagonal^2 = tv.horizontal^2 + tv.vertical^2

/-- The theorem stating the horizontal length of the TV screen -/
theorem tv_horizontal_length (tv : TVScreen) 
  (h : is_valid_tv_screen tv) : 
  tv.horizontal = (16 * 36) / Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_tv_horizontal_length_l2067_206752


namespace NUMINAMATH_CALUDE_gcd_2015_15_l2067_206714

theorem gcd_2015_15 : Nat.gcd 2015 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2015_15_l2067_206714


namespace NUMINAMATH_CALUDE_total_vegetarian_count_l2067_206709

/-- Represents the dietary preferences in a family -/
structure DietaryPreferences where
  only_vegetarian : ℕ
  only_non_vegetarian : ℕ
  both_veg_and_non_veg : ℕ
  vegan : ℕ
  vegan_and_vegetarian : ℕ
  gluten_free_from_both : ℕ

/-- Calculates the total number of people eating vegetarian food -/
def total_vegetarian (d : DietaryPreferences) : ℕ :=
  d.only_vegetarian + d.both_veg_and_non_veg + (d.vegan - d.vegan_and_vegetarian)

/-- Theorem stating the total number of people eating vegetarian food -/
theorem total_vegetarian_count (d : DietaryPreferences)
  (h1 : d.only_vegetarian = 15)
  (h2 : d.only_non_vegetarian = 8)
  (h3 : d.both_veg_and_non_veg = 11)
  (h4 : d.vegan = 5)
  (h5 : d.vegan_and_vegetarian = 3)
  (h6 : d.gluten_free_from_both = 2)
  : total_vegetarian d = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetarian_count_l2067_206709


namespace NUMINAMATH_CALUDE_no_nonzero_integer_solution_l2067_206721

theorem no_nonzero_integer_solution (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_integer_solution_l2067_206721


namespace NUMINAMATH_CALUDE_largest_side_of_special_rectangle_l2067_206777

/-- A rectangle with specific properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 1920

/-- The largest side of a special rectangle is 101 --/
theorem largest_side_of_special_rectangle (r : SpecialRectangle) : 
  max r.length r.width = 101 := by
  sorry

end NUMINAMATH_CALUDE_largest_side_of_special_rectangle_l2067_206777


namespace NUMINAMATH_CALUDE_six_hardcover_books_l2067_206747

/-- Represents the purchase of a set of books with two price options --/
structure BookPurchase where
  totalVolumes : ℕ
  paperbackPrice : ℕ
  hardcoverPrice : ℕ
  totalCost : ℕ

/-- Calculates the number of hardcover books purchased --/
def hardcoverCount (purchase : BookPurchase) : ℕ :=
  sorry

/-- Theorem stating that for the given purchase scenario, 6 hardcover books were bought --/
theorem six_hardcover_books (purchase : BookPurchase) 
  (h1 : purchase.totalVolumes = 12)
  (h2 : purchase.paperbackPrice = 18)
  (h3 : purchase.hardcoverPrice = 28)
  (h4 : purchase.totalCost = 276) : 
  hardcoverCount purchase = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_hardcover_books_l2067_206747


namespace NUMINAMATH_CALUDE_worker_count_l2067_206770

theorem worker_count (total : ℕ) (extra_total : ℕ) (extra_per_worker : ℕ) : 
  (total = 300000) → 
  (extra_total = 375000) → 
  (extra_per_worker = 50) → 
  (∃ (w : ℕ), w * (extra_total / w - total / w) = extra_per_worker ∧ w = 1500) :=
by sorry

end NUMINAMATH_CALUDE_worker_count_l2067_206770


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l2067_206732

/-- The number of ways to arrange books on a shelf -/
def arrange_books (math_books : ℕ) (history_books : ℕ) (english_books : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial history_books) * (Nat.factorial english_books)

/-- Theorem: The number of ways to arrange 3 math books, 4 history books, and 5 English books
    on a shelf, where all books of the same subject must stay together and books within
    each subject are distinct, is equal to 103680. -/
theorem book_arrangement_theorem :
  arrange_books 3 4 5 = 103680 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l2067_206732


namespace NUMINAMATH_CALUDE_orange_apple_pear_weight_equivalence_l2067_206737

/-- Represents the weight of a fruit -/
structure FruitWeight where
  weight : ℝ

/-- Represents the count of fruits -/
structure FruitCount where
  count : ℕ

/-- Given 9 oranges weigh the same as 6 apples and 1 pear, 
    prove that 36 oranges weigh the same as 24 apples and 4 pears -/
theorem orange_apple_pear_weight_equivalence 
  (orange : FruitWeight) 
  (apple : FruitWeight) 
  (pear : FruitWeight) 
  (h : 9 * orange.weight = 6 * apple.weight + pear.weight) : 
  36 * orange.weight = 24 * apple.weight + 4 * pear.weight := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_pear_weight_equivalence_l2067_206737


namespace NUMINAMATH_CALUDE_blocks_required_for_specified_wall_l2067_206764

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : ℕ
  length₁ : ℕ
  length₂ : ℕ

/-- Calculates the number of blocks required for a wall with given specifications --/
def calculateBlocksRequired (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating that the number of blocks required for the specified wall is 404 --/
theorem blocks_required_for_specified_wall :
  let wall := WallDimensions.mk 150 8
  let block := BlockDimensions.mk 1 3 2
  calculateBlocksRequired wall block = 404 :=
by sorry

end NUMINAMATH_CALUDE_blocks_required_for_specified_wall_l2067_206764


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2067_206755

/-- The function f(x) = 2a^(x+1) - 3 has a fixed point at (-1, -1) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x + 1) - 3
  f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2067_206755


namespace NUMINAMATH_CALUDE_polynomial_root_product_l2067_206729

theorem polynomial_root_product (d e f : ℝ) : 
  let Q : ℝ → ℝ := λ x ↦ x^3 + d*x^2 + e*x + f
  (Q (Real.cos (π/5)) = 0) ∧ 
  (Q (Real.cos (3*π/5)) = 0) ∧ 
  (Q (Real.cos (4*π/5)) = 0) →
  d * e * f = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l2067_206729


namespace NUMINAMATH_CALUDE_trapezoid_top_side_length_l2067_206768

theorem trapezoid_top_side_length 
  (height : ℝ) 
  (area : ℝ) 
  (bottom_top_difference : ℝ) :
  height = 8 →
  area = 72 →
  bottom_top_difference = 6 →
  let bottom := (2 * area / height + bottom_top_difference) / 2
  let top := bottom - bottom_top_difference
  top = 6 := by sorry

end NUMINAMATH_CALUDE_trapezoid_top_side_length_l2067_206768


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2067_206749

theorem circle_radius_proof (x y : ℝ) (h : x + y = 100 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l2067_206749


namespace NUMINAMATH_CALUDE_f_composition_seven_l2067_206708

-- Define the function f
def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

-- State the theorem
theorem f_composition_seven : f (f (f (f (f (f 7))))) = 116 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_seven_l2067_206708


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2067_206730

/-- Proves that the cost price of one meter of cloth is 85 rupees given the selling price and profit per meter. -/
theorem cloth_cost_price
  (selling_price : ℕ)
  (cloth_length : ℕ)
  (profit_per_meter : ℕ)
  (h1 : selling_price = 8500)
  (h2 : cloth_length = 85)
  (h3 : profit_per_meter = 15) :
  (selling_price - profit_per_meter * cloth_length) / cloth_length = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2067_206730


namespace NUMINAMATH_CALUDE_convert_22_mps_to_kmph_l2067_206783

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph_factor : ℝ := 3.6

/-- Convert meters per second to kilometers per hour -/
def convert_mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * mps_to_kmph_factor

/-- Theorem: Converting 22 mps to kmph results in 79.2 kmph -/
theorem convert_22_mps_to_kmph :
  convert_mps_to_kmph 22 = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_convert_22_mps_to_kmph_l2067_206783


namespace NUMINAMATH_CALUDE_football_team_addition_l2067_206787

theorem football_team_addition : 36 + 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_football_team_addition_l2067_206787


namespace NUMINAMATH_CALUDE_gcd_problem_l2067_206785

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1729 * k) : 
  Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l2067_206785


namespace NUMINAMATH_CALUDE_florist_roses_total_l2067_206753

theorem florist_roses_total (initial : ℝ) (first_pick : ℝ) (second_pick : ℝ)
  (h1 : initial = 37.0)
  (h2 : first_pick = 16.0)
  (h3 : second_pick = 19.0) :
  initial + first_pick + second_pick = 72.0 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_total_l2067_206753


namespace NUMINAMATH_CALUDE_tigers_games_count_l2067_206743

theorem tigers_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (60 * initial_games) / 100 →
    ∀ (final_games : ℕ),
      final_games = initial_games + 11 →
      (initial_wins + 8) = (65 * final_games) / 100 →
      final_games = 28 := by
sorry

end NUMINAMATH_CALUDE_tigers_games_count_l2067_206743


namespace NUMINAMATH_CALUDE_transformation_result_l2067_206703

def initial_point : ℝ × ℝ × ℝ := (1, 1, 1)

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def transformation_sequence (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180 |> reflect_yz |> reflect_xz |> rotate_y_180 |> reflect_xz

theorem transformation_result :
  transformation_sequence initial_point = (-1, 1, 1) := by
  sorry

#eval transformation_sequence initial_point

end NUMINAMATH_CALUDE_transformation_result_l2067_206703


namespace NUMINAMATH_CALUDE_adjacent_units_conversion_rate_l2067_206762

-- Define the units of length
inductive LengthUnit
  | Kilometer
  | Meter
  | Decimeter
  | Centimeter
  | Millimeter

-- Define the concept of adjacent units
def adjacent (u v : LengthUnit) : Prop :=
  (u = LengthUnit.Kilometer ∧ v = LengthUnit.Meter) ∨
  (u = LengthUnit.Meter ∧ v = LengthUnit.Decimeter) ∨
  (u = LengthUnit.Decimeter ∧ v = LengthUnit.Centimeter) ∨
  (u = LengthUnit.Centimeter ∧ v = LengthUnit.Millimeter)

-- Define the conversion rate function
def conversionRate (u v : LengthUnit) : ℕ := 10

-- State the theorem
theorem adjacent_units_conversion_rate (u v : LengthUnit) :
  adjacent u v → conversionRate u v = 10 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_units_conversion_rate_l2067_206762


namespace NUMINAMATH_CALUDE_increasing_interval_transformed_l2067_206713

-- Define an even function f
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define an increasing function on an interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

-- Main theorem
theorem increasing_interval_transformed (f : ℝ → ℝ) :
  even_function f →
  increasing_on f 2 6 →
  increasing_on (fun x ↦ f (2 - x)) 4 8 :=
sorry

end NUMINAMATH_CALUDE_increasing_interval_transformed_l2067_206713


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2067_206717

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.000000301 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.01 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2067_206717


namespace NUMINAMATH_CALUDE_negation_equivalence_l2067_206722

theorem negation_equivalence :
  (¬ ∃ a : ℝ, a < 0 ∧ a + 4 / a ≤ -4) ↔ (∀ a : ℝ, a < 0 → a + 4 / a > -4) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2067_206722


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2067_206778

/-- A geometric sequence with first term a₁, nth term aₙ, sum Sₙ, and common ratio q. -/
structure GeometricSequence where
  a₁ : ℝ
  aₙ : ℝ
  Sₙ : ℝ
  n : ℕ
  q : ℝ
  geom_seq : a₁ * q^(n-1) = aₙ
  sum_formula : Sₙ = a₁ * (1 - q^n) / (1 - q)

/-- The common ratio of a geometric sequence with a₁ = 2, aₙ = -64, and Sₙ = -42 is -2. -/
theorem geometric_sequence_common_ratio :
  ∀ (seq : GeometricSequence),
    seq.a₁ = 2 →
    seq.aₙ = -64 →
    seq.Sₙ = -42 →
    seq.q = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2067_206778


namespace NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l2067_206718

theorem sin_negative_31pi_over_6 : Real.sin (-31 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l2067_206718


namespace NUMINAMATH_CALUDE_a_power_of_two_l2067_206740

def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * a n + 2^n

theorem a_power_of_two (k : ℕ) : ∃ m : ℕ, a (2^k) = 2^m := by
  sorry

end NUMINAMATH_CALUDE_a_power_of_two_l2067_206740


namespace NUMINAMATH_CALUDE_r_fourth_plus_inv_r_fourth_l2067_206794

theorem r_fourth_plus_inv_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inv_r_fourth_l2067_206794


namespace NUMINAMATH_CALUDE_martha_clothes_count_l2067_206715

/-- Calculates the total number of clothes Martha takes home given the number of jackets and t-shirts bought -/
def total_clothes (jackets_bought : ℕ) (tshirts_bought : ℕ) : ℕ :=
  let free_jackets := jackets_bought / 2
  let free_tshirts := tshirts_bought / 3
  (jackets_bought + free_jackets) + (tshirts_bought + free_tshirts)

/-- Proves that Martha takes home 18 clothes given the conditions of the sale and her purchases -/
theorem martha_clothes_count : total_clothes 4 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_clothes_count_l2067_206715


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l2067_206793

theorem probability_of_y_selection (p_x p_both : ℝ) (h1 : p_x = 1/3) (h2 : p_both = 0.13333333333333333) : 
  ∃ p_y : ℝ, p_y = 0.4 ∧ p_both = p_x * p_y :=
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l2067_206793


namespace NUMINAMATH_CALUDE_dean_5000th_number_l2067_206712

/-- Represents the number of numbers spoken by a player in a given round -/
def numbers_spoken (player : Nat) (round : Nat) : Nat :=
  player + round - 1

/-- Calculates the sum of numbers spoken by all players up to a given round -/
def total_numbers_spoken (round : Nat) : Nat :=
  (1 + 2 + 3 + 4) * round + (0 + 1 + 2 + 3) * (round * (round - 1) / 2)

/-- Calculates the starting number for a player in a given round -/
def start_number (player : Nat) (round : Nat) : Nat :=
  total_numbers_spoken (round - 1) + 
  (if player > 1 then (numbers_spoken 1 round + numbers_spoken 2 round + numbers_spoken 3 round) else 0) + 1

/-- The main theorem to be proved -/
theorem dean_5000th_number : 
  ∃ (round : Nat), start_number 4 round ≤ 5000 ∧ 5000 ≤ start_number 4 round + numbers_spoken 4 round - 1 :=
by sorry

end NUMINAMATH_CALUDE_dean_5000th_number_l2067_206712


namespace NUMINAMATH_CALUDE_cats_and_dogs_sum_l2067_206744

/-- Represents the number of individuals of each type on the ship --/
structure ShipPopulation where
  cats : ℕ
  parrots : ℕ
  dogs : ℕ
  sailors : ℕ
  cook : ℕ := 1
  captain : ℕ := 1

/-- The total number of heads on the ship --/
def totalHeads (p : ShipPopulation) : ℕ :=
  p.cats + p.parrots + p.dogs + p.sailors + p.cook + p.captain

/-- The total number of legs on the ship --/
def totalLegs (p : ShipPopulation) : ℕ :=
  4 * p.cats + 2 * p.parrots + 4 * p.dogs + 2 * p.sailors + 2 * p.cook + 1 * p.captain

/-- Theorem stating that the total number of cats and dogs is 14 --/
theorem cats_and_dogs_sum (p : ShipPopulation) 
    (h1 : totalHeads p = 38) 
    (h2 : totalLegs p = 103) : 
  p.cats + p.dogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_cats_and_dogs_sum_l2067_206744


namespace NUMINAMATH_CALUDE_equation_solution_l2067_206748

theorem equation_solution :
  ∃ x : ℚ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 73 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2067_206748


namespace NUMINAMATH_CALUDE_integral_power_x_l2067_206756

theorem integral_power_x (a : ℝ) (h : a > 0) : ∫ x in (0:ℝ)..1, x^a = 1 / (a + 1) := by sorry

end NUMINAMATH_CALUDE_integral_power_x_l2067_206756


namespace NUMINAMATH_CALUDE_tan_150_degrees_l2067_206734

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l2067_206734


namespace NUMINAMATH_CALUDE_balloon_difference_l2067_206780

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of balloons Jake initially brought to the park -/
def jake_initial_balloons : ℕ := 3

/-- The number of additional balloons Jake bought at the park -/
def jake_additional_balloons : ℕ := 4

/-- The total number of balloons Jake had at the park -/
def jake_total_balloons : ℕ := jake_initial_balloons + jake_additional_balloons

/-- Theorem stating the difference in balloons between Jake and Allan -/
theorem balloon_difference : jake_total_balloons - allan_balloons = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l2067_206780


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2067_206769

theorem sum_of_a_and_b (a b : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hab : |a - b| = b - a) :
  a + b = -2 ∨ a + b = -8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2067_206769


namespace NUMINAMATH_CALUDE_simplify_expression_l2067_206757

theorem simplify_expression : (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2067_206757


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2067_206719

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 2) →
  (a 5 + a 6 + a 7 + a 8 = 12) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2067_206719


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2067_206716

def U : Finset ℕ := {1, 2, 3, 4, 6}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_intersection_equals_set : (U \ (A ∩ B)) = {1, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2067_206716


namespace NUMINAMATH_CALUDE_min_distance_C₁_to_C₂_sum_distances_to_intersection_points_l2067_206705

-- Define the curves and point
def C₁ : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 1}
def C₂ : Set (ℝ × ℝ) := {(x, y) | y = x + 2}
def C₃ : Set (ℝ × ℝ) := {(x, y) | (x/2)^2 + (y/Real.sqrt 3)^2 = 1}
def P : ℝ × ℝ := (-1, 1)

-- State the theorems to be proved
theorem min_distance_C₁_to_C₂ :
  ∃ d : ℝ, d = Real.sqrt 2 - 1 ∧
  ∀ p ∈ C₁, ∀ q ∈ C₂, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
sorry

theorem sum_distances_to_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ C₂ ∧ A ∈ C₃ ∧ B ∈ C₂ ∧ B ∈ C₃ ∧
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_min_distance_C₁_to_C₂_sum_distances_to_intersection_points_l2067_206705


namespace NUMINAMATH_CALUDE_linear_system_solution_l2067_206775

theorem linear_system_solution :
  ∀ x y : ℚ,
  (2 * x + y = 6) →
  (x + 2 * y = 5) →
  ((x + y) / 3 = 11 / 9) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2067_206775


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2067_206735

/-- Given a line L1 with equation x - 2y + 1 = 0, prove that the line L2 with equation 2x + y + 1 = 0
    passes through the point (-2, 3) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 2*y + 1 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 2*x + y + 1 = 0
  let point : ℝ × ℝ := (-2, 3)
  (L2 point.1 point.2) ∧
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) *
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) =
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1)) *
    ((x2 - x1) * (y2 - y1) - (y2 - y1) * (x2 - x1))) :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2067_206735


namespace NUMINAMATH_CALUDE_polka_dot_blankets_l2067_206797

theorem polka_dot_blankets (initial_blankets : ℕ) (added_blankets : ℕ) : 
  initial_blankets = 24 →
  added_blankets = 2 →
  (initial_blankets / 3 + added_blankets : ℕ) = 10 := by
sorry

end NUMINAMATH_CALUDE_polka_dot_blankets_l2067_206797


namespace NUMINAMATH_CALUDE_lauras_remaining_pay_l2067_206772

/-- Calculates the remaining amount of Laura's pay after expenses --/
def remaining_pay (hourly_rate : ℚ) (hours_per_day : ℚ) (days_worked : ℚ) 
                  (food_clothing_percentage : ℚ) (rent : ℚ) : ℚ :=
  let total_earnings := hourly_rate * hours_per_day * days_worked
  let food_clothing_expense := total_earnings * food_clothing_percentage
  let remaining_after_food_clothing := total_earnings - food_clothing_expense
  remaining_after_food_clothing - rent

/-- Theorem stating that Laura's remaining pay is $250 --/
theorem lauras_remaining_pay :
  remaining_pay 10 8 10 (1/4) 350 = 250 := by
  sorry

end NUMINAMATH_CALUDE_lauras_remaining_pay_l2067_206772


namespace NUMINAMATH_CALUDE_certain_number_solution_l2067_206711

theorem certain_number_solution : 
  ∃ x : ℝ, (5100 - (102 / x) = 5095) ∧ (x = 20.4) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l2067_206711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2067_206790

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 4)
  (h3 : arithmetic_sequence a d)
  (h4 : geometric_sequence (a 1) (a 3) (a 4)) :
  (∀ n : ℕ, a n = 5 - n) ∧
  (∃ max_sum : ℝ, max_sum = 10 ∧
    ∀ n : ℕ, (n * (2 * a 1 + (n - 1) * d)) / 2 ≤ max_sum) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2067_206790


namespace NUMINAMATH_CALUDE_gravitational_force_on_space_station_l2067_206763

/-- Gravitational force calculation -/
theorem gravitational_force_on_space_station 
  (inverse_square_law : ∀ (d : ℝ) (f : ℝ), f * d^2 = (400 : ℝ) * 6000^2)
  (earth_surface_distance : ℝ := 6000)
  (earth_surface_force : ℝ := 400)
  (space_station_distance : ℝ := 360000) :
  (earth_surface_force * earth_surface_distance^2) / space_station_distance^2 = 1/9 := by
sorry

end NUMINAMATH_CALUDE_gravitational_force_on_space_station_l2067_206763


namespace NUMINAMATH_CALUDE_greatest_possible_median_l2067_206789

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  t = 40 →
  r ≤ 23 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 40) / 5 = 18 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 40 ∧
    r' = 23 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l2067_206789


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_nonnegative_reals_l2067_206701

open Set

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = 2 * x}
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Define the intersection set
def intersection_set : Set ℝ := {y | y ≥ 0}

-- Theorem statement
theorem A_intersect_B_equals_nonnegative_reals : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_nonnegative_reals_l2067_206701


namespace NUMINAMATH_CALUDE_fraction_problem_l2067_206788

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  (3/10 : ℝ) * N = 64.8 →
  F * ((1/4 : ℝ) * N) = 18 →
  F = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2067_206788


namespace NUMINAMATH_CALUDE_siblings_have_extra_money_l2067_206758

def perfume_cost : ℚ := 100
def christian_savings : ℚ := 7
def sue_savings : ℚ := 9
def bob_savings : ℚ := 3
def christian_yards : ℕ := 7
def christian_yard_rate : ℚ := 7
def sue_dogs : ℕ := 10
def sue_dog_rate : ℚ := 4
def bob_families : ℕ := 5
def bob_family_rate : ℚ := 2
def discount_rate : ℚ := 20 / 100

def total_earnings : ℚ :=
  christian_savings + sue_savings + bob_savings +
  christian_yards * christian_yard_rate +
  sue_dogs * sue_dog_rate +
  bob_families * bob_family_rate

def discounted_price : ℚ :=
  perfume_cost * (1 - discount_rate)

theorem siblings_have_extra_money :
  total_earnings - discounted_price = 38 := by sorry

end NUMINAMATH_CALUDE_siblings_have_extra_money_l2067_206758


namespace NUMINAMATH_CALUDE_solution_exists_l2067_206751

def f (x : ℝ) : ℝ := 2 * x - 3

def d : ℝ := 2

theorem solution_exists : ∃ x : ℝ, 2 * (f x) - 11 = f (x - d) :=
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2067_206751


namespace NUMINAMATH_CALUDE_pear_juice_percentage_l2067_206720

-- Define the juice production rates
def pear_juice_rate : ℚ := 10 / 4
def orange_juice_rate : ℚ := 12 / 3

-- Define the number of fruits used in the blend
def pears_in_blend : ℕ := 8
def oranges_in_blend : ℕ := 6

-- Define the total amount of juice in the blend
def total_juice : ℚ := pear_juice_rate * pears_in_blend + orange_juice_rate * oranges_in_blend

-- Define the amount of pear juice in the blend
def pear_juice_in_blend : ℚ := pear_juice_rate * pears_in_blend

-- Theorem statement
theorem pear_juice_percentage :
  (pear_juice_in_blend / total_juice) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_l2067_206720


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l2067_206700

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 6 → 
  (10 * x + y) - (10 * y + x) = 54 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l2067_206700


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2067_206741

theorem least_number_with_remainder (n : Nat) : n = 125 ↔ 
  (n % 12 = 5 ∧ ∀ m : Nat, m % 12 = 5 → m ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2067_206741


namespace NUMINAMATH_CALUDE_dunkers_lineups_l2067_206710

/-- The number of players in the team -/
def total_players : ℕ := 15

/-- The number of players who can't play together -/
def special_players : ℕ := 3

/-- The number of players in a starting lineup -/
def lineup_size : ℕ := 5

/-- The number of possible starting lineups -/
def possible_lineups : ℕ := 2277

/-- Function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dunkers_lineups :
  choose (total_players - special_players) lineup_size +
  special_players * choose (total_players - special_players) (lineup_size - 1) =
  possible_lineups :=
sorry

end NUMINAMATH_CALUDE_dunkers_lineups_l2067_206710


namespace NUMINAMATH_CALUDE_min_gb_for_y_cheaper_l2067_206782

/-- Cost of Plan X in cents for g gigabytes -/
def cost_x (g : ℕ) : ℕ := 15 * g

/-- Cost of Plan Y in cents for g gigabytes -/
def cost_y (g : ℕ) : ℕ :=
  if g ≤ 500 then
    3000 + 8 * g
  else
    3000 + 8 * 500 + 6 * (g - 500)

/-- Predicate to check if Plan Y is cheaper than Plan X for g gigabytes -/
def y_cheaper_than_x (g : ℕ) : Prop :=
  cost_y g < cost_x g

theorem min_gb_for_y_cheaper :
  ∀ g : ℕ, g < 778 → ¬(y_cheaper_than_x g) ∧
  y_cheaper_than_x 778 :=
sorry

end NUMINAMATH_CALUDE_min_gb_for_y_cheaper_l2067_206782


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2067_206784

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (1 / Real.sqrt (x + 2 * y + 6) + 1 / Real.sqrt (y + 2 * z + 6) + 1 / Real.sqrt (z + 2 * x + 6)) ≤
  (x / Real.sqrt (x^2 + 4 * Real.sqrt y + 4 * Real.sqrt z) +
   y / Real.sqrt (y^2 + 4 * Real.sqrt z + 4 * Real.sqrt x) +
   z / Real.sqrt (z^2 + 4 * Real.sqrt x + 4 * Real.sqrt y)) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2067_206784


namespace NUMINAMATH_CALUDE_square_plus_two_times_plus_one_equals_eleven_l2067_206796

theorem square_plus_two_times_plus_one_equals_eleven :
  let a : ℝ := Real.sqrt 11 - 1
  a^2 + 2*a + 1 = 11 := by
sorry

end NUMINAMATH_CALUDE_square_plus_two_times_plus_one_equals_eleven_l2067_206796


namespace NUMINAMATH_CALUDE_money_left_calculation_l2067_206702

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_cost := 4 * drink_cost + small_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that the money left is equal to 50 - 13q -/
theorem money_left_calculation (q : ℝ) : money_left q = 50 - 13 * q := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l2067_206702


namespace NUMINAMATH_CALUDE_quadratic_problem_l2067_206773

def quadratic_function (b c : ℝ) : ℝ → ℝ := λ x => x^2 + b*x + c

theorem quadratic_problem (b c : ℝ) :
  (∀ x, quadratic_function b c x < 0 ↔ 1 < x ∧ x < 3) →
  (quadratic_function b c = λ x => x^2 - 4*x + 3) ∧
  (∀ m, (∀ x, quadratic_function b c x > m*x - 1) ↔ -8 < m ∧ m < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_problem_l2067_206773


namespace NUMINAMATH_CALUDE_rectangle_area_l2067_206795

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2067_206795


namespace NUMINAMATH_CALUDE_exactly_two_support_probability_l2067_206776

theorem exactly_two_support_probability (p : ℝ) (h : p = 0.6) :
  let q := 1 - p
  3 * p^2 * q = 0.432 := by sorry

end NUMINAMATH_CALUDE_exactly_two_support_probability_l2067_206776


namespace NUMINAMATH_CALUDE_quadratic_roots_l2067_206745

/-- Given a quadratic function f(x) = ax^2 + bx with specific values, 
    prove that the roots of f(x) = 6 are -2 and 3. -/
theorem quadratic_roots (a b : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 + b * x)
    (h_m2 : f (-2) = 6)
    (h_m1 : f (-1) = 2)
    (h_0  : f 0 = 0)
    (h_1  : f 1 = 0)
    (h_2  : f 2 = 2)
    (h_3  : f 3 = 6) :
  (∃ x, f x = 6) ∧ (f (-2) = 6 ∧ f 3 = 6) ∧ 
  (∀ x, f x = 6 → x = -2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2067_206745


namespace NUMINAMATH_CALUDE_complex_calculation_theorem_logarithm_calculation_theorem_l2067_206760

theorem complex_calculation_theorem :
  (2 ^ (1/3) * 3 ^ (1/2)) ^ 6 + (2 * 2 ^ (1/2)) ^ (4/3) - 4 * (16/49) ^ (-1/2) - 2 ^ (1/4) * 8 ^ 0.25 - (-2005) ^ 0 = 100 :=
by sorry

theorem logarithm_calculation_theorem :
  ((1 - Real.log 3 / Real.log 6) ^ 2 + Real.log 2 / Real.log 6 * Real.log 18 / Real.log 6) / (Real.log 4 / Real.log 6) = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_theorem_logarithm_calculation_theorem_l2067_206760


namespace NUMINAMATH_CALUDE_A_must_be_four_l2067_206738

/-- Represents a six-digit number in the form 32BA33 -/
def SixDigitNumber (A : Nat) : Nat :=
  320000 + A * 100 + 33

/-- Rounds a number to the nearest hundred -/
def roundToNearestHundred (n : Nat) : Nat :=
  ((n + 50) / 100) * 100

/-- Theorem stating that if 32BA33 rounds to 323400, then A must be 4 -/
theorem A_must_be_four :
  ∀ A : Nat, A < 10 →
  roundToNearestHundred (SixDigitNumber A) = 323400 →
  A = 4 := by
sorry

end NUMINAMATH_CALUDE_A_must_be_four_l2067_206738


namespace NUMINAMATH_CALUDE_natural_numbers_less_than_two_l2067_206728

theorem natural_numbers_less_than_two :
  {n : ℕ | n < 2} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_natural_numbers_less_than_two_l2067_206728


namespace NUMINAMATH_CALUDE_median_to_AC_altitude_to_AB_l2067_206704

-- Define the triangle ABC
def A : ℝ × ℝ := (8, 5)
def B : ℝ × ℝ := (4, -2)
def C : ℝ × ℝ := (-6, 3)

-- Define the equations of the lines
def median_equation (x y : ℝ) : Prop := 2 * x + y - 6 = 0
def altitude_equation (x y : ℝ) : Prop := 4 * x + 7 * y + 3 = 0

-- Theorem for the median equation
theorem median_to_AC : 
  ∃ (m : ℝ × ℝ → ℝ × ℝ → Prop), 
    (∀ p, m ((A.1 + C.1) / 2, (A.2 + C.2) / 2) p ↔ m B p) ∧
    (∀ x y, m (x, y) (x, y) ↔ median_equation x y) :=
sorry

-- Theorem for the altitude equation
theorem altitude_to_AB :
  ∃ (l : ℝ × ℝ → ℝ × ℝ → Prop),
    (∀ p, l C p → (B.2 - A.2) * (p.1 - C.1) = (A.1 - B.1) * (p.2 - C.2)) ∧
    (∀ x y, l (x, y) (x, y) ↔ altitude_equation x y) :=
sorry

end NUMINAMATH_CALUDE_median_to_AC_altitude_to_AB_l2067_206704


namespace NUMINAMATH_CALUDE_vintik_shpuntik_journey_l2067_206706

/-- The problem of Vintik and Shpuntik's journey to school -/
theorem vintik_shpuntik_journey 
  (distance : ℝ) 
  (vintik_scooter_speed : ℝ) 
  (walking_speed : ℝ) 
  (h_distance : distance = 6) 
  (h_vintik_scooter : vintik_scooter_speed = 10) 
  (h_walking : walking_speed = 5) :
  ∃ (shpuntik_bicycle_speed : ℝ),
    -- Vintik's journey
    ∃ (vintik_time : ℝ),
      vintik_time * (vintik_scooter_speed / 2 + walking_speed / 2) = distance ∧
    -- Shpuntik's journey
    (distance / 2) / shpuntik_bicycle_speed + (distance / 2) / walking_speed = vintik_time ∧
    -- Shpuntik's bicycle speed
    shpuntik_bicycle_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_vintik_shpuntik_journey_l2067_206706


namespace NUMINAMATH_CALUDE_food_waste_scientific_notation_l2067_206767

theorem food_waste_scientific_notation :
  (530 : ℝ) * (10^9 : ℝ) = 5.3 * (10^10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_food_waste_scientific_notation_l2067_206767


namespace NUMINAMATH_CALUDE_min_tetrahedron_volume_l2067_206733

/-- Given a point P(1, 4, 5) in 3D Cartesian coordinate system O-xyz,
    and a plane passing through P intersecting positive axes at points A, B, and C,
    prove that the minimum volume V of tetrahedron O-ABC is 15. -/
theorem min_tetrahedron_volume (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_plane : 1 / a + 4 / b + 5 / c = 1) :
  (1 / 6 : ℝ) * a * b * c ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_tetrahedron_volume_l2067_206733


namespace NUMINAMATH_CALUDE_pete_and_raymond_spending_l2067_206771

theorem pete_and_raymond_spending :
  let initial_amount : ℕ := 250 -- $2.50 in cents
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let pete_nickels_spent : ℕ := 4
  let raymond_dimes_left : ℕ := 7
  
  let pete_spent : ℕ := pete_nickels_spent * nickel_value
  let raymond_spent : ℕ := initial_amount - (raymond_dimes_left * dime_value)
  let total_spent : ℕ := pete_spent + raymond_spent

  total_spent = 200
  := by sorry

end NUMINAMATH_CALUDE_pete_and_raymond_spending_l2067_206771


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l2067_206746

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l2067_206746


namespace NUMINAMATH_CALUDE_hattie_jumps_l2067_206786

theorem hattie_jumps (H : ℚ) 
  (total_jumps : H + (3/4 * H) + (2/3 * H) + (2/3 * H + 50) = 605) : 
  H = 180 := by
sorry

end NUMINAMATH_CALUDE_hattie_jumps_l2067_206786


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2067_206725

theorem complex_equation_solution (x : ℂ) : 
  Complex.abs x = 1 + 3 * Complex.I - x → x = -4 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2067_206725


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2067_206707

/-- The line mx + y - m = 0 passes through the point (1, 0) for all real m. -/
theorem line_passes_through_point :
  ∀ (m : ℝ), m * 1 + 0 - m = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2067_206707


namespace NUMINAMATH_CALUDE_B_work_time_l2067_206774

/-- The number of days it takes for B to complete the work alone -/
def days_for_B : ℝ := 20

/-- The fraction of work completed by A and B together in 2 days -/
def work_completed_in_2_days : ℝ := 1 - 0.7666666666666666

theorem B_work_time (days_for_A : ℝ) (h1 : days_for_A = 15) :
  2 * (1 / days_for_A + 1 / days_for_B) = work_completed_in_2_days := by
  sorry

#check B_work_time

end NUMINAMATH_CALUDE_B_work_time_l2067_206774


namespace NUMINAMATH_CALUDE_no_preimage_range_l2067_206765

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := Set.univ

def f (x : ℝ) : ℝ := -x^2 + 2*x - 1

theorem no_preimage_range :
  {p : ℝ | p ∈ B ∧ ∀ x ∈ A, f x ≠ p} = {p : ℝ | p ≥ -1} := by
  sorry

end NUMINAMATH_CALUDE_no_preimage_range_l2067_206765


namespace NUMINAMATH_CALUDE_baker_sales_difference_l2067_206791

theorem baker_sales_difference (cakes_made pastries_made cakes_sold pastries_sold : ℕ) : 
  cakes_made = 14 →
  pastries_made = 153 →
  cakes_sold = 97 →
  pastries_sold = 8 →
  cakes_sold - pastries_sold = 89 := by
sorry

end NUMINAMATH_CALUDE_baker_sales_difference_l2067_206791


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_l2067_206727

theorem reciprocal_of_negative_five :
  ∃ x : ℝ, x * (-5) = 1 ∧ x = -(1/5) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_l2067_206727


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2067_206731

def vector_a : ℝ × ℝ := (3, -4)

theorem angle_between_vectors (b : ℝ × ℝ) 
  (h1 : ‖b‖ = 2) 
  (h2 : vector_a.fst * b.fst + vector_a.snd * b.snd = -5) : 
  Real.arccos ((vector_a.fst * b.fst + vector_a.snd * b.snd) / (‖vector_a‖ * ‖b‖)) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2067_206731


namespace NUMINAMATH_CALUDE_total_students_amc8_l2067_206779

/-- Represents a math teacher at Pythagoras Middle School -/
inductive Teacher : Type
| Euler : Teacher
| Noether : Teacher
| Gauss : Teacher
| Riemann : Teacher

/-- Returns the number of students in a teacher's class -/
def studentsInClass (t : Teacher) : Nat :=
  match t with
  | Teacher.Euler => 13
  | Teacher.Noether => 10
  | Teacher.Gauss => 12
  | Teacher.Riemann => 7

/-- The list of all teachers at Pythagoras Middle School -/
def allTeachers : List Teacher :=
  [Teacher.Euler, Teacher.Noether, Teacher.Gauss, Teacher.Riemann]

/-- Theorem stating that the total number of students taking the AMC 8 contest is 42 -/
theorem total_students_amc8 :
  (allTeachers.map studentsInClass).sum = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_students_amc8_l2067_206779


namespace NUMINAMATH_CALUDE_simplified_robot_ratio_l2067_206759

/-- The number of animal robots Michael has -/
def michaels_robots : ℕ := 8

/-- The number of animal robots Tom has -/
def toms_robots : ℕ := 16

/-- The ratio of Tom's robots to Michael's robots -/
def robot_ratio : Rat := toms_robots / michaels_robots

theorem simplified_robot_ratio : robot_ratio = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_robot_ratio_l2067_206759


namespace NUMINAMATH_CALUDE_fourth_sample_number_l2067_206742

def systematic_sample (population_size : ℕ) (sample_size : ℕ) (sample : Finset ℕ) : Prop :=
  sample.card = sample_size ∧
  ∃ k, ∀ x ∈ sample, ∃ i, x = k + i * (population_size / sample_size)

theorem fourth_sample_number
  (population_size : ℕ)
  (sample_size : ℕ)
  (sample : Finset ℕ)
  (h1 : population_size = 56)
  (h2 : sample_size = 4)
  (h3 : 6 ∈ sample)
  (h4 : 34 ∈ sample)
  (h5 : 48 ∈ sample)
  (h6 : systematic_sample population_size sample_size sample) :
  20 ∈ sample :=
sorry

end NUMINAMATH_CALUDE_fourth_sample_number_l2067_206742


namespace NUMINAMATH_CALUDE_total_dresses_l2067_206754

theorem total_dresses (emily_dresses : ℕ) (melissa_dresses : ℕ) (debora_dresses : ℕ) : 
  emily_dresses = 16 →
  melissa_dresses = emily_dresses / 2 →
  debora_dresses = melissa_dresses + 12 →
  emily_dresses + melissa_dresses + debora_dresses = 44 := by
sorry

end NUMINAMATH_CALUDE_total_dresses_l2067_206754


namespace NUMINAMATH_CALUDE_translated_line_y_intercept_l2067_206766

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy }

/-- The original line y = x + 4 -/
def originalLine : Line :=
  { slope := 1, intercept := 4 }

/-- The amount to translate the line downwards -/
def translationAmount : ℝ := -6

theorem translated_line_y_intercept :
  (translateLine originalLine translationAmount).intercept = -2 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_y_intercept_l2067_206766


namespace NUMINAMATH_CALUDE_consecutive_lucky_tickets_exist_l2067_206726

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def is_lucky (n : ℕ) : Prop := sum_of_digits n % 7 = 0

/-- There exist two consecutive lucky bus ticket numbers -/
theorem consecutive_lucky_tickets_exist : ∃ n : ℕ, is_lucky n ∧ is_lucky (n + 1) := by sorry

end NUMINAMATH_CALUDE_consecutive_lucky_tickets_exist_l2067_206726


namespace NUMINAMATH_CALUDE_vector_b_value_l2067_206798

/-- Given two vectors a and b in ℝ², prove that b = (√2, √2) under the specified conditions. -/
theorem vector_b_value (a b : ℝ × ℝ) : 
  a = (1, 1) →                   -- a is (1,1)
  ‖b‖ = 2 →                      -- magnitude of b is 2
  ∃ (k : ℝ), b = k • a →         -- b is parallel to a
  k > 0 →                        -- a and b have the same direction
  b = (Real.sqrt 2, Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_value_l2067_206798


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2067_206724

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x^3 + 4 * x - 5) * (4 * x^4 - 3 * x^2 + 2 * x - 7) = 
    12 * x^7 + 9 * x^5 + 10 * x^4 + 53 * x^2 - 14 * x + 25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2067_206724


namespace NUMINAMATH_CALUDE_rotation_theorem_l2067_206792

/-- The original line -/
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- The point where the original line intersects the y-axis -/
def intersection_point : ℝ × ℝ := (0, -2)

/-- The rotated line -/
def rotated_line (x y : ℝ) : Prop := x + 2 * y + 4 = 0

/-- Theorem stating that rotating the original line 90° counterclockwise around the intersection point results in the rotated line -/
theorem rotation_theorem :
  ∀ (x y : ℝ),
  original_line x y →
  ∃ (x' y' : ℝ),
  (x' - intersection_point.1) ^ 2 + (y' - intersection_point.2) ^ 2 = (x - intersection_point.1) ^ 2 + (y - intersection_point.2) ^ 2 ∧
  (x' - intersection_point.1) * (x - intersection_point.1) + (y' - intersection_point.2) * (y - intersection_point.2) = 0 ∧
  rotated_line x' y' :=
sorry

end NUMINAMATH_CALUDE_rotation_theorem_l2067_206792


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l2067_206761

/-- The amount Mary spent on clothing -/
def total_spent : ℚ := 25.31

/-- The amount Mary spent on the shirt -/
def shirt_cost : ℚ := 13.04

/-- The number of shops Mary visited -/
def shops_visited : ℕ := 2

/-- The amount Mary spent on the jacket -/
def jacket_cost : ℚ := total_spent - shirt_cost

theorem jacket_cost_calculation : jacket_cost = 12.27 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l2067_206761


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l2067_206736

/-- Custom multiplication operation -/
def customMult (x y : ℝ) : ℝ := x^2 - x*y + y^2

/-- Theorem stating that 4 * 3 = 13 under the custom multiplication -/
theorem custom_mult_four_three : customMult 4 3 = 13 := by sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l2067_206736


namespace NUMINAMATH_CALUDE_cubic_root_product_sum_l2067_206723

theorem cubic_root_product_sum (p q r : ℝ) : 
  (6 * p^3 - 9 * p^2 + 14 * p - 10 = 0) →
  (6 * q^3 - 9 * q^2 + 14 * q - 10 = 0) →
  (6 * r^3 - 9 * r^2 + 14 * r - 10 = 0) →
  p * q + p * r + q * r = 7/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_sum_l2067_206723
