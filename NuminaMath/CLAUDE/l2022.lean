import Mathlib

namespace NUMINAMATH_CALUDE_divide_algebraic_expression_l2022_202264

theorem divide_algebraic_expression (a b c : ℝ) (h : b ≠ 0) :
  4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end NUMINAMATH_CALUDE_divide_algebraic_expression_l2022_202264


namespace NUMINAMATH_CALUDE_three_digit_sum_9_l2022_202260

/-- A function that generates all three-digit numbers using digits 1 to 5 -/
def generateNumbers : List (Fin 5 × Fin 5 × Fin 5) := sorry

/-- A function that checks if the sum of digits in a three-digit number is 9 -/
def sumIs9 (n : Fin 5 × Fin 5 × Fin 5) : Bool := sorry

/-- The theorem to be proved -/
theorem three_digit_sum_9 : 
  (generateNumbers.filter sumIs9).length = 19 := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_9_l2022_202260


namespace NUMINAMATH_CALUDE_y_axis_intersection_l2022_202244

/-- The quadratic function f(x) = 3x^2 - 4x + 5 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

/-- The y-axis intersection point of f(x) is (0, 5) -/
theorem y_axis_intersection :
  f 0 = 5 :=
by sorry

end NUMINAMATH_CALUDE_y_axis_intersection_l2022_202244


namespace NUMINAMATH_CALUDE_largest_difference_l2022_202268

def Digits : Finset ℕ := {1, 3, 7, 8, 9}

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 5) ∧
  (∀ d ∈ Digits, (d ∈ Finset.filter (λ x => x ∈ Digits) (Finset.range 10)))

theorem largest_difference :
  ∃ (a b : ℕ), is_valid_pair a b ∧
    ∀ (x y : ℕ), is_valid_pair x y → (a - b ≥ x - y) ∧ (a - b = 9868) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l2022_202268


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l2022_202233

/-- The ratio of the area of a square formed by connecting the centers of four adjacent unit squares
    in a 5x5 grid to the area of the entire 5x5 grid is 2/25. -/
theorem shaded_square_area_ratio : 
  let grid_side : ℕ := 5
  let unit_square_side : ℝ := 1
  let grid_area : ℝ := (grid_side ^ 2 : ℝ) * unit_square_side ^ 2
  let shaded_square_side : ℝ := Real.sqrt 2 * unit_square_side
  let shaded_square_area : ℝ := shaded_square_side ^ 2
  shaded_square_area / grid_area = 2 / 25 := by
sorry


end NUMINAMATH_CALUDE_shaded_square_area_ratio_l2022_202233


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2022_202287

/-- Given a hyperbola with the equation (x^2/a^2) - (y^2/b^2) = 1, where a > 0 and b > 0,
    if the eccentricity is 2 and the distance from the right focus to one of the asymptotes is √3,
    then the equation of the hyperbola is x^2 - (y^2/3) = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a →  -- eccentricity is 2
  b = Real.sqrt 3 →  -- distance from right focus to asymptote is √3
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2022_202287


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2022_202293

/-- Given a rectangular prism with dimensions a, b, and c, if the total surface area
    is 11 and the sum of all edge lengths is 24, then the length of the body diagonal is 5. -/
theorem rectangular_prism_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 11)  -- total surface area
  (h2 : 4 * (a + b + c) = 24) :            -- sum of all edge lengths
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2022_202293


namespace NUMINAMATH_CALUDE_repeating_37_equals_fraction_l2022_202203

/-- The repeating decimal 0.373737... -/
def repeating_37 : ℚ := 37 / 99

theorem repeating_37_equals_fraction : 
  repeating_37 = 37 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_37_equals_fraction_l2022_202203


namespace NUMINAMATH_CALUDE_sprite_liters_sprite_liters_value_l2022_202224

def maaza_liters : ℕ := 60
def pepsi_liters : ℕ := 144
def total_cans : ℕ := 143

theorem sprite_liters : ℕ :=
  let can_size := Nat.gcd maaza_liters pepsi_liters
  let maaza_cans := maaza_liters / can_size
  let pepsi_cans := pepsi_liters / can_size
  let sprite_cans := total_cans - (maaza_cans + pepsi_cans)
  sprite_cans * can_size

theorem sprite_liters_value : sprite_liters = 1512 := by sorry

end NUMINAMATH_CALUDE_sprite_liters_sprite_liters_value_l2022_202224


namespace NUMINAMATH_CALUDE_intersection_M_N_l2022_202289

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2022_202289


namespace NUMINAMATH_CALUDE_shift_graph_l2022_202216

-- Define a function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem shift_graph (h : f 0 = 1) : f ((-1) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_shift_graph_l2022_202216


namespace NUMINAMATH_CALUDE_pencil_transfer_l2022_202220

/-- Given that Gloria has 2 pencils and Lisa has 99 pencils, 
    if Lisa gives all of her pencils to Gloria, 
    then Gloria will have 101 pencils. -/
theorem pencil_transfer (gloria_initial : ℕ) (lisa_initial : ℕ) 
  (h1 : gloria_initial = 2) 
  (h2 : lisa_initial = 99) : 
  gloria_initial + lisa_initial = 101 := by
  sorry

end NUMINAMATH_CALUDE_pencil_transfer_l2022_202220


namespace NUMINAMATH_CALUDE_subtraction_property_l2022_202200

theorem subtraction_property (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_property_l2022_202200


namespace NUMINAMATH_CALUDE_rahul_savings_l2022_202249

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℚ) : 
  (1/3 : ℚ) * nsc = (1/2 : ℚ) * ppf →
  nsc + ppf = 180000 →
  ppf = 72000 := by
sorry

end NUMINAMATH_CALUDE_rahul_savings_l2022_202249


namespace NUMINAMATH_CALUDE_mrs_wonderful_class_size_l2022_202221

theorem mrs_wonderful_class_size :
  ∀ (girls boys jelly_beans_given : ℕ),
  girls + boys = 28 →
  boys = girls + 2 →
  jelly_beans_given = girls * girls + boys * boys →
  jelly_beans_given = 400 - 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_wonderful_class_size_l2022_202221


namespace NUMINAMATH_CALUDE_parabola_equation_l2022_202218

/-- A parabola passing through points (0, 5) and (3, 2) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ (b c : ℝ), y = x^2 + b*x + c ∧ 5 = c ∧ 2 = 9 + 3*b + c

/-- The specific parabola y = x^2 - 4x + 5 -/
def SpecificParabola (x y : ℝ) : Prop :=
  y = x^2 - 4*x + 5

theorem parabola_equation : ∀ x y : ℝ, Parabola x y ↔ SpecificParabola x y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2022_202218


namespace NUMINAMATH_CALUDE_inheritance_distribution_correct_l2022_202258

/-- Represents the distribution of an inheritance among three sons and a hospital. -/
structure InheritanceDistribution where
  eldest : ℕ
  middle : ℕ
  youngest : ℕ
  hospital : ℕ

/-- Checks if the given distribution satisfies the inheritance conditions. -/
def satisfies_conditions (d : InheritanceDistribution) : Prop :=
  -- Total inheritance is $1320
  d.eldest + d.middle + d.youngest + d.hospital = 1320 ∧
  -- If hospital's portion went to eldest son
  d.eldest + d.hospital = d.middle + d.youngest ∧
  -- If hospital's portion went to middle son
  d.middle + d.hospital = 2 * (d.eldest + d.youngest) ∧
  -- If hospital's portion went to youngest son
  d.youngest + d.hospital = 3 * (d.eldest + d.middle)

/-- The theorem stating that the given distribution satisfies all conditions. -/
theorem inheritance_distribution_correct : 
  satisfies_conditions ⟨55, 275, 385, 605⟩ := by
  sorry

end NUMINAMATH_CALUDE_inheritance_distribution_correct_l2022_202258


namespace NUMINAMATH_CALUDE_f_properties_l2022_202214

def f (x m : ℝ) : ℝ := |x + 1| + |x + m + 1|

theorem f_properties (m : ℝ) :
  (∀ x, f x m ≥ |m - 2|) ↔ m ≥ 1 ∧
  (m ≤ 0 → ∀ x, ¬(f (-x) m < 2*m)) ∧
  (m > 0 → ∀ x, f (-x) m < 2*m ↔ 1 - m/2 < x ∧ x < 3*m/2 + 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2022_202214


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2022_202296

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (47 * x - 35) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -708 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2022_202296


namespace NUMINAMATH_CALUDE_erased_numbers_sum_l2022_202243

/-- Represents a sequence of consecutive odd numbers -/
def OddSequence : ℕ → ℕ := λ n => 2 * n - 1

/-- Sum of the first n odd numbers -/
def SumOfOddNumbers (n : ℕ) : ℕ := n * n

theorem erased_numbers_sum (first_segment_sum second_segment_sum : ℕ) 
  (h1 : first_segment_sum = 961) 
  (h2 : second_segment_sum = 1001) : 
  ∃ (k1 k2 : ℕ), 
    k1 < k2 ∧ 
    SumOfOddNumbers (k1 - 1) = first_segment_sum ∧
    SumOfOddNumbers (k2 - 1) - SumOfOddNumbers k1 = second_segment_sum ∧
    OddSequence k1 + OddSequence k2 = 154 := by
  sorry

end NUMINAMATH_CALUDE_erased_numbers_sum_l2022_202243


namespace NUMINAMATH_CALUDE_rosy_fish_count_l2022_202239

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 21

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := total_fish - lilly_fish

theorem rosy_fish_count : rosy_fish = 11 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l2022_202239


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2022_202277

/-- Represents the coefficient of x^n in the expansion of (x^2 + x + 1)^k -/
def generalized_pascal (k n : ℕ) : ℕ := sorry

/-- The coefficient of x^8 in the expansion of (1+ax)(x^2+x+1)^5 -/
def coeff_x8 (a : ℝ) : ℝ := generalized_pascal 5 2 + a * generalized_pascal 5 1

theorem expansion_coefficient (a : ℝ) : coeff_x8 a = 75 → a = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2022_202277


namespace NUMINAMATH_CALUDE_bisection_method_accuracy_l2022_202281

theorem bisection_method_accuracy (f : ℝ → ℝ) (x₀ : ℝ) :
  ContinuousOn f (Set.Ioi 0) →
  Irrational x₀ →
  x₀ ∈ Set.Ioo 2 3 →
  f x₀ = 0 →
  ∃ (a b : ℝ), a < x₀ ∧ x₀ < b ∧ b - a ≤ 1 / 2^9 ∧ b - a > 1 / 2^8 := by
  sorry

end NUMINAMATH_CALUDE_bisection_method_accuracy_l2022_202281


namespace NUMINAMATH_CALUDE_no_tiling_with_all_tetrominoes_l2022_202238

/-- A tetromino is a shape consisting of 4 squares that can be rotated but not reflected. -/
structure Tetromino :=
  (squares : Fin 4 → (Fin 2 × Fin 2))

/-- There are exactly 7 different tetrominoes. -/
axiom num_tetrominoes : {n : ℕ // n = 7}

/-- A 4 × n rectangle. -/
def Rectangle (n : ℕ) := Fin 4 × Fin n

/-- A tiling of a rectangle with tetrominoes. -/
def Tiling (n : ℕ) := Rectangle n → Tetromino

/-- Theorem: It is impossible to tile a 4 × n rectangle with one copy of each of the 7 different tetrominoes. -/
theorem no_tiling_with_all_tetrominoes (n : ℕ) :
  ¬∃ (t : Tiling n), (∀ tetromino : Tetromino, ∃! (x : Rectangle n), t x = tetromino) :=
sorry

end NUMINAMATH_CALUDE_no_tiling_with_all_tetrominoes_l2022_202238


namespace NUMINAMATH_CALUDE_missing_number_solution_l2022_202269

theorem missing_number_solution : ∃ x : ℤ, 10111 - 10 * x * 5 = 10011 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_missing_number_solution_l2022_202269


namespace NUMINAMATH_CALUDE_dice_hidden_dots_l2022_202255

/-- Represents a standard six-sided die -/
def Die := Fin 6

/-- The sum of dots on all faces of a standard six-sided die -/
def total_dots_on_die : ℕ := (List.range 6).sum + 6

/-- The visible faces of the two dice -/
def visible_faces : List ℕ := [2, 3, 5]

/-- The theorem statement -/
theorem dice_hidden_dots :
  let total_dots := 2 * total_dots_on_die
  let visible_dots := visible_faces.sum
  total_dots - visible_dots = 32 := by sorry

end NUMINAMATH_CALUDE_dice_hidden_dots_l2022_202255


namespace NUMINAMATH_CALUDE_reach_probability_l2022_202248

-- Define the type for a point in the coordinate plane
structure Point where
  x : Int
  y : Int

-- Define the type for a step direction
inductive Direction
  | Left
  | Right
  | Up
  | Down

-- Define the function to calculate the probability
def probability_reach_target (start : Point) (target : Point) (max_steps : Nat) : Rat :=
  sorry

-- Theorem statement
theorem reach_probability :
  probability_reach_target ⟨0, 0⟩ ⟨2, 3⟩ 7 = 179 / 8192 := by sorry

end NUMINAMATH_CALUDE_reach_probability_l2022_202248


namespace NUMINAMATH_CALUDE_units_digit_of_large_power_of_seven_l2022_202245

theorem units_digit_of_large_power_of_seven : 
  ∃ n : ℕ, 7^(3^(5^2)) ≡ 3 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_large_power_of_seven_l2022_202245


namespace NUMINAMATH_CALUDE_parallelogram_base_l2022_202267

/-- Given a parallelogram with area 416 cm² and height 16 cm, its base is 26 cm. -/
theorem parallelogram_base (area height : ℝ) (h_area : area = 416) (h_height : height = 16) :
  area / height = 26 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2022_202267


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_two_sqrt_three_equals_zero_l2022_202280

theorem sqrt_twelve_minus_two_sqrt_three_equals_zero :
  Real.sqrt 12 - 2 * Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_two_sqrt_three_equals_zero_l2022_202280


namespace NUMINAMATH_CALUDE_sqrt_product_property_l2022_202251

theorem sqrt_product_property : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_property_l2022_202251


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2022_202290

/-- For a quadratic equation px^2 - 20x + 4 = 0, where p is nonzero,
    the equation has only one solution if and only if p = 25. -/
theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x, p * x^2 - 20 * x + 4 = 0) ↔ p = 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2022_202290


namespace NUMINAMATH_CALUDE_max_z_value_l2022_202213

theorem max_z_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (prod_sum_eq : x * y + x * z + y * z = 12)
  (x_pos : x > 0)
  (y_pos : y > 0)
  (z_pos : z > 0) :
  z ≤ 1 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ + 1 = 7 ∧ x₀ * y₀ + x₀ * 1 + y₀ * 1 = 12 :=
sorry

end NUMINAMATH_CALUDE_max_z_value_l2022_202213


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l2022_202242

theorem earth_inhabitable_fraction :
  let land_fraction : ℚ := 2/3
  let inhabitable_land_fraction : ℚ := 3/4
  (land_fraction * inhabitable_land_fraction : ℚ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l2022_202242


namespace NUMINAMATH_CALUDE_solution_set_theorem_min_value_g_min_value_fraction_l2022_202299

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Theorem 1: Solution set of f(x) + |x+1| < 2
theorem solution_set_theorem :
  {x : ℝ | f x + |x + 1| < 2} = {x : ℝ | 0 < x ∧ x < 2/3} :=
sorry

-- Theorem 2: Minimum value of g(x)
theorem min_value_g :
  ∀ x : ℝ, g x ≥ 2 :=
sorry

-- Theorem 3: Minimum value of 4/m + 1/n
theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 2) :
  4/m + 1/n ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_min_value_g_min_value_fraction_l2022_202299


namespace NUMINAMATH_CALUDE_johns_grocery_spend_l2022_202234

/-- Represents the cost of John's purchase at the grocery store. -/
def grocery_purchase (chip_price corn_chip_price : ℚ) (chip_quantity corn_chip_quantity : ℕ) : ℚ :=
  chip_price * chip_quantity + corn_chip_price * corn_chip_quantity

/-- Proves that John's total spend is $45 given the specified conditions. -/
theorem johns_grocery_spend :
  grocery_purchase 2 1.5 15 10 = 45 := by
sorry

end NUMINAMATH_CALUDE_johns_grocery_spend_l2022_202234


namespace NUMINAMATH_CALUDE_pig_farm_area_l2022_202240

/-- Represents a rectangular pig farm with specific properties -/
structure PigFarm where
  short_side : ℝ
  long_side : ℝ
  fence_length : ℝ
  area : ℝ

/-- Creates a PigFarm given the length of the shorter side -/
def make_pig_farm (x : ℝ) : PigFarm :=
  { short_side := x
  , long_side := 2 * x
  , fence_length := 4 * x
  , area := 2 * x * x
  }

/-- Theorem stating the area of the pig farm with given conditions -/
theorem pig_farm_area :
  ∃ (farm : PigFarm), farm.fence_length = 150 ∧ farm.area = 2812.5 := by
  sorry


end NUMINAMATH_CALUDE_pig_farm_area_l2022_202240


namespace NUMINAMATH_CALUDE_literary_club_probability_l2022_202202

theorem literary_club_probability : 
  let num_clubs : ℕ := 2
  let num_students : ℕ := 3
  let total_outcomes : ℕ := num_clubs ^ num_students
  let same_club_outcomes : ℕ := num_clubs
  let diff_club_probability : ℚ := 1 - (same_club_outcomes : ℚ) / total_outcomes
  diff_club_probability = 3/4 := by sorry

end NUMINAMATH_CALUDE_literary_club_probability_l2022_202202


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l2022_202232

/-- Represents the Stewart farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  total_horse_food : ℕ
  food_per_horse : ℕ

/-- Calculates the number of horses on the farm -/
def num_horses (f : Farm) : ℕ := f.total_horse_food / f.food_per_horse

/-- Calculates the ratio of sheep to horses as a pair of natural numbers -/
def sheep_to_horse_ratio (f : Farm) : ℕ × ℕ :=
  let gcd := Nat.gcd f.sheep (num_horses f)
  (f.sheep / gcd, num_horses f / gcd)

/-- Theorem stating that for the given farm conditions, the sheep to horse ratio is 2:7 -/
theorem stewart_farm_ratio :
  let f : Farm := ⟨16, 12880, 230⟩
  sheep_to_horse_ratio f = (2, 7) := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l2022_202232


namespace NUMINAMATH_CALUDE_nelly_paid_correct_amount_l2022_202226

/-- Nelly's payment for a painting at an auction -/
def nellys_payment (joe_bid sarah_bid : ℕ) : ℕ :=
  max
    (3 * joe_bid + 2000)
    (4 * sarah_bid + 1500)

/-- Theorem stating the correct amount Nelly paid for the painting -/
theorem nelly_paid_correct_amount :
  nellys_payment 160000 50000 = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_paid_correct_amount_l2022_202226


namespace NUMINAMATH_CALUDE_pfd_product_theorem_l2022_202206

/-- Partial fraction decomposition coefficients -/
structure PFDCoefficients where
  A : ℚ
  B : ℚ
  C : ℚ

/-- The partial fraction decomposition of (x^2 - 25) / ((x - 1)(x + 3)(x - 4)) -/
def partial_fraction_decomposition : (ℚ → ℚ) → PFDCoefficients → Prop :=
  λ f coeffs =>
    ∀ x, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 →
      f x = coeffs.A / (x - 1) + coeffs.B / (x + 3) + coeffs.C / (x - 4)

/-- The original rational function -/
def original_function (x : ℚ) : ℚ :=
  (x^2 - 25) / ((x - 1) * (x + 3) * (x - 4))

theorem pfd_product_theorem :
  ∃ coeffs : PFDCoefficients,
    partial_fraction_decomposition original_function coeffs ∧
    coeffs.A * coeffs.B * coeffs.C = 24/49 := by
  sorry

end NUMINAMATH_CALUDE_pfd_product_theorem_l2022_202206


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2022_202262

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first four terms of the sequence equals 30. -/
def SumEquals30 (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 = 30

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumEquals30 a) : a 2 + a 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2022_202262


namespace NUMINAMATH_CALUDE_unique_exponent_solution_l2022_202278

theorem unique_exponent_solution :
  ∃! w : ℤ, (3 : ℝ) ^ 6 * (3 : ℝ) ^ w = (3 : ℝ) ^ 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_exponent_solution_l2022_202278


namespace NUMINAMATH_CALUDE_min_value_f_l2022_202256

/-- The function f(x) = 12x - x³ -/
def f (x : ℝ) : ℝ := 12 * x - x^3

/-- The theorem stating that the minimum value of f(x) on [-3, 3] is -16 -/
theorem min_value_f : 
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-3) 3 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-3) 3 → f x ≥ f x₀) ∧
  f x₀ = -16 := by
  sorry


end NUMINAMATH_CALUDE_min_value_f_l2022_202256


namespace NUMINAMATH_CALUDE_carla_drink_problem_l2022_202274

theorem carla_drink_problem (w s : ℝ) 
  (h1 : s = 3 * w - 6)
  (h2 : s + w = 54) : 
  w = 15 := by
sorry

end NUMINAMATH_CALUDE_carla_drink_problem_l2022_202274


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l2022_202265

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := 3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $11 --/
theorem sheila_hourly_wage :
  let sheila_schedule := WorkSchedule.mk 8 6 396
  hourly_wage sheila_schedule = 11 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l2022_202265


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l2022_202275

theorem greatest_integer_for_all_real_domain : ∃ (a : ℤ),
  (∀ (x : ℝ), (x^2 + a*x + 15 ≠ 0)) ∧
  (∀ (b : ℤ), b > a → ∃ (x : ℝ), x^2 + b*x + 15 = 0) ∧
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l2022_202275


namespace NUMINAMATH_CALUDE_integer_solution_l2022_202217

theorem integer_solution (x : ℤ) : x + 8 > 9 ∧ -3*x > -15 → x = 2 ∨ x = 3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_l2022_202217


namespace NUMINAMATH_CALUDE_drinks_calculation_l2022_202222

/-- Given a number of pitchers and the number of glasses each pitcher can fill,
    calculate the total number of glasses that can be filled. -/
def total_glasses (num_pitchers : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  num_pitchers * glasses_per_pitcher

/-- Theorem: With 9 pitchers and 6 glasses per pitcher, the total number of glasses is 54. -/
theorem drinks_calculation :
  total_glasses 9 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_drinks_calculation_l2022_202222


namespace NUMINAMATH_CALUDE_min_stamps_47_cents_l2022_202291

/-- Represents the number of ways to make 47 cents using 5 and 7 cent stamps -/
def stamp_combinations : Set (ℕ × ℕ) :=
  {(c, f) | c * 5 + f * 7 = 47 ∧ c ≥ 0 ∧ f ≥ 0}

/-- The total number of stamps used in a combination -/
def total_stamps (combo : ℕ × ℕ) : ℕ :=
  combo.1 + combo.2

/-- The theorem stating the minimum number of stamps needed is 7 -/
theorem min_stamps_47_cents :
  ∃ (min_combo : ℕ × ℕ),
    min_combo ∈ stamp_combinations ∧
    ∀ combo ∈ stamp_combinations, total_stamps min_combo ≤ total_stamps combo ∧
    total_stamps min_combo = 7 :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_47_cents_l2022_202291


namespace NUMINAMATH_CALUDE_remainder_equality_l2022_202292

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_equality : 
  (sum_factorials 20) % 21 = (sum_factorials 4) % 21 := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l2022_202292


namespace NUMINAMATH_CALUDE_smallest_zack_students_correct_l2022_202257

/-- Represents the number of students in a group for each tutor -/
structure TutorGroup where
  zack : Nat
  karen : Nat
  julie : Nat

/-- Represents the ratio of students for each tutor -/
structure TutorRatio where
  zack : Nat
  karen : Nat
  julie : Nat

/-- The smallest number of students Zack can have given the conditions -/
def smallestZackStudents (g : TutorGroup) (r : TutorRatio) : Nat :=
  630

theorem smallest_zack_students_correct (g : TutorGroup) (r : TutorRatio) :
  g.zack = 14 →
  g.karen = 10 →
  g.julie = 15 →
  r.zack = 3 →
  r.karen = 2 →
  r.julie = 5 →
  smallestZackStudents g r = 630 ∧
  smallestZackStudents g r % g.zack = 0 ∧
  (smallestZackStudents g r / r.zack * r.karen) % g.karen = 0 ∧
  (smallestZackStudents g r / r.zack * r.julie) % g.julie = 0 ∧
  ∀ n : Nat, n < smallestZackStudents g r →
    (n % g.zack = 0 ∧ (n / r.zack * r.karen) % g.karen = 0 ∧ (n / r.zack * r.julie) % g.julie = 0) →
    False :=
by
  sorry

#check smallest_zack_students_correct

end NUMINAMATH_CALUDE_smallest_zack_students_correct_l2022_202257


namespace NUMINAMATH_CALUDE_jennys_bottle_cap_distance_l2022_202261

theorem jennys_bottle_cap_distance (x : ℝ) : 
  (x + (1/3) * x) + 21 = (15 + 2 * 15) → x = 18 := by sorry

end NUMINAMATH_CALUDE_jennys_bottle_cap_distance_l2022_202261


namespace NUMINAMATH_CALUDE_cos_power_six_expansion_l2022_202270

theorem cos_power_six_expansion (b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ θ : ℝ, Real.cos θ ^ 6 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) +
    b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 = 131 / 512 :=
by sorry

end NUMINAMATH_CALUDE_cos_power_six_expansion_l2022_202270


namespace NUMINAMATH_CALUDE_work_done_by_resultant_force_l2022_202252

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two 2D vectors -/
def dotProduct (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Adds two 2D vectors -/
def addVectors (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

/-- Calculates the work done by a force over a displacement -/
def workDone (force displacement : Vector2D) : ℝ :=
  dotProduct force displacement

theorem work_done_by_resultant_force : 
  let f1 : Vector2D := ⟨3, -4⟩
  let f2 : Vector2D := ⟨2, -5⟩
  let f3 : Vector2D := ⟨3, 1⟩
  let a : Vector2D := ⟨1, 1⟩
  let b : Vector2D := ⟨0, 5⟩
  let resultantForce := addVectors (addVectors f1 f2) f3
  let displacement := ⟨b.x - a.x, b.y - a.y⟩
  workDone resultantForce displacement = -40 := by
  sorry


end NUMINAMATH_CALUDE_work_done_by_resultant_force_l2022_202252


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l2022_202284

/-- Represents the number of students in each grade --/
structure Students where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- The ratio of 9th-graders to 7th-graders is 4:5 --/
def ratio_ninth_to_seventh (s : Students) : Prop :=
  5 * s.ninth = 4 * s.seventh

/-- The ratio of 9th-graders to 8th-graders is 7:6 --/
def ratio_ninth_to_eighth (s : Students) : Prop :=
  6 * s.ninth = 7 * s.eighth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.eighth + s.seventh

/-- The statement to be proved --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_ninth_to_seventh s ∧
    ratio_ninth_to_eighth s ∧
    total_students s = 87 ∧
    (∀ (t : Students),
      ratio_ninth_to_seventh t ∧
      ratio_ninth_to_eighth t →
      total_students t ≥ 87) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l2022_202284


namespace NUMINAMATH_CALUDE_original_sheet_area_l2022_202225

/-- Represents the dimensions and properties of a cardboard box created from a rectangular sheet. -/
structure CardboardBox where
  base_length : ℝ
  base_width : ℝ
  volume : ℝ

/-- Theorem stating that given the specified conditions, the original sheet area is 110 cm². -/
theorem original_sheet_area
  (box : CardboardBox)
  (base_length_eq : box.base_length = 5)
  (base_width_eq : box.base_width = 4)
  (volume_eq : box.volume = 60)
  : ℝ :=
by
  -- The proof goes here
  sorry

#check original_sheet_area

end NUMINAMATH_CALUDE_original_sheet_area_l2022_202225


namespace NUMINAMATH_CALUDE_triangle_side_length_l2022_202273

theorem triangle_side_length (a b c : ℝ) (C : Real) :
  a + b = 5 →
  a * b = 2 →
  C = Real.pi / 3 →  -- 60° in radians
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2022_202273


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l2022_202282

/-- The number of additional people needed to mow a lawn in a shorter time -/
def additional_people_needed (initial_people initial_time target_time : ℕ) : ℕ :=
  (initial_people * initial_time / target_time) - initial_people

/-- Proof that 24 additional people are needed to mow the lawn in 2 hours -/
theorem lawn_mowing_problem :
  additional_people_needed 8 8 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l2022_202282


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l2022_202204

-- Define the polynomial Q(x) = x³ - 5x² + 6x - 2
def Q (x : ℝ) : ℝ := x^3 - 5*x^2 + 6*x - 2

-- Theorem statement
theorem cubic_polynomial_root :
  -- Q is a monic cubic polynomial with integer coefficients
  (∀ x, Q x = x^3 - 5*x^2 + 6*x - 2) ∧
  -- The leading coefficient is 1 (monic)
  (∃ a b c, ∀ x, Q x = x^3 + a*x^2 + b*x + c) ∧
  -- All coefficients are integers
  (∃ a b c : ℤ, ∀ x, Q x = x^3 + a*x^2 + b*x + c) ∧
  -- √2 + 2 is a root of Q
  Q (Real.sqrt 2 + 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l2022_202204


namespace NUMINAMATH_CALUDE_driving_time_to_airport_l2022_202286

-- Define time in minutes since midnight
def flight_time : ℕ := 20 * 60
def check_in_buffer : ℕ := 2 * 60
def house_departure_time : ℕ := 17 * 60
def parking_and_terminal_time : ℕ := 15

-- Theorem statement
theorem driving_time_to_airport :
  let check_in_time := flight_time - check_in_buffer
  let airport_arrival_time := check_in_time - parking_and_terminal_time
  airport_arrival_time - house_departure_time = 45 := by
sorry

end NUMINAMATH_CALUDE_driving_time_to_airport_l2022_202286


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2022_202272

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), x + y = r ∧ x^2 + y^2 = r) → 
  (∀ (x y : ℝ), x + y = r → x^2 + y^2 ≥ r) → 
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2022_202272


namespace NUMINAMATH_CALUDE_ladder_distance_l2022_202212

theorem ladder_distance (c a b : ℝ) : 
  c = 25 → a = 20 → c^2 = a^2 + b^2 → b = 15 :=
by sorry

end NUMINAMATH_CALUDE_ladder_distance_l2022_202212


namespace NUMINAMATH_CALUDE_total_gum_packages_l2022_202271

theorem total_gum_packages : ∀ (robin_pieces_per_package : ℕ) 
                               (robin_extra_pieces : ℕ) 
                               (robin_total_pieces : ℕ)
                               (alex_pieces_per_package : ℕ) 
                               (alex_extra_pieces : ℕ) 
                               (alex_total_pieces : ℕ),
  robin_pieces_per_package = 7 →
  robin_extra_pieces = 6 →
  robin_total_pieces = 41 →
  alex_pieces_per_package = 5 →
  alex_extra_pieces = 3 →
  alex_total_pieces = 23 →
  (robin_total_pieces - robin_extra_pieces) / robin_pieces_per_package +
  (alex_total_pieces - alex_extra_pieces) / alex_pieces_per_package = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_total_gum_packages_l2022_202271


namespace NUMINAMATH_CALUDE_cosine_product_equality_l2022_202228

theorem cosine_product_equality : Real.cos (2 * Real.pi / 31) * Real.cos (4 * Real.pi / 31) * Real.cos (8 * Real.pi / 31) * Real.cos (16 * Real.pi / 31) * Real.cos (32 * Real.pi / 31) * 3.418 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equality_l2022_202228


namespace NUMINAMATH_CALUDE_bankers_gain_example_l2022_202205

/-- Calculate the banker's gain given the banker's discount, time, and interest rate. -/
def bankers_gain (bankers_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let face_value := (bankers_discount * 100) / (rate * time)
  let true_discount := (face_value * rate * time) / (100 + rate * time)
  bankers_discount - true_discount

/-- Theorem stating that the banker's gain is 360 given the specified conditions. -/
theorem bankers_gain_example : 
  bankers_gain 1360 3 12 = 360 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_example_l2022_202205


namespace NUMINAMATH_CALUDE_accurate_estimation_l2022_202237

/-- Represents a scale with a lower and upper bound -/
structure Scale where
  lower : ℝ
  upper : ℝ
  h : lower < upper

/-- Represents the position of an arrow on the scale -/
def ArrowPosition (s : Scale) := {x : ℝ // s.lower ≤ x ∧ x ≤ s.upper}

/-- The set of possible readings -/
def PossibleReadings : Set ℝ := {10.1, 10.2, 10.3, 10.4, 10.5}

/-- Function to determine the most accurate estimation -/
noncomputable def mostAccurateEstimation (s : Scale) (arrow : ArrowPosition s) : ℝ :=
  sorry

/-- Theorem stating that 10.3 is the most accurate estimation -/
theorem accurate_estimation (s : Scale) (arrow : ArrowPosition s) 
    (h1 : s.lower = 10.15) (h2 : s.upper = 10.4) : 
    mostAccurateEstimation s arrow = 10.3 := by
  sorry

end NUMINAMATH_CALUDE_accurate_estimation_l2022_202237


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2022_202219

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2022_202219


namespace NUMINAMATH_CALUDE_marinara_stains_count_l2022_202215

def grass_stain_time : ℕ := 4
def marinara_stain_time : ℕ := 7
def num_grass_stains : ℕ := 3
def total_soak_time : ℕ := 19

theorem marinara_stains_count :
  ∃ (num_marinara_stains : ℕ),
    num_marinara_stains * marinara_stain_time + num_grass_stains * grass_stain_time = total_soak_time ∧
    num_marinara_stains = 1 := by
  sorry

end NUMINAMATH_CALUDE_marinara_stains_count_l2022_202215


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2022_202229

/-- Given an arithmetic sequence 3, 7, 11, ..., x, y, 31, prove that x + y = 50 -/
theorem arithmetic_sequence_sum (x y : ℝ) : 
  (∃ (a : ℕ → ℝ), a 0 = 3 ∧ a 1 = 7 ∧ a 2 = 11 ∧ (∃ i j : ℕ, a i = x ∧ a (i + 1) = y ∧ a (j + 2) = 31) ∧ 
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)) → 
  x + y = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2022_202229


namespace NUMINAMATH_CALUDE_infinitely_many_composite_numbers_l2022_202253

theorem infinitely_many_composite_numbers :
  ∃ (N : Set ℕ), Set.Infinite N ∧
    ∀ n ∈ N, ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 50^n + (50*n + 1)^50 = a * b :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_numbers_l2022_202253


namespace NUMINAMATH_CALUDE_equation_solution_l2022_202283

theorem equation_solution : ∃! x : ℝ, x ≥ 2 ∧ 
  Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 3 ∧ 
  x = 44.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2022_202283


namespace NUMINAMATH_CALUDE_tangent_circles_count_l2022_202208

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2 ∨
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius - c2.radius)^2

/-- The theorem statement -/
theorem tangent_circles_count 
  (C1 C2 : Circle)
  (h1 : C1.radius = 2)
  (h2 : C2.radius = 2)
  (h3 : are_tangent C1 C2) :
  ∃! (s : Finset Circle), 
    (∀ c ∈ s, c.radius = 4 ∧ are_tangent c C1 ∧ are_tangent c C2) ∧ 
    s.card = 4 :=
  sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l2022_202208


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l2022_202298

theorem gcd_powers_of_two : Nat.gcd (2^115 - 1) (2^105 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l2022_202298


namespace NUMINAMATH_CALUDE_school_bus_seats_l2022_202247

theorem school_bus_seats (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 60) (h2 : num_buses = 6) (h3 : total_students % num_buses = 0) :
  total_students / num_buses = 10 := by
sorry

end NUMINAMATH_CALUDE_school_bus_seats_l2022_202247


namespace NUMINAMATH_CALUDE_specific_cone_volume_l2022_202254

/-- Represents a cone with given slant height and lateral area -/
structure Cone where
  slant_height : ℝ
  lateral_area : ℝ

/-- The volume of a cone given its slant height and lateral area -/
def cone_volume (c : Cone) : ℝ :=
  sorry

/-- Theorem stating that a cone with slant height 3 and lateral area 3√5π has volume 10π/3 -/
theorem specific_cone_volume :
  let c : Cone := { slant_height := 3, lateral_area := 3 * Real.sqrt 5 * Real.pi }
  cone_volume c = 10 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_cone_volume_l2022_202254


namespace NUMINAMATH_CALUDE_transportation_problem_l2022_202210

/-- Represents a transportation plan -/
structure TransportPlan where
  a_trucks : ℕ
  b_trucks : ℕ

/-- Represents the problem parameters -/
structure ProblemParams where
  a_capacity : ℕ
  b_capacity : ℕ
  a_cost : ℕ
  b_cost : ℕ
  total_goods : ℕ

def is_valid_plan (params : ProblemParams) (plan : TransportPlan) : Prop :=
  params.a_capacity * plan.a_trucks + params.b_capacity * plan.b_trucks = params.total_goods

def plan_cost (params : ProblemParams) (plan : TransportPlan) : ℕ :=
  params.a_cost * plan.a_trucks + params.b_cost * plan.b_trucks

def is_most_cost_effective (params : ProblemParams) (plan : TransportPlan) : Prop :=
  is_valid_plan params plan ∧
  ∀ other_plan : TransportPlan, is_valid_plan params other_plan →
    plan_cost params plan ≤ plan_cost params other_plan

theorem transportation_problem :
  ∃ (params : ProblemParams) (best_plan : TransportPlan),
    params.a_capacity = 20 ∧
    params.b_capacity = 15 ∧
    params.a_cost = 500 ∧
    params.b_cost = 400 ∧
    params.total_goods = 190 ∧
    best_plan.a_trucks = 8 ∧
    best_plan.b_trucks = 2 ∧
    plan_cost params best_plan = 4800 ∧
    (1 * params.a_capacity + 2 * params.b_capacity = 50) ∧
    (5 * params.a_capacity + 4 * params.b_capacity = 160) ∧
    is_most_cost_effective params best_plan :=
by
  sorry

end NUMINAMATH_CALUDE_transportation_problem_l2022_202210


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l2022_202235

theorem inverse_proportion_relationship (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁)
  (h2 : y₂ = 2 / x₂)
  (h3 : x₁ > 0)
  (h4 : 0 > x₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l2022_202235


namespace NUMINAMATH_CALUDE_min_sum_sequence_l2022_202285

theorem min_sum_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  (∃ r : ℚ, C - B = B - A ∧ C / B = r ∧ D / C = r) →
  C / B = 7 / 3 →
  A + B + C + D ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_sequence_l2022_202285


namespace NUMINAMATH_CALUDE_correct_bird_count_l2022_202230

/-- Given a number of feet on tree branches and the number of feet per bird,
    calculate the number of birds on the tree. -/
def birds_on_tree (total_feet : ℕ) (feet_per_bird : ℕ) : ℕ :=
  total_feet / feet_per_bird

theorem correct_bird_count : birds_on_tree 92 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_correct_bird_count_l2022_202230


namespace NUMINAMATH_CALUDE_systematic_sampling_40th_number_l2022_202211

/-- Given a systematic sample of 50 students from 1000, with the first number drawn being 0015,
    prove that the 40th number drawn is 0795. -/
theorem systematic_sampling_40th_number
  (total_students : Nat)
  (sample_size : Nat)
  (first_number : Nat)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_number = 15)
  : (first_number + (39 * (total_students / sample_size))) % total_students = 795 := by
  sorry

#eval (15 + (39 * (1000 / 50))) % 1000  -- Should output 795

end NUMINAMATH_CALUDE_systematic_sampling_40th_number_l2022_202211


namespace NUMINAMATH_CALUDE_find_y_l2022_202227

def v (y : ℝ) : Fin 2 → ℝ := ![1, y]
def w : Fin 2 → ℝ := ![9, 3]
def proj_w_v : Fin 2 → ℝ := ![-6, -2]

theorem find_y : ∃ y : ℝ, v y = v y ∧ w = w ∧ proj_w_v = proj_w_v → y = -23 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2022_202227


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2022_202201

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2022_202201


namespace NUMINAMATH_CALUDE_cross_section_ratio_cube_l2022_202236

theorem cross_section_ratio_cube (a : ℝ) (ha : a > 0) :
  let cube_diagonal := a * Real.sqrt 3
  let min_area := (a / Real.sqrt 2) * cube_diagonal
  let max_area := Real.sqrt 2 * a^2
  max_area / min_area = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_cross_section_ratio_cube_l2022_202236


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2022_202279

theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2022_202279


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2022_202263

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_problem (a : ℝ) : 
  (f_derivative a 1 * (2 - 1) + f a 1 = 7) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2022_202263


namespace NUMINAMATH_CALUDE_plan_A_rate_correct_l2022_202241

/-- The per-minute charge after the first 5 minutes under plan A -/
def plan_A_rate : ℝ := 0.06

/-- The fixed charge for the first 5 minutes under plan A -/
def plan_A_fixed_charge : ℝ := 0.60

/-- The per-minute charge under plan B -/
def plan_B_rate : ℝ := 0.08

/-- The duration at which both plans cost the same -/
def equal_cost_duration : ℝ := 14.999999999999996

theorem plan_A_rate_correct :
  plan_A_rate * (equal_cost_duration - 5) + plan_A_fixed_charge =
  plan_B_rate * equal_cost_duration := by sorry

end NUMINAMATH_CALUDE_plan_A_rate_correct_l2022_202241


namespace NUMINAMATH_CALUDE_max_sphere_cone_volume_ratio_l2022_202276

/-- The maximum volume ratio of a sphere inscribed in a cone to the cone itself -/
theorem max_sphere_cone_volume_ratio :
  ∃ (r m R : ℝ) (α : ℝ),
    r > 0 ∧ m > 0 ∧ R > 0 ∧ 0 < α ∧ α < π / 2 ∧
    r = m * Real.tan α ∧
    R = (m - R) * Real.sin α ∧
    ∀ (r' m' R' : ℝ) (α' : ℝ),
      r' > 0 → m' > 0 → R' > 0 → 0 < α' → α' < π / 2 →
      r' = m' * Real.tan α' →
      R' = (m' - R') * Real.sin α' →
      (4 / 3 * π * R' ^ 3) / ((1 / 3) * π * r' ^ 2 * m') ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_cone_volume_ratio_l2022_202276


namespace NUMINAMATH_CALUDE_part_I_part_II_l2022_202295

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

-- Define the set A
def A (m : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ y = f m x}

-- Theorem for part (I)
theorem part_I (m : ℝ) : 
  (∀ x, f m x ≥ x - m*x) → m ∈ Set.Icc (-7 : ℝ) 1 := by sorry

-- Theorem for part (II)
theorem part_II : 
  (∃ m : ℝ, A m ⊆ Set.Ici 0 ∧ ∀ m' : ℝ, A m' ⊆ Set.Ici 0 → m' ≤ m) → 
  (∃ m : ℝ, m = 1 ∧ A m ⊆ Set.Ici 0 ∧ ∀ m' : ℝ, A m' ⊆ Set.Ici 0 → m' ≤ m) := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l2022_202295


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2022_202209

theorem diophantine_equation_solution (x y : ℤ) : x^4 - 2*y^2 = 1 → x = 1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2022_202209


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l2022_202288

/-- For a polynomial (x-m)(x-n), the condition for it to not contain a linear term in x is m + n = 0. -/
theorem no_linear_term_condition (x m n : ℝ) : 
  (∀ (a b c : ℝ), (x - m) * (x - n) = a * x^2 + c → b = 0) ↔ m + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l2022_202288


namespace NUMINAMATH_CALUDE_count_convex_polygons_l2022_202297

/-- The number of ways to select 33 non-adjacent vertices from a 100-vertex polygon -/
def select_nonadjacent_vertices (n m : ℕ) : ℕ :=
  Nat.choose (n - 33 + 1) 33 + Nat.choose (n - 34 + 1) 32

/-- Theorem: The number of ways to select 33 non-adjacent vertices from a 100-vertex polygon,
    ensuring no shared sides, is equal to ⁽⁶⁷₃₃⁾ + ⁽⁶⁶₃₂⁾ -/
theorem count_convex_polygons :
  select_nonadjacent_vertices 100 33 = Nat.choose 67 33 + Nat.choose 66 32 := by
  sorry

end NUMINAMATH_CALUDE_count_convex_polygons_l2022_202297


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2022_202246

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → (3 ∣ a) ∨ (3 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2022_202246


namespace NUMINAMATH_CALUDE_total_wages_calculation_l2022_202259

/-- Represents the number of days it takes for a worker to complete the job alone -/
structure WorkerSpeed where
  days : ℕ

/-- Represents the wages received by a worker -/
structure Wages where
  amount : ℕ

/-- Calculates the total wages for a job given the speeds of two workers and the wages of one worker -/
def calculateTotalWages (workerA workerB : WorkerSpeed) (wagesA : Wages) : Wages :=
  sorry

theorem total_wages_calculation (workerA workerB : WorkerSpeed) (wagesA : Wages) :
  workerA.days = 10 →
  workerB.days = 15 →
  wagesA.amount = 1860 →
  (calculateTotalWages workerA workerB wagesA).amount = 3100 := by
  sorry

end NUMINAMATH_CALUDE_total_wages_calculation_l2022_202259


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2022_202223

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2022_202223


namespace NUMINAMATH_CALUDE_alternating_exponent_inequality_l2022_202250

theorem alternating_exponent_inequality (n : ℕ) (h : n ≥ 1) :
  2^(3^n) > 3^(2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_alternating_exponent_inequality_l2022_202250


namespace NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_ace_l2022_202231

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of cards that are either diamonds or aces -/
def targetCards : ℕ := 16

/-- The probability of drawing a card that is neither a diamond nor an ace -/
def probNonTarget : ℚ := (deckSize - targetCards) / deckSize

/-- The number of draws -/
def numDraws : ℕ := 3

theorem probability_at_least_one_diamond_or_ace :
  1 - probNonTarget ^ numDraws = 1468 / 2197 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_ace_l2022_202231


namespace NUMINAMATH_CALUDE_distance_between_trees_l2022_202266

/-- Given a curved path of length 300 meters with 26 trees planted at equal arc lengths,
    including one at each end, the distance between consecutive trees is 12 meters. -/
theorem distance_between_trees (path_length : ℝ) (num_trees : ℕ) :
  path_length = 300 ∧ num_trees = 26 →
  (path_length / (num_trees - 1 : ℝ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2022_202266


namespace NUMINAMATH_CALUDE_shaded_area_proof_shaded_area_is_sqrt_288_l2022_202294

theorem shaded_area_proof (small_square_area : ℝ) 
  (h1 : small_square_area = 3) 
  (num_small_squares : ℕ) 
  (h2 : num_small_squares = 9) : ℝ :=
  let small_square_side := Real.sqrt small_square_area
  let small_square_diagonal := small_square_side * Real.sqrt 2
  let large_square_side := 2 * small_square_diagonal + small_square_side
  let large_square_area := large_square_side ^ 2
  let total_small_squares_area := num_small_squares * small_square_area
  let shaded_area := large_square_area - total_small_squares_area
  Real.sqrt 288

theorem shaded_area_is_sqrt_288 : shaded_area_proof 3 rfl 9 rfl = Real.sqrt 288 := by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_shaded_area_is_sqrt_288_l2022_202294


namespace NUMINAMATH_CALUDE_dow_jones_problem_l2022_202207

/-- The Dow Jones Industrial Average problem -/
theorem dow_jones_problem (end_value : ℝ) (percent_fall : ℝ) :
  end_value = 8722 →
  percent_fall = 2 →
  (1 - percent_fall / 100) * 8900 = end_value :=
by
  sorry

end NUMINAMATH_CALUDE_dow_jones_problem_l2022_202207
