import Mathlib

namespace NUMINAMATH_CALUDE_count_rational_roots_l219_21960

/-- The number of different possible rational roots for a polynomial of the form
    12x^4 + b₃x³ + b₂x² + b₁x + 18 = 0 with integer coefficients -/
def num_rational_roots (b₃ b₂ b₁ : ℤ) : ℕ := 28

/-- Theorem stating that the number of different possible rational roots for the given polynomial is 28 -/
theorem count_rational_roots (b₃ b₂ b₁ : ℤ) : 
  num_rational_roots b₃ b₂ b₁ = 28 := by sorry

end NUMINAMATH_CALUDE_count_rational_roots_l219_21960


namespace NUMINAMATH_CALUDE_completing_square_sum_l219_21923

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 6*x = 1

-- Define the transformed equation
def transformed_equation (x m n : ℝ) : Prop := (x - m)^2 = n

-- Theorem statement
theorem completing_square_sum (m n : ℝ) :
  (∀ x, original_equation x ↔ transformed_equation x m n) →
  m + n = 13 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l219_21923


namespace NUMINAMATH_CALUDE_certain_value_proof_l219_21958

theorem certain_value_proof (n v : ℝ) : n = 10 → (1/2) * n + v = 11 → v = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l219_21958


namespace NUMINAMATH_CALUDE_cone_surface_area_l219_21939

/-- The surface area of a cone formed from a 270-degree sector of a circle with radius 20, divided by π, is 525. -/
theorem cone_surface_area (r : ℝ) (θ : ℝ) : 
  r = 20 → θ = 270 → (π * r^2 + π * r * (2 * π * r * θ / 360) / (2 * π)) / π = 525 := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l219_21939


namespace NUMINAMATH_CALUDE_tobias_shoe_purchase_l219_21987

/-- Tobias's shoe purchase problem -/
theorem tobias_shoe_purchase (shoe_cost : ℕ) (saving_months : ℕ) (monthly_allowance : ℕ)
  (lawn_charge : ℕ) (lawns_mowed : ℕ) (driveways_shoveled : ℕ) (change : ℕ)
  (h1 : shoe_cost = 95)
  (h2 : saving_months = 3)
  (h3 : monthly_allowance = 5)
  (h4 : lawn_charge = 15)
  (h5 : lawns_mowed = 4)
  (h6 : driveways_shoveled = 5)
  (h7 : change = 15) :
  ∃ (driveway_charge : ℕ),
    shoe_cost + change =
      saving_months * monthly_allowance +
      lawns_mowed * lawn_charge +
      driveways_shoveled * driveway_charge ∧
    driveway_charge = 7 :=
by sorry

end NUMINAMATH_CALUDE_tobias_shoe_purchase_l219_21987


namespace NUMINAMATH_CALUDE_brianna_reread_books_l219_21983

/-- The number of old books Brianna needs to reread in a year --/
def old_books_to_reread (books_per_month : ℕ) (months_in_year : ℕ) (gifted_books : ℕ) (bought_books : ℕ) (borrowed_books_difference : ℕ) : ℕ :=
  let total_books_needed := books_per_month * months_in_year
  let new_books := gifted_books + bought_books + (bought_books - borrowed_books_difference)
  total_books_needed - new_books

/-- Theorem stating the number of old books Brianna needs to reread --/
theorem brianna_reread_books : 
  old_books_to_reread 2 12 6 8 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_brianna_reread_books_l219_21983


namespace NUMINAMATH_CALUDE_bob_distance_from_start_l219_21921

-- Define the regular pentagon
def regularPentagon (sideLength : ℝ) : Set (ℝ × ℝ) :=
  sorry

-- Define Bob's position after walking a certain distance
def bobPosition (distance : ℝ) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem bob_distance_from_start :
  let pentagon := regularPentagon 3
  let finalPosition := bobPosition 7
  let distance := Real.sqrt ((finalPosition.1)^2 + (finalPosition.2)^2)
  distance = Real.sqrt 6.731 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_from_start_l219_21921


namespace NUMINAMATH_CALUDE_intersecting_lines_theorem_l219_21906

/-- Given two lines that intersect at (-7, 9), prove that the line passing through their coefficients as points has equation -7x + 9y = 1 -/
theorem intersecting_lines_theorem (A₁ B₁ A₂ B₂ : ℝ) : 
  (A₁ * (-7) + B₁ * 9 = 1) → 
  (A₂ * (-7) + B₂ * 9 = 1) → 
  ∃ (k : ℝ), k * (A₂ - A₁) = B₂ - B₁ ∧ 
             ∀ (x y : ℝ), y - B₁ = k * (x - A₁) → -7 * x + 9 * y = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_lines_theorem_l219_21906


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l219_21945

/-- An unfair 8-sided die with specific probability distribution -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  p_eight : ℝ
  /-- The probability of rolling any number from 1 to 7 -/
  p_others : ℝ
  /-- The die has 8 sides -/
  sides : Nat
  sides_eq : sides = 8
  /-- The probability of rolling an 8 is 3/8 -/
  p_eight_eq : p_eight = 3/8
  /-- The sum of all probabilities is 1 -/
  prob_sum : p_eight + 7 * p_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℝ :=
  d.p_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.p_eight * 8

/-- Theorem: The expected value of rolling this unfair 8-sided die is 5.5 -/
theorem unfair_die_expected_value (d : UnfairDie) : expected_value d = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l219_21945


namespace NUMINAMATH_CALUDE_monotonicity_condition_inequality_solution_correct_l219_21998

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (3*m - 1) * x + m - 2

-- Part 1: Monotonicity condition
theorem monotonicity_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (f m)) ↔ m ≥ -1/3 :=
sorry

-- Part 2: Inequality solution
def inequality_solution (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Ioi 2
  else if m > 0 then Set.Iio ((m-1)/m) ∪ Set.Ioi 2
  else if -1 < m ∧ m < 0 then Set.Ioo 2 ((m-1)/m)
  else if m = -1 then ∅
  else Set.Ioo ((m-1)/m) 2

theorem inequality_solution_correct (m : ℝ) (x : ℝ) :
  x ∈ inequality_solution m ↔ f m x + m > 0 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_inequality_solution_correct_l219_21998


namespace NUMINAMATH_CALUDE_triangle_sides_with_inscribed_rhombus_l219_21935

/-- A right triangle with a 60° angle and an inscribed rhombus -/
structure TriangleWithRhombus where
  /-- Side length of the inscribed rhombus -/
  rhombus_side : ℝ
  /-- The rhombus shares the 60° angle with the triangle -/
  shares_angle : Bool
  /-- All vertices of the rhombus lie on the sides of the triangle -/
  vertices_on_sides : Bool

/-- Theorem about the sides of the triangle given the inscribed rhombus -/
theorem triangle_sides_with_inscribed_rhombus 
  (t : TriangleWithRhombus) 
  (h1 : t.rhombus_side = 6) 
  (h2 : t.shares_angle) 
  (h3 : t.vertices_on_sides) : 
  ∃ (a b c : ℝ), a = 9 ∧ b = 9 * Real.sqrt 3 ∧ c = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_with_inscribed_rhombus_l219_21935


namespace NUMINAMATH_CALUDE_min_distance_to_line_l219_21962

/-- The minimum squared distance from the origin to the line 4x + 3y - 10 = 0 is 4 -/
theorem min_distance_to_line : 
  (∀ m n : ℝ, 4*m + 3*n - 10 = 0 → m^2 + n^2 ≥ 4) ∧ 
  (∃ m n : ℝ, 4*m + 3*n - 10 = 0 ∧ m^2 + n^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l219_21962


namespace NUMINAMATH_CALUDE_garden_yield_calculation_l219_21953

/-- Represents the dimensions of a garden section in steps -/
structure GardenSection where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield for an L-shaped garden -/
def expected_potato_yield (section1 : GardenSection) (section2 : GardenSection) 
    (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let area1 := (section1.length * section1.width * step_length ^ 2 : ℝ)
  let area2 := (section2.length * section2.width * step_length ^ 2 : ℝ)
  (area1 + area2) * yield_per_sqft

/-- Theorem stating the expected potato yield for the given garden -/
theorem garden_yield_calculation :
  let section1 : GardenSection := { length := 10, width := 25 }
  let section2 : GardenSection := { length := 10, width := 10 }
  let step_length : ℝ := 1.5
  let yield_per_sqft : ℝ := 0.75
  expected_potato_yield section1 section2 step_length yield_per_sqft = 590.625 := by
  sorry

end NUMINAMATH_CALUDE_garden_yield_calculation_l219_21953


namespace NUMINAMATH_CALUDE_max_apartments_in_complex_l219_21908

/-- Represents an apartment complex -/
structure ApartmentComplex where
  num_buildings : ℕ
  num_floors : ℕ
  apartments_per_floor : ℕ

/-- The maximum number of apartments in the complex -/
def max_apartments (complex : ApartmentComplex) : ℕ :=
  complex.num_buildings * complex.num_floors * complex.apartments_per_floor

/-- Theorem stating the maximum number of apartments in the given complex -/
theorem max_apartments_in_complex :
  ∃ (complex : ApartmentComplex),
    complex.num_buildings ≤ 22 ∧
    complex.num_buildings > 0 ∧
    complex.num_floors ≤ 6 ∧
    complex.apartments_per_floor = 5 ∧
    max_apartments complex = 660 := by
  sorry

end NUMINAMATH_CALUDE_max_apartments_in_complex_l219_21908


namespace NUMINAMATH_CALUDE_cubic_unique_solution_iff_l219_21950

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ := x^3 - a*x^2 - 3*a*x + a^2 - 1

/-- The property that the cubic equation has exactly one real solution -/
def has_unique_real_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a x = 0

theorem cubic_unique_solution_iff (a : ℝ) :
  has_unique_real_solution a ↔ a < -5/4 :=
sorry

end NUMINAMATH_CALUDE_cubic_unique_solution_iff_l219_21950


namespace NUMINAMATH_CALUDE_phone_number_probability_l219_21951

/-- The probability of randomly dialing the correct seven-digit number -/
theorem phone_number_probability :
  let first_three_options : ℕ := 2  -- 298 or 299
  let last_four_digits : ℕ := 4  -- 0, 2, 6, 7
  let total_combinations := first_three_options * (Nat.factorial last_four_digits)
  (1 : ℚ) / total_combinations = 1 / 48 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l219_21951


namespace NUMINAMATH_CALUDE_cookies_in_fridge_l219_21916

theorem cookies_in_fridge (total cookies_to_tim cookies_to_mike : ℕ) 
  (h1 : total = 512)
  (h2 : cookies_to_tim = 30)
  (h3 : cookies_to_mike = 45)
  (h4 : cookies_to_anna = 3 * cookies_to_tim) :
  total - (cookies_to_tim + cookies_to_mike + cookies_to_anna) = 347 :=
by sorry

end NUMINAMATH_CALUDE_cookies_in_fridge_l219_21916


namespace NUMINAMATH_CALUDE_divisibility_by_six_l219_21911

theorem divisibility_by_six (m n : ℤ) 
  (h1 : ∃ x y : ℤ, x^2 + m*x - n = 0 ∧ y^2 + m*y - n = 0)
  (h2 : ∃ x y : ℤ, x^2 - m*x + n = 0 ∧ y^2 - m*y + n = 0) : 
  6 ∣ n :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l219_21911


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l219_21963

/-- A quadratic function satisfying the given conditions -/
def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x > 2 * x + m) ↔ m < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l219_21963


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l219_21997

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define angle congruence
def are_congruent (α β : Angle) : Prop := α = β

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) :
  are_vertical_angles α β → are_congruent α β := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l219_21997


namespace NUMINAMATH_CALUDE_recipe_fraction_is_two_thirds_l219_21946

/-- Represents the amount of an ingredient required for a recipe --/
structure RecipeIngredient where
  amount : ℚ
  deriving Repr

/-- Represents the amount of an ingredient available --/
structure AvailableIngredient where
  amount : ℚ
  deriving Repr

/-- Calculates the fraction of the recipe that can be made for a single ingredient --/
def ingredientFraction (required : RecipeIngredient) (available : AvailableIngredient) : ℚ :=
  available.amount / required.amount

/-- Finds the maximum fraction of the recipe that can be made given all ingredients --/
def maxRecipeFraction (sugar : RecipeIngredient × AvailableIngredient) 
                      (milk : RecipeIngredient × AvailableIngredient)
                      (flour : RecipeIngredient × AvailableIngredient) : ℚ :=
  min (ingredientFraction sugar.1 sugar.2)
      (min (ingredientFraction milk.1 milk.2)
           (ingredientFraction flour.1 flour.2))

theorem recipe_fraction_is_two_thirds :
  let sugar_required := RecipeIngredient.mk (3/4)
  let sugar_available := AvailableIngredient.mk (2/4)
  let milk_required := RecipeIngredient.mk (2/3)
  let milk_available := AvailableIngredient.mk (1/2)
  let flour_required := RecipeIngredient.mk (3/8)
  let flour_available := AvailableIngredient.mk (1/4)
  maxRecipeFraction (sugar_required, sugar_available)
                    (milk_required, milk_available)
                    (flour_required, flour_available) = 2/3 := by
  sorry

#eval maxRecipeFraction (RecipeIngredient.mk (3/4), AvailableIngredient.mk (2/4))
                        (RecipeIngredient.mk (2/3), AvailableIngredient.mk (1/2))
                        (RecipeIngredient.mk (3/8), AvailableIngredient.mk (1/4))

end NUMINAMATH_CALUDE_recipe_fraction_is_two_thirds_l219_21946


namespace NUMINAMATH_CALUDE_solutions_to_quartic_equation_l219_21957

theorem solutions_to_quartic_equation : 
  {x : ℂ | x^4 - 81 = 0} = {3, -3, 3*I, -3*I} := by sorry

end NUMINAMATH_CALUDE_solutions_to_quartic_equation_l219_21957


namespace NUMINAMATH_CALUDE_womens_bathing_suits_l219_21914

theorem womens_bathing_suits (total : ℕ) (mens : ℕ) (womens : ℕ) : 
  total = 19766 → mens = 14797 → womens = total - mens → womens = 4969 := by
  sorry

end NUMINAMATH_CALUDE_womens_bathing_suits_l219_21914


namespace NUMINAMATH_CALUDE_find_m_value_l219_21910

/-- Given two functions f and g, prove that m = -7 when f(5) - g(5) = 55 -/
theorem find_m_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + 3 * x + 7
  let g : ℝ → ℝ := λ x => 2 * x^2 - m * x + 1
  (f 5 - g 5 = 55) → m = -7 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l219_21910


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_300_l219_21949

theorem rectangle_area_with_inscribed_circle : ℝ → ℝ → ℝ → Prop :=
  fun radius ratio area =>
    let width := 2 * radius
    let length := ratio * width
    area = length * width

theorem rectangle_area_is_300 :
  ∃ (radius ratio : ℝ),
    radius = 5 ∧
    ratio = 3 ∧
    rectangle_area_with_inscribed_circle radius ratio 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_300_l219_21949


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l219_21952

theorem smaller_circle_radius (R : ℝ) (r : ℝ) :
  R = 10 → -- Radius of the larger circle is 10 meters
  (4 * (2 * r) = 2 * R) → -- Four diameters of smaller circles span the diameter of the larger circle
  r = 2.5 := by sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l219_21952


namespace NUMINAMATH_CALUDE_sculpture_cost_theorem_l219_21902

/-- Calculates the total cost of John's custom sculpture --/
def calculate_sculpture_cost (base_price : ℝ) (standard_discount : ℝ) (marble_increase : ℝ) 
  (glass_increase : ℝ) (shipping_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_price := base_price * (1 - standard_discount)
  let marble_price := discounted_price * (1 + marble_increase)
  let glass_price := marble_price * (1 + glass_increase)
  let pre_tax_price := glass_price
  let tax := pre_tax_price * tax_rate
  pre_tax_price + tax + shipping_cost

/-- The total cost of John's sculpture is $1058.18 --/
theorem sculpture_cost_theorem : 
  calculate_sculpture_cost 450 0.15 0.70 0.35 75 0.12 = 1058.18 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_theorem_l219_21902


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l219_21982

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l219_21982


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l219_21959

/-- The number of stingrays in the aquarium -/
def num_stingrays : ℕ := 28

/-- The number of sharks in the aquarium -/
def num_sharks : ℕ := 2 * num_stingrays

/-- The total number of fish (sharks and stingrays) in the aquarium -/
def total_fish : ℕ := num_sharks + num_stingrays

/-- Theorem stating that the total number of fish in the aquarium is 84 -/
theorem aquarium_fish_count : total_fish = 84 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l219_21959


namespace NUMINAMATH_CALUDE_max_peak_consumption_l219_21907

/-- Proves that the maximum average monthly electricity consumption during peak hours is 118 kw•h -/
theorem max_peak_consumption (original_price peak_price off_peak_price total_consumption : ℝ)
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_consumption →
    (original_price - peak_price) * x + (original_price - off_peak_price) * (total_consumption - x) ≥ 
    total_consumption * original_price * 0.1) :
  ∃ max_peak : ℝ, max_peak = 118 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_consumption →
      (original_price - peak_price) * x + (original_price - off_peak_price) * (total_consumption - x) ≥ 
      total_consumption * original_price * 0.1 → 
      x ≤ max_peak :=
sorry

end NUMINAMATH_CALUDE_max_peak_consumption_l219_21907


namespace NUMINAMATH_CALUDE_probability_A_and_B_not_together_l219_21925

def total_students : ℕ := 6
def selected_students : ℕ := 4

theorem probability_A_and_B_not_together :
  let total_combinations := Nat.choose total_students selected_students
  let combinations_with_A_and_B := Nat.choose (total_students - 2) (selected_students - 2)
  (1 : ℚ) - (combinations_with_A_and_B : ℚ) / (total_combinations : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_and_B_not_together_l219_21925


namespace NUMINAMATH_CALUDE_angle_value_l219_21917

theorem angle_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan β = 1/2) (h4 : Real.tan (α - β) = 1/3) : α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l219_21917


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l219_21990

/-- A parabola with x-axis as its axis of symmetry, vertex at the origin, and latus rectum length of 8 -/
structure Parabola where
  symmetry_axis : ℝ → ℝ
  vertex : ℝ × ℝ
  latus_rectum : ℝ
  h_symmetry : symmetry_axis = λ y => 0
  h_vertex : vertex = (0, 0)
  h_latus_rectum : latus_rectum = 8

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 8*x ∨ y^2 = -8*x}

theorem parabola_equation_correct (p : Parabola) :
  ∀ (x y : ℝ), (x, y) ∈ parabola_equation p ↔
    (∃ t : ℝ, x = t^2 / 2 ∧ y = t) ∨ (∃ t : ℝ, x = -t^2 / 2 ∧ y = t) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l219_21990


namespace NUMINAMATH_CALUDE_inequality_proof_l219_21947

theorem inequality_proof (a b x y : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l219_21947


namespace NUMINAMATH_CALUDE_probability_divisible_by_five_l219_21922

/-- A three-digit positive integer -/
def ThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- An integer ending in 5 -/
def EndsInFive (n : ℕ) : Prop :=
  n % 10 = 5

/-- The probability that a three-digit positive integer ending in 5 is divisible by 5 is 1 -/
theorem probability_divisible_by_five :
  ∀ n : ℕ, ThreeDigitInteger n → EndsInFive n → n % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_five_l219_21922


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l219_21934

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  is_geometric : (a 3)^2 = a 1 * a 7

/-- The main theorem -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 3) / (seq.a 2 + seq.a 4) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l219_21934


namespace NUMINAMATH_CALUDE_ladder_slide_l219_21918

theorem ladder_slide (L d s : ℝ) (h1 : L = 20) (h2 : d = 4) (h3 : s = 3) :
  ∃ y : ℝ, y = Real.sqrt (400 - (2 * Real.sqrt 96 - 3)^2) - 4 :=
sorry

end NUMINAMATH_CALUDE_ladder_slide_l219_21918


namespace NUMINAMATH_CALUDE_ratio_problem_l219_21964

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 3 / 4) :
  x / y = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l219_21964


namespace NUMINAMATH_CALUDE_no_20_digit_square_starting_with_11_ones_l219_21967

theorem no_20_digit_square_starting_with_11_ones :
  ¬∃ (n : ℕ), 
    (10^19 ≤ n) ∧ 
    (n < 10^20) ∧ 
    (11111111111 * 10^9 ≤ n) ∧ 
    (n < 11111111112 * 10^9) ∧ 
    (∃ (k : ℕ), n = k^2) :=
by sorry

end NUMINAMATH_CALUDE_no_20_digit_square_starting_with_11_ones_l219_21967


namespace NUMINAMATH_CALUDE_no_solution_implies_positive_b_l219_21981

theorem no_solution_implies_positive_b (a b : ℝ) :
  (∀ x y : ℝ, y ≠ x^2 + a*x + b ∨ x ≠ y^2 + a*y + b) →
  b > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_positive_b_l219_21981


namespace NUMINAMATH_CALUDE_coin_array_digit_sum_l219_21973

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

/-- Theorem: For a triangular array of 3003 coins, where the n-th row has n coins,
    the sum of the digits of the total number of rows is 14 -/
theorem coin_array_digit_sum :
  ∃ (n : ℕ), triangular_sum n = 3003 ∧ digit_sum n = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_digit_sum_l219_21973


namespace NUMINAMATH_CALUDE_proportion_proof_1_proportion_proof_2_l219_21972

theorem proportion_proof_1 : 
  let x : ℚ := 1/12
  (x : ℚ) / (5/9 : ℚ) = (1/20 : ℚ) / (1/3 : ℚ) := by sorry

theorem proportion_proof_2 : 
  let x : ℚ := 5/4
  (x : ℚ) / (1/4 : ℚ) = (1/2 : ℚ) / (1/10 : ℚ) := by sorry

end NUMINAMATH_CALUDE_proportion_proof_1_proportion_proof_2_l219_21972


namespace NUMINAMATH_CALUDE_complex_number_location_l219_21961

theorem complex_number_location (z : ℂ) :
  (z * (1 + Complex.I) = 3 - Complex.I) →
  (0 < z.re ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l219_21961


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l219_21913

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l219_21913


namespace NUMINAMATH_CALUDE_firewood_collection_l219_21988

/-- Firewood collection problem -/
theorem firewood_collection (K H : ℝ) (x E : ℝ) 
  (hK : K = 0.8)
  (hH : H = 1.5)
  (eq1 : 10 * K + x * E + 12 * H = 44)
  (eq2 : 10 + x + 12 = 35) :
  x = 13 ∧ E = 18 / 13 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_l219_21988


namespace NUMINAMATH_CALUDE_oblong_perimeter_l219_21943

theorem oblong_perimeter :
  ∀ l w : ℕ,
    l > w →
    l * 3 = w * 4 →
    l * w = 4624 →
    2 * l + 2 * w = 182 := by
  sorry

end NUMINAMATH_CALUDE_oblong_perimeter_l219_21943


namespace NUMINAMATH_CALUDE_florist_roses_l219_21984

/-- A problem about a florist's roses -/
theorem florist_roses (initial : ℕ) (sold : ℕ) (final : ℕ) (picked : ℕ) : 
  initial = 11 → sold = 2 → final = 41 → picked = final - (initial - sold) → picked = 32 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l219_21984


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l219_21996

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : subset m α)
  (h4 : perp m β) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l219_21996


namespace NUMINAMATH_CALUDE_complement_of_A_l219_21956

-- Define the set A
def A : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_l219_21956


namespace NUMINAMATH_CALUDE_power_of_a_l219_21932

theorem power_of_a (a b : ℝ) : b = Real.sqrt (3 - a) + Real.sqrt (a - 3) + 2 → a^b = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_a_l219_21932


namespace NUMINAMATH_CALUDE_mixed_fraction_division_l219_21903

theorem mixed_fraction_division :
  (7 + 1/3) / (2 + 1/2) = 44/15 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_division_l219_21903


namespace NUMINAMATH_CALUDE_no_valid_tournament_l219_21927

/-- Represents a round-robin chess tournament -/
structure ChessTournament where
  num_players : ℕ
  wins : Fin num_players → ℕ
  draws : Fin num_players → ℕ
  losses : Fin num_players → ℕ

/-- Definition of a valid round-robin tournament -/
def is_valid_tournament (t : ChessTournament) : Prop :=
  t.num_players = 20 ∧
  ∀ i : Fin t.num_players, 
    t.wins i + t.draws i + t.losses i = t.num_players - 1 ∧
    t.wins i = t.draws i

/-- Theorem stating that a valid tournament as described is impossible -/
theorem no_valid_tournament : ¬∃ t : ChessTournament, is_valid_tournament t := by
  sorry


end NUMINAMATH_CALUDE_no_valid_tournament_l219_21927


namespace NUMINAMATH_CALUDE_equal_students_after_transfer_total_students_after_transfer_l219_21930

/-- Represents a section in Grade 4 -/
inductive Section
| Diligence
| Industry

/-- The number of students in a section before the transfer -/
def students_before (s : Section) : ℕ :=
  match s with
  | Section.Diligence => 23
  | Section.Industry => sorry  -- We don't know this value

/-- The number of students transferred from Industry to Diligence -/
def transferred_students : ℕ := 2

/-- The number of students in a section after the transfer -/
def students_after (s : Section) : ℕ :=
  match s with
  | Section.Diligence => students_before Section.Diligence + transferred_students
  | Section.Industry => students_before Section.Industry - transferred_students

/-- Theorem stating that the sections have equal students after transfer -/
theorem equal_students_after_transfer :
  students_after Section.Diligence = students_after Section.Industry := by sorry

/-- The main theorem to prove -/
theorem total_students_after_transfer :
  students_after Section.Diligence + students_after Section.Industry = 50 := by sorry

end NUMINAMATH_CALUDE_equal_students_after_transfer_total_students_after_transfer_l219_21930


namespace NUMINAMATH_CALUDE_complex_moduli_equality_l219_21993

theorem complex_moduli_equality (a : ℝ) : 
  let z₁ : ℂ := a + 2 * Complex.I
  let z₂ : ℂ := 2 - Complex.I
  Complex.abs z₁ = Complex.abs z₂ → a^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_moduli_equality_l219_21993


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l219_21954

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} =
  {(0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l219_21954


namespace NUMINAMATH_CALUDE_tangent_circles_radii_l219_21928

/-- Given a sequence of six circles tangent to each other and two parallel lines,
    where the radii form a geometric sequence, prove that if the smallest radius
    is 5 and the largest is 20, then the radius of the third circle is 5 * 2^(2/5). -/
theorem tangent_circles_radii (r : Fin 6 → ℝ) : 
  (∀ i : Fin 5, r i > 0) →  -- All radii are positive
  (∀ i : Fin 5, r i < r i.succ) →  -- Radii are in increasing order
  (∀ i j : Fin 5, i < j → r j / r i = r (j+1) / r (j : Fin 6)) →  -- Geometric sequence
  r 0 = 5 →  -- Smallest radius
  r 5 = 20 →  -- Largest radius
  r 2 = 5 * 2^(2/5) := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_l219_21928


namespace NUMINAMATH_CALUDE_triangle_radii_relation_l219_21926

/-- Given a triangle with sides a, b, c, semi-perimeter p, inradius r, and excircle radii r_a, r_b, r_c,
    prove that 1/((p-a)(p-b)) + 1/((p-b)(p-c)) + 1/((p-c)(p-a)) = 1/r^2 -/
theorem triangle_radii_relation (a b c p r r_a r_b r_c : ℝ) 
  (h_p : p = (a + b + c) / 2)
  (h_r : r > 0)
  (h_ra : r_a > 0)
  (h_rb : r_b > 0)
  (h_rc : r_c > 0)
  (h_pbc : 1 / ((p - b) * (p - c)) = 1 / (r * r_a))
  (h_pca : 1 / ((p - c) * (p - a)) = 1 / (r * r_b))
  (h_pab : 1 / ((p - a) * (p - b)) = 1 / (r * r_c))
  (h_sum : 1 / r_a + 1 / r_b + 1 / r_c = 1 / r) :
  1 / ((p - a) * (p - b)) + 1 / ((p - b) * (p - c)) + 1 / ((p - c) * (p - a)) = 1 / r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_radii_relation_l219_21926


namespace NUMINAMATH_CALUDE_min_value_expression_l219_21999

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_prod : x * y * z = 3 / 4) (h_sum : x + y + z = 4) :
  x^3 + x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 + z^3 ≥ 21/2 ∧
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 3 / 4 ∧ x + y + z = 4 ∧
    x^3 + x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 + z^3 = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l219_21999


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l219_21940

/-- Given two points on the inverse proportion function y = -4/x, 
    if the x-coordinate of the first point is negative and 
    the x-coordinate of the second point is positive, 
    then the y-coordinate of the first point is greater than 
    the y-coordinate of the second point. -/
theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ < 0 → 0 < x₂ → y₁ = -4 / x₁ → y₂ = -4 / x₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l219_21940


namespace NUMINAMATH_CALUDE_two_digit_sum_ten_l219_21941

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The digit sum of a natural number is the sum of its digits. -/
def DigitSum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- There are exactly 9 two-digit numbers whose digits sum to 10. -/
theorem two_digit_sum_ten :
  ∃! (s : Finset ℕ), (∀ n ∈ s, TwoDigitNumber n ∧ DigitSum n = 10) ∧ s.card = 9 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_ten_l219_21941


namespace NUMINAMATH_CALUDE_sequence_value_at_50_l219_21969

def f (n : ℕ) : ℕ := 2 * n^3 + 3 * n^2 + n + 1

theorem sequence_value_at_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 25 ∧ f 3 = 65 → f 50 = 257551 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_value_at_50_l219_21969


namespace NUMINAMATH_CALUDE_value_of_a_l219_21965

theorem value_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {1, 3, a} → 
  B = {1, a^2} → 
  A ∩ B = {1, a} → 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l219_21965


namespace NUMINAMATH_CALUDE_rhino_weight_theorem_l219_21986

/-- The weight of a full-grown white rhino in pounds -/
def full_grown_white_rhino_weight : ℝ := 5100

/-- The weight of a newborn white rhino in pounds -/
def newborn_white_rhino_weight : ℝ := 150

/-- The weight of a full-grown black rhino in pounds -/
def full_grown_black_rhino_weight : ℝ := 2000

/-- The weight of a newborn black rhino in pounds -/
def newborn_black_rhino_weight : ℝ := 100

/-- The conversion factor from pounds to kilograms -/
def pounds_to_kg : ℝ := 0.453592

/-- The number of full-grown white rhinos -/
def num_full_grown_white : ℕ := 6

/-- The number of newborn white rhinos -/
def num_newborn_white : ℕ := 3

/-- The number of full-grown black rhinos -/
def num_full_grown_black : ℕ := 7

/-- The number of newborn black rhinos -/
def num_newborn_black : ℕ := 4

/-- The total weight of all rhinos in kilograms -/
def total_weight_kg : ℝ :=
  ((num_full_grown_white : ℝ) * full_grown_white_rhino_weight +
   (num_newborn_white : ℝ) * newborn_white_rhino_weight +
   (num_full_grown_black : ℝ) * full_grown_black_rhino_weight +
   (num_newborn_black : ℝ) * newborn_black_rhino_weight) * pounds_to_kg

theorem rhino_weight_theorem :
  total_weight_kg = 20616.436 := by
  sorry

end NUMINAMATH_CALUDE_rhino_weight_theorem_l219_21986


namespace NUMINAMATH_CALUDE_relative_minimum_condition_l219_21992

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - x^2 + a*x + 1

/-- The first derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 - 3*x^2 - 2*x + a

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 12*x^2 - 6*x - 2

/-- Theorem stating that f(a) = a is a relative minimum iff a = 1 -/
theorem relative_minimum_condition (a : ℝ) :
  (f a a = a ∧ ∀ x, x ≠ a → f a x ≥ f a a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_relative_minimum_condition_l219_21992


namespace NUMINAMATH_CALUDE_class_vision_most_suitable_l219_21901

/-- Represents a survey option -/
inductive SurveyOption
  | SleepTimeNationwide
  | RiverWaterQuality
  | PocketMoneyCity
  | ClassVision

/-- Checks if a survey option is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.ClassVision => true
  | _ => false

/-- Theorem stating that investigating the vision of all classmates in a class
    is the most suitable for a comprehensive survey -/
theorem class_vision_most_suitable :
  isSuitableForComprehensiveSurvey SurveyOption.ClassVision ∧
  (∀ (option : SurveyOption),
    isSuitableForComprehensiveSurvey option →
    option = SurveyOption.ClassVision) :=
by
  sorry

#check class_vision_most_suitable

end NUMINAMATH_CALUDE_class_vision_most_suitable_l219_21901


namespace NUMINAMATH_CALUDE_cos_two_alpha_plus_pi_third_l219_21975

theorem cos_two_alpha_plus_pi_third (α : ℝ) 
  (h : Real.sin (π / 6 - α) - Real.cos α = 1 / 3) : 
  Real.cos (2 * α + π / 3) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_plus_pi_third_l219_21975


namespace NUMINAMATH_CALUDE_michael_fish_count_l219_21979

def total_pets : ℕ := 160
def dog_percentage : ℚ := 225 / 1000
def cat_percentage : ℚ := 375 / 1000
def bunny_percentage : ℚ := 15 / 100
def bird_percentage : ℚ := 1 / 10

theorem michael_fish_count :
  let dogs := (dog_percentage * total_pets).floor
  let cats := (cat_percentage * total_pets).floor
  let bunnies := (bunny_percentage * total_pets).floor
  let birds := (bird_percentage * total_pets).floor
  let fish := total_pets - (dogs + cats + bunnies + birds)
  fish = 24 := by
sorry

end NUMINAMATH_CALUDE_michael_fish_count_l219_21979


namespace NUMINAMATH_CALUDE_waiter_customers_l219_21938

/-- The total number of customers a waiter has after new arrivals -/
def total_customers (initial : ℕ) (new_arrivals : ℕ) : ℕ :=
  initial + new_arrivals

/-- Theorem stating that with 3 initial customers and 5 new arrivals, the total is 8 -/
theorem waiter_customers : total_customers 3 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l219_21938


namespace NUMINAMATH_CALUDE_range_of_a_l219_21920

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a^2 + a)*x + a^3 > 0 ↔ x < a^2 ∨ x > a) →
  0 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l219_21920


namespace NUMINAMATH_CALUDE_businessmen_drink_neither_l219_21904

theorem businessmen_drink_neither (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ)
  (h_total : total = 30)
  (h_coffee : coffee = 15)
  (h_tea : tea = 13)
  (h_both : both = 7) :
  total - ((coffee + tea) - both) = 9 :=
by sorry

end NUMINAMATH_CALUDE_businessmen_drink_neither_l219_21904


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l219_21991

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h_pos : x > 0) (h_eq : x + 1/x = 100) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 102 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l219_21991


namespace NUMINAMATH_CALUDE_only_cooking_count_l219_21931

/-- Given information about curriculum participation --/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cooking_and_yoga : ℕ
  all_curriculums : ℕ
  cooking_and_weaving : ℕ

/-- Theorem stating the number of people who study only cooking --/
theorem only_cooking_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 25)
  (h2 : cp.cooking = 15)
  (h3 : cp.weaving = 8)
  (h4 : cp.cooking_and_yoga = 7)
  (h5 : cp.all_curriculums = 3)
  (h6 : cp.cooking_and_weaving = 3) :
  cp.cooking - (cp.cooking_and_yoga - cp.all_curriculums) - (cp.cooking_and_weaving - cp.all_curriculums) - cp.all_curriculums = 8 :=
by sorry

end NUMINAMATH_CALUDE_only_cooking_count_l219_21931


namespace NUMINAMATH_CALUDE_storage_blocks_count_l219_21955

/-- Calculates the number of blocks needed for a rectangular storage --/
def blocksNeeded (length width height thickness : ℕ) : ℕ :=
  let totalVolume := length * width * height
  let interiorLength := length - 2 * thickness
  let interiorWidth := width - 2 * thickness
  let interiorHeight := height - thickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating the number of blocks needed for the specific storage --/
theorem storage_blocks_count :
  blocksNeeded 20 15 10 2 = 1592 := by
  sorry

end NUMINAMATH_CALUDE_storage_blocks_count_l219_21955


namespace NUMINAMATH_CALUDE_intersection_M_N_l219_21989

open Set

-- Define set M
def M : Set ℝ := {x : ℝ | (x + 3) * (x - 2) < 0}

-- Define set N
def N : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_M_N :
  M ∩ N = Icc 1 2 ∩ Iio 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_M_N_l219_21989


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_l219_21900

theorem range_of_x_plus_y (x y : Real) 
  (h1 : 0 ≤ y) (h2 : y ≤ x) (h3 : x ≤ π/2)
  (h4 : 4 * (Real.cos y)^2 + 4 * Real.cos x * Real.sin y - 4 * (Real.cos x)^2 ≤ 1) :
  (x + y ∈ Set.Icc 0 (π/6)) ∨ (x + y ∈ Set.Icc (5*π/6) π) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_l219_21900


namespace NUMINAMATH_CALUDE_trigonometric_sum_l219_21905

theorem trigonometric_sum (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) :
  (Real.sin θ ^ 6 / a + Real.cos θ ^ 6 / b = 1 / (a + b)) →
  (Real.sin θ ^ 12 / a ^ 5 + Real.cos θ ^ 12 / b ^ 5 = 1 / (a + b) ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_l219_21905


namespace NUMINAMATH_CALUDE_cards_given_away_ben_cards_given_away_l219_21912

theorem cards_given_away (basketball_boxes : Nat) (basketball_cards_per_box : Nat)
                         (baseball_boxes : Nat) (baseball_cards_per_box : Nat)
                         (cards_left : Nat) : Nat :=
  let total_cards := basketball_boxes * basketball_cards_per_box + 
                     baseball_boxes * baseball_cards_per_box
  total_cards - cards_left

theorem ben_cards_given_away : 
  cards_given_away 4 10 5 8 22 = 58 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_away_ben_cards_given_away_l219_21912


namespace NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l219_21974

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of the leg divided by the point of tangency -/
  a : ℝ
  /-- Length of the other leg -/
  b : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The leg a is divided into segments of 6 and 10 by the point of tangency -/
  h_a : a = 16
  /-- The radius of the inscribed circle is 6 -/
  h_r : r = 6
  /-- The semi-perimeter of the triangle -/
  p : ℝ
  /-- Relation between semi-perimeter and leg b -/
  h_p : p = b + 10

/-- The area of the right triangle with an inscribed circle is 240 -/
theorem area_of_right_triangle_with_inscribed_circle 
  (t : RightTriangleWithInscribedCircle) : t.a * t.b / 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l219_21974


namespace NUMINAMATH_CALUDE_fibonacci_determinant_l219_21919

/-- An arbitrary Fibonacci sequence -/
def FibonacciSequence (u : ℕ → ℤ) : Prop :=
  ∀ n, u (n + 2) = u n + u (n + 1)

/-- The main theorem about the determinant of consecutive Fibonacci terms -/
theorem fibonacci_determinant (u : ℕ → ℤ) (h : FibonacciSequence u) :
  ∀ n : ℕ, u (n - 1) * u (n + 1) - u n ^ 2 = (-1) ^ n :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_determinant_l219_21919


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_l219_21944

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The smallest positive integer k(m) satisfying the Fibonacci periodicity modulo m -/
def k_m (m : ℕ) : ℕ := sorry

theorem fibonacci_periodicity (m : ℕ) (h : m > 0) :
  (∃ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ m^2 ∧ fib i % m = fib j % m ∧ fib (i + 1) % m = fib (j + 1) % m) ∧
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, fib (n + k) % m = fib n % m) ∧
  (∀ n : ℕ, fib (n + k_m m) % m = fib n % m) ∧
  (fib (k_m m) % m = 0 ∧ fib (k_m m + 1) % m = 1) ∧
  (∀ k : ℕ, k > 0 → (∀ n : ℕ, fib (n + k) % m = fib n % m) ↔ k_m m ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_l219_21944


namespace NUMINAMATH_CALUDE_area_code_combinations_l219_21976

/-- The number of digits in the area code -/
def n : ℕ := 4

/-- The set of digits used in the area code -/
def digits : Finset ℕ := {9, 8, 7, 6}

/-- The number of possible combinations for the area code -/
def num_combinations : ℕ := n.factorial

theorem area_code_combinations :
  Finset.card (Finset.powerset digits) = n ∧ num_combinations = 24 := by sorry

end NUMINAMATH_CALUDE_area_code_combinations_l219_21976


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l219_21948

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 19881 → n + (n + 1) = 283 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l219_21948


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l219_21933

theorem sqrt_product_equality : Real.sqrt 50 * Real.sqrt 18 * Real.sqrt 8 = 60 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l219_21933


namespace NUMINAMATH_CALUDE_bounded_harmonic_constant_l219_21968

/-- A function f: ℤ² → ℝ is harmonic if it satisfies the discrete Laplace equation -/
def Harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1) = 4 * f (x, y)

/-- A function f: ℤ² → ℝ is bounded if there exists a positive constant M such that |f(x, y)| ≤ M for all (x, y) in ℤ² -/
def Bounded (f : ℤ × ℤ → ℝ) : Prop :=
  ∃ M > 0, ∀ x y, |f (x, y)| ≤ M

/-- If a function f: ℤ² → ℝ is both harmonic and bounded, then it is constant -/
theorem bounded_harmonic_constant (f : ℤ × ℤ → ℝ) (hf_harmonic : Harmonic f) (hf_bounded : Bounded f) :
  ∃ c : ℝ, ∀ x y, f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_bounded_harmonic_constant_l219_21968


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l219_21942

theorem smallest_next_divisor_after_437 (m : ℕ) (h1 : 10000 ≤ m ∧ m ≤ 99999) 
  (h2 : Odd m) (h3 : 437 ∣ m) :
  ∃ (d : ℕ), d ∣ m ∧ 437 < d ∧ d ≤ 874 ∧ ∀ (x : ℕ), x ∣ m → 437 < x → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l219_21942


namespace NUMINAMATH_CALUDE_train_length_l219_21924

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 108 → time = 7 → speed * time * (1000 / 3600) = 210 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l219_21924


namespace NUMINAMATH_CALUDE_f_difference_l219_21971

def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + 4*x

theorem f_difference : f 3 - f (-3) = 672 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l219_21971


namespace NUMINAMATH_CALUDE_circle_area_difference_l219_21970

theorem circle_area_difference (r₁ r₂ : ℝ) (h₁ : r₁ = 14) (h₂ : r₂ = 10) :
  π * r₁^2 - π * r₂^2 = 96 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l219_21970


namespace NUMINAMATH_CALUDE_jill_first_bus_wait_time_l219_21937

/-- Represents Jill's bus journey times -/
structure BusJourney where
  first_bus_wait : ℕ
  first_bus_ride : ℕ
  second_bus_ride : ℕ

/-- The conditions of Jill's bus journey -/
def journey_conditions (j : BusJourney) : Prop :=
  j.first_bus_ride = 30 ∧
  j.second_bus_ride = 21 ∧
  j.second_bus_ride * 2 = j.first_bus_wait + j.first_bus_ride

theorem jill_first_bus_wait_time (j : BusJourney) 
  (h : journey_conditions j) : j.first_bus_wait = 12 := by
  sorry

end NUMINAMATH_CALUDE_jill_first_bus_wait_time_l219_21937


namespace NUMINAMATH_CALUDE_c_nonzero_necessary_not_sufficient_l219_21980

/-- Represents a conic section defined by the equation ax^2 + y^2 = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Determines if a conic section is an ellipse or hyperbola -/
def isEllipseOrHyperbola (conic : ConicSection) : Prop :=
  (conic.a > 0 ∧ conic.c > 0) ∨ (conic.a < 0 ∧ conic.c ≠ 0)

/-- The main theorem stating that c ≠ 0 is necessary but not sufficient -/
theorem c_nonzero_necessary_not_sufficient :
  (∀ conic : ConicSection, isEllipseOrHyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬isEllipseOrHyperbola conic) :=
sorry

end NUMINAMATH_CALUDE_c_nonzero_necessary_not_sufficient_l219_21980


namespace NUMINAMATH_CALUDE_johns_total_cost_l219_21977

/-- Calculates the total cost of a cell phone plan --/
def calculate_total_cost (base_cost : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
  (texts_sent : ℕ) (hours_talked : ℕ) : ℝ :=
  let text_charge := text_cost * texts_sent
  let extra_hours := max (hours_talked - 50) 0
  let extra_minutes := extra_hours * 60
  let extra_minute_charge := extra_minute_cost * extra_minutes
  base_cost + text_charge + extra_minute_charge

/-- Theorem stating that John's total cost is $69.00 --/
theorem johns_total_cost : 
  calculate_total_cost 30 0.10 0.20 150 52 = 69 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_cost_l219_21977


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l219_21909

def A : Set ℝ := {x | |x| * (x^2 - 4*x + 3) < 0}

def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l219_21909


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l219_21995

/-- Given two lines l₁: ax + y + 1 = 0 and l₂: x - 2y + 1 = 0,
    if they are perpendicular, then a = 2 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ x y, ax + y + 1 = 0 ∧ x - 2*y + 1 = 0) →
  (∀ x₁ y₁ x₂ y₂, ax₁ + y₁ + 1 = 0 ∧ x₁ - 2*y₁ + 1 = 0 ∧
                   ax₂ + y₂ + 1 = 0 ∧ x₂ - 2*y₂ + 1 = 0 →
                   (x₂ - x₁) * ((y₂ - y₁) / (x₂ - x₁)) = -1) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l219_21995


namespace NUMINAMATH_CALUDE_cello_viola_pairs_count_l219_21994

/-- The number of cello-viola pairs in a music store, where each pair consists of
    a cello and a viola made from the same tree. -/
def cello_viola_pairs : ℕ := 70

theorem cello_viola_pairs_count (total_cellos : ℕ) (total_violas : ℕ) 
  (prob_same_tree : ℚ) (h1 : total_cellos = 800) (h2 : total_violas = 600) 
  (h3 : prob_same_tree = 14583333333333335 / 100000000000000000) : 
  cello_viola_pairs = (prob_same_tree * total_cellos * total_violas : ℚ).num := by
  sorry

end NUMINAMATH_CALUDE_cello_viola_pairs_count_l219_21994


namespace NUMINAMATH_CALUDE_equation_solution_l219_21966

theorem equation_solution : 
  ∃ x : ℚ, (8 * x^2 + 80 * x + 4) / (4 * x + 45) = 2 * x + 3 ∧ x = -131/22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l219_21966


namespace NUMINAMATH_CALUDE_exists_100_same_polygons_l219_21929

/-- Represents a convex polygon --/
structure ConvexPolygon where
  vertices : ℕ

/-- Represents the state of the paper after some cuts --/
structure PaperState where
  polygons : List ConvexPolygon

/-- A function that performs a single cut --/
def cut (state : PaperState) : PaperState :=
  sorry

/-- A function that checks if there are 100 polygons with the same number of vertices --/
def has_100_same_polygons (state : PaperState) : Bool :=
  sorry

/-- The main theorem --/
theorem exists_100_same_polygons :
  ∃ (n : ℕ), ∀ (initial : PaperState),
    has_100_same_polygons (n.iterate cut initial) = true :=
  sorry

end NUMINAMATH_CALUDE_exists_100_same_polygons_l219_21929


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l219_21915

/-- Given a positive constant s, prove that an isosceles triangle with base 2s
    and area equal to a rectangle with dimensions 2s and s has height 2s. -/
theorem isosceles_triangle_height (s : ℝ) (hs : s > 0) : 
  let rectangle_area := 2 * s * s
  let triangle_base := 2 * s
  let triangle_height := 2 * s
  rectangle_area = 1/2 * triangle_base * triangle_height := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l219_21915


namespace NUMINAMATH_CALUDE_combine_like_terms_l219_21978

theorem combine_like_terms (a : ℝ) : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l219_21978


namespace NUMINAMATH_CALUDE_line1_passes_through_points_line2_satisfies_conditions_l219_21985

-- Define the points
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (0, -3)
def point3 : ℝ × ℝ := (0, 5)

-- Define the sum of intercepts
def sum_of_intercepts : ℝ := 2

-- Define the equations of the lines
def line1_equation (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2_equation (x y : ℝ) : Prop := 5 * x - 3 * y + 15 = 0

-- Theorem for the first line
theorem line1_passes_through_points :
  line1_equation point1.1 point1.2 ∧ line1_equation point2.1 point2.2 :=
sorry

-- Theorem for the second line
theorem line2_satisfies_conditions :
  line2_equation point3.1 point3.2 ∧
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a + b = sum_of_intercepts ∧
  ∀ x y : ℝ, line2_equation x y ↔ x / a + y / b = 1) :=
sorry

end NUMINAMATH_CALUDE_line1_passes_through_points_line2_satisfies_conditions_l219_21985


namespace NUMINAMATH_CALUDE_derivative_of_f_l219_21936

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt 2) * Real.log (Real.sqrt 2 * Real.tan x + Real.sqrt (1 + 2 * Real.tan x ^ 2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 1 / (Real.cos x ^ 2 * Real.sqrt (1 + 2 * Real.tan x ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l219_21936
