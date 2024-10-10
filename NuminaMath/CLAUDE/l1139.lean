import Mathlib

namespace smallest_x_value_l1139_113965

theorem smallest_x_value (x y : ℝ) 
  (hx : 4 < x ∧ x < 8) 
  (hy : 8 < y ∧ y < 12) 
  (h_diff : ∃ (n : ℕ), n = 7 ∧ n = ⌊y - x⌋) : 
  4 < x :=
sorry

end smallest_x_value_l1139_113965


namespace inequality_always_true_l1139_113949

theorem inequality_always_true (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
sorry

end inequality_always_true_l1139_113949


namespace neighbor_took_twelve_eggs_l1139_113913

/-- Represents the egg-laying scenario with Myrtle's hens -/
structure EggScenario where
  hens : ℕ
  eggs_per_hen_per_day : ℕ
  days_gone : ℕ
  dropped_eggs : ℕ
  remaining_eggs : ℕ

/-- Calculates the number of eggs taken by the neighbor -/
def neighbor_eggs (scenario : EggScenario) : ℕ :=
  scenario.hens * scenario.eggs_per_hen_per_day * scenario.days_gone -
  (scenario.remaining_eggs + scenario.dropped_eggs)

/-- Theorem stating that the neighbor took 12 eggs -/
theorem neighbor_took_twelve_eggs :
  let scenario : EggScenario := {
    hens := 3,
    eggs_per_hen_per_day := 3,
    days_gone := 7,
    dropped_eggs := 5,
    remaining_eggs := 46
  }
  neighbor_eggs scenario = 12 := by sorry

end neighbor_took_twelve_eggs_l1139_113913


namespace not_all_positive_k_real_roots_not_all_negative_k_nonzero_im_not_all_real_k_not_pure_imaginary_l1139_113999

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z k : ℂ) : Prop := 10 * z^2 - 7 * i * z - k = 0

-- Statement A is false
theorem not_all_positive_k_real_roots :
  ¬ ∀ (k : ℝ), k > 0 → ∀ (z : ℂ), equation z k → z.im = 0 :=
sorry

-- Statement B is false
theorem not_all_negative_k_nonzero_im :
  ¬ ∀ (k : ℝ), k < 0 → ∀ (z : ℂ), equation z k → z.im ≠ 0 :=
sorry

-- Statement C is false
theorem not_all_real_k_not_pure_imaginary :
  ¬ ∀ (k : ℝ), ∀ (z : ℂ), equation z k → z.re ≠ 0 :=
sorry

end not_all_positive_k_real_roots_not_all_negative_k_nonzero_im_not_all_real_k_not_pure_imaginary_l1139_113999


namespace function_inequality_l1139_113964

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x > 1 → y > x → f x < f y)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- State the theorem
theorem function_inequality : f (-1) < f 0 ∧ f 0 < f 4 := by
  sorry

end function_inequality_l1139_113964


namespace simplify_expression_solve_inequality_system_l1139_113943

-- Problem 1
theorem simplify_expression (x : ℝ) : (2*x + 1)^2 + x*(x - 4) = 5*x^2 + 1 := by
  sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) : 
  (3*x - 6 > 0 ∧ (5 - x)/2 < 1) ↔ x > 3 := by
  sorry

end simplify_expression_solve_inequality_system_l1139_113943


namespace arc_length_30_degree_sector_l1139_113907

/-- The length of an arc in a sector with radius 1 cm and central angle 30° is π/6 cm. -/
theorem arc_length_30_degree_sector (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 1 → θ = 30 * π / 180 → l = r * θ → l = π / 6 := by
  sorry

end arc_length_30_degree_sector_l1139_113907


namespace max_value_of_product_l1139_113978

/-- Given real numbers x and y that satisfy x + y = 1, 
    the maximum value of (x^3 + 1)(y^3 + 1) is 4. -/
theorem max_value_of_product (x y : ℝ) (h : x + y = 1) :
  ∃ M : ℝ, M = 4 ∧ ∀ x y : ℝ, x + y = 1 → (x^3 + 1) * (y^3 + 1) ≤ M :=
by sorry

end max_value_of_product_l1139_113978


namespace additive_function_properties_l1139_113926

/-- A function satisfying f(x + y) = f(x) + f(y) for all real x and y -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_function_properties (f : ℝ → ℝ) (h : AdditiveFunction f) :
  (∀ x : ℝ, f (-x) = -f x) ∧ f 24 = -8 * f (-3) := by
  sorry

end additive_function_properties_l1139_113926


namespace quadrilateral_diagonal_length_l1139_113968

theorem quadrilateral_diagonal_length 
  (offset1 offset2 total_area : ℝ) 
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : total_area = 180)
  (h4 : total_area = (offset1 + offset2) * diagonal / 2) :
  diagonal = 24 :=
sorry

end quadrilateral_diagonal_length_l1139_113968


namespace rabbit_problem_l1139_113993

/-- The cost price of an Auspicious Rabbit -/
def auspicious_cost : ℝ := 40

/-- The cost price of a Lucky Rabbit -/
def lucky_cost : ℝ := 44

/-- The selling price of an Auspicious Rabbit -/
def auspicious_price : ℝ := 60

/-- The selling price of a Lucky Rabbit -/
def lucky_price : ℝ := 70

/-- The total number of rabbits to be purchased -/
def total_rabbits : ℕ := 200

/-- The minimum required profit -/
def min_profit : ℝ := 4120

/-- The quantity ratio of Lucky Rabbits to Auspicious Rabbits based on the given costs -/
axiom quantity_ratio : (8800 / lucky_cost) = 2 * (4000 / auspicious_cost)

/-- The cost difference between Lucky and Auspicious Rabbits -/
axiom cost_difference : lucky_cost = auspicious_cost + 4

/-- Theorem stating the correct cost prices and minimum number of Lucky Rabbits -/
theorem rabbit_problem :
  (auspicious_cost = 40 ∧ lucky_cost = 44) ∧
  (∀ m : ℕ, m ≥ 20 →
    (lucky_price - lucky_cost) * m + (auspicious_price - auspicious_cost) * (total_rabbits - m) ≥ min_profit) ∧
  (∀ m : ℕ, m < 20 →
    (lucky_price - lucky_cost) * m + (auspicious_price - auspicious_cost) * (total_rabbits - m) < min_profit) :=
sorry

end rabbit_problem_l1139_113993


namespace smallest_sum_of_factors_l1139_113952

theorem smallest_sum_of_factors (r s t : ℕ+) (h : r * s * t = 1230) :
  ∃ (r' s' t' : ℕ+), r' * s' * t' = 1230 ∧ r' + s' + t' = 52 ∧
  ∀ (x y z : ℕ+), x * y * z = 1230 → r' + s' + t' ≤ x + y + z :=
sorry

end smallest_sum_of_factors_l1139_113952


namespace perpendicular_condition_l1139_113990

theorem perpendicular_condition (x : ℝ) :
  let a : ℝ × ℝ := (1, 2*x)
  let b : ℝ × ℝ := (4, -x)
  (x = Real.sqrt 2 → a.1 * b.1 + a.2 * b.2 = 0) ∧
  ¬(a.1 * b.1 + a.2 * b.2 = 0 → x = Real.sqrt 2) :=
by sorry

end perpendicular_condition_l1139_113990


namespace jake_lawn_mowing_earnings_l1139_113974

/-- Jake's desired hourly rate in dollars -/
def desired_hourly_rate : ℝ := 20

/-- Time taken to mow the lawn in hours -/
def lawn_mowing_time : ℝ := 1

/-- Time taken to plant flowers in hours -/
def flower_planting_time : ℝ := 2

/-- Total charge for planting flowers in dollars -/
def flower_planting_charge : ℝ := 45

/-- Earnings for mowing the lawn in dollars -/
def lawn_mowing_earnings : ℝ := desired_hourly_rate * lawn_mowing_time

theorem jake_lawn_mowing_earnings :
  lawn_mowing_earnings = 20 := by sorry

end jake_lawn_mowing_earnings_l1139_113974


namespace cyclic_fraction_product_l1139_113941

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 :=
by sorry

end cyclic_fraction_product_l1139_113941


namespace sqrt_meaningful_iff_geq_neg_one_l1139_113918

theorem sqrt_meaningful_iff_geq_neg_one (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end sqrt_meaningful_iff_geq_neg_one_l1139_113918


namespace quadratic_roots_and_triangle_l1139_113908

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : Prop := x^2 - (2*k + 1)*x + k^2 + k = 0

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a right triangle with sides a, b, c
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

-- Main theorem
theorem quadratic_roots_and_triangle (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_eq k x ∧ quadratic_eq k y) ∧
  (∃ a b : ℝ, quadratic_eq k a ∧ quadratic_eq k b ∧ is_right_triangle a b 5 → k = 3 ∨ k = 12) :=
sorry

end quadratic_roots_and_triangle_l1139_113908


namespace rectangle_area_rectangle_area_proof_l1139_113916

theorem rectangle_area (square_area : Real) (rectangle_length_factor : Real) : Real :=
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_factor * rectangle_width
  rectangle_width * rectangle_length

theorem rectangle_area_proof :
  rectangle_area 36 3 = 108 := by
  sorry

end rectangle_area_rectangle_area_proof_l1139_113916


namespace compute_expression_l1139_113914

theorem compute_expression : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 := by
sorry

end compute_expression_l1139_113914


namespace interest_rate_is_four_percent_l1139_113989

/-- Given a loan with simple interest, prove that the interest rate is 4% per annum -/
theorem interest_rate_is_four_percent
  (P : ℚ) -- Principal amount
  (t : ℚ) -- Time in years
  (I : ℚ) -- Interest amount
  (h1 : P = 250) -- Sum lent is Rs. 250
  (h2 : t = 8) -- Time period is 8 years
  (h3 : I = P - 170) -- Interest is Rs. 170 less than sum lent
  (h4 : I = P * r * t / 100) -- Simple interest formula
  : r = 4 := by
  sorry

end interest_rate_is_four_percent_l1139_113989


namespace notebook_statements_l1139_113963

theorem notebook_statements :
  ∃! n : Fin 40, (∀ m : Fin 40, (m.val + 1 = n.val) ↔ (m = n)) ∧ n.val = 39 :=
sorry

end notebook_statements_l1139_113963


namespace inscribed_squares_area_ratio_l1139_113983

theorem inscribed_squares_area_ratio (r : ℝ) (r_pos : r > 0) :
  let semicircle_square_area := (4 / 5) * r^2
  let equilateral_triangle_side := 2 * r
  let triangle_square_area := r^2
  semicircle_square_area / triangle_square_area = 4 / 5 := by
sorry

end inscribed_squares_area_ratio_l1139_113983


namespace complex_magnitude_eighth_power_l1139_113973

theorem complex_magnitude_eighth_power : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^8 = 1 := by sorry

end complex_magnitude_eighth_power_l1139_113973


namespace minimal_additional_squares_l1139_113923

/-- A point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- The grid configuration --/
structure Grid where
  size : Nat
  shaded : List Point

/-- Check if a point is within the grid --/
def inGrid (p : Point) (g : Grid) : Prop :=
  p.x < g.size ∧ p.y < g.size

/-- Check if a point is shaded --/
def isShaded (p : Point) (g : Grid) : Prop :=
  p ∈ g.shaded

/-- Reflect a point horizontally --/
def reflectHorizontal (p : Point) (g : Grid) : Point :=
  ⟨p.x, g.size - 1 - p.y⟩

/-- Reflect a point vertically --/
def reflectVertical (p : Point) (g : Grid) : Point :=
  ⟨g.size - 1 - p.x, p.y⟩

/-- Check if the grid has horizontal symmetry --/
def hasHorizontalSymmetry (g : Grid) : Prop :=
  ∀ p, inGrid p g → (isShaded p g ↔ isShaded (reflectHorizontal p g) g)

/-- Check if the grid has vertical symmetry --/
def hasVerticalSymmetry (g : Grid) : Prop :=
  ∀ p, inGrid p g → (isShaded p g ↔ isShaded (reflectVertical p g) g)

/-- The initial grid configuration --/
def initialGrid : Grid :=
  { size := 6
  , shaded := [⟨0,5⟩, ⟨2,3⟩, ⟨3,2⟩, ⟨5,0⟩] }

/-- The theorem to prove --/
theorem minimal_additional_squares :
  ∃ (additionalSquares : List Point),
    additionalSquares.length = 1 ∧
    let newGrid : Grid := { size := initialGrid.size, shaded := initialGrid.shaded ++ additionalSquares }
    hasHorizontalSymmetry newGrid ∧ hasVerticalSymmetry newGrid ∧
    ∀ (otherSquares : List Point),
      otherSquares.length < additionalSquares.length →
      let otherGrid : Grid := { size := initialGrid.size, shaded := initialGrid.shaded ++ otherSquares }
      ¬(hasHorizontalSymmetry otherGrid ∧ hasVerticalSymmetry otherGrid) :=
by sorry

end minimal_additional_squares_l1139_113923


namespace base_prime_rep_360_l1139_113975

def base_prime_representation (n : ℕ) : List ℕ := sorry

theorem base_prime_rep_360 :
  base_prime_representation 360 = [3, 2, 0, 1] :=
by
  sorry

end base_prime_rep_360_l1139_113975


namespace sin_210_degrees_l1139_113955

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_degrees_l1139_113955


namespace multiple_problem_l1139_113927

theorem multiple_problem (m : ℤ) : 17 = m * (2625 / 1000) - 4 ↔ m = 8 := by
  sorry

end multiple_problem_l1139_113927


namespace jason_initial_cards_l1139_113971

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end jason_initial_cards_l1139_113971


namespace paulo_children_ages_l1139_113925

theorem paulo_children_ages :
  ∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 12 ∧ a * b * c = 30 :=
by sorry

end paulo_children_ages_l1139_113925


namespace multiply_72519_9999_l1139_113966

theorem multiply_72519_9999 : 72519 * 9999 = 725117481 := by
  sorry

end multiply_72519_9999_l1139_113966


namespace cost_of_chips_l1139_113977

/-- The cost of chips when three friends split the bill equally -/
theorem cost_of_chips (num_friends : ℕ) (num_bags : ℕ) (payment_per_friend : ℚ) : 
  num_friends = 3 → num_bags = 5 → payment_per_friend = 5 →
  (num_friends * payment_per_friend) / num_bags = 3 := by
  sorry

end cost_of_chips_l1139_113977


namespace letter_distribution_l1139_113996

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- There are 5 distinct letters -/
def num_letters : ℕ := 5

/-- There are 3 distinct mailboxes -/
def num_mailboxes : ℕ := 3

/-- The number of ways to distribute 5 letters into 3 mailboxes is 3^5 -/
theorem letter_distribution : distribute num_letters num_mailboxes = 3^5 := by
  sorry

end letter_distribution_l1139_113996


namespace trigonometric_identity_l1139_113901

theorem trigonometric_identity :
  Real.cos (17 * π / 180) * Real.sin (43 * π / 180) +
  Real.sin (163 * π / 180) * Real.sin (47 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end trigonometric_identity_l1139_113901


namespace rectangle_width_proof_l1139_113997

/-- Proves that the width of a rectangle is 14 cm given specific conditions -/
theorem rectangle_width_proof (length width perimeter : ℝ) (triangle_side : ℝ) : 
  length = 10 →
  perimeter = 2 * (length + width) →
  perimeter = 3 * triangle_side →
  triangle_side = 16 →
  width = 14 :=
by sorry

end rectangle_width_proof_l1139_113997


namespace valid_triplets_l1139_113940

def is_valid_triplet (m n p : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ Nat.Prime p ∧ (Nat.choose m 3 - 4 = p^n)

theorem valid_triplets :
  ∀ m n p : ℕ, is_valid_triplet m n p ↔ (m = 7 ∧ n = 1 ∧ p = 31) ∨ (m = 6 ∧ n = 4 ∧ p = 2) :=
sorry

end valid_triplets_l1139_113940


namespace circle_center_l1139_113960

/-- The equation of a circle C in the form x^2 + y^2 + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center of a circle -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a circle C with equation x^2 + y^2 - 2x + y + 1/4 = 0,
    its center is the point (1, -1/2) -/
theorem circle_center (C : Circle) 
  (h : C = { a := -2, b := 1, c := 1/4 }) : 
  ∃ center : Point, center = { x := 1, y := -1/2 } :=
sorry

end circle_center_l1139_113960


namespace fish_count_l1139_113933

theorem fish_count (num_bowls : ℕ) (fish_per_bowl : ℕ) (h1 : num_bowls = 261) (h2 : fish_per_bowl = 23) :
  num_bowls * fish_per_bowl = 6003 := by
  sorry

end fish_count_l1139_113933


namespace max_slope_no_lattice_points_l1139_113920

theorem max_slope_no_lattice_points :
  ∃ (a : ℚ), a = 17/51 ∧
  (∀ (m : ℚ) (x y : ℤ), 1/3 < m → m < a → 1 ≤ x → x ≤ 50 →
    y = m * x + 3 → ¬(∃ (x' y' : ℤ), x' = x ∧ y' = y)) ∧
  (∀ (a' : ℚ), a < a' →
    ∃ (m : ℚ) (x y : ℤ), 1/3 < m → m < a' → 1 ≤ x → x ≤ 50 →
      y = m * x + 3 ∧ (∃ (x' y' : ℤ), x' = x ∧ y' = y)) :=
by sorry

end max_slope_no_lattice_points_l1139_113920


namespace total_cost_is_17_l1139_113931

/-- The total cost of ingredients for Pauline's tacos -/
def total_cost (taco_shells_cost : ℝ) (bell_pepper_cost : ℝ) (bell_pepper_quantity : ℕ) (meat_cost_per_pound : ℝ) (meat_quantity : ℝ) : ℝ :=
  taco_shells_cost + bell_pepper_cost * bell_pepper_quantity + meat_cost_per_pound * meat_quantity

/-- Proof that the total cost of ingredients for Pauline's tacos is $17 -/
theorem total_cost_is_17 :
  total_cost 5 1.5 4 3 2 = 17 := by
  sorry

end total_cost_is_17_l1139_113931


namespace trigonometric_identity_l1139_113946

theorem trigonometric_identity (α : Real) 
  (h : (1 + Real.tan α) / (1 - Real.tan α) = 2016) : 
  1 / Real.cos (2 * α) + Real.tan (2 * α) = 2016 := by
  sorry

end trigonometric_identity_l1139_113946


namespace binomial_150_150_l1139_113953

theorem binomial_150_150 : (150 : ℕ).choose 150 = 1 := by sorry

end binomial_150_150_l1139_113953


namespace system_solution_l1139_113985

theorem system_solution :
  ∃! (x y : ℚ), (2 * x + 3 * y = (7 - 2 * x) + (7 - 3 * y)) ∧
                 (3 * x - 2 * y = (x - 2) + (y - 2)) ∧
                 x = 3 / 4 ∧ y = 11 / 6 := by
  sorry

end system_solution_l1139_113985


namespace hyperbola_eccentricity_range_l1139_113998

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + ((a + 1) / a) ^ 2)
  Real.sqrt 2 < e ∧ e < Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_range_l1139_113998


namespace unique_function_existence_l1139_113938

/-- Given positive real numbers a and b, and X being the set of non-negative real numbers,
    there exists a unique function f: X → X such that f(f(x)) = b(a + b)x - af(x) for all x ∈ X,
    and this function is f(x) = bx. -/
theorem unique_function_existence (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃! f : {x : ℝ | 0 ≤ x} → {x : ℝ | 0 ≤ x},
    (∀ x, f (f x) = b * (a + b) * x - a * f x) ∧
    (∀ x, f x = b * x) := by
  sorry

end unique_function_existence_l1139_113938


namespace vector_problem_l1139_113976

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-1, 7)

theorem vector_problem :
  (a.1 * b.1 + a.2 * b.2 = 25) ∧
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4) := by
  sorry

end vector_problem_l1139_113976


namespace min_value_expression_l1139_113905

theorem min_value_expression (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 ≥ 4 * (4:ℝ)^(1/5) ∧
  (3 * Real.sqrt x + 4 / x^2 = 4 * (4:ℝ)^(1/5) ↔ x = (4:ℝ)^(2/5)) :=
sorry

end min_value_expression_l1139_113905


namespace carrot_bundle_price_is_two_dollars_l1139_113930

/-- Represents the farmer's harvest and sales data -/
structure FarmerData where
  potatoes : ℕ
  carrots : ℕ
  potatoesPerBundle : ℕ
  carrotsPerBundle : ℕ
  potatoBundlePrice : ℚ
  totalRevenue : ℚ

/-- Calculates the price of each carrot bundle -/
def carrotBundlePrice (data : FarmerData) : ℚ :=
  let potatoBundles := data.potatoes / data.potatoesPerBundle
  let potatoRevenue := potatoBundles * data.potatoBundlePrice
  let carrotRevenue := data.totalRevenue - potatoRevenue
  let carrotBundles := data.carrots / data.carrotsPerBundle
  carrotRevenue / carrotBundles

/-- Theorem stating that the carrot bundle price is $2.00 -/
theorem carrot_bundle_price_is_two_dollars 
  (data : FarmerData) 
  (h1 : data.potatoes = 250)
  (h2 : data.carrots = 320)
  (h3 : data.potatoesPerBundle = 25)
  (h4 : data.carrotsPerBundle = 20)
  (h5 : data.potatoBundlePrice = 19/10)
  (h6 : data.totalRevenue = 51) :
  carrotBundlePrice data = 2 := by
  sorry

#eval carrotBundlePrice {
  potatoes := 250,
  carrots := 320,
  potatoesPerBundle := 25,
  carrotsPerBundle := 20,
  potatoBundlePrice := 19/10,
  totalRevenue := 51
}

end carrot_bundle_price_is_two_dollars_l1139_113930


namespace max_value_of_e_l1139_113912

def b (n : ℕ) : ℕ := (10^n - 9) / 3

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n+1))

theorem max_value_of_e :
  (∀ n : ℕ, e n ≤ 3) ∧ (∃ n : ℕ, e n = 3) :=
sorry

end max_value_of_e_l1139_113912


namespace weight_of_b_l1139_113934

/-- Given three weights a, b, and c, prove that b = 33 under the given conditions. -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 41 →
  (b + c) / 2 = 43 →
  b = 33 := by
sorry

end weight_of_b_l1139_113934


namespace area_ratio_of_angle_bisector_l1139_113994

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the lengths of the sides
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the angle bisector
def is_angle_bisector (X P Y Z : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_ratio_of_angle_bisector (XYZ : Triangle) (P : ℝ × ℝ) :
  side_length XYZ.X XYZ.Y = 20 →
  side_length XYZ.X XYZ.Z = 30 →
  side_length XYZ.Y XYZ.Z = 26 →
  is_angle_bisector XYZ.X P XYZ.Y XYZ.Z →
  (triangle_area XYZ.X XYZ.Y P) / (triangle_area XYZ.X XYZ.Z P) = 2 / 3 := by
  sorry

end area_ratio_of_angle_bisector_l1139_113994


namespace incorrect_equation_is_false_l1139_113959

/-- Represents the number of 1-yuan stamps purchased -/
def x : ℕ := sorry

/-- The total number of stamps purchased -/
def total_stamps : ℕ := 12

/-- The total amount spent in yuan -/
def total_spent : ℕ := 20

/-- The equation representing the correct relationship between x, total stamps, and total spent -/
def correct_equation : Prop := x + 2 * (total_stamps - x) = total_spent

/-- The incorrect equation to be proven false -/
def incorrect_equation : Prop := 2 * (total_stamps - x) - total_spent = x

theorem incorrect_equation_is_false :
  correct_equation → ¬incorrect_equation := by sorry

end incorrect_equation_is_false_l1139_113959


namespace unique_solution_for_equation_l1139_113956

theorem unique_solution_for_equation (x r p n : ℕ+) : 
  (x ^ r.val - 1 = p ^ n.val) ∧ 
  (Nat.Prime p.val) ∧ 
  (r.val ≥ 2) ∧ 
  (n.val ≥ 2) → 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := by
sorry

end unique_solution_for_equation_l1139_113956


namespace new_triangle_is_right_triangle_l1139_113911

/-- Given a right triangle with legs a and b, hypotenuse c, and altitude h on the hypotenuse,
    prove that the triangle formed by sides c+h, a+b, and h is also a right triangle. -/
theorem new_triangle_is_right_triangle
  (a b c h : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (altitude_relation : a * b = c * h)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (h_pos : 0 < h) :
  (c + h)^2 = (a + b)^2 + h^2 :=
sorry

end new_triangle_is_right_triangle_l1139_113911


namespace complement_intersection_theorem_l1139_113950

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≥ 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.univ \ A) ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end complement_intersection_theorem_l1139_113950


namespace root_product_l1139_113932

theorem root_product (d e : ℤ) : 
  (∀ s : ℂ, s^2 - 2*s - 1 = 0 → s^5 - d*s - e = 0) → 
  d * e = 348 := by
sorry

end root_product_l1139_113932


namespace parabola_circle_intersection_l1139_113972

/-- Parabola with focus F and equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Circle intersecting y-axis -/
structure IntersectingCircle (M : PointOnParabola C) where
  radius : ℝ
  chord_length : ℝ
  h_chord : chord_length = 2 * Real.sqrt 5
  h_radius_eq : radius^2 = M.x^2 + 5

/-- Line intersecting parabola -/
structure IntersectingLine (C : Parabola) where
  slope : ℝ
  x_intercept : ℝ
  h_slope : slope = Real.pi / 4
  h_intercept : x_intercept = 2

/-- Intersection points of line and parabola -/
structure IntersectionPoints (C : Parabola) (l : IntersectingLine C) where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h_on_parabola₁ : y₁^2 = 2 * C.p * x₁
  h_on_parabola₂ : y₂^2 = 2 * C.p * x₂
  h_on_line₁ : y₁ = l.slope * (x₁ - l.x_intercept)
  h_on_line₂ : y₂ = l.slope * (x₂ - l.x_intercept)

theorem parabola_circle_intersection
  (C : Parabola)
  (M : PointOnParabola C)
  (circle : IntersectingCircle M)
  (l : IntersectingLine C)
  (points : IntersectionPoints C l) :
  circle.radius = 3 ∧ x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end parabola_circle_intersection_l1139_113972


namespace geometric_sequence_property_l1139_113969

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 8 = 1/2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 1/4 := by
  sorry

end geometric_sequence_property_l1139_113969


namespace equation_graph_is_x_axis_l1139_113935

theorem equation_graph_is_x_axis : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 - 2*x*y ↔ y = 0 :=
by sorry

end equation_graph_is_x_axis_l1139_113935


namespace shopping_spending_l1139_113900

/-- The total spending of Elizabeth, Emma, and Elsa given their spending relationships -/
theorem shopping_spending (emma_spending : ℕ) : emma_spending = 58 →
  (emma_spending + 2 * emma_spending + 4 * (2 * emma_spending) = 638) := by
  sorry

#check shopping_spending

end shopping_spending_l1139_113900


namespace fifty_men_left_l1139_113987

/-- Represents the scenario of a hostel with changing occupancy and food provisions. -/
structure Hostel where
  initialMen : ℕ
  initialDays : ℕ
  finalDays : ℕ

/-- Calculates the number of men who left the hostel based on the change in provision duration. -/
def menWhoLeft (h : Hostel) : ℕ :=
  h.initialMen - (h.initialMen * h.initialDays) / h.finalDays

/-- Theorem stating that in the given hostel scenario, 50 men left. -/
theorem fifty_men_left (h : Hostel)
  (h_initial_men : h.initialMen = 250)
  (h_initial_days : h.initialDays = 36)
  (h_final_days : h.finalDays = 45) :
  menWhoLeft h = 50 := by
  sorry

end fifty_men_left_l1139_113987


namespace initial_money_calculation_l1139_113962

theorem initial_money_calculation (initial_money : ℚ) : 
  (initial_money * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 400) → 
  initial_money = 1000 := by
  sorry

end initial_money_calculation_l1139_113962


namespace perpendicular_line_equation_l1139_113970

/-- The equation of a line passing through the intersection of two lines and perpendicular to a third line -/
theorem perpendicular_line_equation (a b c d e f g h i j : ℝ) :
  let l₁ : ℝ × ℝ → Prop := λ p => a * p.1 + b * p.2 = 0
  let l₂ : ℝ × ℝ → Prop := λ p => c * p.1 + d * p.2 + e = 0
  let l₃ : ℝ × ℝ → Prop := λ p => f * p.1 + g * p.2 + h = 0
  let l₄ : ℝ × ℝ → Prop := λ p => i * p.1 + j * p.2 + 5 = 0
  (∃! p, l₁ p ∧ l₂ p) →  -- l₁ and l₂ intersect at a unique point
  (∀ p q : ℝ × ℝ, l₃ p ∧ l₃ q → (p.1 - q.1) * (f * (p.1 - q.1) + g * (p.2 - q.2)) + (p.2 - q.2) * (g * (p.1 - q.1) - f * (p.2 - q.2)) = 0) →  -- l₄ is perpendicular to l₃
  (a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = -2 ∧ f = 2 ∧ g = 1 ∧ h = 3 ∧ i = 1 ∧ j = -2) →
  ∀ p, l₁ p ∧ l₂ p → l₄ p  -- The point of intersection of l₁ and l₂ satisfies l₄
  := by sorry

end perpendicular_line_equation_l1139_113970


namespace arithmetic_sequences_ratio_l1139_113910

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_ratio :
  let num_sum := arithmetic_sum 4 4 72
  let den_sum := arithmetic_sum 5 5 90
  num_sum / den_sum = 76 / 95 := by sorry

end arithmetic_sequences_ratio_l1139_113910


namespace second_concert_attendance_l1139_113995

theorem second_concert_attendance 
  (first_concert : Nat) 
  (attendance_increase : Nat) 
  (h1 : first_concert = 65899)
  (h2 : attendance_increase = 119) :
  first_concert + attendance_increase = 66018 := by
  sorry

end second_concert_attendance_l1139_113995


namespace f_difference_l1139_113992

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(420) - f(360) = 143/20 -/
theorem f_difference : f 420 - f 360 = 143 / 20 := by sorry

end f_difference_l1139_113992


namespace investment_increase_l1139_113903

/-- Represents the broker's investments over three years -/
def investment_change (S R : ℝ) : ℝ := by
  -- Define the changes for each year
  let year1_stock := S * 1.5
  let year1_real_estate := R * 1.2
  
  let year2_stock := year1_stock * 0.7
  let year2_real_estate := year1_real_estate * 1.1
  
  let year3_stock_initial := year2_stock + 0.5 * S
  let year3_real_estate_initial := year2_real_estate - 0.2 * R
  
  let year3_stock_final := year3_stock_initial * 1.25
  let year3_real_estate_final := year3_real_estate_initial * 0.95
  
  -- Calculate the net change
  let net_change := (year3_stock_final + year3_real_estate_final) - (S + R)
  
  exact net_change

/-- Theorem stating the net increase in investment wealth -/
theorem investment_increase (S R : ℝ) : 
  investment_change S R = 0.9375 * S + 0.064 * R := by
  sorry

end investment_increase_l1139_113903


namespace zoo_camels_l1139_113991

theorem zoo_camels (a : ℕ) 
  (h1 : ∃ x y : ℕ, x = y + 10 ∧ x + y = a)
  (h2 : ∃ x y : ℕ, x + 2*y = 55 ∧ x + y = a) : 
  a = 40 := by
sorry

end zoo_camels_l1139_113991


namespace min_chord_length_l1139_113915

/-- The minimum chord length of a circle intersected by a line passing through a fixed point -/
theorem min_chord_length (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : 
  O = (2, 3) → r = 3 → P = (1, 1) → 
  let d := Real.sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2)
  ∃ (A B : ℝ × ℝ), (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧ 
                   (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
                   (∀ (X Y : ℝ × ℝ), 
                     (X.1 - O.1)^2 + (X.2 - O.2)^2 = r^2 → 
                     (Y.1 - O.1)^2 + (Y.2 - O.2)^2 = r^2 →
                     Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) ≥ 
                     Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
                   Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
by sorry


end min_chord_length_l1139_113915


namespace complement_of_16_51_l1139_113961

/-- Represents an angle in degrees and minutes -/
structure DegreeMinute where
  degrees : ℕ
  minutes : ℕ

/-- Calculates the complement of an angle given in degrees and minutes -/
def complement (angle : DegreeMinute) : DegreeMinute :=
  let totalMinutes := 90 * 60 - (angle.degrees * 60 + angle.minutes)
  { degrees := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem complement_of_16_51 :
  complement { degrees := 16, minutes := 51 } = { degrees := 73, minutes := 9 } := by
  sorry

end complement_of_16_51_l1139_113961


namespace angies_age_l1139_113921

theorem angies_age :
  ∀ (A : ℕ), (2 * A + 4 = 20) → A = 8 := by
  sorry

end angies_age_l1139_113921


namespace square_diagonal_length_l1139_113929

/-- The length of the diagonal of a square with area 72 and perimeter 33.94112549695428 is 12 -/
theorem square_diagonal_length (area : ℝ) (perimeter : ℝ) (h_area : area = 72) (h_perimeter : perimeter = 33.94112549695428) :
  let side := (perimeter / 4 : ℝ)
  Real.sqrt (2 * side ^ 2) = 12 := by sorry

end square_diagonal_length_l1139_113929


namespace factorization_equality_l1139_113979

theorem factorization_equality (m a b : ℝ) : 3*m*a^2 - 6*m*a*b + 3*m*b^2 = 3*m*(a-b)^2 := by
  sorry

end factorization_equality_l1139_113979


namespace train_length_l1139_113944

/-- The length of a train given its speed, time to cross a platform, and the platform's length. -/
theorem train_length (train_speed : ℝ) (cross_time : ℝ) (platform_length : ℝ) : 
  train_speed = 72 * (1000 / 3600) → 
  cross_time = 25 → 
  platform_length = 300.04 →
  (train_speed * cross_time - platform_length) = 199.96 := by
  sorry

end train_length_l1139_113944


namespace parallel_lines_b_value_l1139_113937

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ : ℝ} : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ∧ y = m₂ * x + b₂) → m₁ = m₂

/-- The first line equation: 3y - 3b = 9x -/
def line1 (b : ℝ) (x y : ℝ) : Prop := 3 * y - 3 * b = 9 * x

/-- The second line equation: y - 2 = (b + 9)x -/
def line2 (b : ℝ) (x y : ℝ) : Prop := y - 2 = (b + 9) * x

theorem parallel_lines_b_value :
  ∀ b : ℝ, (∀ x y : ℝ, line1 b x y ∧ line2 b x y) → b = -6 := by
  sorry

end parallel_lines_b_value_l1139_113937


namespace isosceles_triangle_from_angle_ratio_l1139_113984

/-- A triangle with angles in the ratio 2:2:1 is isosceles -/
theorem isosceles_triangle_from_angle_ratio (A B C : ℝ) 
  (h_sum : A + B + C = 180) 
  (h_ratio : ∃ (k : ℝ), A = 2*k ∧ B = 2*k ∧ C = k) : 
  A = B ∨ B = C ∨ A = C := by
sorry

end isosceles_triangle_from_angle_ratio_l1139_113984


namespace remainder_987670_div_128_l1139_113957

theorem remainder_987670_div_128 : 987670 % 128 = 22 := by
  sorry

end remainder_987670_div_128_l1139_113957


namespace c_investment_is_10500_l1139_113942

/-- Calculates the investment of partner C given the investments of A and B, 
    the total profit, and A's share of the profit. -/
def calculate_c_investment (a_investment b_investment total_profit a_profit : ℚ) : ℚ :=
  (a_investment * total_profit / a_profit) - a_investment - b_investment

/-- Theorem stating that given the specified conditions, C's investment is 10500. -/
theorem c_investment_is_10500 :
  let a_investment : ℚ := 6300
  let b_investment : ℚ := 4200
  let total_profit : ℚ := 12700
  let a_profit : ℚ := 3810
  calculate_c_investment a_investment b_investment total_profit a_profit = 10500 := by
  sorry

#eval calculate_c_investment 6300 4200 12700 3810

end c_investment_is_10500_l1139_113942


namespace trig_equality_l1139_113904

theorem trig_equality : (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_equality_l1139_113904


namespace basketball_win_percentage_l1139_113981

theorem basketball_win_percentage (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) (remaining_wins : ℕ) :
  total_games = first_games + remaining_games →
  first_games = 55 →
  first_wins = 45 →
  remaining_games = 50 →
  remaining_wins = 34 →
  (first_wins + remaining_wins : ℚ) / total_games = 3 / 4 := by
  sorry

end basketball_win_percentage_l1139_113981


namespace min_contestants_solved_all_l1139_113902

theorem min_contestants_solved_all (total : ℕ) (solved1 solved2 solved3 solved4 : ℕ) 
  (h_total : total = 100)
  (h_solved1 : solved1 = 90)
  (h_solved2 : solved2 = 85)
  (h_solved3 : solved3 = 80)
  (h_solved4 : solved4 = 75) :
  ∃ (min_solved_all : ℕ), 
    min_solved_all ≤ solved1 ∧
    min_solved_all ≤ solved2 ∧
    min_solved_all ≤ solved3 ∧
    min_solved_all ≤ solved4 ∧
    min_solved_all ≥ solved1 + solved2 + solved3 + solved4 - 3 * total ∧
    min_solved_all = 30 :=
sorry

end min_contestants_solved_all_l1139_113902


namespace negation_existence_statement_l1139_113967

theorem negation_existence_statement :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 > 0) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) :=
by sorry

end negation_existence_statement_l1139_113967


namespace two_sqrt_two_less_than_three_l1139_113948

theorem two_sqrt_two_less_than_three : 2 * Real.sqrt 2 < 3 := by
  sorry

end two_sqrt_two_less_than_three_l1139_113948


namespace nested_root_simplification_l1139_113909

theorem nested_root_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x^9)^(1/4) := by
  sorry

end nested_root_simplification_l1139_113909


namespace days_without_email_is_244_l1139_113928

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 365

/-- Represents the email frequency of the first niece -/
def niece1_frequency : ℕ := 4

/-- Represents the email frequency of the second niece -/
def niece2_frequency : ℕ := 6

/-- Represents the email frequency of the third niece -/
def niece3_frequency : ℕ := 8

/-- Calculates the number of days Mr. Thompson did not receive an email from any niece -/
def days_without_email : ℕ :=
  days_in_year - 
  (days_in_year / niece1_frequency + 
   days_in_year / niece2_frequency + 
   days_in_year / niece3_frequency - 
   days_in_year / (niece1_frequency * niece2_frequency) - 
   days_in_year / (niece1_frequency * niece3_frequency) - 
   days_in_year / (niece2_frequency * niece3_frequency) + 
   days_in_year / (niece1_frequency * niece2_frequency * niece3_frequency))

theorem days_without_email_is_244 : days_without_email = 244 := by
  sorry

end days_without_email_is_244_l1139_113928


namespace tens_digit_of_19_pow_2023_l1139_113917

theorem tens_digit_of_19_pow_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
sorry

end tens_digit_of_19_pow_2023_l1139_113917


namespace max_value_of_inverse_sum_l1139_113980

open Real

-- Define the quadratic equation and its roots
def quadratic (t q : ℝ) (x : ℝ) : ℝ := x^2 - t*x + q

-- Define the condition for the roots
def roots_condition (α β : ℝ) : Prop :=
  α + β = α^2 + β^2 ∧ α + β = α^3 + β^3 ∧ α + β = α^4 + β^4 ∧ α + β = α^5 + β^5

-- Theorem statement
theorem max_value_of_inverse_sum (t q α β : ℝ) :
  (∀ x, quadratic t q x = 0 ↔ x = α ∨ x = β) →
  roots_condition α β →
  (∀ γ δ : ℝ, roots_condition γ δ → 1/γ^6 + 1/δ^6 ≤ 2) :=
sorry

end max_value_of_inverse_sum_l1139_113980


namespace sum_of_xyz_l1139_113945

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end sum_of_xyz_l1139_113945


namespace quadratic_equation_integer_roots_l1139_113947

theorem quadratic_equation_integer_roots (a : ℚ) :
  (∃ x y : ℤ, a * x^2 + (a + 1) * x + (a - 1) = 0 ∧
               a * y^2 + (a + 1) * y + (a - 1) = 0 ∧
               x ≠ y) →
  (a = 0 ∨ a = -1/7 ∨ a = 1) :=
by sorry

end quadratic_equation_integer_roots_l1139_113947


namespace zero_point_implies_a_range_l1139_113958

/-- The function f(x) = x^2 + x - 2a has a zero point in the interval (-1, 1) if and only if a ∈ [-1/8, 1) -/
theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 + x - 2*a = 0) ↔ -1/8 ≤ a ∧ a < 1 := by
  sorry

end zero_point_implies_a_range_l1139_113958


namespace expression_value_l1139_113954

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
  sorry

end expression_value_l1139_113954


namespace roses_ratio_l1139_113951

theorem roses_ratio (roses_day1 : ℕ) (roses_day2 : ℕ) (roses_day3 : ℕ) 
  (h1 : roses_day1 = 50)
  (h2 : roses_day2 = roses_day1 + 20)
  (h3 : roses_day1 + roses_day2 + roses_day3 = 220) :
  roses_day3 / roses_day1 = 2 := by
sorry

end roses_ratio_l1139_113951


namespace novel_pages_count_prove_novel_pages_l1139_113936

theorem novel_pages_count : ℕ → Prop :=
  fun total_pages =>
    let day1_read := total_pages / 6 + 10
    let day1_remaining := total_pages - day1_read
    let day2_read := day1_remaining / 5 + 20
    let day2_remaining := day1_remaining - day2_read
    let day3_read := day2_remaining / 4 + 25
    let day3_remaining := day2_remaining - day3_read
    day3_remaining = 80 ∧ total_pages = 252

theorem prove_novel_pages : novel_pages_count 252 := by
  sorry

end novel_pages_count_prove_novel_pages_l1139_113936


namespace meals_neither_kosher_nor_vegan_l1139_113922

/-- Proves the number of meals that are neither kosher nor vegan -/
theorem meals_neither_kosher_nor_vegan 
  (total_clients : ℕ) 
  (vegan_meals : ℕ) 
  (kosher_meals : ℕ) 
  (both_vegan_and_kosher : ℕ) 
  (h1 : total_clients = 30)
  (h2 : vegan_meals = 7)
  (h3 : kosher_meals = 8)
  (h4 : both_vegan_and_kosher = 3) :
  total_clients - (vegan_meals + kosher_meals - both_vegan_and_kosher) = 18 :=
by
  sorry

#check meals_neither_kosher_nor_vegan

end meals_neither_kosher_nor_vegan_l1139_113922


namespace coin_collection_value_l1139_113986

theorem coin_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) 
  (h1 : total_coins = 20)
  (h2 : sample_coins = 4)
  (h3 : sample_value = 16) :
  (total_coins : ℚ) * (sample_value : ℚ) / (sample_coins : ℚ) = 80 := by
  sorry

end coin_collection_value_l1139_113986


namespace no_solution_inequality_l1139_113988

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end no_solution_inequality_l1139_113988


namespace f_of_one_eq_two_l1139_113906

def f (x : ℝ) := x^2 + |x - 2|

theorem f_of_one_eq_two : f 1 = 2 := by sorry

end f_of_one_eq_two_l1139_113906


namespace reduced_banana_price_l1139_113982

/-- Given a 60% reduction in banana prices and the ability to obtain 120 more bananas
    for Rs. 150 after the reduction, prove that the reduced price per dozen bananas
    is Rs. 48/17. -/
theorem reduced_banana_price (P : ℚ) : 
  (150 / (0.4 * P) = 150 / P + 120) →
  (12 * (0.4 * P) = 48 / 17) :=
by sorry

end reduced_banana_price_l1139_113982


namespace parallelogram_has_multiple_altitudes_l1139_113939

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- An altitude of a parallelogram is a line segment from a vertex perpendicular to the opposite side or its extension. -/
structure Altitude (p : Parallelogram) where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- A parallelogram has more than one altitude. -/
theorem parallelogram_has_multiple_altitudes (p : Parallelogram) : ∃ (a b : Altitude p), a ≠ b := by
  sorry

end parallelogram_has_multiple_altitudes_l1139_113939


namespace intersection_sum_l1139_113919

/-- Two circles with centers on the line x + y = 0 intersect at points M(m, 1) and N(-1, n) -/
def circles_intersection (m n : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ), 
    (c₁.1 + c₁.2 = 0) ∧ 
    (c₂.1 + c₂.2 = 0) ∧ 
    ((m - c₁.1)^2 + (1 - c₁.2)^2 = (-1 - c₁.1)^2 + (n - c₁.2)^2) ∧
    ((m - c₂.1)^2 + (1 - c₂.2)^2 = (-1 - c₂.1)^2 + (n - c₂.2)^2)

/-- The theorem to be proved -/
theorem intersection_sum (m n : ℝ) (h : circles_intersection m n) : m + n = 0 := by
  sorry

end intersection_sum_l1139_113919


namespace solution_characterization_l1139_113924

def is_solution (x y z w : ℝ) : Prop :=
  x + y + z + w = 10 ∧
  x^2 + y^2 + z^2 + w^2 = 30 ∧
  x^3 + y^3 + z^3 + w^3 = 100 ∧
  x * y * z * w = 24

def is_permutation_of_1234 (x y z w : ℝ) : Prop :=
  ({x, y, z, w} : Set ℝ) = {1, 2, 3, 4}

theorem solution_characterization :
  ∀ x y z w : ℝ, is_solution x y z w ↔ is_permutation_of_1234 x y z w :=
by sorry

end solution_characterization_l1139_113924
