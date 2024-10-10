import Mathlib

namespace linear_coefficient_of_quadratic_l364_36484

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 -/
def linear_coefficient (a b c : ℚ) : ℚ := b

/-- The quadratic equation 2x^2 - 3x - 4 = 0 -/
def quadratic_equation (x : ℚ) : Prop := 2 * x^2 - 3 * x - 4 = 0

theorem linear_coefficient_of_quadratic :
  linear_coefficient 2 (-3) (-4) = -3 := by sorry

end linear_coefficient_of_quadratic_l364_36484


namespace system_solution_l364_36407

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = m - 3) → 
  (x - y = 2) → 
  (m = 8) := by
sorry

end system_solution_l364_36407


namespace sin_270_degrees_l364_36412

theorem sin_270_degrees : Real.sin (270 * π / 180) = -1 := by
  sorry

end sin_270_degrees_l364_36412


namespace ninth_root_unity_sum_l364_36452

theorem ninth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (Complex.I * (2 * Real.pi / 9)) →
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^6 / (1 + z^9) = 0 := by
  sorry

end ninth_root_unity_sum_l364_36452


namespace cylinder_height_relationship_l364_36449

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end cylinder_height_relationship_l364_36449


namespace triangle_area_inequality_l364_36410

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
  (hα_def : α = 2 * Real.sqrt (b * c))
  (hβ_def : β = 2 * Real.sqrt (c * a))
  (hγ_def : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by
  sorry

end triangle_area_inequality_l364_36410


namespace exp_ln_one_equals_one_l364_36472

theorem exp_ln_one_equals_one : Real.exp (Real.log 1) = 1 := by
  sorry

end exp_ln_one_equals_one_l364_36472


namespace reciprocal_of_two_l364_36498

theorem reciprocal_of_two (m : ℚ) : m - 3 = 1 / 2 → m = 7 / 2 := by
  sorry

end reciprocal_of_two_l364_36498


namespace rice_grains_difference_l364_36477

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ :=
  (List.range n).map grains_on_square |>.sum

theorem rice_grains_difference :
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := by
  sorry

end rice_grains_difference_l364_36477


namespace factory_production_rate_l364_36431

/-- Represents a chocolate factory's production parameters and calculates the hourly production rate. -/
def ChocolateFactory (total_candies : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  total_candies / (days * hours_per_day)

/-- Theorem stating that for the given production parameters, the factory produces 50 candies per hour. -/
theorem factory_production_rate :
  ChocolateFactory 4000 8 10 = 50 := by
  sorry

end factory_production_rate_l364_36431


namespace expression_equality_l364_36461

theorem expression_equality (x y : ℝ) (h : 2 * x - 3 * y = 1) : 
  10 - 4 * x + 6 * y = 8 := by
sorry

end expression_equality_l364_36461


namespace safe_combination_l364_36429

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (n i m a : Digit) : Prop :=
  n ≠ i ∧ n ≠ m ∧ n ≠ a ∧ i ≠ m ∧ i ≠ a ∧ m ≠ a

/-- Converts a three-digit number in base 10 to its decimal value -/
def toDecimal (n i m : Digit) : Nat :=
  100 * n.val + 10 * i.val + m.val

/-- Checks if the equation NIM + AM + MIA = MINA holds in base 10 -/
def equationHolds (n i m a : Digit) : Prop :=
  (100 * n.val + 10 * i.val + m.val) +
  (10 * a.val + m.val) +
  (100 * m.val + 10 * i.val + a.val) =
  (1000 * m.val + 100 * i.val + 10 * n.val + a.val)

theorem safe_combination :
  ∃! (n i m a : Digit), distinct n i m a ∧
  equationHolds n i m a ∧
  toDecimal n i m = 845 := by
sorry

end safe_combination_l364_36429


namespace total_cups_in_trays_l364_36432

theorem total_cups_in_trays (first_tray second_tray : ℕ) 
  (h1 : second_tray = first_tray - 20) 
  (h2 : second_tray = 240) : 
  first_tray + second_tray = 500 := by
  sorry

end total_cups_in_trays_l364_36432


namespace system_solution_l364_36426

theorem system_solution : ∃ x y : ℤ, (3 * x - 14 * y = 2) ∧ (4 * y - x = 6) ∧ x = -46 ∧ y = -10 := by
  sorry

end system_solution_l364_36426


namespace square_value_l364_36475

theorem square_value : ∃ (square : ℚ), (7863 : ℚ) / 13 = 604 + square / 13 ∧ square = 11 := by
  sorry

end square_value_l364_36475


namespace possible_values_of_a_l364_36430

theorem possible_values_of_a (x y a : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : x^3 + y^3 = a) 
  (eq3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
sorry

end possible_values_of_a_l364_36430


namespace larger_square_side_length_l364_36451

theorem larger_square_side_length 
  (shaded_area unshaded_area : ℝ) 
  (h1 : shaded_area = 18)
  (h2 : unshaded_area = 18) : 
  ∃ (side_length : ℝ), side_length = 6 ∧ side_length^2 = shaded_area + unshaded_area :=
by
  sorry

end larger_square_side_length_l364_36451


namespace sara_popsicle_consumption_l364_36441

/-- The number of Popsicles Sara can eat in a given time period -/
def popsicles_eaten (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / minutes_per_popsicle

theorem sara_popsicle_consumption :
  popsicles_eaten 20 340 = 17 := by
  sorry

end sara_popsicle_consumption_l364_36441


namespace shopping_trip_percentage_l364_36402

/-- Represents the percentage of the total amount spent on other items -/
def percentage_other : ℝ := sorry

theorem shopping_trip_percentage :
  let total_amount : ℝ := 100 -- Assume total amount is 100 for percentage calculations
  let clothing_percent : ℝ := 60
  let food_percent : ℝ := 10
  let clothing_tax_rate : ℝ := 4
  let other_tax_rate : ℝ := 8
  let total_tax_percent : ℝ := 4.8

  -- Condition 1, 2, and 3
  clothing_percent + food_percent + percentage_other = total_amount ∧
  -- Condition 4, 5, and 6 (tax calculations)
  clothing_percent * clothing_tax_rate / 100 + percentage_other * other_tax_rate / 100 =
    total_tax_percent ∧
  -- Conclusion
  percentage_other = 30 := by sorry

end shopping_trip_percentage_l364_36402


namespace tan_eleven_pi_thirds_l364_36488

theorem tan_eleven_pi_thirds : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end tan_eleven_pi_thirds_l364_36488


namespace even_composition_l364_36495

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_composition (g : ℝ → ℝ) (h : IsEven g) : IsEven (g ∘ g) := by
  sorry

end even_composition_l364_36495


namespace present_cost_difference_l364_36487

theorem present_cost_difference (cost_first cost_second cost_third : ℕ) : 
  cost_first = 18 →
  cost_third = cost_first - 11 →
  cost_first + cost_second + cost_third = 50 →
  cost_second > cost_first →
  cost_second - cost_first = 7 := by
sorry

end present_cost_difference_l364_36487


namespace correct_propositions_l364_36406

-- Define the planes
variable (α β : Set (Point))

-- Define the property of being a plane
def is_plane (p : Set (Point)) : Prop := sorry

-- Define the property of being distinct
def distinct (p q : Set (Point)) : Prop := p ≠ q

-- Define line
def Line : Type := sorry

-- Define the property of a line being within a plane
def line_in_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define perpendicularity between lines
def perp_lines (l1 l2 : Line) : Prop := sorry

-- Define perpendicularity between a line and a plane
def perp_line_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define perpendicularity between planes
def perp_planes (p q : Set (Point)) : Prop := sorry

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define parallelism between planes
def parallel_planes (p q : Set (Point)) : Prop := sorry

-- State the theorem
theorem correct_propositions 
  (h_planes : is_plane α ∧ is_plane β) 
  (h_distinct : distinct α β) :
  (∀ (l : Line), line_in_plane l α → 
    (∀ (m : Line), line_in_plane m β → perp_lines l m) → 
    perp_planes α β) ∧ 
  (∀ (l : Line), line_in_plane l α → 
    parallel_line_plane l β → 
    parallel_planes α β) ∧ 
  (perp_planes α β → 
    ∃ (l : Line), line_in_plane l α ∧ ¬(perp_line_plane l β)) ∧
  (parallel_planes α β → 
    ∀ (l : Line), line_in_plane l α → 
    parallel_line_plane l β) :=
sorry

end correct_propositions_l364_36406


namespace geometric_sequence_m_value_l364_36422

/-- Definition of the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) (m : ℝ) : ℝ := m * 2^(n - 1) - 3

/-- Definition of the nth term of the geometric sequence -/
def a (n : ℕ) (m : ℝ) : ℝ :=
  if n = 1 then S 1 m
  else S n m - S (n - 1) m

/-- Theorem stating that m = 6 for the given geometric sequence -/
theorem geometric_sequence_m_value :
  ∃ (m : ℝ), ∀ (n : ℕ), n ≥ 1 → (a n m) / (a 1 m) = 2^(n - 1) ∧ m = 6 :=
sorry

end geometric_sequence_m_value_l364_36422


namespace gratuity_calculation_correct_l364_36444

/-- Calculates the gratuity for a restaurant bill given the individual dish prices,
    discount rate, sales tax rate, and tip rate. -/
def calculate_gratuity (prices : List ℝ) (discount_rate sales_tax_rate tip_rate : ℝ) : ℝ :=
  let total_before_discount := prices.sum
  let discounted_total := total_before_discount * (1 - discount_rate)
  let total_with_tax := discounted_total * (1 + sales_tax_rate)
  total_with_tax * tip_rate

/-- The gratuity calculated for the given restaurant bill is correct. -/
theorem gratuity_calculation_correct :
  let prices := [21, 15, 26, 13, 20]
  let discount_rate := 0.15
  let sales_tax_rate := 0.08
  let tip_rate := 0.18
  calculate_gratuity prices discount_rate sales_tax_rate tip_rate = 15.70 := by
  sorry

#eval calculate_gratuity [21, 15, 26, 13, 20] 0.15 0.08 0.18

end gratuity_calculation_correct_l364_36444


namespace fayes_rows_l364_36470

theorem fayes_rows (pencils_per_row : ℕ) (crayons_per_row : ℕ) (total_items : ℕ) : 
  pencils_per_row = 31 →
  crayons_per_row = 27 →
  total_items = 638 →
  total_items / (pencils_per_row + crayons_per_row) = 11 := by
sorry

end fayes_rows_l364_36470


namespace smallest_n_for_equal_candy_costs_l364_36433

theorem smallest_n_for_equal_candy_costs : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p y o : ℕ), p > 0 ∧ y > 0 ∧ o > 0 ∧ 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬∃ (p y o : ℕ), p > 0 ∧ y > 0 ∧ o > 0 ∧ 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * m) ∧
  n = 8 := by
  sorry

end smallest_n_for_equal_candy_costs_l364_36433


namespace polynomial_perfect_square_l364_36405

theorem polynomial_perfect_square (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5*x + 5)^2 := by
  sorry

end polynomial_perfect_square_l364_36405


namespace unique_solution_system_l364_36409

/-- The system of equations has a unique solution when a = 5/3 and no solutions otherwise -/
theorem unique_solution_system (a x y : ℝ) : 
  (3 * (x - a)^2 + y = 2 - a) ∧ 
  (y^2 + ((x - 2) / (|x| - 2))^2 = 1) ∧ 
  (x ≥ 0) ∧ 
  (x ≠ 2) ↔ 
  (a = 5/3 ∧ x = 4/3 ∧ y = 0) :=
sorry

end unique_solution_system_l364_36409


namespace smallest_b_in_arithmetic_sequence_l364_36485

theorem smallest_b_in_arithmetic_sequence (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- All terms are positive
  a = b - d →              -- a is the first term
  c = b + d →              -- c is the third term
  a * b * c = 125 →        -- Product of terms is 125
  ∀ x : ℝ, x > 0 ∧ x < b → ¬∃ y : ℝ, 
    (x - y) > 0 ∧ (x + y) > 0 ∧ (x - y) * x * (x + y) = 125 →
  b = 5 := by
sorry

end smallest_b_in_arithmetic_sequence_l364_36485


namespace geometric_sequence_ratio_l364_36490

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
    (h_a1 : a 1 = 8) (h_a4 : a 4 = 64) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end geometric_sequence_ratio_l364_36490


namespace rhombus_area_l364_36467

/-- The area of a rhombus given its vertices in a rectangular coordinate system -/
theorem rhombus_area (A B C D : ℝ × ℝ) : 
  A = (2, 5.5) → 
  B = (8.5, 1) → 
  C = (2, -3.5) → 
  D = (-4.5, 1) → 
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)
  let cross_product : ℝ := AC.1 * BD.2 - AC.2 * BD.1
  0.5 * |cross_product| = 58.5 := by sorry

end rhombus_area_l364_36467


namespace quadrilateral_JMIT_cyclic_l364_36479

structure Triangle (α : Type*) [Field α] where
  a : α
  b : α
  c : α

def incenter {α : Type*} [Field α] (t : Triangle α) : α :=
  -(t.a * t.b + t.b * t.c + t.c * t.a)

def excenter {α : Type*} [Field α] (t : Triangle α) : α :=
  t.a * t.b - t.b * t.c + t.c * t.a

def midpoint_BC {α : Type*} [Field α] (t : Triangle α) : α :=
  (t.b^2 + t.c^2) / 2

def symmetric_point {α : Type*} [Field α] (t : Triangle α) : α :=
  2 * t.a^2 - t.b * t.c

def is_cyclic {α : Type*} [Field α] (a b c d : α) : Prop :=
  ∃ (k : α), k ≠ 0 ∧ (b - a) * (d - c) = k * (c - a) * (d - b)

theorem quadrilateral_JMIT_cyclic {α : Type*} [Field α] (t : Triangle α) 
  (h1 : t.a^2 ≠ 0) (h2 : t.b^2 ≠ 0) (h3 : t.c^2 ≠ 0)
  (h4 : t.a^2 * t.a^2 = 1) (h5 : t.b^2 * t.b^2 = 1) (h6 : t.c^2 * t.c^2 = 1) :
  is_cyclic (excenter t) (midpoint_BC t) (incenter t) (symmetric_point t) :=
sorry

end quadrilateral_JMIT_cyclic_l364_36479


namespace consecutive_integers_sum_l364_36448

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1 ∧ c = b + 1 ∧ a * b * c = 990) → a + b + c = 30 := by
  sorry

end consecutive_integers_sum_l364_36448


namespace excursion_min_parents_l364_36443

/-- The minimum number of parents needed for an excursion -/
def min_parents_needed (num_students : ℕ) (car_capacity : ℕ) : ℕ :=
  Nat.ceil (num_students / (car_capacity - 1))

/-- Theorem: The minimum number of parents needed for 30 students with 5-seat cars is 8 -/
theorem excursion_min_parents :
  min_parents_needed 30 5 = 8 := by
  sorry

end excursion_min_parents_l364_36443


namespace smallest_integer_l364_36418

theorem smallest_integer (a b : ℕ) (x : ℕ) (h1 : b = 18) (h2 : x > 0)
  (h3 : Nat.gcd a b = x + 3) (h4 : Nat.lcm a b = x * (x + 3)) :
  ∃ (a_min : ℕ), a_min = 6 ∧ ∀ (a' : ℕ), (∃ (x' : ℕ), x' > 0 ∧
    Nat.gcd a' b = x' + 3 ∧ Nat.lcm a' b = x' * (x' + 3)) → a' ≥ a_min :=
sorry

end smallest_integer_l364_36418


namespace a_3_range_l364_36425

def is_convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def b (n : ℕ) : ℝ := n^2 - 6*n + 10

theorem a_3_range (a : ℕ → ℝ) :
  is_convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  (∀ n : ℕ, 1 ≤ n → n < 10 → |a n - b n| ≤ 20) →
  7 ≤ a 3 ∧ a 3 ≤ 19 := by
sorry

end a_3_range_l364_36425


namespace gray_area_calculation_l364_36400

/-- Given two rectangles with dimensions 8x10 and 12x9, and an overlapping area of 37,
    the non-overlapping area in the second rectangle (gray part) is 65. -/
theorem gray_area_calculation (rect1_width rect1_height rect2_width rect2_height black_area : ℕ)
  (h1 : rect1_width = 8)
  (h2 : rect1_height = 10)
  (h3 : rect2_width = 12)
  (h4 : rect2_height = 9)
  (h5 : black_area = 37) :
  rect2_width * rect2_height - (rect1_width * rect1_height - black_area) = 65 := by
sorry

end gray_area_calculation_l364_36400


namespace snake_shedding_decimal_l364_36482

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

/-- The number of ways a snake can shed its skin in octal --/
def snake_shedding_octal : ℕ := 453

theorem snake_shedding_decimal :
  octal_to_decimal snake_shedding_octal = 299 := by
  sorry

end snake_shedding_decimal_l364_36482


namespace equal_temperature_proof_l364_36460

/-- The temperature at which Fahrenheit and Celsius scales are equal -/
def equal_temperature : ℚ := -40

/-- The relation between Fahrenheit (f) and Celsius (c) temperatures -/
def fahrenheit_celsius_relation (c : ℚ) : ℚ := (9/5) * c + 32

/-- Theorem stating that the equal_temperature is the point where Fahrenheit and Celsius scales meet -/
theorem equal_temperature_proof :
  fahrenheit_celsius_relation equal_temperature = equal_temperature := by
  sorry

end equal_temperature_proof_l364_36460


namespace expand_and_simplify_l364_36466

theorem expand_and_simplify (x : ℝ) : 
  (1 + x^3) * (1 - x^4)^2 = 1 + x^3 - 2*x^4 - 2*x^7 + x^8 + x^11 := by
  sorry

end expand_and_simplify_l364_36466


namespace student_sample_size_l364_36469

theorem student_sample_size :
  ∀ (total juniors sophomores freshmen seniors : ℕ),
  juniors = (26 * total) / 100 →
  sophomores = (25 * total) / 100 →
  seniors = 160 →
  freshmen = sophomores + 32 →
  total = freshmen + sophomores + juniors + seniors →
  total = 800 := by
sorry

end student_sample_size_l364_36469


namespace arithmetic_sequence_property_l364_36478

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n}, if a_1 + a_19 = 10, then a_10 = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 1 + a 19 = 10) : 
  a 10 = 5 := by
  sorry

end arithmetic_sequence_property_l364_36478


namespace hash_four_one_l364_36497

-- Define the # operation
def hash (a b : ℤ) : ℤ := (a + b + 2) * (a - b - 2)

-- Theorem statement
theorem hash_four_one : hash 4 1 = 7 := by
  sorry

end hash_four_one_l364_36497


namespace blue_garden_yield_l364_36437

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected carrot yield from a rectangular garden -/
def expectedCarrotYield (garden : GardenDimensions) (stepLength : ℝ) (yieldPerSqFt : ℝ) : ℝ :=
  (garden.length : ℝ) * stepLength * (garden.width : ℝ) * stepLength * yieldPerSqFt

/-- Theorem stating the expected carrot yield for Mr. Blue's garden -/
theorem blue_garden_yield :
  let garden : GardenDimensions := ⟨18, 25⟩
  let stepLength : ℝ := 3
  let yieldPerSqFt : ℝ := 3 / 4
  expectedCarrotYield garden stepLength yieldPerSqFt = 3037.5 := by
  sorry

end blue_garden_yield_l364_36437


namespace abs_one_minus_i_l364_36439

theorem abs_one_minus_i : Complex.abs (1 - Complex.I) = Real.sqrt 2 := by
  sorry

end abs_one_minus_i_l364_36439


namespace factors_imply_unique_h_k_l364_36440

-- Define the polynomial
def P (h k : ℝ) (x : ℝ) : ℝ := 3 * x^4 - h * x^2 + k * x - 12

-- State the theorem
theorem factors_imply_unique_h_k :
  ∀ h k : ℝ,
  (∀ x : ℝ, P h k x = 0 ↔ x = 3 ∨ x = -4) →
  ∃! (h' k' : ℝ), P h' k' = P h k :=
by sorry

end factors_imply_unique_h_k_l364_36440


namespace playground_fence_posts_l364_36458

/-- Calculates the number of fence posts required for a rectangular playground -/
def fence_posts (width : ℕ) (length : ℕ) (post_interval : ℕ) : ℕ :=
  let long_side_posts := length / post_interval + 2
  let short_side_posts := width / post_interval + 1
  long_side_posts + 2 * short_side_posts

/-- Theorem stating the number of fence posts for a 50m by 90m playground -/
theorem playground_fence_posts :
  fence_posts 50 90 10 = 25 := by
  sorry

#eval fence_posts 50 90 10

end playground_fence_posts_l364_36458


namespace distance_MN_equals_5_l364_36436

def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 5)

theorem distance_MN_equals_5 : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 5 := by
  sorry

end distance_MN_equals_5_l364_36436


namespace binomial_probability_5_to_7_successes_in_8_trials_l364_36403

theorem binomial_probability_5_to_7_successes_in_8_trials :
  let n : ℕ := 8
  let p : ℝ := 1/2
  let q : ℝ := 1 - p
  let X : ℕ → ℝ := λ k => Nat.choose n k * p^k * q^(n-k)
  (X 5 + X 6 + X 7) = 23/64 := by sorry

end binomial_probability_5_to_7_successes_in_8_trials_l364_36403


namespace no_factors_l364_36459

def f (x : ℝ) : ℝ := x^4 + 2*x^2 + 9

def g₁ (x : ℝ) : ℝ := x^2 + 3
def g₂ (x : ℝ) : ℝ := x + 1
def g₃ (x : ℝ) : ℝ := x^2 - 3
def g₄ (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem no_factors : 
  (∀ x : ℝ, f x ≠ 0 → g₁ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₂ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₃ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₄ x ≠ 0) :=
by sorry

end no_factors_l364_36459


namespace number_of_model_X_computers_prove_number_of_model_X_computers_l364_36411

/-- Represents the time (in minutes) for a model X computer to complete the task -/
def modelXTime : ℝ := 72

/-- Represents the time (in minutes) for a model Y computer to complete the task -/
def modelYTime : ℝ := 36

/-- Represents the total time (in minutes) for the combined computers to complete the task -/
def totalTime : ℝ := 1

/-- Theorem stating that the number of model X computers used is 24 -/
theorem number_of_model_X_computers : ℕ :=
  24

/-- Proof that the number of model X computers used is 24 -/
theorem prove_number_of_model_X_computers :
  (modelXTime : ℝ) > 0 ∧ (modelYTime : ℝ) > 0 ∧ totalTime > 0 →
  ∃ (n : ℕ), n > 0 ∧ n = number_of_model_X_computers ∧
  (n : ℝ) * (1 / modelXTime + 1 / modelYTime) = 1 / totalTime :=
by
  sorry

end number_of_model_X_computers_prove_number_of_model_X_computers_l364_36411


namespace arithmetic_calculation_l364_36419

theorem arithmetic_calculation : 28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44 = 7275 := by
  sorry

end arithmetic_calculation_l364_36419


namespace equation_solution_l364_36486

theorem equation_solution : ∃! x : ℝ, (Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 4) ∧ (x = 4 * Real.sqrt 11) := by
  sorry

end equation_solution_l364_36486


namespace quadrilateral_properties_l364_36435

-- Define the points
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (4, 1)
def D : ℝ × ℝ := (-2, 4)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AD : ℝ × ℝ := (D.1 - A.1, D.2 - A.2)
def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicular
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Define parallel
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Define trapezoid
def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  parallel (B.1 - A.1, B.2 - A.2) (D.1 - C.1, D.2 - C.2) ∧
  ¬parallel (A.1 - D.1, A.2 - D.2) (B.1 - C.1, B.2 - C.2)

theorem quadrilateral_properties :
  perpendicular AB AD ∧ parallel AB DC ∧ is_trapezoid A B C D := by
  sorry

end quadrilateral_properties_l364_36435


namespace smallest_angle_EBC_l364_36401

theorem smallest_angle_EBC (ABC ABD DBE : ℝ) (h1 : ABC = 40) (h2 : ABD = 30) (h3 : DBE = 10) : 
  ∃ (EBC : ℝ), EBC = 20 ∧ ∀ (x : ℝ), x ≥ 20 → x ≥ EBC := by
  sorry

end smallest_angle_EBC_l364_36401


namespace m_is_smallest_l364_36434

/-- The smallest positive integer satisfying the given divisibility conditions -/
def m : ℕ := 60

theorem m_is_smallest : m = 60 ∧ 
  (∃ (n : ℕ), m = 13 * n + 8 ∧ m = 15 * n) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → 
    ¬(∃ (n : ℕ), k = 13 * n + 8 ∧ k = 15 * n)) := by
  sorry

#check m_is_smallest

end m_is_smallest_l364_36434


namespace geometric_progression_condition_l364_36446

/-- Given real numbers a, b, c with b < 0, prove that b^2 = ac is necessary and 
    sufficient for a, b, c to form a geometric progression -/
theorem geometric_progression_condition (a b c : ℝ) (h : b < 0) :
  (b^2 = a*c) ↔ ∃ r : ℝ, (r ≠ 0 ∧ b = a*r ∧ c = b*r) :=
sorry

end geometric_progression_condition_l364_36446


namespace line_intersection_point_sum_l364_36454

/-- The line equation y = -1/2x + 8 -/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (16, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 8)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs (P.1 * Q.2 - Q.1 * P.2) / 2 = 4 * abs (r * P.2 - P.1 * s) / 2

theorem line_intersection_point_sum : 
  ∀ r s : ℝ, line_equation r s → T_on_PQ r s → area_condition r s → r + s = 14 := by
sorry

end line_intersection_point_sum_l364_36454


namespace min_value_product_quotient_l364_36468

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 3*x + 2) * (y^2 + 3*y + 2) * (z^2 + 3*z + 2) / (x*y*z) ≥ 216 ∧
  (x^2 + 3*x + 2) * (y^2 + 3*y + 2) * (z^2 + 3*z + 2) / (x*y*z) = 216 ↔ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end min_value_product_quotient_l364_36468


namespace certain_number_base_l364_36414

theorem certain_number_base (x y : ℕ) (a : ℝ) 
  (h1 : 3^x * a^y = 3^12) 
  (h2 : x - y = 12) 
  (h3 : x = 12) : 
  a = 1 := by
  sorry

end certain_number_base_l364_36414


namespace player_A_wins_l364_36491

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents the state of the blackboard -/
def BoardState : Type := List Nat

/-- Checks if a move is valid according to the game rules -/
def isValidMove (current : BoardState) (next : BoardState) : Prop :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy : Type := BoardState → BoardState

/-- Checks if a strategy is winning for a given player -/
def isWinningStrategy (player : Player) (strat : Strategy) : Prop :=
  sorry

/-- The initial state of the board -/
def initialState : BoardState := [10^2007]

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_A_wins :
  ∃ (strat : Strategy), isWinningStrategy Player.A strat :=
sorry

end player_A_wins_l364_36491


namespace scientific_notation_929000_l364_36494

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_929000 :
  scientific_notation 929000 = (9.29, 5) :=
sorry

end scientific_notation_929000_l364_36494


namespace selection_methods_with_female_l364_36423

def total_students : ℕ := 8
def male_students : ℕ := 4
def female_students : ℕ := 4
def students_to_select : ℕ := 3

theorem selection_methods_with_female (h1 : total_students = male_students + female_students) 
  (h2 : total_students ≥ students_to_select) :
  (Nat.choose total_students students_to_select) - (Nat.choose male_students students_to_select) = 52 := by
  sorry

end selection_methods_with_female_l364_36423


namespace perpendicular_lines_parallel_l364_36496

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations for parallel and perpendicular
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end perpendicular_lines_parallel_l364_36496


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l364_36492

-- Problem 1
theorem problem_1 : (1) - 27 + (-32) + (-8) + 27 = -40 := by sorry

-- Problem 2
theorem problem_2 : (2) * (-5) + |(-3)| = -2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℤ) (h1 : -x = 3) (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := by sorry

-- Problem 4
theorem problem_4 : (-1 - 1/2) + (1 + 1/4) + (-2 - 1/2) - (-3 - 1/4) - (1 + 1/4) = -3/4 := by sorry

-- Problem 5
theorem problem_5 (a b : ℝ) (h : |a - 4| + |b + 5| = 0) : a - b = 9 := by sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l364_36492


namespace emily_age_proof_l364_36481

/-- Rachel's current age -/
def rachel_current_age : ℕ := 24

/-- Rachel's age when Emily was half her age -/
def rachel_past_age : ℕ := 8

/-- Emily's age when Rachel was 8 -/
def emily_past_age : ℕ := rachel_past_age / 2

/-- The constant age difference between Rachel and Emily -/
def age_difference : ℕ := rachel_past_age - emily_past_age

/-- Emily's current age -/
def emily_current_age : ℕ := rachel_current_age - age_difference

theorem emily_age_proof : emily_current_age = 20 := by
  sorry

end emily_age_proof_l364_36481


namespace quadratic_equation_identification_l364_36455

/-- Definition of a quadratic equation in one variable -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equations given in the problem -/
def eq_A : ℝ → ℝ := λ x => 2 * x - 1
def eq_B : ℝ → ℝ := λ x => x^2
def eq_C : ℝ → ℝ → ℝ := λ x y => 5 * x * y - 1
def eq_D : ℝ → ℝ := λ x => 2 * (x + 1)

/-- Theorem stating that eq_B is quadratic while others are not -/
theorem quadratic_equation_identification :
  is_quadratic eq_B ∧ 
  ¬is_quadratic eq_A ∧ 
  ¬is_quadratic (λ x => eq_C x x) ∧ 
  ¬is_quadratic eq_D :=
sorry

end quadratic_equation_identification_l364_36455


namespace frac_x_div_2_gt_0_is_linear_inequality_l364_36427

/-- A linear inequality in one variable is of the form ax + b > 0 or ax + b < 0,
    where a and b are constants and a ≠ 0. --/
def IsLinearInequality (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ) (rel : ℝ → ℝ → Prop), a ≠ 0 ∧
    (rel = (· > ·) ∨ rel = (· < ·)) ∧
    (∀ x, f x ↔ rel (a * x + b) 0)

/-- The function f(x) = x/2 > 0 is a linear inequality in one variable. --/
theorem frac_x_div_2_gt_0_is_linear_inequality :
  IsLinearInequality (fun x => x / 2 > 0) :=
sorry

end frac_x_div_2_gt_0_is_linear_inequality_l364_36427


namespace set_equality_implies_m_zero_l364_36424

theorem set_equality_implies_m_zero (m : ℝ) : 
  ({3, m} : Set ℝ) = ({3 * m, 3} : Set ℝ) → m = 0 := by
  sorry

end set_equality_implies_m_zero_l364_36424


namespace quadratic_inequality_range_l364_36464

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a < 0 ∨ a ≥ 3) := by
  sorry

end quadratic_inequality_range_l364_36464


namespace equivalent_operation_l364_36445

theorem equivalent_operation (x : ℝ) : (x / (5/6)) * (4/7) = x * (24/35) := by
  sorry

end equivalent_operation_l364_36445


namespace sqrt_equation_solution_l364_36480

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (2 * x - 3) = 10 → x = 103 / 2 := by
  sorry

end sqrt_equation_solution_l364_36480


namespace mike_mark_height_difference_l364_36476

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- The height difference between two people in inches -/
def height_difference (height1 : ℕ) (height2 : ℕ) : ℕ := 
  if height1 ≥ height2 then height1 - height2 else height2 - height1

theorem mike_mark_height_difference :
  let mark_height := height_to_inches 5 3
  let mike_height := height_to_inches 6 1
  height_difference mike_height mark_height = 10 := by
sorry

end mike_mark_height_difference_l364_36476


namespace fourth_house_number_l364_36420

theorem fourth_house_number (x : ℕ) (k : ℕ) : 
  k ≥ 4 → 
  (k + 1) * (x + k) = 78 → 
  x + 6 = 14 :=
by sorry

end fourth_house_number_l364_36420


namespace paint_set_cost_l364_36463

def total_cost (has : ℝ) (needs : ℝ) : ℝ := has + needs
def paintbrush_cost : ℝ := 1.50
def easel_cost : ℝ := 12.65
def albert_has : ℝ := 6.50
def albert_needs : ℝ := 12.00

theorem paint_set_cost :
  total_cost albert_has albert_needs - (paintbrush_cost + easel_cost) = 4.35 := by
  sorry

end paint_set_cost_l364_36463


namespace unique_solution_3x_eq_12_l364_36404

theorem unique_solution_3x_eq_12 : 
  ∀ (x₁ x₂ : ℝ), (3 : ℝ) ^ x₁ = 12 → (3 : ℝ) ^ x₂ = 12 → x₁ = x₂ := by
  sorry

end unique_solution_3x_eq_12_l364_36404


namespace volume_is_304_l364_36462

/-- The volume of the described set of points -/
def total_volume (central_box : ℝ × ℝ × ℝ) (extension : ℝ) : ℝ :=
  let (l, w, h) := central_box
  let box_volume := l * w * h
  let bounding_boxes_volume := 2 * (l * w + l * h + w * h) * extension
  let edge_prism_volume := 2 * (l + w + h) * extension * extension
  box_volume + bounding_boxes_volume + edge_prism_volume

/-- The theorem stating that the total volume is 304 cubic units -/
theorem volume_is_304 :
  total_volume (2, 3, 4) 2 = 304 := by sorry

end volume_is_304_l364_36462


namespace quadratic_function_a_range_l364_36471

theorem quadratic_function_a_range (a b c : ℝ) : 
  a ≠ 0 →
  a * (-1)^2 + b * (-1) + c = 3 →
  a * 1^2 + b * 1 + c = 1 →
  0 < c →
  c < 1 →
  1 < a ∧ a < 2 := by
sorry

end quadratic_function_a_range_l364_36471


namespace sphere_surface_area_tangent_to_cube_l364_36417

/-- The surface area of a sphere tangent to all faces of a cube -/
theorem sphere_surface_area_tangent_to_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) :
  cube_edge_length = 2 →
  sphere_radius = cube_edge_length / 2 →
  4 * Real.pi * sphere_radius^2 = 4 * Real.pi := by
  sorry

end sphere_surface_area_tangent_to_cube_l364_36417


namespace complex_number_modulus_l364_36453

/-- Given a complex number z = (3ai)/(1-2i) where a < 0 and i is the imaginary unit,
    if |z| = √5, then a = -5/3 -/
theorem complex_number_modulus (a : ℝ) (h1 : a < 0) :
  let z : ℂ := (3 * a * Complex.I) / (1 - 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 → a = -5/3 := by
  sorry

end complex_number_modulus_l364_36453


namespace high_correlation_implies_r_close_to_one_l364_36413

/-- Represents the correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Represents the degree of linear correlation between two variables -/
def linear_correlation_degree (x y : ℝ → ℝ) : ℝ := sorry

/-- A high degree of linear correlation -/
def high_correlation : ℝ := sorry

theorem high_correlation_implies_r_close_to_one (x y : ℝ → ℝ) :
  linear_correlation_degree x y ≥ high_correlation →
  ∀ ε > 0, ∃ δ > 0, linear_correlation_degree x y > 1 - δ →
  |correlation_coefficient x y| > 1 - ε :=
sorry

end high_correlation_implies_r_close_to_one_l364_36413


namespace solution_set_l364_36499

def f (x : ℝ) := 3 - 2*x

theorem solution_set (x : ℝ) : 
  x ∈ Set.Icc 0 3 ↔ |f (x + 1) + 2| ≤ 3 :=
by sorry

end solution_set_l364_36499


namespace triangle_equilateral_l364_36447

theorem triangle_equilateral (m n p : ℝ) (h1 : m + n + p = 180) 
  (h2 : |m - n| + (n - p)^2 = 0) : m = n ∧ n = p := by
  sorry

end triangle_equilateral_l364_36447


namespace polynomial_is_perfect_square_l364_36438

theorem polynomial_is_perfect_square (x : ℝ) : 
  ∃ (t u : ℝ), (49/4 : ℝ) * x^2 + 21 * x + 9 = (t * x + u)^2 := by
  sorry

end polynomial_is_perfect_square_l364_36438


namespace tan_alpha_value_l364_36421

theorem tan_alpha_value (α : Real) (h : Real.cos α + 2 * Real.sin α = Real.sqrt 5) :
  Real.tan α = 2 := by
  sorry

end tan_alpha_value_l364_36421


namespace odd_sum_ends_with_1379_l364_36474

theorem odd_sum_ends_with_1379 (S : Finset ℕ) 
  (h1 : S.card = 10000)
  (h2 : ∀ n ∈ S, Odd n)
  (h3 : ∀ n ∈ S, ¬(5 ∣ n)) :
  ∃ T ⊆ S, (T.sum id) % 10000 = 1379 := by
  sorry

end odd_sum_ends_with_1379_l364_36474


namespace chicken_eggs_l364_36457

/-- The number of eggs laid by a chicken over two days -/
def total_eggs (today : ℕ) (yesterday : ℕ) : ℕ := today + yesterday

/-- Theorem: The chicken laid 49 eggs in total over two days -/
theorem chicken_eggs : total_eggs 30 19 = 49 := by
  sorry

end chicken_eggs_l364_36457


namespace basketball_weight_proof_l364_36450

/-- The weight of one basketball in pounds -/
def basketball_weight : ℚ := 125 / 9

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℚ := 25

theorem basketball_weight_proof :
  (9 : ℚ) * basketball_weight = (5 : ℚ) * bicycle_weight ∧
  (3 : ℚ) * bicycle_weight = 75 :=
by sorry

end basketball_weight_proof_l364_36450


namespace f_simplification_f_value_at_specific_angle_l364_36408

noncomputable def f (θ : ℝ) : ℝ :=
  (Real.sin (θ + 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 - θ) * Real.cos (θ + 3 * Real.pi)) /
  (Real.cos (-Real.pi / 2 - θ) * Real.sin (-3 * Real.pi / 2 - θ))

theorem f_simplification (θ : ℝ) : f θ = -Real.cos θ := by sorry

theorem f_value_at_specific_angle (θ : ℝ) (h : Real.sin (θ - Real.pi / 6) = 3 / 5) :
  f (θ + Real.pi / 3) = 3 / 5 := by sorry

end f_simplification_f_value_at_specific_angle_l364_36408


namespace sum_of_modified_numbers_l364_36493

theorem sum_of_modified_numbers (R x y : ℝ) (h : x + y = R) :
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 := by
  sorry

end sum_of_modified_numbers_l364_36493


namespace number_of_divisors_of_M_l364_36473

def M : ℕ := 2^6 * 3^4 * 5^2 * 7^2 * 11^1

theorem number_of_divisors_of_M : (Nat.divisors M).card = 630 := by
  sorry

end number_of_divisors_of_M_l364_36473


namespace bread_baking_time_l364_36483

/-- The time it takes for one ball of dough to rise -/
def rise_time : ℕ := 3

/-- The time it takes to bake one ball of dough -/
def bake_time : ℕ := 2

/-- The number of balls of dough Ellen makes -/
def num_balls : ℕ := 4

/-- The total time taken to make and bake all balls of dough -/
def total_time : ℕ := rise_time + (num_balls - 1) * rise_time + num_balls * bake_time

theorem bread_baking_time :
  total_time = 14 :=
sorry

end bread_baking_time_l364_36483


namespace perpendicular_lines_from_parallel_planes_l364_36456

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (l m : Line) (α β : Plane)
  (different_lines : l ≠ m)
  (non_coincident_planes : α ≠ β)
  (l_perp_α : perpendicular l α)
  (α_parallel_β : parallel α β)
  (m_in_β : contained_in m β) :
  line_perpendicular l m :=
sorry

end perpendicular_lines_from_parallel_planes_l364_36456


namespace gift_distribution_sequences_l364_36416

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of ways to distribute gifts in one class session -/
def ways_per_session : ℕ := num_students * num_students

/-- The total number of different gift distribution sequences in a week -/
def total_sequences : ℕ := ways_per_session ^ meetings_per_week

/-- Theorem stating the total number of different gift distribution sequences -/
theorem gift_distribution_sequences :
  total_sequences = 11390625 := by
  sorry

end gift_distribution_sequences_l364_36416


namespace jim_paycheck_amount_l364_36428

/-- Calculates the final amount on a paycheck after retirement and tax deductions -/
def final_paycheck_amount (gross_pay : ℝ) (retirement_rate : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_rate) - tax_deduction

/-- Theorem stating that given the specific conditions, the final paycheck amount is $740 -/
theorem jim_paycheck_amount :
  final_paycheck_amount 1120 0.25 100 = 740 := by
  sorry

#eval final_paycheck_amount 1120 0.25 100

end jim_paycheck_amount_l364_36428


namespace intersection_points_on_circle_l364_36489

/-- Given two parabolas, prove that their intersection points lie on a circle with radius squared equal to 16 -/
theorem intersection_points_on_circle (x y : ℝ) : 
  y = (x - 2)^2 ∧ x = (y - 5)^2 - 1 → (x - 2)^2 + (y - 5)^2 = 16 := by
  sorry

end intersection_points_on_circle_l364_36489


namespace g_at_negative_three_l364_36465

-- Define the property of g being a rational function satisfying the given equation
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2

-- State the theorem
theorem g_at_negative_three (g : ℝ → ℝ) (h : is_valid_g g) : g (-3) = 247 / 39 := by
  sorry

end g_at_negative_three_l364_36465


namespace point_outside_circle_l364_36442

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 24}
  P ∉ circle ∧ ∀ (x y : ℝ), (x, y) ∈ circle → (m^2 - x)^2 + (5 - y)^2 > 0 :=
by sorry

end point_outside_circle_l364_36442


namespace fraction_simplification_l364_36415

theorem fraction_simplification : 48 / (7 - 3/4) = 192/25 := by
  sorry

end fraction_simplification_l364_36415
