import Mathlib

namespace rulers_remaining_l1885_188507

theorem rulers_remaining (initial_rulers : ℕ) (removed_rulers : ℕ) : 
  initial_rulers = 14 → removed_rulers = 11 → initial_rulers - removed_rulers = 3 :=
by sorry

end rulers_remaining_l1885_188507


namespace product_remainder_mod_seven_l1885_188512

def product_sequence : List ℕ := List.range 10 |>.map (λ i => 3 + 10 * i)

theorem product_remainder_mod_seven :
  (product_sequence.prod) % 7 = 2 := by
  sorry

end product_remainder_mod_seven_l1885_188512


namespace reflect_point_over_x_axis_l1885_188530

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflect_point_over_x_axis :
  let P : Point := { x := -6, y := -9 }
  reflectOverXAxis P = { x := -6, y := 9 } := by
  sorry

end reflect_point_over_x_axis_l1885_188530


namespace problem_statement_l1885_188515

theorem problem_statement (p q : ℝ) (h : p^2 / q^3 = 4 / 5) :
  11/7 + (2*q^3 - p^2) / (2*q^3 + p^2) = 2 := by
  sorry

end problem_statement_l1885_188515


namespace gumballs_eaten_l1885_188561

/-- The number of gumballs in each package -/
def gumballs_per_package : ℝ := 5.0

/-- The number of packages Nathan ate -/
def packages_eaten : ℝ := 20.0

/-- The total number of gumballs Nathan ate -/
def total_gumballs : ℝ := gumballs_per_package * packages_eaten

theorem gumballs_eaten :
  total_gumballs = 100.0 := by sorry

end gumballs_eaten_l1885_188561


namespace nail_trimming_customers_l1885_188587

/-- The number of nails per person -/
def nails_per_person : ℕ := 20

/-- The total number of sounds produced by the nail cutter -/
def total_sounds : ℕ := 100

/-- The number of customers whose nails were trimmed -/
def num_customers : ℕ := total_sounds / nails_per_person

theorem nail_trimming_customers :
  num_customers = 5 :=
sorry

end nail_trimming_customers_l1885_188587


namespace x_range_for_f_l1885_188568

-- Define the function f
def f (x : ℝ) := x^3 + 3*x

-- State the theorem
theorem x_range_for_f (x : ℝ) :
  (∀ m ∈ Set.Icc (-2 : ℝ) 2, f (m*x - 2) + f x < 0) →
  x ∈ Set.Ioo (-2 : ℝ) (2/3) :=
by sorry

end x_range_for_f_l1885_188568


namespace geometric_sequence_problem_l1885_188529

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 210 * r = b ∧ b * r = 140 / 60) → b = 7 * Real.sqrt 10 := by
  sorry

end geometric_sequence_problem_l1885_188529


namespace complex_expression_evaluation_l1885_188552

theorem complex_expression_evaluation : (3 * (3 * (4 * (3 * (4 * (2 + 1) + 1) + 2) + 1) + 2) + 1) = 1492 := by
  sorry

end complex_expression_evaluation_l1885_188552


namespace car_sale_profit_l1885_188508

theorem car_sale_profit (original_price : ℝ) (h : original_price > 0) :
  let purchase_price := 0.80 * original_price
  let selling_price := 1.6000000000000001 * original_price
  let profit_percentage := (selling_price - purchase_price) / purchase_price * 100
  profit_percentage = 100.00000000000001 := by
sorry

end car_sale_profit_l1885_188508


namespace product_trailing_zeros_l1885_188565

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 500 -/
def product : ℕ := 45 * 500

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end product_trailing_zeros_l1885_188565


namespace f_is_odd_iff_l1885_188569

-- Define the function f
def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

-- State the theorem
theorem f_is_odd_iff (a b : ℝ) :
  (∀ x, f a b (-x) = -f a b x) ↔ a^2 + b^2 = 0 :=
by sorry

end f_is_odd_iff_l1885_188569


namespace swallow_flock_max_weight_l1885_188501

/-- Represents the weight capacity of different swallow types and their quantities in a flock -/
structure SwallowFlock where
  american_capacity : ℕ
  european_capacity : ℕ
  african_capacity : ℕ
  total_swallows : ℕ
  american_count : ℕ
  european_count : ℕ
  african_count : ℕ

/-- Calculates the maximum weight a flock of swallows can carry -/
def max_carry_weight (flock : SwallowFlock) : ℕ :=
  flock.american_count * flock.american_capacity +
  flock.european_count * flock.european_capacity +
  flock.african_count * flock.african_capacity

/-- Theorem stating the maximum weight the specific flock can carry -/
theorem swallow_flock_max_weight :
  ∃ (flock : SwallowFlock),
    flock.american_capacity = 5 ∧
    flock.european_capacity = 2 * flock.american_capacity ∧
    flock.african_capacity = 3 * flock.american_capacity ∧
    flock.total_swallows = 120 ∧
    flock.american_count = 2 * flock.european_count ∧
    flock.african_count = 3 * flock.american_count ∧
    flock.american_count + flock.european_count + flock.african_count = flock.total_swallows ∧
    max_carry_weight flock = 1415 :=
  sorry

end swallow_flock_max_weight_l1885_188501


namespace arctan_identity_l1885_188543

theorem arctan_identity (x : Real) : 
  Real.arctan (Real.tan (70 * π / 180) - 2 * Real.tan (35 * π / 180)) = 20 * π / 180 := by
  sorry

end arctan_identity_l1885_188543


namespace simple_interest_problem_l1885_188573

/-- Calculates the total amount after a given period using simple interest -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, prove that the total amount after 7 years is $595 -/
theorem simple_interest_problem :
  ∃ (rate : ℝ),
    (totalAmount 350 rate 2 = 420) →
    (totalAmount 350 rate 7 = 595) := by
  sorry

end simple_interest_problem_l1885_188573


namespace school_classrooms_l1885_188536

/-- Given a school with a total number of students and a fixed number of students per classroom,
    calculate the number of classrooms. -/
def number_of_classrooms (total_students : ℕ) (students_per_classroom : ℕ) : ℕ :=
  total_students / students_per_classroom

/-- Theorem stating that in a school with 120 students and 5 students per classroom,
    there are 24 classrooms. -/
theorem school_classrooms :
  number_of_classrooms 120 5 = 24 := by
  sorry

#eval number_of_classrooms 120 5

end school_classrooms_l1885_188536


namespace geometric_sequence_formula_l1885_188566

def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  isGeometricSequence a →
  a 1 = 2 →
  (∀ n, a (n + 2)^2 + 4 * a n^2 = 4 * a (n + 1)^2) →
  ∀ n, a n = 2^((n + 1) / 2) :=
by sorry

end geometric_sequence_formula_l1885_188566


namespace average_of_remaining_numbers_l1885_188554

theorem average_of_remaining_numbers 
  (total : ℝ) 
  (group1 : ℝ) 
  (group2 : ℝ) 
  (h1 : total = 6 * 3.95) 
  (h2 : group1 = 2 * 3.8) 
  (h3 : group2 = 2 * 3.85) : 
  (total - group1 - group2) / 2 = 4.2 := by
sorry

end average_of_remaining_numbers_l1885_188554


namespace car_catchup_l1885_188556

/-- The time (in hours) it takes for the second car to catch up with the first car -/
def catchup_time : ℝ :=
  1.5

/-- The speed of the first car in km/h -/
def speed_first : ℝ :=
  60

/-- The speed of the second car in km/h -/
def speed_second : ℝ :=
  80

/-- The head start of the first car in hours -/
def head_start : ℝ :=
  0.5

theorem car_catchup :
  speed_second * catchup_time = speed_first * (catchup_time + head_start) :=
sorry

end car_catchup_l1885_188556


namespace circle_rolling_in_triangle_l1885_188590

/-- The distance traveled by the center of a circle rolling inside a right triangle -/
theorem circle_rolling_in_triangle (a b c : ℝ) (r : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) (h_radius : r = 2) : 
  (a - 2*r) + (b - 2*r) + (c - 2*r) = 24 := by
sorry


end circle_rolling_in_triangle_l1885_188590


namespace quiz_team_payment_l1885_188562

/-- The set of possible values for B in the number 2B5 -/
def possible_B : Set Nat :=
  {b | b ∈ Finset.range 10 ∧ (200 + 10 * b + 5) % 15 = 0}

/-- The theorem stating that the only possible values for B are 2, 5, and 8 -/
theorem quiz_team_payment :
  possible_B = {2, 5, 8} := by sorry

end quiz_team_payment_l1885_188562


namespace det_trig_matrix_l1885_188597

open Real Matrix

theorem det_trig_matrix (a b : ℝ) : 
  det !![1, sin (a + b), cos a; 
         sin (a + b), 1, sin b; 
         cos a, sin b, 1] = 
  2 * sin (a + b) * sin b * cos a + sin (a + b)^2 - 1 := by
  sorry

end det_trig_matrix_l1885_188597


namespace find_k_value_l1885_188584

/-- Represents a point on a line segment --/
structure SegmentPoint where
  position : ℝ
  min : ℝ
  max : ℝ
  h : min ≤ position ∧ position ≤ max

/-- The theorem stating the value of k --/
theorem find_k_value (AB CD : ℝ × ℝ) (h_AB : AB = (0, 6)) (h_CD : CD = (0, 9)) :
  ∃ k : ℝ, 
    (∀ (P : SegmentPoint) (Q : SegmentPoint), 
      P.min = 0 ∧ P.max = 6 ∧ Q.min = 0 ∧ Q.max = 9 →
      P.position = 3 * k → P.position + Q.position = 12 * k) →
    k = 2 := by
  sorry

end find_k_value_l1885_188584


namespace affine_preserves_ratio_l1885_188547

/-- An affine transformation in a vector space -/
noncomputable def AffineTransformation (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  V → V

/-- The ratio in which a point divides a line segment -/
def divides_segment_ratio {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (A B C : V) (p q : ℝ) : Prop :=
  q • (C - A) = p • (B - C)

/-- Theorem: Affine transformations preserve segment division ratios -/
theorem affine_preserves_ratio {V : Type*} [AddCommGroup V] [Module ℝ V]
  (L : AffineTransformation V) (A B C A' B' C' : V) (p q : ℝ) :
  L A = A' → L B = B' → L C = C' →
  divides_segment_ratio A B C p q →
  divides_segment_ratio A' B' C' p q :=
by sorry

end affine_preserves_ratio_l1885_188547


namespace employee_payment_l1885_188545

theorem employee_payment (total : ℝ) (a_multiplier : ℝ) (b_payment : ℝ) :
  total = 580 →
  a_multiplier = 1.5 →
  total = b_payment + a_multiplier * b_payment →
  b_payment = 232 := by
sorry

end employee_payment_l1885_188545


namespace intersection_of_lines_l1885_188563

/-- The x-coordinate of the intersection point of two lines -/
def intersection_x (m₁ b₁ a₂ b₂ c₂ : ℚ) : ℚ :=
  (c₂ + 2 * b₁) / (2 * m₁ + a₂)

theorem intersection_of_lines :
  let line1 : ℚ → ℚ := λ x => 3 * x - 24
  let line2 : ℚ → ℚ → Prop := λ x y => 5 * x + 2 * y = 102
  ∃ x y : ℚ, line2 x y ∧ y = line1 x ∧ x = 150 / 11 :=
by sorry

end intersection_of_lines_l1885_188563


namespace thirteen_factorial_mod_seventeen_l1885_188527

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating that 13! ≡ 9 (mod 17) -/
theorem thirteen_factorial_mod_seventeen :
  factorial 13 % 17 = 9 := by
  sorry

end thirteen_factorial_mod_seventeen_l1885_188527


namespace cos_seven_arccos_two_fifths_l1885_188538

theorem cos_seven_arccos_two_fifths (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, abs (Real.cos (7 * Real.arccos (2/5)) - x) < ε ∧ abs (x + 0.2586) < ε :=
sorry

end cos_seven_arccos_two_fifths_l1885_188538


namespace mike_seeds_count_mike_total_seeds_l1885_188575

theorem mike_seeds_count : ℕ → Prop :=
  fun total_seeds =>
    let seeds_left : ℕ := 20
    let seeds_right : ℕ := 2 * seeds_left
    let seeds_joining : ℕ := 30
    let seeds_remaining : ℕ := 30
    total_seeds = seeds_left + seeds_right + seeds_joining + seeds_remaining

theorem mike_total_seeds :
  ∃ (total_seeds : ℕ), mike_seeds_count total_seeds ∧ total_seeds = 120 := by
  sorry

end mike_seeds_count_mike_total_seeds_l1885_188575


namespace lana_total_pages_l1885_188581

/-- Calculate the total number of pages Lana will have after receiving pages from Duane and Alexa -/
theorem lana_total_pages
  (lana_initial : ℕ)
  (duane_pages : ℕ)
  (duane_percentage : ℚ)
  (alexa_pages : ℕ)
  (alexa_percentage : ℚ)
  (h1 : lana_initial = 8)
  (h2 : duane_pages = 42)
  (h3 : duane_percentage = 70 / 100)
  (h4 : alexa_pages = 48)
  (h5 : alexa_percentage = 25 / 100)
  : ℕ := by
  sorry

#check lana_total_pages

end lana_total_pages_l1885_188581


namespace g_increasing_range_l1885_188542

/-- A piecewise function g(x) defined on [0, +∞) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x ≥ m then (1/4) * x^2 else x

/-- The theorem stating the range of m for which g is increasing on [0, +∞) -/
theorem g_increasing_range (m : ℝ) :
  (m > 0) →
  (∀ x y, 0 ≤ x ∧ x < y → g m x ≤ g m y) →
  m ∈ Set.Ici 4 :=
sorry

end g_increasing_range_l1885_188542


namespace jacks_sock_purchase_l1885_188519

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  four_dollar : ℕ

/-- Checks if the given SockPurchase satisfies all conditions --/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.four_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 4 * p.four_dollar = 36 ∧
  p.two_dollar ≥ 1 ∧ p.three_dollar ≥ 1 ∧ p.four_dollar ≥ 1

/-- Theorem stating that the only valid purchase has 10 pairs of $2 socks --/
theorem jacks_sock_purchase :
  ∀ p : SockPurchase, is_valid_purchase p → p.two_dollar = 10 := by
  sorry

end jacks_sock_purchase_l1885_188519


namespace cylinder_height_relationship_l1885_188577

/-- Given two right circular cylinders with identical volumes, where the radius of the second cylinder
    is 20% more than the radius of the first, prove that the height of the first cylinder
    is 44% more than the height of the second cylinder. -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) : 
  r₁ > 0 → h₁ > 0 → r₂ > 0 → h₂ > 0 →
  (π * r₁^2 * h₁ = π * r₂^2 * h₂) →  -- Volumes are equal
  (r₂ = 1.2 * r₁) →                  -- Second radius is 20% more than the first
  (h₁ = 1.44 * h₂) :=                -- First height is 44% more than the second
by sorry

end cylinder_height_relationship_l1885_188577


namespace chocolate_ratio_l1885_188523

/-- Proves that the ratio of chocolates with nuts to chocolates without nuts is 1:1 given the problem conditions. -/
theorem chocolate_ratio (total : ℕ) (eaten_with_nuts : ℚ) (eaten_without_nuts : ℚ) (left : ℕ)
  (h_total : total = 80)
  (h_eaten_with_nuts : eaten_with_nuts = 4/5)
  (h_eaten_without_nuts : eaten_without_nuts = 1/2)
  (h_left : left = 28) :
  ∃ (with_nuts without_nuts : ℕ),
    with_nuts + without_nuts = total ∧
    (1 - eaten_with_nuts) * with_nuts + (1 - eaten_without_nuts) * without_nuts = left ∧
    with_nuts = without_nuts := by
  sorry

#check chocolate_ratio

end chocolate_ratio_l1885_188523


namespace olivia_chocolate_sales_l1885_188539

/-- Calculates the money made from selling chocolate bars --/
def money_made (cost_per_bar : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * cost_per_bar

/-- Proves that Olivia made $9 from selling chocolate bars --/
theorem olivia_chocolate_sales : money_made 3 7 4 = 9 := by
  sorry

end olivia_chocolate_sales_l1885_188539


namespace largest_divisor_of_polynomial_l1885_188558

theorem largest_divisor_of_polynomial (n : ℤ) : 
  ∃ (k : ℕ), k = 120 ∧ (k : ℤ) ∣ (n^5 - 5*n^3 + 4*n) ∧ 
  ∀ (m : ℕ), m > k → ¬((m : ℤ) ∣ (n^5 - 5*n^3 + 4*n)) :=
by sorry

end largest_divisor_of_polynomial_l1885_188558


namespace playground_run_distance_l1885_188511

theorem playground_run_distance (length width laps : ℕ) : 
  length = 55 → width = 35 → laps = 2 → 
  2 * (length + width) * laps = 360 := by
sorry

end playground_run_distance_l1885_188511


namespace base4_division_theorem_l1885_188598

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent. -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

/-- Represents a number in base 4. -/
structure Base4 where
  digits : List Nat
  valid : ∀ d ∈ digits.toFinset, d < 4

/-- The dividend in base 4. -/
def dividend : Base4 := {
  digits := [0, 2, 3, 2, 1]
  valid := by sorry
}

/-- The divisor in base 4. -/
def divisor : Base4 := {
  digits := [2, 1]
  valid := by sorry
}

/-- The quotient in base 4. -/
def quotient : Base4 := {
  digits := [1, 2, 1, 1]
  valid := by sorry
}

/-- Theorem stating that the division of the dividend by the divisor equals the quotient in base 4. -/
theorem base4_division_theorem :
  (base4ToDecimal dividend.digits) / (base4ToDecimal divisor.digits) = base4ToDecimal quotient.digits := by
  sorry

end base4_division_theorem_l1885_188598


namespace bowtie_equation_solution_l1885_188506

-- Define the operation ⊗ (using ⊗ instead of ⭐ as it's more readily available)
noncomputable def bowtie (x y : ℝ) : ℝ := x + Real.sqrt (y + Real.sqrt (y + Real.sqrt y))

-- State the theorem
theorem bowtie_equation_solution (h : ℝ) : bowtie 3 h = 5 → h = 2 := by
  sorry

end bowtie_equation_solution_l1885_188506


namespace largest_root_bound_l1885_188533

theorem largest_root_bound (b₂ b₁ b₀ : ℤ) (h₂ : |b₂| ≤ 3) (h₁ : |b₁| ≤ 3) (h₀ : |b₀| ≤ 3) :
  ∃ r : ℝ, 3.5 < r ∧ r < 4 ∧
  (∀ x : ℝ, x > r → x^3 + (b₂ : ℝ) * x^2 + (b₁ : ℝ) * x + (b₀ : ℝ) ≠ 0) ∧
  (∃ x : ℝ, x ≤ r ∧ x^3 + (b₂ : ℝ) * x^2 + (b₁ : ℝ) * x + (b₀ : ℝ) = 0) :=
by sorry

end largest_root_bound_l1885_188533


namespace isosceles_triangle_leg_length_l1885_188535

/-- An isosceles triangle with perimeter 16 and base 4 has legs of length 6 -/
theorem isosceles_triangle_leg_length :
  ∀ (leg_length : ℝ),
  leg_length > 0 →
  leg_length + leg_length + 4 = 16 →
  leg_length = 6 :=
by sorry

end isosceles_triangle_leg_length_l1885_188535


namespace average_of_remaining_numbers_l1885_188580

theorem average_of_remaining_numbers 
  (total : ℝ) 
  (group1 : ℝ) 
  (group2 : ℝ) 
  (h1 : total = 6 * 3.95) 
  (h2 : group1 = 2 * 4.2) 
  (h3 : group2 = 2 * 3.85) : 
  (total - group1 - group2) / 2 = 3.8 := by
sorry

end average_of_remaining_numbers_l1885_188580


namespace ratio_equality_l1885_188528

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : a / b = 3 / 2 := by
  sorry

end ratio_equality_l1885_188528


namespace lift_cars_and_trucks_l1885_188516

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars to be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks to be lifted -/
def num_trucks : ℕ := 3

/-- The total number of people needed to lift the given number of cars and trucks -/
def total_people : ℕ := num_cars * people_per_car + num_trucks * people_per_truck

theorem lift_cars_and_trucks : total_people = 60 := by
  sorry

end lift_cars_and_trucks_l1885_188516


namespace max_value_fraction_l1885_188525

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ a b, -4 ≤ a ∧ a ≤ -2 ∧ 3 ≤ b ∧ b ≤ 5 → (a + b) / a ≤ (x + y) / x) →
  (x + y) / x = -1/4 :=
sorry

end max_value_fraction_l1885_188525


namespace circle_tangency_l1885_188510

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the given line
def givenLine (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the y-axis
def yAxis (x : ℝ) : Prop := x = 0

-- Define the possible circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y + Real.sqrt 3)^2 = 1
def circle3 (x y : ℝ) : Prop := (x - 2*Real.sqrt 3 - 3)^2 + (y + 2 + Real.sqrt 3)^2 = 21 + 12*Real.sqrt 3
def circle4 (x y : ℝ) : Prop := (x + 2*Real.sqrt 3 + 3)^2 + (y - 2 - Real.sqrt 3)^2 = 21 + 12*Real.sqrt 3

-- Define external tangency
def externallyTangent (c1 c2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define tangency to a line
def tangentToLine (c : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop := sorry

-- Define tangency to y-axis
def tangentToYAxis (c : ℝ → ℝ → Prop) : Prop := sorry

theorem circle_tangency :
  (externallyTangent circle1 givenCircle ∧ tangentToLine circle1 givenLine ∧ tangentToYAxis circle1) ∨
  (externallyTangent circle2 givenCircle ∧ tangentToLine circle2 givenLine ∧ tangentToYAxis circle2) ∨
  (externallyTangent circle3 givenCircle ∧ tangentToLine circle3 givenLine ∧ tangentToYAxis circle3) ∨
  (externallyTangent circle4 givenCircle ∧ tangentToLine circle4 givenLine ∧ tangentToYAxis circle4) :=
by sorry

end circle_tangency_l1885_188510


namespace zero_at_specific_point_l1885_188548

/-- A polynomial of degree 3 in x and y -/
def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*x^2*y + b₈*x*y^2 + b₉*y^3

theorem zero_at_specific_point 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) : 
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 2 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (5/19) (16/19) = 0 := by
  sorry

end zero_at_specific_point_l1885_188548


namespace cosine_sum_special_case_l1885_188537

theorem cosine_sum_special_case (α β : Real) 
  (h1 : α - β = π/3)
  (h2 : Real.tan α - Real.tan β = 3 * Real.sqrt 3) :
  Real.cos (α + β) = -1/6 := by sorry

end cosine_sum_special_case_l1885_188537


namespace min_value_of_exponential_sum_l1885_188518

/-- A line passing through (1,2) with equal x and y intercepts -/
structure Line where
  slope : ℝ
  intercept : ℝ
  passes_through_one_two : 2 = slope * 1 + intercept
  equal_intercepts : intercept = slope * intercept

/-- A point (a,b) on the line -/
structure Point (l : Line) where
  a : ℝ
  b : ℝ
  on_line : b = l.slope * a + l.intercept

/-- The theorem statement -/
theorem min_value_of_exponential_sum (l : Line) (p : Point l) 
  (h_non_zero : l.intercept ≠ 0) :
  (3 : ℝ) ^ p.a + (3 : ℝ) ^ p.b ≥ 6 * Real.sqrt 3 := by
  sorry

end min_value_of_exponential_sum_l1885_188518


namespace qin_jiushao_v3_value_l1885_188578

def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

def v_3 (x : ℝ) : ℝ := ((7*x + 6)*x + 5)*x + 4

theorem qin_jiushao_v3_value : v_3 3 = 262 := by
  sorry

end qin_jiushao_v3_value_l1885_188578


namespace property_holds_iff_one_or_two_l1885_188513

-- Define the property for a given k
def has_property (k : ℕ) : Prop :=
  k ≥ 1 ∧
  ∀ (coloring : ℤ → Fin k),
  ∃ (a : ℕ → ℤ),
    (∀ i < 2023, a i < a (i + 1)) ∧
    (∀ i < 2023, ∃ n : ℕ, a (i + 1) - a i = 2^n) ∧
    (∀ i < 2023, coloring (a i) = coloring (a 0))

-- State the theorem
theorem property_holds_iff_one_or_two :
  ∀ k : ℕ, has_property k ↔ k = 1 ∨ k = 2 := by sorry

end property_holds_iff_one_or_two_l1885_188513


namespace count_integers_with_at_most_three_divisors_cubic_plus_eight_l1885_188540

def has_at_most_three_divisors (x : ℤ) : Prop :=
  (∃ p : ℕ, Prime p ∧ x = p^2) ∨ (∃ p : ℕ, Prime p ∧ x = p) ∨ x = 1

theorem count_integers_with_at_most_three_divisors_cubic_plus_eight :
  ∃! (S : Finset ℤ), ∀ n : ℤ, n ∈ S ↔ has_at_most_three_divisors (n^3 + 8) ∧ Finset.card S = 2 :=
sorry

end count_integers_with_at_most_three_divisors_cubic_plus_eight_l1885_188540


namespace vector_b_value_l1885_188532

theorem vector_b_value (a b : ℝ × ℝ × ℝ) :
  a = (4, 0, -2) →
  a - b = (0, 1, -2) →
  b = (4, -1, 0) := by
sorry

end vector_b_value_l1885_188532


namespace min_value_3a_minus_2ab_l1885_188559

theorem min_value_3a_minus_2ab :
  ∀ a b : ℕ+, a < 8 → b < 8 → (3 * a - 2 * a * b : ℤ) ≥ -77 ∧
  ∃ a₀ b₀ : ℕ+, a₀ < 8 ∧ b₀ < 8 ∧ (3 * a₀ - 2 * a₀ * b₀ : ℤ) = -77 := by
  sorry

end min_value_3a_minus_2ab_l1885_188559


namespace boatman_journey_l1885_188503

/-- Represents the boatman's journey on the river -/
structure RiverJourney where
  v : ℝ  -- Speed of the boat in still water
  v_T : ℝ  -- Speed of the current
  upstream_distance : ℝ  -- Distance traveled upstream
  total_time : ℝ  -- Total time for the round trip

/-- Theorem stating the conditions and results of the boatman's journey -/
theorem boatman_journey (j : RiverJourney) : 
  j.upstream_distance = 12.5 ∧ 
  (3 / (j.v - j.v_T) = 5 / (j.v + j.v_T)) ∧ 
  (j.upstream_distance / (j.v - j.v_T) + j.upstream_distance / (j.v + j.v_T) = j.total_time) ∧ 
  j.total_time = 8 → 
  j.v_T = 5/6 ∧ 
  j.upstream_distance / (j.v - j.v_T) = 5 := by
  sorry

end boatman_journey_l1885_188503


namespace either_false_sufficient_not_necessary_l1885_188591

variable (p q : Prop)

theorem either_false_sufficient_not_necessary :
  (((¬p ∨ ¬q) → ¬p) ∧ ¬(¬p → (¬p ∨ ¬q))) := by sorry

end either_false_sufficient_not_necessary_l1885_188591


namespace lcm_of_primes_l1885_188544

theorem lcm_of_primes : 
  let p₁ : Nat := 1223
  let p₂ : Nat := 1399
  let p₃ : Nat := 2687
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ →
  Nat.lcm p₁ (Nat.lcm p₂ p₃) = 4583641741 :=
by sorry

end lcm_of_primes_l1885_188544


namespace factorial_division_l1885_188531

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end factorial_division_l1885_188531


namespace sum_of_four_repeated_digit_terms_l1885_188522

/-- A function that checks if a natural number consists of repeated digits --/
def is_repeated_digit (n : ℕ) : Prop := sorry

/-- A function that returns the number of digits in a natural number --/
def num_digits (n : ℕ) : ℕ := sorry

theorem sum_of_four_repeated_digit_terms : 
  ∃ (a b c d : ℕ), 
    2017 = a + b + c + d ∧ 
    is_repeated_digit a ∧ 
    is_repeated_digit b ∧ 
    is_repeated_digit c ∧ 
    is_repeated_digit d ∧ 
    num_digits a ≠ num_digits b ∧ 
    num_digits a ≠ num_digits c ∧ 
    num_digits a ≠ num_digits d ∧ 
    num_digits b ≠ num_digits c ∧ 
    num_digits b ≠ num_digits d ∧ 
    num_digits c ≠ num_digits d :=
by sorry

end sum_of_four_repeated_digit_terms_l1885_188522


namespace min_xy_value_l1885_188564

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z : ℝ, x * y ≥ z → z ≥ 64 :=
sorry

end min_xy_value_l1885_188564


namespace cricket_team_age_difference_l1885_188514

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : Nat
  captain_age : Nat
  wicket_keeper_age : Nat
  team_average_age : Nat

/-- The difference between the average age of remaining players and the whole team -/
def age_difference (team : CricketTeam) : Rat :=
  let remaining_members := team.total_members - 2
  let total_age := team.team_average_age * team.total_members
  let remaining_age := total_age - team.captain_age - team.wicket_keeper_age
  let remaining_average := remaining_age / remaining_members
  team.team_average_age - remaining_average

/-- Theorem stating the age difference for a specific cricket team -/
theorem cricket_team_age_difference :
  ∃ (team : CricketTeam),
    team.total_members = 11 ∧
    team.captain_age = 26 ∧
    team.wicket_keeper_age = team.captain_age + 5 ∧
    team.team_average_age = 24 ∧
    age_difference team = 1 := by
  sorry

end cricket_team_age_difference_l1885_188514


namespace existence_of_number_with_four_prime_factors_l1885_188589

theorem existence_of_number_with_four_prime_factors : ∃ N : ℕ,
  (∃ p₁ p₂ p₃ p₄ : ℕ, 
    (Nat.Prime p₁) ∧ (Nat.Prime p₂) ∧ (Nat.Prime p₃) ∧ (Nat.Prime p₄) ∧
    (p₁ ≠ p₂) ∧ (p₁ ≠ p₃) ∧ (p₁ ≠ p₄) ∧ (p₂ ≠ p₃) ∧ (p₂ ≠ p₄) ∧ (p₃ ≠ p₄) ∧
    (1 < p₁) ∧ (p₁ ≤ 100) ∧
    (1 < p₂) ∧ (p₂ ≤ 100) ∧
    (1 < p₃) ∧ (p₃ ≤ 100) ∧
    (1 < p₄) ∧ (p₄ ≤ 100) ∧
    (N = p₁ * p₂ * p₃ * p₄) ∧
    (∀ q : ℕ, Nat.Prime q → q ∣ N → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄))) ∧
  N = 210 :=
by
  sorry


end existence_of_number_with_four_prime_factors_l1885_188589


namespace gcd_linear_combination_l1885_188585

theorem gcd_linear_combination (a b d : ℕ) :
  d = Nat.gcd a b →
  d = Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) := by
  sorry

end gcd_linear_combination_l1885_188585


namespace commonMaterialChoices_eq_120_l1885_188582

/-- The number of ways to choose r items from n items without regard to order -/
def binomial (n r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items out of n items -/
def permutation (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways two students can choose 2 materials each from 6 materials, 
    with exactly 1 material in common -/
def commonMaterialChoices : ℕ :=
  binomial 6 1 * permutation 5 2

theorem commonMaterialChoices_eq_120 : commonMaterialChoices = 120 := by
  sorry

end commonMaterialChoices_eq_120_l1885_188582


namespace village_population_l1885_188557

theorem village_population (initial_population : ℝ) : 
  (initial_population * 1.2 * 0.8 = 9600) → initial_population = 10000 := by
  sorry

end village_population_l1885_188557


namespace divisible_by_41_l1885_188592

theorem divisible_by_41 (n : ℕ) : ∃ k : ℤ, 5 * 7^(2*(n+1)) + 2^(3*n) = 41 * k := by
  sorry

end divisible_by_41_l1885_188592


namespace second_day_sales_l1885_188546

/-- Represents the ticket sales for a choral performance --/
structure TicketSales where
  senior_price : ℝ
  student_price : ℝ
  day1_senior : ℕ
  day1_student : ℕ
  day1_total : ℝ
  day2_senior : ℕ
  day2_student : ℕ

/-- The theorem to prove --/
theorem second_day_sales (ts : TicketSales)
  (h1 : ts.student_price = 9)
  (h2 : ts.day1_senior * ts.senior_price + ts.day1_student * ts.student_price = ts.day1_total)
  (h3 : ts.day1_senior = 4)
  (h4 : ts.day1_student = 3)
  (h5 : ts.day1_total = 79)
  (h6 : ts.day2_senior = 12)
  (h7 : ts.day2_student = 10) :
  ts.day2_senior * ts.senior_price + ts.day2_student * ts.student_price = 246 := by
  sorry


end second_day_sales_l1885_188546


namespace difference_of_squares_253_247_l1885_188526

theorem difference_of_squares_253_247 : 253^2 - 247^2 = 3000 := by
  sorry

end difference_of_squares_253_247_l1885_188526


namespace cricket_average_l1885_188550

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 10 → 
  next_runs = 81 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase → 
  current_average = 37 := by
sorry

end cricket_average_l1885_188550


namespace bus_equations_l1885_188524

/-- Given m buses and n people, if 40 people per bus leaves 10 people without a seat
    and 43 people per bus leaves 1 person without a seat, then two equations hold. -/
theorem bus_equations (m n : ℕ) 
    (h1 : 40 * m + 10 = n) 
    (h2 : 43 * m + 1 = n) : 
    (40 * m + 10 = 43 * m + 1) ∧ ((n - 10) / 40 = (n - 1) / 43) := by
  sorry

end bus_equations_l1885_188524


namespace cone_volume_not_equal_base_height_product_l1885_188593

/-- The volume of a cone is not equal to the product of its base area and height. -/
theorem cone_volume_not_equal_base_height_product (S h : ℝ) (S_pos : S > 0) (h_pos : h > 0) :
  ∃ V : ℝ, V = (1/3) * S * h ∧ V ≠ S * h := by
  sorry

end cone_volume_not_equal_base_height_product_l1885_188593


namespace M_intersect_N_empty_l1885_188595

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ (N.prod Set.univ) = ∅ := by sorry

end M_intersect_N_empty_l1885_188595


namespace fraction_inverse_addition_l1885_188505

theorem fraction_inverse_addition (a b : ℚ) (h : a ≠ b) :
  let c := -(a + b)
  (a + c) / (b + c) = b / a :=
by sorry

end fraction_inverse_addition_l1885_188505


namespace integral_sqrt_rational_equals_pi_sixth_l1885_188551

theorem integral_sqrt_rational_equals_pi_sixth :
  ∫ x in (2 : ℝ)..3, Real.sqrt ((3 - 2*x) / (2*x - 7)) = π / 6 := by
  sorry

end integral_sqrt_rational_equals_pi_sixth_l1885_188551


namespace perpendicular_vectors_l1885_188520

/-- Given vectors a and b in ℝ², prove that k = -1 makes k*a - b perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (-3, 1)) :
  ∃ k : ℝ, k = -1 ∧ (k • a - b) • a = 0 := by
  sorry

end perpendicular_vectors_l1885_188520


namespace quadratic_sum_l1885_188560

/-- Given a quadratic function f(x) = 10x^2 + 100x + 1000, 
    proves that when written in the form a(x+b)^2 + c, 
    the sum of a, b, and c is 765 -/
theorem quadratic_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
    (∀ x, 10 * x^2 + 100 * x + 1000 = a * (x + b)^2 + c) ∧
    a + b + c = 765 :=
by sorry

end quadratic_sum_l1885_188560


namespace norbs_age_l1885_188576

def guesses : List Nat := [26, 30, 35, 39, 42, 43, 45, 47, 49, 52, 53]

def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

def countLowerGuesses (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (· < age)).length

def countOffByOne (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (fun g => g = age - 1 || g = age + 1)).length

theorem norbs_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    isPrime age ∧
    countLowerGuesses age guesses ≥ guesses.length / 2 ∧
    countOffByOne age guesses = 3 ∧
    age = 47 :=
  by sorry

end norbs_age_l1885_188576


namespace least_positive_integer_with_remainders_l1885_188572

theorem least_positive_integer_with_remainders : ∃ M : ℕ, 
  (M > 0) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (∀ n : ℕ, n > 0 ∧ 
    n % 6 = 5 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 ∧
    n % 10 = 9 ∧
    n % 11 = 10 → n ≥ M) ∧
  M = 27719 :=
by sorry

end least_positive_integer_with_remainders_l1885_188572


namespace gcf_factorial_seven_eight_l1885_188502

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcf_factorial_seven_eight : 
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end gcf_factorial_seven_eight_l1885_188502


namespace extreme_value_and_range_l1885_188517

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Theorem statement
theorem extreme_value_and_range :
  (f 2 = -8) ∧
  (∀ x ∈ Set.Icc (-3) 3, -8 ≤ f x ∧ f x ≤ 24) ∧
  (∃ x ∈ Set.Icc (-3) 3, f x = -8) ∧
  (∃ x ∈ Set.Icc (-3) 3, f x = 24) :=
by sorry

end extreme_value_and_range_l1885_188517


namespace incorrect_reasoning_l1885_188534

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_on_plane : Line → Plane → Prop)

-- Define the theorem
theorem incorrect_reasoning 
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ l α A, ¬(line_on_plane l α) → on_line A l → ¬(on_plane A α)) :=
sorry

end incorrect_reasoning_l1885_188534


namespace erasers_per_box_l1885_188553

theorem erasers_per_box 
  (num_boxes : ℕ) 
  (price_per_eraser : ℚ) 
  (total_money : ℚ) 
  (h1 : num_boxes = 48)
  (h2 : price_per_eraser = 3/4)
  (h3 : total_money = 864) : 
  (total_money / price_per_eraser) / num_boxes = 24 := by
sorry

end erasers_per_box_l1885_188553


namespace parallel_vectors_x_value_l1885_188596

/-- Given two parallel vectors a and b in R², if a = (4, 2) and b = (x, 3), then x = 6 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (4, 2)) 
  (h2 : b = (x, 3)) 
  (h3 : ∃ (k : ℝ), b = k • a) : 
  x = 6 := by
  sorry

end parallel_vectors_x_value_l1885_188596


namespace plywood_cut_perimeter_difference_l1885_188588

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its possible cuts --/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Generates all possible ways to cut the plywood into congruent pieces --/
def possible_cuts (p : Plywood) : List Rectangle :=
  sorry -- Implementation details omitted

/-- Finds the maximum perimeter from a list of rectangles --/
def max_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

/-- Finds the minimum perimeter from a list of rectangles --/
def min_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

theorem plywood_cut_perimeter_difference :
  let p : Plywood := { length := 9, width := 6, num_pieces := 6 }
  let cuts := possible_cuts p
  max_perimeter cuts - min_perimeter cuts = 10 := by
  sorry

end plywood_cut_perimeter_difference_l1885_188588


namespace right_triangle_hypotenuse_l1885_188571

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 24 → b = 32 → c^2 = a^2 + b^2 → c = 40 := by
  sorry

end right_triangle_hypotenuse_l1885_188571


namespace complex_equation_solution_l1885_188574

theorem complex_equation_solution (a : ℝ) : (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end complex_equation_solution_l1885_188574


namespace square_region_perimeter_l1885_188567

theorem square_region_perimeter (area : ℝ) (num_squares : ℕ) (perimeter : ℝ) :
  area = 392 →
  num_squares = 8 →
  (area / num_squares).sqrt * (2 * num_squares + 2) = perimeter →
  perimeter = 70 := by
  sorry

end square_region_perimeter_l1885_188567


namespace factorization_equality_l1885_188555

theorem factorization_equality (x : ℝ) : (x + 2) * x - (x + 2) = (x + 2) * (x - 1) := by
  sorry

end factorization_equality_l1885_188555


namespace average_value_equals_combination_l1885_188549

def average_value (n : ℕ) : ℚ :=
  (n + 1) * n * (n - 1) / 6

theorem average_value_equals_combination (n : ℕ) (h : n > 0) :
  average_value n = Nat.choose (n + 1) 3 := by sorry

end average_value_equals_combination_l1885_188549


namespace collinear_vectors_x_value_l1885_188594

/-- Two vectors in R² are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2) ∨ w = (k * v.1, k * v.2)

/-- The problem statement -/
theorem collinear_vectors_x_value :
  ∀ (x : ℝ), collinear (x, 1) (4, x) → x = 2 ∨ x = -2 :=
by sorry

end collinear_vectors_x_value_l1885_188594


namespace textile_firm_profit_decrease_l1885_188541

/-- Represents the decrease in profit due to loom breakdowns -/
def decrease_in_profit (
  total_looms : ℕ)
  (monthly_sales : ℝ)
  (monthly_manufacturing_expenses : ℝ)
  (monthly_establishment_charges : ℝ)
  (breakdown_days : List ℕ)
  (repair_cost_per_loom : ℝ)
  : ℝ :=
  sorry

/-- Theorem stating the decrease in profit for the given scenario -/
theorem textile_firm_profit_decrease :
  decrease_in_profit 70 1000000 150000 75000 [10, 5, 15] 2000 = 20285.70 :=
sorry

end textile_firm_profit_decrease_l1885_188541


namespace sixth_term_of_geometric_sequence_l1885_188586

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Main theorem -/
theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 5 = 16)
  (h_fourth : a 4 = 8) :
  a 6 = 32 := by
sorry

end sixth_term_of_geometric_sequence_l1885_188586


namespace solve_r_system_l1885_188579

theorem solve_r_system (r s : ℚ) : 
  (r - 60) / 3 = (5 - 3 * r) / 4 → 
  s + 2 * r = 10 → 
  r = 255 / 13 := by
sorry

end solve_r_system_l1885_188579


namespace smallest_prime_divisor_of_sum_l1885_188570

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  Nat.minFac (3^15 + 11^21) = 2 := by
  sorry

end smallest_prime_divisor_of_sum_l1885_188570


namespace quadratic_root_implies_coefficient_l1885_188521

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I) ^ 2 + a * (1 - Complex.I) + 2 = 0 →
  a = -2 :=
by sorry

end quadratic_root_implies_coefficient_l1885_188521


namespace white_balls_count_white_balls_count_specific_l1885_188500

/-- The number of white balls in a bag, given the total number of balls,
    the number of balls of each color (except white), and the probability
    of choosing a ball that is neither red nor purple. -/
theorem white_balls_count (total green yellow red purple : ℕ)
                          (prob_not_red_purple : ℚ) : ℕ :=
  let total_balls : ℕ := total
  let green_balls : ℕ := green
  let yellow_balls : ℕ := yellow
  let red_balls : ℕ := red
  let purple_balls : ℕ := purple
  let prob_not_red_or_purple : ℚ := prob_not_red_purple
  24

/-- The number of white balls is 24 given the specific conditions. -/
theorem white_balls_count_specific : white_balls_count 60 18 2 15 3 (7/10) = 24 := by
  sorry

end white_balls_count_white_balls_count_specific_l1885_188500


namespace percentage_to_fraction_l1885_188599

theorem percentage_to_fraction (p : ℚ) : p = 166 / 1000 → p = 83 / 50 := by
  sorry

end percentage_to_fraction_l1885_188599


namespace tan_double_angle_l1885_188509

theorem tan_double_angle (x : ℝ) (h : Real.tan x = 2) : Real.tan (2 * x) = -4/3 := by
  sorry

end tan_double_angle_l1885_188509


namespace largest_two_digit_prime_factor_of_150_choose_75_l1885_188504

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

theorem largest_two_digit_prime_factor_of_150_choose_75 :
  ∃ (p : ℕ), p = 47 ∧ 
  Prime p ∧ 
  10 ≤ p ∧ p < 100 ∧
  p ∣ binomial_coefficient 150 75 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial_coefficient 150 75 → q ≤ p :=
by sorry

end largest_two_digit_prime_factor_of_150_choose_75_l1885_188504


namespace rock_collection_contest_l1885_188583

theorem rock_collection_contest (sydney_start conner_start : ℕ) 
  (sydney_day1 conner_day2 conner_day3 : ℕ) : 
  sydney_start = 837 → 
  conner_start = 723 → 
  sydney_day1 = 4 → 
  conner_day2 = 123 → 
  conner_day3 = 27 → 
  ∃ (conner_day1 : ℕ), 
    conner_start + conner_day1 + conner_day2 + conner_day3 
    = sydney_start + sydney_day1 + 2 * conner_day1 
    ∧ conner_day1 / sydney_day1 = 8 := by
  sorry

end rock_collection_contest_l1885_188583
