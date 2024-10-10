import Mathlib

namespace k_range_theorem_l2643_264319

-- Define the function f(x) = (e^x / x) + x^2 - 2x
noncomputable def f (x : ℝ) : ℝ := (Real.exp x / x) + x^2 - 2*x

theorem k_range_theorem (k : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x / Real.exp x < 1 / (k + 2*x - x^2)) → 
  k ∈ Set.Icc 0 (Real.exp 1 - 1) :=
sorry

end k_range_theorem_l2643_264319


namespace product_of_diff_of_squares_l2643_264343

-- Define the property of being a difference of two squares
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = x^2 - y^2 ∧ x > y

-- Theorem statement
theorem product_of_diff_of_squares (a b c d : ℕ) 
  (ha : is_diff_of_squares a)
  (hb : is_diff_of_squares b)
  (hc : is_diff_of_squares c)
  (hd : is_diff_of_squares d)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0) :
  is_diff_of_squares (a * b * c * d) :=
by
  sorry

end product_of_diff_of_squares_l2643_264343


namespace union_complement_equality_l2643_264388

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 3, 4}

theorem union_complement_equality : A ∪ (U \ B) = {1, 2, 3} := by
  sorry

end union_complement_equality_l2643_264388


namespace simplify_and_rationalize_l2643_264338

theorem simplify_and_rationalize (x : ℝ) (hx : x > 0) :
  (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt x / Real.sqrt 12) * (Real.sqrt 6 / Real.sqrt 8) = 
  Real.sqrt (1260 * x) / 168 := by
  sorry

end simplify_and_rationalize_l2643_264338


namespace international_shipping_charge_l2643_264322

/-- The additional charge per letter for international shipping -/
def additional_charge (standard_postage : ℚ) (total_letters : ℕ) (international_letters : ℕ) (total_cost : ℚ) : ℚ :=
  ((total_cost - (standard_postage * total_letters)) / international_letters) * 100

theorem international_shipping_charge :
  let standard_postage : ℚ := 108 / 100  -- $1.08 in decimal form
  let total_letters : ℕ := 4
  let international_letters : ℕ := 2
  let total_cost : ℚ := 460 / 100  -- $4.60 in decimal form
  additional_charge standard_postage total_letters international_letters total_cost = 14 := by
  sorry

#eval additional_charge (108/100) 4 2 (460/100)

end international_shipping_charge_l2643_264322


namespace box_volume_l2643_264327

/-- Given a rectangular box with dimensions L, W, and H satisfying certain conditions,
    prove that its volume is 5184. -/
theorem box_volume (L W H : ℝ) (h1 : H * W = 288) (h2 : L * W = 1.5 * 288) 
    (h3 : L * H = 0.5 * (L * W)) : L * W * H = 5184 := by
  sorry

end box_volume_l2643_264327


namespace base_8_properties_l2643_264359

-- Define the base 10 number
def base_10_num : ℕ := 9257

-- Define the base 8 representation as a list of digits
def base_8_rep : List ℕ := [2, 2, 0, 5, 1]

-- Theorem stating the properties we want to prove
theorem base_8_properties :
  -- The base 8 representation is correct
  (List.foldl (λ acc d => acc * 8 + d) 0 base_8_rep = base_10_num) ∧
  -- The product of the digits is 0
  (List.foldl (· * ·) 1 base_8_rep = 0) ∧
  -- The sum of the digits is 10
  (List.sum base_8_rep = 10) := by
  sorry

end base_8_properties_l2643_264359


namespace tetrahedrons_from_triangular_prism_l2643_264324

/-- The number of tetrahedrons that can be formed from a regular triangular prism -/
def tetrahedrons_from_prism (n : ℕ) : ℕ :=
  Nat.choose n 4 - 3

/-- Theorem stating that the number of tetrahedrons formed from a regular triangular prism with 6 vertices is 12 -/
theorem tetrahedrons_from_triangular_prism : 
  tetrahedrons_from_prism 6 = 12 := by
  sorry

end tetrahedrons_from_triangular_prism_l2643_264324


namespace puppies_theorem_l2643_264304

/-- The number of puppies Alyssa gave away -/
def alyssa_gave : ℕ := 20

/-- The number of puppies Alyssa kept -/
def alyssa_kept : ℕ := 8

/-- The number of puppies Bella gave away -/
def bella_gave : ℕ := 10

/-- The number of puppies Bella kept -/
def bella_kept : ℕ := 6

/-- The total number of puppies Alyssa and Bella had to start with -/
def total_puppies : ℕ := alyssa_gave + alyssa_kept + bella_gave + bella_kept

theorem puppies_theorem : total_puppies = 44 := by
  sorry

end puppies_theorem_l2643_264304


namespace max_value_of_f_l2643_264350

def f (x : ℝ) : ℝ := -2 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 8 := by
  sorry

end max_value_of_f_l2643_264350


namespace pigeonhole_on_permutation_sums_l2643_264332

theorem pigeonhole_on_permutation_sums (n : ℕ) (h_even : Even n) (h_pos : 0 < n)
  (A B : Fin n → Fin n) (h_A : Function.Bijective A) (h_B : Function.Bijective B) :
  ∃ (i j : Fin n), i ≠ j ∧ (A i + B i) % n = (A j + B j) % n := by
sorry

end pigeonhole_on_permutation_sums_l2643_264332


namespace first_term_is_one_l2643_264344

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1
  sum_formula : ∀ n : ℕ, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)

/-- Theorem: For a geometric sequence with specific sum values, the first term is 1 -/
theorem first_term_is_one (seq : GeometricSequence) (m : ℕ) 
    (h1 : seq.S (m - 2) = 1)
    (h2 : seq.S m = 3)
    (h3 : seq.S (m + 2) = 5) :
  seq.a 1 = 1 := by
  sorry

end first_term_is_one_l2643_264344


namespace cylinder_volume_relation_l2643_264379

/-- Given two cylinders A and B, where A's radius is r and height is h,
    and B's radius is h and height is r, prove that if A's volume is
    three times B's volume, then A's volume is 9πh^3. -/
theorem cylinder_volume_relation (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 3 * (π * h^2 * r) →
  π * r^2 * h = 9 * π * h^3 := by
sorry

end cylinder_volume_relation_l2643_264379


namespace circle_and_symmetry_line_l2643_264326

-- Define the center of the circle
def C : ℝ × ℝ := (-1, 0)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 3 = 0

-- Define the symmetry line
def symmetry_line (m x y : ℝ) : Prop := m * x + y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Theorem statement
theorem circle_and_symmetry_line :
  ∃ (r : ℝ), r > 0 ∧
  (∀ x y : ℝ, (x - C.1)^2 + (y - C.2)^2 = r^2 →
    (∃ x' y' : ℝ, tangent_line x' y' ∧ (x' - C.1)^2 + (y' - C.2)^2 = r^2)) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    ∃ m : ℝ, symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂ ∧ x₁ ≠ x₂) →
  (∀ x y : ℝ, circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = 4) ∧
  (∃! m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂ ∧ x₁ ≠ x₂ ∧ m = 1) :=
sorry

end circle_and_symmetry_line_l2643_264326


namespace no_prime_satisfies_equation_l2643_264312

theorem no_prime_satisfies_equation : 
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ 
  (2 * p^3 + 0 * p^2 + 3 * p + 4) + 
  (4 * p^2 + 0 * p + 5) + 
  (2 * p^2 + 1 * p + 7) + 
  (1 * p^2 + 5 * p + 0) + 
  4 = 
  (3 * p^2 + 0 * p + 2) + 
  (5 * p^2 + 2 * p + 0) + 
  (4 * p^2 + 3 * p + 1) :=
sorry

end no_prime_satisfies_equation_l2643_264312


namespace inequality_solution_set_l2643_264381

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := -x^2 - 4*x + 5

-- Define the solution set
def solution_set : Set ℝ := {x | -5 < x ∧ x < 1}

-- Theorem stating that the solution set of the inequality is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set := by
  sorry

end inequality_solution_set_l2643_264381


namespace line_y_intercept_l2643_264356

/-- A line with slope 6 and x-intercept (8, 0) has y-intercept (0, -48) -/
theorem line_y_intercept (f : ℝ → ℝ) (h_slope : ∀ x y, f y - f x = 6 * (y - x)) 
  (h_x_intercept : f 8 = 0) : f 0 = -48 := by
  sorry

end line_y_intercept_l2643_264356


namespace quadratic_equation_from_means_l2643_264301

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 12 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_equation_from_means_l2643_264301


namespace greatest_three_digit_multiple_of_17_l2643_264347

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, n < 1000 → n ≥ 100 → n % 17 = 0 → n ≤ 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l2643_264347


namespace soccer_teams_count_l2643_264302

theorem soccer_teams_count : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n * (n - 1) = 20 ∧ 
    n = 5 := by
  sorry

end soccer_teams_count_l2643_264302


namespace total_cost_is_24_l2643_264393

/-- The number of index fingers on a person's hands. -/
def num_index_fingers : ℕ := 2

/-- The cost of one gold ring in dollars. -/
def cost_per_ring : ℕ := 12

/-- The total cost of buying gold rings for all index fingers. -/
def total_cost : ℕ := num_index_fingers * cost_per_ring

/-- Theorem stating that the total cost for buying gold rings for all index fingers is $24. -/
theorem total_cost_is_24 : total_cost = 24 := by sorry

end total_cost_is_24_l2643_264393


namespace polygon_sides_count_l2643_264373

theorem polygon_sides_count (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end polygon_sides_count_l2643_264373


namespace thirty_percent_of_hundred_l2643_264360

theorem thirty_percent_of_hundred : ∃ x : ℝ, 30 = 0.30 * x ∧ x = 100 := by
  sorry

end thirty_percent_of_hundred_l2643_264360


namespace vector_operation_l2643_264362

/-- Given two vectors AB and AC in R², prove that 2AB - AC equals (5,7) -/
theorem vector_operation (AB AC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : AC = (-1, -1)) : 
  (2 : ℝ) • AB - AC = (5, 7) := by
  sorry

end vector_operation_l2643_264362


namespace minimum_students_l2643_264396

theorem minimum_students (b g : ℕ) : 
  (3 * b = 4 * g) →  -- Equal number of boys and girls passed
  (∃ k : ℕ, b = 4 * k ∧ g = 3 * k) →  -- b and g are integers
  (b + g ≥ 7) ∧ (∀ m n : ℕ, (3 * m = 4 * n) → (m + n < 7 → m = 0 ∨ n = 0)) :=
by sorry

end minimum_students_l2643_264396


namespace bus_ride_cost_l2643_264391

theorem bus_ride_cost (train_cost bus_cost : ℝ) : 
  train_cost = bus_cost + 6.35 →
  train_cost + bus_cost = 9.85 →
  bus_cost = 1.75 := by
sorry

end bus_ride_cost_l2643_264391


namespace henrys_action_figures_l2643_264346

def action_figure_problem (total_needed : ℕ) (cost_per_figure : ℕ) (money_needed : ℕ) : Prop :=
  let figures_to_buy : ℕ := money_needed / cost_per_figure
  let initial_figures : ℕ := total_needed - figures_to_buy
  initial_figures = 3

theorem henrys_action_figures :
  action_figure_problem 8 6 30 := by
  sorry

end henrys_action_figures_l2643_264346


namespace circle_non_intersect_l2643_264378

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line
def line (k : ℤ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for non-intersection
def non_intersect (k : ℤ) (l : ℝ) : Prop :=
  ∀ x y : ℝ, line k x y →
    (x - 1)^2 + y^2 > (1 + l)^2

-- Main theorem
theorem circle_non_intersect :
  ∃ k : ℤ, ∀ l : ℝ, l > 0 → non_intersect k l ∧ k = -1 :=
sorry

end circle_non_intersect_l2643_264378


namespace simon_change_theorem_l2643_264309

/-- Calculates the discounted price for a flower purchase -/
def discountedPrice (quantity : ℕ) (price : ℚ) (discount : ℚ) : ℚ :=
  (quantity : ℚ) * price * (1 - discount)

/-- Calculates the total price after tax -/
def totalPriceAfterTax (prices : List ℚ) (taxRate : ℚ) : ℚ :=
  let subtotal := prices.sum
  subtotal * (1 + taxRate)

theorem simon_change_theorem (pansyPrice petuniasPrice lilyPrice orchidPrice : ℚ)
    (pansyDiscount hydrangeaDiscount petuniaDiscount lilyDiscount orchidDiscount : ℚ)
    (hydrangeaPrice : ℚ) (taxRate : ℚ) :
    pansyPrice = 2.5 →
    petuniasPrice = 1 →
    lilyPrice = 5 →
    orchidPrice = 7.5 →
    hydrangeaPrice = 12.5 →
    pansyDiscount = 0.1 →
    hydrangeaDiscount = 0.15 →
    petuniaDiscount = 0.2 →
    lilyDiscount = 0.12 →
    orchidDiscount = 0.08 →
    taxRate = 0.06 →
    let pansies := discountedPrice 5 pansyPrice pansyDiscount
    let hydrangea := discountedPrice 1 hydrangeaPrice hydrangeaDiscount
    let petunias := discountedPrice 5 petuniasPrice petuniaDiscount
    let lilies := discountedPrice 3 lilyPrice lilyDiscount
    let orchids := discountedPrice 2 orchidPrice orchidDiscount
    let total := totalPriceAfterTax [pansies, hydrangea, petunias, lilies, orchids] taxRate
    100 - total = 43.95 := by sorry

end simon_change_theorem_l2643_264309


namespace prime_divides_or_coprime_l2643_264348

theorem prime_divides_or_coprime (p n : ℕ) (hp : Prime p) :
  p ∣ n ∨ Nat.gcd p n = 1 := by sorry

end prime_divides_or_coprime_l2643_264348


namespace max_product_at_endpoints_l2643_264331

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  m : ℤ
  n : ℤ
  f : ℝ → ℝ := λ x ↦ 10 * x^2 + m * x + n

/-- The property that a function has two distinct real roots in (1, 3) -/
def has_two_distinct_roots_in_open_interval (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, 1 < x ∧ x < y ∧ y < 3 ∧ f x = 0 ∧ f y = 0

/-- The theorem statement -/
theorem max_product_at_endpoints (qf : QuadraticFunction) 
  (h : has_two_distinct_roots_in_open_interval qf.f) :
  (qf.f 1) * (qf.f 3) ≤ 99 := by
  sorry

end max_product_at_endpoints_l2643_264331


namespace log_relation_l2643_264315

theorem log_relation (a b : ℝ) : 
  a = Real.log 1024 / Real.log 16 → b = Real.log 32 / Real.log 2 → a = (1/2) * b := by
  sorry

end log_relation_l2643_264315


namespace complex_fraction_sum_l2643_264374

theorem complex_fraction_sum (z : ℂ) (a b : ℝ) : 
  z = (1 + Complex.I) / (1 - Complex.I) → 
  z = Complex.mk a b → 
  a + b = 1 := by sorry

end complex_fraction_sum_l2643_264374


namespace complex_multiplication_l2643_264398

theorem complex_multiplication (P F G : ℂ) : 
  P = 3 + 4*Complex.I ∧ 
  F = 2*Complex.I ∧ 
  G = 3 - 4*Complex.I → 
  (P + F) * G = 21 + 6*Complex.I :=
by sorry

end complex_multiplication_l2643_264398


namespace inequality_preservation_l2643_264358

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end inequality_preservation_l2643_264358


namespace xy_squared_l2643_264397

theorem xy_squared (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + x + y = 7) : 
  x^2 * y^2 = 49/36 := by
sorry

end xy_squared_l2643_264397


namespace midpoint_coordinate_ratio_range_l2643_264329

/-- Given two parallel lines and a point between them, prove the ratio of its coordinates is within a specific range. -/
theorem midpoint_coordinate_ratio_range 
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (x₀ : ℝ) (y₀ : ℝ)
  (hP : P.1 + 2 * P.2 - 1 = 0)
  (hQ : Q.1 + 2 * Q.2 + 3 = 0)
  (hM : (x₀, y₀) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))
  (h_ineq : y₀ > x₀ + 2)
  : -1/2 < y₀ / x₀ ∧ y₀ / x₀ < -1/5 :=
sorry

end midpoint_coordinate_ratio_range_l2643_264329


namespace sarah_initial_followers_l2643_264321

/-- Represents the number of followers gained by Sarah in a week -/
structure WeeklyGain where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the data for a student's social media followers -/
structure StudentData where
  school_size : ℕ
  initial_followers : ℕ
  weekly_gain : WeeklyGain

theorem sarah_initial_followers (susy sarah : StudentData) 
  (h1 : susy.school_size = 800)
  (h2 : sarah.school_size = 300)
  (h3 : susy.initial_followers = 100)
  (h4 : sarah.weekly_gain.first = 90)
  (h5 : sarah.weekly_gain.second = sarah.weekly_gain.first / 3)
  (h6 : sarah.weekly_gain.third = sarah.weekly_gain.second / 3)
  (h7 : max (susy.initial_followers + susy.weekly_gain.first + susy.weekly_gain.second + susy.weekly_gain.third)
            (sarah.initial_followers + sarah.weekly_gain.first + sarah.weekly_gain.second + sarah.weekly_gain.third) = 180) :
  sarah.initial_followers = 50 := by
  sorry


end sarah_initial_followers_l2643_264321


namespace smallest_z_value_l2643_264371

/-- Given four positive integers w, x, y, and z such that:
    1. w³, x³, y³, and z³ are distinct, consecutive positive perfect cubes
    2. There's a gap of 1 between w, x, and y
    3. There's a gap of 3 between y and z
    4. w³ + x³ + y³ = z³
    Then the smallest possible value of z is 9. -/
theorem smallest_z_value (w x y z : ℕ+) 
  (h1 : w.val + 1 = x.val)
  (h2 : x.val + 1 = y.val)
  (h3 : y.val + 3 = z.val)
  (h4 : w.val^3 + x.val^3 + y.val^3 = z.val^3)
  (h5 : w.val^3 < x.val^3 ∧ x.val^3 < y.val^3 ∧ y.val^3 < z.val^3) :
  z.val ≥ 9 := by
  sorry

#check smallest_z_value

end smallest_z_value_l2643_264371


namespace complete_square_sum_l2643_264352

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + b)^2 = c
    where b and c are integers, prove that b + c = 1 -/
theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + b)^2 = c) → b + c = 1 := by
  sorry

end complete_square_sum_l2643_264352


namespace kubosdivision_l2643_264317

theorem kubosdivision (k m : ℕ) (hk : k > 0) (hm : m > 0) (hkm : k > m) 
  (hdiv : (k^3 - m^3) ∣ (k * m * (k^2 - m^2))) : (k - m)^3 > 3 * k * m := by
  sorry

end kubosdivision_l2643_264317


namespace fabric_difference_total_fabric_l2643_264363

/-- The amount of fabric used to make a coat, in meters -/
def coat_fabric : ℝ := 1.55

/-- The amount of fabric used to make a pair of pants, in meters -/
def pants_fabric : ℝ := 1.05

/-- The difference in fabric usage between a coat and pants is 0.5 meters -/
theorem fabric_difference : coat_fabric - pants_fabric = 0.5 := by sorry

/-- The total fabric needed for a coat and pants is 2.6 meters -/
theorem total_fabric : coat_fabric + pants_fabric = 2.6 := by sorry

end fabric_difference_total_fabric_l2643_264363


namespace total_meal_cost_l2643_264386

def meal_cost (num_people : ℕ) (cost_per_person : ℚ) (tax_rate : ℚ) (tip_percentages : List ℚ) : ℚ :=
  let base_cost := num_people * cost_per_person
  let tax := base_cost * tax_rate
  let cost_with_tax := base_cost + tax
  let avg_tip_percentage := (tip_percentages.sum + 1) / tip_percentages.length
  let tip := cost_with_tax * avg_tip_percentage
  cost_with_tax + tip

theorem total_meal_cost :
  let num_people : ℕ := 5
  let cost_per_person : ℚ := 90
  let tax_rate : ℚ := 825 / 10000
  let tip_percentages : List ℚ := [15/100, 18/100, 20/100, 22/100, 25/100]
  meal_cost num_people cost_per_person tax_rate tip_percentages = 97426 / 100 := by
  sorry

end total_meal_cost_l2643_264386


namespace julia_monday_kids_l2643_264369

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 10

/-- The additional number of kids Julia played with on Monday compared to Tuesday -/
def additional_monday_kids : ℕ := 8

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := tuesday_kids + additional_monday_kids

theorem julia_monday_kids : monday_kids = 18 := by
  sorry

end julia_monday_kids_l2643_264369


namespace some_number_equation_l2643_264341

theorem some_number_equation : ∃ n : ℤ, (69842^2 - n^2) / (69842 - n) = 100000 ∧ n = 30158 := by
  sorry

end some_number_equation_l2643_264341


namespace rectangle_area_ratio_l2643_264380

theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  (a * b) / (c * d) = 4 / 9 := by
  sorry

end rectangle_area_ratio_l2643_264380


namespace parabola_intersection_value_l2643_264328

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - x - 1

-- Define the theorem
theorem parabola_intersection_value :
  ∀ m : ℝ, parabola m = 0 → -2 * m^2 + 2 * m + 2023 = 2021 := by
  sorry

end parabola_intersection_value_l2643_264328


namespace expansion_properties_l2643_264372

/-- Given that for some natural number n, the expansion of (x^(1/6) + x^(-1/6))^n has
    binomial coefficients of the 2nd, 3rd, and 4th terms forming an arithmetic sequence,
    prove that n = 7 and there is no constant term in the expansion. -/
theorem expansion_properties (n : ℕ) 
  (h : 2 * (n.choose 2) = n.choose 1 + n.choose 3) : 
  (n = 7) ∧ (∀ k : ℕ, (7 : ℚ) - 2 * k ≠ 0) := by
  sorry

end expansion_properties_l2643_264372


namespace expected_vote_percentage_a_l2643_264306

/-- Percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 70

/-- Percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 100 - democrat_percentage

/-- Percentage of Democrats expected to vote for candidate A -/
def democrat_vote_a : ℝ := 80

/-- Percentage of Republicans expected to vote for candidate A -/
def republican_vote_a : ℝ := 30

/-- Theorem stating the percentage of registered voters expected to vote for candidate A -/
theorem expected_vote_percentage_a : 
  (democrat_percentage / 100 * democrat_vote_a + 
   republican_percentage / 100 * republican_vote_a) = 65 := by
  sorry

end expected_vote_percentage_a_l2643_264306


namespace hurricane_damage_conversion_l2643_264349

def damage_in_euros : ℝ := 45000000
def exchange_rate : ℝ := 0.9

theorem hurricane_damage_conversion :
  damage_in_euros * (1 / exchange_rate) = 49995000 := by
  sorry

end hurricane_damage_conversion_l2643_264349


namespace number_calculation_l2643_264377

theorem number_calculation (n : ℝ) : 0.125 * 0.20 * 0.40 * 0.75 * n = 148.5 → n = 23760 := by
  sorry

end number_calculation_l2643_264377


namespace sphere_packing_ratio_l2643_264395

/-- Configuration of four spheres with two radii -/
structure SpherePacking where
  r : ℝ  -- radius of smaller spheres
  R : ℝ  -- radius of larger spheres
  r_positive : r > 0
  R_positive : R > 0
  touch_plane : True  -- represents that all spheres touch the plane
  touch_others : True  -- represents that each sphere touches three others

/-- Theorem stating the ratio of radii in the sphere packing configuration -/
theorem sphere_packing_ratio (config : SpherePacking) : config.R / config.r = 1 + Real.sqrt 3 := by
  sorry

end sphere_packing_ratio_l2643_264395


namespace half_triangles_isosceles_l2643_264361

/-- A function that returns the number of pairwise non-congruent triangles
    that can be formed from N points on a circle. -/
def totalTriangles (N : ℕ) : ℕ := N * (N - 1) * (N - 2) / 6

/-- A function that returns the number of isosceles triangles
    that can be formed from N points on a circle. -/
def isoscelesTriangles (N : ℕ) : ℕ := N * (N - 2) / 3

/-- The theorem stating that exactly half of the triangles are isosceles
    if and only if N is 10 or 11, for N > 2. -/
theorem half_triangles_isosceles (N : ℕ) (h : N > 2) :
  2 * isoscelesTriangles N = totalTriangles N ↔ N = 10 ∨ N = 11 :=
sorry

end half_triangles_isosceles_l2643_264361


namespace lcm_gcd_1365_910_l2643_264339

theorem lcm_gcd_1365_910 :
  (Nat.lcm 1365 910 = 2730) ∧ (Nat.gcd 1365 910 = 455) := by
sorry

end lcm_gcd_1365_910_l2643_264339


namespace ways_without_first_grade_ways_with_all_grades_l2643_264382

/-- Represents the number of products of each grade -/
structure ProductCounts where
  total : Nat
  firstGrade : Nat
  secondGrade : Nat
  thirdGrade : Nat

/-- The given product counts in the problem -/
def givenCounts : ProductCounts :=
  { total := 8
  , firstGrade := 3
  , secondGrade := 3
  , thirdGrade := 2 }

/-- Number of products to draw -/
def drawCount : Nat := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem for the first question -/
theorem ways_without_first_grade (counts : ProductCounts) :
  choose (counts.secondGrade + counts.thirdGrade) drawCount = 5 :=
sorry

/-- Theorem for the second question -/
theorem ways_with_all_grades (counts : ProductCounts) :
  choose counts.firstGrade 2 * choose counts.secondGrade 1 * choose counts.thirdGrade 1 +
  choose counts.firstGrade 1 * choose counts.secondGrade 2 * choose counts.thirdGrade 1 +
  choose counts.firstGrade 1 * choose counts.secondGrade 1 * choose counts.thirdGrade 2 = 45 :=
sorry

end ways_without_first_grade_ways_with_all_grades_l2643_264382


namespace percent_product_theorem_l2643_264390

theorem percent_product_theorem :
  let p1 : ℝ := 15
  let p2 : ℝ := 20
  let p3 : ℝ := 25
  (p1 / 100) * (p2 / 100) * (p3 / 100) * 100 = 0.75
  := by sorry

end percent_product_theorem_l2643_264390


namespace cricket_game_target_runs_l2643_264345

/-- Calculates the target number of runs in a cricket game -/
def targetRuns (totalOvers runRateFirst10 runRateRemaining : ℝ) : ℝ :=
  10 * runRateFirst10 + (totalOvers - 10) * runRateRemaining

/-- Theorem stating the target number of runs in the given cricket game -/
theorem cricket_game_target_runs :
  targetRuns 50 6.2 5.5 = 282 := by
  sorry

end cricket_game_target_runs_l2643_264345


namespace floor_sqrt_150_l2643_264389

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by
  sorry

end floor_sqrt_150_l2643_264389


namespace sqrt_equation_solution_l2643_264303

theorem sqrt_equation_solution : 
  ∃ x : ℚ, (Real.sqrt (8 * x) / Real.sqrt (2 * (x - 2)) = 3) ∧ (x = 18/5) := by
  sorry

end sqrt_equation_solution_l2643_264303


namespace inequality_proof_l2643_264351

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) :
  x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end inequality_proof_l2643_264351


namespace dasha_flag_count_l2643_264337

/-- Represents the number of flags held by each first-grader -/
structure FlagCount where
  tata : ℕ
  yasha : ℕ
  vera : ℕ
  maxim : ℕ
  dasha : ℕ

/-- The problem statement -/
def flag_problem (fc : FlagCount) : Prop :=
  fc.tata + fc.yasha + fc.vera + fc.maxim + fc.dasha = 37 ∧
  fc.yasha + fc.vera + fc.maxim + fc.dasha = 32 ∧
  fc.vera + fc.maxim + fc.dasha = 20 ∧
  fc.maxim + fc.dasha = 14 ∧
  fc.dasha = 8

/-- The theorem to prove -/
theorem dasha_flag_count :
  ∀ fc : FlagCount, flag_problem fc → fc.dasha = 8 := by
  sorry

end dasha_flag_count_l2643_264337


namespace apple_orchard_problem_l2643_264307

theorem apple_orchard_problem (total : ℝ) (fuji : ℝ) (gala : ℝ) : 
  (0.1 * total = total - fuji - gala) →
  (fuji + 0.1 * total = 238) →
  (fuji = 0.75 * total) →
  (gala = 42) := by
sorry

end apple_orchard_problem_l2643_264307


namespace course_selection_count_l2643_264366

def num_courses_A : ℕ := 3
def num_courses_B : ℕ := 4
def total_courses_selected : ℕ := 3

theorem course_selection_count : 
  (Nat.choose num_courses_A 2 * Nat.choose num_courses_B 1) + 
  (Nat.choose num_courses_A 1 * Nat.choose num_courses_B 2) = 30 := by
  sorry

end course_selection_count_l2643_264366


namespace first_ring_at_start_of_day_l2643_264316

-- Define the clock's properties
def ring_interval : ℕ := 3
def rings_per_day : ℕ := 8
def hours_per_day : ℕ := 24

-- Theorem to prove
theorem first_ring_at_start_of_day :
  ring_interval * rings_per_day = hours_per_day →
  ring_interval ∣ hours_per_day →
  (0 : ℕ) = hours_per_day % ring_interval :=
by
  sorry

#check first_ring_at_start_of_day

end first_ring_at_start_of_day_l2643_264316


namespace trigonometric_properties_l2643_264367

theorem trigonometric_properties :
  (¬ ∃ α : ℝ, Real.sin α + Real.cos α = 3/2) ∧
  (∀ x : ℝ, Real.cos (7 * Real.pi / 2 - 3 * x) = -Real.cos (7 * Real.pi / 2 + 3 * x)) ∧
  (∀ x : ℝ, 4 * Real.sin (2 * (-9 * Real.pi / 8 + x) + 5 * Real.pi / 4) = 
            4 * Real.sin (2 * (-9 * Real.pi / 8 - x) + 5 * Real.pi / 4)) ∧
  (∃ x : ℝ, Real.sin (2 * x - Real.pi / 4) ≠ Real.sin (2 * (x - Real.pi / 8))) :=
by sorry

end trigonometric_properties_l2643_264367


namespace complex_arithmetic_equality_l2643_264336

theorem complex_arithmetic_equality : (5 - 5*Complex.I) + (-2 - Complex.I) - (3 + 4*Complex.I) = -10*Complex.I := by
  sorry

end complex_arithmetic_equality_l2643_264336


namespace product_inequality_l2643_264311

theorem product_inequality (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) : 
  |((a - b) * (b - c) * (c - d) * (d - a))| ≤ (a * b * c * d) / 4 := by
sorry

end product_inequality_l2643_264311


namespace tan_squared_f_equals_neg_cos_double_l2643_264325

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)))

-- State the theorem
theorem tan_squared_f_equals_neg_cos_double (t : ℝ) 
  (h1 : 0 ≤ t) (h2 : t ≤ π/2) : f (Real.tan t ^ 2) = -Real.cos (2 * t) :=
by
  sorry


end tan_squared_f_equals_neg_cos_double_l2643_264325


namespace reciprocal_of_repeating_decimal_l2643_264394

def repeating_decimal : ℚ := 36 / 99

theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = 11 / 4 := by
  sorry

end reciprocal_of_repeating_decimal_l2643_264394


namespace two_distinct_roots_l2643_264375

/-- The custom operation ⊗ for real numbers -/
def otimes (a b : ℝ) : ℝ := b^2 - a*b

/-- Theorem stating that the equation (k-3) ⊗ x = k-1 has two distinct real roots for any real k -/
theorem two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ otimes (k-3) x₁ = k-1 ∧ otimes (k-3) x₂ = k-1 :=
sorry

end two_distinct_roots_l2643_264375


namespace same_color_probability_l2643_264300

/-- The number of red candies in the box -/
def num_red : ℕ := 12

/-- The number of green candies in the box -/
def num_green : ℕ := 8

/-- The number of candies Alice and Bob each pick -/
def num_pick : ℕ := 3

/-- The probability that Alice and Bob pick the same number of candies of each color -/
def same_color_prob : ℚ := 231 / 1060

theorem same_color_probability :
  let total := num_red + num_green
  same_color_prob = (Nat.choose num_red num_pick * Nat.choose (num_red - num_pick) num_pick) / 
    (Nat.choose total num_pick * Nat.choose (total - num_pick) num_pick) +
    (Nat.choose num_red 2 * Nat.choose num_green 1 * Nat.choose (num_red - 2) 2 * 
    Nat.choose (num_green - 1) 1) / (Nat.choose total num_pick * Nat.choose (total - num_pick) num_pick) := by
  sorry

end same_color_probability_l2643_264300


namespace salt_solution_mixture_l2643_264383

/-- Represents the amount of pure water added in liters -/
def W : ℝ := 1

/-- The initial volume of salt solution in liters -/
def initial_volume : ℝ := 1

/-- The initial concentration of salt in the solution -/
def initial_concentration : ℝ := 0.40

/-- The final concentration of salt in the mixture -/
def final_concentration : ℝ := 0.20

theorem salt_solution_mixture :
  initial_volume * initial_concentration = 
  (initial_volume + W) * final_concentration := by sorry

end salt_solution_mixture_l2643_264383


namespace relationship_between_a_and_b_l2643_264342

theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : (1003 : ℝ) ^ a + (1004 : ℝ) ^ b = (2006 : ℝ) ^ b)
  (h2 : (997 : ℝ) ^ a + (1009 : ℝ) ^ b = (2007 : ℝ) ^ a) : 
  a < b := by
sorry

end relationship_between_a_and_b_l2643_264342


namespace parabola_sum_zero_l2643_264340

/-- A parabola passing through two specific points has a + b + c = 0 --/
theorem parabola_sum_zero (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (a * (-2)^2 + b * (-2) + c = -3) →
  (a * 2^2 + b * 2 + c = 5) →
  a + b + c = 0 := by
sorry

end parabola_sum_zero_l2643_264340


namespace jellybean_problem_l2643_264353

theorem jellybean_problem (initial_count : ℕ) : 
  (((initial_count : ℚ) * (3/4)^3).floor = 27) → initial_count = 64 := by
sorry

end jellybean_problem_l2643_264353


namespace vector_BC_l2643_264330

/-- Given points A(0,1), B(3,2), and vector AC(-4,-3), prove that vector BC is (-7,-4) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (0, 1) → 
  B = (3, 2) → 
  C - A = (-4, -3) → 
  C - B = (-7, -4) := by
sorry

end vector_BC_l2643_264330


namespace log_sum_equation_l2643_264323

theorem log_sum_equation (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  Real.log p + Real.log q = Real.log (p + q + p * q) → p = -q :=
by sorry

end log_sum_equation_l2643_264323


namespace reciprocal_fraction_l2643_264314

theorem reciprocal_fraction (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 1) 
  (h3 : (2/3) * x = y * (1/x)) : y = 2/3 := by
  sorry

end reciprocal_fraction_l2643_264314


namespace number_of_factors_34650_l2643_264357

def number_to_factor := 34650

theorem number_of_factors_34650 :
  (Finset.filter (· ∣ number_to_factor) (Finset.range (number_to_factor + 1))).card = 72 := by
  sorry

end number_of_factors_34650_l2643_264357


namespace binomial_8_3_l2643_264384

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end binomial_8_3_l2643_264384


namespace constant_function_proof_l2643_264364

theorem constant_function_proof (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ 1 + f x * f (y * z)) : 
  ∀ x : ℝ, f x = 1 := by
  sorry

end constant_function_proof_l2643_264364


namespace monochromatic_triangle_k6_exists_no_monochromatic_triangle_k5_l2643_264318

/-- A type representing the two colors used in the graph coloring -/
inductive Color
| Red
| Blue

/-- A type representing a complete graph with n vertices -/
def CompleteGraph (n : ℕ) := Fin n → Fin n → Color

/-- A predicate that checks if a triangle is monochromatic in a given graph -/
def HasMonochromaticTriangle (g : CompleteGraph n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g i j = g j k ∧ g j k = g i k

theorem monochromatic_triangle_k6 :
  ∀ (g : CompleteGraph 6), HasMonochromaticTriangle g :=
sorry

theorem exists_no_monochromatic_triangle_k5 :
  ∃ (g : CompleteGraph 5), ¬HasMonochromaticTriangle g :=
sorry

end monochromatic_triangle_k6_exists_no_monochromatic_triangle_k5_l2643_264318


namespace sequence_property_l2643_264355

-- Define the sequence a_n
def a (n : ℕ) : ℚ := n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n)

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := (n^2 + n) / 2

-- Theorem statement
theorem sequence_property (n k : ℕ) (h1 : k > 2) :
  (∀ m : ℕ, S m = (m^2 + m) / 2) →
  (2 * b (n + 2) = b n + b (n + k)) →
  (k ≠ 4 ∧ k ≠ 10) ∧ (k = 6 ∨ k = 8) :=
by sorry

end sequence_property_l2643_264355


namespace timothy_land_cost_l2643_264387

/-- Represents the cost breakdown of Timothy's farm --/
structure FarmCosts where
  land_acres : ℕ
  house_cost : ℕ
  cow_count : ℕ
  cow_cost : ℕ
  chicken_count : ℕ
  chicken_cost : ℕ
  solar_install_hours : ℕ
  solar_install_rate : ℕ
  solar_equipment_cost : ℕ
  total_cost : ℕ

/-- Calculates the cost per acre of land given the farm costs --/
def land_cost_per_acre (costs : FarmCosts) : ℕ :=
  (costs.total_cost - 
   (costs.house_cost + 
    costs.cow_count * costs.cow_cost + 
    costs.chicken_count * costs.chicken_cost + 
    costs.solar_install_hours * costs.solar_install_rate + 
    costs.solar_equipment_cost)) / costs.land_acres

/-- Theorem stating that the cost per acre of Timothy's land is $20 --/
theorem timothy_land_cost (costs : FarmCosts) 
  (h1 : costs.land_acres = 30)
  (h2 : costs.house_cost = 120000)
  (h3 : costs.cow_count = 20)
  (h4 : costs.cow_cost = 1000)
  (h5 : costs.chicken_count = 100)
  (h6 : costs.chicken_cost = 5)
  (h7 : costs.solar_install_hours = 6)
  (h8 : costs.solar_install_rate = 100)
  (h9 : costs.solar_equipment_cost = 6000)
  (h10 : costs.total_cost = 147700) :
  land_cost_per_acre costs = 20 := by
  sorry


end timothy_land_cost_l2643_264387


namespace participants_in_both_competitions_l2643_264365

theorem participants_in_both_competitions
  (total : ℕ)
  (chinese : ℕ)
  (math : ℕ)
  (neither : ℕ)
  (h1 : total = 50)
  (h2 : chinese = 30)
  (h3 : math = 38)
  (h4 : neither = 2) :
  chinese + math - (total - neither) = 20 :=
by
  sorry

end participants_in_both_competitions_l2643_264365


namespace no_integer_solution_l2643_264385

theorem no_integer_solution : ∀ x y : ℤ, 5 * x^2 - 4 * y^2 ≠ 2017 := by
  sorry

end no_integer_solution_l2643_264385


namespace stating_lunch_potatoes_count_l2643_264399

/-- Represents the number of potatoes used for different purposes -/
structure PotatoUsage where
  total : ℕ
  dinner : ℕ
  lunch : ℕ

/-- 
Theorem stating that given a total of 7 potatoes and 2 used for dinner,
the number of potatoes used for lunch must be 5.
-/
theorem lunch_potatoes_count (usage : PotatoUsage) 
    (h1 : usage.total = 7)
    (h2 : usage.dinner = 2)
    (h3 : usage.total = usage.lunch + usage.dinner) : 
  usage.lunch = 5 := by
  sorry

end stating_lunch_potatoes_count_l2643_264399


namespace equation_solution_l2643_264305

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end equation_solution_l2643_264305


namespace no_solution_for_gcd_equation_l2643_264368

theorem no_solution_for_gcd_equation :
  ¬ ∃ (a b c : ℕ+), 
    Nat.gcd (a.val^2) (b.val^2) + 
    Nat.gcd a.val (Nat.gcd b.val c.val) + 
    Nat.gcd b.val (Nat.gcd a.val c.val) + 
    Nat.gcd c.val (Nat.gcd a.val b.val) = 199 := by
  sorry

end no_solution_for_gcd_equation_l2643_264368


namespace remaining_volleyballs_l2643_264376

/-- Given an initial number of volleyballs and a number of volleyballs lent out,
    calculate the number of volleyballs remaining. -/
def volleyballs_remaining (initial : ℕ) (lent_out : ℕ) : ℕ :=
  initial - lent_out

/-- Theorem stating that given 9 initial volleyballs and 5 lent out,
    the number of volleyballs remaining is 4. -/
theorem remaining_volleyballs :
  volleyballs_remaining 9 5 = 4 := by
  sorry

end remaining_volleyballs_l2643_264376


namespace coordinate_system_change_l2643_264310

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The relative position of two points -/
def relativePosition (p q : Point) : Point :=
  ⟨p.x - q.x, p.y - q.y⟩

theorem coordinate_system_change (A B : Point) :
  relativePosition A B = ⟨2, 5⟩ → relativePosition B A = ⟨-2, -5⟩ := by
  sorry

end coordinate_system_change_l2643_264310


namespace sum_integers_50_to_70_l2643_264354

theorem sum_integers_50_to_70 (x y : ℕ) : 
  (x = (50 + 70) * (70 - 50 + 1) / 2) →  -- Sum of integers from 50 to 70
  (y = ((70 - 50) / 2 + 1)) →            -- Number of even integers from 50 to 70
  (x + y = 1271) → 
  (x = 1260) := by sorry

end sum_integers_50_to_70_l2643_264354


namespace max_knight_moves_5x6_l2643_264333

/-- Represents a chess board --/
structure ChessBoard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a knight's move type --/
inductive MoveType
  | Normal
  | Short

/-- Represents a sequence of knight moves --/
def MoveSequence := List MoveType

/-- Checks if a move sequence is valid (alternating between Normal and Short, starting with Normal) --/
def isValidMoveSequence : MoveSequence → Bool
  | [] => true
  | [MoveType.Normal] => true
  | (MoveType.Normal :: MoveType.Short :: rest) => isValidMoveSequence rest
  | _ => false

/-- The maximum number of moves a knight can make on the given board --/
def maxKnightMoves (board : ChessBoard) (seq : MoveSequence) : Nat :=
  seq.length

/-- The main theorem to prove --/
theorem max_knight_moves_5x6 :
  ∀ (seq : MoveSequence),
    isValidMoveSequence seq →
    maxKnightMoves ⟨5, 6⟩ seq ≤ 24 :=
by sorry

end max_knight_moves_5x6_l2643_264333


namespace probability_not_pulling_prize_l2643_264335

/-- Given odds of 5:8 for pulling a prize, the probability of not pulling the prize is 8/13 -/
theorem probability_not_pulling_prize (favorable_outcomes unfavorable_outcomes : ℕ) 
  (h_odds : favorable_outcomes = 5 ∧ unfavorable_outcomes = 8) :
  (unfavorable_outcomes : ℚ) / (favorable_outcomes + unfavorable_outcomes) = 8 / 13 := by
  sorry


end probability_not_pulling_prize_l2643_264335


namespace square_plot_area_l2643_264320

/-- Proves that a square plot with given fencing costs has an area of 36 square feet -/
theorem square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) :
  cost_per_foot = 58 →
  total_cost = 1392 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    4 * side_length * cost_per_foot = total_cost ∧
    side_length ^ 2 = 36 := by
  sorry

#check square_plot_area

end square_plot_area_l2643_264320


namespace quadratic_equation_roots_ratio_l2643_264392

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end quadratic_equation_roots_ratio_l2643_264392


namespace opposites_imply_x_equals_one_l2643_264370

theorem opposites_imply_x_equals_one : 
  ∀ x : ℝ, (-2 * x) = -(3 * x - 1) → x = 1 := by
  sorry

end opposites_imply_x_equals_one_l2643_264370


namespace geometric_mean_minimum_l2643_264308

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : 3 = Real.sqrt (9^a * 27^b)) :
  (3/a + 2/b) ≥ 12 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 = Real.sqrt (9^a₀ * 27^b₀) ∧ (3/a₀ + 2/b₀) = 12 := by
  sorry

end geometric_mean_minimum_l2643_264308


namespace card_purchase_cost_l2643_264313

/-- Calculates the total cost of cards purchased, including discounts and tax --/
def totalCost (typeA_price typeB_price typeC_price typeD_price : ℚ)
               (typeA_count typeB_count typeC_count typeD_count : ℕ)
               (discount_AB discount_CD : ℚ)
               (min_count_AB min_count_CD : ℕ)
               (tax_rate : ℚ) : ℚ :=
  let subtotal := typeA_price * typeA_count + typeB_price * typeB_count +
                  typeC_price * typeC_count + typeD_price * typeD_count
  let discount_amount := 
    (if typeA_count ≥ min_count_AB ∧ typeB_count ≥ min_count_AB then
      discount_AB * (typeA_price * typeA_count + typeB_price * typeB_count)
    else 0) +
    (if typeC_count ≥ min_count_CD ∧ typeD_count ≥ min_count_CD then
      discount_CD * (typeC_price * typeC_count + typeD_price * typeD_count)
    else 0)
  let discounted_total := subtotal - discount_amount
  let tax := tax_rate * discounted_total
  discounted_total + tax

/-- The total cost of cards is $60.82 given the specified conditions --/
theorem card_purchase_cost : 
  totalCost 1.25 1.50 2.25 2.50  -- Card prices
            6 4 10 12            -- Number of cards purchased
            0.1 0.15             -- Discount rates
            5 8                  -- Minimum count for discounts
            0.06                 -- Tax rate
  = 60.82 := by
  sorry

end card_purchase_cost_l2643_264313


namespace jason_bought_four_dozens_l2643_264334

/-- The number of cupcakes Jason gives to each cousin -/
def cupcakes_per_cousin : ℕ := 3

/-- The number of cousins Jason has -/
def number_of_cousins : ℕ := 16

/-- The number of cupcakes in a dozen -/
def cupcakes_per_dozen : ℕ := 12

/-- Theorem: Jason bought 4 dozens of cupcakes -/
theorem jason_bought_four_dozens :
  (cupcakes_per_cousin * number_of_cousins) / cupcakes_per_dozen = 4 := by
  sorry

end jason_bought_four_dozens_l2643_264334
