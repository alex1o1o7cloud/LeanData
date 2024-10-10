import Mathlib

namespace sqrt_equation_solution_l161_16139

theorem sqrt_equation_solution : ∃ x : ℝ, x = 2401 / 100 ∧ Real.sqrt x + Real.sqrt (x + 2) = 10 := by
  sorry

end sqrt_equation_solution_l161_16139


namespace fractional_decomposition_sum_l161_16173

theorem fractional_decomposition_sum (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end fractional_decomposition_sum_l161_16173


namespace max_draws_until_white_l161_16150

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  white : Nat

/-- Represents the process of drawing balls from the bag -/
def drawUntilWhite (bag : BagContents) : Nat :=
  sorry

/-- Theorem stating the maximum number of draws needed -/
theorem max_draws_until_white (bag : BagContents) 
  (h1 : bag.red = 6) 
  (h2 : bag.white = 5) : 
  drawUntilWhite bag ≤ 7 :=
sorry

end max_draws_until_white_l161_16150


namespace min_disks_to_cover_l161_16128

/-- Represents a disk in 2D space -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a covering of a disk by smaller disks -/
def DiskCovering (large : Disk) (small : List Disk) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - large.center.1)^2 + (p.2 - large.center.2)^2 ≤ large.radius^2 →
    ∃ d ∈ small, (p.1 - d.center.1)^2 + (p.2 - d.center.2)^2 ≤ d.radius^2

/-- The theorem stating that 7 is the minimum number of smaller disks needed -/
theorem min_disks_to_cover (large : Disk) (small : List Disk) :
  large.radius = 1 →
  (∀ d ∈ small, d.radius = 1/2) →
  DiskCovering large small →
  small.length ≥ 7 :=
sorry

end min_disks_to_cover_l161_16128


namespace textbook_distribution_is_four_l161_16145

/-- The number of ways to distribute 8 identical textbooks between the classroom and students,
    given that at least 2 books must be in the classroom and at least 3 books must be with students. -/
def textbook_distribution : ℕ :=
  let total_books : ℕ := 8
  let min_classroom : ℕ := 2
  let min_students : ℕ := 3
  let valid_distributions := List.range (total_books + 1)
    |>.filter (λ classroom_books => 
      classroom_books ≥ min_classroom ∧ 
      (total_books - classroom_books) ≥ min_students)
  valid_distributions.length

/-- Proof that the number of valid distributions is 4 -/
theorem textbook_distribution_is_four : textbook_distribution = 4 := by
  sorry

end textbook_distribution_is_four_l161_16145


namespace base_conversion_1729_l161_16165

theorem base_conversion_1729 :
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = 1729 := by
  sorry

#eval 2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0

end base_conversion_1729_l161_16165


namespace wall_length_calculation_l161_16182

/-- Given a square mirror and a rectangular wall, if the mirror's area is half the wall's area,
    prove that the wall's length is approximately 27 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 24 →
  wall_width = 42 →
  (mirror_side * mirror_side) * 2 = wall_width * (27 : ℝ) := by
  sorry

end wall_length_calculation_l161_16182


namespace cube_volume_from_doubled_cuboid_edges_l161_16121

theorem cube_volume_from_doubled_cuboid_edges (l w h : ℝ) : 
  l * w * h = 36 → (2 * l) * (2 * w) * (2 * h) = 288 := by sorry

end cube_volume_from_doubled_cuboid_edges_l161_16121


namespace youngest_child_age_l161_16195

def is_valid_age (x : ℕ) : Prop :=
  Nat.Prime x ∧
  Nat.Prime (x + 2) ∧
  Nat.Prime (x + 6) ∧
  Nat.Prime (x + 8) ∧
  Nat.Prime (x + 12) ∧
  Nat.Prime (x + 14)

theorem youngest_child_age :
  ∃ (x : ℕ), is_valid_age x ∧ ∀ (y : ℕ), y < x → ¬is_valid_age y :=
by sorry

end youngest_child_age_l161_16195


namespace gcd_8_factorial_10_factorial_l161_16111

theorem gcd_8_factorial_10_factorial :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_8_factorial_10_factorial_l161_16111


namespace waiter_customers_l161_16136

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  num_tables = 6 →
  women_per_table = 3 →
  men_per_table = 5 →
  num_tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end waiter_customers_l161_16136


namespace percentage_error_calculation_l161_16117

theorem percentage_error_calculation : 
  let correct_operation (x : ℝ) := 3 * x
  let incorrect_operation (x : ℝ) := x / 5
  let error (x : ℝ) := correct_operation x - incorrect_operation x
  let percentage_error (x : ℝ) := (error x / correct_operation x) * 100
  ∀ x : ℝ, x ≠ 0 → percentage_error x = (14 / 15) * 100 := by
sorry

end percentage_error_calculation_l161_16117


namespace asterisk_value_l161_16118

theorem asterisk_value : ∃ x : ℚ, (x / 21) * (42 / 84) = 1 ∧ x = 21 := by
  sorry

end asterisk_value_l161_16118


namespace range_of_power_function_l161_16196

theorem range_of_power_function (m : ℝ) (h : m > 0) :
  Set.range (fun x : ℝ => x ^ m) ∩ Set.Ioo 0 1 = Set.Ioo 0 1 := by
  sorry

end range_of_power_function_l161_16196


namespace suraya_vs_mia_l161_16119

/-- The number of apples picked by each person -/
structure ApplePickers where
  kayla : ℕ
  caleb : ℕ
  suraya : ℕ
  mia : ℕ

/-- The conditions of the apple-picking scenario -/
def apple_picking_conditions (a : ApplePickers) : Prop :=
  a.kayla = 20 ∧
  a.caleb = a.kayla / 2 - 5 ∧
  a.suraya = 3 * a.caleb ∧
  a.mia = 2 * a.caleb

/-- The theorem stating that Suraya picked 5 more apples than Mia -/
theorem suraya_vs_mia (a : ApplePickers) 
  (h : apple_picking_conditions a) : a.suraya = a.mia + 5 := by
  sorry


end suraya_vs_mia_l161_16119


namespace inequality_solution_set_l161_16192

theorem inequality_solution_set (x : ℝ) : -x + 1 > 7*x - 3 ↔ x < 1/2 := by
  sorry

end inequality_solution_set_l161_16192


namespace quadratic_equation_root_l161_16170

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x - 4 = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3 := by
  sorry

end quadratic_equation_root_l161_16170


namespace S_is_infinite_l161_16135

-- Define the set of points satisfying the conditions
def S : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 10}

-- Theorem stating that the set S is infinite
theorem S_is_infinite : Set.Infinite S := by
  sorry

end S_is_infinite_l161_16135


namespace constant_difference_of_equal_derivatives_l161_16160

theorem constant_difference_of_equal_derivatives 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, deriv f x = deriv g x) : 
  ∃ C, ∀ x, f x - g x = C :=
sorry

end constant_difference_of_equal_derivatives_l161_16160


namespace expression_value_l161_16112

theorem expression_value :
  let x : ℚ := 2
  let y : ℚ := 3
  let z : ℚ := 4
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 := by
  sorry

end expression_value_l161_16112


namespace unique_cube_difference_nineteen_l161_16142

theorem unique_cube_difference_nineteen :
  ∀ x y : ℕ, x^3 - y^3 = 19 → x = 3 ∧ y = 2 := by
  sorry

end unique_cube_difference_nineteen_l161_16142


namespace ivan_payment_l161_16105

/-- The total amount paid for discounted Uno Giant Family Cards -/
def total_paid (original_price discount quantity : ℕ) : ℕ :=
  (original_price - discount) * quantity

/-- Theorem: Ivan paid $100 for 10 Uno Giant Family Cards with a $2 discount each -/
theorem ivan_payment :
  let original_price : ℕ := 12
  let discount : ℕ := 2
  let quantity : ℕ := 10
  total_paid original_price discount quantity = 100 := by
sorry

end ivan_payment_l161_16105


namespace sin_n_equals_cos_522_l161_16110

theorem sin_n_equals_cos_522 :
  ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (522 * π / 180) :=
by
  use -72
  sorry

end sin_n_equals_cos_522_l161_16110


namespace root_product_equality_l161_16147

-- Define the quadratic equations
def quadratic1 (x p c : ℝ) : ℝ := x^2 + p*x + c
def quadratic2 (x q c : ℝ) : ℝ := x^2 + q*x + c

-- Define the theorem
theorem root_product_equality (p q c : ℝ) (α β γ δ : ℝ) :
  quadratic1 α p c = 0 →
  quadratic1 β p c = 0 →
  quadratic2 γ q c = 0 →
  quadratic2 δ q c = 0 →
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = (p^2 - q^2) * c + c^2 - p*c - q*c :=
by sorry

end root_product_equality_l161_16147


namespace planes_lines_relations_l161_16127

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem planes_lines_relations 
  (α β : Plane) (l m : Line) 
  (h1 : perpendicular l α) 
  (h2 : contained_in m β) :
  (parallel α β → line_perpendicular l m) ∧ 
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end planes_lines_relations_l161_16127


namespace complex_geometry_problem_l161_16103

/-- A complex number z with specific properties -/
def z : ℂ :=
  sorry

/-- The condition that |z| = √2 -/
axiom z_norm : Complex.abs z = Real.sqrt 2

/-- The condition that the imaginary part of z² is 2 -/
axiom z_sq_im : Complex.im (z ^ 2) = 2

/-- The condition that z is in the first quadrant -/
axiom z_first_quadrant : Complex.re z > 0 ∧ Complex.im z > 0

/-- Point A corresponds to z -/
def A : ℂ := z

/-- Point B corresponds to z² -/
def B : ℂ := z ^ 2

/-- Point C corresponds to z - z² -/
def C : ℂ := z - z ^ 2

/-- The main theorem to be proved -/
theorem complex_geometry_problem :
  z = 1 + Complex.I ∧
  Real.cos (Complex.arg (B - A) - Complex.arg (C - B)) = -2 * Real.sqrt 5 / 5 :=
sorry

end complex_geometry_problem_l161_16103


namespace book_cost_problem_l161_16179

/-- Proves that given two books with a total cost of 480, where one is sold at a 15% loss 
and the other at a 19% gain, and both are sold at the same price, 
the cost of the book sold at a loss is 280. -/
theorem book_cost_problem (c1 c2 : ℝ) : 
  c1 + c2 = 480 →
  c1 * 0.85 = c2 * 1.19 →
  c1 = 280 := by
  sorry

end book_cost_problem_l161_16179


namespace wheel_speed_l161_16184

/-- The speed of the wheel in miles per hour -/
def r : ℝ := sorry

/-- The circumference of the wheel in feet -/
def circumference : ℝ := 11

/-- The time for one rotation in hours -/
def t : ℝ := sorry

/-- Conversion factor from feet to miles -/
def feet_per_mile : ℝ := 5280

/-- Conversion factor from hours to seconds -/
def seconds_per_hour : ℝ := 3600

/-- The relationship between speed, time, and distance -/
axiom speed_time_distance : r * t = circumference / feet_per_mile

/-- The relationship when time is decreased and speed is increased -/
axiom increased_speed_decreased_time : 
  (r + 5) * (t - 1 / (4 * seconds_per_hour)) = circumference / feet_per_mile

theorem wheel_speed : r = 10 := by sorry

end wheel_speed_l161_16184


namespace triangle_shape_l161_16181

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem triangle_shape (t : Triangle) 
  (p : Vector2D) 
  (q : Vector2D) 
  (hp : p = ⟨t.c^2, t.a^2⟩) 
  (hq : q = ⟨Real.tan t.C, Real.tan t.A⟩) 
  (hpq : parallel p q) : 
  (t.a = t.c) ∨ (t.b^2 = t.a^2 + t.c^2) := by
  sorry

end triangle_shape_l161_16181


namespace sara_purse_value_l161_16108

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  quarters * coin_value "quarter"

/-- Converts a number of cents to a percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem sara_purse_value :
  cents_to_percentage (total_value 3 2 1 2) = 73 / 100 := by
  sorry

end sara_purse_value_l161_16108


namespace factorial_10_mod_13_l161_16164

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end factorial_10_mod_13_l161_16164


namespace roots_of_equation_number_of_roots_l161_16194

def f (x : ℝ) : ℝ := x + |x^2 - 1|

theorem roots_of_equation (k : ℝ) :
  (∀ x, f x ≠ k) ∨
  (∃! x, f x = k) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∨
  (∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧ f x₄ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
by
  sorry

theorem number_of_roots (k : ℝ) :
  (k < -1 → ∀ x, f x ≠ k) ∧
  (k = -1 → ∃! x, f x = k) ∧
  (-1 < k ∧ k < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) ∧
  (k = 1 → ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (1 < k ∧ k < 5/4 → ∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧ f x₄ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
  (k = 5/4 → ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (k > 5/4 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) :=
by
  sorry

end roots_of_equation_number_of_roots_l161_16194


namespace candy_duration_l161_16156

theorem candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) :
  neighbors_candy = 66 →
  sister_candy = 15 →
  daily_consumption = 9 →
  (neighbors_candy + sister_candy) / daily_consumption = 9 :=
by sorry

end candy_duration_l161_16156


namespace second_plan_fee_calculation_l161_16122

/-- The monthly fee for the first plan -/
def first_plan_monthly_fee : ℚ := 22

/-- The per-minute fee for the first plan -/
def first_plan_per_minute : ℚ := 13 / 100

/-- The monthly fee for the second plan -/
def second_plan_monthly_fee : ℚ := 8

/-- The number of minutes at which both plans cost the same -/
def equal_cost_minutes : ℚ := 280

/-- The per-minute fee for the second plan -/
def second_plan_per_minute : ℚ := 18 / 100

theorem second_plan_fee_calculation :
  first_plan_monthly_fee + first_plan_per_minute * equal_cost_minutes =
  second_plan_monthly_fee + second_plan_per_minute * equal_cost_minutes := by
  sorry

end second_plan_fee_calculation_l161_16122


namespace perpendicular_vectors_m_value_l161_16162

-- Define the vectors
def a : ℝ × ℝ := (3, 1)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = -6 := by
  sorry

end perpendicular_vectors_m_value_l161_16162


namespace spinner_probability_theorem_l161_16180

/-- Represents the probability of landing on each part of a circular spinner -/
structure SpinnerProbabilities where
  A : ℚ
  B : ℚ
  C : ℚ
  D : ℚ

/-- Theorem: If a circular spinner has probabilities 1/4 for A, 1/3 for B, and 1/6 for D,
    then the probability for C is 1/4 -/
theorem spinner_probability_theorem (sp : SpinnerProbabilities) 
  (hA : sp.A = 1/4)
  (hB : sp.B = 1/3)
  (hD : sp.D = 1/6)
  (hSum : sp.A + sp.B + sp.C + sp.D = 1) :
  sp.C = 1/4 := by
sorry

end spinner_probability_theorem_l161_16180


namespace roller_coaster_friends_l161_16157

theorem roller_coaster_friends (tickets_per_ride : ℕ) (total_tickets : ℕ) (num_friends : ℕ) : 
  tickets_per_ride = 6 → total_tickets = 48 → num_friends * tickets_per_ride = total_tickets → num_friends = 8 := by
  sorry

end roller_coaster_friends_l161_16157


namespace star_four_eight_two_l161_16177

-- Define the ⋆ operation
def star (a b c : ℕ+) : ℚ := (a * b + c) / (a + b + c)

-- Theorem statement
theorem star_four_eight_two :
  star 4 8 2 = 17 / 7 := by sorry

end star_four_eight_two_l161_16177


namespace probability_of_three_positive_answers_l161_16174

/-- The probability of getting exactly k successes in n trials,
    where the probability of success on each trial is p. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of questions asked -/
def total_questions : ℕ := 7

/-- The number of positive answers we're interested in -/
def positive_answers : ℕ := 3

/-- The probability of a positive answer for each question -/
def positive_probability : ℚ := 3/7

theorem probability_of_three_positive_answers :
  binomial_probability total_questions positive_answers positive_probability = 242112/823543 := by
  sorry

end probability_of_three_positive_answers_l161_16174


namespace quadratic_equation_roots_relation_l161_16167

theorem quadratic_equation_roots_relation (p : ℚ) : 
  (∃ x1 x2 : ℚ, 3 * x1^2 - 5*(p-1)*x1 + p^2 + 2 = 0 ∧
                3 * x2^2 - 5*(p-1)*x2 + p^2 + 2 = 0 ∧
                x1 + 4*x2 = 14) ↔ 
  (p = 742/127 ∨ p = 4) :=
by sorry

end quadratic_equation_roots_relation_l161_16167


namespace kaylee_biscuit_sales_l161_16186

/-- The number of boxes Kaylee needs to sell -/
def total_boxes : ℕ := 33

/-- The number of lemon biscuit boxes sold -/
def lemon_boxes : ℕ := 12

/-- The number of chocolate biscuit boxes sold -/
def chocolate_boxes : ℕ := 5

/-- The number of oatmeal biscuit boxes sold -/
def oatmeal_boxes : ℕ := 4

/-- The number of additional boxes Kaylee needs to sell -/
def additional_boxes : ℕ := total_boxes - (lemon_boxes + chocolate_boxes + oatmeal_boxes)

theorem kaylee_biscuit_sales : additional_boxes = 12 := by
  sorry

end kaylee_biscuit_sales_l161_16186


namespace a_greater_than_b_l161_16149

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1) 
  (h_b_eq : b^(2*n) = b + 3*a) : 
  a > b := by
  sorry

end a_greater_than_b_l161_16149


namespace rackets_sold_l161_16169

def total_sales : ℝ := 588
def average_price : ℝ := 9.8

theorem rackets_sold (pairs : ℝ) : pairs = total_sales / average_price → pairs = 60 := by
  sorry

end rackets_sold_l161_16169


namespace cubic_equation_one_root_strategy_l161_16193

theorem cubic_equation_one_root_strategy :
  ∃ (strategy : ℝ → ℝ → ℝ),
    ∀ (a b c : ℝ),
      ∃ (root : ℝ),
        (root^3 + a*root^2 + b*root + c = 0) ∧
        (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 → x = root) :=
sorry

end cubic_equation_one_root_strategy_l161_16193


namespace apple_plum_ratio_l161_16159

theorem apple_plum_ratio :
  ∀ (apples plums : ℕ),
    apples = 180 →
    apples + plums = 240 →
    (2 : ℚ) / 5 * (apples + plums) = 96 →
    (apples : ℚ) / plums = 3 / 1 := by
  sorry

end apple_plum_ratio_l161_16159


namespace diego_payment_is_9800_l161_16120

def total_payment : ℝ := 50000
def celina_payment (diego_payment : ℝ) : ℝ := 1000 + 4 * diego_payment

theorem diego_payment_is_9800 :
  ∃ (diego_payment : ℝ),
    diego_payment + celina_payment diego_payment = total_payment ∧
    diego_payment = 9800 :=
by sorry

end diego_payment_is_9800_l161_16120


namespace starship_age_conversion_l161_16109

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_number (octal : List Nat) : Nat :=
  octal.enum.foldr (fun (i, digit) acc => acc + octal_to_decimal digit * (8^i)) 0

theorem starship_age_conversion :
  octal_to_decimal_number [6, 7, 2, 4] = 3540 := by
  sorry

end starship_age_conversion_l161_16109


namespace meal_base_cost_is_28_l161_16126

/-- Represents the cost structure of a meal --/
structure MealCost where
  baseCost : ℝ
  taxRate : ℝ
  tipRate : ℝ
  totalCost : ℝ

/-- Calculates the total cost of a meal given its base cost, tax rate, and tip rate --/
def calculateTotalCost (m : MealCost) : ℝ :=
  m.baseCost * (1 + m.taxRate + m.tipRate)

/-- Theorem stating that given the specified conditions, the base cost of the meal is $28 --/
theorem meal_base_cost_is_28 (m : MealCost) 
  (h1 : m.taxRate = 0.08)
  (h2 : m.tipRate = 0.18)
  (h3 : m.totalCost = 35.20)
  (h4 : calculateTotalCost m = m.totalCost) :
  m.baseCost = 28 := by
  sorry

#eval (28 : ℚ) * (1 + 0.08 + 0.18)

end meal_base_cost_is_28_l161_16126


namespace cone_lateral_surface_angle_l161_16188

/-- The angle in the lateral surface unfolding of a cone, given that its lateral surface area is twice the area of its base. -/
theorem cone_lateral_surface_angle (r : ℝ) (h : r > 0) : 
  let l := 2 * r
  let base_area := π * r^2
  let lateral_area := π * r * l
  lateral_area = 2 * base_area →
  (lateral_area / (π * l^2)) * 360 = 180 :=
by sorry

end cone_lateral_surface_angle_l161_16188


namespace clean_city_people_l161_16114

/-- The number of people in group A -/
def group_A : ℕ := 54

/-- The number of people in group B -/
def group_B : ℕ := group_A - 17

/-- The number of people in group C -/
def group_C : ℕ := 2 * group_B

/-- The number of people in group D -/
def group_D : ℕ := group_A / 3

/-- The total number of people working together to clean the city -/
def total_people : ℕ := group_A + group_B + group_C + group_D

theorem clean_city_people : total_people = 183 := by
  sorry

end clean_city_people_l161_16114


namespace train_speed_l161_16198

/-- Given a train of length 200 meters that takes 5 seconds to cross an electric pole,
    prove that its speed is 40 meters per second. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 200) (h2 : crossing_time = 5) :
  train_length / crossing_time = 40 :=
by sorry

end train_speed_l161_16198


namespace water_speed_proof_l161_16138

/-- Proves that the speed of the water is 2 km/h, given the conditions of the swimming problem. -/
theorem water_speed_proof (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : still_water_speed = 4) (h2 : distance = 10) (h3 : time = 5) :
  ∃ water_speed : ℝ, water_speed = 2 ∧ still_water_speed - water_speed = distance / time :=
by sorry

end water_speed_proof_l161_16138


namespace boat_speed_ratio_l161_16191

/-- Given a boat's upstream and downstream travel times, 
    prove the ratio of current speed to boat speed in still water -/
theorem boat_speed_ratio 
  (distance : ℝ) 
  (upstream_time downstream_time : ℝ) 
  (h1 : distance = 15)
  (h2 : upstream_time = 5)
  (h3 : downstream_time = 3) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧
    current_speed > 0 ∧
    distance / upstream_time = boat_speed - current_speed ∧
    distance / downstream_time = boat_speed + current_speed ∧
    current_speed / boat_speed = 1 / 4 := by
  sorry

end boat_speed_ratio_l161_16191


namespace triangle_count_l161_16171

/-- Calculates the total number of triangles in a triangular figure composed of n rows of small isosceles triangles. -/
def totalTriangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The number of rows in our specific triangular figure -/
def numRows : ℕ := 7

/-- Theorem stating that the total number of triangles in our specific figure is 28 -/
theorem triangle_count : totalTriangles numRows = 28 := by
  sorry

end triangle_count_l161_16171


namespace mozzarella_count_l161_16104

def cheese_pack (cheddar pepperjack mozzarella : ℕ) : Prop :=
  cheddar = 15 ∧ 
  pepperjack = 45 ∧ 
  (pepperjack : ℚ) / (cheddar + pepperjack + mozzarella) = 1/2

theorem mozzarella_count : ∃ m : ℕ, cheese_pack 15 45 m ∧ m = 30 := by
  sorry

end mozzarella_count_l161_16104


namespace recurring_decimal_fraction_l161_16148

theorem recurring_decimal_fraction (a b : ℚ) :
  a = 36 * (1 / 99) ∧ b = 12 * (1 / 99) → a / b = 3 := by
  sorry

end recurring_decimal_fraction_l161_16148


namespace unique_solution_l161_16189

/-- A function y is a solution to the differential equation y' - y = cos x - sin x
    and is bounded as x approaches positive infinity -/
def IsSolution (y : ℝ → ℝ) : Prop :=
  (∀ x, (deriv y x) - y x = Real.cos x - Real.sin x) ∧
  (∃ M, ∀ x, x ≥ 0 → |y x| ≤ M)

/-- The unique solution to the differential equation y' - y = cos x - sin x
    that is bounded as x approaches positive infinity is y = - cos x -/
theorem unique_solution :
  ∃! y, IsSolution y ∧ (∀ x, y x = - Real.cos x) :=
sorry

end unique_solution_l161_16189


namespace variance_scaling_l161_16185

-- Define a function to calculate the variance of a list of numbers
noncomputable def variance (data : List ℝ) : ℝ := sorry

-- Define our theorem
theorem variance_scaling (data : List ℝ) :
  variance data = 4 → variance (List.map (· * 2) data) = 16 := by
  sorry

end variance_scaling_l161_16185


namespace horse_journey_l161_16140

theorem horse_journey (a₁ : ℚ) : 
  (a₁ * (1 - (1/2)^7) / (1 - 1/2) = 700) → 
  (a₁ * (1/2)^6 = 700/127) := by
sorry

end horse_journey_l161_16140


namespace lcm_of_36_and_125_l161_16130

theorem lcm_of_36_and_125 : Nat.lcm 36 125 = 4500 := by
  sorry

end lcm_of_36_and_125_l161_16130


namespace complex_distance_l161_16124

theorem complex_distance (z : ℂ) : z = 1 - 2*I → Complex.abs z = Real.sqrt 5 := by sorry

end complex_distance_l161_16124


namespace largest_rational_satisfying_equation_l161_16133

theorem largest_rational_satisfying_equation :
  ∀ x : ℚ, |x - 7/2| = 25/2 → x ≤ 16 :=
by
  sorry

end largest_rational_satisfying_equation_l161_16133


namespace no_delightful_eight_digit_integers_l161_16166

/-- Represents an 8-digit positive integer as a list of its digits -/
def EightDigitInteger := List Nat

/-- Checks if a list of digits forms a valid 8-digit integer -/
def isValid (n : EightDigitInteger) : Prop :=
  n.length = 8 ∧ n.toFinset = Finset.range 9 \ {0}

/-- Checks if the sum of the first k digits is divisible by k for all k from 1 to 8 -/
def isDelightful (n : EightDigitInteger) : Prop :=
  ∀ k : Nat, k ∈ Finset.range 9 \ {0} → (n.take k).sum % k = 0

/-- The main theorem: there are no delightful 8-digit integers -/
theorem no_delightful_eight_digit_integers :
  ¬∃ n : EightDigitInteger, isValid n ∧ isDelightful n := by
  sorry

end no_delightful_eight_digit_integers_l161_16166


namespace house_rent_percentage_l161_16101

-- Define the percentages as real numbers
def food_percentage : ℝ := 0.50
def education_percentage : ℝ := 0.15
def remaining_percentage : ℝ := 0.175

-- Define the theorem
theorem house_rent_percentage :
  let total_income : ℝ := 100
  let remaining_after_food_education : ℝ := total_income * (1 - food_percentage - education_percentage)
  let spent_on_rent : ℝ := remaining_after_food_education - (total_income * remaining_percentage)
  (spent_on_rent / remaining_after_food_education) = 0.5 := by sorry

end house_rent_percentage_l161_16101


namespace fifty_square_divisible_by_one_by_four_strips_l161_16176

/-- Represents a rectangular strip --/
structure Strip where
  width : ℕ
  length : ℕ

/-- Represents a square --/
structure Square where
  side : ℕ

/-- Checks if a square can be divided into strips --/
def isDivisible (s : Square) (strip : Strip) : Prop :=
  s.side * s.side % (strip.width * strip.length) = 0

/-- Theorem: A 50x50 square can be divided into 1x4 strips --/
theorem fifty_square_divisible_by_one_by_four_strips :
  isDivisible (Square.mk 50) (Strip.mk 1 4) := by
  sorry

end fifty_square_divisible_by_one_by_four_strips_l161_16176


namespace bacteria_population_after_15_days_l161_16106

/-- Calculates the population of bacteria cells after a given number of days -/
def bacteriaPopulation (initialCells : ℕ) (daysPerDivision : ℕ) (totalDays : ℕ) : ℕ :=
  initialCells * (3 ^ (totalDays / daysPerDivision))

/-- Theorem stating that the bacteria population after 15 days is 1215 cells -/
theorem bacteria_population_after_15_days :
  bacteriaPopulation 5 3 15 = 1215 := by
  sorry

end bacteria_population_after_15_days_l161_16106


namespace quadratic_roots_integrality_l161_16158

theorem quadratic_roots_integrality (q : ℤ) :
  (q > 0 → ∃ (p : ℤ), ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) ∧
  (q < 0 → ¬∃ (p : ℤ), ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) :=
by sorry

end quadratic_roots_integrality_l161_16158


namespace vector_parallel_implies_m_equals_two_l161_16172

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem vector_parallel_implies_m_equals_two (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (2, 1)
  parallel (a.1 - 2 * b.1, a.2 - 2 * b.2) b →
  m = 2 := by
sorry

end vector_parallel_implies_m_equals_two_l161_16172


namespace range_of_a_l161_16199

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2 * (1 - x^2) - 3 * x > 0 → x > a) ∧ 
  (∃ y : ℝ, y > a ∧ 2 * (1 - y^2) - 3 * y ≤ 0)) → 
  a ∈ Set.Iic (-2 : ℝ) :=
sorry

end range_of_a_l161_16199


namespace matrix_computation_l161_16144

theorem matrix_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![1, 3] = ![2, 5])
  (h2 : N.mulVec ![-2, 4] = ![3, 1]) :
  N.mulVec ![3, 11] = ![7.4, 17.2] := by
sorry

end matrix_computation_l161_16144


namespace investment_return_calculation_l161_16102

theorem investment_return_calculation (total_investment small_investment large_investment : ℝ)
  (combined_return_rate small_return_rate : ℝ) :
  total_investment = small_investment + large_investment →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.085 →
  small_return_rate = 0.07 →
  (small_return_rate * small_investment + 
   (combined_return_rate * total_investment - small_return_rate * small_investment) / large_investment)
  = 0.09 := by
sorry

end investment_return_calculation_l161_16102


namespace intersection_of_A_and_B_l161_16125

def A : Set ℝ := {x | x^2 - 2*x = 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end intersection_of_A_and_B_l161_16125


namespace min_sum_of_distances_l161_16187

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_eq_ac : dist A B = dist A C)
  (ab_eq_5 : dist A B = 5)
  (bc_eq_6 : dist B C = 6)

/-- Point on the sides of the triangle -/
def PointOnSides (t : Triangle) : Set (ℝ × ℝ) :=
  {P | ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧
    (P = (1 - s) • t.A + s • t.B ∨
     P = (1 - s) • t.B + s • t.C ∨
     P = (1 - s) • t.C + s • t.A)}

/-- Sum of distances from P to vertices -/
def SumOfDistances (t : Triangle) (P : ℝ × ℝ) : ℝ :=
  dist P t.A + dist P t.B + dist P t.C

/-- Theorem: Minimum sum of distances is 16 -/
theorem min_sum_of_distances (t : Triangle) :
  ∀ P ∈ PointOnSides t, SumOfDistances t P ≥ 16 :=
sorry

end min_sum_of_distances_l161_16187


namespace olympic_quiz_probability_l161_16137

theorem olympic_quiz_probability (A B C : ℝ) 
  (hA : A = 3/4)
  (hAC : (1 - A) * (1 - C) = 1/12)
  (hBC : B * C = 1/4) :
  A * B * (1 - C) + A * (1 - B) * C + (1 - A) * B * C = 15/32 := by
  sorry

end olympic_quiz_probability_l161_16137


namespace anion_and_salt_identification_l161_16116

/-- Represents an anion with an O-O bond -/
structure AnionWithOOBond where
  has_oo_bond : Bool

/-- Represents a salt formed during anodic oxidation of bisulfate -/
structure SaltFromBisulfateOxidation where
  is_sulfate_based : Bool

/-- Theorem stating that an anion with an O-O bond is a peroxide ion and 
    the salt formed from bisulfate oxidation is sulfate-based -/
theorem anion_and_salt_identification 
  (anion : AnionWithOOBond) 
  (salt : SaltFromBisulfateOxidation) : 
  (anion.has_oo_bond → (∃ x : String, x = "O₂²⁻")) ∧ 
  (salt.is_sulfate_based → (∃ y : String, y = "K₂SO₄")) := by
  sorry

end anion_and_salt_identification_l161_16116


namespace fraction_simplification_l161_16100

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2*d*e) / (d^2 + f^2 - e^2 + 3*d*f) = (d + e - f) / (d + f - e) :=
by sorry

end fraction_simplification_l161_16100


namespace pet_insurance_coverage_calculation_l161_16143

/-- Calculates the amount covered by pet insurance for a cat's visit -/
def pet_insurance_coverage (
  doctor_visit_cost : ℝ
  ) (health_insurance_rate : ℝ
  ) (cat_visit_cost : ℝ
  ) (total_out_of_pocket : ℝ
  ) : ℝ :=
  cat_visit_cost - (total_out_of_pocket - (doctor_visit_cost * (1 - health_insurance_rate)))

theorem pet_insurance_coverage_calculation :
  pet_insurance_coverage 300 0.75 120 135 = 60 := by
  sorry

end pet_insurance_coverage_calculation_l161_16143


namespace compound_composition_l161_16152

/-- The number of Aluminium atoms in the compound -/
def n : ℕ := 1

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℚ := 26.98

/-- Atomic weight of Phosphorus in g/mol -/
def P_weight : ℚ := 30.97

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℚ := 16.00

/-- Total molecular weight of the compound in g/mol -/
def total_weight : ℚ := 122

/-- Number of Phosphorus atoms in the compound -/
def P_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 4

theorem compound_composition :
  n * Al_weight + P_count * P_weight + O_count * O_weight = total_weight :=
sorry

end compound_composition_l161_16152


namespace circle_equation_tangent_line_equation_l161_16151

-- Define the circle
def circle_center : ℝ × ℝ := (-1, 2)
def line_m (x y : ℝ) : ℝ := x + 2*y + 7

-- Define the point Q
def point_Q : ℝ × ℝ := (1, 6)

-- Theorem for the circle equation
theorem circle_equation : 
  ∃ (r : ℝ), ∀ (x y : ℝ), 
  (x + 1)^2 + (y - 2)^2 = r^2 ∧ 
  (∃ (x₀ y₀ : ℝ), line_m x₀ y₀ = 0 ∧ ((x₀ + 1)^2 + (y₀ - 2)^2 = r^2)) :=
sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 2)^2 = 20) →
  (point_Q.1 + 1)^2 + (point_Q.2 - 2)^2 = 20 →
  (y - point_Q.2 = -(x - point_Q.1) / 2) ↔ (x + 2*y - 13 = 0) :=
sorry

end circle_equation_tangent_line_equation_l161_16151


namespace hardey_fitness_center_ratio_l161_16175

theorem hardey_fitness_center_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℝ),
  f_avg = 55 →
  m_avg = 80 →
  total_avg = 70 →
  (f_avg * f + m_avg * m) / (f + m) = total_avg →
  (f : ℝ) / m = 2 / 3 := by
  sorry

end hardey_fitness_center_ratio_l161_16175


namespace chess_tournament_games_l161_16131

theorem chess_tournament_games (P : ℕ) (total_games : ℕ) (h1 : P = 21) (h2 : total_games = 210) :
  (P * (P - 1)) / 2 = total_games ∧ P - 1 = 20 := by
  sorry

end chess_tournament_games_l161_16131


namespace unique_six_digit_number_l161_16163

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def rightmost_digit (n : ℕ) : ℕ := n % 10

def move_rightmost_to_leftmost (n : ℕ) : ℕ :=
  (n / 10) + (rightmost_digit n * 100000)

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
            rightmost_digit n = 2 ∧
            move_rightmost_to_leftmost n = 2 * n + 2 :=
by
  use 105262
  sorry

end unique_six_digit_number_l161_16163


namespace discount_profit_calculation_l161_16113

/-- Given a discount percentage and an original profit percentage,
    calculate the new profit percentage after applying the discount. -/
def profit_after_discount (discount : ℝ) (original_profit : ℝ) : ℝ :=
  let original_price := 1 + original_profit
  let discounted_price := original_price * (1 - discount)
  (discounted_price - 1) * 100

/-- Theorem stating that a 5% discount on an item with 50% original profit
    results in a 42.5% profit. -/
theorem discount_profit_calculation :
  profit_after_discount 0.05 0.5 = 42.5 := by sorry

end discount_profit_calculation_l161_16113


namespace tank_capacity_l161_16197

theorem tank_capacity : 
  ∀ (initial_fraction final_fraction added_water : ℚ),
  initial_fraction = 1/8 →
  final_fraction = 2/3 →
  added_water = 150 →
  ∃ (total_capacity : ℚ),
  (final_fraction - initial_fraction) * total_capacity = added_water ∧
  total_capacity = 3600/13 := by
sorry

end tank_capacity_l161_16197


namespace complex_magnitude_sum_l161_16155

theorem complex_magnitude_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end complex_magnitude_sum_l161_16155


namespace trig_expression_value_l161_16134

theorem trig_expression_value (α : Real) 
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) : 
  2 + 2/3 * (Real.sin α)^2 + 1/4 * (Real.cos α)^2 = 21/8 := by
  sorry

end trig_expression_value_l161_16134


namespace equal_distance_trajectory_length_l161_16168

/-- Rectilinear distance between two points -/
def rectilinearDistance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- The set of points C(x, y) with equal rectilinear distance to A and B -/
def equalDistancePoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 ∧
               rectilinearDistance x y 1 3 = rectilinearDistance x y 6 9}

/-- The sum of the lengths of the trajectories of all points in equalDistancePoints -/
noncomputable def trajectoryLength : ℝ :=
  5 * (Real.sqrt 2 + 1)

theorem equal_distance_trajectory_length :
  trajectoryLength = 5 * (Real.sqrt 2 + 1) := by
  sorry

end equal_distance_trajectory_length_l161_16168


namespace marias_gum_count_l161_16123

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem marias_gum_count 
  (x y z : ℕ) 
  (hx : is_two_digit x) 
  (hy : is_two_digit y) 
  (hz : is_two_digit z) : 
  58 + x + y + z = 58 + x + y + z :=
by sorry

end marias_gum_count_l161_16123


namespace tan_half_alpha_l161_16115

theorem tan_half_alpha (α : ℝ) (h1 : π < α) (h2 : α < 3*π/2) 
  (h3 : Real.sin (3*π/2 + α) = 4/5) : Real.tan (α/2) = -3 := by
  sorry

end tan_half_alpha_l161_16115


namespace probability_one_second_class_l161_16153

def total_products : ℕ := 12
def first_class_products : ℕ := 10
def second_class_products : ℕ := 2
def selected_products : ℕ := 4

theorem probability_one_second_class :
  (Nat.choose second_class_products 1 * Nat.choose first_class_products (selected_products - 1)) /
  (Nat.choose total_products selected_products) = 16 / 33 :=
by sorry

end probability_one_second_class_l161_16153


namespace divisors_of_squared_number_l161_16107

theorem divisors_of_squared_number (n : ℕ) (h : n > 1) :
  (Finset.card (Nat.divisors n) = 4) → (Finset.card (Nat.divisors (n^2)) = 9) := by
  sorry

end divisors_of_squared_number_l161_16107


namespace workers_count_l161_16146

-- Define the work function
def work (workers : ℕ) (hours : ℕ) : ℕ := workers * hours

-- Define the problem parameters
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def second_hours : ℕ := 6
def second_depth : ℕ := 55
def extra_workers : ℕ := 65

theorem workers_count :
  ∃ (W : ℕ), 
    (work W initial_hours) * second_depth = 
    (work (W + extra_workers) second_hours) * initial_depth ∧
    W = 45 := by
  sorry

end workers_count_l161_16146


namespace queens_attack_probability_l161_16190

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of ways to choose two different squares -/
def totalChoices : Nat := totalSquares * (totalSquares - 1) / 2

/-- The number of ways two queens can attack each other -/
def attackingChoices : Nat := 
  -- Same row
  boardSize * (boardSize * (boardSize - 1) / 2) +
  -- Same column
  boardSize * (boardSize * (boardSize - 1) / 2) +
  -- Same diagonal (main and anti-diagonals)
  (2 * (1 + 3 + 6 + 10 + 15 + 21) + 28)

/-- The probability of two queens attacking each other -/
def attackProbability : Rat := attackingChoices / totalChoices

theorem queens_attack_probability : 
  attackProbability = 7 / 24 := by sorry

end queens_attack_probability_l161_16190


namespace baseball_team_groups_l161_16183

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) 
  (h1 : new_players = 48)
  (h2 : returning_players = 6)
  (h3 : players_per_group = 6) :
  (new_players + returning_players) / players_per_group = 9 := by
  sorry

end baseball_team_groups_l161_16183


namespace cow_count_is_16_l161_16129

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs for a given animal count -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads for a given animal count -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that if the total number of legs is 32 more than twice the number of heads,
    then the number of cows is 16 -/
theorem cow_count_is_16 (count : AnimalCount) :
    totalLegs count = 2 * totalHeads count + 32 → count.cows = 16 := by
  sorry

end cow_count_is_16_l161_16129


namespace absolute_value_inequality_l161_16178

theorem absolute_value_inequality (x y z : ℝ) :
  |x| + |y| + |z| - |x + y| - |y + z| - |z + x| + |x + y + z| ≥ 0 := by
  sorry

end absolute_value_inequality_l161_16178


namespace vacation_cost_division_l161_16154

theorem vacation_cost_division (total_cost : ℝ) (initial_people : ℕ) (cost_reduction : ℝ) (n : ℕ) :
  total_cost = 1000 →
  initial_people = 4 →
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  cost_reduction = 50 →
  n = 5 := by
  sorry

end vacation_cost_division_l161_16154


namespace polygon_sides_and_diagonals_l161_16132

theorem polygon_sides_and_diagonals (n : ℕ) : 
  n + (n * (n - 3)) / 2 = 77 → n = 14 := by sorry

end polygon_sides_and_diagonals_l161_16132


namespace ampersand_eight_two_squared_l161_16161

def ampersand (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem ampersand_eight_two_squared :
  (ampersand 8 2)^2 = 3600 := by
  sorry

end ampersand_eight_two_squared_l161_16161


namespace complex_product_negative_l161_16141

theorem complex_product_negative (a : ℝ) :
  let z : ℂ := (a + Complex.I) * (-3 + a * Complex.I)
  (z.re < 0) → a = Real.sqrt 3 := by
sorry

end complex_product_negative_l161_16141
