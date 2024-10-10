import Mathlib

namespace sequence_convergence_l2726_272682

def is_smallest_prime_divisor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ p ∣ n ∧ ∀ q, Nat.Prime q → q ∣ n → p ≤ q

def sequence_condition (a p : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0 ∧ p n > 0) ∧
  a 1 ≥ 2 ∧
  (∀ n, is_smallest_prime_divisor (p n) (a n)) ∧
  (∀ n, a (n + 1) = a n + a n / p n)

theorem sequence_convergence (a p : ℕ → ℕ) (h : sequence_condition a p) :
  ∃ N, ∀ n > N, a (n + 3) = 3 * a n := by
  sorry

end sequence_convergence_l2726_272682


namespace solve_flowers_problem_l2726_272602

def flowers_problem (lilies sunflowers daisies total_flowers : ℕ) : Prop :=
  let other_flowers := lilies + sunflowers + daisies
  let roses := total_flowers - other_flowers
  (lilies = 40) ∧ (sunflowers = 40) ∧ (daisies = 40) ∧ (total_flowers = 160) →
  roses = 40

theorem solve_flowers_problem :
  ∀ (lilies sunflowers daisies total_flowers : ℕ),
  flowers_problem lilies sunflowers daisies total_flowers :=
by
  sorry

end solve_flowers_problem_l2726_272602


namespace IMO_2002_problem_l2726_272691

theorem IMO_2002_problem (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 → 
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  (∀ X Y Z : ℕ, X > 0 → Y > 0 → Z > 0 → X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
    A + B + C ≤ X + Y + Z) →
  (∀ X Y Z : ℕ, X > 0 → Y > 0 → Z > 0 → X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
    A + B + C ≥ X + Y + Z) →
  A + B + C = 52 ∧ A + B + C = 390 :=
by sorry

end IMO_2002_problem_l2726_272691


namespace steve_socks_count_l2726_272684

/-- The number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := 4

/-- The number of mismatching socks Steve has -/
def mismatching_socks : ℕ := 17

/-- The total number of socks Steve has -/
def total_socks : ℕ := 2 * matching_pairs + mismatching_socks

theorem steve_socks_count : total_socks = 25 := by
  sorry

end steve_socks_count_l2726_272684


namespace quadratic_equation_unique_solution_l2726_272601

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 8 * x + c = 0) →
  a + 2 * c = 14 →
  a < c →
  (a = (7 - Real.sqrt 17) / 2 ∧ c = (7 + Real.sqrt 17) / 2) :=
by sorry

end quadratic_equation_unique_solution_l2726_272601


namespace partial_fraction_product_l2726_272620

/-- Given a rational function and its partial fraction decomposition, prove that the product of the numerator coefficients is zero. -/
theorem partial_fraction_product (x : ℝ) (A B C : ℝ) : 
  (x^2 - 25) / (x^3 - x^2 - 7*x + 15) = A / (x - 3) + B / (x + 3) + C / (x - 5) →
  A * B * C = 0 := by
  sorry

end partial_fraction_product_l2726_272620


namespace factorization_3ax2_minus_3ay2_l2726_272625

theorem factorization_3ax2_minus_3ay2 (a x y : ℝ) : 3*a*x^2 - 3*a*y^2 = 3*a*(x+y)*(x-y) := by
  sorry

end factorization_3ax2_minus_3ay2_l2726_272625


namespace candy_difference_l2726_272656

theorem candy_difference (sandra_bags : Nat) (sandra_pieces_per_bag : Nat)
  (roger_bag1 : Nat) (roger_bag2 : Nat) :
  sandra_bags = 2 →
  sandra_pieces_per_bag = 6 →
  roger_bag1 = 11 →
  roger_bag2 = 3 →
  (roger_bag1 + roger_bag2) - (sandra_bags * sandra_pieces_per_bag) = 2 := by
  sorry

end candy_difference_l2726_272656


namespace present_age_of_b_l2726_272687

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39                     -- B's present age is 39 years
:= by sorry

end present_age_of_b_l2726_272687


namespace replaced_person_weight_l2726_272626

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (average_increase : ℝ) (weight_of_new_person : ℝ) : ℝ :=
  weight_of_new_person - 2 * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 4.5 74 = 65 := by
  sorry

end replaced_person_weight_l2726_272626


namespace book_pages_l2726_272628

/-- Calculates the total number of pages in a book given reading rate and time spent reading. -/
def total_pages (pages_per_hour : ℝ) (monday_hours : ℝ) (tuesday_hours : ℝ) (remaining_hours : ℝ) : ℝ :=
  pages_per_hour * (monday_hours + tuesday_hours + remaining_hours)

/-- Theorem stating that the book has 248 pages given Joanna's reading rate and time spent. -/
theorem book_pages : 
  let pages_per_hour : ℝ := 16
  let monday_hours : ℝ := 3
  let tuesday_hours : ℝ := 6.5
  let remaining_hours : ℝ := 6
  total_pages pages_per_hour monday_hours tuesday_hours remaining_hours = 248 := by
  sorry

end book_pages_l2726_272628


namespace f_even_and_increasing_l2726_272668

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Statement: f is an even function and increasing on (0, +∞)
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_even_and_increasing_l2726_272668


namespace inequality_one_inequality_two_l2726_272619

-- Statement 1
theorem inequality_one (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by sorry

-- Statement 2
theorem inequality_two {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  Real.sqrt ((a^2 + b^2) / 2) ≥ (a + b) / 2 := by sorry

end inequality_one_inequality_two_l2726_272619


namespace celebrity_baby_matching_probability_l2726_272655

theorem celebrity_baby_matching_probability :
  let n : ℕ := 4
  let total_arrangements := n.factorial
  let correct_arrangements : ℕ := 1
  (correct_arrangements : ℚ) / total_arrangements = 1 / 24 :=
by sorry

end celebrity_baby_matching_probability_l2726_272655


namespace enrique_shredder_y_feeds_l2726_272669

/-- Calculates the number of times a shredder needs to be fed to shred all pages of a contract type -/
def shredder_feeds (num_contracts : ℕ) (pages_per_contract : ℕ) (pages_per_shred : ℕ) : ℕ :=
  ((num_contracts * pages_per_contract + pages_per_shred - 1) / pages_per_shred : ℕ)

theorem enrique_shredder_y_feeds : 
  let type_b_contracts : ℕ := 350
  let pages_per_type_b : ℕ := 10
  let shredder_y_capacity : ℕ := 8
  shredder_feeds type_b_contracts pages_per_type_b shredder_y_capacity = 438 := by
sorry

end enrique_shredder_y_feeds_l2726_272669


namespace jake_snake_revenue_l2726_272665

/-- Calculates the total revenue from selling baby snakes --/
def total_revenue (num_snakes : ℕ) (eggs_per_snake : ℕ) (regular_price : ℕ) (rare_price_multiplier : ℕ) : ℕ :=
  let total_babies := num_snakes * eggs_per_snake
  let num_regular_babies := total_babies - 1
  let rare_price := regular_price * rare_price_multiplier
  num_regular_babies * regular_price + rare_price

/-- The revenue from Jake's snake business --/
theorem jake_snake_revenue :
  total_revenue 3 2 250 4 = 2250 := by
  sorry

end jake_snake_revenue_l2726_272665


namespace f_properties_l2726_272650

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem f_properties :
  (∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x = 5) ∧
  (∀ (y : ℝ), -2 < y ∧ y < 2 → f y ≤ 5) ∧
  (¬ ∃ (z : ℝ), -2 < z ∧ z < 2 ∧ ∀ (w : ℝ), -2 < w ∧ w < 2 → f z ≤ f w) :=
by sorry

end f_properties_l2726_272650


namespace some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire_l2726_272641

-- Define the universe of discourse
def Dragon : Type := sorry

-- Define the property of breathing fire
def breathes_fire : Dragon → Prop := sorry

-- Theorem: "Some dragons do not breathe fire" is equivalent to 
-- the negation of "All dragons breathe fire"
theorem some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire :
  (∃ d : Dragon, ¬(breathes_fire d)) ↔ ¬(∀ d : Dragon, breathes_fire d) := by
  sorry

end some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire_l2726_272641


namespace solve_for_a_l2726_272667

theorem solve_for_a : ∀ a : ℝ, (2 * 2 + a - 5 = 0) → a = 1 := by sorry

end solve_for_a_l2726_272667


namespace probability_square_divisor_15_factorial_l2726_272604

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of positive integer divisors of n that are perfect squares -/
def num_square_divisors (n : ℕ) : ℕ := sorry

/-- The total number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Two natural numbers are coprime -/
def coprime (a b : ℕ) : Prop := sorry

theorem probability_square_divisor_15_factorial :
  ∃ m n : ℕ, 
    coprime m n ∧ 
    (m : ℚ) / n = (num_square_divisors (factorial 15) : ℚ) / (num_divisors (factorial 15)) ∧
    m = 1 ∧ n = 84 := by
  sorry

end probability_square_divisor_15_factorial_l2726_272604


namespace pen_pencil_ratio_l2726_272640

/-- Given 54 pencils and 9 more pencils than pens, prove that the ratio of pens to pencils is 5:6 -/
theorem pen_pencil_ratio : 
  ∀ (num_pens num_pencils : ℕ), 
  num_pencils = 54 → 
  num_pencils = num_pens + 9 → 
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end pen_pencil_ratio_l2726_272640


namespace emily_phone_bill_l2726_272610

/-- Calculates the total cost of a cell phone plan based on usage. -/
def calculate_total_cost (base_cost : ℚ) (included_hours : ℚ) (text_cost : ℚ) 
  (extra_minute_cost : ℚ) (data_cost : ℚ) (texts_sent : ℚ) (hours_used : ℚ) 
  (data_used : ℚ) : ℚ :=
  base_cost + 
  (text_cost * texts_sent) + 
  (max (hours_used - included_hours) 0 * 60 * extra_minute_cost) + 
  (data_cost * data_used)

theorem emily_phone_bill :
  let base_cost : ℚ := 25
  let included_hours : ℚ := 25
  let text_cost : ℚ := 0.1
  let extra_minute_cost : ℚ := 0.15
  let data_cost : ℚ := 2
  let texts_sent : ℚ := 150
  let hours_used : ℚ := 26
  let data_used : ℚ := 3
  calculate_total_cost base_cost included_hours text_cost extra_minute_cost 
    data_cost texts_sent hours_used data_used = 55 := by
  sorry

end emily_phone_bill_l2726_272610


namespace max_value_when_h_3_h_values_when_max_negative_one_l2726_272694

-- Define the quadratic function
def f (h : ℝ) (x : ℝ) : ℝ := -(x - h)^2

-- Define the range of x
def x_range (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5

-- Part 1: Maximum value when h = 3
theorem max_value_when_h_3 :
  ∀ x, x_range x → f 3 x ≤ 0 ∧ ∃ x₀, x_range x₀ ∧ f 3 x₀ = 0 :=
sorry

-- Part 2: Values of h when maximum is -1
theorem h_values_when_max_negative_one :
  (∀ x, x_range x → f h x ≤ -1 ∧ ∃ x₀, x_range x₀ ∧ f h x₀ = -1) →
  h = 6 ∨ h = 1 :=
sorry

end max_value_when_h_3_h_values_when_max_negative_one_l2726_272694


namespace root_inequality_l2726_272646

noncomputable def f (x : ℝ) : ℝ := (1 + 2 * Real.log x) / x^2

theorem root_inequality (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f x₁ = k) 
  (h2 : f x₂ = k) 
  (h3 : x₁ < x₂) :
  x₁ + x₂ > 2 ∧ 2 > 1/x₁ + 1/x₂ := by
  sorry

end root_inequality_l2726_272646


namespace eliminate_x_from_system_l2726_272627

theorem eliminate_x_from_system : ∀ x y : ℝ,
  (2 * x - 3 * y = 11 ∧ 2 * x + 5 * y = -5) →
  -8 * y = 16 := by
  sorry

end eliminate_x_from_system_l2726_272627


namespace locus_of_circle_centers_l2726_272637

/-- Given a point O and a radius R, the locus of centers C of circles with radius R
    passing through O is a circle with center O and radius R. -/
theorem locus_of_circle_centers (O : ℝ × ℝ) (R : ℝ) :
  {C : ℝ × ℝ | ∃ P, dist P C = R ∧ P = O} = {C : ℝ × ℝ | dist C O = R} := by sorry

end locus_of_circle_centers_l2726_272637


namespace simplify_expression_1_expand_expression_2_l2726_272642

-- First expression
theorem simplify_expression_1 (x y : ℝ) (h : y ≠ 0) :
  (3 * x^2 * y - 6 * x * y) / (3 * x * y) = x - 2 := by sorry

-- Second expression
theorem expand_expression_2 (a b : ℝ) :
  (a + b + 2) * (a + b - 2) = a^2 + 2*a*b + b^2 - 4 := by sorry

end simplify_expression_1_expand_expression_2_l2726_272642


namespace water_displacement_theorem_l2726_272638

/-- Represents a cylindrical barrel --/
structure Barrel where
  radius : ℝ
  height : ℝ

/-- Represents a cube --/
structure Cube where
  side_length : ℝ

/-- Calculates the volume of water displaced by a cube in a barrel --/
def water_displaced (barrel : Barrel) (cube : Cube) : ℝ :=
  cube.side_length ^ 3

/-- The main theorem about water displacement and its square --/
theorem water_displacement_theorem (barrel : Barrel) (cube : Cube)
    (h1 : barrel.radius = 5)
    (h2 : barrel.height = 15)
    (h3 : cube.side_length = 10) :
    let v := water_displaced barrel cube
    v = 1000 ∧ v^2 = 1000000 := by
  sorry

#check water_displacement_theorem

end water_displacement_theorem_l2726_272638


namespace largest_integer_satisfying_inequality_l2726_272631

theorem largest_integer_satisfying_inequality :
  ∀ y : ℤ, y ≤ 5 ↔ y / 3 + 5 / 3 < 11 / 3 :=
by sorry

end largest_integer_satisfying_inequality_l2726_272631


namespace value_not_unique_l2726_272633

/-- A quadratic function passing through (0, 1) and (1, 0), and concave down -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  pass_origin : c = 1
  pass_one : a + b + c = 0
  concave_down : a < 0

/-- The value of a - b + c cannot be uniquely determined for all quadratic functions satisfying the given conditions -/
theorem value_not_unique (f g : QuadraticFunction) : ∃ (f g : QuadraticFunction), f.a - f.b + f.c ≠ g.a - g.b + g.c := by
  sorry

end value_not_unique_l2726_272633


namespace y_greater_than_one_l2726_272685

theorem y_greater_than_one (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 := by
  sorry

end y_greater_than_one_l2726_272685


namespace clearance_savings_l2726_272624

def coat_price : ℝ := 100
def pants_price : ℝ := 50
def coat_discount : ℝ := 0.30
def pants_discount : ℝ := 0.60

theorem clearance_savings : 
  let total_original := coat_price + pants_price
  let total_savings := coat_price * coat_discount + pants_price * pants_discount
  total_savings / total_original = 0.40 := by sorry

end clearance_savings_l2726_272624


namespace xy_value_l2726_272651

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 15) : x * y = 15 := by
  sorry

end xy_value_l2726_272651


namespace product_xyz_equals_one_l2726_272681

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (eq1 : x + 1/y = 2) 
  (eq2 : y + 1/z = 3) 
  (eq3 : z + 1/x = 4) : 
  x * y * z = 1 := by
sorry

end product_xyz_equals_one_l2726_272681


namespace f_at_three_l2726_272636

/-- The polynomial function f(x) = 9x^4 + 7x^3 - 5x^2 + 3x - 6 -/
def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

/-- Theorem stating that f(3) = 876 -/
theorem f_at_three : f 3 = 876 := by
  sorry

end f_at_three_l2726_272636


namespace range_of_dot_product_trajectory_of_P_l2726_272643

noncomputable section

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 / 2 - y^2 / 3 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define a point M on the right branch of C
def M (x y : ℝ) : Prop := C x y ∧ x ≥ Real.sqrt 2

-- Define the dot product of vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Theorem 1: Range of OM · F₁M
theorem range_of_dot_product (x y : ℝ) :
  M x y → dot_product (x, y) (x + F₁.1, y + F₁.2) ≥ 2 + Real.sqrt 10 := by sorry

-- Define a point P with constant sum of distances from F₁ and F₂
def P (x y : ℝ) : Prop :=
  ∃ (k : ℝ), Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
             Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = k

-- Define the cosine of angle F₁PF₂
def cos_F₁PF₂ (x y : ℝ) : ℝ :=
  let d₁ := Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2)
  let d₂ := Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2)
  ((x - F₁.1) * (x - F₂.1) + (y - F₁.2) * (y - F₂.2)) / (d₁ * d₂)

-- Theorem 2: Trajectory of P
theorem trajectory_of_P (x y : ℝ) :
  P x y ∧ (∀ (u v : ℝ), P u v → cos_F₁PF₂ x y ≤ cos_F₁PF₂ u v) ∧ cos_F₁PF₂ x y = -1/9
  → x^2/9 + y^2/4 = 1 := by sorry

end range_of_dot_product_trajectory_of_P_l2726_272643


namespace right_triangle_equality_l2726_272629

/-- In a right triangle ABC with point M on the hypotenuse, if BM + MA = BC + CA,
    MB = x, CB = 2h, and CA = d, then x = hd / (2h + d) -/
theorem right_triangle_equality (h d x : ℝ) :
  h > 0 → d > 0 →
  x > 0 →
  x + Real.sqrt ((x + 2*h)^2 + d^2) = 2*h + d →
  x = h * d / (2*h + d) := by
  sorry

end right_triangle_equality_l2726_272629


namespace garden_flower_ratio_l2726_272680

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  red : ℕ
  orange : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- The conditions of the garden problem -/
def gardenConditions (f : FlowerCounts) : Prop :=
  f.red + f.orange + f.yellow + f.pink + f.purple = 105 ∧
  f.orange = 10 ∧
  f.yellow = f.red - 5 ∧
  f.pink = f.purple ∧
  f.pink + f.purple = 30

theorem garden_flower_ratio : 
  ∀ f : FlowerCounts, gardenConditions f → (f.red : ℚ) / f.orange = 7 / 2 := by
  sorry

end garden_flower_ratio_l2726_272680


namespace canoe_upstream_speed_l2726_272671

/-- Proves that the upstream speed of a canoe is 9 km/hr given its downstream speed and the stream speed -/
theorem canoe_upstream_speed 
  (downstream_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : downstream_speed = 12) 
  (h2 : stream_speed = 1.5) : 
  downstream_speed - 2 * stream_speed = 9 := by
  sorry

end canoe_upstream_speed_l2726_272671


namespace range_of_a_perpendicular_case_l2726_272611

-- Define the line and hyperbola
def line (a : ℝ) (x : ℝ) : ℝ := a * x + 1
def hyperbola (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

-- Define the intersection condition
def intersects (a : ℝ) : Prop := ∃ x y, hyperbola x y ∧ y = line a x

-- Define the range of a
def valid_range (a : ℝ) : Prop := -Real.sqrt 6 < a ∧ a < Real.sqrt 6 ∧ a ≠ Real.sqrt 3 ∧ a ≠ -Real.sqrt 3

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂, 
  hyperbola x₁ y₁ ∧ y₁ = line a x₁ ∧
  hyperbola x₂ y₂ ∧ y₂ = line a x₂ ∧
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Range of a
theorem range_of_a : ∀ a : ℝ, intersects a ↔ valid_range a :=
sorry

-- Theorem 2: Perpendicular case
theorem perpendicular_case : ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -1) :=
sorry

end range_of_a_perpendicular_case_l2726_272611


namespace sin_960_degrees_l2726_272679

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_960_degrees_l2726_272679


namespace range_of_a_proof_l2726_272661

/-- Proposition p: there exists a real x₀ such that x₀² + 2ax₀ - 2a = 0 -/
def p (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ - 2*a = 0

/-- Proposition q: for all real x, ax² + 4x + a > -2x² + 1 -/
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + 4*x + a > -2*x^2 + 1

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := {a : ℝ | a ≤ -2}

theorem range_of_a_proof (h1 : ∀ a : ℝ, p a ∨ q a) (h2 : ∀ a : ℝ, ¬(p a ∧ q a)) :
  ∀ a : ℝ, a ∈ range_of_a ↔ (p a ∧ ¬q a) :=
sorry

end range_of_a_proof_l2726_272661


namespace dice_probability_l2726_272612

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice re-rolled -/
def numReRolled : ℕ := 3

/-- The probability of a single re-rolled die matching the set-aside pair -/
def probSingleMatch : ℚ := 1 / numSides

/-- The probability of all re-rolled dice matching the set-aside pair -/
def probAllMatch : ℚ := probSingleMatch ^ numReRolled

theorem dice_probability : probAllMatch = 1 / 216 := by
  sorry

end dice_probability_l2726_272612


namespace f_2019_l2726_272607

def f : ℕ → ℕ
| x => if x ≤ 2015 then x + 2 else f (x - 5)

theorem f_2019 : f 2019 = 2016 := by
  sorry

end f_2019_l2726_272607


namespace tan_double_angle_special_case_l2726_272603

theorem tan_double_angle_special_case (θ : Real) :
  (∃ (x y : Real), y = (1/2) * x ∧ x ≥ 0 ∧ y ≥ 0 ∧ Real.tan θ = y / x) →
  Real.tan (2 * θ) = 4/3 := by
  sorry

end tan_double_angle_special_case_l2726_272603


namespace split_99_into_four_numbers_l2726_272606

theorem split_99_into_four_numbers : ∃ (a b c d : ℚ),
  a + b + c + d = 99 ∧
  a + 2 = b - 2 ∧
  a + 2 = 2 * c ∧
  a + 2 = d / 2 ∧
  a = 20 ∧ b = 24 ∧ c = 11 ∧ d = 44 := by
  sorry

end split_99_into_four_numbers_l2726_272606


namespace least_number_of_grapes_l2726_272663

theorem least_number_of_grapes (n : ℕ) : 
  (n > 0) → 
  (n % 3 = 1) → 
  (n % 5 = 1) → 
  (n % 7 = 1) → 
  (∀ m : ℕ, m > 0 → m % 3 = 1 → m % 5 = 1 → m % 7 = 1 → m ≥ n) → 
  n = 106 := by
  sorry

end least_number_of_grapes_l2726_272663


namespace investment_problem_l2726_272689

/-- Prove the existence and uniqueness of the investment amount and interest rate -/
theorem investment_problem :
  ∃! (P y : ℝ), P > 0 ∧ y > 0 ∧
    P * y * 2 / 100 = 800 ∧
    P * ((1 + y / 100)^2 - 1) = 820 ∧
    P = 8000 := by
  sorry

end investment_problem_l2726_272689


namespace fraction_sum_theorem_l2726_272676

theorem fraction_sum_theorem : (1/2 : ℚ) * (1/3 : ℚ) + (1/3 : ℚ) * (1/4 : ℚ) + (1/4 : ℚ) * (1/5 : ℚ) = (3/10 : ℚ) := by
  sorry

end fraction_sum_theorem_l2726_272676


namespace min_value_ab_l2726_272686

/-- Given that ab > 0 and points A(a,0), B(0,b), and C(-2,-2) are collinear, 
    the minimum value of ab is 16 -/
theorem min_value_ab (a b : ℝ) (hab : a * b > 0) 
    (hcollinear : (0 - a) * (b + 2) = (b - 0) * (0 + 2)) : 
  ∀ x y : ℝ, x * y > 0 ∧ 
    (0 - x) * (y + 2) = (y - 0) * (0 + 2) → 
    a * b ≤ x * y ∧ a * b = 16 := by
  sorry

end min_value_ab_l2726_272686


namespace ratio_sum_equation_solver_l2726_272666

theorem ratio_sum_equation_solver (x y z a : ℚ) : 
  (∃ k : ℚ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * a - 5 →
  x + y + z = 70 →
  a = 5 / 3 := by
sorry

end ratio_sum_equation_solver_l2726_272666


namespace scientific_notation_218_million_l2726_272692

theorem scientific_notation_218_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    218000000 = a * (10 : ℝ) ^ n ∧
    a = 2.18 ∧ n = 8 := by
  sorry

end scientific_notation_218_million_l2726_272692


namespace stan_typing_speed_l2726_272634

theorem stan_typing_speed : 
  -- Define constants
  let pages : ℕ := 5
  let words_per_page : ℕ := 400
  let water_per_hour : ℚ := 15
  let water_needed : ℚ := 10
  -- Calculate total words and time
  let total_words : ℕ := pages * words_per_page
  let time_hours : ℚ := water_needed / water_per_hour
  -- Calculate words per minute
  let words_per_minute : ℚ := total_words / (time_hours * 60)
  -- Prove the result
  words_per_minute = 50 := by sorry

end stan_typing_speed_l2726_272634


namespace missing_number_is_eight_l2726_272660

/-- Represents a 1 × 12 table filled with numbers -/
def Table := Fin 12 → ℝ

/-- The sum of any four adjacent cells in the table is 11 -/
def SumAdjacent (t : Table) : Prop :=
  ∀ i : Fin 9, t i + t (i + 1) + t (i + 2) + t (i + 3) = 11

/-- The table contains the known numbers 4, 1, and 2 -/
def ContainsKnownNumbers (t : Table) : Prop :=
  ∃ (i j k : Fin 12), t i = 4 ∧ t j = 1 ∧ t k = 2

/-- The theorem to be proved -/
theorem missing_number_is_eight
  (t : Table)
  (h1 : SumAdjacent t)
  (h2 : ContainsKnownNumbers t) :
  ∃ (l : Fin 12), t l = 8 :=
sorry

end missing_number_is_eight_l2726_272660


namespace quadratic_is_square_of_binomial_l2726_272695

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℝ), (225/16 : ℝ) * x^2 + 15 * x + 4 = (r * x + s)^2 := by
  sorry

end quadratic_is_square_of_binomial_l2726_272695


namespace cost_price_percentage_l2726_272600

theorem cost_price_percentage (C S : ℝ) (h : C > 0) (h' : S > 0) :
  (S - C) / C = 3 → C / S = 1 / 4 := by
  sorry

end cost_price_percentage_l2726_272600


namespace book_cost_l2726_272690

/-- The cost of a book given partial payment and a condition on the remaining amount -/
theorem book_cost (paid : ℝ) (total_cost : ℝ) : 
  paid = 100 →
  (total_cost - paid) = (total_cost - (total_cost - paid)) →
  total_cost = 200 := by
  sorry

end book_cost_l2726_272690


namespace customer_departure_l2726_272648

theorem customer_departure (initial_customers : Real) 
  (second_departure : Real) (final_customers : Real) :
  initial_customers = 36.0 →
  second_departure = 14.0 →
  final_customers = 3 →
  ∃ (first_departure : Real),
    initial_customers - first_departure - second_departure = final_customers ∧
    first_departure = 19.0 := by
  sorry

end customer_departure_l2726_272648


namespace simplify_expression_1_simplify_expression_2_l2726_272674

-- Define variables
variable (a b x y : ℝ)

-- Theorem 1
theorem simplify_expression_1 : 
  6*a + 7*b^2 - 9 + 4*a - b^2 + 6 = 6*b^2 + 10*a - 3 := by sorry

-- Theorem 2
theorem simplify_expression_2 :
  5*x - 2*(4*x + 5*y) + 3*(3*x - 4*y) = 6*x - 22*y := by sorry

end simplify_expression_1_simplify_expression_2_l2726_272674


namespace quadratic_roots_bounds_l2726_272688

theorem quadratic_roots_bounds (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m < 0)
  (hroots : x₁^2 - x₁ - 6 = m ∧ x₂^2 - x₂ - 6 = m)
  (horder : x₁ < x₂) :
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by sorry

end quadratic_roots_bounds_l2726_272688


namespace ratio_equality_l2726_272657

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := by
  sorry

end ratio_equality_l2726_272657


namespace girls_in_class_l2726_272632

theorem girls_in_class (total : ℕ) (difference : ℕ) : 
  total = 63 → difference = 7 → (total + difference) / 2 = 35 := by
  sorry

end girls_in_class_l2726_272632


namespace sum_of_divisors_of_420_l2726_272605

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

-- State the theorem
theorem sum_of_divisors_of_420 : sumOfDivisors 420 = 1344 := by
  sorry

end sum_of_divisors_of_420_l2726_272605


namespace inequality_proof_l2726_272653

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l2726_272653


namespace mikes_house_payments_l2726_272673

theorem mikes_house_payments (lower_rate higher_rate total_payments num_lower_payments num_higher_payments : ℚ) :
  higher_rate = 310 →
  total_payments = 3615 →
  num_lower_payments = 5 →
  num_higher_payments = 7 →
  num_lower_payments + num_higher_payments = 12 →
  num_lower_payments * lower_rate + num_higher_payments * higher_rate = total_payments →
  lower_rate = 289 := by
sorry

end mikes_house_payments_l2726_272673


namespace largest_divisor_of_n_fourth_minus_n_l2726_272639

theorem largest_divisor_of_n_fourth_minus_n (n : ℤ) (h : 4 ∣ n) :
  (∃ k : ℤ, n^4 - n = 4 * k) ∧ 
  (∀ m : ℤ, m > 4 → ¬(∀ n : ℤ, 4 ∣ n → m ∣ (n^4 - n))) :=
sorry

end largest_divisor_of_n_fourth_minus_n_l2726_272639


namespace total_cost_theorem_l2726_272696

/-- The cost of a single shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a single trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a single tie -/
def tie_cost : ℝ := sorry

/-- The total cost of 7 shirts, 2 trousers, and 2 ties is $50 -/
axiom condition1 : 7 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50

/-- The total cost of 3 trousers, 5 shirts, and 2 ties is $70 -/
axiom condition2 : 5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 70

/-- The theorem to be proved -/
theorem total_cost_theorem : 
  3 * shirt_cost + 4 * trouser_cost + 2 * tie_cost = 90 := by sorry

end total_cost_theorem_l2726_272696


namespace binary_digit_difference_l2726_272672

/-- Returns the number of digits in the base-2 representation of a natural number -/
def numDigitsBinary (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

/-- The difference in the number of binary digits between 800 and 250 is 2 -/
theorem binary_digit_difference : numDigitsBinary 800 - numDigitsBinary 250 = 2 := by
  sorry

end binary_digit_difference_l2726_272672


namespace ln_sufficient_not_necessary_l2726_272645

-- Define the statement that ln a > ln b implies e^a > e^b
def ln_implies_exp (a b : ℝ) : Prop :=
  (∀ a b : ℝ, Real.log a > Real.log b → Real.exp a > Real.exp b)

-- Define the statement that e^a > e^b does not always imply ln a > ln b
def exp_not_always_implies_ln (a b : ℝ) : Prop :=
  (∃ a b : ℝ, Real.exp a > Real.exp b ∧ ¬(Real.log a > Real.log b))

-- Theorem stating that ln a > ln b is sufficient but not necessary for e^a > e^b
theorem ln_sufficient_not_necessary :
  (∀ a b : ℝ, ln_implies_exp a b) ∧ (∃ a b : ℝ, exp_not_always_implies_ln a b) :=
sorry

end ln_sufficient_not_necessary_l2726_272645


namespace multiplication_after_division_l2726_272623

theorem multiplication_after_division (x y : ℝ) : x = 6 → (x / 6) * y = 12 → y = 12 := by
  sorry

end multiplication_after_division_l2726_272623


namespace lines_2_3_parallel_l2726_272614

-- Define the slopes of the lines
def slope1 : ℚ := 3 / 4
def slope2 : ℚ := -3 / 4
def slope3 : ℚ := -3 / 4
def slope4 : ℚ := -4 / 3

-- Define the equations of the lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 16
def line2 (x y : ℚ) : Prop := -3 * x - 4 * y = 15
def line3 (x y : ℚ) : Prop := 4 * y + 3 * x = 16
def line4 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Theorem: Lines 2 and 3 are parallel
theorem lines_2_3_parallel : 
  ∀ (x1 y1 x2 y2 : ℚ), 
    line2 x1 y1 → line3 x2 y2 → 
    slope2 = slope3 ∧ slope2 ≠ slope1 ∧ slope2 ≠ slope4 := by
  sorry

end lines_2_3_parallel_l2726_272614


namespace three_unique_circles_l2726_272664

/-- A square with vertices P, Q, R, and S -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- A circle defined by two points as its diameter endpoints -/
structure Circle where
  endpoint1 : Point
  endpoint2 : Point

/-- Function to count unique circles defined by square vertices -/
def count_unique_circles (s : Square) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 3 unique circles -/
theorem three_unique_circles (s : Square) : 
  count_unique_circles s = 3 := by
  sorry

end three_unique_circles_l2726_272664


namespace six_good_points_l2726_272677

/-- A lattice point on a 9x9 grid -/
structure LatticePoint where
  x : Fin 9
  y : Fin 9

/-- A triangle defined by three lattice points -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Calculates the area of a triangle given three lattice points -/
def triangleArea (P Q R : LatticePoint) : ℚ :=
  sorry

/-- Checks if a point is a "good point" for a given triangle -/
def isGoodPoint (T : Triangle) (P : LatticePoint) : Prop :=
  triangleArea P T.A T.B = triangleArea P T.A T.C

/-- The main theorem stating that there are exactly 6 "good points" -/
theorem six_good_points (T : Triangle) : 
  ∃! (goodPoints : Finset LatticePoint), 
    (∀ P ∈ goodPoints, isGoodPoint T P) ∧ 
    goodPoints.card = 6 :=
  sorry

end six_good_points_l2726_272677


namespace range_of_k_min_value_when_k_4_min_value_of_reciprocal_sum_l2726_272647

-- Define the function f
def f (x k : ℝ) : ℝ := |2*x - 1| + |2*x - k|

-- Part 1
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f x k ≥ 1) ↔ k ≤ 0 ∨ k ≥ 2 :=
sorry

-- Part 2
theorem min_value_when_k_4 :
  ∃ m : ℝ, m = 3 ∧ (∀ x : ℝ, f x 4 ≥ m) :=
sorry

-- Part 3
theorem min_value_of_reciprocal_sum (a b : ℝ) :
  a > 0 → b > 0 → a + 4*b = 3 →
  1/a + 1/b ≥ 3 :=
sorry

end range_of_k_min_value_when_k_4_min_value_of_reciprocal_sum_l2726_272647


namespace fraction_division_equality_l2726_272616

theorem fraction_division_equality : (2 / 5) / (3 / 7) = 14 / 15 := by
  sorry

end fraction_division_equality_l2726_272616


namespace complement_A_union_B_eq_univ_A_inter_B_ne_B_l2726_272659

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

-- Statement for part 1
theorem complement_A_union_B_eq_univ (a : ℝ) :
  (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 :=
sorry

-- Statement for part 2
theorem A_inter_B_ne_B (a : ℝ) :
  A ∩ B a ≠ B a ↔ a < 1/2 :=
sorry

end complement_A_union_B_eq_univ_A_inter_B_ne_B_l2726_272659


namespace unique_four_digit_number_l2726_272670

theorem unique_four_digit_number :
  ∃! (a b c d : ℕ), 
    0 < a ∧ a < 10 ∧
    0 ≤ b ∧ b < 10 ∧
    0 ≤ c ∧ c < 10 ∧
    0 ≤ d ∧ d < 10 ∧
    a + b + c + d = 10 * a + b ∧
    a * b * c * d = 10 * c + d :=
by sorry

end unique_four_digit_number_l2726_272670


namespace power_of_two_equality_l2726_272617

theorem power_of_two_equality (K : ℕ) : 32^5 * 64^2 = 2^K → K = 37 := by
  have h1 : 32 = 2^5 := by sorry
  have h2 : 64 = 2^6 := by sorry
  sorry

end power_of_two_equality_l2726_272617


namespace cube_root_simplification_l2726_272652

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 + 60^3 : ℝ)^(1/3) = 10 * 315^(1/3) :=
by sorry

end cube_root_simplification_l2726_272652


namespace arctan_sum_three_seven_l2726_272630

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end arctan_sum_three_seven_l2726_272630


namespace concession_stand_soda_cost_l2726_272618

/-- Proves that the cost of each soda is $0.50 given the conditions of the concession stand problem -/
theorem concession_stand_soda_cost 
  (total_revenue : ℝ)
  (total_items : ℕ)
  (hot_dogs_sold : ℕ)
  (hot_dog_cost : ℝ)
  (h1 : total_revenue = 78.50)
  (h2 : total_items = 87)
  (h3 : hot_dogs_sold = 35)
  (h4 : hot_dog_cost = 1.50) :
  let soda_cost := (total_revenue - hot_dogs_sold * hot_dog_cost) / (total_items - hot_dogs_sold)
  soda_cost = 0.50 := by
    sorry

#check concession_stand_soda_cost

end concession_stand_soda_cost_l2726_272618


namespace odd_function_negative_l2726_272635

/-- An odd function f with a specific definition for non-negative x -/
def odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x ≥ 0, f x = x * (1 - x))

/-- Theorem stating the form of f(x) for non-positive x -/
theorem odd_function_negative (f : ℝ → ℝ) (h : odd_function f) :
  ∀ x ≤ 0, f x = x * (1 + x) := by
  sorry

end odd_function_negative_l2726_272635


namespace factory_earnings_l2726_272608

/-- Represents a factory with machines producing material -/
structure Factory where
  machines_23h : ℕ  -- Number of machines working 23 hours
  machines_12h : ℕ  -- Number of machines working 12 hours
  production_rate : ℝ  -- Production rate in kg per hour per machine
  price_per_kg : ℝ  -- Selling price per kg of material

/-- Calculates the daily earnings of the factory -/
def daily_earnings (f : Factory) : ℝ :=
  (f.machines_23h * 23 + f.machines_12h * 12) * f.production_rate * f.price_per_kg

/-- Theorem stating that the factory's daily earnings are $8100 -/
theorem factory_earnings :
  let f : Factory := {
    machines_23h := 3,
    machines_12h := 1,
    production_rate := 2,
    price_per_kg := 50
  }
  daily_earnings f = 8100 := by sorry

end factory_earnings_l2726_272608


namespace product_is_term_iff_first_term_is_power_of_ratio_l2726_272698

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  a : ℝ  -- First term
  q : ℝ  -- Common ratio

/-- The nth term of a geometric progression -/
def GeometricProgression.nthTerm (gp : GeometricProgression) (n : ℕ) : ℝ :=
  gp.a * gp.q ^ n

/-- Condition for product of terms to be another term -/
def productIsTermCondition (gp : GeometricProgression) : Prop :=
  ∃ m : ℤ, gp.a = gp.q ^ m

theorem product_is_term_iff_first_term_is_power_of_ratio (gp : GeometricProgression) :
  (∀ n p k : ℕ, ∃ k : ℕ, gp.nthTerm n * gp.nthTerm p = gp.nthTerm k) ↔
  productIsTermCondition gp :=
sorry

end product_is_term_iff_first_term_is_power_of_ratio_l2726_272698


namespace afternoon_to_morning_ratio_is_two_to_one_l2726_272622

/-- Represents the sales of pears by a salesman in a day -/
structure PearSales where
  total : ℕ
  morning : ℕ
  afternoon : ℕ

/-- Theorem stating that the ratio of afternoon to morning pear sales is 2:1 -/
theorem afternoon_to_morning_ratio_is_two_to_one (sales : PearSales)
  (h_total : sales.total = 360)
  (h_morning : sales.morning = 120)
  (h_afternoon : sales.afternoon = 240) :
  sales.afternoon / sales.morning = 2 := by
  sorry

#check afternoon_to_morning_ratio_is_two_to_one

end afternoon_to_morning_ratio_is_two_to_one_l2726_272622


namespace pyramid_ball_count_l2726_272675

/-- The number of layers in the pyramid -/
def n : ℕ := 13

/-- The number of balls in the top layer -/
def first_term : ℕ := 4

/-- The number of balls in the bottom layer -/
def last_term : ℕ := 40

/-- The sum of the arithmetic sequence representing the number of balls in each layer -/
def sum_of_sequence : ℕ := n * (first_term + last_term) / 2

theorem pyramid_ball_count :
  sum_of_sequence = 286 := by sorry

end pyramid_ball_count_l2726_272675


namespace lcm_gcf_product_24_60_l2726_272644

theorem lcm_gcf_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end lcm_gcf_product_24_60_l2726_272644


namespace sanya_towels_per_wash_l2726_272658

/-- The number of bath towels Sanya can wash in one wash -/
def towels_per_wash : ℕ := sorry

/-- The number of hours Sanya has per day for washing -/
def hours_per_day : ℕ := 2

/-- The total number of bath towels Sanya has -/
def total_towels : ℕ := 98

/-- The number of days it takes to wash all towels -/
def days_to_wash_all : ℕ := 7

/-- Theorem stating that Sanya can wash 7 towels in one wash -/
theorem sanya_towels_per_wash :
  towels_per_wash = 7 := by sorry

end sanya_towels_per_wash_l2726_272658


namespace lilia_initial_peaches_l2726_272621

/-- The number of peaches Lilia sold to friends -/
def peaches_sold_to_friends : ℕ := 10

/-- The price of each peach sold to friends -/
def price_for_friends : ℚ := 2

/-- The number of peaches Lilia sold to relatives -/
def peaches_sold_to_relatives : ℕ := 4

/-- The price of each peach sold to relatives -/
def price_for_relatives : ℚ := 5/4

/-- The number of peaches Lilia kept for herself -/
def peaches_kept : ℕ := 1

/-- The total amount of money Lilia earned -/
def total_earned : ℚ := 25

/-- The initial number of peaches Lilia had -/
def initial_peaches : ℕ := peaches_sold_to_friends + peaches_sold_to_relatives + peaches_kept

theorem lilia_initial_peaches :
  initial_peaches = 15 ∧
  total_earned = peaches_sold_to_friends * price_for_friends + peaches_sold_to_relatives * price_for_relatives :=
by sorry

end lilia_initial_peaches_l2726_272621


namespace root_sum_transformation_l2726_272683

theorem root_sum_transformation (α β γ : ℂ) : 
  (x^3 - x + 1 = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 := by
  sorry

end root_sum_transformation_l2726_272683


namespace greatest_valid_sequence_length_l2726_272649

/-- A sequence of distinct positive integers satisfying the given condition -/
def ValidSequence (s : Nat → Nat) (n : Nat) : Prop :=
  (∀ i j, i < n → j < n → i ≠ j → s i ≠ s j) ∧ 
  (∀ i, i < n - 1 → (s i) ^ (s (i + 1)) = (s (i + 1)) ^ (s (i + 2)))

/-- The theorem stating that 5 is the greatest positive integer satisfying the condition -/
theorem greatest_valid_sequence_length : 
  (∃ s : Nat → Nat, ValidSequence s 5) ∧ 
  (∀ n : Nat, n > 5 → ¬∃ s : Nat → Nat, ValidSequence s n) :=
sorry

end greatest_valid_sequence_length_l2726_272649


namespace exists_composite_evaluation_l2726_272662

/-- A polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Evaluate a polynomial at a given integer -/
def evalPoly (p : IntPolynomial) (x : Int) : Int :=
  p.foldr (fun a b => a + x * b) 0

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : Int) : Prop :=
  ∃ m, 1 < m ∧ m < n.natAbs ∧ n % m = 0

theorem exists_composite_evaluation (polys : List IntPolynomial) :
  ∃ a : Int, ∀ p ∈ polys, isComposite (evalPoly p a) := by
  sorry

#check exists_composite_evaluation

end exists_composite_evaluation_l2726_272662


namespace overtime_hours_calculation_l2726_272699

theorem overtime_hours_calculation 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_pay : ℝ) 
  (h1 : regular_rate = 3)
  (h2 : regular_hours = 40)
  (h3 : total_pay = 192) :
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate = 12 := by
sorry

end overtime_hours_calculation_l2726_272699


namespace tens_digit_of_nine_power_2010_l2726_272615

def last_two_digits (n : ℕ) : ℕ := n % 100

def cycle_of_nine : List ℕ := [09, 81, 29, 61, 49, 41, 69, 21, 89, 01]

theorem tens_digit_of_nine_power_2010 :
  (last_two_digits (9^2010)) / 10 = 0 :=
sorry

end tens_digit_of_nine_power_2010_l2726_272615


namespace difference_of_squares_fifty_thirty_l2726_272697

theorem difference_of_squares_fifty_thirty : 50^2 - 30^2 = 1600 := by
  sorry

end difference_of_squares_fifty_thirty_l2726_272697


namespace nested_fraction_equality_l2726_272609

theorem nested_fraction_equality : (1 / (1 + 1 / (4 + 1 / 5))) = 21 / 26 := by
  sorry

end nested_fraction_equality_l2726_272609


namespace counterexample_exists_l2726_272654

def is_in_set (n : ℕ) : Prop := n = 14 ∨ n = 18 ∨ n = 20 ∨ n = 24 ∨ n = 30

theorem counterexample_exists : 
  ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 2)) ∧ is_in_set n :=
sorry

end counterexample_exists_l2726_272654


namespace amusement_park_theorem_l2726_272613

/-- Represents the amusement park scenario with two roller coasters and a group of friends. -/
structure AmusementPark where
  friends : ℕ
  first_coaster_cost : ℕ
  second_coaster_cost : ℕ
  first_coaster_rides : ℕ
  second_coaster_rides : ℕ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the total number of tickets needed for the group. -/
def total_tickets (park : AmusementPark) : ℕ :=
  park.friends * (park.first_coaster_cost * park.first_coaster_rides + 
                  park.second_coaster_cost * park.second_coaster_rides)

/-- Calculates the cost difference between non-discounted and discounted tickets. -/
def cost_difference (park : AmusementPark) : ℚ :=
  let total_cost := total_tickets park
  if total_tickets park ≥ park.discount_threshold then
    (total_cost : ℚ) * park.discount_rate
  else
    0

/-- Theorem stating the correct number of tickets and cost difference for the given scenario. -/
theorem amusement_park_theorem (park : AmusementPark) 
  (h1 : park.friends = 8)
  (h2 : park.first_coaster_cost = 6)
  (h3 : park.second_coaster_cost = 8)
  (h4 : park.first_coaster_rides = 2)
  (h5 : park.second_coaster_rides = 1)
  (h6 : park.discount_rate = 15 / 100)
  (h7 : park.discount_threshold = 10) :
  total_tickets park = 160 ∧ cost_difference park = 24 := by
  sorry

end amusement_park_theorem_l2726_272613


namespace tims_children_treats_l2726_272678

/-- The total number of treats Tim's children get while trick or treating --/
def total_treats (num_children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_kid : ℕ) : ℕ :=
  num_children * hours * houses_per_hour * treats_per_kid

/-- Theorem stating that Tim's children get 180 treats in total --/
theorem tims_children_treats :
  total_treats 3 4 5 3 = 180 := by
  sorry

#eval total_treats 3 4 5 3

end tims_children_treats_l2726_272678


namespace waiter_remaining_customers_l2726_272693

/-- Calculates the number of remaining customers after some customers leave. -/
def remainingCustomers (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem waiter_remaining_customers :
  remainingCustomers 21 9 = 12 := by
  sorry

end waiter_remaining_customers_l2726_272693
