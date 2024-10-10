import Mathlib

namespace quadratic_minimum_l1494_149434

theorem quadratic_minimum (x : ℝ) (h : x > 0) : x^2 - 2*x + 3 ≥ 2 := by
  sorry

end quadratic_minimum_l1494_149434


namespace opposite_of_negative_three_fourths_l1494_149428

theorem opposite_of_negative_three_fourths :
  let x : ℚ := -3/4
  let y : ℚ := 3/4
  (∀ z : ℚ, z + x = 0 ↔ z = y) :=
by sorry

end opposite_of_negative_three_fourths_l1494_149428


namespace fencing_cost_is_5300_l1494_149429

/-- Calculates the total cost of fencing a rectangular plot -/
def totalFencingCost (length width fenceCostPerMeter : ℝ) : ℝ :=
  2 * (length + width) * fenceCostPerMeter

/-- Theorem: The total cost of fencing the given rectangular plot is $5300 -/
theorem fencing_cost_is_5300 :
  let length : ℝ := 70
  let width : ℝ := 30
  let fenceCostPerMeter : ℝ := 26.50
  totalFencingCost length width fenceCostPerMeter = 5300 := by
  sorry

end fencing_cost_is_5300_l1494_149429


namespace triangle_mass_l1494_149469

-- Define the shapes
variable (Square Circle Triangle : ℝ)

-- Define the scale equations
axiom scale1 : Square + Circle = 8
axiom scale2 : Square + 2 * Circle = 11
axiom scale3 : Circle + 2 * Triangle = 15

-- Theorem to prove
theorem triangle_mass : Triangle = 6 := by
  sorry

end triangle_mass_l1494_149469


namespace at_most_one_root_l1494_149499

theorem at_most_one_root {f : ℝ → ℝ} (h : ∀ a b, a < b → f a < f b) :
  ∃! x, f x = 0 ∨ ∀ x, f x ≠ 0 :=
sorry

end at_most_one_root_l1494_149499


namespace sin_cos_square_identity_l1494_149492

theorem sin_cos_square_identity (α : ℝ) : (Real.sin α + Real.cos α)^2 = 1 + Real.sin (2 * α) := by
  sorry

end sin_cos_square_identity_l1494_149492


namespace softball_team_composition_l1494_149418

theorem softball_team_composition (total : ℕ) (ratio : ℚ) : 
  total = 14 → ratio = 5/9 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 4 := by
  sorry

end softball_team_composition_l1494_149418


namespace department_age_analysis_l1494_149488

/-- Represents the age data for a department -/
def DepartmentData := List Nat

/-- Calculate the mode of a list of numbers -/
def mode (data : DepartmentData) : Nat :=
  sorry

/-- Calculate the median of a list of numbers -/
def median (data : DepartmentData) : Nat :=
  sorry

/-- Calculate the average of a list of numbers -/
def average (data : DepartmentData) : Rat :=
  sorry

theorem department_age_analysis 
  (dept_A dept_B : DepartmentData)
  (h1 : dept_A.length = 10)
  (h2 : dept_B.length = 10)
  (h3 : dept_A = [21, 23, 25, 26, 27, 28, 30, 32, 32, 32])
  (h4 : dept_B = [20, 22, 24, 24, 26, 28, 28, 30, 34, 40]) :
  (mode dept_A = 32) ∧ 
  (median dept_B = 26) ∧ 
  (average dept_A < average dept_B) :=
sorry

end department_age_analysis_l1494_149488


namespace game_probability_l1494_149409

/-- The probability of a specific outcome in a game with 8 rounds -/
theorem game_probability : 
  -- Total number of rounds
  (total_rounds : ℕ) →
  -- Alex's probability of winning a round
  (alex_prob : ℚ) →
  -- Mel's probability of winning a round
  (mel_prob : ℚ) →
  -- Chelsea's probability of winning a round
  (chelsea_prob : ℚ) →
  -- Number of rounds Alex wins
  (alex_wins : ℕ) →
  -- Number of rounds Mel wins
  (mel_wins : ℕ) →
  -- Number of rounds Chelsea wins
  (chelsea_wins : ℕ) →
  -- Conditions
  total_rounds = 8 →
  alex_prob = 2/5 →
  mel_prob = 3 * chelsea_prob →
  alex_prob + mel_prob + chelsea_prob = 1 →
  alex_wins + mel_wins + chelsea_wins = total_rounds →
  alex_wins = 3 →
  mel_wins = 4 →
  chelsea_wins = 1 →
  -- Conclusion
  (Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins *
   alex_prob ^ alex_wins * mel_prob ^ mel_wins * chelsea_prob ^ chelsea_wins : ℚ) = 881/1000 := by
sorry

end game_probability_l1494_149409


namespace problem_statement_l1494_149411

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := Real.log x - x + 2

theorem problem_statement :
  (∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 1) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ≥ 1 → m * f x ≥ (x - 1) / (x + 1)) ↔ m ≥ 1/2) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi/2 →
    ((0 < α ∧ α < Real.pi/4 → f (Real.tan α) < -Real.cos (2*α)) ∧
     (α = Real.pi/4 → f (Real.tan α) = -Real.cos (2*α)) ∧
     (Real.pi/4 < α ∧ α < Real.pi/2 → f (Real.tan α) > -Real.cos (2*α)))) := by
  sorry

end problem_statement_l1494_149411


namespace binomial_coefficient_x_plus_two_to_seven_coefficient_of_x_fifth_power_l1494_149481

theorem binomial_coefficient_x_plus_two_to_seven (x : ℝ) : 
  (Finset.range 8).sum (λ k => (Nat.choose 7 k : ℝ) * x^k * 2^(7-k)) = 
    x^7 + 14*x^6 + 84*x^5 + 280*x^4 + 560*x^3 + 672*x^2 + 448*x + 128 :=
by sorry

theorem coefficient_of_x_fifth_power : 
  (Finset.range 8).sum (λ k => (Nat.choose 7 k : ℝ) * 1^k * 2^(7-k) * 
    (if k = 5 then 1 else 0)) = 84 :=
by sorry

end binomial_coefficient_x_plus_two_to_seven_coefficient_of_x_fifth_power_l1494_149481


namespace base_eight_47_equals_39_l1494_149407

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-eight number 47 is equal to the base-ten number 39 -/
theorem base_eight_47_equals_39 : base_eight_to_ten 4 7 = 39 := by
  sorry

end base_eight_47_equals_39_l1494_149407


namespace rope_division_l1494_149495

def rope_length : ℝ := 3
def num_segments : ℕ := 7

theorem rope_division (segment_fraction : ℝ) (segment_length : ℝ) :
  (segment_fraction = 1 / num_segments) ∧
  (segment_length = rope_length / num_segments) ∧
  (segment_fraction = 1 / 7) ∧
  (segment_length = 3 / 7) := by
  sorry

end rope_division_l1494_149495


namespace tv_price_calculation_l1494_149456

/-- Calculates the final price of an item given the original price, discount rate, tax rate, and rebate amount. -/
def finalPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) (rebate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discountRate)
  let priceWithTax := salePrice * (1 + taxRate)
  priceWithTax - rebate

/-- Theorem stating that the final price of a $1200 item with 30% discount, 8% tax, and $50 rebate is $857.2. -/
theorem tv_price_calculation :
  finalPrice 1200 0.30 0.08 50 = 857.2 := by
  sorry

end tv_price_calculation_l1494_149456


namespace cubic_equation_roots_inequality_l1494_149450

/-- Given a cubic equation x³ + ax² + bx + c = 0 with three real roots p ≤ q ≤ r,
    prove that a² - 3b ≥ 0 and √(a² - 3b) ≤ r - p -/
theorem cubic_equation_roots_inequality (a b c p q r : ℝ) :
  p ≤ q ∧ q ≤ r ∧
  p^3 + a*p^2 + b*p + c = 0 ∧
  q^3 + a*q^2 + b*q + c = 0 ∧
  r^3 + a*r^2 + b*r + c = 0 →
  a^2 - 3*b ≥ 0 ∧ Real.sqrt (a^2 - 3*b) ≤ r - p :=
by sorry

end cubic_equation_roots_inequality_l1494_149450


namespace gnome_distribution_ways_l1494_149496

/-- The number of ways to distribute n identical objects among k recipients,
    with each recipient receiving at least m objects. -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * (m - 1) + k - 1) (k - 1)

/-- The number of gnomes -/
def num_gnomes : ℕ := 3

/-- The total number of stones -/
def total_stones : ℕ := 70

/-- The minimum number of stones each gnome must receive -/
def min_stones : ℕ := 10

theorem gnome_distribution_ways : 
  distribution_ways total_stones num_gnomes min_stones = 946 := by
  sorry

end gnome_distribution_ways_l1494_149496


namespace margaret_score_l1494_149416

theorem margaret_score (average_score : ℝ) (marco_percentage : ℝ) (margaret_difference : ℝ) : 
  average_score = 90 →
  marco_percentage = 0.1 →
  margaret_difference = 5 →
  let marco_score := average_score * (1 - marco_percentage)
  let margaret_score := marco_score + margaret_difference
  margaret_score = 86 := by sorry

end margaret_score_l1494_149416


namespace cone_volume_l1494_149480

/-- Given a cone with base radius 1 and slant height equal to the diameter of the base,
    prove that its volume is (√3 * π) / 3 -/
theorem cone_volume (r : ℝ) (l : ℝ) (h : ℝ) :
  r = 1 →
  l = 2 * r →
  h ^ 2 + r ^ 2 = l ^ 2 →
  (1 / 3) * π * r ^ 2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end cone_volume_l1494_149480


namespace terry_age_proof_l1494_149426

/-- Nora's current age -/
def nora_age : ℕ := 10

/-- Terry's age in 10 years -/
def terry_future_age : ℕ := 4 * nora_age

/-- Terry's current age -/
def terry_current_age : ℕ := terry_future_age - 10

theorem terry_age_proof : terry_current_age = 30 := by
  sorry

end terry_age_proof_l1494_149426


namespace orvin_max_balloons_l1494_149442

/-- Represents the price of a balloon in cents -/
def regularPrice : ℕ := 200

/-- Represents the number of balloons Orvin can afford at regular price -/
def regularAffordable : ℕ := 40

/-- Represents the maximum number of discounted balloons -/
def maxDiscounted : ℕ := 10

/-- Calculates the total money Orvin has in cents -/
def totalMoney : ℕ := regularPrice * regularAffordable

/-- Calculates the price of a discounted balloon in cents -/
def discountedPrice : ℕ := regularPrice / 2

/-- Calculates the cost of buying a regular and a discounted balloon in cents -/
def pairCost : ℕ := regularPrice + discountedPrice

/-- Represents the maximum number of balloons Orvin can buy -/
def maxBalloons : ℕ := 42

theorem orvin_max_balloons :
  regularPrice > 0 →
  (totalMoney - (maxDiscounted / 2 * pairCost)) / regularPrice + maxDiscounted = maxBalloons :=
by sorry

end orvin_max_balloons_l1494_149442


namespace f_36_l1494_149424

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, f (x * y) = f x + f y)
variable (h2 : f 2 = p)
variable (h3 : f 3 = q)

-- State the theorem
theorem f_36 (p q : ℝ) : f 36 = 2 * (p + q) := by sorry

end f_36_l1494_149424


namespace second_chapter_pages_l1494_149417

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ

/-- Properties of the book -/
def book_properties (b : Book) : Prop :=
  b.total_pages = 93 ∧ b.first_chapter_pages = 60 ∧ b.total_pages = b.first_chapter_pages + b.second_chapter_pages

/-- Theorem stating that the second chapter has 33 pages -/
theorem second_chapter_pages (b : Book) (h : book_properties b) : b.second_chapter_pages = 33 := by
  sorry

end second_chapter_pages_l1494_149417


namespace fraction_power_equality_l1494_149430

theorem fraction_power_equality : (125000 : ℝ)^5 / (25000 : ℝ)^5 = 3125 := by sorry

end fraction_power_equality_l1494_149430


namespace exists_cheaper_bulk_purchase_l1494_149422

/-- The original price of a notebook --/
def original_price : ℝ := 8

/-- The discounted price of a notebook when buying more than 100 --/
def discounted_price : ℝ := original_price - 2

/-- The cost of buying n books under Plan 1 (n ≤ 100) --/
def cost_plan1 (n : ℝ) : ℝ := original_price * n

/-- The cost of buying n books under Plan 2 (n > 100) --/
def cost_plan2 (n : ℝ) : ℝ := discounted_price * n

/-- Theorem stating that there exists a scenario where buying n books (n > 100) 
    costs less than buying 80 books under Plan 1 --/
theorem exists_cheaper_bulk_purchase :
  ∃ n : ℝ, n > 100 ∧ cost_plan2 n < cost_plan1 80 := by
  sorry

end exists_cheaper_bulk_purchase_l1494_149422


namespace ratio_to_percentage_difference_l1494_149410

theorem ratio_to_percentage_difference (A B : ℝ) (h : A / B = 3 / 4) :
  (B - A) / B = 1 / 4 := by
  sorry

end ratio_to_percentage_difference_l1494_149410


namespace line_equation_proof_l1494_149452

/-- Given a line defined by (3, -4) · ((x, y) - (2, 7)) = 0, prove that its slope-intercept form y = mx + b has m = 3/4 and b = 11/2 -/
theorem line_equation_proof (x y : ℝ) : 
  (3 * (x - 2) + (-4) * (y - 7) = 0) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 3/4 ∧ b = 11/2) := by
  sorry

end line_equation_proof_l1494_149452


namespace triangle_side_length_range_l1494_149420

theorem triangle_side_length_range (a b c : ℝ) :
  (|a + b - 4| + (a - b + 2)^2 = 0) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (2 < c ∧ c < 4) :=
by sorry

end triangle_side_length_range_l1494_149420


namespace quadratic_vertex_ordinate_l1494_149438

theorem quadratic_vertex_ordinate 
  (a b c : ℝ) 
  (d : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b^2 - 4*a*c > 0) 
  (h3 : d = (Real.sqrt (b^2 - 4*a*c)) / a) :
  ∃! y : ℝ, y = -a * d^2 / 4 ∧ 
    y = a * (-b / (2*a))^2 + b * (-b / (2*a)) + c :=
sorry

end quadratic_vertex_ordinate_l1494_149438


namespace four_students_two_groups_l1494_149479

/-- The number of different ways to assign n students to 2 groups -/
def signUpMethods (n : ℕ) : ℕ := 2^n

/-- The problem statement -/
theorem four_students_two_groups : 
  signUpMethods 4 = 16 := by
  sorry

end four_students_two_groups_l1494_149479


namespace f_properties_l1494_149464

-- Define the function f(x) = lg|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem f_properties :
  -- f is defined for all real numbers except 0
  (∀ x : ℝ, x ≠ 0 → f x = Real.log (abs x)) →
  -- f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_properties_l1494_149464


namespace quartic_polynomial_satisfies_conditions_l1494_149405

def q (x : ℚ) : ℚ := -1/6 * x^4 + 4/3 * x^3 - 4/3 * x^2 - 8/3 * x

theorem quartic_polynomial_satisfies_conditions :
  q 1 = -3 ∧ q 2 = -5 ∧ q 3 = -9 ∧ q 4 = -17 ∧ q 5 = -35 := by
  sorry

end quartic_polynomial_satisfies_conditions_l1494_149405


namespace intersection_of_A_and_B_l1494_149406

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 - 2*x}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1 : ℝ) (1 : ℝ) := by
  sorry

end intersection_of_A_and_B_l1494_149406


namespace min_faces_prism_min_vertices_pyramid_l1494_149468

/-- A prism is a three-dimensional shape with two identical ends and flat sides. -/
structure Prism where
  base : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  height : ℝ

/-- A pyramid is a three-dimensional shape with a polygonal base and triangular faces meeting at a point. -/
structure Pyramid where
  base : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  apex : ℝ × ℝ × ℝ    -- The apex point in 3D

/-- The number of faces in a prism. -/
def num_faces_prism (p : Prism) : ℕ := sorry

/-- The number of vertices in a pyramid. -/
def num_vertices_pyramid (p : Pyramid) : ℕ := sorry

/-- The minimum number of faces in any prism is 5. -/
theorem min_faces_prism : ∀ p : Prism, num_faces_prism p ≥ 5 := sorry

/-- The number of vertices in a pyramid with the minimum number of faces is 4. -/
theorem min_vertices_pyramid : ∃ p : Pyramid, num_vertices_pyramid p = 4 ∧ 
  (∀ q : Pyramid, num_vertices_pyramid q ≥ num_vertices_pyramid p) := sorry

end min_faces_prism_min_vertices_pyramid_l1494_149468


namespace problem_solution_l1494_149408

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) : ℝ := f (x + 1) - x

theorem problem_solution :
  (∃ (x_max : ℝ), ∀ (x : ℝ), g x ≤ g x_max ∧ g x_max = 0) ∧
  (∀ (n : ℕ), n > 0 → (1 + 1 / n : ℝ) ^ n < Real.exp 1) ∧
  (∀ (a b : ℝ), 0 < a → a < b → f b - f a > 2 * a * (b - a) / (a^2 + b^2)) :=
sorry

end problem_solution_l1494_149408


namespace pencil_buyers_difference_l1494_149445

theorem pencil_buyers_difference : ∀ (pencil_cost : ℕ) 
  (seventh_graders : ℕ) (sixth_graders : ℕ),
  pencil_cost > 0 →
  pencil_cost * seventh_graders = 143 →
  pencil_cost * sixth_graders = 195 →
  sixth_graders ≤ 30 →
  sixth_graders - seventh_graders = 4 :=
by sorry

end pencil_buyers_difference_l1494_149445


namespace line_passes_through_center_line_is_diameter_l1494_149436

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Theorem: The line passes through the center of the circle
theorem line_passes_through_center :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y :=
sorry

-- Theorem: The line is a diameter of the circle
theorem line_is_diameter :
  ∀ (x y : ℝ), circle_eq x y → line_eq x y → 
  ∃ (x' y' : ℝ), circle_eq x' y' ∧ line_eq x' y' ∧ 
  (x - x')^2 + (y - y')^2 = 4 :=
sorry

end line_passes_through_center_line_is_diameter_l1494_149436


namespace team_division_probabilities_l1494_149493

/-- The total number of teams -/
def total_teams : ℕ := 8

/-- The number of weak teams -/
def weak_teams : ℕ := 3

/-- The number of teams in each group -/
def group_size : ℕ := 4

/-- The probability that one group has exactly two weak teams -/
def prob_two_weak : ℚ := 6/7

/-- The probability that group A has at least two weak teams -/
def prob_A_at_least_two : ℚ := 1/2

/-- Theorem stating the probabilities for the team division problem -/
theorem team_division_probabilities :
  (prob_two_weak = 6/7) ∧ (prob_A_at_least_two = 1/2) := by sorry

end team_division_probabilities_l1494_149493


namespace petrol_consumption_reduction_l1494_149477

/-- Theorem: Calculation of required reduction in petrol consumption to maintain constant expenditure --/
theorem petrol_consumption_reduction
  (price_increase_A : ℝ) (price_increase_B : ℝ)
  (maintenance_cost_ratio : ℝ) (maintenance_cost_increase : ℝ)
  (h1 : price_increase_A = 0.20)
  (h2 : price_increase_B = 0.15)
  (h3 : maintenance_cost_ratio = 0.30)
  (h4 : maintenance_cost_increase = 0.10) :
  let avg_price_increase := (1 + price_increase_A + 1 + price_increase_B) / 2 - 1
  let total_maintenance_increase := maintenance_cost_ratio * maintenance_cost_increase
  let total_increase := avg_price_increase + total_maintenance_increase
  total_increase = 0.205 := by sorry

end petrol_consumption_reduction_l1494_149477


namespace line_through_points_l1494_149401

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨8, 9⟩
  let p2 : Point := ⟨2, -3⟩
  let p3 : Point := ⟨5, 3⟩
  let p4 : Point := ⟨6, 6⟩
  let p5 : Point := ⟨3, 0⟩
  let p6 : Point := ⟨0, -9⟩
  let p7 : Point := ⟨4, 1⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p7 ∧ 
  ¬collinear p1 p2 p4 ∧ 
  ¬collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p6 := by
  sorry

end line_through_points_l1494_149401


namespace quadratic_minimum_l1494_149487

/-- The function f(x) = x^2 - px + q reaches its minimum when x = p/2, given p > 0 and q > 0 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := fun x ↦ x^2 - p*x + q
  ∃ (x_min : ℝ), x_min = p/2 ∧ ∀ (x : ℝ), f x_min ≤ f x :=
by
  sorry

end quadratic_minimum_l1494_149487


namespace juan_running_time_l1494_149443

theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 80) (h2 : speed = 10) :
  distance / speed = 8 := by
  sorry

end juan_running_time_l1494_149443


namespace equation_solution_l1494_149403

theorem equation_solution :
  ∃ x : ℝ, x = 1 ∧ 2021 * x = 2022 * (x^2021)^(1/2021) - 1 :=
by
  sorry

end equation_solution_l1494_149403


namespace leadership_selection_count_l1494_149431

/-- The number of ways to choose a president, vice president, and a 3-person committee from a group of people. -/
def choose_leadership (total : ℕ) (males : ℕ) (females : ℕ) : ℕ :=
  let remaining := total - 2  -- After choosing president and vice president
  let committee_choices := 
    (males.choose 1 * females.choose 2) +  -- 1 male and 2 females
    (males.choose 2 * females.choose 1)    -- 2 males and 1 female
  (total * (total - 1)) * committee_choices

/-- The theorem stating the number of ways to choose leadership positions from a specific group. -/
theorem leadership_selection_count : 
  choose_leadership 10 6 4 = 8640 := by
  sorry


end leadership_selection_count_l1494_149431


namespace ellipse_intersection_theorem_l1494_149404

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  m : ℝ

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_intersection_theorem (E : Ellipse) (L : IntersectingLine) : 
  (E.a^2 - E.b^2 = 1) →  -- Focal length is 2
  (1 / E.a^2 + (9/4) / E.b^2 = 1) →  -- Ellipse passes through (1, 3/2)
  (∃ (x₁ y₁ x₂ y₂ : ℝ),  -- Intersection points exist
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    y₁ = 3/2 * x₁ + L.m ∧
    y₂ = 3/2 * x₂ + L.m) →
  (∃ (k₁ k₂ : ℝ),  -- Slope ratio condition
    k₁ / k₂ = 2 ∧
    k₁ = y₂ / (x₂ + 2) ∧
    k₂ = y₁ / (x₁ - 2)) →
  L.m = 1 :=
sorry

end ellipse_intersection_theorem_l1494_149404


namespace complex_imaginary_operation_l1494_149498

theorem complex_imaginary_operation : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end complex_imaginary_operation_l1494_149498


namespace intersection_complement_sets_l1494_149425

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_sets : M ∩ (U \ N) = {0, 3} := by sorry

end intersection_complement_sets_l1494_149425


namespace smallest_y_coordinate_l1494_149412

theorem smallest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y = 3 := by
  sorry

end smallest_y_coordinate_l1494_149412


namespace not_always_complete_gear_possible_l1494_149447

-- Define the number of teeth on each gear
def num_teeth : ℕ := 13

-- Define the number of pairs of teeth removed
def num_removed : ℕ := 4

-- Define a type for the positions of removed teeth
def RemovedTeeth := Fin num_teeth

-- Define a function to check if two positions overlap after rotation
def overlaps (x y : RemovedTeeth) (rotation : ℕ) : Prop :=
  (x.val + rotation) % num_teeth = y.val

-- State the theorem
theorem not_always_complete_gear_possible : ∃ (removed : Fin num_removed → RemovedTeeth),
  ∀ (rotation : ℕ), ∃ (i j : Fin num_removed), i ≠ j ∧ overlaps (removed i) (removed j) rotation :=
sorry

end not_always_complete_gear_possible_l1494_149447


namespace hilt_detergent_usage_l1494_149446

/-- The amount of detergent Mrs. Hilt uses per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The number of pounds of clothes to be washed -/
def pounds_of_clothes : ℝ := 9

/-- Theorem: Mrs. Hilt will use 18 ounces of detergent to wash 9 pounds of clothes -/
theorem hilt_detergent_usage : detergent_per_pound * pounds_of_clothes = 18 := by
  sorry

end hilt_detergent_usage_l1494_149446


namespace solve_linear_equation_l1494_149439

theorem solve_linear_equation :
  ∀ x : ℝ, 5 + 3.6 * x = 2.1 * x - 25 → x = -20 :=
by
  sorry

end solve_linear_equation_l1494_149439


namespace church_rows_count_l1494_149435

/-- Represents the seating arrangement in a church --/
structure ChurchSeating where
  chairs_per_row : ℕ
  people_per_chair : ℕ
  total_people : ℕ

/-- Calculates the number of rows in the church --/
def number_of_rows (s : ChurchSeating) : ℕ :=
  s.total_people / (s.chairs_per_row * s.people_per_chair)

/-- Theorem stating the number of rows in the church --/
theorem church_rows_count (s : ChurchSeating) 
  (h1 : s.chairs_per_row = 6)
  (h2 : s.people_per_chair = 5)
  (h3 : s.total_people = 600) :
  number_of_rows s = 20 := by
  sorry

#eval number_of_rows ⟨6, 5, 600⟩

end church_rows_count_l1494_149435


namespace correct_freshmen_sample_l1494_149449

/-- Represents a stratified sampling scenario in a college -/
structure CollegeSampling where
  total_students : ℕ
  freshmen : ℕ
  sample_size : ℕ

/-- Calculates the number of freshmen to be sampled in a stratified sampling -/
def freshmen_in_sample (cs : CollegeSampling) : ℕ :=
  cs.sample_size * cs.freshmen / cs.total_students

/-- Theorem stating the correct number of freshmen to be sampled -/
theorem correct_freshmen_sample (cs : CollegeSampling) 
  (h1 : cs.total_students = 3000)
  (h2 : cs.freshmen = 800)
  (h3 : cs.sample_size = 300) :
  freshmen_in_sample cs = 80 :=
sorry

end correct_freshmen_sample_l1494_149449


namespace arithmetic_sequence_range_l1494_149463

/-- 
For an arithmetic sequence with first term a₁ = -10 and common difference d,
if the 10th term and all subsequent terms are positive,
then 10/9 < d ≤ 5/4.
-/
theorem arithmetic_sequence_range (d : ℝ) : 
  (∀ n : ℕ, n ≥ 10 → -10 + (n - 1) * d > 0) → 
  10/9 < d ∧ d ≤ 5/4 := by
  sorry

end arithmetic_sequence_range_l1494_149463


namespace passengers_per_bus_l1494_149427

def total_people : ℕ := 1230
def num_buses : ℕ := 26

theorem passengers_per_bus :
  (total_people / num_buses : ℕ) = 47 := by sorry

end passengers_per_bus_l1494_149427


namespace john_piggy_bank_balance_l1494_149465

/-- The amount John saves monthly in dollars -/
def monthly_savings : ℕ := 25

/-- The number of months John saves -/
def saving_period : ℕ := 2 * 12

/-- The amount John spends on car repairs in dollars -/
def car_repair_cost : ℕ := 400

/-- The amount left in John's piggy bank after savings and car repair -/
def piggy_bank_balance : ℕ := monthly_savings * saving_period - car_repair_cost

theorem john_piggy_bank_balance : piggy_bank_balance = 200 := by
  sorry

end john_piggy_bank_balance_l1494_149465


namespace two_prime_roots_equation_l1494_149421

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem two_prime_roots_equation (n : ℕ) (h_pos : n > 0) :
  ∃ (x₁ x₂ : ℕ), 
    is_prime x₁ ∧ 
    is_prime x₂ ∧ 
    x₁ ≠ x₂ ∧
    2 * x₁^2 - 8*n*x₁ + 10*x₁ - n^2 + 35*n - 76 = 0 ∧
    2 * x₂^2 - 8*n*x₂ + 10*x₂ - n^2 + 35*n - 76 = 0 →
  n = 3 ∧ x₁ = 2 ∧ x₂ = 5 :=
sorry

end two_prime_roots_equation_l1494_149421


namespace quadratic_one_root_l1494_149402

theorem quadratic_one_root (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃! x : ℝ, x^2 + 6*m*x + m - n = 0) →
  (0 < m ∧ m < 1/9 ∧ n = m - 9*m^2) :=
by sorry

end quadratic_one_root_l1494_149402


namespace min_expression_proof_l1494_149486

theorem min_expression_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) ≥ 3 ∧
  (((x^2 * y * z) / 324 = (144 * y) / (x * z) ∧ (144 * y) / (x * z) = 9 / (4 * x * y^2)) →
    z / (16 * y) + x / 9 ≥ 2) ∧
  ((x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) = 3 ∧
   z / (16 * y) + x / 9 = 2) ↔ (x = 9 ∧ y = 1/2 ∧ z = 16) := by
  sorry

#check min_expression_proof

end min_expression_proof_l1494_149486


namespace prob_third_term_four_sum_of_fraction_parts_l1494_149453

/-- Set of permutations of 1,2,3,4,5,6 with restrictions -/
def T : Set (Fin 6 → Fin 6) :=
  { σ | Function.Bijective σ ∧ 
        σ 0 ≠ 0 ∧ σ 0 ≠ 1 ∧
        σ 1 ≠ 2 }

/-- The cardinality of set T -/
def T_size : ℕ := 48

/-- The number of permutations in T where the third term is 4 -/
def favorable_outcomes : ℕ := 12

/-- The probability of the third term being 4 in a randomly chosen permutation from T -/
theorem prob_third_term_four : 
  (favorable_outcomes : ℚ) / T_size = 1 / 4 :=
sorry

/-- The sum of numerator and denominator in the probability fraction -/
theorem sum_of_fraction_parts : 
  1 + 4 = 5 :=
sorry

end prob_third_term_four_sum_of_fraction_parts_l1494_149453


namespace total_investment_sum_l1494_149462

/-- Proves that the total sum of investments is 6358 given the specified conditions --/
theorem total_investment_sum (raghu_investment : ℝ) 
  (h1 : raghu_investment = 2200)
  (h2 : ∃ trishul_investment : ℝ, trishul_investment = raghu_investment * 0.9)
  (h3 : ∃ vishal_investment : ℝ, vishal_investment = trishul_investment * 1.1) :
  ∃ total_investment : ℝ, total_investment = raghu_investment + trishul_investment + vishal_investment ∧ 
  total_investment = 6358 :=
by sorry

end total_investment_sum_l1494_149462


namespace min_y_value_l1494_149414

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 50*y + 64) : y ≥ 0 := by
  sorry

end min_y_value_l1494_149414


namespace final_debt_calculation_l1494_149461

def calculate_debt (initial_loan : ℝ) (repayment1_percent : ℝ) (loan2 : ℝ) 
                   (repayment2_percent : ℝ) (loan3 : ℝ) (repayment3_percent : ℝ) : ℝ :=
  let debt1 := initial_loan * (1 - repayment1_percent)
  let debt2 := debt1 + loan2
  let debt3 := debt2 * (1 - repayment2_percent)
  let debt4 := debt3 + loan3
  debt4 * (1 - repayment3_percent)

theorem final_debt_calculation :
  calculate_debt 40 0.25 25 0.5 30 0.1 = 51.75 := by
  sorry

end final_debt_calculation_l1494_149461


namespace seven_abba_divisible_by_eleven_l1494_149478

theorem seven_abba_divisible_by_eleven (A : Nat) :
  A < 10 →
  (∃ B : Nat, B < 10 ∧ (70000 + A * 1000 + B * 100 + B * 10 + A) % 11 = 0) ↔
  A = 7 := by
sorry

end seven_abba_divisible_by_eleven_l1494_149478


namespace pig_count_l1494_149441

theorem pig_count (initial_pigs : ℕ) : initial_pigs + 86 = 150 → initial_pigs = 64 := by
  sorry

end pig_count_l1494_149441


namespace david_weighted_average_l1494_149491

def david_marks : List ℕ := [76, 65, 82, 67, 85, 93, 71]

def english_weight : ℕ := 2
def math_weight : ℕ := 3
def science_weight : ℕ := 1

def weighted_sum : ℕ := 
  david_marks[0] * english_weight + 
  david_marks[1] * math_weight + 
  david_marks[2] * science_weight + 
  david_marks[3] * science_weight + 
  david_marks[4] * science_weight

def total_weight : ℕ := english_weight + math_weight + 3 * science_weight

theorem david_weighted_average :
  (weighted_sum : ℚ) / total_weight = 581 / 8 := by sorry

end david_weighted_average_l1494_149491


namespace flour_added_indeterminate_l1494_149473

/-- Represents the ingredients in cups -/
structure Ingredients where
  sugar : ℕ
  flour : ℕ
  salt : ℕ

/-- Represents the current state of Mary's baking process -/
structure BakingState where
  recipe : Ingredients
  flour_added : ℕ
  sugar_to_add : ℕ
  salt_to_add : ℕ

/-- The recipe requirements -/
def recipe : Ingredients :=
  { sugar := 11, flour := 6, salt := 9 }

/-- Theorem stating that the amount of flour already added cannot be uniquely determined -/
theorem flour_added_indeterminate (state : BakingState) : 
  state.recipe = recipe → 
  state.sugar_to_add = state.salt_to_add + 2 → 
  ∃ (x y : ℕ), x ≠ y ∧ 
    (∃ (state1 state2 : BakingState), 
      state1.flour_added = x ∧ 
      state2.flour_added = y ∧ 
      state1.recipe = state.recipe ∧ 
      state2.recipe = state.recipe ∧ 
      state1.sugar_to_add = state.sugar_to_add ∧ 
      state2.sugar_to_add = state.sugar_to_add ∧ 
      state1.salt_to_add = state.salt_to_add ∧ 
      state2.salt_to_add = state.salt_to_add) :=
by
  sorry

end flour_added_indeterminate_l1494_149473


namespace probability_four_ones_in_five_rolls_l1494_149467

theorem probability_four_ones_in_five_rolls :
  let n_rolls : ℕ := 5
  let n_desired : ℕ := 4
  let die_sides : ℕ := 6
  let p_success : ℚ := 1 / die_sides
  let p_failure : ℚ := 1 - p_success
  let combinations : ℕ := Nat.choose n_rolls n_desired
  combinations * p_success ^ n_desired * p_failure ^ (n_rolls - n_desired) = 25 / 7776 :=
by sorry

end probability_four_ones_in_five_rolls_l1494_149467


namespace correct_logarithms_l1494_149432

-- Define the logarithm function
noncomputable def log (x : ℝ) : ℝ := Real.log x

-- Define the variables a, b, and c
variable (a b c : ℝ)

-- Define the given logarithmic relationships
axiom log_3 : log 3 = 2*a - b
axiom log_5 : log 5 = a + c
axiom log_2 : log 2 = 1 - a - c
axiom log_9 : log 9 = 4*a - 2*b
axiom log_14 : log 14 = 1 - c + 2*b

-- State the theorem to be proved
theorem correct_logarithms :
  log 1.5 = 3*a - b + c - 1 ∧ log 7 = 2*b + c :=
by sorry

end correct_logarithms_l1494_149432


namespace absolute_value_inequality_l1494_149494

theorem absolute_value_inequality (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 8) ↔ ((-10 ≤ x ∧ x ≤ -5) ∨ (1 ≤ x ∧ x ≤ 6)) := by
  sorry

end absolute_value_inequality_l1494_149494


namespace factor_of_polynomial_l1494_149458

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (y : ℝ), x^4 + 16 = (x^2 - 4*x + 4) * y :=
sorry

end factor_of_polynomial_l1494_149458


namespace necessary_but_not_sufficient_condition_for_positive_x_l1494_149440

theorem necessary_but_not_sufficient_condition_for_positive_x :
  (∀ x : ℝ, x > 0 → x > -2) ∧
  (∃ x : ℝ, x > -2 ∧ x ≤ 0) :=
by sorry

end necessary_but_not_sufficient_condition_for_positive_x_l1494_149440


namespace store_a_more_cost_effective_for_large_x_l1494_149451

/-- Represents the cost of purchasing table tennis rackets from Store A -/
def cost_store_a (x : ℕ) : ℚ :=
  if x ≤ 10 then 30 * x else 300 + 21 * (x - 10)

/-- Represents the cost of purchasing table tennis rackets from Store B -/
def cost_store_b (x : ℕ) : ℚ := 25.5 * x

/-- Theorem stating that Store A is more cost-effective than Store B for x > 20 -/
theorem store_a_more_cost_effective_for_large_x :
  ∀ x : ℕ, x > 20 → cost_store_a x < cost_store_b x :=
by
  sorry

/-- Helper lemma to show that cost_store_a simplifies to 21x + 90 for x > 10 -/
lemma cost_store_a_simplification (x : ℕ) (h : x > 10) :
  cost_store_a x = 21 * x + 90 :=
by
  sorry

end store_a_more_cost_effective_for_large_x_l1494_149451


namespace moles_CH₄_required_l1494_149471

/-- Represents a chemical species in a reaction --/
inductive Species
| CH₄ : Species
| Cl₂ : Species
| CHCl₃ : Species
| HCl : Species

/-- Represents the stoichiometric coefficients in a chemical reaction --/
def reaction_coefficients : Species → ℚ
| Species.CH₄ => -1
| Species.Cl₂ => -3
| Species.CHCl₃ => 1
| Species.HCl => 3

/-- The number of moles of CHCl₃ formed --/
def moles_CHCl₃_formed : ℚ := 3

/-- Theorem stating that the number of moles of CH₄ required to form 3 moles of CHCl₃ is 3 moles --/
theorem moles_CH₄_required :
  -reaction_coefficients Species.CH₄ * moles_CHCl₃_formed = 3 := by sorry

end moles_CH₄_required_l1494_149471


namespace pure_imaginary_complex_number_l1494_149444

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ (b : ℝ), a + (5 * Complex.I) / (1 - 2 * Complex.I) = b * Complex.I) → a = 2 := by
  sorry

end pure_imaginary_complex_number_l1494_149444


namespace square_sum_given_sum_square_and_product_l1494_149466

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end square_sum_given_sum_square_and_product_l1494_149466


namespace negation_of_existence_negation_of_proposition_l1494_149433

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_existence_negation_of_proposition_l1494_149433


namespace circle_symmetry_l1494_149400

/-- Given two circles and a line of symmetry, prove that the parameter 'a' in the first circle's equation must equal 2 for the circles to be symmetrical. -/
theorem circle_symmetry (x y : ℝ) (a : ℝ) : 
  (∀ x y, x^2 + y^2 - a*x + 2*y + 1 = 0) →  -- First circle equation
  (∀ x y, x^2 + y^2 = 1) →                  -- Second circle equation
  (∀ x y, x - y = 1) →                      -- Line of symmetry
  a = 2 := by
sorry


end circle_symmetry_l1494_149400


namespace solution_range_l1494_149475

theorem solution_range (b : ℝ) : 
  (∀ x : ℝ, x = -2 → x^2 - b*x - 5 = 5) ∧
  (∀ x : ℝ, x = -1 → x^2 - b*x - 5 = -1) ∧
  (∀ x : ℝ, x = 4 → x^2 - b*x - 5 = -1) ∧
  (∀ x : ℝ, x = 5 → x^2 - b*x - 5 = 5) →
  ∀ x : ℝ, x^2 - b*x - 5 = 0 ↔ (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) :=
by sorry

end solution_range_l1494_149475


namespace arithmetic_geometric_sequence_product_l1494_149485

theorem arithmetic_geometric_sequence_product (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, a n ≠ 0) →  -- a_n is non-zero for all n
  (a 3 - (a 7)^2 / 2 + a 11 = 0) →  -- given condition
  (∃ r, ∀ n, b (n + 1) = r * b n) →  -- b is a geometric sequence
  (b 7 = a 7) →  -- given condition
  (b 1 * b 13 = 16) :=
by sorry

end arithmetic_geometric_sequence_product_l1494_149485


namespace solution_satisfies_system_l1494_149457

theorem solution_satisfies_system :
  ∃ (x y z w : ℝ), 
    (x = 2 ∧ y = 2 ∧ z = 0 ∧ w = 0) ∧
    (x + y + Real.sqrt z = 4) ∧
    (Real.sqrt x * Real.sqrt y - Real.sqrt w = 2) :=
by sorry

end solution_satisfies_system_l1494_149457


namespace two_pythagorean_triples_l1494_149474

-- Define a Pythagorean triple
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- State the theorem
theorem two_pythagorean_triples :
  isPythagoreanTriple 3 4 5 ∧ isPythagoreanTriple 5 12 13 := by
  sorry

end two_pythagorean_triples_l1494_149474


namespace exit_times_theorem_l1494_149476

/-- Represents the time in minutes it takes to exit through a door -/
structure ExitTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the cinema with two doors -/
structure Cinema where
  wide_door : ExitTime
  narrow_door : ExitTime
  combined_exit_time : ℝ
  combined_exit_time_value : combined_exit_time = 3.75
  door_time_difference : narrow_door.time = wide_door.time + 4

theorem exit_times_theorem (c : Cinema) :
  c.wide_door.time = 6 ∧ c.narrow_door.time = 10 := by
  sorry

#check exit_times_theorem

end exit_times_theorem_l1494_149476


namespace jakes_weight_l1494_149489

theorem jakes_weight (j k : ℝ) 
  (h1 : j - 8 = 2 * k)  -- If Jake loses 8 pounds, he will weigh twice as much as Kendra
  (h2 : j + k = 290)    -- Together they now weigh 290 pounds
  : j = 196 :=          -- Jake's present weight is 196 pounds
by sorry

end jakes_weight_l1494_149489


namespace lcm_n_n_plus_3_l1494_149415

theorem lcm_n_n_plus_3 (n : ℕ) :
  lcm n (n + 3) = if n % 3 = 0 then n * (n + 3) / 3 else n * (n + 3) := by
  sorry

end lcm_n_n_plus_3_l1494_149415


namespace six_pointed_star_perimeter_l1494_149482

/-- A regular hexagon with perimeter 3 meters -/
structure RegularHexagon :=
  (perimeter : ℝ)
  (is_regular : perimeter = 3)

/-- A six-pointed star formed by extending the sides of a regular hexagon -/
structure SixPointedStar (h : RegularHexagon) :=
  (perimeter : ℝ)

/-- The perimeter of the six-pointed star is 4√3 meters -/
theorem six_pointed_star_perimeter (h : RegularHexagon) (s : SixPointedStar h) :
  s.perimeter = 4 * Real.sqrt 3 :=
sorry

end six_pointed_star_perimeter_l1494_149482


namespace sum_of_digits_of_greatest_prime_divisor_of_n_l1494_149455

-- Define the number we're working with
def n : ℕ := 9999

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define a function to check if a number is a divisor of n
def is_divisor_of_n (d : ℕ) : Prop := n % d = 0

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem stating the sum of digits of the greatest prime divisor of n is 2
theorem sum_of_digits_of_greatest_prime_divisor_of_n : 
  ∃ p : ℕ, is_prime p ∧ is_divisor_of_n p ∧ 
    (∀ q : ℕ, is_prime q → is_divisor_of_n q → q ≤ p) ∧
    sum_of_digits p = 2 :=
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_n_l1494_149455


namespace starting_number_is_271_l1494_149483

/-- A function that checks if a natural number contains the digit 1 -/
def contains_one (n : ℕ) : Bool := sorry

/-- The count of numbers from 1 to 1000 (exclusive) that do not contain the digit 1 -/
def count_no_one_to_1000 : ℕ := sorry

/-- The theorem to prove -/
theorem starting_number_is_271 (count_between : ℕ) 
  (h1 : count_between = 728) 
  (h2 : ∀ n ∈ Finset.range (1000 - 271), 
    ¬contains_one (n + 271) ↔ n < count_between) : 
  271 = 1000 - count_between - 1 :=
sorry

end starting_number_is_271_l1494_149483


namespace upstream_speed_is_26_l1494_149470

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (stillWater : ℝ)
  (downstream : ℝ)

/-- Calculate the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem: The speed of the man rowing upstream is 26 kmph -/
theorem upstream_speed_is_26 (s : RowingSpeed)
  (h1 : s.stillWater = 28)
  (h2 : s.downstream = 30) :
  upstreamSpeed s = 26 := by
  sorry

end upstream_speed_is_26_l1494_149470


namespace fraction_simplification_l1494_149423

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  12 / (x^2 - 9) - 2 / (x - 3) = -2 / (x + 3) := by
  sorry

end fraction_simplification_l1494_149423


namespace house_selling_price_l1494_149448

/-- Represents the total number of houses in the village -/
def total_houses : ℕ := 15

/-- Represents the total cost of construction for the entire village in millions of units -/
def total_cost : ℕ := 150 + 105 + 225 + 45

/-- Represents the markup percentage as a rational number -/
def markup : ℚ := 1 / 5

/-- Theorem: The selling price of each house in the village is 42 million units -/
theorem house_selling_price : 
  ∃ (cost_per_house : ℕ) (selling_price : ℕ),
    cost_per_house * total_houses = total_cost ∧
    selling_price = cost_per_house + cost_per_house * markup ∧
    selling_price = 42 :=
by sorry

end house_selling_price_l1494_149448


namespace sequence_product_l1494_149437

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that the product of the second term of the geometric sequence and
    the difference of the second and first terms of the arithmetic sequence is -8. -/
theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (∀ d : ℝ, -9 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -1) →  -- arithmetic sequence condition
  (∃ r : ℝ, -9 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -1) →  -- geometric sequence condition
  b₂ * (a₂ - a₁) = -8 := by
sorry

end sequence_product_l1494_149437


namespace initial_cards_equals_sum_l1494_149459

/-- The number of Pokemon cards Jason had initially --/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason gave away --/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left --/
def cards_left : ℕ := 4

/-- Theorem stating that the initial number of cards equals the sum of cards given away and cards left --/
theorem initial_cards_equals_sum : initial_cards = cards_given_away + cards_left := by
  sorry

end initial_cards_equals_sum_l1494_149459


namespace distance_to_line_segment_equidistant_points_vertical_line_equidistant_points_diagonal_l1494_149460

-- Define the distance function from a point to a line segment
def distance_point_to_segment (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the line segment l: x-y-3=0 (3 ≤ x ≤ 5)
def line_segment_l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - 3 = 0 ∧ 3 ≤ p.1 ∧ p.1 ≤ 5}

-- Theorem 1
theorem distance_to_line_segment :
  distance_point_to_segment (1, 1) line_segment_l = Real.sqrt 5 := by sorry

-- Define the set of points equidistant from two line segments
def equidistant_points (l₁ l₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | distance_point_to_segment p l₁ = distance_point_to_segment p l₂}

-- Define line segments AB and CD for Theorem 2
def line_segment_AB : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
def line_segment_CD : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Theorem 2
theorem equidistant_points_vertical_line :
  equidistant_points line_segment_AB line_segment_CD = {p : ℝ × ℝ | p.1 = 0} := by sorry

-- Define line segments AB and CD for Theorem 3
def line_segment_AB' : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0 ∧ -1 ≤ p.1 ∧ p.1 ≤ 1}
def line_segment_CD' : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1}

-- Theorem 3
theorem equidistant_points_diagonal :
  equidistant_points line_segment_AB' line_segment_CD' = {p : ℝ × ℝ | p.1^2 - p.2^2 = 0} := by sorry

end distance_to_line_segment_equidistant_points_vertical_line_equidistant_points_diagonal_l1494_149460


namespace total_prizes_l1494_149419

def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def yo_yos : ℕ := 18

theorem total_prizes : stuffed_animals + frisbees + yo_yos = 50 := by
  sorry

end total_prizes_l1494_149419


namespace vector_problem_l1494_149484

def a : Fin 2 → ℝ := ![2, 4]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

theorem vector_problem (x : ℝ) :
  (∀ (i : Fin 2), (a i * b x i) > 0) →
  (x > -2 ∧ x ≠ 1/2) ∧
  ((∀ (i : Fin 2), ((2 * a i - b x i) * a i) = 0) →
   Real.sqrt ((a 0 + b x 0)^2 + (a 1 + b x 1)^2) = 5 * Real.sqrt 17) :=
by sorry

end vector_problem_l1494_149484


namespace h_is_even_l1494_149454

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the function h
def h (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ g x * |f x|

-- State the theorem
theorem h_is_even (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) : 
  IsEven (h f g) := by
  sorry

end h_is_even_l1494_149454


namespace sphere_surface_area_from_rectangular_solid_l1494_149490

/-- The surface area of a sphere that circumscribes a rectangular solid -/
theorem sphere_surface_area_from_rectangular_solid 
  (length width height : ℝ) 
  (h_length : length = 4) 
  (h_width : width = 3) 
  (h_height : height = 2) : 
  ∃ (radius : ℝ), 4 * Real.pi * radius^2 = 29 * Real.pi := by
  sorry

end sphere_surface_area_from_rectangular_solid_l1494_149490


namespace ellipse_intersection_slope_l1494_149497

/-- Given an ellipse ax² + by² = 1 intersecting the line y = 1 - x, 
    if a line through the origin and the midpoint of the intersection points 
    has slope √3/2, then a/b = √3/2 -/
theorem ellipse_intersection_slope (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : 
  (∃ x₁ x₂ : ℝ, 
    a * x₁^2 + b * (1 - x₁)^2 = 1 ∧ 
    a * x₂^2 + b * (1 - x₂)^2 = 1 ∧
    x₁ ≠ x₂ ∧
    (a / (a + b)) / (b / (a + b)) = Real.sqrt 3 / 2) →
  a / b = Real.sqrt 3 / 2 :=
sorry

end ellipse_intersection_slope_l1494_149497


namespace absolute_value_inequality_solution_l1494_149413

theorem absolute_value_inequality_solution (x : ℝ) :
  (|2*x - 3| < 5) ↔ (-1 < x ∧ x < 4) :=
sorry

end absolute_value_inequality_solution_l1494_149413


namespace fraction_simplification_l1494_149472

theorem fraction_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + x*y) / (x*y) * y^2 / (x + y) = y :=
sorry

end fraction_simplification_l1494_149472
