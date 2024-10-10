import Mathlib

namespace largest_sum_is_three_fourths_l2115_211586

theorem largest_sum_is_three_fourths : 
  let sums : List ℚ := [1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/2, 1/4 + 1/12, 1/4 + 1/11]
  (∀ s ∈ sums, s ≤ 3/4) ∧ (3/4 ∈ sums) := by
  sorry

end largest_sum_is_three_fourths_l2115_211586


namespace polynomial_remainder_theorem_l2115_211588

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^12 - 1) % (x + 1) = 0 := by
  sorry

end polynomial_remainder_theorem_l2115_211588


namespace sally_has_hundred_l2115_211507

/-- Sally's current amount of money -/
def sally_money : ℕ := sorry

/-- The condition that if Sally had $20 less, she would have $80 -/
axiom sally_condition : sally_money - 20 = 80

/-- Theorem: Sally has $100 -/
theorem sally_has_hundred : sally_money = 100 := by sorry

end sally_has_hundred_l2115_211507


namespace fenced_area_calculation_l2115_211527

theorem fenced_area_calculation (yard_length yard_width cutout_side : ℕ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 18)
  (h3 : cutout_side = 4) :
  yard_length * yard_width - cutout_side * cutout_side = 344 := by
  sorry

end fenced_area_calculation_l2115_211527


namespace sheep_flock_size_l2115_211543

theorem sheep_flock_size :
  ∀ (x y : ℕ),
  (x - 1 : ℚ) / y = 7 / 5 →
  x / (y - 1 : ℚ) = 5 / 3 →
  x + y = 25 :=
by
  sorry

end sheep_flock_size_l2115_211543


namespace easel_cost_l2115_211545

def paintbrush_cost : ℚ := 2.4
def paints_cost : ℚ := 9.2
def rose_has : ℚ := 7.1
def rose_needs : ℚ := 11

theorem easel_cost : 
  let total_cost := rose_has + rose_needs
  let other_items_cost := paintbrush_cost + paints_cost
  total_cost - other_items_cost = 6.5 := by sorry

end easel_cost_l2115_211545


namespace stating_tree_structure_equation_l2115_211587

/-- Represents a tree structure with a trunk, branches, and small branches. -/
structure TreeStructure where
  x : ℕ  -- number of branches grown by the trunk
  total : ℕ  -- total count of trunk, branches, and small branches
  h_total : total = x^2 + x + 1  -- relation between x and total

/-- 
Theorem stating that for a tree structure with 43 total elements,
the equation x^2 + x + 1 = 43 correctly represents the structure.
-/
theorem tree_structure_equation (t : TreeStructure) (h : t.total = 43) :
  t.x^2 + t.x + 1 = 43 := by
  sorry

end stating_tree_structure_equation_l2115_211587


namespace prove_additional_cans_l2115_211520

/-- The number of additional cans Alyssa and Abigail need to collect. -/
def additional_cans_needed (total_needed alyssa_collected abigail_collected : ℕ) : ℕ :=
  total_needed - (alyssa_collected + abigail_collected)

/-- Theorem: Given the conditions, the additional cans needed is 27. -/
theorem prove_additional_cans : additional_cans_needed 100 30 43 = 27 := by
  sorry

end prove_additional_cans_l2115_211520


namespace local_extremum_implies_a_equals_four_l2115_211517

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_implies_a_equals_four :
  ∀ a b : ℝ,
  (f a b 1 = 10) →  -- f(1) = 10
  (f' a b 1 = 0) →  -- f'(1) = 0 (condition for local extremum)
  (∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → f a b x ≤ f a b 1) →  -- local maximum condition
  a = 4 :=
sorry

end local_extremum_implies_a_equals_four_l2115_211517


namespace expression_evaluation_l2115_211523

theorem expression_evaluation : 2 + 3 * 4 - 1^2 + 6 / 3 = 15 := by
  sorry

end expression_evaluation_l2115_211523


namespace percentage_relation_l2115_211533

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = x / 100 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 50 := by
sorry

end percentage_relation_l2115_211533


namespace pants_price_proof_l2115_211528

-- Define the total cost
def total_cost : ℝ := 70.93

-- Define the price difference between belt and pants
def price_difference : ℝ := 2.93

-- Define the price of pants
def price_of_pants : ℝ := 34.00

-- Theorem statement
theorem pants_price_proof :
  ∃ (belt_price : ℝ),
    price_of_pants + belt_price = total_cost ∧
    price_of_pants = belt_price - price_difference :=
by sorry

end pants_price_proof_l2115_211528


namespace average_of_c_and_d_l2115_211565

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 18 → (c + d) / 2 = 36 := by
  sorry

end average_of_c_and_d_l2115_211565


namespace male_students_count_l2115_211546

theorem male_students_count (total : ℕ) (difference : ℕ) (male : ℕ) (female : ℕ) : 
  total = 1443 →
  difference = 141 →
  male = female + difference →
  total = male + female →
  male = 792 := by
sorry

end male_students_count_l2115_211546


namespace root_implies_difference_l2115_211505

theorem root_implies_difference (a b : ℝ) :
  (∃ x, x^2 + 4*a^2*b^2*x = 4 ∧ x = (a^2 - b^2)^2) →
  (b^4 - a^4 = 2 ∨ b^4 - a^4 = -2) :=
by sorry

end root_implies_difference_l2115_211505


namespace ellipse_and_range_of_m_l2115_211571

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the square formed by foci and vertices of minor axis -/
def square_perimeter (a b : ℝ) : Prop := 4 * a = 4 * Real.sqrt 2 ∧ b = Real.sqrt (a^2 - b^2)

/-- Definition of point B -/
def point_B (m : ℝ) : ℝ × ℝ := (0, m)

/-- Definition of point D symmetric to B with respect to origin -/
def point_D (m : ℝ) : ℝ × ℝ := (0, -m)

/-- Definition of line l passing through B -/
def line_l (x y m k : ℝ) : Prop := y = k * x + m

/-- Definition of intersection points E and F -/
def intersection_points (x y m k : ℝ) : Prop :=
  ellipse_C x y (Real.sqrt 2) 1 ∧ line_l x y m k

/-- Definition of D being inside circle with diameter EF -/
def D_inside_circle (m : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection_points x₁ y₁ m k ∧
    intersection_points x₂ y₂ m k ∧
    (0 - (x₁ + x₂)/2)^2 + (-m - (y₁ + y₂)/2)^2 < ((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4

/-- Main theorem -/
theorem ellipse_and_range_of_m (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : square_perimeter a b) :
  (ellipse_C x y (Real.sqrt 2) 1 ↔ ellipse_C x y a b) ∧
  (∀ m : ℝ, m > 0 → D_inside_circle m → 0 < m ∧ m < Real.sqrt 3 / 3) :=
sorry

end ellipse_and_range_of_m_l2115_211571


namespace renu_work_time_l2115_211580

/-- The number of days it takes Renu to complete the work alone -/
def renu_days : ℝ := 8

/-- The number of days it takes Suma to complete the work alone -/
def suma_days : ℝ := 4.8

/-- The number of days it takes Renu and Suma to complete the work together -/
def combined_days : ℝ := 3

theorem renu_work_time :
  (1 / renu_days) + (1 / suma_days) = (1 / combined_days) :=
sorry

end renu_work_time_l2115_211580


namespace problem_statement_l2115_211567

theorem problem_statement (a b : ℝ) (h : |a - 3| + (b + 2)^2 = 0) :
  (a + b)^2023 = 1 := by
  sorry

end problem_statement_l2115_211567


namespace range_of_a_l2115_211541

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.exp x + x - a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.cos x₀ ∧ f a (f a y₀) = y₀) →
  2 ≤ a ∧ a ≤ Real.exp 1 + 1 :=
sorry

end range_of_a_l2115_211541


namespace courtyard_paving_l2115_211552

/-- Calculates the number of bricks required to pave a courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℝ) :
  courtyard_length = 35 ∧ 
  courtyard_width = 24 ∧ 
  brick_length = 0.15 ∧ 
  brick_width = 0.08 →
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 70000 := by
  sorry

#check courtyard_paving

end courtyard_paving_l2115_211552


namespace friction_coefficient_inclined_plane_l2115_211595

/-- The coefficient of kinetic friction for a block sliding down an inclined plane,
    given that it reaches the bottom simultaneously with a hollow cylinder rolling without slipping -/
theorem friction_coefficient_inclined_plane (θ : Real) (g : Real) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : g > 0) :
  let μ := (1 / 2) * Real.tan θ
  let a_cylinder := (1 / 2) * g * Real.sin θ
  let a_block := g * Real.sin θ - μ * g * Real.cos θ
  a_cylinder = a_block :=
by sorry

end friction_coefficient_inclined_plane_l2115_211595


namespace relationship_between_A_B_C_l2115_211556

-- Define the variables and functions
variable (a : ℝ)
def A : ℝ := 2 * a - 7
def B : ℝ := a^2 - 4 * a + 3
def C : ℝ := a^2 + 6 * a - 28

-- Theorem statement
theorem relationship_between_A_B_C (h : a > 2) :
  (B a - A a > 0) ∧
  (∀ x, 2 < x ∧ x < 3 → C x - A x < 0) ∧
  (C 3 - A 3 = 0) ∧
  (∀ y, y > 3 → C y - A y > 0) := by
  sorry

end relationship_between_A_B_C_l2115_211556


namespace baseball_card_value_decrease_l2115_211534

theorem baseball_card_value_decrease : 
  let initial_value : ℝ := 100
  let year1_decrease : ℝ := 0.60
  let year2_decrease : ℝ := 0.30
  let year3_decrease : ℝ := 0.20
  let year4_decrease : ℝ := 0.10
  
  let value_after_year1 : ℝ := initial_value * (1 - year1_decrease)
  let value_after_year2 : ℝ := value_after_year1 * (1 - year2_decrease)
  let value_after_year3 : ℝ := value_after_year2 * (1 - year3_decrease)
  let value_after_year4 : ℝ := value_after_year3 * (1 - year4_decrease)
  
  let total_decrease : ℝ := (initial_value - value_after_year4) / initial_value * 100

  total_decrease = 79.84 := by sorry

end baseball_card_value_decrease_l2115_211534


namespace weighted_sum_inequality_l2115_211559

theorem weighted_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (sum_geq_one : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end weighted_sum_inequality_l2115_211559


namespace unique_divisor_sequence_l2115_211562

theorem unique_divisor_sequence : ∃! (x y z w : ℕ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  x % y = 0 ∧ y % z = 0 ∧ z % w = 0 ∧
  x + y + z + w = 329 ∧
  x = 231 ∧ y = 77 ∧ z = 14 ∧ w = 7 := by
sorry

end unique_divisor_sequence_l2115_211562


namespace geometric_sequence_ratio_l2115_211525

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  2 * ((1/2) * a 3) = a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticCondition a →
  (∀ n : ℕ, a n > 0) →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_ratio_l2115_211525


namespace intersection_when_a_half_subset_iff_a_range_l2115_211585

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Theorem 1: When a = 1/2, A ∩ B = B
theorem intersection_when_a_half : A (1/2) ∩ B = B := by sorry

-- Theorem 2: B ⊆ A if and only if 0 ≤ a ≤ 1
theorem subset_iff_a_range : B ⊆ A a ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end intersection_when_a_half_subset_iff_a_range_l2115_211585


namespace balloon_distribution_l2115_211573

theorem balloon_distribution (yellow_balloons : ℕ) (black_balloons_diff : ℕ) (balloons_per_school : ℕ) : 
  yellow_balloons = 3414 →
  black_balloons_diff = 1762 →
  balloons_per_school = 859 →
  (yellow_balloons + (yellow_balloons + black_balloons_diff)) / balloons_per_school = 10 := by
  sorry

end balloon_distribution_l2115_211573


namespace no_common_solution_l2115_211558

theorem no_common_solution : ¬ ∃ (x y : ℝ), (x^2 - 6*x + y + 9 = 0) ∧ (x^2 + 4*y + 5 = 0) := by
  sorry

end no_common_solution_l2115_211558


namespace flour_for_cookies_l2115_211549

/-- Given a recipe where 24 cookies require 1.5 cups of flour,
    calculate the amount of flour needed for 72 cookies. -/
theorem flour_for_cookies (original_cookies : ℕ) (original_flour : ℚ) (new_cookies : ℕ) :
  original_cookies = 24 →
  original_flour = 3/2 →
  new_cookies = 72 →
  (original_flour / original_cookies) * new_cookies = 9/2 :=
by sorry

end flour_for_cookies_l2115_211549


namespace book_reading_competition_l2115_211594

/-- Represents the number of pages read by each girl -/
structure PageCount where
  ivana : ℕ
  majka : ℕ
  lucka : ℕ
  sasa : ℕ
  zuzka : ℕ

/-- Checks if all values in the PageCount are distinct -/
def allDistinct (p : PageCount) : Prop :=
  p.ivana ≠ p.majka ∧ p.ivana ≠ p.lucka ∧ p.ivana ≠ p.sasa ∧ p.ivana ≠ p.zuzka ∧
  p.majka ≠ p.lucka ∧ p.majka ≠ p.sasa ∧ p.majka ≠ p.zuzka ∧
  p.lucka ≠ p.sasa ∧ p.lucka ≠ p.zuzka ∧
  p.sasa ≠ p.zuzka

/-- The theorem representing the book reading competition -/
theorem book_reading_competition :
  ∃! (p : PageCount),
    p.lucka = 32 ∧
    p.lucka = (p.sasa + p.zuzka) / 2 ∧
    p.ivana = p.zuzka + 5 ∧
    p.majka = p.sasa - 8 ∧
    allDistinct p ∧
    (∀ x ∈ [p.ivana, p.majka, p.lucka, p.sasa, p.zuzka], x ≥ 27) ∧
    p.ivana = 34 ∧ p.majka = 27 ∧ p.lucka = 32 ∧ p.sasa = 35 ∧ p.zuzka = 29 :=
by sorry

end book_reading_competition_l2115_211594


namespace symmetric_point_theorem_l2115_211512

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The point symmetric to (2, -3) with respect to the origin is (-2, 3) -/
theorem symmetric_point_theorem :
  let P : Point := { x := 2, y := -3 }
  let P' : Point := symmetricToOrigin P
  P'.x = -2 ∧ P'.y = 3 := by
  sorry

end symmetric_point_theorem_l2115_211512


namespace marble_bag_size_l2115_211501

/-- Represents a bag of marbles with blue, red, and white colors. -/
structure MarbleBag where
  total : ℕ
  blue : ℕ
  red : ℕ
  white : ℕ

/-- The probability of selecting a red or white marble from the bag. -/
def redOrWhiteProbability (bag : MarbleBag) : ℚ :=
  (bag.red + bag.white : ℚ) / bag.total

theorem marble_bag_size :
  ∃ (bag : MarbleBag),
    bag.blue = 5 ∧
    bag.red = 7 ∧
    redOrWhiteProbability bag = 3/4 ∧
    bag.total = 20 :=
by
  sorry

end marble_bag_size_l2115_211501


namespace quadratic_transformation_l2115_211578

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := 6 * x^2 - 12 * x + 4

-- Define the transformed expression
def transformed (x a h k : ℝ) : ℝ := a * (x - h)^2 + k

-- Theorem statement
theorem quadratic_transformation :
  ∃ (a h k : ℝ), (∀ x, quadratic x = transformed x a h k) ∧ (a + h + k = 5) := by
  sorry

end quadratic_transformation_l2115_211578


namespace isosceles_trapezoid_area_l2115_211524

noncomputable def trapezoidArea (h α β : ℝ) : ℝ :=
  2 * h^2 * (Real.tan β + Real.tan α)

theorem isosceles_trapezoid_area
  (h α β : ℝ)
  (h_pos : h > 0)
  (α_pos : α > 0)
  (β_pos : β > 0)
  (α_lt_90 : α < π / 2)
  (β_lt_90 : β < π / 2)
  (h_eq : h = 2)
  (α_eq : α = 15 * π / 180)
  (β_eq : β = 75 * π / 180) :
  trapezoidArea h α β = 16 := by
  sorry

end isosceles_trapezoid_area_l2115_211524


namespace geometry_propositions_l2115_211560

-- Define the concept of vertical angles
def are_vertical_angles (α β : Real) : Prop := sorry

-- Define the concept of complementary angles
def are_complementary (α β : Real) : Prop := α + β = 90

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem geometry_propositions :
  -- Proposition 1: Vertical angles are equal
  ∀ (α β : Real), are_vertical_angles α β → α = β
  
  -- Proposition 2: Complementary angles of equal angles are equal
  ∧ ∀ (α β γ δ : Real), α = β ∧ are_complementary α γ ∧ are_complementary β δ → γ = δ
  
  -- Proposition 3: If b is parallel to a and c is parallel to a, then b is parallel to c
  ∧ ∀ (a b c : Line), parallel b a ∧ parallel c a → parallel b c :=
by sorry

end geometry_propositions_l2115_211560


namespace largest_n_divisibility_l2115_211569

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > n → ¬((m + 12) ∣ (m^3 + 160))) ∧ 
  ((n + 12) ∣ (n^3 + 160)) ∧ n = 1748 := by
  sorry

end largest_n_divisibility_l2115_211569


namespace tan_alpha_value_l2115_211519

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = (1 - Real.sqrt 3) / 2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.tan α = -Real.sqrt 3 / 3 := by
sorry

end tan_alpha_value_l2115_211519


namespace shortest_wire_length_l2115_211583

/-- The length of the shortest wire around two circular poles -/
theorem shortest_wire_length (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 18) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_section := 2 * Real.sqrt ((r1 + r2)^2 - (r2 - r1)^2)
  let small_circle_arc := 2 * π * r1 * (1/3)
  let large_circle_arc := 2 * π * r2 * (2/3)
  straight_section + small_circle_arc + large_circle_arc = 12 * Real.sqrt 3 + 14 * π :=
by sorry

end shortest_wire_length_l2115_211583


namespace A_equals_Z_l2115_211561

-- Define the set A
def A : Set Int :=
  {n | ∃ (a b : Nat), a ≥ 1 ∧ b ≥ 1 ∧ n = 2^a - 2^b}

-- Define the closure property of A
axiom A_closure (a b : Int) : a ∈ A → b ∈ A → (a + b) ∈ A

-- Axiom stating that A contains at least one odd number
axiom A_contains_odd : ∃ (n : Int), n ∈ A ∧ n % 2 ≠ 0

-- Theorem to prove
theorem A_equals_Z : A = Set.univ := by sorry

end A_equals_Z_l2115_211561


namespace area_of_overlapping_squares_l2115_211532

/-- The area covered by two overlapping congruent squares -/
theorem area_of_overlapping_squares (side_length : ℝ) (h : side_length = 12) :
  let square_area := side_length ^ 2
  let overlap_area := square_area / 4
  let total_area := 2 * square_area - overlap_area
  total_area = 252 := by sorry

end area_of_overlapping_squares_l2115_211532


namespace complement_of_B_relative_to_A_l2115_211579

def A : Set Int := {-1, 0, 1, 2, 3}
def B : Set Int := {-1, 1}

theorem complement_of_B_relative_to_A :
  {x : Int | x ∈ A ∧ x ∉ B} = {0, 2, 3} := by sorry

end complement_of_B_relative_to_A_l2115_211579


namespace seating_arrangements_with_restriction_l2115_211589

def number_of_people : ℕ := 4

def total_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_pair_together (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem seating_arrangements_with_restriction :
  total_arrangements number_of_people - arrangements_with_pair_together number_of_people = 12 := by
  sorry

end seating_arrangements_with_restriction_l2115_211589


namespace C_power_50_l2115_211550

def C : Matrix (Fin 2) (Fin 2) ℤ := !![2, 1; -4, -1]

theorem C_power_50 : C^50 = !![4^49 + 1, 4^49; -4^50, -2 * 4^49 + 1] := by sorry

end C_power_50_l2115_211550


namespace computer_pricing_l2115_211529

/-- Proves that if a selling price of $2240 yields a 40% profit on cost, 
    then a selling price of $2560 yields a 60% profit on the same cost. -/
theorem computer_pricing (cost : ℝ) 
  (h1 : 2240 = cost + 0.4 * cost) 
  (h2 : 2560 = cost + 0.6 * cost) : 
  2240 = cost * 1.4 ∧ 2560 = cost * 1.6 := by
  sorry

#check computer_pricing

end computer_pricing_l2115_211529


namespace rectangular_plot_length_l2115_211521

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 24 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 62 := by
sorry

end rectangular_plot_length_l2115_211521


namespace opposite_of_2023_l2115_211539

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + 2023 = 0 → x = -2023) ∧ (-2023 + 2023 = 0) := by
  sorry

end opposite_of_2023_l2115_211539


namespace tangent_line_circle_minimum_l2115_211554

theorem tangent_line_circle_minimum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2 ∧ 
    ∀ x' y' : ℝ, (x' - b)^2 + (y' - 1)^2 ≤ 2 → (x' + y' + a)^2 > 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x y : ℝ, x + y + a' = 0 ∧ (x - b')^2 + (y - 1)^2 = 2 ∧ 
      ∀ x' y' : ℝ, (x' - b')^2 + (y' - 1)^2 ≤ 2 → (x' + y' + a')^2 > 0) → 
    (3 - 2*b')^2 / (2*a') ≥ (3 - 2*b)^2 / (2*a)) →
  (3 - 2*b)^2 / (2*a) = 4 := by
sorry

end tangent_line_circle_minimum_l2115_211554


namespace polynomial_multiplication_l2115_211518

/-- Given a polynomial P(x) that satisfies P(x) - 3x^2 = x^2 - (1/2)x + 1,
    prove that (-3x^2) * P(x) = -12x^4 + (3/2)x^3 - 3x^2 -/
theorem polynomial_multiplication (x : ℝ) (P : ℝ → ℝ) 
    (h : P x - 3 * x^2 = x^2 - (1/2) * x + 1) :
  (-3 * x^2) * P x = -12 * x^4 + (3/2) * x^3 - 3 * x^2 := by
  sorry

end polynomial_multiplication_l2115_211518


namespace product_equals_three_eighths_l2115_211590

-- Define the fractions and mixed number
def a : ℚ := 1/2
def b : ℚ := 2/3
def c : ℚ := 3/4
def d : ℚ := 3/2  -- 1.5 as a fraction

-- State the theorem
theorem product_equals_three_eighths :
  a * b * c * d = 3/8 := by
  sorry

end product_equals_three_eighths_l2115_211590


namespace simone_apple_fraction_l2115_211530

theorem simone_apple_fraction (x : ℚ) : 
  (16 * x + 15 * (1 / 3 : ℚ) = 13) → x = 1 / 2 := by
  sorry

end simone_apple_fraction_l2115_211530


namespace black_ball_probability_l2115_211570

theorem black_ball_probability 
  (n₁ n₂ k₁ k₂ : ℕ) 
  (h_total : n₁ + n₂ = 25)
  (h_white_prob : (k₁ : ℚ) / n₁ * k₂ / n₂ = 54 / 100) :
  (n₁ - k₁ : ℚ) / n₁ * (n₂ - k₂) / n₂ = 4 / 100 := by
sorry

end black_ball_probability_l2115_211570


namespace simplify_expression_l2115_211564

theorem simplify_expression :
  (Real.sqrt 10 + Real.sqrt 15) / (Real.sqrt 3 + Real.sqrt 5 - Real.sqrt 2) =
  (2 * Real.sqrt 30 + 5 * Real.sqrt 2 + 11 * Real.sqrt 5 + 5 * Real.sqrt 3) / 6 := by
sorry

end simplify_expression_l2115_211564


namespace hcf_of_36_and_84_l2115_211597

theorem hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_of_36_and_84_l2115_211597


namespace product_of_three_numbers_l2115_211514

theorem product_of_three_numbers (a b c m : ℝ) : 
  a + b + c = 180 ∧
  5 * a = m ∧
  b = m + 12 ∧
  c = m - 6 →
  a * b * c = 42184 := by
sorry

end product_of_three_numbers_l2115_211514


namespace helen_cookies_l2115_211536

/-- The number of raisin cookies Helen baked this morning -/
def raisin_cookies : ℕ := 231

/-- The difference between chocolate chip cookies and raisin cookies -/
def cookie_difference : ℕ := 25

/-- The number of chocolate chip cookies Helen baked this morning -/
def choc_chip_cookies : ℕ := raisin_cookies + cookie_difference

theorem helen_cookies : choc_chip_cookies = 256 := by
  sorry

end helen_cookies_l2115_211536


namespace quadratic_roots_relation_l2115_211509

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2*x₁ ∨ x = 2*x₂)) →
  a / c = 1 / 8 :=
by sorry


end quadratic_roots_relation_l2115_211509


namespace initial_milk_water_ratio_l2115_211551

theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_milk : ℝ) 
  (new_ratio : ℝ) :
  total_volume = 20 →
  added_milk = 5 →
  new_ratio = 4 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    (initial_milk + added_milk) / initial_water = new_ratio ∧
    initial_milk / initial_water = 3 := by
sorry

end initial_milk_water_ratio_l2115_211551


namespace fine_on_fifth_day_l2115_211500

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (previousFine : ℚ) : ℚ :=
  min (previousFine + 0.3) (previousFine * 2)

/-- Calculates the fine for a given number of days overdue -/
def fineFordaysOverdue (days : ℕ) : ℚ :=
  match days with
  | 0 => 0
  | 1 => 0.05
  | n + 1 => nextDayFine (fineFordaysOverdue n)

theorem fine_on_fifth_day :
  fineFordaysOverdue 5 = 0.7 := by
  sorry

end fine_on_fifth_day_l2115_211500


namespace calculate_expression_l2115_211563

theorem calculate_expression : (42 / (3^2 + 2 * 3 - 1)) * 7 = 21 := by
  sorry

end calculate_expression_l2115_211563


namespace jimmy_payment_l2115_211508

/-- Represents the cost of a pizza in dollars -/
def pizza_cost : ℕ := 12

/-- Represents the delivery charge in dollars for distances over 1 km -/
def delivery_charge : ℕ := 2

/-- Represents the distance threshold in meters for applying delivery charge -/
def distance_threshold : ℕ := 1000

/-- Represents the number of pizzas delivered to the park -/
def park_pizzas : ℕ := 3

/-- Represents the distance to the park in meters -/
def park_distance : ℕ := 100

/-- Represents the number of pizzas delivered to the building -/
def building_pizzas : ℕ := 2

/-- Represents the distance to the building in meters -/
def building_distance : ℕ := 2000

/-- Calculates the total amount Jimmy got paid for the pizzas -/
def total_amount : ℕ :=
  (park_pizzas + building_pizzas) * pizza_cost +
  (if building_distance > distance_threshold then building_pizzas * delivery_charge else 0)

theorem jimmy_payment : total_amount = 64 := by
  sorry

end jimmy_payment_l2115_211508


namespace perpendicular_bisector_value_l2115_211596

/-- The perpendicular bisector of a line segment passing through two points -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) (b : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = b

theorem perpendicular_bisector_value : 
  perpendicular_bisector 2 4 6 8 10 := by
  sorry

#check perpendicular_bisector_value

end perpendicular_bisector_value_l2115_211596


namespace factorization_of_4a_minus_a_cubed_l2115_211582

theorem factorization_of_4a_minus_a_cubed (a : ℝ) : 4*a - a^3 = a*(2-a)*(2+a) := by
  sorry

end factorization_of_4a_minus_a_cubed_l2115_211582


namespace max_value_on_circle_l2115_211584

theorem max_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 1 → (y / (x + 2) ≤ Real.sqrt 3 / 3) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 1 ∧ y₀ / (x₀ + 2) = Real.sqrt 3 / 3) := by
  sorry

end max_value_on_circle_l2115_211584


namespace james_earnings_l2115_211557

/-- Represents the amount of water collected per inch of rain -/
def water_per_inch : ℝ := 15

/-- Represents the amount of rain on Monday in inches -/
def monday_rain : ℝ := 4

/-- Represents the amount of rain on Tuesday in inches -/
def tuesday_rain : ℝ := 3

/-- Represents the price per gallon of water in dollars -/
def price_per_gallon : ℝ := 1.2

/-- Calculates the total amount of money James made from selling all the water -/
def total_money : ℝ :=
  (monday_rain * water_per_inch + tuesday_rain * water_per_inch) * price_per_gallon

/-- Theorem stating that James made $126 from selling all the water -/
theorem james_earnings : total_money = 126 := by
  sorry

end james_earnings_l2115_211557


namespace gcd_special_numbers_l2115_211591

theorem gcd_special_numbers : Nat.gcd 333333333 555555555 = 111111111 := by
  sorry

end gcd_special_numbers_l2115_211591


namespace tim_pay_per_task_l2115_211568

/-- Represents the pay per task for Tim's work --/
def pay_per_task (tasks_per_day : ℕ) (days_per_week : ℕ) (weekly_pay : ℚ) : ℚ :=
  weekly_pay / (tasks_per_day * days_per_week)

/-- Theorem stating that Tim's pay per task is $1.20 --/
theorem tim_pay_per_task :
  pay_per_task 100 6 720 = 1.20 := by
  sorry

end tim_pay_per_task_l2115_211568


namespace quadratic_inequality_range_l2115_211555

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) ↔ a ≤ 0 := by sorry

end quadratic_inequality_range_l2115_211555


namespace quadratic_inequality_solution_l2115_211593

theorem quadratic_inequality_solution (a b : ℝ) (h : Set ℝ) : 
  (∀ x, x ∈ h ↔ (a * x^2 - 3*x + 6 > 4 ∧ (x < 1 ∨ x > b))) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, 
    (c > 2 → {x | 2 < x ∧ x < c} = {x | x^2 - (2 + c)*x + 2*c < 0}) ∧
    (c < 2 → {x | c < x ∧ x < 2} = {x | x^2 - (2 + c)*x + 2*c < 0}) ∧
    (c = 2 → (∅ : Set ℝ) = {x | x^2 - (2 + c)*x + 2*c < 0})) :=
by sorry

end quadratic_inequality_solution_l2115_211593


namespace stratified_sampling_seniors_l2115_211547

theorem stratified_sampling_seniors (total_students : ℕ) (senior_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : senior_students = 1500)
  (h3 : sample_size = 300)
  (h4 : senior_students ≤ total_students) :
  (senior_students * sample_size) / total_students = 100 := by
  sorry

end stratified_sampling_seniors_l2115_211547


namespace exists_integers_satisfying_inequality_l2115_211535

theorem exists_integers_satisfying_inequality :
  ∃ (A B : ℤ), (0.999 : ℝ) < (A : ℝ) + (B : ℝ) * Real.sqrt 2 ∧ (A : ℝ) + (B : ℝ) * Real.sqrt 2 < 1 :=
by sorry

end exists_integers_satisfying_inequality_l2115_211535


namespace inequality_solution_l2115_211581

theorem inequality_solution (a : ℝ) :
  (a = 0 → ¬∃ x, (1 - a * x)^2 < 1) ∧
  (a < 0 → ∀ x, (1 - a * x)^2 < 1 ↔ (2 / a < x ∧ x < 0)) ∧
  (a > 0 → ∀ x, (1 - a * x)^2 < 1 ↔ (0 < x ∧ x < 2 / a)) :=
by sorry

end inequality_solution_l2115_211581


namespace parabola_two_distinct_roots_l2115_211503

/-- Given a real number m, the quadratic equation x^2 - (2m-1)x + (m^2 - m) = 0 has two distinct real roots. -/
theorem parabola_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 - (2*m - 1)*x₁ + (m^2 - m) = 0 ∧
    x₂^2 - (2*m - 1)*x₂ + (m^2 - m) = 0 :=
sorry

end parabola_two_distinct_roots_l2115_211503


namespace odd_function_max_to_min_l2115_211599

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has a maximum value on [a, b] if there exists a point c in [a, b] 
    such that f(c) ≥ f(x) for all x in [a, b] -/
def HasMaxOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c

/-- A function f has a minimum value on [a, b] if there exists a point c in [a, b] 
    such that f(c) ≤ f(x) for all x in [a, b] -/
def HasMinOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f c ≤ f x

theorem odd_function_max_to_min (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) 
  (h2 : IsOdd f) (h3 : HasMaxOn f a b) : HasMinOn f (-b) (-a) := by
  sorry

end odd_function_max_to_min_l2115_211599


namespace airplane_passenger_ratio_l2115_211575

/-- Given an airplane with 80 passengers, of which 30 are men, prove that the ratio of men to women is 3:5. -/
theorem airplane_passenger_ratio :
  let total_passengers : ℕ := 80
  let num_men : ℕ := 30
  let num_women : ℕ := total_passengers - num_men
  (num_men : ℚ) / (num_women : ℚ) = 3 / 5 := by
  sorry

end airplane_passenger_ratio_l2115_211575


namespace student_height_survey_is_comprehensive_l2115_211506

/-- Represents a survey --/
structure Survey where
  population : ℕ
  measurementType : Type
  isFeasible : Bool

/-- Defines the conditions for a comprehensive survey --/
def isComprehensiveSurvey (s : Survey) : Prop :=
  s.population ≤ 100 ∧ s.isFeasible = true

/-- Represents the survey of students' heights in a class --/
def studentHeightSurvey : Survey :=
  { population := 45,
    measurementType := ℝ,
    isFeasible := true }

/-- Theorem stating that the student height survey is a comprehensive survey --/
theorem student_height_survey_is_comprehensive :
  isComprehensiveSurvey studentHeightSurvey :=
by
  sorry


end student_height_survey_is_comprehensive_l2115_211506


namespace complex_modulus_equality_l2115_211598

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end complex_modulus_equality_l2115_211598


namespace hot_dog_bun_packages_l2115_211544

theorem hot_dog_bun_packages : ∃ n : ℕ, n > 0 ∧ 12 * n % 9 = 0 ∧ ∀ m : ℕ, m > 0 → 12 * m % 9 = 0 → n ≤ m := by
  sorry

end hot_dog_bun_packages_l2115_211544


namespace total_distance_is_6300_l2115_211515

/-- The distance Bomin walked in kilometers -/
def bomin_km : ℝ := 2

/-- The additional distance Bomin walked in meters -/
def bomin_additional_m : ℝ := 600

/-- The distance Yunshik walked in meters -/
def yunshik_m : ℝ := 3700

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The total distance walked by Bomin and Yunshik in meters -/
def total_distance : ℝ := (bomin_km * km_to_m + bomin_additional_m) + yunshik_m

theorem total_distance_is_6300 : total_distance = 6300 := by
  sorry

end total_distance_is_6300_l2115_211515


namespace sequence_general_term_l2115_211504

theorem sequence_general_term (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) = a n / (1 + 3 * a n)) →
  a 1 = 2 →
  ∀ n : ℕ, n ≥ 1 → a n = 2 / (6 * n - 5) :=
by sorry

end sequence_general_term_l2115_211504


namespace rectangle_triangle_equal_area_l2115_211522

/-- Given a rectangle with perimeter 60 and length 3 times its width, 
    and a triangle with height 36, if their areas are equal, 
    then the base of the triangle (which is also one side of the rectangle) is 9.375. -/
theorem rectangle_triangle_equal_area (w : ℝ) (x : ℝ) : 
  (2 * (w + 3*w) = 60) →  -- Rectangle perimeter is 60
  (w * (3*w) = (1/2) * 36 * x) →  -- Rectangle and triangle have equal area
  x = 9.375 := by sorry

end rectangle_triangle_equal_area_l2115_211522


namespace conjugate_complex_abs_l2115_211538

theorem conjugate_complex_abs (α β : ℂ) : 
  (∃ (x y : ℝ), α = x + y * I ∧ β = x - y * I) →  -- α and β are conjugates
  (∃ (r : ℝ), α / β^2 = r) →                     -- α/β² is real
  Complex.abs (α - β) = 4 * Real.sqrt 3 →        -- |α - β| = 4√3
  Complex.abs α = 4 :=                           -- |α| = 4
by sorry

end conjugate_complex_abs_l2115_211538


namespace integer_solutions_of_manhattan_distance_equation_l2115_211510

def solution_set : Set (ℤ × ℤ) := {(2,2), (2,0), (3,1), (1,1)}

theorem integer_solutions_of_manhattan_distance_equation :
  {(x, y) : ℤ × ℤ | |x - 2| + |y - 1| = 1} = solution_set := by sorry

end integer_solutions_of_manhattan_distance_equation_l2115_211510


namespace rectangle_circles_l2115_211574

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end rectangle_circles_l2115_211574


namespace train_length_is_500_l2115_211513

/-- The length of a train that passes a pole in 50 seconds and a 500 m long platform in 100 seconds -/
def train_length : ℝ := by sorry

/-- The time it takes for the train to pass a pole -/
def pole_passing_time : ℝ := 50

/-- The time it takes for the train to pass a platform -/
def platform_passing_time : ℝ := 100

/-- The length of the platform -/
def platform_length : ℝ := 500

theorem train_length_is_500 :
  train_length = 500 :=
by
  have h1 : train_length / pole_passing_time = (train_length + platform_length) / platform_passing_time :=
    by sorry
  sorry

#check train_length_is_500

end train_length_is_500_l2115_211513


namespace unique_solution_condition_inequality_condition_l2115_211592

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The function g(x) = a|x-1| -/
def g (a x : ℝ) : ℝ := a * |x - 1|

/-- If |f(x)| = g(x) has only one real solution, then a < 0 -/
theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, |f x| = g a x) → a < 0 := by sorry

/-- If f(x) ≥ g(x) for all x ∈ ℝ, then a ≤ -2 -/
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 := by sorry

end unique_solution_condition_inequality_condition_l2115_211592


namespace parabola_y_relationship_l2115_211553

theorem parabola_y_relationship (x₁ x₂ y₁ y₂ : ℝ) : 
  (y₁ = x₁^2 - 3) →  -- Point A lies on the parabola
  (y₂ = x₂^2 - 3) →  -- Point B lies on the parabola
  (0 < x₁) →         -- x₁ is positive
  (x₁ < x₂) →        -- x₁ is less than x₂
  y₁ < y₂ :=         -- Conclusion: y₁ is less than y₂
by sorry

end parabola_y_relationship_l2115_211553


namespace salary_increase_l2115_211537

/-- Given an original salary and a salary increase, 
    proves that the new salary is $90,000 if the percent increase is 38.46153846153846% --/
theorem salary_increase (S : ℝ) (increase : ℝ) : 
  increase = 25000 →
  (increase / S) * 100 = 38.46153846153846 →
  S + increase = 90000 := by
  sorry

end salary_increase_l2115_211537


namespace power_function_below_identity_l2115_211577

theorem power_function_below_identity (α : ℝ) : 
  (∀ x : ℝ, x > 1 → x^α < x) → α < 1 := by sorry

end power_function_below_identity_l2115_211577


namespace lidia_money_is_66_l2115_211572

/-- The amount of money Lidia has for buying apps -/
def lidia_money (app_cost : ℝ) (num_apps : ℕ) (remaining : ℝ) : ℝ :=
  app_cost * (num_apps : ℝ) + remaining

/-- Theorem stating that Lidia has $66 for buying apps -/
theorem lidia_money_is_66 :
  lidia_money 4 15 6 = 66 := by
  sorry

end lidia_money_is_66_l2115_211572


namespace midpoint_octagon_area_l2115_211540

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

theorem midpoint_octagon_area (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o := by
  sorry

end midpoint_octagon_area_l2115_211540


namespace faster_speed_problem_l2115_211576

/-- Proves that the faster speed is 15 km/hr given the conditions of the problem -/
theorem faster_speed_problem (actual_distance : ℝ) (original_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 10)
  (h2 : original_speed = 5)
  (h3 : additional_distance = 20) :
  let time := actual_distance / original_speed
  let faster_speed := (actual_distance + additional_distance) / time
  faster_speed = 15 := by sorry

end faster_speed_problem_l2115_211576


namespace time_difference_is_56_minutes_l2115_211542

def minnie_uphill_distance : ℝ := 12
def minnie_flat_distance : ℝ := 18
def minnie_downhill_distance : ℝ := 22
def minnie_uphill_speed : ℝ := 4
def minnie_flat_speed : ℝ := 25
def minnie_downhill_speed : ℝ := 32

def penny_downhill_distance : ℝ := 22
def penny_flat_distance : ℝ := 18
def penny_uphill_distance : ℝ := 12
def penny_downhill_speed : ℝ := 15
def penny_flat_speed : ℝ := 35
def penny_uphill_speed : ℝ := 8

theorem time_difference_is_56_minutes :
  let minnie_time := minnie_uphill_distance / minnie_uphill_speed +
                     minnie_flat_distance / minnie_flat_speed +
                     minnie_downhill_distance / minnie_downhill_speed
  let penny_time := penny_downhill_distance / penny_downhill_speed +
                    penny_flat_distance / penny_flat_speed +
                    penny_uphill_distance / penny_uphill_speed
  (minnie_time - penny_time) * 60 = 56 := by
  sorry

end time_difference_is_56_minutes_l2115_211542


namespace total_fruits_eaten_l2115_211502

/-- Prove that the total number of fruits eaten by three dogs is 240 given the specified conditions -/
theorem total_fruits_eaten (dog1_apples dog2_blueberries dog3_bonnies : ℕ) : 
  dog3_bonnies = 60 →
  dog2_blueberries = (3 * dog3_bonnies) / 4 →
  dog1_apples = 3 * dog2_blueberries →
  dog1_apples + dog2_blueberries + dog3_bonnies = 240 := by
  sorry

#check total_fruits_eaten

end total_fruits_eaten_l2115_211502


namespace no_perfect_square_300_ones_l2115_211566

/-- Represents the count of digits '1' in the decimal representation of a number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- Checks if a number's decimal representation contains only '0' and '1' -/
def only_zero_and_one (n : ℕ) : Prop := sorry

/-- Theorem: There does not exist a perfect square integer with exactly 300 digits of '1' 
    and no other digits except '0' in its decimal representation -/
theorem no_perfect_square_300_ones : 
  ¬ ∃ (n : ℕ), count_ones n = 300 ∧ only_zero_and_one n ∧ ∃ (k : ℕ), n = k^2 := by
  sorry

end no_perfect_square_300_ones_l2115_211566


namespace sum_of_roots_l2115_211548

theorem sum_of_roots (x : ℝ) : 
  (∃ a b : ℝ, (2*x + 3)*(x - 4) + (2*x + 3)*(x - 6) = 0 ∧ 
   {y : ℝ | (2*y + 3)*(y - 4) + (2*y + 3)*(y - 6) = 0} = {a, b} ∧
   a + b = 7/2) :=
by
  sorry

end sum_of_roots_l2115_211548


namespace cafeteria_apples_l2115_211516

theorem cafeteria_apples (initial : ℕ) : 
  initial - 20 + 28 = 46 → initial = 38 := by
  sorry

end cafeteria_apples_l2115_211516


namespace intersection_of_M_and_N_l2115_211526

-- Define set M
def M : Set ℝ := {x | x * (x - 5) ≤ 6}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 6} := by
  sorry

end intersection_of_M_and_N_l2115_211526


namespace max_rooks_on_chessboard_l2115_211531

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a rook placement on a chessboard --/
structure RookPlacement :=
  (board : Chessboard)
  (num_rooks : ℕ)

/-- Predicate to check if a rook placement satisfies the condition --/
def satisfies_condition (placement : RookPlacement) : Prop :=
  ∀ (removed : ℕ), removed < placement.num_rooks →
    ∃ (square : ℕ × ℕ), 
      square.1 ≤ placement.board.size ∧ 
      square.2 ≤ placement.board.size ∧
      (∀ (rook : ℕ × ℕ), rook ≠ removed → 
        (rook.1 ≠ square.1 ∧ rook.2 ≠ square.2))

/-- The main theorem --/
theorem max_rooks_on_chessboard :
  ∃ (placement : RookPlacement),
    placement.board.size = 10 ∧
    placement.num_rooks = 81 ∧
    satisfies_condition placement ∧
    (∀ (other_placement : RookPlacement),
      other_placement.board.size = 10 →
      satisfies_condition other_placement →
      other_placement.num_rooks ≤ 81) :=
sorry

end max_rooks_on_chessboard_l2115_211531


namespace temperature_difference_l2115_211511

/-- The temperature difference problem -/
theorem temperature_difference (T_NY T_M T_SD : ℝ) : 
  T_NY = 80 →
  T_M = T_NY + 10 →
  (T_NY + T_M + T_SD) / 3 = 95 →
  T_SD - T_M = 25 := by
  sorry

end temperature_difference_l2115_211511
