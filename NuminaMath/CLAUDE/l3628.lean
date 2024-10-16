import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_2549_l3628_362840

theorem sqrt_product_plus_one_equals_2549 :
  Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_2549_l3628_362840


namespace NUMINAMATH_CALUDE_lost_shoes_count_l3628_362818

/-- Given a number of initial shoe pairs and remaining matching pairs,
    calculates the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * remaining_pairs

/-- Theorem stating that with 24 initial pairs and 19 remaining pairs,
    10 individual shoes were lost. -/
theorem lost_shoes_count :
  shoes_lost 24 19 = 10 := by
  sorry


end NUMINAMATH_CALUDE_lost_shoes_count_l3628_362818


namespace NUMINAMATH_CALUDE_smaller_rectangle_perimeter_is_9_l3628_362809

/-- Represents a rectangle with its dimensions and division properties. -/
structure DividedRectangle where
  perimeter : ℝ
  verticalCuts : ℕ
  horizontalCuts : ℕ
  smallRectangles : ℕ
  totalCutLength : ℝ

/-- Calculates the perimeter of a smaller rectangle given a DividedRectangle. -/
def smallRectanglePerimeter (r : DividedRectangle) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating the perimeter of each smaller rectangle is 9 cm under given conditions. -/
theorem smaller_rectangle_perimeter_is_9 (r : DividedRectangle) 
    (h1 : r.perimeter = 96)
    (h2 : r.verticalCuts = 8)
    (h3 : r.horizontalCuts = 11)
    (h4 : r.smallRectangles = 108)
    (h5 : r.totalCutLength = 438) :
    smallRectanglePerimeter r = 9 := by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_perimeter_is_9_l3628_362809


namespace NUMINAMATH_CALUDE_C_always_answers_yes_l3628_362881

-- Define the type of islander
inductive IslanderType
  | Knight
  | Liar

-- Define the islanders
def A : IslanderType := sorry
def B : IslanderType := sorry
def C : IslanderType := sorry

-- Define A's statement
def A_statement : Prop := (B = C)

-- Define the question asked to C
def question_to_C : Prop := (A = B)

-- Define C's answer
def C_answer : Prop := 
  match C with
  | IslanderType.Knight => question_to_C
  | IslanderType.Liar => ¬question_to_C

-- Theorem: C will always answer "Yes"
theorem C_always_answers_yes :
  ∀ (A B C : IslanderType),
  (A_statement ↔ (B = C)) →
  C_answer = true :=
sorry

end NUMINAMATH_CALUDE_C_always_answers_yes_l3628_362881


namespace NUMINAMATH_CALUDE_exponent_sum_l3628_362839

theorem exponent_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 8) : a^(m+n) = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l3628_362839


namespace NUMINAMATH_CALUDE_alternating_color_probability_l3628_362805

/-- The probability of drawing 8 balls from a box containing 5 white and 3 black balls,
    such that the draws alternate in color starting with a white ball. -/
theorem alternating_color_probability :
  let total_balls : ℕ := 8
  let white_balls : ℕ := 5
  let black_balls : ℕ := 3
  let total_arrangements : ℕ := Nat.choose total_balls black_balls
  let favorable_arrangements : ℕ := 1
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l3628_362805


namespace NUMINAMATH_CALUDE_total_door_replacement_cost_l3628_362892

/-- The total cost of replacing doors for John -/
theorem total_door_replacement_cost :
  let num_bedroom_doors : ℕ := 3
  let num_outside_doors : ℕ := 2
  let outside_door_cost : ℕ := 20
  let bedroom_door_cost : ℕ := outside_door_cost / 2
  let total_cost : ℕ := num_bedroom_doors * bedroom_door_cost + num_outside_doors * outside_door_cost
  total_cost = 70 := by sorry

end NUMINAMATH_CALUDE_total_door_replacement_cost_l3628_362892


namespace NUMINAMATH_CALUDE_arrangements_equal_78_l3628_362810

/-- The number of different arrangements to select 2 workers for typesetting and 2 for printing
    from a group of 7 workers, where 5 are proficient in typesetting and 4 are proficient in printing. -/
def num_arrangements (total : ℕ) (typesetters : ℕ) (printers : ℕ) (typeset_needed : ℕ) (print_needed : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 78 -/
theorem arrangements_equal_78 :
  num_arrangements 7 5 4 2 2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_equal_78_l3628_362810


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_l3628_362889

/-- Given two spheres inscribed in a dihedral angle, this theorem proves that
    the cosine of the measure of the dihedral angle is 5/9 under specific conditions. -/
theorem dihedral_angle_cosine (α : Real) (R r : Real) (β : Real) :
  -- Two spheres are inscribed in a dihedral angle
  -- The spheres touch each other
  -- R is the radius of the larger sphere, r is the radius of the smaller sphere
  (R = 2 * r) →
  -- The line connecting the centers of the spheres forms a 45° angle with the edge of the dihedral angle
  (β = Real.pi / 4) →
  -- α is the measure of the dihedral angle
  -- The cosine of the measure of the dihedral angle is 5/9
  (Real.cos α = 5 / 9) :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_l3628_362889


namespace NUMINAMATH_CALUDE_paddyfield_warbler_percentage_l3628_362823

/-- Represents the composition of birds in a nature reserve -/
structure BirdPopulation where
  total : ℝ
  hawk_percent : ℝ
  other_percent : ℝ
  kingfisher_to_warbler_ratio : ℝ

/-- Theorem about the percentage of paddyfield-warblers among non-hawks -/
theorem paddyfield_warbler_percentage
  (pop : BirdPopulation)
  (h1 : pop.hawk_percent = 0.3)
  (h2 : pop.other_percent = 0.35)
  (h3 : pop.kingfisher_to_warbler_ratio = 0.25)
  : (((1 - pop.hawk_percent - pop.other_percent) * pop.total) / ((1 - pop.hawk_percent) * pop.total)) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_paddyfield_warbler_percentage_l3628_362823


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3628_362842

/-- The area of a square with diagonal length 8√2 is 64 -/
theorem square_area_from_diagonal : 
  ∀ (s : ℝ), s > 0 → s * s * 2 = (8 * Real.sqrt 2) ^ 2 → s * s = 64 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3628_362842


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3628_362879

theorem multiply_mixed_number : 8 * (11 + 1/4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3628_362879


namespace NUMINAMATH_CALUDE_condition_iff_in_solution_set_l3628_362836

/-- A pair of positive integers (x, y) satisfies the given condition -/
def satisfies_condition (x y : ℕ+) : Prop :=
  ∃ k : ℕ, x^2 * y + x = k * (x * y^2 + 7)

/-- The set of all pairs (x, y) that satisfy the condition -/
def solution_set : Set (ℕ+ × ℕ+) :=
  {p | p = (7, 1) ∨ p = (14, 1) ∨ p = (35, 1) ∨ p = (7, 2) ∨
       ∃ k : ℕ+, p = (7 * k, 7)}

/-- The main theorem stating the equivalence between the condition and the solution set -/
theorem condition_iff_in_solution_set (x y : ℕ+) :
  satisfies_condition x y ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_condition_iff_in_solution_set_l3628_362836


namespace NUMINAMATH_CALUDE_pencil_distribution_l3628_362884

/-- The number of ways to distribute n identical objects among k people, 
    where each person gets at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 8 identical pencils among 4 friends, 
    where each friend has at least one pencil. -/
theorem pencil_distribution : distribute 8 4 = 35 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3628_362884


namespace NUMINAMATH_CALUDE_angle_ADE_measure_l3628_362833

/-- Triangle ABC -/
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

/-- Pentagon ABCDE -/
structure Pentagon :=
  (A B C D E : ℝ)
  (sum_angles : A + B + C + D + E = 540)

/-- Circle circumscribed around a pentagon -/
structure CircumscribedCircle (p : Pentagon) := 
  (is_circumscribed : Bool)

/-- Pentagon with sides tangent to a circle -/
structure TangentPentagon (p : Pentagon) (c : CircumscribedCircle p) :=
  (is_tangent : Bool)

/-- Theorem: In a pentagon ABCDE constructed as described, the measure of angle ADE is 108° -/
theorem angle_ADE_measure 
  (t : Triangle)
  (p : Pentagon)
  (c : CircumscribedCircle p)
  (tp : TangentPentagon p c)
  (h1 : t.A = 60)
  (h2 : t.B = 50)
  (h3 : t.C = 70)
  (h4 : p.D ∈ Set.Ioo 0 (t.A + t.B))  -- D is on side AB
  (h5 : p.E ∈ Set.Ioo (t.A + t.B) (t.A + t.B + t.C))  -- E is on side BC
  : p.D = 108 :=
sorry

end NUMINAMATH_CALUDE_angle_ADE_measure_l3628_362833


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3628_362822

theorem diophantine_equation_solutions :
  ∃! (solutions : Set (ℤ × ℤ)),
    solutions = {(4, 9), (4, -9), (-4, 9), (-4, -9)} ∧
    ∀ (x y : ℤ), (x, y) ∈ solutions ↔ 3 * x^2 + 5 * y^2 = 453 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3628_362822


namespace NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l3628_362883

theorem cos_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 6 + α) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l3628_362883


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l3628_362855

theorem first_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 0.765 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l3628_362855


namespace NUMINAMATH_CALUDE_rogers_final_amount_l3628_362853

def rogers_money (initial : ℕ) (gift : ℕ) (spent : ℕ) : ℕ :=
  initial + gift - spent

theorem rogers_final_amount :
  rogers_money 16 28 25 = 19 := by
  sorry

end NUMINAMATH_CALUDE_rogers_final_amount_l3628_362853


namespace NUMINAMATH_CALUDE_correct_understanding_of_philosophy_l3628_362849

-- Define the characteristics of philosophy
def originatesFromLife (p : Type) : Prop := sorry
def affectsLife (p : Type) : Prop := sorry
def formsSpontaneously (p : Type) : Prop := sorry
def summarizesKnowledge (p : Type) : Prop := sorry

-- Define Yu Wujin's statement
def yuWujinStatement (p : Type) : Prop := sorry

-- Theorem to prove
theorem correct_understanding_of_philosophy (p : Type) :
  yuWujinStatement p ↔ (originatesFromLife p ∧ affectsLife p) :=
sorry

end NUMINAMATH_CALUDE_correct_understanding_of_philosophy_l3628_362849


namespace NUMINAMATH_CALUDE_ratio_unchanged_l3628_362886

theorem ratio_unchanged (a b : ℝ) (h : b ≠ 0) :
  (3 * a) / (b / (1 / 3)) = a / b :=
by sorry

end NUMINAMATH_CALUDE_ratio_unchanged_l3628_362886


namespace NUMINAMATH_CALUDE_dinner_seating_arrangements_l3628_362835

theorem dinner_seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 6) :
  (Nat.choose n k) * (Nat.factorial (k - 1)) = 3360 := by
  sorry

end NUMINAMATH_CALUDE_dinner_seating_arrangements_l3628_362835


namespace NUMINAMATH_CALUDE_point_on_line_l3628_362873

/-- Given three points in a 2D plane, this function checks if they are collinear --/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem point_on_line : are_collinear (0, 4) (-6, 1) (4, 6) := by sorry

end NUMINAMATH_CALUDE_point_on_line_l3628_362873


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3628_362866

theorem fractional_equation_solution :
  ∃ (x : ℝ), (x + 2 ≠ 0 ∧ x - 2 ≠ 0) →
  (1 / (x + 2) + 4 * x / (x^2 - 4) = 1 / (x - 2)) ∧ x = 1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3628_362866


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l3628_362894

variable {a b c x1 x2 : ℝ}

theorem quadratic_roots_transformation (ha : a ≠ 0)
  (hroots : a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ k, k * ((x1 + 2*x2) - x)* ((x2 + 2*x1) - x) = a^2 * x^2 + 3*a*b * x + 2*b^2 + a*c :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l3628_362894


namespace NUMINAMATH_CALUDE_equal_values_l3628_362890

theorem equal_values (p q a b : ℝ) 
  (h1 : p + q = 1)
  (h2 : p * q ≠ 0)
  (h3 : (p / a) + (q / b) = 1 / (p * a + q * b)) :
  a = b := by sorry

end NUMINAMATH_CALUDE_equal_values_l3628_362890


namespace NUMINAMATH_CALUDE_tan_sum_15_30_l3628_362880

theorem tan_sum_15_30 : 
  ∀ (tan : Real → Real),
  (∀ α β, tan (α + β) = (tan α + tan β) / (1 - tan α * tan β)) →
  tan (45 * π / 180) = 1 →
  tan (15 * π / 180) + tan (30 * π / 180) + tan (15 * π / 180) * tan (30 * π / 180) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_15_30_l3628_362880


namespace NUMINAMATH_CALUDE_number_of_brown_dogs_l3628_362817

/-- Given a group of dogs with white, black, and brown colors, 
    prove that the number of brown dogs is 20. -/
theorem number_of_brown_dogs 
  (total : ℕ) 
  (white : ℕ) 
  (black : ℕ) 
  (h1 : total = 45) 
  (h2 : white = 10) 
  (h3 : black = 15) : 
  total - (white + black) = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_brown_dogs_l3628_362817


namespace NUMINAMATH_CALUDE_min_value_of_w_l3628_362871

theorem min_value_of_w (x y : ℝ) :
  let w := 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 34
  w ≥ 71 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_w_l3628_362871


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l3628_362804

theorem set_equality_implies_values (A B : Set ℝ) (x y : ℝ) :
  A = {3, 4, x} → B = {2, 3, y} → A = B → x = 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l3628_362804


namespace NUMINAMATH_CALUDE_M_remainder_mod_32_l3628_362827

def M : ℕ := (List.filter (fun p => Nat.Prime p ∧ p % 2 = 1) (List.range 32)).prod

theorem M_remainder_mod_32 : M % 32 = 17 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_32_l3628_362827


namespace NUMINAMATH_CALUDE_total_cost_before_markup_is_47_l3628_362845

/-- The markup percentage as a decimal -/
def markup : ℚ := 0.10

/-- The selling prices of the three books -/
def sellingPrices : List ℚ := [11.00, 16.50, 24.20]

/-- Calculate the original price before markup -/
def originalPrice (sellingPrice : ℚ) : ℚ := sellingPrice / (1 + markup)

/-- Calculate the total cost before markup -/
def totalCostBeforeMarkup : ℚ := (sellingPrices.map originalPrice).sum

/-- Theorem stating that the total cost before markup is $47.00 -/
theorem total_cost_before_markup_is_47 : totalCostBeforeMarkup = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_before_markup_is_47_l3628_362845


namespace NUMINAMATH_CALUDE_expression_evaluation_l3628_362828

theorem expression_evaluation (x : ℝ) (h : x = 2) : x^2 + 5*x - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3628_362828


namespace NUMINAMATH_CALUDE_shirt_count_l3628_362848

theorem shirt_count (total_shirt_price : ℝ) (total_sweater_price : ℝ) (sweater_count : ℕ) (price_difference : ℝ) :
  total_shirt_price = 400 →
  total_sweater_price = 1500 →
  sweater_count = 75 →
  (total_sweater_price / sweater_count) = (total_shirt_price / (total_shirt_price / 16)) + price_difference →
  price_difference = 4 →
  (total_shirt_price / 16 : ℝ) = 25 :=
by sorry

end NUMINAMATH_CALUDE_shirt_count_l3628_362848


namespace NUMINAMATH_CALUDE_consecutive_blue_gumballs_probability_l3628_362808

theorem consecutive_blue_gumballs_probability :
  let p_pink : ℝ := 0.1428571428571428
  let p_blue : ℝ := 1 - p_pink
  p_blue * p_blue = 0.7346938775510203 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_blue_gumballs_probability_l3628_362808


namespace NUMINAMATH_CALUDE_sector_max_area_l3628_362831

/-- Given a sector with constant perimeter a, prove that the maximum area is a²/16
    and this occurs when the central angle α is 2. -/
theorem sector_max_area (a : ℝ) (h : a > 0) :
  ∃ (S : ℝ) (α : ℝ),
    S = a^2 / 16 ∧
    α = 2 ∧
    ∀ (S' : ℝ) (α' : ℝ),
      (∃ (r : ℝ), 2 * r + r * α' = a ∧ S' = r^2 * α' / 2) →
      S' ≤ S :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l3628_362831


namespace NUMINAMATH_CALUDE_door_lock_problem_l3628_362832

def num_buttons : ℕ := 10
def buttons_to_press : ℕ := 3
def time_per_attempt : ℕ := 2

def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

theorem door_lock_problem :
  (total_combinations * time_per_attempt = 240) ∧
  ((1 + total_combinations) / 2 * time_per_attempt = 121) ∧
  (((60 / time_per_attempt) - 1 : ℚ) / total_combinations = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_door_lock_problem_l3628_362832


namespace NUMINAMATH_CALUDE_brick_weighs_32_kg_l3628_362838

-- Define the weight of one brick
def brick_weight : ℝ := sorry

-- Define the weight of one statue
def statue_weight : ℝ := sorry

-- Theorem stating the weight of one brick is 32 kg
theorem brick_weighs_32_kg : brick_weight = 32 :=
  by
  -- Condition 1: 5 bricks weigh the same as 4 statues
  have h1 : 5 * brick_weight = 4 * statue_weight := sorry
  -- Condition 2: 2 statues weigh 80 kg
  have h2 : 2 * statue_weight = 80 := sorry
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_brick_weighs_32_kg_l3628_362838


namespace NUMINAMATH_CALUDE_cuboctahedron_volume_side_length_one_l3628_362821

/-- A cuboctahedron is a polyhedron with 8 triangular faces and 6 square faces. -/
structure Cuboctahedron where
  side_length : ℝ

/-- The volume of a cuboctahedron. -/
noncomputable def volume (c : Cuboctahedron) : ℝ :=
  (5 * Real.sqrt 2) / 3

/-- Theorem: The volume of a cuboctahedron with side length 1 is (5 * √2) / 3. -/
theorem cuboctahedron_volume_side_length_one :
  volume { side_length := 1 } = (5 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cuboctahedron_volume_side_length_one_l3628_362821


namespace NUMINAMATH_CALUDE_set_cardinality_lower_bound_l3628_362850

variable (m : ℕ) (A : Finset ℤ) (B : Fin m → Finset ℤ)

theorem set_cardinality_lower_bound
  (h_m : m ≥ 2)
  (h_subset : ∀ i : Fin m, B i ⊆ A)
  (h_sum : ∀ i : Fin m, (B i).sum id = m ^ (i : ℕ).succ) :
  A.card ≥ m / 2 :=
sorry

end NUMINAMATH_CALUDE_set_cardinality_lower_bound_l3628_362850


namespace NUMINAMATH_CALUDE_parabola_b_value_l3628_362800

/-- A parabola passing through three given points has a specific b value -/
theorem parabola_b_value (b c : ℚ) :
  ((-1)^2 + b*(-1) + c = -11) →
  (3^2 + b*3 + c = 17) →
  (2^2 + b*2 + c = 6) →
  b = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3628_362800


namespace NUMINAMATH_CALUDE_smallest_number_l3628_362895

theorem smallest_number (a b c d : ℝ) (ha : a = 0) (hb : b = -1/2) (hc : c = -1) (hd : d = Real.sqrt 2) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l3628_362895


namespace NUMINAMATH_CALUDE_greatest_k_value_l3628_362826

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 7 = 0 ∧ 
    x₂^2 + k*x₂ + 7 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 85) →
  k ≤ Real.sqrt 113 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l3628_362826


namespace NUMINAMATH_CALUDE_modified_geometric_progression_sum_of_squares_l3628_362854

/-- The sum of squares of a modified geometric progression -/
theorem modified_geometric_progression_sum_of_squares
  (b c s : ℝ) (h : abs s < 1) :
  let modifiedSum := (c^2 * b^2 * s^4) / (1 - s)
  let modifiedSequence := fun n => if n < 3 then b * s^(n-1) else c * b * s^(n-1)
  ∑' n, (modifiedSequence n)^2 = modifiedSum :=
sorry

end NUMINAMATH_CALUDE_modified_geometric_progression_sum_of_squares_l3628_362854


namespace NUMINAMATH_CALUDE_dhoni_doll_expenditure_l3628_362843

theorem dhoni_doll_expenditure :
  ∀ (total_spent : ℕ) (large_price small_price : ℕ),
    large_price = 6 →
    small_price = large_price - 2 →
    (total_spent / small_price) - (total_spent / large_price) = 25 →
    total_spent = 300 := by
  sorry

end NUMINAMATH_CALUDE_dhoni_doll_expenditure_l3628_362843


namespace NUMINAMATH_CALUDE_not_heart_zero_sum_property_l3628_362803

def heart (x y : ℝ) : ℝ := |x + y|

theorem not_heart_zero_sum_property : ¬ ∀ x y : ℝ, (heart x 0 + heart 0 y = heart x y) := by
  sorry

end NUMINAMATH_CALUDE_not_heart_zero_sum_property_l3628_362803


namespace NUMINAMATH_CALUDE_octopus_leg_configuration_l3628_362891

-- Define the possible number of legs for an octopus
inductive LegCount : Type where
  | six : LegCount
  | seven : LegCount
  | eight : LegCount

-- Define the colors of the octopuses
inductive Color : Type where
  | blue : Color
  | green : Color
  | yellow : Color
  | red : Color

-- Define a function to determine if an octopus is telling the truth
def isTruthful (legs : LegCount) : Prop :=
  match legs with
  | LegCount.six => True
  | LegCount.seven => False
  | LegCount.eight => True

-- Define a function to convert LegCount to a natural number
def legCountToNat (legs : LegCount) : Nat :=
  match legs with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

-- Define the claims made by each octopus
def claim (color : Color) : Nat :=
  match color with
  | Color.blue => 28
  | Color.green => 27
  | Color.yellow => 26
  | Color.red => 25

-- Define the theorem
theorem octopus_leg_configuration :
  ∃ (legs : Color → LegCount),
    (legs Color.green = LegCount.six) ∧
    (legs Color.blue = LegCount.seven) ∧
    (legs Color.yellow = LegCount.seven) ∧
    (legs Color.red = LegCount.seven) ∧
    (∀ c, isTruthful (legs c) ↔ (legCountToNat (legs Color.blue) + legCountToNat (legs Color.green) + legCountToNat (legs Color.yellow) + legCountToNat (legs Color.red) = claim c)) :=
sorry

end NUMINAMATH_CALUDE_octopus_leg_configuration_l3628_362891


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l3628_362897

/-- Given two parabolas that intersect the coordinate axes at four points forming a rectangle with area 36, prove that the sum of their coefficients is 4/27 -/
theorem parabola_intersection_sum (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x^2 + 3 ∧ (x = 0 ∨ y = 0)) ∧ 
  (∃ x y : ℝ, y = 7 - b * x^2 ∧ (x = 0 ∨ y = 0)) ∧
  (∃ x1 x2 y1 y2 : ℝ, 
    (x1 ≠ 0 ∧ y1 = 0 ∧ y1 = a * x1^2 + 3) ∧
    (x2 ≠ 0 ∧ y2 = 0 ∧ y2 = 7 - b * x2^2) ∧
    (x1 * y2 = 36)) →
  a + b = 4/27 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l3628_362897


namespace NUMINAMATH_CALUDE_gum_pack_size_l3628_362847

theorem gum_pack_size (initial_peach : ℕ) (initial_mint : ℕ) (y : ℚ) 
  (h1 : initial_peach = 40)
  (h2 : initial_mint = 50)
  (h3 : y > 0) :
  (initial_peach - 2 * y) / initial_mint = initial_peach / (initial_mint + 3 * y) → 
  y = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_gum_pack_size_l3628_362847


namespace NUMINAMATH_CALUDE_sum_of_root_products_l3628_362846

theorem sum_of_root_products (p q r : ℂ) : 
  (4 * p^3 - 2 * p^2 + 13 * p - 9 = 0) →
  (4 * q^3 - 2 * q^2 + 13 * q - 9 = 0) →
  (4 * r^3 - 2 * r^2 + 13 * r - 9 = 0) →
  p * q + p * r + q * r = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_root_products_l3628_362846


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3628_362857

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n - 1
  let r : ℕ := 4^s - s
  r = 16377 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3628_362857


namespace NUMINAMATH_CALUDE_new_conveyor_belt_time_l3628_362858

theorem new_conveyor_belt_time (old_time new_time combined_time : ℝ) 
  (h1 : old_time = 21)
  (h2 : combined_time = 8.75)
  (h3 : 1 / old_time + 1 / new_time = 1 / combined_time) : 
  new_time = 15 := by
sorry

end NUMINAMATH_CALUDE_new_conveyor_belt_time_l3628_362858


namespace NUMINAMATH_CALUDE_sqrt_equality_l3628_362819

theorem sqrt_equality (x : ℝ) (hx : x > 0) : -x * Real.sqrt (2 / x) = -Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l3628_362819


namespace NUMINAMATH_CALUDE_tangent_lines_at_k_zero_equal_angles_point_l3628_362876

-- Define the curve C and the line
def C (x y : ℝ) : Prop := x^2 = 4*y
def L (k a x y : ℝ) : Prop := y = k*x + a

-- Define the intersection points M and N
def intersection_points (k a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ C x y ∧ L k a x y}

-- Theorem for tangent lines when k = 0
theorem tangent_lines_at_k_zero (a : ℝ) (ha : a > 0) :
  ∃ (M N : ℝ × ℝ), M ∈ intersection_points 0 a ∧ N ∈ intersection_points 0 a ∧
  (∃ (x y : ℝ), M = (x, y) ∧ Real.sqrt a * x - y - a = 0) ∧
  (∃ (x y : ℝ), N = (x, y) ∧ Real.sqrt a * x + y + a = 0) :=
sorry

-- Theorem for the existence of point P
theorem equal_angles_point (a : ℝ) (ha : a > 0) :
  ∃ (P : ℝ × ℝ), P.1 = 0 ∧
  ∀ (k : ℝ) (M N : ℝ × ℝ), M ∈ intersection_points k a → N ∈ intersection_points k a →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), M = (x₁, y₁) ∧ N = (x₂, y₂) ∧
   (y₁ - P.2) / x₁ = -(y₂ - P.2) / x₂) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_at_k_zero_equal_angles_point_l3628_362876


namespace NUMINAMATH_CALUDE_coefficient_x9_eq_240_l3628_362802

/-- The coefficient of x^9 in the expansion of (1+3x-2x^2)^5 -/
def coefficient_x9 : ℤ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^9 in (1+3x-2x^2)^5 is 240 -/
theorem coefficient_x9_eq_240 : coefficient_x9 = 240 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x9_eq_240_l3628_362802


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3628_362830

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Generates the nth element of a systematic sample -/
def nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_element + (n - 1) * s.interval

theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population_size = 52)
  (h2 : s.sample_size = 4)
  (h3 : s.first_element = 5)
  (h4 : nth_element s 3 = 31)
  (h5 : nth_element s 4 = 44) :
  nth_element s 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3628_362830


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l3628_362882

/-- Given a geometric progression a, b, c, d, prove that 
    (a^2 + b^2 + c^2)(b^2 + c^2 + d^2) = (ab + bc + cd)^2 -/
theorem geometric_progression_equality 
  (a b c d : ℝ) (h : ∃ (q : ℝ), b = a * q ∧ c = b * q ∧ d = c * q) : 
  (a^2 + b^2 + c^2) * (b^2 + c^2 + d^2) = (a*b + b*c + c*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l3628_362882


namespace NUMINAMATH_CALUDE_abs_sum_min_value_l3628_362837

theorem abs_sum_min_value :
  (∀ x : ℝ, |x + 1| + |2 - x| ≥ 3) ∧
  (∃ x : ℝ, |x + 1| + |2 - x| = 3) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_min_value_l3628_362837


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3628_362801

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 1 →                     -- first term condition
  d ≠ 0 →                       -- non-zero common difference
  (a 2) ^ 2 = a 1 * a 5 →       -- geometric sequence condition
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3628_362801


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l3628_362868

theorem dot_product_specific_vectors (α : ℝ) : 
  let a : ℝ × ℝ := (Real.cos α, Real.sin α)
  let b : ℝ × ℝ := (Real.cos (π/3 + α), Real.sin (π/3 + α))
  (a.1 * b.1 + a.2 * b.2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l3628_362868


namespace NUMINAMATH_CALUDE_sixth_root_equation_solution_l3628_362834

theorem sixth_root_equation_solution (x : ℝ) :
  (x^2 * (x^4)^(1/3))^(1/6) = 4 ↔ x = 4^(18/5) := by sorry

end NUMINAMATH_CALUDE_sixth_root_equation_solution_l3628_362834


namespace NUMINAMATH_CALUDE_abs_even_and_increasing_l3628_362861

-- Define the absolute value function
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_abs_even_and_increasing_l3628_362861


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l3628_362865

theorem quadratic_real_solutions (x y : ℝ) :
  (9 * y^2 + 6 * x * y + 2 * x + 10 = 0) ↔ (x ≤ -10/3 ∨ x ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l3628_362865


namespace NUMINAMATH_CALUDE_julio_fish_count_l3628_362806

/-- Calculates the number of fish Julio has after fishing for a given number of hours and losing some fish. -/
def fish_count (catch_rate : ℕ) (hours : ℕ) (fish_lost : ℕ) : ℕ :=
  catch_rate * hours - fish_lost

/-- Theorem stating that Julio has 48 fish after 9 hours of fishing at 7 fish per hour and losing 15 fish. -/
theorem julio_fish_count :
  fish_count 7 9 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_julio_fish_count_l3628_362806


namespace NUMINAMATH_CALUDE_multiply_x_equals_5_l3628_362874

theorem multiply_x_equals_5 (x y : ℝ) (h1 : x * y ≠ 0) 
  (h2 : (1/5 * x) / (1/6 * y) = 0.7200000000000001) : 
  ∃ n : ℝ, n * x = 3 * y ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_x_equals_5_l3628_362874


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3628_362893

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) :
  ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3628_362893


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_max_on_interval_l3628_362824

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Theorem for monotonically decreasing intervals
theorem f_monotone_decreasing (a : ℝ) :
  ∀ x, (x < -1 ∨ x > 3) → (∀ y, y > x → f a y < f a x) :=
sorry

-- Theorem for minimum and maximum values when a = -2
theorem f_min_max_on_interval :
  let a := -2
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f a y ≤ f a x) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a x ≥ -7) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a x ≤ 20) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a x = -7) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a x = 20) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_max_on_interval_l3628_362824


namespace NUMINAMATH_CALUDE_line_trig_identity_l3628_362807

/-- Given a line with direction vector (-1, 2) and inclination angle α, 
    prove that sin(2α) - cos²(α) - 1 = -2 -/
theorem line_trig_identity (α : Real) (h : Real.tan α = -2) : 
  Real.sin (2 * α) - Real.cos α ^ 2 - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_trig_identity_l3628_362807


namespace NUMINAMATH_CALUDE_jerrys_shelf_l3628_362872

theorem jerrys_shelf (books : ℕ) (added_figures : ℕ) (difference : ℕ) : 
  books = 7 → added_figures = 2 → difference = 2 →
  ∃ initial_figures : ℕ, 
    initial_figures = 3 ∧ 
    books = (initial_figures + added_figures) + difference :=
by sorry

end NUMINAMATH_CALUDE_jerrys_shelf_l3628_362872


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l3628_362811

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l3628_362811


namespace NUMINAMATH_CALUDE_percentage_relationship_l3628_362878

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.375)) :
  y = x * 1.6 :=
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l3628_362878


namespace NUMINAMATH_CALUDE_unique_negative_solution_implies_positive_a_l3628_362813

theorem unique_negative_solution_implies_positive_a (a : ℝ) : 
  (∃! x : ℝ, (abs x = 2 * x + a) ∧ (x < 0)) → a > 0 := by
sorry

end NUMINAMATH_CALUDE_unique_negative_solution_implies_positive_a_l3628_362813


namespace NUMINAMATH_CALUDE_parabola_symmetric_axis_l3628_362820

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (1/2) * x^2 - 6*x + 21

/-- The symmetric axis of the parabola -/
def symmetric_axis (x : ℝ) : Prop :=
  x = 6

/-- Theorem: The symmetric axis of the given parabola is x = 6 -/
theorem parabola_symmetric_axis :
  ∀ x y : ℝ, parabola x y → symmetric_axis x :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_axis_l3628_362820


namespace NUMINAMATH_CALUDE_tadpole_fish_difference_l3628_362867

def initial_fish : ℕ := 50
def tadpole_ratio : ℕ := 3
def fish_caught : ℕ := 7
def tadpole_development_ratio : ℚ := 1/2

theorem tadpole_fish_difference : 
  (tadpole_ratio * initial_fish) * tadpole_development_ratio - (initial_fish - fish_caught) = 32 := by
  sorry

end NUMINAMATH_CALUDE_tadpole_fish_difference_l3628_362867


namespace NUMINAMATH_CALUDE_marys_max_earnings_l3628_362814

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : Nat
  regularRate : ℕ
  overtimeRate1 : ℕ
  overtimeRate2 : ℕ
  weekendBonus : ℕ
  milestoneBonus : ℕ

/-- Calculates the maximum weekly earnings based on the given work schedule --/
def maxWeeklyEarnings (schedule : WorkSchedule) : ℕ :=
  let regularPay := schedule.regularRate * 40
  let overtimePay1 := schedule.overtimeRate1 * 10
  let overtimePay2 := schedule.overtimeRate2 * 10
  let weekendBonus := schedule.weekendBonus * 2
  regularPay + overtimePay1 + overtimePay2 + weekendBonus + schedule.milestoneBonus

/-- Mary's work schedule --/
def marysSchedule : WorkSchedule := {
  maxHours := 60
  regularRate := 10
  overtimeRate1 := 12
  overtimeRate2 := 15
  weekendBonus := 50
  milestoneBonus := 100
}

/-- Theorem stating that Mary's maximum weekly earnings are $875 --/
theorem marys_max_earnings :
  maxWeeklyEarnings marysSchedule = 875 := by
  sorry


end NUMINAMATH_CALUDE_marys_max_earnings_l3628_362814


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l3628_362860

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral (P : Type*) [MetricSpace P] :=
  (A B C D : P)
  (cyclic : ∃ (center : P) (radius : ℝ), dist center A = radius ∧ dist center B = radius ∧ dist center C = radius ∧ dist center D = radius)

/-- The inequality for cyclic quadrilaterals -/
theorem cyclic_quadrilateral_inequality {P : Type*} [MetricSpace P] (ABCD : CyclicQuadrilateral P) :
  |dist ABCD.A ABCD.B - dist ABCD.C ABCD.D| + |dist ABCD.A ABCD.D - dist ABCD.B ABCD.C| ≥ 2 * |dist ABCD.A ABCD.C - dist ABCD.B ABCD.D| :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l3628_362860


namespace NUMINAMATH_CALUDE_smallest_number_quotient_remainder_difference_l3628_362851

theorem smallest_number_quotient_remainder_difference : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 5 = 0) ∧
  (n / 5 > n % 34) ∧
  (∀ m : ℕ, m > 0 → m % 5 = 0 → m / 5 > m % 34 → m ≥ n) ∧
  (n / 5 - n % 34 = 8) := by
sorry

end NUMINAMATH_CALUDE_smallest_number_quotient_remainder_difference_l3628_362851


namespace NUMINAMATH_CALUDE_train_speed_Q_l3628_362844

/-- The distance between stations P and Q in kilometers -/
def distance_PQ : ℝ := 65

/-- The speed of the train starting from station P in kilometers per hour -/
def speed_P : ℝ := 20

/-- The time difference between the start of the two trains in hours -/
def time_difference : ℝ := 1

/-- The total time until the trains meet in hours -/
def total_time : ℝ := 2

/-- The speed of the train starting from station Q in kilometers per hour -/
def speed_Q : ℝ := 25

theorem train_speed_Q : speed_Q = (distance_PQ - speed_P * total_time) / (total_time - time_difference) :=
sorry

end NUMINAMATH_CALUDE_train_speed_Q_l3628_362844


namespace NUMINAMATH_CALUDE_eight_queens_exists_l3628_362863

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Checks if two positions are on the same diagonal -/
def sameDiagonal (p1 p2 : Position) : Prop :=
  (p1.row.val : Int) - (p1.col.val : Int) = (p2.row.val : Int) - (p2.col.val : Int) ∨
  (p1.row.val : Int) + (p1.col.val : Int) = (p2.row.val : Int) + (p2.col.val : Int)

/-- Checks if two queens threaten each other -/
def threaten (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col ∨ sameDiagonal p1 p2

/-- Represents an arrangement of eight queens on the chessboard -/
def QueenArrangement := Fin 8 → Position

/-- Checks if a queen arrangement is valid (no queens threaten each other) -/
def validArrangement (arrangement : QueenArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → ¬threaten (arrangement i) (arrangement j)

/-- Theorem: There exists a valid arrangement of eight queens on an 8x8 chessboard -/
theorem eight_queens_exists : ∃ arrangement : QueenArrangement, validArrangement arrangement :=
sorry

end NUMINAMATH_CALUDE_eight_queens_exists_l3628_362863


namespace NUMINAMATH_CALUDE_complete_square_l3628_362829

theorem complete_square (x : ℝ) : 
  (x^2 + 6*x + 5 = 0) ↔ ((x + 3)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_l3628_362829


namespace NUMINAMATH_CALUDE_no_positive_integers_satisfying_equation_l3628_362812

theorem no_positive_integers_satisfying_equation : 
  ¬∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 2) = n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integers_satisfying_equation_l3628_362812


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3628_362825

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 3 / (x - 4)
  ∃ (s : ℝ), s = 4 - Real.sqrt 3 ∧
    f s = 0 ∧
    ∀ (x : ℝ), f x = 0 → x ≥ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3628_362825


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3628_362869

theorem two_digit_number_property (a b m y : ℕ) : 
  (1 ≤ a) → (a ≤ 9) → (1 ≤ b) → (b ≤ 9) → 
  (10 * a + b = m * (a * b)) → 
  (10 * b + a = y * (a + b)) →
  y = 10 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3628_362869


namespace NUMINAMATH_CALUDE_symmetric_function_k_range_l3628_362864

/-- A function f is symmetric if it's monotonic on its domain D and there exists an interval [a,b] ⊆ D such that the range of f on [a,b] is [-b,-a] -/
def IsSymmetric (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  Monotone f ∧ ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Set.image f (Set.Icc a b) = Set.Icc (-b) (-a)

/-- The main theorem stating that if f(x) = √(2 - x) - k is symmetric on (-∞, 2], then k ∈ [2, 9/4) -/
theorem symmetric_function_k_range :
  ∀ k : ℝ, IsSymmetric (fun x ↦ Real.sqrt (2 - x) - k) (Set.Iic 2) →
  k ∈ Set.Icc 2 (9/4) := by sorry

end NUMINAMATH_CALUDE_symmetric_function_k_range_l3628_362864


namespace NUMINAMATH_CALUDE_triangle_inequality_l3628_362877

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3628_362877


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l3628_362815

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 --/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 51)
  (h2 : stream_speed = 17) : 
  (boat_speed + stream_speed) / (boat_speed - stream_speed) = 2 := by
  sorry

#check upstream_downstream_time_ratio

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l3628_362815


namespace NUMINAMATH_CALUDE_candy_problem_l3628_362859

theorem candy_problem :
  ∀ (S : ℕ) (N : ℕ),
    (∀ (i : ℕ), i < N → S / N = (S - S / N - 11)) →
    (S / N > 1) →
    (N > 1) →
    S = 33 :=
by sorry

end NUMINAMATH_CALUDE_candy_problem_l3628_362859


namespace NUMINAMATH_CALUDE_value_of_a_l3628_362870

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3628_362870


namespace NUMINAMATH_CALUDE_eliot_account_balance_l3628_362898

theorem eliot_account_balance (al eliot : ℝ) 
  (h1 : al > eliot)
  (h2 : al - eliot = (1 / 12) * (al + eliot))
  (h3 : 1.1 * al = 1.2 * eliot + 21) :
  eliot = 210 := by
sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l3628_362898


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_36_l3628_362885

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_27_times_36 : unitsDigit (27 * 36) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_36_l3628_362885


namespace NUMINAMATH_CALUDE_volunteer_average_age_l3628_362816

theorem volunteer_average_age (total_members : ℕ) (teens : ℕ) (parents : ℕ) (volunteers : ℕ)
  (teen_avg_age : ℝ) (parent_avg_age : ℝ) (overall_avg_age : ℝ) :
  total_members = 50 →
  teens = 30 →
  parents = 15 →
  volunteers = 5 →
  teen_avg_age = 16 →
  parent_avg_age = 35 →
  overall_avg_age = 23 →
  (total_members : ℝ) * overall_avg_age = 
    (teens : ℝ) * teen_avg_age + (parents : ℝ) * parent_avg_age + (volunteers : ℝ) * ((total_members : ℝ) * overall_avg_age - (teens : ℝ) * teen_avg_age - (parents : ℝ) * parent_avg_age) / (volunteers : ℝ) →
  ((total_members : ℝ) * overall_avg_age - (teens : ℝ) * teen_avg_age - (parents : ℝ) * parent_avg_age) / (volunteers : ℝ) = 29 :=
by
  sorry

#check volunteer_average_age

end NUMINAMATH_CALUDE_volunteer_average_age_l3628_362816


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3628_362899

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
   ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 1 := by sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3628_362899


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_two_primes_l3628_362841

/-- A function that returns true if a number is divisible by at least two different primes -/
def divisible_by_two_primes (x : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ∣ x ∧ q ∣ x

/-- The theorem stating that 5 is the smallest positive integer n ≥ 5 such that n^2 - n + 6 is divisible by at least two different primes -/
theorem smallest_n_divisible_by_two_primes :
  ∀ n : ℕ, n ≥ 5 → (divisible_by_two_primes (n^2 - n + 6) → n ≥ 5) ∧
  (n = 5 → divisible_by_two_primes (5^2 - 5 + 6)) :=
by sorry

#check smallest_n_divisible_by_two_primes

end NUMINAMATH_CALUDE_smallest_n_divisible_by_two_primes_l3628_362841


namespace NUMINAMATH_CALUDE_triangle_table_height_l3628_362888

theorem triangle_table_height (DE EF FD : ℝ) (hDE : DE = 20) (hEF : EF = 21) (hFD : FD = 29) :
  let s := (DE + EF + FD) / 2
  let A := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_d := 2 * A / FD
  let h_f := 2 * A / EF
  let k := (h_f * h_d) / (h_f + h_d)
  k = 7 * Real.sqrt 210 / 5 := by sorry

end NUMINAMATH_CALUDE_triangle_table_height_l3628_362888


namespace NUMINAMATH_CALUDE_perpendicular_line_with_same_intercept_l3628_362856

/-- Given a line l with equation x/3 - y/4 = 1, 
    prove that the line with equation 3x + 4y + 16 = 0 
    has the same y-intercept as l and is perpendicular to l -/
theorem perpendicular_line_with_same_intercept 
  (x y : ℝ) (l : x / 3 - y / 4 = 1) :
  ∃ (m b : ℝ), 
    (-- Same y-intercept condition
     b = -4) ∧ 
    (-- Perpendicular condition
     m * (4 / 3) = -1) ∧
    (-- Equation of the new line
     3 * x + 4 * y + 16 = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_with_same_intercept_l3628_362856


namespace NUMINAMATH_CALUDE_regression_line_change_l3628_362852

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.intercept + line.slope * x

/-- Theorem: For the regression line y = 2 - 1.5x, 
    when x increases by 1 unit, y decreases by 1.5 units -/
theorem regression_line_change 
  (line : RegressionLine) 
  (h1 : line.intercept = 2) 
  (h2 : line.slope = -1.5) 
  (x : ℝ) : 
  line.predict (x + 1) = line.predict x - 1.5 := by
  sorry


end NUMINAMATH_CALUDE_regression_line_change_l3628_362852


namespace NUMINAMATH_CALUDE_T_equals_five_l3628_362887

noncomputable def T : ℝ :=
  1 / (3 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
  1 / (Real.sqrt 7 - Real.sqrt 6) - 1 / (Real.sqrt 6 - Real.sqrt 5) + 
  1 / (Real.sqrt 5 - 2)

theorem T_equals_five : T = 5 := by
  sorry

end NUMINAMATH_CALUDE_T_equals_five_l3628_362887


namespace NUMINAMATH_CALUDE_shane_garret_age_ratio_l3628_362875

theorem shane_garret_age_ratio : 
  let shane_current_age : ℕ := 44
  let garret_current_age : ℕ := 12
  let years_ago : ℕ := 20
  let shane_past_age : ℕ := shane_current_age - years_ago
  shane_past_age / garret_current_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_shane_garret_age_ratio_l3628_362875


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3628_362862

theorem contrapositive_equivalence :
  (∀ x y : ℝ, (x = 3 ∧ y = 5) → x + y = 8) ↔
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 3 ∨ y ≠ 5)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3628_362862


namespace NUMINAMATH_CALUDE_pages_read_on_tuesday_l3628_362896

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Berry's daily reading goal -/
def daily_goal : ℕ := 50

/-- Pages read on Sunday -/
def sunday_pages : ℕ := 43

/-- Pages read on Monday -/
def monday_pages : ℕ := 65

/-- Pages read on Wednesday -/
def wednesday_pages : ℕ := 0

/-- Pages read on Thursday -/
def thursday_pages : ℕ := 70

/-- Pages read on Friday -/
def friday_pages : ℕ := 56

/-- Pages to be read on Saturday -/
def saturday_pages : ℕ := 88

/-- Theorem stating that Berry must have read 28 pages on Tuesday to achieve his weekly goal -/
theorem pages_read_on_tuesday : 
  ∃ (tuesday_pages : ℕ), 
    (sunday_pages + monday_pages + tuesday_pages + wednesday_pages + 
     thursday_pages + friday_pages + saturday_pages) = 
    (daily_goal * days_in_week) ∧ tuesday_pages = 28 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_on_tuesday_l3628_362896
