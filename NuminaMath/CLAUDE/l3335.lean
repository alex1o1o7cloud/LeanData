import Mathlib

namespace ratio_x_to_y_l3335_333530

theorem ratio_x_to_y (x y : ℚ) (h : (8 * x + 5 * y) / (10 * x + 3 * y) = 4 / 7) :
  x / y = -23 / 16 := by
  sorry

end ratio_x_to_y_l3335_333530


namespace count_words_theorem_l3335_333529

/-- The set of all available letters -/
def Letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants -/
def Consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels -/
def Vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering -/
def WordLength : Nat := 5

/-- Function to count the number of 5-letter words with at least two consonants -/
def count_words_with_at_least_two_consonants : Nat :=
  sorry

/-- Theorem stating that the number of 5-letter words with at least two consonants is 7424 -/
theorem count_words_theorem : count_words_with_at_least_two_consonants = 7424 := by
  sorry

end count_words_theorem_l3335_333529


namespace accounting_majors_l3335_333553

theorem accounting_majors (p q r s t u : ℕ) : 
  p * q * r * s * t * u = 51030 → 
  1 < p → p < q → q < r → r < s → s < t → t < u → 
  p = 2 := by
sorry

end accounting_majors_l3335_333553


namespace max_value_trig_expression_l3335_333554

open Real

theorem max_value_trig_expression :
  ∃ (M : ℝ), M = Real.sqrt 10 + 3 ∧
  (∀ x : ℝ, cos x + 3 * sin x + tan x ≤ M) ∧
  (∃ x : ℝ, cos x + 3 * sin x + tan x = M) := by
  sorry

end max_value_trig_expression_l3335_333554


namespace symmetry_implies_a_equals_one_l3335_333549

/-- A function f: ℝ → ℝ is symmetric about the line x = c if f(c + t) = f(c - t) for all t ∈ ℝ -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ t, f (c + t) = f (c - t)

/-- The main theorem: If sin x + a cos x is symmetric about x = π/4, then a = 1 -/
theorem symmetry_implies_a_equals_one (a : ℝ) :
  SymmetricAbout (fun x ↦ Real.sin x + a * Real.cos x) (π/4) → a = 1 := by
  sorry

end symmetry_implies_a_equals_one_l3335_333549


namespace tshirts_per_package_l3335_333516

theorem tshirts_per_package (total_tshirts : ℕ) (num_packages : ℕ) 
  (h1 : total_tshirts = 70) 
  (h2 : num_packages = 14) : 
  total_tshirts / num_packages = 5 := by
  sorry

end tshirts_per_package_l3335_333516


namespace complex_vector_sum_l3335_333572

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) :
  z₁ = -1 + 2*Complex.I →
  z₂ = 1 - Complex.I →
  z₃ = 3 - 2*Complex.I →
  z₃ = x * z₁ + y * z₂ →
  x + y = 5 := by sorry

end complex_vector_sum_l3335_333572


namespace children_tickets_count_l3335_333540

/-- Proves that the number of children's tickets is 21 given the ticket prices, total amount paid, and total number of tickets. -/
theorem children_tickets_count (adult_price child_price total_amount total_tickets : ℕ)
  (h_adult_price : adult_price = 8)
  (h_child_price : child_price = 5)
  (h_total_amount : total_amount = 201)
  (h_total_tickets : total_tickets = 33) :
  ∃ (adult_count child_count : ℕ),
    adult_count + child_count = total_tickets ∧
    adult_count * adult_price + child_count * child_price = total_amount ∧
    child_count = 21 :=
by sorry

end children_tickets_count_l3335_333540


namespace bills_steps_correct_l3335_333569

/-- The length of each step Bill takes, in metres -/
def step_length : ℚ := 1/2

/-- The total distance Bill walks, in metres -/
def total_distance : ℚ := 12

/-- The number of steps Bill takes to walk the total distance -/
def number_of_steps : ℕ := 24

/-- Theorem stating that the number of steps Bill takes is correct -/
theorem bills_steps_correct : 
  (step_length * number_of_steps : ℚ) = total_distance :=
by sorry

end bills_steps_correct_l3335_333569


namespace sin_double_angle_l3335_333507

theorem sin_double_angle (x : Real) (h : Real.sin (x + π/4) = 3/5) : 
  Real.sin (2*x) = 8*Real.sqrt 2/25 := by
  sorry

end sin_double_angle_l3335_333507


namespace candies_left_l3335_333582

def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_bought_friday : ℕ := 2
def candies_eaten : ℕ := 6

theorem candies_left : 
  candies_bought_tuesday + candies_bought_thursday + candies_bought_friday - candies_eaten = 4 := by
  sorry

end candies_left_l3335_333582


namespace components_upper_bound_l3335_333566

/-- Represents a square grid with diagonals --/
structure DiagonalGrid (n : ℕ) where
  size : n > 8
  cells : Fin n → Fin n → Bool
  -- True represents one diagonal, False represents the other

/-- Counts the number of connected components in the grid --/
def countComponents (g : DiagonalGrid n) : ℕ := sorry

/-- Theorem stating that the number of components is not greater than n²/4 --/
theorem components_upper_bound (n : ℕ) (g : DiagonalGrid n) :
  countComponents g ≤ n^2 / 4 := by sorry

end components_upper_bound_l3335_333566


namespace fraction_equality_l3335_333561

theorem fraction_equality (b : ℕ+) : 
  (b : ℚ) / ((b : ℚ) + 35) = 869 / 1000 → b = 232 := by
  sorry

end fraction_equality_l3335_333561


namespace fraction_division_l3335_333544

theorem fraction_division : (4/9) / (5/8) = 32/45 := by
  sorry

end fraction_division_l3335_333544


namespace dans_seashells_l3335_333586

def seashells_problem (initial_seashells : ℕ) (remaining_seashells : ℕ) : Prop :=
  initial_seashells ≥ remaining_seashells →
  ∃ (given_seashells : ℕ), given_seashells = initial_seashells - remaining_seashells

theorem dans_seashells : seashells_problem 56 22 := by
  sorry

end dans_seashells_l3335_333586


namespace additional_money_needed_l3335_333535

def phone_cost : ℝ := 1300
def percentage_owned : ℝ := 40

theorem additional_money_needed : 
  phone_cost - (percentage_owned / 100) * phone_cost = 780 := by
  sorry

end additional_money_needed_l3335_333535


namespace quadratic_equation_solution_l3335_333552

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 11 * x - 20 = 0 :=
by
  use 4/3
  sorry

end quadratic_equation_solution_l3335_333552


namespace prob_exceeds_175_l3335_333560

/-- The probability that a randomly selected student's height is less than 160cm -/
def prob_less_than_160 : ℝ := 0.2

/-- The probability that a randomly selected student's height is between 160cm and 175cm -/
def prob_between_160_and_175 : ℝ := 0.5

/-- Theorem: Given the probabilities of a student's height being less than 160cm and between 160cm and 175cm,
    the probability of a student's height exceeding 175cm is 0.3 -/
theorem prob_exceeds_175 : 1 - (prob_less_than_160 + prob_between_160_and_175) = 0.3 := by
  sorry

end prob_exceeds_175_l3335_333560


namespace sports_club_membership_l3335_333593

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : tennis = 18)
  (h4 : both = 3) :
  total - (badminton + tennis - both) = 5 := by
  sorry

end sports_club_membership_l3335_333593


namespace inscribed_cube_volume_l3335_333523

theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_edge : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  small_cube_edge * Real.sqrt 3 = sphere_diameter →
  small_cube_edge ^ 3 = 192 * Real.sqrt 3 :=
by sorry

end inscribed_cube_volume_l3335_333523


namespace function_expressions_and_minimum_l3335_333513

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x * (x + 2)
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x + 2

def has_same_tangent_at_zero (f g : ℝ → ℝ) : Prop :=
  (deriv f) 0 = (deriv g) 0 ∧ f 0 = g 0

theorem function_expressions_and_minimum (a b : ℝ) (t : ℝ) 
  (h1 : has_same_tangent_at_zero (f a) (g b))
  (h2 : t > -4) :
  (∃ (a' b' : ℝ), f a' = f 1 ∧ g b' = g 3) ∧
  (∀ x ∈ Set.Icc t (t + 1),
    (t < -3 → f 1 x ≥ -Real.exp (-3)) ∧
    (t ≥ -3 → f 1 x ≥ Real.exp t * (t + 2))) :=
by sorry

end function_expressions_and_minimum_l3335_333513


namespace unique_prime_for_equiangular_polygons_l3335_333587

theorem unique_prime_for_equiangular_polygons :
  ∃! k : ℕ, 
    Prime k ∧ 
    k > 1 ∧
    ∃ (x n₁ n₂ : ℕ),
      -- Angle formula for P1
      x = 180 - 360 / n₁ ∧ 
      -- Angle formula for P2
      k * x = 180 - 360 / n₂ ∧ 
      -- Angles must be positive and less than 180°
      0 < x ∧ x < 180 ∧
      0 < k * x ∧ k * x < 180 ∧
      -- Number of sides must be at least 3
      n₁ ≥ 3 ∧ n₂ ≥ 3 :=
sorry

end unique_prime_for_equiangular_polygons_l3335_333587


namespace cricket_bat_weight_proof_l3335_333515

/-- The weight of one cricket bat in pounds -/
def cricket_bat_weight : ℝ := 18

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 36

/-- The number of cricket bats -/
def num_cricket_bats : ℕ := 8

/-- The number of basketballs -/
def num_basketballs : ℕ := 4

theorem cricket_bat_weight_proof :
  cricket_bat_weight * num_cricket_bats = basketball_weight * num_basketballs :=
by sorry

end cricket_bat_weight_proof_l3335_333515


namespace parallelogram_area_l3335_333550

-- Define the parallelogram ABCD
variable (A B C D : Point)

-- Define point E as midpoint of BC
variable (E : Point)

-- Define point F on AD
variable (F : Point)

-- Define the area function
variable (area : Set Point → ℝ)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define E as midpoint of BC
def is_midpoint (E B C : Point) : Prop := sorry

-- Define the condition DF = 2FC
def segment_ratio (D F C : Point) : Prop := sorry

-- Define triangles
def triangle (P Q R : Point) : Set Point := sorry

-- Define parallelogram
def parallelogram (A B C D : Point) : Set Point := sorry

-- Theorem statement
theorem parallelogram_area 
  (h1 : is_parallelogram A B C D)
  (h2 : is_midpoint E B C)
  (h3 : segment_ratio D F C)
  (h4 : area (triangle A F C) + area (triangle A B E) = 10) :
  area (parallelogram A B C D) = 24 := by sorry

end parallelogram_area_l3335_333550


namespace art_gallery_pieces_l3335_333521

theorem art_gallery_pieces (total : ℕ) 
  (displayed : ℕ) (sculptures_displayed : ℕ) 
  (paintings_not_displayed : ℕ) (sculptures_not_displayed : ℕ) :
  displayed = total / 3 →
  sculptures_displayed = displayed / 6 →
  paintings_not_displayed = (total - displayed) / 3 →
  sculptures_not_displayed = 1400 →
  total = 3150 := by
sorry

end art_gallery_pieces_l3335_333521


namespace beam_max_strength_l3335_333531

/-- The strength of a rectangular beam cut from a circular log is maximized when its width is 2R/√3 and its height is 2R√2/√3, where R is the radius of the log. -/
theorem beam_max_strength (R : ℝ) (R_pos : R > 0) :
  let strength (x y : ℝ) := x * y^2
  let constraint (x y : ℝ) := x^2 + y^2 = 4 * R^2
  ∃ (k : ℝ), k > 0 ∧
    ∀ (x y : ℝ), constraint x y →
      strength x y ≤ k * strength (2*R/Real.sqrt 3) (2*R*Real.sqrt 2/Real.sqrt 3) :=
by sorry

end beam_max_strength_l3335_333531


namespace salary_left_unspent_l3335_333595

/-- The fraction of salary spent in the first week -/
def first_week_spending : ℚ := 1/4

/-- The fraction of salary spent in each of the following three weeks -/
def other_weeks_spending : ℚ := 1/5

/-- The number of weeks after the first week -/
def remaining_weeks : ℕ := 3

/-- Theorem: Given the spending conditions, the fraction of salary left unspent at the end of the month is 3/20 -/
theorem salary_left_unspent :
  1 - (first_week_spending + remaining_weeks * other_weeks_spending) = 3/20 := by
  sorry

end salary_left_unspent_l3335_333595


namespace op_theorem_l3335_333573

/-- The type representing elements in our set -/
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

/-- The operation ⊕ -/
def op : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem op_theorem : 
  op (op Element.four Element.one) (op Element.two Element.three) = Element.three :=
by sorry

end op_theorem_l3335_333573


namespace sum_removal_proof_l3335_333547

theorem sum_removal_proof : 
  let original_sum := (1/2 : ℚ) + 1/3 + 1/4 + 1/6 + 1/8 + 1/9 + 1/12
  let removed_terms := (1/8 : ℚ) + 1/9
  original_sum - removed_terms = 4/3 := by
  sorry

end sum_removal_proof_l3335_333547


namespace joan_apples_total_l3335_333500

/-- The number of apples Joan has now, given her initial pick and Melanie's gift -/
def total_apples (initial_pick : ℕ) (melanie_gift : ℕ) : ℕ :=
  initial_pick + melanie_gift

/-- Theorem stating that Joan has 70 apples in total -/
theorem joan_apples_total :
  total_apples 43 27 = 70 := by
  sorry

end joan_apples_total_l3335_333500


namespace cafeteria_extra_fruits_l3335_333506

/-- The number of extra fruits ordered by the cafeteria -/
def extra_fruits (total_fruits students max_per_student : ℕ) : ℕ :=
  total_fruits - (students * max_per_student)

/-- Theorem stating that the cafeteria ordered 43 extra fruits -/
theorem cafeteria_extra_fruits :
  extra_fruits 85 21 2 = 43 := by
  sorry

end cafeteria_extra_fruits_l3335_333506


namespace complex_equation_solutions_l3335_333518

theorem complex_equation_solutions (z : ℂ) : 
  z^3 + z = 2 * Complex.abs z^2 → 
  z = 0 ∨ z = 1 ∨ z = -1 + 2*Complex.I ∨ z = -1 - 2*Complex.I :=
by sorry

end complex_equation_solutions_l3335_333518


namespace line_of_symmetry_l3335_333545

/-- Definition of circle O -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

/-- Definition of line l -/
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

/-- Theorem stating that line l is the line of symmetry for circles O and C -/
theorem line_of_symmetry :
  ∀ (x y : ℝ), line_l x y → (∃ (x' y' : ℝ), circle_O x' y' ∧ circle_C x y ∧
    x' = 2*x - x ∧ y' = 2*y - y) :=
sorry

end line_of_symmetry_l3335_333545


namespace polynomial_roots_l3335_333591

theorem polynomial_roots : 
  let f : ℝ → ℝ := λ x => (x + 1998) * (x + 1999) * (x + 2000) * (x + 2001) + 1
  ∀ x : ℝ, f x = 0 ↔ x = -1999.5 - Real.sqrt 5 / 2 ∨ x = -1999.5 + Real.sqrt 5 / 2 := by
  sorry

end polynomial_roots_l3335_333591


namespace family_of_lines_fixed_point_l3335_333514

/-- The point that all lines in the family kx+y+2k+1=0 pass through -/
theorem family_of_lines_fixed_point (k : ℝ) : 
  k * (-2) + (-1) + 2 * k + 1 = 0 := by
  sorry

#check family_of_lines_fixed_point

end family_of_lines_fixed_point_l3335_333514


namespace derivative_f_at_neg_one_l3335_333568

noncomputable def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1/2) * (3 + x^3)^(1/3)

theorem derivative_f_at_neg_one :
  deriv f (-1) = Real.sqrt 3 * 2^(1/3) :=
sorry

end derivative_f_at_neg_one_l3335_333568


namespace percentage_calculation_l3335_333508

theorem percentage_calculation (n : ℝ) (h : n = 6000) :
  (0.1 * (0.3 * (0.5 * n))) = 90 := by
  sorry

end percentage_calculation_l3335_333508


namespace binomial_coefficient_2000_3_l3335_333509

theorem binomial_coefficient_2000_3 : Nat.choose 2000 3 = 1331000333 := by
  sorry

end binomial_coefficient_2000_3_l3335_333509


namespace ellipse_equation_l3335_333504

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    if a triangle formed by the intersection of a line through one focus with the ellipse
    has perimeter p, then the standard equation of the ellipse is x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (e : ℝ) (he : e = Real.sqrt 3 / 3)
  (p : ℝ) (hp : p = 4 * Real.sqrt 3) :
  ∃ (x y : ℝ), x^2 / 3 + y^2 / 2 = 1 :=
sorry

end ellipse_equation_l3335_333504


namespace half_power_decreasing_l3335_333533

theorem half_power_decreasing (a b : ℝ) (h : a > b) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end half_power_decreasing_l3335_333533


namespace intersection_M_N_l3335_333585

-- Define set M
def M : Set ℝ := {x | x^2 + 2*x - 3 = 0}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (2^x - 1/2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end intersection_M_N_l3335_333585


namespace distance_is_35_over_13_l3335_333539

def point : ℝ × ℝ × ℝ := (0, -1, 4)
def line_point : ℝ × ℝ × ℝ := (-3, 2, 5)
def line_direction : ℝ × ℝ × ℝ := (4, 1, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ := 
  sorry

theorem distance_is_35_over_13 : 
  distance_to_line point line_point line_direction = 35 / 13 := by sorry

end distance_is_35_over_13_l3335_333539


namespace correct_assignment_count_l3335_333532

/-- The number of ways to assign volunteers to pavilions. -/
def assign_volunteers (total_volunteers : ℕ) (female_volunteers : ℕ) (male_volunteers : ℕ) (pavilions : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating the correct number of ways to assign volunteers. -/
theorem correct_assignment_count :
  assign_volunteers 8 3 5 3 = 180 :=
sorry

end correct_assignment_count_l3335_333532


namespace electricity_scientific_notation_l3335_333564

-- Define the number of kilowatt-hours
def electricity_delivered : ℝ := 105.9e9

-- Theorem to prove the scientific notation
theorem electricity_scientific_notation :
  electricity_delivered = 1.059 * (10 : ℝ)^10 := by
  sorry

end electricity_scientific_notation_l3335_333564


namespace estimate_value_l3335_333527

theorem estimate_value : 6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
                         (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 := by
  sorry

end estimate_value_l3335_333527


namespace square_perimeter_l3335_333590

theorem square_perimeter (A : ℝ) (h : A = 625) :
  2 * (4 * Real.sqrt A) = 200 :=
by sorry

end square_perimeter_l3335_333590


namespace average_candies_sigyeong_group_l3335_333517

def sigyeong_group : List Nat := [16, 22, 30, 26, 18, 20]

theorem average_candies_sigyeong_group : 
  (sigyeong_group.sum / sigyeong_group.length : ℚ) = 22 := by
  sorry

end average_candies_sigyeong_group_l3335_333517


namespace tan_neg_seven_pi_sixths_l3335_333512

theorem tan_neg_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_seven_pi_sixths_l3335_333512


namespace max_diagonal_area_ratio_l3335_333511

/-- A triangle with an inscribed rectangle -/
structure TriangleWithInscribedRectangle where
  /-- The area of the triangle -/
  area : ℝ
  /-- The length of the shortest diagonal of any inscribed rectangle -/
  shortest_diagonal : ℝ
  /-- The area is positive -/
  area_pos : 0 < area

/-- The theorem statement -/
theorem max_diagonal_area_ratio (T : TriangleWithInscribedRectangle) :
  T.shortest_diagonal ^ 2 / T.area ≤ 4 * Real.sqrt 3 / 7 := by
  sorry


end max_diagonal_area_ratio_l3335_333511


namespace prime_cube_plus_one_l3335_333505

theorem prime_cube_plus_one (p : ℕ) (x y : ℕ+) (h_prime : Nat.Prime p) 
  (h_eq : p ^ x.val = y.val ^ 3 + 1) :
  ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := by
  sorry

end prime_cube_plus_one_l3335_333505


namespace negative_integer_sum_square_twelve_l3335_333555

theorem negative_integer_sum_square_twelve (M : ℤ) : 
  M < 0 → M^2 + M = 12 → M = -4 := by
  sorry

end negative_integer_sum_square_twelve_l3335_333555


namespace mutually_exclusive_events_l3335_333542

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that the events are mutually exclusive
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(hit_at_least_once ω ∧ miss_both_times ω) :=
by
  sorry

end mutually_exclusive_events_l3335_333542


namespace problem_statement_l3335_333546

theorem problem_statement (x : ℚ) : 5 * x - 10 = 15 * x + 5 → 5 * (x + 3) = 15 / 2 := by
  sorry

end problem_statement_l3335_333546


namespace lemons_removed_is_thirty_l3335_333594

/-- Represents the number of lemons picked by each person and eaten by animals --/
structure LemonCounts where
  sally : ℕ
  mary : ℕ
  tom : ℕ
  eaten : ℕ

/-- Calculates the total number of lemons removed from the tree --/
def total_lemons_removed (counts : LemonCounts) : ℕ :=
  counts.sally + counts.mary + counts.tom + counts.eaten

/-- Theorem stating that the total number of lemons removed is 30 --/
theorem lemons_removed_is_thirty : 
  ∀ (counts : LemonCounts), 
  counts.sally = 7 → 
  counts.mary = 9 → 
  counts.tom = 12 → 
  counts.eaten = 2 → 
  total_lemons_removed counts = 30 := by
  sorry


end lemons_removed_is_thirty_l3335_333594


namespace function_values_l3335_333501

def is_prime (n : ℕ) : Prop := sorry

def coprime (a b : ℕ) : Prop := sorry

def number_theory_function (f : ℕ → ℕ) : Prop :=
  (∀ a b, coprime a b → f (a * b) = f a * f b) ∧
  (∀ p q, is_prime p → is_prime q → f (p + q) = f p + f q)

theorem function_values (f : ℕ → ℕ) (h : number_theory_function f) :
  f 2 = 2 ∧ f 3 = 3 ∧ f 1999 = 1999 := by sorry

end function_values_l3335_333501


namespace f_not_mapping_l3335_333599

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | 0 ≤ y ∧ y ≤ 8}

-- Define the correspondence rule f
def f (x : ℝ) : ℝ := 4

-- Theorem stating that f is not a mapping from A to B
theorem f_not_mapping : ¬(∀ x ∈ A, f x ∈ B) :=
sorry

end f_not_mapping_l3335_333599


namespace system_solution_iff_a_in_interval_l3335_333565

/-- The system of equations has a solution for some b if and only if a is in the interval (-8, 7] -/
theorem system_solution_iff_a_in_interval (a : ℝ) : 
  (∃ (b x y : ℝ), x^2 + y^2 + 2*a*(a - x - y) = 64 ∧ y = 7 / ((x + b)^2 + 1)) ↔ 
  -8 < a ∧ a ≤ 7 := by sorry

end system_solution_iff_a_in_interval_l3335_333565


namespace trig_inequality_l3335_333526

theorem trig_inequality (x y : Real) 
  (hx : 0 < x ∧ x < Real.pi / 2)
  (hy : 0 < y ∧ y < Real.pi / 2)
  (h_eq : Real.sin x = x * Real.cos y) : 
  x / 2 < y ∧ y < x :=
by sorry

end trig_inequality_l3335_333526


namespace intersection_distance_l3335_333557

/-- Given ω > 0, if the distance between the two closest intersection points 
    of y = 4sin(ωx) and y = 4cos(ωx) is 6, then ω = π/2 -/
theorem intersection_distance (ω : Real) (h1 : ω > 0) : 
  (∃ x₁ x₂ : Real, 
    x₁ ≠ x₂ ∧ 
    4 * Real.sin (ω * x₁) = 4 * Real.cos (ω * x₁) ∧
    4 * Real.sin (ω * x₂) = 4 * Real.cos (ω * x₂) ∧
    ∀ x : Real, 4 * Real.sin (ω * x) = 4 * Real.cos (ω * x) → 
      (x = x₁ ∨ x = x₂ ∨ |x - x₁| ≥ |x₁ - x₂| ∧ |x - x₂| ≥ |x₁ - x₂|) ∧
    (x₁ - x₂)^2 = 36) →
  ω = π / 2 := by sorry

end intersection_distance_l3335_333557


namespace min_stamps_for_39_cents_l3335_333579

theorem min_stamps_for_39_cents : 
  ∃ (c f : ℕ), 3 * c + 5 * f = 39 ∧ 
  c + f = 9 ∧ 
  ∀ (c' f' : ℕ), 3 * c' + 5 * f' = 39 → c + f ≤ c' + f' :=
by sorry

end min_stamps_for_39_cents_l3335_333579


namespace inverse_inequality_conditions_l3335_333528

theorem inverse_inequality_conditions (a b : ℝ) :
  (1 / a < 1 / b) ↔ (b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0) :=
sorry

end inverse_inequality_conditions_l3335_333528


namespace odd_function_with_period_4_sum_l3335_333548

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_with_period_4_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) (h_period : has_period f 4) :
  f 2005 + f 2006 + f 2007 = 0 := by
  sorry

end odd_function_with_period_4_sum_l3335_333548


namespace quadratic_sum_of_constants_l3335_333596

theorem quadratic_sum_of_constants (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 48 * x + 162 = a * (x + b)^2 + c) ∧ (a + b + c = 76) := by
  sorry

end quadratic_sum_of_constants_l3335_333596


namespace sum_reciprocal_bound_l3335_333580

theorem sum_reciprocal_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a + b = 2) :
  c / a + c / b ≥ 2 * c ∧ ∀ ε > 0, ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ c / a' + c / b' > ε := by
  sorry

end sum_reciprocal_bound_l3335_333580


namespace determinant_trig_matrix_l3335_333556

theorem determinant_trig_matrix (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos (α + β), Real.sin (α + β), -Real.sin α],
    ![-Real.sin β, Real.cos β, 0],
    ![Real.sin α * Real.cos β, Real.sin α * Real.sin β, Real.cos α]
  ]
  Matrix.det M = 1 := by sorry

end determinant_trig_matrix_l3335_333556


namespace unique_natural_solution_l3335_333551

theorem unique_natural_solution :
  ∀ n : ℕ, n ≠ 0 → (2 * n - 1 / (n^5 : ℚ) = 3 - 2 / (n : ℚ)) ↔ n = 1 := by
sorry

end unique_natural_solution_l3335_333551


namespace sum_after_removal_is_perfect_square_l3335_333519

-- Define the set M
def M : Set Nat := {n | 1 ≤ n ∧ n ≤ 2017}

-- Define the sum of all elements in M
def sum_M : Nat := (2017 * 2018) / 2

-- Define the element to be removed
def removed_element : Nat := 1677

-- Theorem to prove
theorem sum_after_removal_is_perfect_square :
  ∃ k : Nat, sum_M - removed_element = k^2 ∧ removed_element ∈ M :=
sorry

end sum_after_removal_is_perfect_square_l3335_333519


namespace bf_length_l3335_333574

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : (A.1 = C.1 ∧ A.2 = D.2) ∧ (C.1 = D.1 ∧ C.2 = B.2))  -- right angles at A and C
variable (h2 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C)  -- E is on AC
variable (h3 : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (1 - s) • A + s • C)  -- F is on AC
variable (h4 : (D.1 - E.1) * (C.1 - A.1) + (D.2 - E.2) * (C.2 - A.2) = 0)  -- DE perpendicular to AC
variable (h5 : (B.1 - F.1) * (C.1 - A.1) + (B.2 - F.2) * (C.2 - A.2) = 0)  -- BF perpendicular to AC
variable (h6 : Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) = 4)  -- AE = 4
variable (h7 : Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 4)  -- DE = 4
variable (h8 : Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) = 6)  -- CE = 6

-- Theorem statement
theorem bf_length : Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 4 :=
sorry

end bf_length_l3335_333574


namespace max_value_expression_l3335_333592

theorem max_value_expression (a b c d : ℝ) 
  (ha : -7.5 ≤ a ∧ a ≤ 7.5)
  (hb : -7.5 ≤ b ∧ b ≤ 7.5)
  (hc : -7.5 ≤ c ∧ c ≤ 7.5)
  (hd : -7.5 ≤ d ∧ d ≤ 7.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 240 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    (-7.5 ≤ a₀ ∧ a₀ ≤ 7.5) ∧
    (-7.5 ≤ b₀ ∧ b₀ ≤ 7.5) ∧
    (-7.5 ≤ c₀ ∧ c₀ ≤ 7.5) ∧
    (-7.5 ≤ d₀ ∧ d₀ ≤ 7.5) ∧
    (a₀ + 2*b₀ + c₀ + 2*d₀ - a₀*b₀ - b₀*c₀ - c₀*d₀ - d₀*a₀) = 240 :=
by sorry

end max_value_expression_l3335_333592


namespace base_10_to_7_2023_l3335_333584

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  sorry

/-- Converts a list of digits in base 7 to a natural number --/
def fromBase7 (digits : List Nat) : Nat :=
  sorry

theorem base_10_to_7_2023 :
  toBase7 2023 = [5, 6, 2, 0] ∧ fromBase7 [5, 6, 2, 0] = 2023 := by
  sorry

end base_10_to_7_2023_l3335_333584


namespace negative_cube_squared_l3335_333575

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end negative_cube_squared_l3335_333575


namespace partial_fraction_decomposition_l3335_333597

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) : 
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 → (45 * x - 34) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = -1111 / 4 := by
sorry

end partial_fraction_decomposition_l3335_333597


namespace integral_sqrt_one_minus_x_squared_l3335_333577

theorem integral_sqrt_one_minus_x_squared (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f x = Real.sqrt (1 - x^2)) →
  (∫ x in Set.Icc (-1) 1, f x) = π / 2 := by
  sorry

end integral_sqrt_one_minus_x_squared_l3335_333577


namespace total_wheels_at_station_l3335_333567

/-- The number of trains at the station -/
def num_trains : ℕ := 4

/-- The number of carriages per train -/
def carriages_per_train : ℕ := 4

/-- The number of wheel rows per carriage -/
def wheel_rows_per_carriage : ℕ := 3

/-- The number of wheels per row -/
def wheels_per_row : ℕ := 5

/-- The total number of wheels at the train station -/
def total_wheels : ℕ := num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row

theorem total_wheels_at_station :
  total_wheels = 240 :=
by sorry

end total_wheels_at_station_l3335_333567


namespace mrs_blue_tomato_yield_l3335_333563

/-- Represents the dimensions of a rectangular vegetable patch in steps -/
structure PatchDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected tomato yield from a vegetable patch -/
def expected_tomato_yield (dimensions : PatchDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (dimensions.length : ℝ) * step_length * (dimensions.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected tomato yield for Mrs. Blue's vegetable patch -/
theorem mrs_blue_tomato_yield :
  let dimensions : PatchDimensions := ⟨18, 25⟩
  let step_length : ℝ := 3
  let yield_per_sqft : ℝ := 3 / 4
  expected_tomato_yield dimensions step_length yield_per_sqft = 3037.5 := by
  sorry

end mrs_blue_tomato_yield_l3335_333563


namespace percentage_of_part_to_whole_l3335_333510

theorem percentage_of_part_to_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 25 →
  part = 70 ∧ whole = 280 :=
by sorry

end percentage_of_part_to_whole_l3335_333510


namespace cody_tickets_l3335_333537

theorem cody_tickets (initial : ℝ) (lost : ℝ) (spent : ℝ) : 
  initial = 49.0 → lost = 6.0 → spent = 25.0 → initial - lost - spent = 18.0 :=
by sorry

end cody_tickets_l3335_333537


namespace cylinder_intersection_area_l3335_333588

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the area of a surface formed by intersecting a cylinder with a plane --/
def intersectionArea (c : Cylinder) (arcAngle : ℝ) : ℝ := sorry

theorem cylinder_intersection_area :
  let c : Cylinder := { radius := 7, height := 9 }
  let arcAngle : ℝ := 150 * (π / 180)  -- Convert degrees to radians
  intersectionArea c arcAngle = 62.4 * π + 112 * Real.sqrt 3 := by sorry

end cylinder_intersection_area_l3335_333588


namespace jeans_bought_l3335_333524

/-- Given a clothing sale with specific prices and quantities, prove the number of jeans bought. -/
theorem jeans_bought (shirt_price hat_price jeans_price total_cost : ℕ) 
  (shirts_bought hats_bought : ℕ) : 
  shirt_price = 5 →
  hat_price = 4 →
  jeans_price = 10 →
  total_cost = 51 →
  shirts_bought = 3 →
  hats_bought = 4 →
  ∃ (jeans_bought : ℕ), 
    jeans_bought = 2 ∧ 
    total_cost = shirt_price * shirts_bought + hat_price * hats_bought + jeans_price * jeans_bought :=
by sorry

end jeans_bought_l3335_333524


namespace hyperbola_other_asymptote_l3335_333536

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- X-coordinate of the foci -/
  foci_x : ℝ

/-- Represents the equation of a line in the form y = mx + b -/
structure LineEquation where
  m : ℝ
  b : ℝ

/-- The other asymptote of a hyperbola given one asymptote and the x-coordinate of the foci -/
def other_asymptote (h : Hyperbola) : LineEquation :=
  { m := -2, b := -16 }

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = -4) : 
  other_asymptote h = { m := -2, b := -16 } := by
  sorry

end hyperbola_other_asymptote_l3335_333536


namespace last_number_in_sequence_l3335_333538

theorem last_number_in_sequence (x : ℕ) : 
  1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + x = 4100 → x = 10 := by
  sorry

end last_number_in_sequence_l3335_333538


namespace increase_by_percentage_l3335_333570

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 50 → percentage = 120 → result = initial * (1 + percentage / 100) → result = 110 := by
  sorry

end increase_by_percentage_l3335_333570


namespace angle_C_value_l3335_333525

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x + (Real.sqrt 3/2) * Real.cos x

theorem angle_C_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  f A = Real.sqrt 3 / 2 ∧
  a = (Real.sqrt 3 / 2) * b ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  C = π / 6 := by
  sorry

end angle_C_value_l3335_333525


namespace sum_of_digits_mod_9_triple_sum_of_digits_4444_power_l3335_333558

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Property: sum of digits is congruent to the number modulo 9 -/
theorem sum_of_digits_mod_9 (n : ℕ) : sum_of_digits n % 9 = n % 9 := sorry

/-- Main theorem -/
theorem triple_sum_of_digits_4444_power :
  let N := 4444^4444
  let f := sum_of_digits
  f (f (f N)) = 7 := by sorry

end sum_of_digits_mod_9_triple_sum_of_digits_4444_power_l3335_333558


namespace find_number_l3335_333541

theorem find_number : ∃ x : ℝ, (0.4 * x - 30 = 50) ∧ (x = 200) := by
  sorry

end find_number_l3335_333541


namespace student_walking_speed_l3335_333559

/-- 
Given two students walking towards each other:
- They start 350 meters apart
- They walk for 100 seconds until they meet
- The first student walks at 1.6 m/s
Prove that the second student's speed is 1.9 m/s
-/
theorem student_walking_speed 
  (initial_distance : ℝ) 
  (time : ℝ) 
  (speed1 : ℝ) 
  (h1 : initial_distance = 350)
  (h2 : time = 100)
  (h3 : speed1 = 1.6) :
  ∃ speed2 : ℝ, 
    speed2 = 1.9 ∧ 
    speed1 * time + speed2 * time = initial_distance := by
  sorry

end student_walking_speed_l3335_333559


namespace sum_of_common_ratios_l3335_333578

-- Define the geometric sequences and their properties
def geometric_sequence (k a₂ a₃ : ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2

-- Theorem statement
theorem sum_of_common_ratios 
  (k a₂ a₃ b₂ b₃ : ℝ) 
  (h₁ : geometric_sequence k a₂ a₃)
  (h₂ : geometric_sequence k b₂ b₃)
  (h₃ : ∃ p r : ℝ, p ≠ r ∧ 
    a₂ = k * p ∧ a₃ = k * p^2 ∧ 
    b₂ = k * r ∧ b₃ = k * r^2)
  (h₄ : a₃ - b₃ = 3 * (a₂ - b₂))
  : ∃ p r : ℝ, p + r = 3 ∧ 
    a₂ = k * p ∧ a₃ = k * p^2 ∧ 
    b₂ = k * r ∧ b₃ = k * r^2 :=
sorry

end sum_of_common_ratios_l3335_333578


namespace percentage_of_sum_l3335_333534

theorem percentage_of_sum (x y : ℝ) (P : ℝ) : 
  (0.5 * (x - y) = (P / 100) * (x + y)) → 
  (y = 0.42857142857142854 * x) → 
  (P = 20) := by
sorry

end percentage_of_sum_l3335_333534


namespace pi_is_irrational_l3335_333503

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define π (since it's not a built-in constant in Lean)
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem pi_is_irrational : ¬ IsRational π := by
  sorry


end pi_is_irrational_l3335_333503


namespace coordinates_wrt_origin_l3335_333576

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the given point
def given_point : Point := (2, -6)

-- Theorem stating that the coordinates of the given point with respect to the origin are (2, -6)
theorem coordinates_wrt_origin (p : Point) : p = given_point → p = (2, -6) := by
  sorry

end coordinates_wrt_origin_l3335_333576


namespace watch_cost_price_l3335_333520

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℚ), 
  (cp * (1 - 1/10) = cp * 0.9) ∧ 
  (cp * (1 + 1/10) = cp * 1.1) ∧ 
  (cp * 1.1 - cp * 0.9 = 500) ∧ 
  cp = 2500 := by
  sorry

end watch_cost_price_l3335_333520


namespace batsman_average_theorem_l3335_333598

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  averageBeforeLastInning : Rat
  lastInningScore : Nat
  averageIncrease : Rat

/-- Calculates the new average after the last inning -/
def newAverage (stats : BatsmanStats) : Rat :=
  (stats.totalRuns + stats.lastInningScore) / stats.innings

/-- Theorem: Given the conditions, prove that the new average is 23 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.innings = 17)
  (h2 : stats.lastInningScore = 87)
  (h3 : stats.averageIncrease = 4)
  (h4 : newAverage stats = stats.averageBeforeLastInning + stats.averageIncrease) :
  newAverage stats = 23 := by
  sorry

end batsman_average_theorem_l3335_333598


namespace cosine_sum_equals_one_l3335_333562

theorem cosine_sum_equals_one (α β : ℝ) :
  ((Real.cos α * Real.cos (β / 2)) / Real.cos (α + β / 2) +
   (Real.cos β * Real.cos (α / 2)) / Real.cos (β + α / 2) = 1) →
  Real.cos α + Real.cos β = 1 := by
  sorry

end cosine_sum_equals_one_l3335_333562


namespace taller_tree_height_l3335_333583

theorem taller_tree_height (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 20 →  -- One tree is 20 feet taller than the other
  h₁ / h₂ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₂ = 70 :=  -- The height of the taller tree is 70 feet
by sorry

end taller_tree_height_l3335_333583


namespace inscribed_sphere_radius_l3335_333522

/-- Represents a cone with given base radius and height -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere with given radius -/
structure Sphere where
  radius : ℝ

/-- Checks if a sphere is inscribed in a cone -/
def isInscribed (c : Cone) (s : Sphere) : Prop :=
  -- This is a placeholder for the actual geometric condition
  True

theorem inscribed_sphere_radius (c : Cone) (s : Sphere) 
  (h1 : c.baseRadius = 15)
  (h2 : c.height = 30)
  (h3 : isInscribed c s) :
  s.radius = 7.5 * Real.sqrt 5 - 7.5 := by
  sorry

end inscribed_sphere_radius_l3335_333522


namespace journal_problem_formula_l3335_333543

def f (x y : ℕ) : ℕ := 5 * x + 60 * (y - 1970) - 4

theorem journal_problem_formula (x y : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 12) (hy : 1970 ≤ y ∧ y ≤ 1989) : 
  (f 1 1970 = 1) ∧
  (∀ x' y', 1 ≤ x' ∧ x' < 12 → f (x' + 1) y' = f x' y' + 5) ∧
  (∀ y', f 1 (y' + 1) = f 1 y' + 60) →
  f x y = 5 * x + 60 * (y - 1970) - 4 :=
by sorry

end journal_problem_formula_l3335_333543


namespace complex_number_properties_l3335_333589

variable (a : ℝ)
variable (b : ℝ)
def z : ℂ := a + Complex.I

theorem complex_number_properties :
  (∀ z, Complex.abs z = 1 → a = 0) ∧
  (∀ z, (z / (1 + Complex.I)).im = 0 → a = 1) ∧
  (∀ z b, z^2 + b*z + 2 = 0 → ((a = 1 ∧ b = -2) ∨ (a = -1 ∧ b = 2))) :=
by sorry

end complex_number_properties_l3335_333589


namespace complete_square_with_integer_l3335_333571

theorem complete_square_with_integer (y : ℝ) : 
  ∃ (k : ℤ) (a : ℝ), y^2 + 12*y + 44 = (y + a)^2 + k ∧ k = 8 := by
  sorry

end complete_square_with_integer_l3335_333571


namespace polynomial_expansion_theorem_l3335_333502

/-- Given (2x-1)^5 = ax^5 + bx^4 + cx^3 + dx^2 + ex + f, prove the following statements -/
theorem polynomial_expansion_theorem (a b c d e f : ℝ) :
  (∀ x, (2*x - 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a + b + c + d + e + f = 1) ∧
  (b + c + d + e = -30) ∧
  (a + c + e = 122) := by
  sorry

end polynomial_expansion_theorem_l3335_333502


namespace community_families_count_l3335_333581

theorem community_families_count :
  let families_with_two_dogs : ℕ := 15
  let families_with_one_dog : ℕ := 20
  let total_animals : ℕ := 80
  let total_dogs : ℕ := families_with_two_dogs * 2 + families_with_one_dog
  let total_cats : ℕ := total_animals - total_dogs
  let families_with_cats : ℕ := total_cats / 2
  families_with_two_dogs + families_with_one_dog + families_with_cats = 50 :=
by sorry

end community_families_count_l3335_333581
