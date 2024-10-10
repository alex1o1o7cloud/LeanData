import Mathlib

namespace tanya_bought_eleven_pears_l1767_176739

/-- Represents the number of pears Tanya bought -/
def num_pears : ℕ := sorry

/-- Represents the number of Granny Smith apples Tanya bought -/
def num_apples : ℕ := 4

/-- Represents the number of pineapples Tanya bought -/
def num_pineapples : ℕ := 2

/-- Represents the basket of plums as a single item -/
def num_plum_baskets : ℕ := 1

/-- Represents the total number of fruit items Tanya bought -/
def total_fruits : ℕ := num_pears + num_apples + num_pineapples + num_plum_baskets

/-- Represents the number of fruits remaining in the bag after half fell out -/
def remaining_fruits : ℕ := 9

theorem tanya_bought_eleven_pears :
  num_pears = 11 ∧
  total_fruits = 2 * remaining_fruits :=
by sorry

end tanya_bought_eleven_pears_l1767_176739


namespace quadratic_t_range_l1767_176741

/-- Represents a quadratic function of the form ax² + bx - 2 --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Theorem statement for the range of t in the given quadratic equation --/
theorem quadratic_t_range (f : QuadraticFunction) 
  (h1 : f.a * (-1)^2 + f.b * (-1) - 2 = 0)  -- -1 is a root
  (h2 : 0 < -f.b / (2 * f.a))  -- vertex x-coordinate is positive (4th quadrant)
  (h3 : 0 < f.a)  -- parabola opens upward (4th quadrant)
  : -2 < 3 * f.a + f.b ∧ 3 * f.a + f.b < 6 := by
  sorry

end quadratic_t_range_l1767_176741


namespace divisor_sum_not_divides_l1767_176792

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n

/-- The set of proper divisors of a natural number -/
def ProperDivisors (n : ℕ) : Set ℕ := {d : ℕ | d ∣ n ∧ 1 < d ∧ d < n}

/-- The set of remaining divisors after removing the smaller of each pair -/
def RemainingDivisors (n : ℕ) : Set ℕ :=
  {d ∈ ProperDivisors n | d ≥ n / d}

theorem divisor_sum_not_divides (a b : ℕ) (ha : IsComposite a) (hb : IsComposite b) :
  ∀ (c : ℕ) (d : ℕ), c ∈ RemainingDivisors a → d ∈ RemainingDivisors b →
    ¬((c + d) ∣ (a + b)) := by
  sorry

#check divisor_sum_not_divides

end divisor_sum_not_divides_l1767_176792


namespace symmetric_points_imply_sum_power_l1767_176797

-- Define the points P and Q
def P (m n : ℝ) : ℝ × ℝ := (m - 1, n + 2)
def Q (m : ℝ) : ℝ × ℝ := (2 * m - 4, 2)

-- Define the symmetry condition
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem symmetric_points_imply_sum_power (m n : ℝ) :
  symmetric_x_axis (P m n) (Q m) → (m + n)^2023 = -1 := by
  sorry

#check symmetric_points_imply_sum_power

end symmetric_points_imply_sum_power_l1767_176797


namespace graph_transformation_l1767_176765

/-- Given a function f, prove that (1/3)f(x) + 2 is equivalent to scaling f(x) vertically by 1/3 and shifting up by 2 -/
theorem graph_transformation (f : ℝ → ℝ) (x : ℝ) :
  (1/3) * (f x) + 2 = ((1/3) * f x) + 2 := by sorry

end graph_transformation_l1767_176765


namespace hyperbola_parabola_property_l1767_176767

/-- Given a hyperbola and a parabola with specific properties, prove that 2e - b² = 4 -/
theorem hyperbola_parabola_property (a b : ℝ) (e : ℝ) :
  a > 0 →
  b > 0 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ y^2 = 4*x) →  -- Common point exists
  (∃ (x₀ y₀ : ℝ), x₀^2 / a^2 - y₀^2 / b^2 = 1 ∧ y₀^2 = 4*x₀ ∧ x₀ + 1 = 2) →  -- Distance to directrix is 2
  e = Real.sqrt (1 + b^2 / a^2) →  -- Definition of hyperbola eccentricity
  2*e - b^2 = 4 := by
sorry

end hyperbola_parabola_property_l1767_176767


namespace probability_specific_arrangement_l1767_176723

theorem probability_specific_arrangement (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  (1 : ℚ) / (n.choose k) = (1 : ℚ) / 35 :=
sorry

end probability_specific_arrangement_l1767_176723


namespace power_three_nineteen_mod_ten_l1767_176768

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by
  sorry

end power_three_nineteen_mod_ten_l1767_176768


namespace prob_and_expectation_l1767_176791

variable (K N M : ℕ) (p : ℝ)

-- Probability that exactly M out of K items are known by at least one of N agents
def prob_exact_M_known : ℝ := 
  (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * (1 - p)^(N * (K - M))

-- Expected number of items known by at least one agent
def expected_items_known : ℝ := K * (1 - (1 - p)^N)

-- Theorem statement
theorem prob_and_expectation (h_p : 0 ≤ p ∧ p ≤ 1) (h_K : K > 0) (h_N : N > 0) (h_M : M ≤ K) :
  (prob_exact_M_known K N M p = (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * (1 - p)^(N * (K - M))) ∧
  (expected_items_known K N p = K * (1 - (1 - p)^N)) := by sorry

end prob_and_expectation_l1767_176791


namespace circle_symmetric_point_theorem_l1767_176799

/-- A circle C in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 + 2*c.a*p.x - 4*p.y + c.b = 0

/-- Find the symmetric point of a given point about the line x + y - 3 = 0 -/
def symmetricPoint (p : Point) : Point :=
  { x := 2 - p.y, y := 2 - p.x }

/-- Main theorem -/
theorem circle_symmetric_point_theorem (c : Circle) : 
  let p : Point := { x := 1, y := 4 }
  (p.onCircle c ∧ (symmetricPoint p).onCircle c) → c.a = -1 ∧ c.b = 1 := by
  sorry

end circle_symmetric_point_theorem_l1767_176799


namespace missing_digit_is_four_l1767_176716

def set_of_numbers : List Nat := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

def arithmetic_mean (numbers : List Nat) : Rat :=
  (numbers.sum : Rat) / numbers.length

theorem missing_digit_is_four :
  let mean := arithmetic_mean set_of_numbers
  ∃ (n : Nat), 
    (n = Int.floor mean) ∧ 
    (n ≥ 100000000 ∧ n < 1000000000) ∧ 
    (∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ d₂) ∧
    (4 ∉ n.digits 10) :=
by sorry

end missing_digit_is_four_l1767_176716


namespace arithmetic_sequence_range_l1767_176750

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem arithmetic_sequence_range :
  ∀ d : ℝ,
  (arithmetic_sequence (-24) d 1 = -24) →
  (arithmetic_sequence (-24) d 10 > 0) →
  (arithmetic_sequence (-24) d 9 ≤ 0) →
  (8/3 < d ∧ d ≤ 3) :=
by sorry

end arithmetic_sequence_range_l1767_176750


namespace sarah_bowling_score_l1767_176708

theorem sarah_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 30 →
  (sarah_score + greg_score) / 2 = 95 →
  sarah_score = 110 := by
  sorry

end sarah_bowling_score_l1767_176708


namespace even_increasing_inequality_l1767_176766

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of f being increasing on (-∞, -1]
def increasing_on_neg_infinity_to_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ -1 → f x < f y

-- State the theorem
theorem even_increasing_inequality 
  (h_even : is_even f) 
  (h_incr : increasing_on_neg_infinity_to_neg_one f) : 
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) :=
sorry

end even_increasing_inequality_l1767_176766


namespace quadratic_equation_transformation_l1767_176740

theorem quadratic_equation_transformation (x : ℝ) :
  (x^2 + 2*x - 2 = 0) ↔ ((x + 1)^2 = 3) :=
sorry

end quadratic_equation_transformation_l1767_176740


namespace division_problem_addition_problem_multiplication_problem_l1767_176778

-- Problem 1
theorem division_problem : 246 / 73 = 3 + 27 / 73 := by sorry

-- Problem 2
theorem addition_problem : 9999 + 999 + 99 + 9 = 11106 := by sorry

-- Problem 3
theorem multiplication_problem : 25 * 29 * 4 = 2900 := by sorry

end division_problem_addition_problem_multiplication_problem_l1767_176778


namespace inverse_sum_property_l1767_176757

-- Define a function f with domain ℝ and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property that f is invertible
def is_inverse (f f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the theorem
theorem inverse_sum_property
  (h1 : is_inverse f f_inv)
  (h2 : ∀ x : ℝ, f x + f (-x) = 1) :
  ∀ x : ℝ, f_inv (2010 - x) + f_inv (x - 2009) = 0 :=
sorry

end inverse_sum_property_l1767_176757


namespace perfect_square_quadratic_l1767_176783

theorem perfect_square_quadratic (n : ℤ) : ∃ m : ℤ, 4 * n^2 + 12 * n + 9 = m^2 := by
  sorry

end perfect_square_quadratic_l1767_176783


namespace total_is_450_l1767_176712

/-- The number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- The number of vacations Grant has -/
def grant_vacations : ℕ := 4 * kelvin_classes

/-- The total number of vacations and classes for Grant and Kelvin -/
def total_vacations_and_classes : ℕ := grant_vacations + kelvin_classes

theorem total_is_450 : total_vacations_and_classes = 450 := by
  sorry

end total_is_450_l1767_176712


namespace every_algorithm_has_sequential_structure_l1767_176754

/-- An algorithm is a sequence of well-defined instructions for solving a problem or performing a task. -/
def Algorithm : Type := Unit

/-- A sequential structure is a series of steps executed in a specific order. -/
def SequentialStructure : Type := Unit

/-- Every algorithm has a sequential structure. -/
theorem every_algorithm_has_sequential_structure :
  ∀ (a : Algorithm), ∃ (s : SequentialStructure), True :=
sorry

end every_algorithm_has_sequential_structure_l1767_176754


namespace range_of_f_on_I_l1767_176743

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Define the interval
def I : Set ℝ := {x | -5 ≤ x ∧ x ≤ 0}

-- State the theorem
theorem range_of_f_on_I :
  {y | ∃ x ∈ I, f x = y} = {y | -12 ≤ y ∧ y ≤ 4} := by sorry

end range_of_f_on_I_l1767_176743


namespace greatest_common_measure_of_segments_l1767_176749

/-- The greatest common measure of two segments of lengths 19 cm and 190 cm is 19 cm, not 1 cm -/
theorem greatest_common_measure_of_segments (segment1 : ℕ) (segment2 : ℕ) 
  (h1 : segment1 = 19) (h2 : segment2 = 190) :
  Nat.gcd segment1 segment2 = 19 ∧ Nat.gcd segment1 segment2 ≠ 1 := by
  sorry

end greatest_common_measure_of_segments_l1767_176749


namespace water_needed_for_mixture_l1767_176784

/-- Given the initial mixture composition and the desired total volume, 
    prove that the amount of water needed is 0.24 liters. -/
theorem water_needed_for_mixture (initial_chemical_b : ℝ) (initial_water : ℝ) 
  (initial_mixture : ℝ) (desired_volume : ℝ) 
  (h1 : initial_chemical_b = 0.05)
  (h2 : initial_water = 0.03)
  (h3 : initial_mixture = 0.08)
  (h4 : desired_volume = 0.64)
  (h5 : initial_chemical_b + initial_water = initial_mixture) : 
  desired_volume * (initial_water / initial_mixture) = 0.24 := by
  sorry

end water_needed_for_mixture_l1767_176784


namespace sum_of_three_numbers_l1767_176747

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 156) (h2 : a*b + b*c + a*c = 50) :
  a + b + c = 16 := by
  sorry

end sum_of_three_numbers_l1767_176747


namespace inverse_of_10_mod_997_l1767_176755

theorem inverse_of_10_mod_997 : 
  ∃ x : ℕ, x < 997 ∧ (10 * x) % 997 = 1 :=
by
  use 709
  sorry

end inverse_of_10_mod_997_l1767_176755


namespace fraction_equivalence_l1767_176759

theorem fraction_equivalence : (3 : ℚ) / 7 = 27 / 63 := by
  sorry

end fraction_equivalence_l1767_176759


namespace election_votes_calculation_l1767_176706

theorem election_votes_calculation (total_votes : ℕ) :
  let valid_votes_percentage : ℚ := 85 / 100
  let candidate_a_percentage : ℚ := 75 / 100
  let candidate_a_votes : ℕ := 357000
  (↑candidate_a_votes : ℚ) = candidate_a_percentage * (valid_votes_percentage * ↑total_votes) →
  total_votes = 560000 := by
  sorry

end election_votes_calculation_l1767_176706


namespace green_blue_difference_l1767_176721

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles needed for a two-layer border of a hexagon -/
def border_tiles : ℕ := 6 * 6

/-- Represents the new figure after adding a border -/
def new_figure (fig : HexFigure) : HexFigure :=
  { blue_tiles := fig.blue_tiles,
    green_tiles := fig.green_tiles + border_tiles }

/-- The main theorem to prove -/
theorem green_blue_difference (fig : HexFigure) 
  (h1 : fig.blue_tiles = 20) 
  (h2 : fig.green_tiles = 8) : 
  (new_figure fig).green_tiles - (new_figure fig).blue_tiles = 24 := by
  sorry

#check green_blue_difference

end green_blue_difference_l1767_176721


namespace remainder_9053_div_98_l1767_176763

theorem remainder_9053_div_98 : 9053 % 98 = 37 := by
  sorry

end remainder_9053_div_98_l1767_176763


namespace octahedron_faces_l1767_176773

/-- An octahedron is a polyhedron with a specific number of faces -/
structure Octahedron where
  faces : ℕ

/-- The number of faces of an octahedron is 8 -/
theorem octahedron_faces (o : Octahedron) : o.faces = 8 := by
  sorry

end octahedron_faces_l1767_176773


namespace gilbert_judah_ratio_l1767_176793

/-- The number of crayons in each person's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ

/-- The conditions of the crayon box problem -/
def crayon_box_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = 2 * boxes.gilbert ∧
  boxes.gilbert = boxes.judah ∧
  boxes.karen = 128 ∧
  boxes.judah = 8

/-- The theorem stating the ratio of crayons in Gilbert's box to Judah's box -/
theorem gilbert_judah_ratio (boxes : CrayonBoxes) 
  (h : crayon_box_conditions boxes) : 
  boxes.gilbert / boxes.judah = 4 := by
  sorry


end gilbert_judah_ratio_l1767_176793


namespace inequality_implies_sum_nonpositive_l1767_176762

theorem inequality_implies_sum_nonpositive 
  {a b x y : ℝ} 
  (h1 : 1 < a) 
  (h2 : a < b) 
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) : 
  x + y ≤ 0 := by
sorry

end inequality_implies_sum_nonpositive_l1767_176762


namespace sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one_l1767_176789

theorem sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one :
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) = 1 := by
  sorry

end sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one_l1767_176789


namespace initial_bottle_caps_l1767_176722

theorem initial_bottle_caps (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 7 → total = 14 → total = initial + added → initial = 7 := by sorry

end initial_bottle_caps_l1767_176722


namespace gene_separation_in_Aa_genotype_l1767_176713

-- Define the stages of spermatogenesis
inductive SpermatogenesisStage
  | formation_primary_spermatocytes
  | formation_secondary_spermatocytes
  | formation_spermatids
  | formation_sperm

-- Define alleles
inductive Allele
  | A
  | a

-- Define the separation event
structure SeparationEvent where
  allele1 : Allele
  allele2 : Allele
  stage : SpermatogenesisStage

-- Define the genotype
def GenotypeAa : List Allele := [Allele.A, Allele.a]

-- Define the correct separation sequence
def CorrectSeparationSequence : List SpermatogenesisStage :=
  [SpermatogenesisStage.formation_spermatids,
   SpermatogenesisStage.formation_spermatids,
   SpermatogenesisStage.formation_secondary_spermatocytes]

-- Theorem statement
theorem gene_separation_in_Aa_genotype :
  ∀ (separation_events : List SeparationEvent),
    (∀ e ∈ separation_events, e.allele1 ∈ GenotypeAa ∧ e.allele2 ∈ GenotypeAa) →
    (∃ (e1 e2 e3 : SeparationEvent),
      e1 ∈ separation_events ∧
      e2 ∈ separation_events ∧
      e3 ∈ separation_events ∧
      e1.allele1 = Allele.A ∧ e1.allele2 = Allele.A ∧
      e2.allele1 = Allele.a ∧ e2.allele2 = Allele.a ∧
      e3.allele1 = Allele.A ∧ e3.allele2 = Allele.a) →
    (separation_events.map (λ e => e.stage)) = CorrectSeparationSequence :=
by sorry

end gene_separation_in_Aa_genotype_l1767_176713


namespace hash_difference_l1767_176796

def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 6 5) - (hash 5 6) = -4 := by
  sorry

end hash_difference_l1767_176796


namespace polynomial_remainder_l1767_176733

def polynomial (y : ℝ) : ℝ := y^5 - 8*y^4 + 12*y^3 + 25*y^2 - 40*y + 24

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, polynomial = (λ y => (y - 4) * q y + 8) := by sorry

end polynomial_remainder_l1767_176733


namespace x_squared_plus_inverse_squared_l1767_176760

theorem x_squared_plus_inverse_squared (x : ℝ) : x^2 - x - 1 = 0 → x^2 + 1/x^2 = 3 := by
  sorry

end x_squared_plus_inverse_squared_l1767_176760


namespace people_studying_cooking_and_weaving_l1767_176761

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem people_studying_cooking_and_weaving 
  (cp : CurriculumParticipation)
  (h1 : cp.yoga = 35)
  (h2 : cp.cooking = 20)
  (h3 : cp.weaving = 15)
  (h4 : cp.cookingOnly = 7)
  (h5 : cp.cookingAndYoga = 5)
  (h6 : cp.allCurriculums = 3) :
  ∃ n : ℕ, n = cp.cooking - cp.cookingOnly - (cp.cookingAndYoga - cp.allCurriculums) - cp.allCurriculums ∧ n = 8 := by
  sorry

#check people_studying_cooking_and_weaving

end people_studying_cooking_and_weaving_l1767_176761


namespace problem_solution_l1767_176790

theorem problem_solution (x y : ℝ) 
  (h1 : 2 * x + y = 7) 
  (h2 : (x + y) / 3 = 1.6666666666666667) : 
  x + 2 * y = 8 := by
  sorry

end problem_solution_l1767_176790


namespace function_properties_l1767_176742

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * a * x - 1 / x

theorem function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1: If f'(1) = -2, then a = 1
  (f_derivative a 1 = -2) → a = 1 ∧
  -- Part 2: When a ≥ 1/8, f(x) is monotonically decreasing
  (a ≥ 1/8 → ∀ x > 0, f_derivative a x ≤ 0) :=
sorry

end

end function_properties_l1767_176742


namespace stall_owner_earnings_l1767_176705

/-- Represents the number of yellow balls in the bag -/
def yellow_balls : ℕ := 3

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + white_balls

/-- Represents the number of balls drawn in each event -/
def balls_drawn : ℕ := 3

/-- Represents the probability of drawing 3 yellow balls -/
def prob_3_yellow : ℚ := 1 / 20

/-- Represents the probability of drawing 3 white balls -/
def prob_3_white : ℚ := 1 / 20

/-- Represents the probability of drawing balls of the same color -/
def prob_same_color : ℚ := prob_3_yellow + prob_3_white

/-- Represents the amount won when drawing 3 balls of the same color -/
def win_amount : ℤ := 10

/-- Represents the amount lost when drawing 3 balls of different colors -/
def loss_amount : ℤ := 2

/-- Represents the number of draws per day -/
def draws_per_day : ℕ := 80

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Theorem: The stall owner's expected earnings in a month are $1920 -/
theorem stall_owner_earnings : 
  (draws_per_day * days_in_month * 
    (prob_same_color * win_amount - (1 - prob_same_color) * loss_amount)) = 1920 := by
  sorry

end stall_owner_earnings_l1767_176705


namespace right_triangle_area_divisibility_l1767_176724

theorem right_triangle_area_divisibility (a b c : ℕ) : 
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  c % 5 ≠ 0 → -- hypotenuse not divisible by 5
  ∃ k : ℕ, a * b = 20 * k -- area is divisible by 10
  := by sorry

end right_triangle_area_divisibility_l1767_176724


namespace weaving_productivity_l1767_176753

/-- Represents the daily increase in fabric production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days -/
def days : ℕ := 30

/-- Represents the initial daily production -/
def initial_production : ℚ := 5

/-- Represents the total production over the given period -/
def total_production : ℚ := 390

/-- Theorem stating the relationship between the daily increase and total production -/
theorem weaving_productivity :
  days * initial_production + (days * (days - 1) / 2) * daily_increase = total_production :=
sorry

end weaving_productivity_l1767_176753


namespace interest_rate_proof_l1767_176774

/-- Given a principal sum and a time period of 2 years, if the simple interest
    is one-fifth of the principal sum, then the rate of interest per annum is 10%. -/
theorem interest_rate_proof (P : ℝ) (P_pos : P > 0) : 
  (P * 2 * 10 / 100 = P / 5) → 10 = (P / 5) / P * 100 / 2 := by
  sorry

#check interest_rate_proof

end interest_rate_proof_l1767_176774


namespace difference_even_plus_five_minus_odd_l1767_176738

/-- Sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * (2 * n - 1)

/-- Sum of the first n even counting numbers plus 5 added to each number -/
def sumEvenNumbersPlusFive (n : ℕ) : ℕ := n * (2 * n + 5)

/-- The difference between the sum of the first 3000 even counting numbers plus 5 
    added to each number and the sum of the first 3000 odd counting numbers is 18000 -/
theorem difference_even_plus_five_minus_odd : 
  sumEvenNumbersPlusFive 3000 - sumOddNumbers 3000 = 18000 := by
  sorry

end difference_even_plus_five_minus_odd_l1767_176738


namespace ethan_candle_coconut_oil_l1767_176777

/-- The amount of coconut oil used in each candle, given the total weight of candles,
    the number of candles, and the amount of beeswax per candle. -/
def coconut_oil_per_candle (total_weight : ℕ) (num_candles : ℕ) (beeswax_per_candle : ℕ) : ℚ :=
  (total_weight - num_candles * beeswax_per_candle) / num_candles

theorem ethan_candle_coconut_oil :
  coconut_oil_per_candle 63 (10 - 3) 8 = 1 := by sorry

end ethan_candle_coconut_oil_l1767_176777


namespace ellipse_higher_focus_coordinates_l1767_176744

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  majorAxis : Point × Point
  minorAxis : Point × Point

/-- The focus of an ellipse with higher y-coordinate -/
def higherFocus (e : Ellipse) : Point :=
  sorry

theorem ellipse_higher_focus_coordinates :
  let e : Ellipse := {
    majorAxis := (⟨3, 0⟩, ⟨3, 8⟩),
    minorAxis := (⟨1, 4⟩, ⟨5, 4⟩)
  }
  let focus := higherFocus e
  focus.x = 3 ∧ focus.y = 4 + 2 * Real.sqrt 3 :=
by sorry

end ellipse_higher_focus_coordinates_l1767_176744


namespace cyclists_time_apart_l1767_176758

/-- Calculates the time taken for two cyclists to be 200 miles apart -/
theorem cyclists_time_apart (v_east : ℝ) (v_west : ℝ) (distance : ℝ) : 
  v_east = 22 →
  v_west = v_east + 4 →
  distance = 200 →
  (distance / (v_east + v_west) : ℝ) = 25 / 6 := by
  sorry

#check cyclists_time_apart

end cyclists_time_apart_l1767_176758


namespace inverse_sum_mod_31_l1767_176785

theorem inverse_sum_mod_31 : ∃ (a b : ℤ), (5 * a) % 31 = 1 ∧ (5 * 5 * 5 * b) % 31 = 1 ∧ (a + b) % 31 = 5 := by
  sorry

end inverse_sum_mod_31_l1767_176785


namespace cheryls_expenses_l1767_176737

/-- Cheryl's golf tournament expenses problem -/
theorem cheryls_expenses (electricity_bill : ℝ) : 
  -- Golf tournament cost is 20% more than monthly cell phone expenses
  -- Monthly cell phone expenses are $400 more than electricity bill
  -- Total payment for golf tournament is $1440
  (1.2 * (electricity_bill + 400) = 1440) →
  -- Cheryl's electricity bill cost is $800
  electricity_bill = 800 := by
  sorry

end cheryls_expenses_l1767_176737


namespace inequality_solution_set_range_of_a_l1767_176779

-- Define the function f
def f (x : ℝ) : ℝ := |3*x + 2|

-- Theorem for part 1
theorem inequality_solution_set :
  {x : ℝ | f x < 4 - |x - 1|} = {x : ℝ | -5/4 < x ∧ x < 1/2} :=
sorry

-- Theorem for part 2
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ |x - a| - f x ≤ 1/m + 1/n) →
  ∃ a : ℝ, 0 < a ∧ a ≤ 10/3 :=
sorry

end inequality_solution_set_range_of_a_l1767_176779


namespace quadratic_equation_root_l1767_176703

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 5 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 5 = 0 ∧ y = -5/3) :=
by sorry

end quadratic_equation_root_l1767_176703


namespace cyclic_sum_squares_identity_l1767_176751

theorem cyclic_sum_squares_identity (a b c x y z : ℝ) :
  (a * x + b * y + c * z)^2 + (b * x + c * y + a * z)^2 + (c * x + a * y + b * z)^2 =
  (c * x + b * y + a * z)^2 + (b * x + a * y + c * z)^2 + (a * x + c * y + b * z)^2 :=
by sorry

end cyclic_sum_squares_identity_l1767_176751


namespace cross_in_square_l1767_176772

theorem cross_in_square (x : ℝ) : 
  x > 0 → 
  (5 / 8) * x^2 = 810 → 
  x = 36 :=
by sorry

end cross_in_square_l1767_176772


namespace cubic_roots_sum_l1767_176731

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a * b / c + b * c / a + c * a / b = 49 / 6 := by
sorry

end cubic_roots_sum_l1767_176731


namespace ice_cream_sales_theorem_l1767_176714

/-- The number of ice cream cones sold on Tuesday -/
def tuesday_sales : ℕ := 12000

/-- The number of ice cream cones sold on Wednesday -/
def wednesday_sales : ℕ := 2 * tuesday_sales

/-- The total number of ice cream cones sold on Tuesday and Wednesday -/
def total_sales : ℕ := tuesday_sales + wednesday_sales

theorem ice_cream_sales_theorem : total_sales = 36000 := by
  sorry

end ice_cream_sales_theorem_l1767_176714


namespace sum_highest_powers_12_18_divides_20_factorial_l1767_176756

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_divides (base k : ℕ) : ℕ → ℕ
| 0 => 0
| n + 1 => if (factorial n) % (base ^ (k + 1)) = 0 then highest_power_divides base (k + 1) n else k

theorem sum_highest_powers_12_18_divides_20_factorial :
  (highest_power_divides 12 0 20) + (highest_power_divides 18 0 20) = 12 := by
  sorry

end sum_highest_powers_12_18_divides_20_factorial_l1767_176756


namespace sum_remainder_mod_nine_l1767_176718

theorem sum_remainder_mod_nine : ∃ k : ℕ, 
  88134 + 88135 + 88136 + 88137 + 88138 + 88139 = 9 * k + 6 := by
  sorry

end sum_remainder_mod_nine_l1767_176718


namespace cider_pints_is_180_l1767_176732

/-- Represents the number of pints of cider that can be made given the following conditions:
  * 20 golden delicious, 40 pink lady, and 30 granny smith apples make one pint of cider
  * Each farmhand can pick 120 golden delicious, 240 pink lady, and 180 granny smith apples per hour
  * There are 6 farmhands working 5 hours
  * The ratio of golden delicious : pink lady : granny smith apples gathered is 1:2:1.5
-/
def cider_pints : ℕ :=
  let golden_per_pint : ℕ := 20
  let pink_per_pint : ℕ := 40
  let granny_per_pint : ℕ := 30
  let golden_per_hour : ℕ := 120
  let pink_per_hour : ℕ := 240
  let granny_per_hour : ℕ := 180
  let farmhands : ℕ := 6
  let hours : ℕ := 5
  let golden_total : ℕ := golden_per_hour * hours * farmhands
  let pink_total : ℕ := pink_per_hour * hours * farmhands
  let granny_total : ℕ := granny_per_hour * hours * farmhands
  golden_total / golden_per_pint

theorem cider_pints_is_180 : cider_pints = 180 := by
  sorry

end cider_pints_is_180_l1767_176732


namespace ab_range_l1767_176748

-- Define the function f
def f (x : ℝ) : ℝ := |2 - x^2|

-- State the theorem
theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  0 < a * b ∧ a * b < 2 := by
  sorry

end ab_range_l1767_176748


namespace third_player_win_probability_l1767_176734

/-- Represents a fair six-sided die --/
def FairDie : Finset ℕ := Finset.range 6

/-- The probability of rolling a 6 on a fair die --/
def probWin : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a fair die --/
def probLose : ℚ := 1 - probWin

/-- The number of players --/
def numPlayers : ℕ := 3

/-- The probability that the third player wins the game --/
def probThirdPlayerWins : ℚ := 1 / 91

theorem third_player_win_probability :
  probThirdPlayerWins = (probWin^numPlayers) / (1 - probLose^numPlayers) :=
by sorry

end third_player_win_probability_l1767_176734


namespace flagpole_shadow_length_l1767_176776

/-- Given a flagpole and a building under similar conditions, 
    prove that the flagpole's shadow length is 45 meters. -/
theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 22)
  (h3 : building_shadow = 55)
  (h4 : flagpole_height / building_height = building_shadow / building_shadow) :
  flagpole_height * building_shadow / building_height = 45 :=
by sorry

end flagpole_shadow_length_l1767_176776


namespace jones_elementary_population_l1767_176771

theorem jones_elementary_population :
  let total_students : ℕ := 360
  let boy_percentage : ℚ := 1/2
  let representative_boys : ℕ := 90
  (representative_boys : ℚ) / (boy_percentage * total_students) = boy_percentage →
  total_students = 360 := by
sorry

end jones_elementary_population_l1767_176771


namespace least_k_value_l1767_176752

theorem least_k_value (k : ℤ) : 
  (0.00010101 * (10 : ℝ)^k > 100) → k ≥ 7 ∧ ∀ m : ℤ, m < 7 → (0.00010101 * (10 : ℝ)^m ≤ 100) :=
by sorry

end least_k_value_l1767_176752


namespace modular_sum_equivalence_l1767_176709

theorem modular_sum_equivalence : ∃ (a b c : ℤ),
  (7 * a) % 80 = 1 ∧
  (13 * b) % 80 = 1 ∧
  (15 * c) % 80 = 1 ∧
  (3 * a + 9 * b + 4 * c) % 80 = 34 := by
  sorry

end modular_sum_equivalence_l1767_176709


namespace shortest_altitude_of_triangle_l1767_176720

/-- Given a triangle with sides 18, 24, and 30, its shortest altitude has length 18 -/
theorem shortest_altitude_of_triangle (a b c h1 h2 h3 : ℝ) : 
  a = 18 ∧ b = 24 ∧ c = 30 →
  a^2 + b^2 = c^2 →
  h1 = a ∧ h2 = b ∧ h3 = (2 * (1/2 * a * b)) / c →
  min h1 (min h2 h3) = 18 := by
sorry

end shortest_altitude_of_triangle_l1767_176720


namespace min_value_abs_sum_l1767_176711

theorem min_value_abs_sum (x : ℝ) : |x - 2| + |x + 1| ≥ 3 ∧ ∃ y : ℝ, |y - 2| + |y + 1| = 3 := by
  sorry

end min_value_abs_sum_l1767_176711


namespace first_three_digits_after_decimal_l1767_176730

/-- The first three digits to the right of the decimal point in (2^10 + 1)^(4/3) are 320. -/
theorem first_three_digits_after_decimal (x : ℝ) : x = (2^10 + 1)^(4/3) →
  ∃ n : ℕ, x - ↑n = 0.320 + r ∧ 0 ≤ r ∧ r < 0.001 := by
  sorry

end first_three_digits_after_decimal_l1767_176730


namespace photo_arrangements_l1767_176736

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n distinct objects. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem photo_arrangements (teachers students : ℕ) 
  (h1 : teachers = 4) (h2 : students = 4) : 
  /- Students stand together -/
  (arrangements students * arrangements (teachers + 1) = 2880) ∧ 
  /- No two students are adjacent -/
  (arrangements teachers * permutations (teachers + 1) students = 2880) ∧
  /- Teachers and students alternate -/
  (2 * arrangements teachers * arrangements students = 1152) := by
  sorry

#check photo_arrangements

end photo_arrangements_l1767_176736


namespace elective_course_selection_l1767_176735

theorem elective_course_selection (type_A : ℕ) (type_B : ℕ) : 
  type_A = 4 → type_B = 3 → (type_A + type_B : ℕ) = 7 := by
  sorry

end elective_course_selection_l1767_176735


namespace sphere_volume_from_surface_area_l1767_176727

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end sphere_volume_from_surface_area_l1767_176727


namespace evaluate_expression_l1767_176775

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  sorry

end evaluate_expression_l1767_176775


namespace prob_two_or_more_fail_ge_0_9_l1767_176788

/-- The probability of failure for a single device -/
def p : ℝ := 0.2

/-- The probability of success for a single device -/
def q : ℝ := 1 - p

/-- The number of devices to be tested -/
def n : ℕ := 18

/-- The probability of at least two devices failing out of n tested devices -/
def prob_at_least_two_fail (n : ℕ) : ℝ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

/-- Theorem stating that testing 18 devices ensures a probability of at least 0.9 
    that two or more devices will fail -/
theorem prob_two_or_more_fail_ge_0_9 : prob_at_least_two_fail n ≥ 0.9 := by
  sorry


end prob_two_or_more_fail_ge_0_9_l1767_176788


namespace greatest_power_of_three_dividing_fifteen_factorial_l1767_176719

theorem greatest_power_of_three_dividing_fifteen_factorial : 
  (∃ k : ℕ, k > 0 ∧ 3^k ∣ Nat.factorial 15 ∧ ∀ m : ℕ, m > k → ¬(3^m ∣ Nat.factorial 15)) → 
  (∃ k : ℕ, k > 0 ∧ 3^k ∣ Nat.factorial 15 ∧ ∀ m : ℕ, m > k → ¬(3^m ∣ Nat.factorial 15) ∧ k = 6) :=
by sorry

end greatest_power_of_three_dividing_fifteen_factorial_l1767_176719


namespace equation_solution_l1767_176729

theorem equation_solution : ∃ x : ℝ, 6*x - 3*2*x - 2*3*x + 6 = 0 ∧ x = 1 := by sorry

end equation_solution_l1767_176729


namespace min_cubes_for_representation_l1767_176786

/-- The number of faces on each cube -/
def faces_per_cube : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The length of the number we want to represent -/
def number_length : ℕ := 30

/-- The minimum number of occurrences needed for digits 1-9 -/
def min_occurrences : ℕ := 30

/-- The minimum number of occurrences needed for digit 0 -/
def min_occurrences_zero : ℕ := 29

/-- The total number of digit occurrences needed -/
def total_occurrences : ℕ := (num_digits - 1) * min_occurrences + min_occurrences_zero

/-- The minimum number of cubes needed -/
def min_cubes : ℕ := (total_occurrences + faces_per_cube - 1) / faces_per_cube

theorem min_cubes_for_representation :
  min_cubes = 50 ∧
  min_cubes * faces_per_cube ≥ total_occurrences ∧
  (min_cubes - 1) * faces_per_cube < total_occurrences :=
by sorry

end min_cubes_for_representation_l1767_176786


namespace s_3_equals_149_l1767_176704

-- Define the function s(n)
def s (n : ℕ) : ℕ :=
  let squares := List.range n |>.map (λ i => (i + 1) ^ 2)
  squares.foldl (λ acc x => acc * 10^(Nat.digits 10 x).length + x) 0

-- State the theorem
theorem s_3_equals_149 : s 3 = 149 := by
  sorry

end s_3_equals_149_l1767_176704


namespace unique_valid_x_l1767_176780

def is_valid_x (x : ℕ) : Prop :=
  x > 4 ∧ (x + 4) * (x - 4) * (x^3 + 25) < 1000

theorem unique_valid_x : ∃! x : ℕ, is_valid_x x :=
sorry

end unique_valid_x_l1767_176780


namespace kaylee_age_l1767_176702

/-- Given that in 7 years, Kaylee will be 3 times as old as Matt is now,
    and Matt is currently 5 years old, prove that Kaylee is currently 8 years old. -/
theorem kaylee_age (matt_age : ℕ) (kaylee_age : ℕ) :
  matt_age = 5 →
  kaylee_age + 7 = 3 * matt_age →
  kaylee_age = 8 := by
sorry

end kaylee_age_l1767_176702


namespace meaningful_fraction_l1767_176710

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 4)) ↔ x ≠ 4 := by sorry

end meaningful_fraction_l1767_176710


namespace regular_polygon_with_90_degree_difference_has_8_sides_l1767_176770

-- Define a regular polygon
structure RegularPolygon where
  n : ℕ  -- number of sides
  n_ge_3 : n ≥ 3  -- a polygon has at least 3 sides

-- Define the interior angle of a regular polygon
def interiorAngle (p : RegularPolygon) : ℚ :=
  (p.n - 2) * 180 / p.n

-- Define the exterior angle of a regular polygon
def exteriorAngle (p : RegularPolygon) : ℚ :=
  360 / p.n

-- Theorem: A regular polygon where each interior angle is 90° larger than each exterior angle has 8 sides
theorem regular_polygon_with_90_degree_difference_has_8_sides :
  ∃ (p : RegularPolygon), interiorAngle p - exteriorAngle p = 90 → p.n = 8 :=
sorry

end regular_polygon_with_90_degree_difference_has_8_sides_l1767_176770


namespace angle_ABF_measure_l1767_176764

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle ABF in a regular octagon -/
def angle_ABF (octagon : RegularOctagon) : ℝ := 22.5

theorem angle_ABF_measure (octagon : RegularOctagon) :
  angle_ABF octagon = 22.5 := by
  sorry

end angle_ABF_measure_l1767_176764


namespace particle_probability_l1767_176725

/-- Represents the probability of a particle hitting (0,0) starting from (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1) + P (x-2) (y-2)) / 4

/-- The probability of hitting (0,0) starting from (5,5) is 3805/16384 -/
theorem particle_probability : P 5 5 = 3805 / 16384 := by
  sorry

end particle_probability_l1767_176725


namespace soccer_ball_distribution_l1767_176728

/-- The number of ways to distribute n identical balls into k boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute balls into numbered boxes with constraints --/
def distributeWithConstraints (totalBalls numBoxes : ℕ) : ℕ :=
  let remainingBalls := totalBalls - (numBoxes * (numBoxes + 1) / 2)
  distribute remainingBalls numBoxes

theorem soccer_ball_distribution :
  distributeWithConstraints 9 3 = 10 := by
  sorry

end soccer_ball_distribution_l1767_176728


namespace expression_equality_l1767_176794

theorem expression_equality : (2^1501 + 5^1502)^2 - (2^1501 - 5^1502)^2 = 20 * 10^1501 := by
  sorry

end expression_equality_l1767_176794


namespace problem_solution_l1767_176781

theorem problem_solution (x y : ℝ) (h : |x + 5| + (y - 4)^2 = 0) : (x + y)^99 = -1 := by
  sorry

end problem_solution_l1767_176781


namespace remaining_score_is_80_l1767_176769

/-- Given 5 students with 4 known scores and an average score, 
    calculate the remaining score -/
def remaining_score (s1 s2 s3 s4 : ℕ) (avg : ℚ) : ℚ :=
  5 * avg - (s1 + s2 + s3 + s4)

/-- Theorem: The remaining score is 80 -/
theorem remaining_score_is_80 :
  remaining_score 85 95 75 65 80 = 80 := by
  sorry

#eval remaining_score 85 95 75 65 80

end remaining_score_is_80_l1767_176769


namespace cubic_function_extrema_difference_l1767_176787

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_extrema_difference (a b c : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent at x = 1 is parallel to 6x + 2y + 5 = 0
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b c x ≤ f a b c x_max) ∧
    (∀ x, f a b c x_min ≤ f a b c x) ∧
    (f a b c x_max - f a b c x_min = 4) := by
  sorry

end cubic_function_extrema_difference_l1767_176787


namespace factor_calculation_l1767_176782

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 9 →
  factor * (2 * initial_number + 13) = 93 →
  factor = 3 := by sorry

end factor_calculation_l1767_176782


namespace range_of_f_l1767_176700

def f (x : ℝ) := |x + 5| - |x - 3|

theorem range_of_f :
  ∀ y ∈ Set.range f, -8 ≤ y ∧ y ≤ 8 ∧
  ∀ z, -8 ≤ z ∧ z ≤ 8 → ∃ x, f x = z :=
by sorry

end range_of_f_l1767_176700


namespace club_membership_increase_l1767_176746

theorem club_membership_increase (current_members additional_members : ℕ) 
  (h1 : current_members = 10)
  (h2 : additional_members = 15) :
  let new_total := current_members + additional_members
  new_total - current_members = 15 ∧ new_total > 2 * current_members :=
by sorry

end club_membership_increase_l1767_176746


namespace problem_solution_l1767_176717

theorem problem_solution (a b c d : ℝ) : 
  2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + Real.sqrt (2 * a + 2 * b + 2 * c - 3 * d) →
  d = 23 / 48 := by
  sorry

end problem_solution_l1767_176717


namespace triangle_area_proof_l1767_176726

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem triangle_area_proof (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2) 
  (h_f : f A = 1) (h_dot : 2 * Real.cos A = Real.sqrt 2) : 
  (1/2) * Real.sin A = Real.sqrt 2 / 2 := by
  sorry

end triangle_area_proof_l1767_176726


namespace sum_of_cubes_l1767_176707

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by
sorry

end sum_of_cubes_l1767_176707


namespace adamek_marbles_l1767_176745

theorem adamek_marbles :
  ∀ (n : ℕ), 
    (∃ (a b : ℕ), n = 3 * a ∧ n = 4 * b) →
    (∃ (k : ℕ), 3 * (k + 8) = 4 * k) →
    n = 96 := by
  sorry

end adamek_marbles_l1767_176745


namespace james_arthur_muffin_ratio_muffin_baking_problem_l1767_176715

theorem james_arthur_muffin_ratio : ℕ → ℕ → ℕ
  | arthur_muffins, james_muffins =>
    james_muffins / arthur_muffins

theorem muffin_baking_problem (arthur_muffins james_muffins : ℕ) 
  (h1 : arthur_muffins = 115)
  (h2 : james_muffins = 1380) :
  james_arthur_muffin_ratio arthur_muffins james_muffins = 12 := by
  sorry

end james_arthur_muffin_ratio_muffin_baking_problem_l1767_176715


namespace mans_rate_in_still_water_l1767_176701

/-- Given a man's rowing speeds with and against a stream, calculate his rate in still water. -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 16) 
  (h2 : speed_against_stream = 12) : 
  (speed_with_stream + speed_against_stream) / 2 = 14 := by
  sorry

#check mans_rate_in_still_water

end mans_rate_in_still_water_l1767_176701


namespace exponential_function_property_l1767_176795

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_property (a : ℝ) (b : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≤ 4) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≥ b) →
  (f a (-2) = 4 ∨ f a 1 = 4) →
  (f a (-2) = b ∨ f a 1 = b) →
  (∀ x y : ℝ, x < y → (2 - 7*b)*x > (2 - 7*b)*y) →
  a = 1/2 :=
by sorry

end exponential_function_property_l1767_176795


namespace cylinder_line_distance_theorem_l1767_176798

/-- A cylinder with a square axial cross-section -/
structure SquareCylinder where
  /-- The side length of the square axial cross-section -/
  side : ℝ
  /-- Assertion that the side length is positive -/
  side_pos : 0 < side

/-- A line segment connecting points on the upper and lower bases of the cylinder -/
structure CylinderLineSegment (c : SquareCylinder) where
  /-- The length of the line segment -/
  length : ℝ
  /-- The angle the line segment makes with the base plane -/
  angle : ℝ
  /-- Assertion that the length is positive -/
  length_pos : 0 < length
  /-- Assertion that the angle is between 0 and π/2 -/
  angle_range : 0 < angle ∧ angle < Real.pi / 2

/-- The theorem stating the distance formula and angle range -/
theorem cylinder_line_distance_theorem (c : SquareCylinder) (l : CylinderLineSegment c) :
  ∃ (d : ℝ), d = (1 / 2) * l.length * Real.sqrt (-Real.cos (2 * l.angle)) ∧
  Real.pi / 4 < l.angle ∧ l.angle < 3 * Real.pi / 4 := by
  sorry

end cylinder_line_distance_theorem_l1767_176798
