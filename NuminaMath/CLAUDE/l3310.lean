import Mathlib

namespace four_corner_holes_l3310_331018

/-- Represents the state of a rectangular paper. -/
structure Paper where
  folded : Bool
  holes : List (Nat × Nat)

/-- Represents the folding operations. -/
inductive FoldOperation
  | BottomToTop
  | LeftToRight
  | TopToBottom

/-- Folds the paper according to the given operation. -/
def fold (p : Paper) (op : FoldOperation) : Paper :=
  { p with folded := true }

/-- Punches a hole in the top left corner of the folded paper. -/
def punchHole (p : Paper) : Paper :=
  { p with holes := (0, 0) :: p.holes }

/-- Unfolds the paper and calculates the final hole positions. -/
def unfold (p : Paper) : Paper :=
  { p with 
    folded := false,
    holes := [(0, 0), (0, 1), (1, 0), (1, 1)] }

/-- The main theorem stating that after folding, punching, and unfolding, 
    the paper will have four holes, one in each corner. -/
theorem four_corner_holes (p : Paper) :
  let p1 := fold p FoldOperation.BottomToTop
  let p2 := fold p1 FoldOperation.LeftToRight
  let p3 := fold p2 FoldOperation.TopToBottom
  let p4 := punchHole p3
  let final := unfold p4
  final.holes = [(0, 0), (0, 1), (1, 0), (1, 1)] :=
by sorry

end four_corner_holes_l3310_331018


namespace gene_mutation_not_valid_reason_l3310_331001

/-- Represents a genotype --/
inductive Genotype
  | AA
  | Aa
  | BB
  | Bb
  | AaBB
  | AaBb
  | AAB
  | AaB
  | AABb

/-- Represents possible reasons for missing genes --/
inductive MissingGeneReason
  | GeneMutation
  | ChromosomeNumberVariation
  | ChromosomeStructureVariation
  | MaleSexLinked

/-- Defines the genotypes of individuals A and B --/
def individualA : Genotype := Genotype.AaB
def individualB : Genotype := Genotype.AABb

/-- Determines if a reason is valid for explaining the missing gene --/
def isValidReason (reason : MissingGeneReason) (genotypeA : Genotype) (genotypeB : Genotype) : Prop :=
  match reason with
  | MissingGeneReason.GeneMutation => False
  | _ => True

/-- Theorem stating that gene mutation is not a valid reason for the missing gene --/
theorem gene_mutation_not_valid_reason :
  ¬(isValidReason MissingGeneReason.GeneMutation individualA individualB) := by
  sorry


end gene_mutation_not_valid_reason_l3310_331001


namespace money_distribution_ratio_l3310_331008

def distribute_money (total : ℝ) (p q r s : ℝ) : Prop :=
  p + q + r + s = total ∧
  p = 2 * q ∧
  q = r ∧
  s - p = 250

theorem money_distribution_ratio :
  ∀ (total p q r s : ℝ),
    total = 1000 →
    distribute_money total p q r s →
    s / r = 4 := by
  sorry

end money_distribution_ratio_l3310_331008


namespace greatest_divisor_with_digit_sum_l3310_331053

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem greatest_divisor_with_digit_sum (a b : ℕ) (ha : a = 4665) (hb : b = 6905) :
  ∃ (n : ℕ), n = 40 ∧ 
  (b - a) % n = 0 ∧
  sum_of_digits n = 4 ∧
  ∀ (m : ℕ), m > n → ((b - a) % m = 0 → sum_of_digits m ≠ 4) :=
sorry

end greatest_divisor_with_digit_sum_l3310_331053


namespace second_player_guarantee_seven_moves_l3310_331055

/-- Represents a game on a polygon where two players mark vertices alternately --/
structure PolygonGame where
  sides : ℕ
  -- Assume sides ≥ 3 for a valid polygon

/-- Represents a strategy for the second player --/
def SecondPlayerStrategy := PolygonGame → ℕ

/-- The maximum number of moves the second player can guarantee --/
def maxGuaranteedMoves (game : PolygonGame) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that for a 129-sided polygon, the second player can guarantee at least 7 moves --/
theorem second_player_guarantee_seven_moves :
  ∀ (game : PolygonGame),
    game.sides = 129 →
    maxGuaranteedMoves game ≥ 7 := by
  sorry

end second_player_guarantee_seven_moves_l3310_331055


namespace tinas_career_difference_l3310_331062

/-- Represents the career of a boxer --/
structure BoxerCareer where
  initial_wins : ℕ
  additional_wins_before_first_loss : ℕ
  losses : ℕ

/-- Calculates the total wins for a boxer's career --/
def total_wins (career : BoxerCareer) : ℕ :=
  career.initial_wins + career.additional_wins_before_first_loss + 
  (career.initial_wins + career.additional_wins_before_first_loss)

/-- Theorem stating the difference between wins and losses for Tina's career --/
theorem tinas_career_difference : 
  ∀ (career : BoxerCareer), 
  career.initial_wins = 10 → 
  career.additional_wins_before_first_loss = 5 → 
  career.losses = 2 → 
  total_wins career - career.losses = 43 :=
by sorry

end tinas_career_difference_l3310_331062


namespace caterpillar_problem_solution_l3310_331049

/-- Represents the caterpillar problem --/
structure CaterpillarProblem where
  initial_caterpillars : ℕ
  fallen_caterpillars : ℕ
  hatched_eggs : ℕ
  leaves_per_day : ℕ
  observation_days : ℕ
  cocooned_caterpillars : ℕ

/-- Calculates the number of caterpillars left on the tree and leaves eaten --/
def solve_caterpillar_problem (problem : CaterpillarProblem) : ℕ × ℕ :=
  let remaining_after_storm := problem.initial_caterpillars - problem.fallen_caterpillars
  let total_after_hatching := remaining_after_storm + problem.hatched_eggs
  let remaining_after_cocooning := total_after_hatching - problem.cocooned_caterpillars
  let final_caterpillars := remaining_after_cocooning / 2
  let leaves_eaten := problem.hatched_eggs * problem.leaves_per_day * problem.observation_days
  (final_caterpillars, leaves_eaten)

/-- Theorem stating the solution to the caterpillar problem --/
theorem caterpillar_problem_solution :
  let problem : CaterpillarProblem := {
    initial_caterpillars := 14,
    fallen_caterpillars := 3,
    hatched_eggs := 6,
    leaves_per_day := 2,
    observation_days := 7,
    cocooned_caterpillars := 9
  }
  solve_caterpillar_problem problem = (4, 84) := by
  sorry


end caterpillar_problem_solution_l3310_331049


namespace max_subsets_with_intersection_property_l3310_331044

/-- The maximum number of distinct subsets satisfying the intersection property -/
theorem max_subsets_with_intersection_property (n : ℕ) :
  (∃ (t : ℕ) (A : Fin t → Finset (Fin n)),
    (∀ i j k, i < j → j < k → (A i ∩ A k) ⊆ A j) ∧
    (∀ i j, i ≠ j → A i ≠ A j)) →
  (∀ (t : ℕ) (A : Fin t → Finset (Fin n)),
    (∀ i j k, i < j → j < k → (A i ∩ A k) ⊆ A j) ∧
    (∀ i j, i ≠ j → A i ≠ A j) →
    t ≤ 2 * n + 1) :=
by sorry

end max_subsets_with_intersection_property_l3310_331044


namespace circle_M_equation_l3310_331032

-- Define the line on which point M lies
def line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

-- Define that points (3,0) and (0,1) lie on circle M
def points_on_circle : Prop := circle_M 3 0 ∧ circle_M 0 1

-- Theorem statement
theorem circle_M_equation : 
  ∃ (x y : ℝ), line x y ∧ points_on_circle → circle_M x y :=
sorry

end circle_M_equation_l3310_331032


namespace total_pumpkins_sold_l3310_331084

/-- Represents the price of a jumbo pumpkin in dollars -/
def jumbo_price : ℚ := 9

/-- Represents the price of a regular pumpkin in dollars -/
def regular_price : ℚ := 4

/-- Represents the total amount collected in dollars -/
def total_collected : ℚ := 395

/-- Represents the number of regular pumpkins sold -/
def regular_sold : ℕ := 65

/-- Theorem stating that the total number of pumpkins sold is 80 -/
theorem total_pumpkins_sold : 
  ∃ (jumbo_sold : ℕ), 
    (jumbo_price * jumbo_sold + regular_price * regular_sold = total_collected) ∧
    (jumbo_sold + regular_sold = 80) := by
  sorry

end total_pumpkins_sold_l3310_331084


namespace expression_equality_l3310_331038

theorem expression_equality (a b : ℝ) :
  (-a * b^2)^3 + a * b^2 * (a * b)^2 * (-2 * b)^2 = 3 * a^3 * b^6 := by
  sorry

end expression_equality_l3310_331038


namespace total_length_of_T_l3310_331033

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; |(|(|x| - 3)| - 1)| + |(|(|y| - 3)| - 1)| = 2}

-- Define the total length function
def totalLength (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem total_length_of_T : totalLength T = 32 * Real.sqrt 2 := by sorry

end total_length_of_T_l3310_331033


namespace usual_time_calculation_l3310_331048

theorem usual_time_calculation (T : ℝ) 
  (h1 : T > 0) 
  (h2 : (1 : ℝ) / 0.25 = (T + 24) / T) : T = 8 := by
  sorry

end usual_time_calculation_l3310_331048


namespace power_three_nineteen_mod_ten_l3310_331077

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by sorry

end power_three_nineteen_mod_ten_l3310_331077


namespace fixed_points_subset_stable_points_quadratic_no_fixed_points_implies_no_stable_points_l3310_331023

/-- Fixed points of a function -/
def fixed_points (f : ℝ → ℝ) : Set ℝ := {x | f x = x}

/-- Stable points of a function -/
def stable_points (f : ℝ → ℝ) : Set ℝ := {x | f (f x) = x}

theorem fixed_points_subset_stable_points (f : ℝ → ℝ) :
  fixed_points f ⊆ stable_points f := by sorry

theorem quadratic_no_fixed_points_implies_no_stable_points
  (a b c : ℝ) (h : a ≠ 0) (f : ℝ → ℝ) (hf : ∀ x, f x = a * x^2 + b * x + c) :
  fixed_points f = ∅ → stable_points f = ∅ := by sorry

end fixed_points_subset_stable_points_quadratic_no_fixed_points_implies_no_stable_points_l3310_331023


namespace fred_total_cents_l3310_331007

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes Fred has -/
def fred_dimes : ℕ := 9

/-- Theorem: Fred's total cents is 90 -/
theorem fred_total_cents : fred_dimes * dime_value = 90 := by
  sorry

end fred_total_cents_l3310_331007


namespace prism_volume_l3310_331057

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h1 : side_area = 18)
  (h2 : front_area = 12)
  (h3 : bottom_area = 8) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 24 * Real.sqrt 3 := by
  sorry

end prism_volume_l3310_331057


namespace postcard_price_calculation_bernie_postcard_problem_l3310_331075

theorem postcard_price_calculation (initial_postcards : Nat) 
  (sold_postcards : Nat) (price_per_sold : Nat) (final_total : Nat) : Nat :=
  let total_earned := sold_postcards * price_per_sold
  let remaining_original := initial_postcards - sold_postcards
  let new_postcards := final_total - remaining_original
  total_earned / new_postcards

theorem bernie_postcard_problem : 
  postcard_price_calculation 18 9 15 36 = 5 := by
  sorry

end postcard_price_calculation_bernie_postcard_problem_l3310_331075


namespace quadratic_coefficients_l3310_331014

-- Define ω as a complex number
variable (ω : ℂ)

-- Define the conditions
def omega_condition := ω^5 = 1 ∧ ω ≠ 1

-- Define α and β
def α := ω + ω^2
def β := ω^3 + ω^4

-- Define the theorem
theorem quadratic_coefficients (h : omega_condition ω) : 
  ∃ (p : ℝ × ℝ), p.1 = 0 ∧ p.2 = 2 ∧ 
  (α ω)^2 + p.1 * (α ω) + p.2 = 0 ∧ 
  (β ω)^2 + p.1 * (β ω) + p.2 = 0 := by
  sorry

end quadratic_coefficients_l3310_331014


namespace bus_trip_distance_l3310_331004

/-- The distance of a bus trip given specific speed conditions -/
theorem bus_trip_distance : ∃ (d : ℝ), 
  (d / 45 = d / 50 + 1) ∧ d = 450 := by
  sorry

end bus_trip_distance_l3310_331004


namespace imaginary_part_of_z_l3310_331022

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end imaginary_part_of_z_l3310_331022


namespace polynomial_independent_implies_m_plus_n_squared_l3310_331017

/-- A polynomial that is independent of x -/
def polynomial (m n x y : ℝ) : ℝ := 4*m*x^2 + 5*x - 2*y^2 + 8*x^2 - n*x + y - 1

/-- The polynomial is independent of x -/
def independent_of_x (m n : ℝ) : Prop :=
  ∀ x y : ℝ, ∃ c : ℝ, ∀ x' : ℝ, polynomial m n x' y = c

/-- The main theorem -/
theorem polynomial_independent_implies_m_plus_n_squared (m n : ℝ) :
  independent_of_x m n → (m + n)^2 = 9 := by
  sorry

end polynomial_independent_implies_m_plus_n_squared_l3310_331017


namespace sum_of_even_and_multiples_of_seven_l3310_331043

/-- The number of five-digit even numbers -/
def X : ℕ := 45000

/-- The number of five-digit multiples of 7 -/
def Y : ℕ := 12857

/-- The sum of five-digit even numbers and five-digit multiples of 7 -/
theorem sum_of_even_and_multiples_of_seven : X + Y = 57857 := by
  sorry

end sum_of_even_and_multiples_of_seven_l3310_331043


namespace polynomial_roots_equivalence_l3310_331000

theorem polynomial_roots_equivalence :
  let p (x : ℝ) := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7
  let y (x : ℝ) := x + 2 / x
  let q (y : ℝ) := 7 * y^2 - 48 * y + 47
  ∀ x : ℝ, x ≠ 0 →
    (p x = 0 ↔ ∃ y : ℝ, q y = 0 ∧ (x + 2 / x = y ∨ x + 2 / x = y)) :=
by sorry

end polynomial_roots_equivalence_l3310_331000


namespace bird_count_correct_bird_count_l3310_331099

theorem bird_count (total_heads : ℕ) (total_legs : ℕ) : ℕ :=
  let birds : ℕ := total_heads - (total_legs - 2 * total_heads) / 2
  birds

theorem correct_bird_count :
  bird_count 300 980 = 110 := by
  sorry

end bird_count_correct_bird_count_l3310_331099


namespace basketball_weight_proof_l3310_331085

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 20.83333

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 37.5

/-- The weight of one skateboard in pounds -/
def skateboard_weight : ℝ := 15

theorem basketball_weight_proof :
  (9 * basketball_weight = 5 * bicycle_weight) ∧
  (2 * bicycle_weight + 3 * skateboard_weight = 120) ∧
  (skateboard_weight = 15) :=
by sorry

end basketball_weight_proof_l3310_331085


namespace third_year_sample_size_l3310_331025

/-- The number of third-year students to be sampled in a stratified sampling scenario -/
theorem third_year_sample_size 
  (total_students : ℕ) 
  (first_year_students : ℕ) 
  (sophomore_probability : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : first_year_students = 760)
  (h3 : sophomore_probability = 37/100)
  (h4 : sample_size = 20) :
  let sophomore_students : ℕ := (sophomore_probability * total_students).num.toNat
  let third_year_students : ℕ := total_students - first_year_students - sophomore_students
  (sample_size * third_year_students) / total_students = 5 :=
by sorry

end third_year_sample_size_l3310_331025


namespace probability_at_least_one_head_l3310_331070

theorem probability_at_least_one_head (p : ℝ) : 
  p = 1 - (1/2)^4 → p = 15/16 := by
sorry

end probability_at_least_one_head_l3310_331070


namespace multiple_of_ten_implies_multiple_of_five_l3310_331024

theorem multiple_of_ten_implies_multiple_of_five 
  (h1 : ∀ n : ℕ, 10 ∣ n → 5 ∣ n) 
  (a : ℕ) 
  (h2 : 10 ∣ a) : 
  5 ∣ a := by
  sorry

end multiple_of_ten_implies_multiple_of_five_l3310_331024


namespace triangle_area_expression_range_l3310_331066

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def arithmeticSequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B = t.A + d ∧ t.C = t.B + d

def triangleConditions (t : Triangle) : Prop :=
  arithmeticSequence t ∧ t.b = 7 ∧ t.a + t.c = 13

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : triangleConditions t) :
  (1/2) * t.a * t.c * Real.sin t.B = 10 * Real.sqrt 3 := by sorry

-- Theorem for the range of the expression
theorem expression_range (t : Triangle) (h : triangleConditions t) :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ 
  x = Real.sqrt 3 * Real.sin t.A + Real.sin (t.C - π/6) := by sorry

end triangle_area_expression_range_l3310_331066


namespace triangle_problem_l3310_331091

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.C = 2 * t.b - t.c)
  (h2 : t.a = Real.sqrt 21)
  (h3 : t.b = 4) :
  t.A = π / 3 ∧ t.c = 5 := by
  sorry


end triangle_problem_l3310_331091


namespace log_equation_solution_l3310_331064

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq1 : q ≠ 1) :
  Real.log p + Real.log (q^2) = Real.log (p + q^2) ↔ p = q^2 / (q^2 - 1) :=
by sorry

end log_equation_solution_l3310_331064


namespace de_morgan_and_jenkins_birth_years_l3310_331054

def birth_year_de_morgan (x : ℕ) : Prop :=
  x^2 - x = 1806

def birth_year_jenkins (a b m n : ℕ) : Prop :=
  (a^4 + b^4) - (a^2 + b^2) = 1860 ∧
  2 * m^2 - 2 * m = 1860 ∧
  3 * n^4 - 3 * n = 1860

theorem de_morgan_and_jenkins_birth_years :
  ∃ (x a b m n : ℕ),
    birth_year_de_morgan x ∧
    birth_year_jenkins a b m n :=
sorry

end de_morgan_and_jenkins_birth_years_l3310_331054


namespace complex_number_quadrant_l3310_331012

theorem complex_number_quadrant : ∀ z : ℂ, 
  (3 - Complex.I) * z = 1 - 2 * Complex.I →
  0 < z.re ∧ z.im < 0 :=
by
  sorry

end complex_number_quadrant_l3310_331012


namespace f_properties_l3310_331052

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

theorem f_properties :
  (∀ x > 0, f (-1) x ≥ 1/2) ∧ 
  (f (-1) 1 = 1/2) ∧
  (∀ x ≥ 1, f 1 x < (2/3) * x^3) := by
  sorry

end f_properties_l3310_331052


namespace total_toys_is_15_8_l3310_331006

-- Define the initial number of toys and daily changes
def initial_toys : ℝ := 5.3
def tuesday_remaining_percent : ℝ := 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_loss_percent : ℝ := 0.502
def wednesday_new_toys : ℝ := 2.4
def thursday_loss_percent : ℝ := 0.308
def thursday_new_toys : ℝ := 4.5

-- Define the function to calculate the total number of toys
def total_toys : ℝ :=
  let tuesday_toys := initial_toys * tuesday_remaining_percent + tuesday_new_toys
  let wednesday_toys := tuesday_toys * (1 - wednesday_loss_percent) + wednesday_new_toys
  let thursday_toys := wednesday_toys * (1 - thursday_loss_percent) + thursday_new_toys
  let lost_tuesday := initial_toys - initial_toys * tuesday_remaining_percent
  let lost_wednesday := tuesday_toys - tuesday_toys * (1 - wednesday_loss_percent)
  let lost_thursday := wednesday_toys - wednesday_toys * (1 - thursday_loss_percent)
  thursday_toys + lost_tuesday + lost_wednesday + lost_thursday

-- Theorem statement
theorem total_toys_is_15_8 : total_toys = 15.8 := by
  sorry

end total_toys_is_15_8_l3310_331006


namespace function_is_even_l3310_331005

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_is_even (f : ℝ → ℝ) 
  (h : ∀ x y, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)) : 
  IsEven f := by
  sorry

end function_is_even_l3310_331005


namespace range_of_m_l3310_331067

theorem range_of_m (m : ℝ) : 
  (∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ Real.sqrt 3 * Real.sin α + Real.cos α = m) → 
  1 < m ∧ m ≤ 2 := by
  sorry

end range_of_m_l3310_331067


namespace complement_A_in_U_equals_open_interval_l3310_331019

-- Define the set U
def U : Set ℝ := {x | (x - 2) / x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 2 - x ≤ 1}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_in_U_equals_open_interval :
  complement_A_in_U = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end complement_A_in_U_equals_open_interval_l3310_331019


namespace curtis_family_children_l3310_331074

/-- Represents the Curtis family -/
structure CurtisFamily where
  mother_age : ℕ
  father_age : ℕ
  num_children : ℕ
  children_ages : Fin num_children → ℕ

/-- The average age of the family -/
def family_average_age (f : CurtisFamily) : ℚ :=
  (f.mother_age + f.father_age + (Finset.sum Finset.univ f.children_ages)) / (2 + f.num_children)

/-- The average age of the mother and children -/
def mother_children_average_age (f : CurtisFamily) : ℚ :=
  (f.mother_age + (Finset.sum Finset.univ f.children_ages)) / (1 + f.num_children)

/-- The theorem stating the number of children in the Curtis family -/
theorem curtis_family_children (f : CurtisFamily) 
  (h1 : family_average_age f = 25)
  (h2 : f.father_age = 50)
  (h3 : mother_children_average_age f = 20) : 
  f.num_children = 4 := by
  sorry


end curtis_family_children_l3310_331074


namespace unique_function_theorem_l3310_331028

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The property that a function satisfies the given conditions -/
def SatisfiesConditions (f : RationalFunction) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

/-- The theorem statement -/
theorem unique_function_theorem :
  ∀ f : RationalFunction, SatisfiesConditions f → ∀ x : ℚ, f x = x + 1 := by
  sorry

end unique_function_theorem_l3310_331028


namespace clown_balloons_l3310_331087

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown initially has -/
def initial_dozens : ℕ := 3

/-- The number of boys who buy a balloon -/
def boys : ℕ := 3

/-- The number of girls who buy a balloon -/
def girls : ℕ := 12

/-- The number of balloons the clown is left with after selling to boys and girls -/
def remaining_balloons : ℕ := initial_dozens * dozen - (boys + girls)

theorem clown_balloons : remaining_balloons = 21 := by
  sorry

end clown_balloons_l3310_331087


namespace lottery_expected_correct_guesses_l3310_331063

/-- Represents the number of matches in the lottery -/
def num_matches : ℕ := 12

/-- Represents the number of possible outcomes for each match -/
def num_outcomes : ℕ := 3

/-- Probability of guessing correctly for a single match -/
def p_correct : ℚ := 1 / num_outcomes

/-- Probability of guessing incorrectly for a single match -/
def p_incorrect : ℚ := 1 - p_correct

/-- Expected number of correct guesses in the lottery -/
def expected_correct_guesses : ℚ := num_matches * p_correct

theorem lottery_expected_correct_guesses :
  expected_correct_guesses = 4 := by sorry

end lottery_expected_correct_guesses_l3310_331063


namespace world_record_rates_l3310_331078

-- Define the world records
def hotdog_record : ℕ := 75
def hotdog_time : ℕ := 10
def hamburger_record : ℕ := 97
def hamburger_time : ℕ := 3
def cheesecake_record : ℚ := 11
def cheesecake_time : ℕ := 9

-- Define Lisa's progress
def lisa_hotdogs : ℕ := 20
def lisa_hotdog_time : ℕ := 5
def lisa_hamburgers : ℕ := 60
def lisa_hamburger_time : ℕ := 2
def lisa_cheesecake : ℚ := 5
def lisa_cheesecake_time : ℕ := 5

-- Define the theorem
theorem world_record_rates : 
  (((hotdog_record - lisa_hotdogs : ℚ) / (hotdog_time - lisa_hotdog_time)) = 11) ∧
  (((hamburger_record - lisa_hamburgers : ℚ) / (hamburger_time - lisa_hamburger_time)) = 37) ∧
  (((cheesecake_record - lisa_cheesecake) / (cheesecake_time - lisa_cheesecake_time)) = 3/2) :=
by sorry

end world_record_rates_l3310_331078


namespace max_safe_destroyers_l3310_331061

/-- Represents the configuration of ships and torpedo boats --/
structure NavalSetup where
  total_ships : Nat
  destroyers : Nat
  small_boats : Nat
  torpedo_boats : Nat
  torpedoes_per_boat : Nat

/-- Represents the targeting capabilities of torpedo boats --/
inductive TargetingStrategy
  | Successive : TargetingStrategy  -- Can target 10 successive ships
  | NextByOne : TargetingStrategy   -- Can target 10 ships next by one

/-- Defines a valid naval setup based on the problem conditions --/
def valid_setup (s : NavalSetup) : Prop :=
  s.total_ships = 30 ∧
  s.destroyers = 10 ∧
  s.small_boats = 20 ∧
  s.torpedo_boats = 2 ∧
  s.torpedoes_per_boat = 10

/-- Defines the maximum number of destroyers that can be targeted --/
def max_targeted_destroyers (s : NavalSetup) : Nat :=
  7  -- Based on the solution analysis

/-- The main theorem to be proved --/
theorem max_safe_destroyers (s : NavalSetup) 
  (h_valid : valid_setup s) :
  ∃ (safe_destroyers : Nat),
    safe_destroyers = s.destroyers - max_targeted_destroyers s ∧
    safe_destroyers = 3 :=
  sorry


end max_safe_destroyers_l3310_331061


namespace cubic_root_difference_l3310_331096

theorem cubic_root_difference : ∃ (r₁ r₂ r₃ : ℝ),
  (∀ x : ℝ, x^3 - 7*x^2 + 11*x - 6 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧
  max r₁ (max r₂ r₃) - min r₁ (min r₂ r₃) = 2 :=
sorry

end cubic_root_difference_l3310_331096


namespace james_purchase_cost_l3310_331021

def shirts_count : ℕ := 10
def shirt_price : ℕ := 6
def pants_price : ℕ := 8

def pants_count : ℕ := shirts_count / 2

def total_cost : ℕ := shirts_count * shirt_price + pants_count * pants_price

theorem james_purchase_cost : total_cost = 100 := by
  sorry

end james_purchase_cost_l3310_331021


namespace smallest_sum_4x4x4_cube_l3310_331035

/-- Represents a 4x4x4 cube made of dice -/
structure LargeCube where
  size : Nat
  dice_count : Nat
  opposite_sides_sum : Nat

/-- Calculates the smallest possible sum of visible faces on the large cube -/
def smallest_visible_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum for a 4x4x4 cube of dice -/
theorem smallest_sum_4x4x4_cube (cube : LargeCube) 
  (h1 : cube.size = 4)
  (h2 : cube.dice_count = 64)
  (h3 : cube.opposite_sides_sum = 7) :
  smallest_visible_sum cube = 144 := by
  sorry

end smallest_sum_4x4x4_cube_l3310_331035


namespace range_of_a_l3310_331040

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - a| < 4) → a ∈ Set.Ioo (-5) 3 := by
  sorry

end range_of_a_l3310_331040


namespace sequence_problem_l3310_331026

def S (n : ℕ) (k : ℕ) : ℚ := -1/2 * n^2 + k*n

theorem sequence_problem (k : ℕ) (h1 : k > 0) 
  (h2 : ∀ n : ℕ, S n k ≤ 8) 
  (h3 : ∃ n : ℕ, S n k = 8) :
  k = 4 ∧ ∀ n : ℕ, n ≥ 1 → ((-1/2 : ℚ) * n^2 + 4*n) - ((-1/2 : ℚ) * (n-1)^2 + 4*(n-1)) = 9/2 - n :=
sorry

end sequence_problem_l3310_331026


namespace minimum_packaging_volume_l3310_331010

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the packaging problem parameters -/
structure PackagingProblem where
  boxDimensions : BoxDimensions
  costPerBox : ℝ
  minTotalCost : ℝ

theorem minimum_packaging_volume (p : PackagingProblem) 
  (h1 : p.boxDimensions.length = 20)
  (h2 : p.boxDimensions.width = 20)
  (h3 : p.boxDimensions.height = 12)
  (h4 : p.costPerBox = 0.4)
  (h5 : p.minTotalCost = 200) :
  (p.minTotalCost / p.costPerBox) * boxVolume p.boxDimensions = 2400000 := by
  sorry

#check minimum_packaging_volume

end minimum_packaging_volume_l3310_331010


namespace final_hair_length_l3310_331093

/-- Given initial hair length x, amount cut off y, and growth z,
    prove that the final hair length F is 17 inches. -/
theorem final_hair_length
  (x y z : ℝ)
  (hx : x = 16)
  (hy : y = 11)
  (hz : z = 12)
  (hF : F = (x - y) + z) :
  F = 17 :=
by sorry

end final_hair_length_l3310_331093


namespace dots_not_visible_is_81_l3310_331047

/-- The number of faces on each die -/
def faces_per_die : ℕ := 6

/-- The number of dice -/
def num_dice : ℕ := 5

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

/-- The total number of dots on all dice -/
def total_dots : ℕ := num_dice * (faces_per_die * (faces_per_die + 1) / 2)

/-- The sum of visible numbers -/
def sum_visible : ℕ := visible_numbers.sum

/-- Theorem: The number of dots not visible is 81 -/
theorem dots_not_visible_is_81 : total_dots - sum_visible = 81 := by
  sorry

end dots_not_visible_is_81_l3310_331047


namespace fraction_equality_implies_value_l3310_331027

theorem fraction_equality_implies_value (a : ℝ) : 
  a / (a + 45) = 0.82 → a = 205 := by sorry

end fraction_equality_implies_value_l3310_331027


namespace square_root_of_four_l3310_331011

theorem square_root_of_four (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end square_root_of_four_l3310_331011


namespace a_plus_2b_equals_one_l3310_331095

theorem a_plus_2b_equals_one (a b : ℝ) 
  (ha : a^3 - 21*a^2 + 140*a - 120 = 0)
  (hb : 4*b^3 - 12*b^2 - 32*b + 448 = 0) :
  a + 2*b = 1 := by
  sorry

end a_plus_2b_equals_one_l3310_331095


namespace line_through_two_points_l3310_331056

-- Define a point in 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line equation
def lineEquation (p1 p2 : Point2D) (x y : ℝ) : Prop :=
  (y - p1.y) * (p2.x - p1.x) = (x - p1.x) * (p2.y - p1.y)

-- Theorem statement
theorem line_through_two_points (p1 p2 : Point2D) :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | lineEquation p1 p2 x y} ↔ 
  ∃ t : ℝ, x = p1.x + t * (p2.x - p1.x) ∧ y = p1.y + t * (p2.y - p1.y) :=
by sorry

end line_through_two_points_l3310_331056


namespace fraction_to_decimal_l3310_331092

theorem fraction_to_decimal : (45 : ℚ) / 72 = 0.625 := by
  sorry

end fraction_to_decimal_l3310_331092


namespace binomial_10_choose_3_l3310_331034

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l3310_331034


namespace bedroom_set_price_l3310_331002

def original_price : ℝ := 2000
def gift_card : ℝ := 200
def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10

def final_price : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  price_after_second_discount - gift_card

theorem bedroom_set_price : final_price = 1330 := by
  sorry

end bedroom_set_price_l3310_331002


namespace number_problem_l3310_331086

theorem number_problem : ∃ x : ℝ, x = 25 ∧ (2/5) * x + 22 = (80/100) * 40 := by
  sorry

end number_problem_l3310_331086


namespace five_sixths_of_thirty_l3310_331065

theorem five_sixths_of_thirty : (5 / 6 : ℚ) * 30 = 25 := by
  sorry

end five_sixths_of_thirty_l3310_331065


namespace plant_is_red_daisy_l3310_331068

structure Plant where
  color : String
  type : String

structure Statement where
  person : String
  plant : Plant

def is_partially_correct (actual : Plant) (statement : Statement) : Prop :=
  (actual.color = statement.plant.color) ≠ (actual.type = statement.plant.type)

theorem plant_is_red_daisy (actual : Plant) 
  (anika_statement : Statement)
  (bill_statement : Statement)
  (cathy_statement : Statement)
  (h1 : anika_statement.person = "Anika" ∧ anika_statement.plant = ⟨"red", "rose"⟩)
  (h2 : bill_statement.person = "Bill" ∧ bill_statement.plant = ⟨"purple", "daisy"⟩)
  (h3 : cathy_statement.person = "Cathy" ∧ cathy_statement.plant = ⟨"red", "dahlia"⟩)
  (h4 : is_partially_correct actual anika_statement)
  (h5 : is_partially_correct actual bill_statement)
  (h6 : is_partially_correct actual cathy_statement)
  : actual = ⟨"red", "daisy"⟩ := by
  sorry

end plant_is_red_daisy_l3310_331068


namespace range_of_a_l3310_331081

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0 → False) ∧ 
  (∀ x : ℝ, x^2 + a*x + 1 > 0) → False ∧
  (∀ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0 → False) ∨ 
  (∀ x : ℝ, x^2 + a*x + 1 > 0) → 
  a ≤ -2 ∨ a = 2 := by sorry

end range_of_a_l3310_331081


namespace sqrt_product_quotient_l3310_331069

theorem sqrt_product_quotient : Real.sqrt 3 * Real.sqrt 10 / Real.sqrt 6 = Real.sqrt 5 := by
  sorry

end sqrt_product_quotient_l3310_331069


namespace binomial_eight_zero_l3310_331042

theorem binomial_eight_zero : Nat.choose 8 0 = 1 := by
  sorry

end binomial_eight_zero_l3310_331042


namespace min_q_geq_half_l3310_331003

def q (a : ℕ) : ℚ := ((48 - a) * (47 - a) + (a - 1) * (a - 2)) / (2 * 1653)

theorem min_q_geq_half (n : ℕ) (h : n ≥ 1 ∧ n ≤ 60) :
  (∀ a : ℕ, a ≥ 1 ∧ a ≤ 60 → q a ≥ 1/2 → a ≥ n) →
  q n ≥ 1/2 →
  n = 10 :=
sorry

end min_q_geq_half_l3310_331003


namespace point_with_distance_6_l3310_331080

def distance_from_origin (x : ℝ) : ℝ := |x|

theorem point_with_distance_6 (A : ℝ) :
  distance_from_origin A = 6 ↔ A = 6 ∨ A = -6 := by
  sorry

end point_with_distance_6_l3310_331080


namespace point_movement_l3310_331015

/-- Given point A(-1, 3), moving it 5 units down and 2 units to the left results in point B(-3, -2) -/
theorem point_movement (A B : ℝ × ℝ) : 
  A = (-1, 3) → 
  B.1 = A.1 - 2 → 
  B.2 = A.2 - 5 → 
  B = (-3, -2) := by
sorry

end point_movement_l3310_331015


namespace applicant_age_standard_deviation_l3310_331009

theorem applicant_age_standard_deviation
  (average_age : ℝ)
  (max_different_ages : ℕ)
  (h_average : average_age = 31)
  (h_max_ages : max_different_ages = 11) :
  let standard_deviation := (max_different_ages - 1) / 2
  standard_deviation = 5 := by
  sorry

end applicant_age_standard_deviation_l3310_331009


namespace always_real_roots_roots_difference_condition_l3310_331082

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * x^2 - (3*m - 1) * x + (2*m - 2)

-- Theorem 1: The equation always has real roots
theorem always_real_roots (m : ℝ) : 
  ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Theorem 2: If the difference between the roots is 2, then m = 1 or m = -1/3
theorem roots_difference_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation m x₁ = 0 ∧ 
    quadratic_equation m x₂ = 0 ∧ 
    |x₁ - x₂| = 2) →
  (m = 1 ∨ m = -1/3) :=
sorry

end always_real_roots_roots_difference_condition_l3310_331082


namespace square_perimeter_from_p_shape_l3310_331072

/-- Represents the width of each rectangle --/
def rectangle_width : ℝ := 4

/-- Represents the length of each rectangle --/
def rectangle_length : ℝ := 4 * rectangle_width

/-- Represents the side length of the original square --/
def square_side : ℝ := rectangle_width + rectangle_length

/-- Represents the perimeter of the "P" shape --/
def p_perimeter : ℝ := 56

theorem square_perimeter_from_p_shape :
  p_perimeter = 2 * (square_side) + rectangle_length →
  4 * square_side = 80 :=
by sorry

end square_perimeter_from_p_shape_l3310_331072


namespace box_volume_increase_l3310_331045

/-- Theorem about the volume of a rectangular box after increasing its dimensions --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + l * h + w * h) = 1950)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7198 := by sorry

end box_volume_increase_l3310_331045


namespace max_regions_theorem_l3310_331050

/-- Represents a circular disk with chords and secant lines. -/
structure DiskWithChords where
  n : ℕ
  chord_count : ℕ := 2 * n + 1
  secant_count : ℕ := 2

/-- Calculates the maximum number of non-overlapping regions in the disk. -/
def max_regions (disk : DiskWithChords) : ℕ :=
  8 * disk.n + 8

/-- Theorem stating the maximum number of non-overlapping regions. -/
theorem max_regions_theorem (disk : DiskWithChords) (h : disk.n > 0) :
  max_regions disk = 8 * disk.n + 8 := by
  sorry

end max_regions_theorem_l3310_331050


namespace pyramid_volume_change_l3310_331073

/-- Given a pyramid with rectangular base and volume 60 cubic feet, 
    prove that tripling its length, doubling its width, and increasing its height by 20% 
    results in a new volume of 432 cubic feet. -/
theorem pyramid_volume_change (V : ℝ) (l w h : ℝ) : 
  V = 60 → 
  V = (1/3) * l * w * h → 
  (1/3) * (3*l) * (2*w) * (1.2*h) = 432 :=
by sorry

end pyramid_volume_change_l3310_331073


namespace minimum_students_in_class_l3310_331016

theorem minimum_students_in_class (boys girls : ℕ) : 
  boys > 0 → girls > 0 →
  2 * (boys / 2) = 3 * (girls / 3) →
  boys + girls ≥ 7 :=
by
  sorry

#check minimum_students_in_class

end minimum_students_in_class_l3310_331016


namespace marble_problem_l3310_331089

theorem marble_problem (r b : ℕ) : 
  (r > 0) →
  (b > 0) →
  ((r - 1 : ℚ) / (r + b - 1) = 1 / 7) →
  (r / (r + b - 2 : ℚ) = 1 / 5) →
  (r + b = 22) :=
by sorry

end marble_problem_l3310_331089


namespace millionthDigitOf1Over41_l3310_331030

-- Define the fraction
def fraction : ℚ := 1 / 41

-- Define the function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- State the theorem
theorem millionthDigitOf1Over41 : 
  nthDigitAfterDecimal fraction 1000000 = 9 := by sorry

end millionthDigitOf1Over41_l3310_331030


namespace child_running_speed_l3310_331029

/-- Verify the child's running speed on a still sidewalk -/
theorem child_running_speed 
  (distance_with : ℝ)
  (distance_against : ℝ)
  (time_against : ℝ)
  (speed_still : ℝ)
  (h1 : distance_with = 372)
  (h2 : distance_against = 165)
  (h3 : time_against = 3)
  (h4 : speed_still = 74)
  (h5 : ∃ t, t > 0 ∧ (speed_still + (distance_against / time_against - speed_still)) * t = distance_with)
  (h6 : (speed_still - (distance_against / time_against - speed_still)) * time_against = distance_against) :
  speed_still = 74 := by
sorry

end child_running_speed_l3310_331029


namespace peters_glass_purchase_l3310_331039

/-- Peter's glass purchase problem -/
theorem peters_glass_purchase
  (small_price : ℕ)
  (large_price : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (large_count : ℕ)
  (h1 : small_price = 3)
  (h2 : large_price = 5)
  (h3 : total_money = 50)
  (h4 : change = 1)
  (h5 : large_count = 5)
  : (total_money - change - large_count * large_price) / small_price = 8 := by
  sorry

end peters_glass_purchase_l3310_331039


namespace volume_ratio_cube_sphere_l3310_331090

/-- The ratio of the volume of a cube to the volume of a sphere -/
theorem volume_ratio_cube_sphere (cube_edge : Real) (other_cube_edge : Real) : 
  cube_edge = 4 → other_cube_edge = 3 →
  (cube_edge ^ 3) / ((4/3) * π * (other_cube_edge ^ 3)) = 16 / (9 * π) := by
  sorry

end volume_ratio_cube_sphere_l3310_331090


namespace average_temperature_l3310_331079

def temperature_data : List ℝ := [90, 90, 90, 79, 71]
def num_years : ℕ := 5

theorem average_temperature : 
  (List.sum temperature_data) / num_years = 84 := by
  sorry

end average_temperature_l3310_331079


namespace average_cost_is_1_85_l3310_331013

/-- Calculates the average cost per fruit given the prices and quantities of fruits, applying special offers --/
def average_cost_per_fruit (apple_price banana_price orange_price : ℚ) 
  (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let apple_cost := apple_price * (apple_qty.div 10 * 10)
  let banana_cost := banana_price * banana_qty
  let orange_cost := orange_price * (orange_qty.div 3 * 3)
  let total_cost := apple_cost + banana_cost + orange_cost
  let total_fruits := apple_qty + banana_qty + orange_qty
  total_cost / total_fruits

/-- The average cost per fruit is $1.85 given the specified prices, quantities, and offers --/
theorem average_cost_is_1_85 :
  average_cost_per_fruit 2 1 3 12 4 4 = 37/20 := by
  sorry

end average_cost_is_1_85_l3310_331013


namespace sum_of_twos_and_threes_1800_l3310_331037

/-- The number of ways to represent a positive integer as a sum of 2s and 3s -/
def waysToSum (n : ℕ) : ℕ :=
  (n / 6 + 1)

/-- 1800 can be represented as a sum of 2s and 3s in 301 ways -/
theorem sum_of_twos_and_threes_1800 : waysToSum 1800 = 301 := by
  sorry

end sum_of_twos_and_threes_1800_l3310_331037


namespace students_with_one_problem_l3310_331076

/-- Represents the number of problems created by students from each course -/
def ProblemsCourses : Type := Fin 5 → ℕ

/-- Represents the number of students in each course -/
def StudentsCourses : Type := Fin 5 → ℕ

/-- The total number of students -/
def TotalStudents : ℕ := 30

/-- The total number of problems created -/
def TotalProblems : ℕ := 40

/-- The condition that students from different courses created different numbers of problems -/
def DifferentProblems (p : ProblemsCourses) : Prop :=
  ∀ i j, i ≠ j → p i ≠ p j

/-- The condition that the total number of problems created matches the given total -/
def MatchesTotalProblems (p : ProblemsCourses) (s : StudentsCourses) : Prop :=
  (Finset.sum Finset.univ (λ i => p i * s i)) = TotalProblems

/-- The condition that the total number of students matches the given total -/
def MatchesTotalStudents (s : StudentsCourses) : Prop :=
  (Finset.sum Finset.univ s) = TotalStudents

theorem students_with_one_problem
  (p : ProblemsCourses)
  (s : StudentsCourses)
  (h1 : DifferentProblems p)
  (h2 : MatchesTotalProblems p s)
  (h3 : MatchesTotalStudents s) :
  (Finset.filter (λ i => p i = 1) Finset.univ).card = 26 := by
  sorry

end students_with_one_problem_l3310_331076


namespace find_S_value_l3310_331051

-- Define the relationship between R, S, and T
def relationship (R S T : ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ R S T, R = c * S^2 / T

-- Define the initial condition
def initial_condition (R S T : ℝ) : Prop :=
  R = 2 ∧ S = 1 ∧ T = 3

-- Theorem to prove
theorem find_S_value (R S T : ℝ) :
  relationship R S T →
  initial_condition R S T →
  R = 18 ∧ T = 2 →
  S = Real.sqrt 6 := by
  sorry

end find_S_value_l3310_331051


namespace ancient_chinese_rope_problem_l3310_331098

theorem ancient_chinese_rope_problem (x y : ℝ) :
  (1/2 : ℝ) * x - y = 5 ∧ y - (1/3 : ℝ) * x = 2 → x = 42 ∧ y = 16 := by
  sorry

end ancient_chinese_rope_problem_l3310_331098


namespace warrens_event_capacity_l3310_331036

theorem warrens_event_capacity :
  let total_tables : ℕ := 252
  let large_tables : ℕ := 93
  let medium_tables : ℕ := 97
  let small_tables : ℕ := total_tables - large_tables - medium_tables
  let unusable_small_tables : ℕ := 20
  let usable_small_tables : ℕ := small_tables - unusable_small_tables
  let large_table_capacity : ℕ := 6
  let medium_table_capacity : ℕ := 5
  let small_table_capacity : ℕ := 4
  
  large_tables * large_table_capacity +
  medium_tables * medium_table_capacity +
  usable_small_tables * small_table_capacity = 1211 :=
by
  sorry

#eval
  let total_tables : ℕ := 252
  let large_tables : ℕ := 93
  let medium_tables : ℕ := 97
  let small_tables : ℕ := total_tables - large_tables - medium_tables
  let unusable_small_tables : ℕ := 20
  let usable_small_tables : ℕ := small_tables - unusable_small_tables
  let large_table_capacity : ℕ := 6
  let medium_table_capacity : ℕ := 5
  let small_table_capacity : ℕ := 4
  
  large_tables * large_table_capacity +
  medium_tables * medium_table_capacity +
  usable_small_tables * small_table_capacity

end warrens_event_capacity_l3310_331036


namespace cube_collinear_triples_l3310_331058

/-- Represents a point in a cube -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | CubeCenter

/-- Represents a set of three collinear points in a cube -/
structure CollinearTriple where
  p1 : CubePoint
  p2 : CubePoint
  p3 : CubePoint

/-- The total number of points in the cube -/
def totalPoints : Nat := 27

/-- The number of vertices in the cube -/
def numVertices : Nat := 8

/-- The number of edge midpoints in the cube -/
def numEdgeMidpoints : Nat := 12

/-- The number of face centers in the cube -/
def numFaceCenters : Nat := 6

/-- The number of cube centers (always 1) -/
def numCubeCenters : Nat := 1

/-- Function to count the number of collinear triples in the cube -/
def countCollinearTriples : List CollinearTriple → Nat :=
  List.length

/-- Theorem: The number of sets of three collinear points in the cube is 49 -/
theorem cube_collinear_triples :
  ∃ (triples : List CollinearTriple),
    countCollinearTriples triples = 49 ∧
    totalPoints = numVertices + numEdgeMidpoints + numFaceCenters + numCubeCenters :=
  sorry

end cube_collinear_triples_l3310_331058


namespace sum_of_m_values_l3310_331097

/-- A triangle with vertices at (0,0), (2,2), and (8m,0) is divided into two equal areas by a line y = mx. -/
def Triangle (m : ℝ) := {A : ℝ × ℝ | A = (0, 0) ∨ A = (2, 2) ∨ A = (8*m, 0)}

/-- The line that divides the triangle into two equal areas -/
def DividingLine (m : ℝ) := {(x, y) : ℝ × ℝ | y = m * x}

/-- The condition that the line divides the triangle into two equal areas -/
def EqualAreasCondition (m : ℝ) : Prop := 
  ∃ (x : ℝ), (x, m*x) ∈ DividingLine m ∧ 
  (x = 4*m + 1) ∧ (m*x = 1)

/-- The theorem stating that the sum of all possible values of m is -1/4 -/
theorem sum_of_m_values (m₁ m₂ : ℝ) : 
  (EqualAreasCondition m₁ ∧ EqualAreasCondition m₂ ∧ m₁ ≠ m₂) → 
  m₁ + m₂ = -1/4 := by sorry

end sum_of_m_values_l3310_331097


namespace only_drunk_drivers_traffic_accidents_correlated_l3310_331046

-- Define the types of quantities
inductive Quantity
  | Time
  | Displacement
  | StudentGrade
  | Weight
  | DrunkDrivers
  | TrafficAccidents
  | Volume

-- Define the relationship between quantities
inductive Relationship
  | NoRelation
  | Correlation
  | FunctionalRelation

-- Define a function to determine the relationship between two quantities
def relationshipBetween (q1 q2 : Quantity) : Relationship :=
  match q1, q2 with
  | Quantity.Time, Quantity.Displacement => Relationship.FunctionalRelation
  | Quantity.StudentGrade, Quantity.Weight => Relationship.NoRelation
  | Quantity.DrunkDrivers, Quantity.TrafficAccidents => Relationship.Correlation
  | Quantity.Volume, Quantity.Weight => Relationship.FunctionalRelation
  | _, _ => Relationship.NoRelation

-- Theorem to prove
theorem only_drunk_drivers_traffic_accidents_correlated :
  ∀ q1 q2 : Quantity,
    relationshipBetween q1 q2 = Relationship.Correlation →
    (q1 = Quantity.DrunkDrivers ∧ q2 = Quantity.TrafficAccidents) ∨
    (q1 = Quantity.TrafficAccidents ∧ q2 = Quantity.DrunkDrivers) :=
by
  sorry


end only_drunk_drivers_traffic_accidents_correlated_l3310_331046


namespace percentage_less_than_y_l3310_331060

theorem percentage_less_than_y (y q w z : ℝ) 
  (hw : w = 0.6 * q) 
  (hq : q = 0.6 * y) 
  (hz : z = 1.5 * w) : 
  z = 0.54 * y := by sorry

end percentage_less_than_y_l3310_331060


namespace solution_set_implies_m_value_l3310_331088

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + a

-- Define the property of m and 1 being roots of the equation f a x = 0
def are_roots (a m : ℝ) : Prop := f a m = 0 ∧ f a 1 = 0

-- Define the property of (m, 1) being the solution set of the inequality
def is_solution_set (a m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ m < x ∧ x < 1

-- State the theorem
theorem solution_set_implies_m_value (a m : ℝ) :
  is_solution_set a m → m = 1/2 := by
  sorry

end solution_set_implies_m_value_l3310_331088


namespace smallest_19_factor_number_is_78732_l3310_331020

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The smallest positive integer with exactly 19 factors -/
def smallest_19_factor_number : ℕ+ := sorry

/-- Theorem stating that the smallest positive integer with exactly 19 factors is 78732 -/
theorem smallest_19_factor_number_is_78732 : 
  smallest_19_factor_number = 78732 ∧ num_factors smallest_19_factor_number = 19 := by sorry

end smallest_19_factor_number_is_78732_l3310_331020


namespace raised_bed_width_l3310_331083

theorem raised_bed_width (num_beds : ℕ) (length height : ℝ) (num_bags : ℕ) (soil_per_bag : ℝ) :
  num_beds = 2 →
  length = 8 →
  height = 1 →
  num_bags = 16 →
  soil_per_bag = 4 →
  (num_bags : ℝ) * soil_per_bag / num_beds / (length * height) = 4 :=
by sorry

end raised_bed_width_l3310_331083


namespace smallest_multiple_of_seven_l3310_331031

theorem smallest_multiple_of_seven (x y : ℤ) 
  (h1 : (x + 1) % 7 = 0) 
  (h2 : (y - 5) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) → 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) ∧ 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) → n = 7 := by
  sorry

end smallest_multiple_of_seven_l3310_331031


namespace decreasing_function_a_range_l3310_331071

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then -4 * x + 2 * a else x^2 - a * x + 4

-- Define what it means for f to be decreasing on ℝ
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Main theorem statement
theorem decreasing_function_a_range :
  ∃ a_min a_max : ℝ, a_min = 2 ∧ a_max = 3 ∧
  (∀ a : ℝ, is_decreasing (f a) ↔ a_min ≤ a ∧ a ≤ a_max) :=
sorry

end decreasing_function_a_range_l3310_331071


namespace fan_shooting_theorem_l3310_331059

/-- Represents a fan with four blades rotating at a given speed -/
structure Fan :=
  (revolution_speed : ℝ)
  (num_blades : ℕ)

/-- Represents a bullet trajectory -/
structure BulletTrajectory :=
  (angle : ℝ)
  (speed : ℝ)

/-- Checks if a bullet trajectory intersects all blades of a fan -/
def intersects_all_blades (f : Fan) (bt : BulletTrajectory) : Prop :=
  sorry

/-- The main theorem stating that there exists a bullet trajectory that intersects all blades -/
theorem fan_shooting_theorem (f : Fan) 
  (h1 : f.revolution_speed = 50)
  (h2 : f.num_blades = 4) : 
  ∃ (bt : BulletTrajectory), intersects_all_blades f bt :=
sorry

end fan_shooting_theorem_l3310_331059


namespace factor_expression_l3310_331041

theorem factor_expression (y : ℝ) : 5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) := by
  sorry

end factor_expression_l3310_331041


namespace total_vacations_and_classes_l3310_331094

/-- The number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- The number of vacations Grant has -/
def grant_vacations : ℕ := 4 * kelvin_classes

/-- The total number of vacations and classes Grant and Kelvin have altogether -/
def total : ℕ := grant_vacations + kelvin_classes

theorem total_vacations_and_classes : total = 450 := by
  sorry

end total_vacations_and_classes_l3310_331094
