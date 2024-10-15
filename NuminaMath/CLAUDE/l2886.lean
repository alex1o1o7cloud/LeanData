import Mathlib

namespace NUMINAMATH_CALUDE_f_analytical_expression_k_range_for_monotonicity_l2886_288604

-- Part 1
def f₁ (x : ℝ) := x^2 - 3*x + 2

theorem f_analytical_expression :
  ∀ x, f₁ (x + 1) = x^2 - 3*x + 2 →
  ∃ g : ℝ → ℝ, (∀ x, g x = x^2 - 6*x + 6) ∧ (∀ x, g x = f₁ x) :=
sorry

-- Part 2
def f₂ (k : ℝ) (x : ℝ) := x^2 - 2*k*x - 8

theorem k_range_for_monotonicity :
  ∀ k, (∀ x ∈ Set.Icc 1 4, Monotone (f₂ k)) →
  k ≥ 4 ∨ k ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_analytical_expression_k_range_for_monotonicity_l2886_288604


namespace NUMINAMATH_CALUDE_prove_initial_person_count_l2886_288660

/-- The initial number of persons in a group where:
  - The average weight increase is 4.2 kg when a new person replaces one of the original group.
  - The weight of the person leaving is 65 kg.
  - The weight of the new person is 98.6 kg.
-/
def initialPersonCount : ℕ := 8

theorem prove_initial_person_count :
  let avgWeightIncrease : ℚ := 21/5
  let oldPersonWeight : ℚ := 65
  let newPersonWeight : ℚ := 493/5
  (newPersonWeight - oldPersonWeight) / avgWeightIncrease = initialPersonCount := by
  sorry

end NUMINAMATH_CALUDE_prove_initial_person_count_l2886_288660


namespace NUMINAMATH_CALUDE_equal_star_set_eq_four_lines_l2886_288688

-- Define the operation ⋆
def star (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def equal_star_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Define the four lines
def four_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 + p.2 = 0}

-- Theorem stating the equivalence of the two sets
theorem equal_star_set_eq_four_lines :
  equal_star_set = four_lines := by sorry

end NUMINAMATH_CALUDE_equal_star_set_eq_four_lines_l2886_288688


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l2886_288669

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℤ, x^3 + b*x + c = 0) →
  (Complex.exp (3 - Real.sqrt 3))^3 + b*(Complex.exp (3 - Real.sqrt 3)) + c = 0 →
  (∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l2886_288669


namespace NUMINAMATH_CALUDE_courtyard_length_l2886_288686

/-- The length of a rectangular courtyard given its width and paving details. -/
theorem courtyard_length (width : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℕ) : 
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 16000 →
  width * (total_bricks : ℝ) * brick_length * brick_width / width = 20 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l2886_288686


namespace NUMINAMATH_CALUDE_characters_with_initial_D_l2886_288651

-- Define the total number of characters
def total_characters : ℕ := 60

-- Define the number of characters with initial A
def characters_A : ℕ := total_characters / 2

-- Define the number of characters with initial C
def characters_C : ℕ := characters_A / 2

-- Define the remaining characters (D and E)
def remaining_characters : ℕ := total_characters - characters_A - characters_C

-- Theorem stating the number of characters with initial D
theorem characters_with_initial_D : 
  ∃ (d e : ℕ), d = 2 * e ∧ d + e = remaining_characters ∧ d = 10 :=
sorry

end NUMINAMATH_CALUDE_characters_with_initial_D_l2886_288651


namespace NUMINAMATH_CALUDE_al_sandwich_options_l2886_288647

-- Define the types of ingredients
structure Ingredients :=
  (bread : Nat)
  (meat : Nat)
  (cheese : Nat)

-- Define the restrictions
structure Restrictions :=
  (turkey_swiss : Nat)
  (rye_roast_beef : Nat)

-- Define the function to calculate the number of sandwiches
def calculate_sandwiches (i : Ingredients) (r : Restrictions) : Nat :=
  i.bread * i.meat * i.cheese - r.turkey_swiss - r.rye_roast_beef

-- Theorem statement
theorem al_sandwich_options (i : Ingredients) (r : Restrictions) 
  (h1 : i.bread = 5)
  (h2 : i.meat = 7)
  (h3 : i.cheese = 6)
  (h4 : r.turkey_swiss = 5)
  (h5 : r.rye_roast_beef = 6) :
  calculate_sandwiches i r = 199 := by
  sorry


end NUMINAMATH_CALUDE_al_sandwich_options_l2886_288647


namespace NUMINAMATH_CALUDE_asymptote_of_hyperbola_l2886_288693

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 16 = 1

/-- The equation of an asymptote -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (4/5) * x

/-- Theorem: The given equation is an asymptote of the hyperbola -/
theorem asymptote_of_hyperbola :
  ∀ x y : ℝ, asymptote_equation x y → (∃ ε > 0, ∀ δ > ε, 
    ∃ x' y' : ℝ, hyperbola_equation x' y' ∧ 
    ((x' - x)^2 + (y' - y)^2 < δ^2)) :=
sorry

end NUMINAMATH_CALUDE_asymptote_of_hyperbola_l2886_288693


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2886_288682

/-- The units digit of m^2 + 2^m is 7, where m = 2021^2 + 3^2021 -/
theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : m = 2021^2 + 3^2021 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2886_288682


namespace NUMINAMATH_CALUDE_root_of_equations_l2886_288683

theorem root_of_equations (a b c d e k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (eq1 : a * k^4 + b * k^3 + c * k^2 + d * k + e = 0)
  (eq2 : b * k^4 + c * k^3 + d * k^2 + e * k + a = 0) :
  k^5 = 1 :=
sorry

end NUMINAMATH_CALUDE_root_of_equations_l2886_288683


namespace NUMINAMATH_CALUDE_fraction_reducibility_fraction_reducibility_2_l2886_288679

theorem fraction_reducibility (n : ℤ) :
  (∃ k : ℤ, n = 3 * k - 1) ↔ 
    ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ b ≠ 1 ∧ b * (n^2 + 2*n + 4) = a * (n^2 + n + 3) :=
sorry

theorem fraction_reducibility_2 (n : ℤ) :
  (∃ k : ℤ, n = 3 * k ∨ n = 3 * k + 1) ↔ 
    ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ b ≠ 1 ∧ b * (n^3 - n^2 - 3*n) = a * (n^2 - n + 3) :=
sorry

end NUMINAMATH_CALUDE_fraction_reducibility_fraction_reducibility_2_l2886_288679


namespace NUMINAMATH_CALUDE_function_monotonically_increasing_l2886_288690

/-- The function f(x) = x^2 - 2x + 8 is monotonically increasing on the interval (1, +∞) -/
theorem function_monotonically_increasing (x y : ℝ) : x > 1 → y > 1 → x < y →
  (x^2 - 2*x + 8) < (y^2 - 2*y + 8) := by sorry

end NUMINAMATH_CALUDE_function_monotonically_increasing_l2886_288690


namespace NUMINAMATH_CALUDE_cubic_max_value_l2886_288696

/-- Given a cubic function with a known maximum value, prove the constant term --/
theorem cubic_max_value (m : ℝ) : 
  (∃ (x : ℝ), ∀ (t : ℝ), -t^3 + 3*t^2 + m ≤ -x^3 + 3*x^2 + m) ∧
  (∃ (x : ℝ), -x^3 + 3*x^2 + m = 10) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_max_value_l2886_288696


namespace NUMINAMATH_CALUDE_number_grid_solution_l2886_288605

theorem number_grid_solution : 
  ∃ (a b c d : ℕ) (s : Finset ℕ),
    s = {1, 2, 3, 4, 5, 6, 7, 8} ∧
    a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a * b = c ∧
    c / b = d ∧
    a = d :=
by sorry

end NUMINAMATH_CALUDE_number_grid_solution_l2886_288605


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_zero_one_l2886_288627

-- Define set M
def M : Set ℝ := {x | x^2 = x}

-- Define set N
def N : Set ℝ := {-1, 0, 1}

-- Theorem statement
theorem M_intersect_N_eq_zero_one : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_zero_one_l2886_288627


namespace NUMINAMATH_CALUDE_xyz_product_l2886_288646

theorem xyz_product (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x * (y + z) = 198)
  (h2 : y * (z + x) = 216)
  (h3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2886_288646


namespace NUMINAMATH_CALUDE_game_probability_difference_l2886_288675

def p_heads : ℚ := 3/4
def p_tails : ℚ := 1/4

def p_win_game_c : ℚ := p_heads^4 + p_tails^4

def p_win_game_d : ℚ := p_heads^4 * p_tails + p_tails^4 * p_heads

theorem game_probability_difference :
  p_win_game_c - p_win_game_d = 61/256 := by sorry

end NUMINAMATH_CALUDE_game_probability_difference_l2886_288675


namespace NUMINAMATH_CALUDE_course_assessment_probabilities_l2886_288662

/-- Represents a student in the course -/
inductive Student := | A | B | C

/-- Represents the type of assessment -/
inductive AssessmentType := | Theory | Experimental

/-- The probability of a student passing a specific assessment type -/
def passProbability (s : Student) (t : AssessmentType) : ℝ :=
  match s, t with
  | Student.A, AssessmentType.Theory => 0.9
  | Student.B, AssessmentType.Theory => 0.8
  | Student.C, AssessmentType.Theory => 0.7
  | Student.A, AssessmentType.Experimental => 0.8
  | Student.B, AssessmentType.Experimental => 0.7
  | Student.C, AssessmentType.Experimental => 0.9

/-- The probability of at least two students passing the theory assessment -/
def atLeastTwoPassTheory : ℝ := sorry

/-- The probability of all three students passing both assessments -/
def allPassBoth : ℝ := sorry

theorem course_assessment_probabilities :
  (atLeastTwoPassTheory = 0.902) ∧ (allPassBoth = 0.254) := by sorry

end NUMINAMATH_CALUDE_course_assessment_probabilities_l2886_288662


namespace NUMINAMATH_CALUDE_division_equality_l2886_288659

theorem division_equality : (124 : ℚ) / (8 + 14 * 3) = 62 / 25 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l2886_288659


namespace NUMINAMATH_CALUDE_refill_count_is_three_l2886_288608

/-- Calculates the number of daily water bottle refills given the parameters. -/
def daily_refills (bottle_capacity : ℕ) (days : ℕ) (spill1 : ℕ) (spill2 : ℕ) (total_drunk : ℕ) : ℕ :=
  ((total_drunk + spill1 + spill2) / (bottle_capacity * days) : ℕ)

/-- Proves that given the specified parameters, the number of daily refills is 3. -/
theorem refill_count_is_three :
  daily_refills 20 7 5 8 407 = 3 := by
  sorry

end NUMINAMATH_CALUDE_refill_count_is_three_l2886_288608


namespace NUMINAMATH_CALUDE_reuleaux_triangle_fits_all_holes_l2886_288619

-- Define a Reuleaux Triangle
structure ReuleauxTriangle where
  -- Add necessary properties of a Reuleaux Triangle
  constant_width : ℝ

-- Define the types of holes
inductive HoleType
  | Triangular
  | Square
  | Circular

-- Define a function to check if a shape fits into a hole
def fits_into (shape : ReuleauxTriangle) (hole : HoleType) : Prop :=
  match hole with
  | HoleType.Triangular => true -- Assume it fits into triangular hole
  | HoleType.Square => true     -- Assume it fits into square hole
  | HoleType.Circular => true   -- Assume it fits into circular hole

-- Theorem statement
theorem reuleaux_triangle_fits_all_holes (r : ReuleauxTriangle) :
  (∀ (h : HoleType), fits_into r h) :=
sorry

end NUMINAMATH_CALUDE_reuleaux_triangle_fits_all_holes_l2886_288619


namespace NUMINAMATH_CALUDE_no_solutions_to_equation_l2886_288636

theorem no_solutions_to_equation :
  ¬ ∃ x : ℝ, (2 * x^2 - 10 * x) / (x^2 - 5 * x) = x - 3 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_to_equation_l2886_288636


namespace NUMINAMATH_CALUDE_power_of_power_l2886_288681

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2886_288681


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2886_288629

/-- The constant term in the binomial expansion of (3x^2 - 2/x^3)^5 is 1080 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (3 * x^2 - 2 / x^3)^5
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = 1080 :=
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2886_288629


namespace NUMINAMATH_CALUDE_first_tier_tax_percentage_l2886_288677

theorem first_tier_tax_percentage
  (first_tier_limit : ℝ)
  (second_tier_rate : ℝ)
  (car_price : ℝ)
  (total_tax : ℝ)
  (h1 : first_tier_limit = 11000)
  (h2 : second_tier_rate = 0.09)
  (h3 : car_price = 18000)
  (h4 : total_tax = 1950) :
  ∃ first_tier_rate : ℝ,
    first_tier_rate = 0.12 ∧
    total_tax = first_tier_rate * first_tier_limit +
                second_tier_rate * (car_price - first_tier_limit) := by
  sorry

end NUMINAMATH_CALUDE_first_tier_tax_percentage_l2886_288677


namespace NUMINAMATH_CALUDE_more_boys_than_girls_boy_girl_difference_l2886_288607

/-- The number of girls in the school -/
def num_girls : ℕ := 635

/-- The number of boys in the school -/
def num_boys : ℕ := 1145

/-- There are more boys than girls -/
theorem more_boys_than_girls : num_boys > num_girls := by sorry

/-- The difference between the number of boys and girls is 510 -/
theorem boy_girl_difference : num_boys - num_girls = 510 := by sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_boy_girl_difference_l2886_288607


namespace NUMINAMATH_CALUDE_tank_weight_calculation_l2886_288602

def tank_capacity : ℝ := 200
def empty_tank_weight : ℝ := 80
def fill_percentage : ℝ := 0.8
def water_weight_per_gallon : ℝ := 8

theorem tank_weight_calculation : 
  let water_volume : ℝ := tank_capacity * fill_percentage
  let water_weight : ℝ := water_volume * water_weight_per_gallon
  let total_weight : ℝ := empty_tank_weight + water_weight
  total_weight = 1360 := by sorry

end NUMINAMATH_CALUDE_tank_weight_calculation_l2886_288602


namespace NUMINAMATH_CALUDE_twentieth_base4_is_110_l2886_288655

/-- Converts a decimal number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The 20th number in the base-4 system -/
def twentieth_base4 : List ℕ := toBase4 20

theorem twentieth_base4_is_110 : twentieth_base4 = [1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_twentieth_base4_is_110_l2886_288655


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l2886_288672

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : ℕ
  num_districts : ℕ
  precincts_per_district : ℕ
  voters_per_precinct : ℕ

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The theorem stating the minimum number of voters for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingStructure) 
  (h1 : vs.total_voters = 135)
  (h2 : vs.num_districts = 5)
  (h3 : vs.precincts_per_district = 9)
  (h4 : vs.voters_per_precinct = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.precincts_per_district * vs.voters_per_precinct) :
  min_voters_to_win vs = 30 := by
  sorry

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l2886_288672


namespace NUMINAMATH_CALUDE_sample_data_properties_l2886_288615

theorem sample_data_properties (x : Fin 6 → ℝ) 
  (h_ordered : ∀ i j, i < j → x i ≤ x j) : 
  (((x 2 + x 3) / 2 = (x 3 + x 4) / 2) ∧ 
  (x 5 - x 2 ≤ x 6 - x 1)) := by sorry

end NUMINAMATH_CALUDE_sample_data_properties_l2886_288615


namespace NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l2886_288668

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem ratio_A_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l2886_288668


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l2886_288695

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l2886_288695


namespace NUMINAMATH_CALUDE_isosceles_triangle_l2886_288618

theorem isosceles_triangle (A B C : Real) (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l2886_288618


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l2886_288674

/-- Two lines in the form A₁x + B₁y + C₁ = 0 and A₂x + B₂y + C₂ = 0 are perpendicular -/
def are_perpendicular (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) : Prop :=
  A₁ * A₂ + B₁ * B₂ = 0

/-- The theorem stating the necessary and sufficient condition for two lines to be perpendicular -/
theorem perpendicular_lines_condition
  (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) :
  (∃ x y : ℝ, A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) →
  (are_perpendicular A₁ B₁ C₁ A₂ B₂ C₂ ↔ 
   ∀ x₁ y₁ x₂ y₂ : ℝ, 
   A₁ * x₁ + B₁ * y₁ + C₁ = 0 ∧ 
   A₁ * x₂ + B₁ * y₂ + C₁ = 0 ∧ 
   A₂ * x₁ + B₂ * y₁ + C₂ = 0 ∧ 
   A₂ * x₂ + B₂ * y₂ + C₂ = 0 →
   (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
   ((x₂ - x₁) * (y₂ - y₁) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l2886_288674


namespace NUMINAMATH_CALUDE_four_coin_stacking_methods_l2886_288637

/-- Represents a coin with two sides -/
inductive Coin
| Head
| Tail

/-- Represents a stack of coins -/
def CoinStack := List Coin

/-- Checks if a given coin stack is valid (no adjacent heads) -/
def is_valid_stack (stack : CoinStack) : Bool :=
  match stack with
  | [] => true
  | [_] => true
  | Coin.Head :: Coin.Head :: _ => false
  | _ :: rest => is_valid_stack rest

/-- Generates all possible coin stacks of a given length -/
def generate_stacks (n : Nat) : List CoinStack :=
  if n = 0 then [[]]
  else
    let prev_stacks := generate_stacks (n - 1)
    prev_stacks.bind (fun stack => [Coin.Head :: stack, Coin.Tail :: stack])

/-- Counts the number of valid coin stacks of a given length -/
def count_valid_stacks (n : Nat) : Nat :=
  (generate_stacks n).filter is_valid_stack |>.length

/-- The main theorem to be proved -/
theorem four_coin_stacking_methods :
  count_valid_stacks 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_coin_stacking_methods_l2886_288637


namespace NUMINAMATH_CALUDE_hausdorff_dim_countable_union_l2886_288609

open MeasureTheory

-- Define a countable collection of sets
variable {α : Type*} [MeasurableSpace α]
variable (A : ℕ → Set α)

-- Define Hausdorff dimension
noncomputable def hausdorffDim (S : Set α) : ℝ := sorry

-- State the theorem
theorem hausdorff_dim_countable_union :
  hausdorffDim (⋃ i, A i) = ⨆ i, hausdorffDim (A i) := by sorry

end NUMINAMATH_CALUDE_hausdorff_dim_countable_union_l2886_288609


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2886_288611

theorem cubic_equation_solution (x : ℝ) : 
  x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2886_288611


namespace NUMINAMATH_CALUDE_population_reaches_capacity_years_to_max_capacity_l2886_288666

/-- The maximum capacity of the realm in people -/
def max_capacity : ℕ := 35000 / 2

/-- The initial population in 2023 -/
def initial_population : ℕ := 500

/-- The population growth factor every 20 years -/
def growth_factor : ℕ := 2

/-- The population after n 20-year periods -/
def population (n : ℕ) : ℕ := initial_population * growth_factor ^ n

/-- The number of 20-year periods after which the population reaches or exceeds the maximum capacity -/
def periods_to_max_capacity : ℕ := 5

theorem population_reaches_capacity :
  population periods_to_max_capacity ≥ max_capacity ∧
  population (periods_to_max_capacity - 1) < max_capacity :=
sorry

theorem years_to_max_capacity : periods_to_max_capacity * 20 = 100 :=
sorry

end NUMINAMATH_CALUDE_population_reaches_capacity_years_to_max_capacity_l2886_288666


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2886_288645

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : b = (a + c) / 2) (h5 : b^2 = a^2 - c^2) : 
  let e := c / a
  0 < e ∧ e < 1 ∧ e = 3/5 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2886_288645


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2886_288603

theorem unique_solution_to_equation :
  ∃! y : ℝ, y ≠ 2 ∧ y ≠ -2 ∧
  (-12 * y) / (y^2 - 4) = (3 * y) / (y + 2) - 9 / (y - 2) ∧
  y = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2886_288603


namespace NUMINAMATH_CALUDE_circle_tangent_to_lines_l2886_288624

/-- The circle with center (1, 1) and radius √5 is tangent to both lines 2x - y + 4 = 0 and 2x - y - 6 = 0 -/
theorem circle_tangent_to_lines :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 5}
  let line1 := {(x, y) : ℝ × ℝ | 2*x - y + 4 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 2*x - y - 6 = 0}
  (∃ p ∈ circle ∩ line1, ∀ q ∈ circle, q ∉ line1 ∨ q = p) ∧
  (∃ p ∈ circle ∩ line2, ∀ q ∈ circle, q ∉ line2 ∨ q = p) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_lines_l2886_288624


namespace NUMINAMATH_CALUDE_no_power_ending_222_l2886_288657

theorem no_power_ending_222 :
  ¬ ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ ∃ (n : ℕ), x^y = 1000*n + 222 :=
sorry

end NUMINAMATH_CALUDE_no_power_ending_222_l2886_288657


namespace NUMINAMATH_CALUDE_expression_equals_sum_l2886_288638

theorem expression_equals_sum (a b c : ℝ) (ha : a = 14) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

#eval (14 : ℝ) + 19 + 23

end NUMINAMATH_CALUDE_expression_equals_sum_l2886_288638


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l2886_288600

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + a*x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 := by
  sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l2886_288600


namespace NUMINAMATH_CALUDE_mean_calculation_l2886_288687

theorem mean_calculation (x y : ℝ) : 
  (28 + x + 50 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + y + x) / 5 = (258 + y) / 5 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l2886_288687


namespace NUMINAMATH_CALUDE_third_month_sale_l2886_288642

/-- Calculates the missing sale amount given the average sale and other known sales. -/
def calculate_missing_sale (average : ℕ) (num_months : ℕ) (known_sales : List ℕ) : ℕ :=
  average * num_months - known_sales.sum

/-- The problem statement -/
theorem third_month_sale (average : ℕ) (num_months : ℕ) (known_sales : List ℕ) :
  average = 5600 ∧ 
  num_months = 6 ∧ 
  known_sales = [5266, 5768, 5678, 6029, 4937] →
  calculate_missing_sale average num_months known_sales = 5922 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l2886_288642


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2886_288628

theorem inequality_equivalence (x : ℝ) : 3 * x^2 + x < 8 ↔ -2 < x ∧ x < 4/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2886_288628


namespace NUMINAMATH_CALUDE_unchanged_temperature_count_is_219_l2886_288652

/-- The count of integer Fahrenheit temperatures between 32 and 2000 (inclusive) 
    that remain unchanged after the specified conversion process -/
def unchangedTemperatureCount : ℕ :=
  let minTemp := 32
  let maxTemp := 2000
  (maxTemp - minTemp) / 9 + 1

theorem unchanged_temperature_count_is_219 : 
  unchangedTemperatureCount = 219 := by
  sorry

end NUMINAMATH_CALUDE_unchanged_temperature_count_is_219_l2886_288652


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2886_288631

/-- Given a point A(m, 2) on the line y = 2x - 4, prove that its coordinates are (3, 2) -/
theorem point_A_coordinates :
  ∀ m : ℝ, (2 : ℝ) = 2 * m - 4 → m = 3 ∧ (3, 2) = (m, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l2886_288631


namespace NUMINAMATH_CALUDE_defective_and_shipped_percentage_l2886_288663

/-- The percentage of defective units produced -/
def defective_rate : ℝ := 0.08

/-- The percentage of defective units shipped -/
def shipped_rate : ℝ := 0.05

/-- The percentage of units that are both defective and shipped -/
def defective_and_shipped_rate : ℝ := defective_rate * shipped_rate

theorem defective_and_shipped_percentage :
  defective_and_shipped_rate = 0.004 := by sorry

end NUMINAMATH_CALUDE_defective_and_shipped_percentage_l2886_288663


namespace NUMINAMATH_CALUDE_flour_measurement_l2886_288649

theorem flour_measurement (flour_needed : ℚ) (cup_capacity : ℚ) : 
  flour_needed = 4 + 3 / 4 →
  cup_capacity = 1 / 2 →
  ⌈flour_needed / cup_capacity⌉ = 10 := by
  sorry

end NUMINAMATH_CALUDE_flour_measurement_l2886_288649


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l2886_288626

def volleyball_team_size : ℕ := 10
def lineup_size : ℕ := 5

theorem volleyball_lineup_combinations :
  (volleyball_team_size.factorial) / ((volleyball_team_size - lineup_size).factorial) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l2886_288626


namespace NUMINAMATH_CALUDE_product_and_sum_of_factors_l2886_288671

theorem product_and_sum_of_factors : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8775 ∧ 
  a + b = 110 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_factors_l2886_288671


namespace NUMINAMATH_CALUDE_quadratic_roots_sine_cosine_l2886_288699

theorem quadratic_roots_sine_cosine (α : Real) (c : Real) :
  (∃ (x y : Real), x = Real.sin α ∧ y = Real.cos α ∧ 
   10 * x^2 - 7 * x - c = 0 ∧ 10 * y^2 - 7 * y - c = 0) →
  c = 2.55 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sine_cosine_l2886_288699


namespace NUMINAMATH_CALUDE_diana_weekly_earnings_l2886_288614

/-- Represents Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  hourly_rate : ℕ

/-- Calculates Diana's weekly earnings based on her work schedule --/
def weekly_earnings (d : DianaWork) : ℕ :=
  (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours) * d.hourly_rate

/-- Diana's actual work schedule --/
def diana : DianaWork :=
  { monday_hours := 10
    tuesday_hours := 15
    wednesday_hours := 10
    thursday_hours := 15
    friday_hours := 10
    hourly_rate := 30 }

/-- Theorem stating that Diana's weekly earnings are $1800 --/
theorem diana_weekly_earnings :
  weekly_earnings diana = 1800 := by
  sorry


end NUMINAMATH_CALUDE_diana_weekly_earnings_l2886_288614


namespace NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l2886_288625

-- Part 1
theorem min_value_theorem (x : ℝ) (hx : x > 0) : 12 / x + 3 * x ≥ 12 := by
  sorry

-- Part 2
theorem max_value_theorem (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) : x * (1 - 3 * x) ≤ 1/12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l2886_288625


namespace NUMINAMATH_CALUDE_music_stand_cost_l2886_288622

/-- The cost of Jason's music stand, given his total spending and the costs of other items. -/
theorem music_stand_cost (total_spent flute_cost book_cost : ℚ) 
  (h1 : total_spent = 158.35)
  (h2 : flute_cost = 142.46)
  (h3 : book_cost = 7) :
  total_spent - (flute_cost + book_cost) = 8.89 := by
  sorry

end NUMINAMATH_CALUDE_music_stand_cost_l2886_288622


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l2886_288630

theorem sphere_surface_area_from_volume (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4 / 3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l2886_288630


namespace NUMINAMATH_CALUDE_servant_worked_nine_months_l2886_288676

/-- Represents the salary and work duration of a servant --/
structure ServantSalary where
  yearly_cash : ℕ  -- Yearly cash salary in Rupees
  turban_value : ℕ  -- Value of the turban in Rupees
  received_cash : ℕ  -- Cash received when leaving in Rupees
  months_worked : ℕ  -- Number of months worked

/-- Calculates the number of months a servant worked based on their salary structure --/
def calculate_months_worked (s : ServantSalary) : ℕ :=
  ((s.received_cash + s.turban_value) * 12) / (s.yearly_cash + s.turban_value)

/-- Theorem stating that under the given conditions, the servant worked for 9 months --/
theorem servant_worked_nine_months (s : ServantSalary) 
  (h1 : s.yearly_cash = 90)
  (h2 : s.turban_value = 90)
  (h3 : s.received_cash = 45) :
  calculate_months_worked s = 9 := by
  sorry

end NUMINAMATH_CALUDE_servant_worked_nine_months_l2886_288676


namespace NUMINAMATH_CALUDE_apples_ordered_per_month_l2886_288616

def chandler_initial : ℕ := 23
def lucy_initial : ℕ := 19
def ross_initial : ℕ := 15
def chandler_increase : ℕ := 2
def lucy_decrease : ℕ := 1
def weeks_per_month : ℕ := 4

def total_apples_month : ℕ :=
  (chandler_initial + (chandler_initial + chandler_increase) + 
   (chandler_initial + 2 * chandler_increase) + 
   (chandler_initial + 3 * chandler_increase)) +
  (lucy_initial + (lucy_initial - lucy_decrease) + 
   (lucy_initial - 2 * lucy_decrease) + 
   (lucy_initial - 3 * lucy_decrease)) +
  (ross_initial * weeks_per_month)

theorem apples_ordered_per_month : 
  total_apples_month = 234 := by sorry

end NUMINAMATH_CALUDE_apples_ordered_per_month_l2886_288616


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2886_288620

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ) := {(4, -1), (-26, -9), (-16, -9), (-6, -1), (50, 15), (-72, -25)}
  ∀ (x y : ℤ), (x^2 - 5*x*y + 6*y^2 - 3*x + 5*y - 25 = 0) ↔ (x, y) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2886_288620


namespace NUMINAMATH_CALUDE_fisherman_catch_l2886_288617

/-- The number of bass caught by the fisherman -/
def bass : ℕ := 32

/-- The number of trout caught by the fisherman -/
def trout : ℕ := bass / 4

/-- The number of bluegill caught by the fisherman -/
def bluegill : ℕ := 2 * bass

/-- The total number of fish caught by the fisherman -/
def total_fish : ℕ := 104

theorem fisherman_catch :
  bass + trout + bluegill = total_fish ∧
  trout = bass / 4 ∧
  bluegill = 2 * bass :=
sorry

end NUMINAMATH_CALUDE_fisherman_catch_l2886_288617


namespace NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_problem_l2886_288644

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ i j k l : ℕ, i + j = k + l → a i * a j = a k * a l :=
sorry

theorem roots_product (p q r : ℝ) (x y : ℝ) (hx : p * x^2 + q * x + r = 0) (hy : p * y^2 + q * y + r = 0) :
  x * y = r / p :=
sorry

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_roots : 3 * a 1^2 - 2 * a 1 - 6 = 0 ∧ 3 * a 10^2 - 2 * a 10 - 6 = 0) :
  a 4 * a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_problem_l2886_288644


namespace NUMINAMATH_CALUDE_sum_of_products_l2886_288643

-- Define the problem statement
theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 2)
  (eq2 : y^2 + y*z + z^2 = 5)
  (eq3 : z^2 + x*z + x^2 = 3) :
  x*y + y*z + x*z = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2886_288643


namespace NUMINAMATH_CALUDE_real_part_z_2017_l2886_288692

def z : ℂ := 1 + Complex.I

theorem real_part_z_2017 : (z^2017).re = 2^1008 := by sorry

end NUMINAMATH_CALUDE_real_part_z_2017_l2886_288692


namespace NUMINAMATH_CALUDE_benny_missed_games_l2886_288641

/-- The number of baseball games Benny missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Benny missed 25 games -/
theorem benny_missed_games :
  let total_games : ℕ := 39
  let attended_games : ℕ := 14
  games_missed total_games attended_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_benny_missed_games_l2886_288641


namespace NUMINAMATH_CALUDE_factorization_implies_sum_l2886_288601

theorem factorization_implies_sum (C D : ℤ) :
  (∀ y : ℝ, 6 * y^2 - 31 * y + 35 = (C * y - 5) * (D * y - 7)) →
  C * D + C = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorization_implies_sum_l2886_288601


namespace NUMINAMATH_CALUDE_product_of_roots_l2886_288665

theorem product_of_roots (x : ℝ) : 
  (x^2 + 2*x - 35 = 0) → 
  ∃ y : ℝ, (y^2 + 2*y - 35 = 0) ∧ (x * y = -35) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2886_288665


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l2886_288653

/-- Proves that given a round trip of 240 miles with a total travel time of 5.4 hours,
    where the return trip speed is 50 miles per hour, the outbound trip speed is 40 miles per hour. -/
theorem round_trip_speed_calculation (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 240 →
  total_time = 5.4 →
  return_speed = 50 →
  ∃ (outbound_speed : ℝ),
    outbound_speed = 40 ∧
    total_time = (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l2886_288653


namespace NUMINAMATH_CALUDE_sin_cos_sum_2023_17_l2886_288678

theorem sin_cos_sum_2023_17 :
  Real.sin (2023 * π / 180) * Real.cos (17 * π / 180) +
  Real.cos (2023 * π / 180) * Real.sin (17 * π / 180) =
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_2023_17_l2886_288678


namespace NUMINAMATH_CALUDE_blake_bought_six_chocolate_packs_l2886_288640

/-- The number of lollipops Blake bought -/
def lollipops : ℕ := 4

/-- The cost of one lollipop in dollars -/
def lollipop_cost : ℕ := 2

/-- The number of $10 bills Blake gave to the cashier -/
def bills_given : ℕ := 6

/-- The amount of change Blake received in dollars -/
def change_received : ℕ := 4

/-- The cost of one pack of chocolate in terms of lollipops -/
def chocolate_pack_cost : ℕ := 4 * lollipop_cost

/-- The total amount Blake spent in dollars -/
def total_spent : ℕ := bills_given * 10 - change_received

/-- Theorem stating that Blake bought 6 packs of chocolate -/
theorem blake_bought_six_chocolate_packs : 
  (total_spent - lollipops * lollipop_cost) / chocolate_pack_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_blake_bought_six_chocolate_packs_l2886_288640


namespace NUMINAMATH_CALUDE_probability_r_successes_correct_l2886_288648

/-- The probability of exactly r successful shots by the time the nth shot is taken -/
def probability_r_successes (n r : ℕ) (p : ℝ) : ℝ :=
  Nat.choose (n - 1) (r - 1) * p ^ r * (1 - p) ^ (n - r)

/-- Theorem stating the probability of exactly r successful shots by the nth shot -/
theorem probability_r_successes_correct (n r : ℕ) (p : ℝ) 
    (h1 : 0 ≤ p) (h2 : p ≤ 1) (h3 : 1 ≤ r) (h4 : r ≤ n) : 
  probability_r_successes n r p = Nat.choose (n - 1) (r - 1) * p ^ r * (1 - p) ^ (n - r) :=
by sorry

end NUMINAMATH_CALUDE_probability_r_successes_correct_l2886_288648


namespace NUMINAMATH_CALUDE_train_speed_proof_l2886_288613

/-- Proves that a train crossing a bridge has a speed of approximately 36 kmph given specific conditions. -/
theorem train_speed_proof (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 150)
  (h3 : time_to_cross = 28.997680185585153) : 
  ∃ (speed : ℝ), abs (speed - 36) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_proof_l2886_288613


namespace NUMINAMATH_CALUDE_table_tennis_probabilities_l2886_288661

/-- Represents the probability of player A winning a serve -/
def p_win : ℝ := 0.6

/-- Probability that player A scores i points in two consecutive serves -/
def p_score (i : Fin 3) : ℝ :=
  match i with
  | 0 => (1 - p_win)^2
  | 1 => 2 * p_win * (1 - p_win)
  | 2 => p_win^2

/-- Theorem stating the probabilities of specific score situations in a table tennis game -/
theorem table_tennis_probabilities :
  let p_b_leads := p_score 0 * p_win + p_score 1 * (1 - p_win)
  let p_a_leads := p_score 1 * p_score 2 + p_score 2 * p_score 1 + p_score 2 * p_score 2
  (p_b_leads = 0.352) ∧ (p_a_leads = 0.3072) := by
  sorry


end NUMINAMATH_CALUDE_table_tennis_probabilities_l2886_288661


namespace NUMINAMATH_CALUDE_product_469158_9999_l2886_288654

theorem product_469158_9999 : 469158 * 9999 = 4690872842 := by
  sorry

end NUMINAMATH_CALUDE_product_469158_9999_l2886_288654


namespace NUMINAMATH_CALUDE_city_mpg_is_32_l2886_288670

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarFuelEfficiency where
  highway_miles_per_tank : ℝ
  city_miles_per_tank : ℝ
  highway_city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
def city_mpg (car : CarFuelEfficiency) : ℝ :=
  sorry

/-- Theorem stating that for the given car data, the city MPG is 32 -/
theorem city_mpg_is_32 (car : CarFuelEfficiency)
  (h1 : car.highway_miles_per_tank = 462)
  (h2 : car.city_miles_per_tank = 336)
  (h3 : car.highway_city_mpg_difference = 12) :
  city_mpg car = 32 := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_is_32_l2886_288670


namespace NUMINAMATH_CALUDE_problem_statement_l2886_288667

theorem problem_statement (a b : ℤ) (h1 : a = -5) (h2 : b = 3) : 
  -a - b^4 + a*b = -91 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2886_288667


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2886_288610

theorem inequality_solution_set :
  {x : ℝ | (1 : ℝ) / (x - 1) < -1} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2886_288610


namespace NUMINAMATH_CALUDE_wire_cutting_l2886_288621

/-- Given a wire of length 28 cm, if one piece is 2.00001/5 times the length of the other,
    then the shorter piece is 20 cm long. -/
theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 28 →
  ratio = 2.00001 / 5 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l2886_288621


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l2886_288664

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = b * Complex.I) →  -- z is purely imaginary
  (∃ r : ℝ, (z + 2) / (1 + Complex.I) = r) →  -- (z+2)/(1+i) is real
  z = -2 * Complex.I :=  -- z = -2i
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l2886_288664


namespace NUMINAMATH_CALUDE_rogers_dimes_l2886_288606

/-- The number of dimes Roger initially collected -/
def initial_dimes : ℕ := 15

/-- The number of pennies Roger collected -/
def pennies : ℕ := 42

/-- The number of nickels Roger collected -/
def nickels : ℕ := 36

/-- The number of coins Roger had left after donating -/
def coins_left : ℕ := 27

/-- The number of coins Roger donated -/
def coins_donated : ℕ := 66

theorem rogers_dimes :
  initial_dimes = 15 ∧
  pennies + nickels + initial_dimes = coins_left + coins_donated :=
by sorry

end NUMINAMATH_CALUDE_rogers_dimes_l2886_288606


namespace NUMINAMATH_CALUDE_digit_101_of_7_12_l2886_288633

/-- The decimal representation of 7/12 has a repeating sequence of 4 digits. -/
def decimal_7_12_period : ℕ := 4

/-- The first digit of the repeating sequence in the decimal representation of 7/12. -/
def first_digit_7_12 : ℕ := 5

/-- The 101st digit after the decimal point in the decimal representation of 7/12 is 5. -/
theorem digit_101_of_7_12 : 
  (101 % decimal_7_12_period = 1) → 
  (Nat.digitChar (first_digit_7_12) = '5') := by
sorry

end NUMINAMATH_CALUDE_digit_101_of_7_12_l2886_288633


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_l2886_288694

theorem stratified_sampling_third_year 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (total_sample : ℕ) 
  (h1 : total_students = 1200)
  (h2 : third_year_students = 300)
  (h3 : total_sample = 100) :
  (third_year_students : ℚ) / total_students * total_sample = 25 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_l2886_288694


namespace NUMINAMATH_CALUDE_shortest_side_length_l2886_288656

theorem shortest_side_length (a b c : ℝ) : 
  a + b + c = 15 ∧ a = 2 * c ∧ b = 2 * c → c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l2886_288656


namespace NUMINAMATH_CALUDE_edge_stop_probability_l2886_288680

-- Define the grid size
def gridSize : Nat := 4

-- Define a position on the grid
structure Position where
  x : Nat
  y : Nat
  deriving Repr

-- Define the possible directions
inductive Direction
  | Up
  | Down
  | Left
  | Right

-- Define whether a position is on the edge
def isEdge (pos : Position) : Bool :=
  pos.x == 1 || pos.x == gridSize || pos.y == 1 || pos.y == gridSize

-- Define the next position after a move, with wrap-around
def nextPosition (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Up => ⟨pos.x, if pos.y == gridSize then 1 else pos.y + 1⟩
  | Direction.Down => ⟨pos.x, if pos.y == 1 then gridSize else pos.y - 1⟩
  | Direction.Left => ⟨if pos.x == 1 then gridSize else pos.x - 1, pos.y⟩
  | Direction.Right => ⟨if pos.x == gridSize then 1 else pos.x + 1, pos.y⟩

-- Define the probability of stopping at an edge within n hops
def probStopAtEdge (start : Position) (n : Nat) : Real :=
  sorry

-- Theorem statement
theorem edge_stop_probability :
  probStopAtEdge ⟨2, 1⟩ 5 =
    probStopAtEdge ⟨2, 1⟩ 1 +
    probStopAtEdge ⟨2, 1⟩ 2 +
    probStopAtEdge ⟨2, 1⟩ 3 +
    probStopAtEdge ⟨2, 1⟩ 4 +
    probStopAtEdge ⟨2, 1⟩ 5 :=
  sorry

end NUMINAMATH_CALUDE_edge_stop_probability_l2886_288680


namespace NUMINAMATH_CALUDE_set_equality_l2886_288697

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem set_equality : (Set.compl A) ∪ B = Set.Iic (-1) ∪ Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_set_equality_l2886_288697


namespace NUMINAMATH_CALUDE_exact_three_correct_deliveries_probability_l2886_288612

def num_packages : ℕ := 5

def num_correct_deliveries : ℕ := 3

def total_permutations : ℕ := num_packages.factorial

def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

def num_ways_correct_deliveries : ℕ := choose num_packages num_correct_deliveries

def num_derangements_remaining : ℕ := 1

theorem exact_three_correct_deliveries_probability :
  (num_ways_correct_deliveries * num_derangements_remaining : ℚ) / total_permutations = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_correct_deliveries_probability_l2886_288612


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9879_l2886_288698

theorem largest_prime_factor_of_9879 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9879 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 9879 → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9879_l2886_288698


namespace NUMINAMATH_CALUDE_complex_fraction_real_implies_a_negative_one_l2886_288689

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition that (a+i)/(1-i) is real
def is_real (a : ℝ) : Prop := ∃ (r : ℝ), (a + i) / (1 - i) = r

-- Theorem statement
theorem complex_fraction_real_implies_a_negative_one (a : ℝ) :
  is_real a → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_real_implies_a_negative_one_l2886_288689


namespace NUMINAMATH_CALUDE_smallest_square_partition_l2886_288632

theorem smallest_square_partition : ∃ (n : ℕ),
  n > 0 ∧
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 10 ∧ a ≥ 8 ∧ n^2 = a * 1^2 + b * 2^2) ∧
  (∀ (m : ℕ), m < n →
    ¬(∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c + d = 10 ∧ c ≥ 8 ∧ m^2 = c * 1^2 + d * 2^2)) ∧
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l2886_288632


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2886_288635

/-- Given two vectors in R³ that satisfy certain conditions, prove that k = -3/2 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ × ℝ) (k : ℝ) :
  a = (1, 2, 1) →
  b = (1, 2, 2) →
  ∃ (t : ℝ), t ≠ 0 ∧ (k • a + b) = t • (a - 2 • b) →
  k = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2886_288635


namespace NUMINAMATH_CALUDE_class_average_mark_l2886_288623

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_average : ℝ) (remaining_average : ℝ) : 
  total_students = 13 →
  excluded_students = 5 →
  excluded_average = 40 →
  remaining_average = 92 →
  (total_students : ℝ) * (total_students * (remaining_average : ℝ) - excluded_students * excluded_average) / 
    (total_students * (total_students - excluded_students)) = 72 := by
  sorry


end NUMINAMATH_CALUDE_class_average_mark_l2886_288623


namespace NUMINAMATH_CALUDE_paper_clip_cost_l2886_288658

/-- The cost of Eldora's purchase -/
def eldora_cost : ℝ := 55.40

/-- The cost of Finn's purchase -/
def finn_cost : ℝ := 61.70

/-- The number of paper clip boxes Eldora bought -/
def eldora_clips : ℕ := 15

/-- The number of index card packages Eldora bought -/
def eldora_cards : ℕ := 7

/-- The number of paper clip boxes Finn bought -/
def finn_clips : ℕ := 12

/-- The number of index card packages Finn bought -/
def finn_cards : ℕ := 10

/-- The cost of one box of paper clips -/
noncomputable def clip_cost : ℝ := 1.835

theorem paper_clip_cost : 
  ∃ (card_cost : ℝ), 
    (eldora_clips : ℝ) * clip_cost + (eldora_cards : ℝ) * card_cost = eldora_cost ∧ 
    (finn_clips : ℝ) * clip_cost + (finn_cards : ℝ) * card_cost = finn_cost :=
by sorry

end NUMINAMATH_CALUDE_paper_clip_cost_l2886_288658


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2886_288691

theorem geometric_sequence_seventh_term (x : ℝ) (b : ℕ → ℝ) 
  (h1 : b 1 = Real.sin x ^ 2)
  (h2 : b 2 = Real.sin x * Real.cos x)
  (h3 : b 3 = (Real.cos x ^ 2) / (Real.sin x))
  (h_geom : ∀ n : ℕ, n ≥ 1 → b (n + 1) = (b 2 / b 1) * b n) :
  b 7 = Real.cos x + Real.sin x :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2886_288691


namespace NUMINAMATH_CALUDE_sum_of_second_progression_l2886_288650

/-- Given two arithmetic progressions with specific conditions, prove that the sum of the terms of the second progression is 14. -/
theorem sum_of_second_progression (a₁ a₅ b₁ bₙ : ℚ) (N : ℕ) : 
  a₁ = 7 →
  a₅ = -5 →
  b₁ = 0 →
  bₙ = 7/2 →
  N > 1 →
  (∃ d D : ℚ, a₁ + 2*d = b₁ + 2*D ∧ a₅ = a₁ + 4*d ∧ bₙ = b₁ + (N-1)*D) →
  (N/2 : ℚ) * (b₁ + bₙ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_second_progression_l2886_288650


namespace NUMINAMATH_CALUDE_congruence_solution_l2886_288639

theorem congruence_solution (n : ℤ) : 
  0 ≤ n ∧ n < 203 ∧ (150 * n) % 203 = 95 % 203 → n = 144 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l2886_288639


namespace NUMINAMATH_CALUDE_bernardo_silvia_game_l2886_288684

theorem bernardo_silvia_game (N : ℕ) : N = 24 ↔ 
  (N ≤ 999) ∧ 
  (3 * N < 800) ∧ 
  (3 * N - 30 < 800) ∧ 
  (9 * N - 90 < 800) ∧ 
  (9 * N - 120 < 800) ∧ 
  (27 * N - 360 < 800) ∧ 
  (27 * N - 390 < 800) ∧ 
  (81 * N - 1170 ≥ 800) ∧ 
  (∀ m : ℕ, m < N → 
    (3 * m < 800) ∧ 
    (3 * m - 30 < 800) ∧ 
    (9 * m - 90 < 800) ∧ 
    (9 * m - 120 < 800) ∧ 
    (27 * m - 360 < 800) ∧ 
    (27 * m - 390 < 800) ∧ 
    (81 * m - 1170 < 800)) := by
  sorry

end NUMINAMATH_CALUDE_bernardo_silvia_game_l2886_288684


namespace NUMINAMATH_CALUDE_ben_win_probability_l2886_288673

theorem ben_win_probability (lose_prob : ℚ) (win_prob : ℚ) : 
  lose_prob = 5/8 → win_prob = 1 - lose_prob → win_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l2886_288673


namespace NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l2886_288685

theorem belt_and_road_population_scientific_notation :
  let billion : ℝ := 10^9
  4.4 * billion = 4.4 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l2886_288685


namespace NUMINAMATH_CALUDE_last_bead_is_blue_l2886_288634

/-- Represents the colors of beads -/
inductive BeadColor
| Red
| Orange
| Yellow
| Green
| Blue
| Purple

/-- Represents the pattern of beads -/
def beadPattern : List BeadColor :=
  [BeadColor.Red, BeadColor.Orange, BeadColor.Yellow, BeadColor.Yellow,
   BeadColor.Green, BeadColor.Blue, BeadColor.Purple]

/-- The total number of beads in the bracelet -/
def totalBeads : Nat := 83

/-- Theorem stating that the last bead of the bracelet is blue -/
theorem last_bead_is_blue :
  (totalBeads % beadPattern.length) = 6 →
  beadPattern[(totalBeads - 1) % beadPattern.length] = BeadColor.Blue :=
by sorry

end NUMINAMATH_CALUDE_last_bead_is_blue_l2886_288634
