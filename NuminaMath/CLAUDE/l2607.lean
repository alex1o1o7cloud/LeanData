import Mathlib

namespace perpendicular_line_through_point_l2607_260732

/-- Given two lines in the 2D plane represented by their equations:
    ax + by + c = 0 and dx + ey + f = 0,
    this function returns true if the lines are perpendicular. -/
def are_perpendicular (a b c d e f : ℝ) : Prop :=
  a * d + b * e = 0

/-- Given a line ax + by + c = 0 and a point (x₀, y₀),
    this function returns true if the point lies on the line. -/
def point_on_line (a b c x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

theorem perpendicular_line_through_point :
  are_perpendicular 4 (-3) 2 3 4 1 ∧
  point_on_line 4 (-3) 2 1 2 :=
by sorry

end perpendicular_line_through_point_l2607_260732


namespace total_chase_time_distance_equality_at_capture_l2607_260701

/-- Represents the chase scenario between Black Cat Detective and One-Ear --/
structure ChaseScenario where
  v : ℝ  -- One-Ear's speed
  initial_time : ℝ  -- Time before chase begins
  chase_time : ℝ  -- Time of chase

/-- Conditions of the chase scenario --/
def chase_conditions (s : ChaseScenario) : Prop :=
  s.initial_time = 13 ∧
  s.chase_time = 1 ∧
  s.v > 0

/-- The theorem stating the total time of the chase --/
theorem total_chase_time (s : ChaseScenario) 
  (h : chase_conditions s) : s.initial_time + s.chase_time = 14 := by
  sorry

/-- The theorem proving the distance equality at the point of capture --/
theorem distance_equality_at_capture (s : ChaseScenario) 
  (h : chase_conditions s) : 
  (5 * s.v + s.v) * s.initial_time = (7.5 * s.v - s.v) * s.chase_time := by
  sorry

end total_chase_time_distance_equality_at_capture_l2607_260701


namespace positive_root_existence_l2607_260798

def f (x : ℝ) := x^5 - x - 1

theorem positive_root_existence :
  ∃ x ∈ Set.Icc 1 2, f x = 0 ∧ x > 0 :=
sorry

end positive_root_existence_l2607_260798


namespace necessary_but_not_sufficient_condition_l2607_260781

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) := by sorry

end necessary_but_not_sufficient_condition_l2607_260781


namespace sum_of_digits_9ab_eq_17965_l2607_260758

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ :=
  d * ((10^n - 1) / 9)

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_9ab_eq_17965 :
  let a := repeatedDigit 4 1995
  let b := repeatedDigit 7 1995
  sumOfDigits (9 * a * b) = 17965 :=
sorry

end sum_of_digits_9ab_eq_17965_l2607_260758


namespace correct_quotient_l2607_260782

theorem correct_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 70) : D / 21 = 40 := by
  sorry

end correct_quotient_l2607_260782


namespace right_trapezoid_inscribed_circle_theorem_l2607_260773

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- Length of the longer base -/
  a : ℝ
  /-- Length of the shorter base -/
  c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The longer base is longer than the shorter base -/
  h1 : a > c
  /-- The bases are positive -/
  h2 : a > 0
  h3 : c > 0
  /-- The radius is positive -/
  h4 : r > 0
  /-- Relation between radius and bases -/
  h5 : r = (a * c) / (a + c)

/-- The theorem to be proved -/
theorem right_trapezoid_inscribed_circle_theorem (t : RightTrapezoidWithInscribedCircle) :
  (2 : ℝ) * t.r = 2 / ((1 / t.a) + (1 / t.c)) :=
by sorry

end right_trapezoid_inscribed_circle_theorem_l2607_260773


namespace zero_location_l2607_260775

theorem zero_location (x y : ℝ) 
  (h1 : x^5 < y^8) 
  (h2 : y^8 < y^3) 
  (h3 : y^3 < x^6)
  (h4 : x < 0)
  (h5 : 0 < y)
  (h6 : y < 1) : 
  x^5 < 0 ∧ 0 < y^8 := by
  sorry

end zero_location_l2607_260775


namespace math_team_selection_l2607_260749

theorem math_team_selection (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 →
  k = 3 →
  total = 10 →
  (Nat.choose (total - 1) k) - (Nat.choose (total - 3) k) = 49 := by
  sorry

end math_team_selection_l2607_260749


namespace sculpture_cost_in_inr_l2607_260779

/-- Exchange rate from British pounds to Indian rupees -/
def gbp_to_inr : ℚ := 20

/-- Exchange rate from British pounds to Namibian dollars -/
def gbp_to_nad : ℚ := 18

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 360

/-- Theorem stating the equivalent cost of the sculpture in Indian rupees -/
theorem sculpture_cost_in_inr :
  (sculpture_cost_nad / gbp_to_nad) * gbp_to_inr = 400 := by
  sorry

end sculpture_cost_in_inr_l2607_260779


namespace pascal_triangle_30_rows_l2607_260736

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the first 30 rows of Pascal's Triangle -/
def total_elements : ℕ := sum_first_n 30

theorem pascal_triangle_30_rows :
  total_elements = 465 := by
  sorry

end pascal_triangle_30_rows_l2607_260736


namespace jerry_shelf_theorem_l2607_260780

/-- Calculates the total number of action figures on Jerry's shelf -/
def total_action_figures (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of action figures is the sum of initial and added figures -/
theorem jerry_shelf_theorem (initial : ℕ) (added : ℕ) :
  total_action_figures initial added = initial + added :=
by sorry

end jerry_shelf_theorem_l2607_260780


namespace max_b_value_l2607_260787

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + 3 * b * x

-- State the theorem
theorem max_b_value (a b : ℝ) (ha : a < 0) (hb : b > 0) 
  (hf : ∀ x ∈ Set.Icc 0 1, f a b x ∈ Set.Icc 0 1) : 
  b ≤ Real.sqrt 3 / 2 ∧ ∃ x ∈ Set.Icc 0 1, f a (Real.sqrt 3 / 2) x = 1 := by
sorry

end max_b_value_l2607_260787


namespace symmetric_points_sum_l2607_260771

/-- Given two points M and N that are symmetric with respect to the y-axis,
    prove that the sum of their x-coordinates is zero. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (m - 1 = -(3: ℝ)) → -- M's x-coordinate is opposite to N's
  (1 : ℝ) = n - 1 →    -- M's y-coordinate equals N's
  m + n = 0 :=
by sorry

end symmetric_points_sum_l2607_260771


namespace expression_equality_l2607_260738

theorem expression_equality : ∀ x : ℤ, x = 3 ∨ x = -3 →
  6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2) = 20 := by
  sorry

end expression_equality_l2607_260738


namespace correct_num_kettles_l2607_260724

/-- The number of kettles of hawks the ornithologists are tracking -/
def num_kettles : ℕ := 6

/-- The average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- The number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- The survival rate of babies -/
def survival_rate : ℚ := 3/4

/-- The total number of expected babies this season -/
def total_babies : ℕ := 270

/-- Theorem stating that the number of kettles is correct given the conditions -/
theorem correct_num_kettles : 
  num_kettles = total_babies / (pregnancies_per_kettle * babies_per_pregnancy * survival_rate) :=
sorry

end correct_num_kettles_l2607_260724


namespace divisors_of_8n_cubed_l2607_260765

theorem divisors_of_8n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 12) :
  (Nat.divisors (8 * n^3)).card = 280 := by
  sorry

end divisors_of_8n_cubed_l2607_260765


namespace negation_of_universal_proposition_l2607_260784

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2^x - 1 > 0) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l2607_260784


namespace circumference_difference_concentric_circles_l2607_260737

/-- Given two concentric circles where the outer circle's radius is 12 feet greater than the inner circle's radius, the difference in their circumferences is 24π feet. -/
theorem circumference_difference_concentric_circles (r : ℝ) : 
  2 * π * (r + 12) - 2 * π * r = 24 * π := by
  sorry

end circumference_difference_concentric_circles_l2607_260737


namespace jack_life_timeline_l2607_260792

theorem jack_life_timeline (jack_lifetime : ℝ) 
  (h1 : jack_lifetime = 84)
  (adolescence : ℝ) (h2 : adolescence = (1/6) * jack_lifetime)
  (facial_hair : ℝ) (h3 : facial_hair = (1/12) * jack_lifetime)
  (marriage : ℝ) (h4 : marriage = (1/7) * jack_lifetime)
  (son_birth : ℝ) (h5 : son_birth = 5)
  (son_lifetime : ℝ) (h6 : son_lifetime = (1/2) * jack_lifetime) :
  jack_lifetime - (adolescence + facial_hair + marriage + son_birth + son_lifetime) = 4 := by
sorry

end jack_life_timeline_l2607_260792


namespace coin_flip_probability_l2607_260777

theorem coin_flip_probability (oliver_prob jayden_prob mia_prob : ℚ) :
  oliver_prob = 1/3 →
  jayden_prob = 1/4 →
  mia_prob = 1/5 →
  (∑' n : ℕ, (1 - oliver_prob)^(n-1) * oliver_prob *
              (1 - jayden_prob)^(n-1) * jayden_prob *
              (1 - mia_prob)^(n-1) * mia_prob) = 1/36 := by
  sorry

#check coin_flip_probability

end coin_flip_probability_l2607_260777


namespace escalator_speed_increase_l2607_260728

theorem escalator_speed_increase (total_steps : ℕ) (first_climb : ℕ) (second_climb : ℕ)
  (h_total : total_steps = 125)
  (h_first : first_climb = 45)
  (h_second : second_climb = 55)
  (h_first_valid : first_climb < total_steps)
  (h_second_valid : second_climb < total_steps) :
  (second_climb : ℚ) / first_climb * (total_steps - first_climb : ℚ) / (total_steps - second_climb) = 88 / 63 :=
by sorry

end escalator_speed_increase_l2607_260728


namespace product_sum_fractions_l2607_260709

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 + 1/6) = 57 := by
  sorry

end product_sum_fractions_l2607_260709


namespace both_glasses_and_hair_tied_l2607_260729

def students : Finset ℕ := Finset.range 30

def glasses : Finset ℕ := {1, 3, 7, 10, 23, 27}

def hairTied : Finset ℕ := {1, 9, 11, 20, 23}

theorem both_glasses_and_hair_tied :
  (glasses ∩ hairTied).card = 2 := by sorry

end both_glasses_and_hair_tied_l2607_260729


namespace carl_pink_hats_solution_l2607_260754

/-- The number of pink hard hats Carl took away from the truck -/
def carl_pink_hats : ℕ := sorry

theorem carl_pink_hats_solution : carl_pink_hats = 4 := by
  have initial_pink : ℕ := 26
  have initial_green : ℕ := 15
  have initial_yellow : ℕ := 24
  have john_pink : ℕ := 6
  have john_green : ℕ := 2 * john_pink
  have remaining_hats : ℕ := 43

  sorry

end carl_pink_hats_solution_l2607_260754


namespace sandbag_weight_sandbag_problem_l2607_260730

theorem sandbag_weight (capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : ℝ :=
  let sand_weight := capacity * fill_percentage
  let extra_weight := sand_weight * weight_increase
  sand_weight + extra_weight

theorem sandbag_problem :
  sandbag_weight 250 0.8 0.4 = 280 := by
  sorry

end sandbag_weight_sandbag_problem_l2607_260730


namespace bushes_needed_for_zucchinis_l2607_260767

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers of blueberries that can be traded for zucchinis -/
def containers_traded : ℕ := 6

/-- Represents the number of zucchinis received in trade for containers_traded -/
def zucchinis_received : ℕ := 3

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- Theorem stating that 12 bushes are needed to obtain 60 zucchinis -/
theorem bushes_needed_for_zucchinis : 
  (target_zucchinis * containers_traded) / (zucchinis_received * containers_per_bush) = 12 :=
sorry

end bushes_needed_for_zucchinis_l2607_260767


namespace equation_solutions_l2607_260741

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4) ∧
  (∃ x : ℝ, (x + 1)^2 = 4*x^2 ↔ x = -1/3 ∨ x = 1) := by
sorry

end equation_solutions_l2607_260741


namespace total_rainfall_2005_l2607_260720

def rainfall_2005 (initial_rainfall : ℝ) (yearly_increase : ℝ) : ℝ :=
  12 * (initial_rainfall + 2 * yearly_increase)

theorem total_rainfall_2005 (initial_rainfall yearly_increase : ℝ) 
  (h1 : initial_rainfall = 30)
  (h2 : yearly_increase = 3) :
  rainfall_2005 initial_rainfall yearly_increase = 432 := by
  sorry

end total_rainfall_2005_l2607_260720


namespace two_color_theorem_l2607_260708

/-- A type representing a plane divided by lines -/
structure DividedPlane where
  n : ℕ  -- number of lines
  regions : Set (Set ℝ × ℝ)  -- regions as sets of points
  adjacent : regions → regions → Prop  -- adjacency relation

/-- A coloring of the plane -/
def Coloring (p : DividedPlane) := p.regions → Bool

/-- A valid two-coloring of the plane -/
def ValidColoring (p : DividedPlane) (c : Coloring p) : Prop :=
  ∀ r1 r2 : p.regions, p.adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem: any divided plane has a valid two-coloring -/
theorem two_color_theorem (p : DividedPlane) : ∃ c : Coloring p, ValidColoring p c := by
  sorry

end two_color_theorem_l2607_260708


namespace inner_circle_distance_l2607_260733

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_lengths : a = 9 ∧ b = 12 ∧ c = 15

/-- The path of the center of a circle rolling inside the triangle -/
def inner_circle_path (t : RightTriangle) (r : ℝ) : ℝ := 
  (t.a - 2*r) + (t.b - 2*r) + (t.c - 2*r)

/-- The theorem to be proved -/
theorem inner_circle_distance (t : RightTriangle) : 
  inner_circle_path t 2 = 24 := by sorry

end inner_circle_distance_l2607_260733


namespace perpendicular_line_inclination_angle_l2607_260718

/-- The inclination angle of a line perpendicular to x + √3y - 1 = 0 is π/3 -/
theorem perpendicular_line_inclination_angle : 
  let original_line : Real → Real → Prop := λ x y => x + Real.sqrt 3 * y - 1 = 0
  let perpendicular_slope : Real := Real.sqrt 3
  let inclination_angle : Real := Real.pi / 3
  ∀ x y, original_line x y → 
    ∃ m : Real, m * perpendicular_slope = -1 ∧ 
    Real.tan inclination_angle = perpendicular_slope :=
by sorry

end perpendicular_line_inclination_angle_l2607_260718


namespace ingrid_income_calculation_l2607_260713

-- Define the given constants
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def john_income : ℝ := 58000
def combined_tax_rate : ℝ := 0.3554

-- Define Ingrid's income as a variable
def ingrid_income : ℝ := 72000

-- Theorem statement
theorem ingrid_income_calculation :
  ingrid_income = 72000 ∧
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end ingrid_income_calculation_l2607_260713


namespace fir_tree_count_l2607_260759

/-- Represents the four children in the problem -/
inductive Child
| Anya
| Borya
| Vera
| Gena

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Returns the gender of a child -/
def childGender (c : Child) : Gender :=
  match c with
  | Child.Anya => Gender.Girl
  | Child.Borya => Gender.Boy
  | Child.Vera => Gender.Girl
  | Child.Gena => Gender.Boy

/-- Represents a statement made by a child -/
def Statement := ℕ → Prop

/-- Returns the statement made by each child -/
def childStatement (c : Child) : Statement :=
  match c with
  | Child.Anya => λ n => n = 15
  | Child.Borya => λ n => n % 11 = 0
  | Child.Vera => λ n => n < 25
  | Child.Gena => λ n => n % 22 = 0

theorem fir_tree_count :
  ∃ (n : ℕ) (truthTellers : Finset Child),
    n = 11 ∧
    truthTellers.card = 2 ∧
    (∃ (boy girl : Child), boy ∈ truthTellers ∧ girl ∈ truthTellers ∧
      childGender boy = Gender.Boy ∧ childGender girl = Gender.Girl) ∧
    (∀ c ∈ truthTellers, childStatement c n) ∧
    (∀ c ∉ truthTellers, ¬(childStatement c n)) :=
  sorry

end fir_tree_count_l2607_260759


namespace intersection_nonempty_implies_b_greater_than_neg_one_l2607_260768

def A : Set ℝ := {x | Real.log (x + 2) / Real.log (1/2) < 0}
def B (a b : ℝ) : Set ℝ := {x | (x - a) * (x - b) < 0}

theorem intersection_nonempty_implies_b_greater_than_neg_one :
  (∀ b : ℝ, (A ∩ B (-3) b).Nonempty) → ∀ b : ℝ, b > -1 :=
by
  sorry

end intersection_nonempty_implies_b_greater_than_neg_one_l2607_260768


namespace smallest_number_with_all_factors_l2607_260761

def alice_number : ℕ := 90

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ m : ℕ, m > 0 ∧ has_all_prime_factors alice_number m ∧
  ∀ k : ℕ, k > 0 → has_all_prime_factors alice_number k → m ≤ k :=
by sorry

end smallest_number_with_all_factors_l2607_260761


namespace max_servings_is_fifty_l2607_260707

/-- Represents the number of chunks per serving for each fruit type -/
structure FruitRatio :=
  (cantaloupe : ℕ)
  (honeydew : ℕ)
  (pineapple : ℕ)
  (watermelon : ℕ)

/-- Represents the available chunks of each fruit type -/
structure AvailableFruit :=
  (cantaloupe : ℕ)
  (honeydew : ℕ)
  (pineapple : ℕ)
  (watermelon : ℕ)

/-- Calculates the maximum number of servings possible given a ratio and available fruit -/
def maxServings (ratio : FruitRatio) (available : AvailableFruit) : ℕ :=
  min
    (available.cantaloupe / ratio.cantaloupe)
    (min
      (available.honeydew / ratio.honeydew)
      (min
        (available.pineapple / ratio.pineapple)
        (available.watermelon / ratio.watermelon)))

theorem max_servings_is_fifty :
  let ratio : FruitRatio := ⟨3, 2, 1, 4⟩
  let available : AvailableFruit := ⟨150, 135, 60, 220⟩
  let minServings : ℕ := 50
  maxServings ratio available = 50 ∧ maxServings ratio available ≥ minServings :=
by sorry

end max_servings_is_fifty_l2607_260707


namespace race_distance_l2607_260734

/-- The race problem -/
theorem race_distance (t_A t_B : ℕ) (lead : ℕ) (h1 : t_A = 36) (h2 : t_B = 45) (h3 : lead = 24) :
  ∃ D : ℕ, D = 24 ∧ (D : ℚ) / t_A * t_B = D + lead :=
by sorry

end race_distance_l2607_260734


namespace opposite_def_opposite_of_neg_four_l2607_260751

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -4 is 4 -/
theorem opposite_of_neg_four : opposite (-4 : ℝ) = 4 := by sorry

end opposite_def_opposite_of_neg_four_l2607_260751


namespace max_value_f_l2607_260745

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem max_value_f (a : ℝ) (h : a ≤ 0) :
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ 
    ∀ (y : ℝ), y ∈ Set.Icc 0 1 → f a x ≥ f a y) ∧
  (∃ (max_val : ℝ), 
    (a = 0 → max_val = 1) ∧
    (-2 < a ∧ a < 0 → max_val = Real.exp a) ∧
    (a ≤ -2 → max_val = 4 / (a^2 * Real.exp 2))) :=
by sorry

end max_value_f_l2607_260745


namespace toby_speed_proof_l2607_260721

/-- Represents the speed of Toby when pulling the unloaded sled -/
def unloaded_speed : ℝ := 20

/-- Represents the speed of Toby when pulling the loaded sled -/
def loaded_speed : ℝ := 10

/-- Represents the total journey time in hours -/
def total_time : ℝ := 39

/-- Represents the distance of the first part of the journey (loaded sled) -/
def distance1 : ℝ := 180

/-- Represents the distance of the second part of the journey (unloaded sled) -/
def distance2 : ℝ := 120

/-- Represents the distance of the third part of the journey (loaded sled) -/
def distance3 : ℝ := 80

/-- Represents the distance of the fourth part of the journey (unloaded sled) -/
def distance4 : ℝ := 140

theorem toby_speed_proof :
  (distance1 / loaded_speed) + (distance2 / unloaded_speed) +
  (distance3 / loaded_speed) + (distance4 / unloaded_speed) = total_time :=
by sorry

end toby_speed_proof_l2607_260721


namespace unique_solution_l2607_260746

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 1

/-- The theorem stating that the function g(x) = 2x + 3 is the unique solution -/
theorem unique_solution :
  ∀ g : ℝ → ℝ, FunctionalEquation g → (∀ x : ℝ, g x = 2 * x + 3) :=
by sorry

end unique_solution_l2607_260746


namespace baseball_card_purchase_l2607_260700

/-- The cost of the rare baseball card -/
def card_cost : ℕ := 100

/-- Patricia's money -/
def patricia_money : ℕ := 6

/-- Lisa's money in terms of Patricia's -/
def lisa_money : ℕ := 5 * patricia_money

/-- Charlotte's money in terms of Lisa's -/
def charlotte_money : ℕ := lisa_money / 2

/-- The total money they have -/
def total_money : ℕ := patricia_money + lisa_money + charlotte_money

/-- The additional amount needed to buy the card -/
def additional_money_needed : ℕ := card_cost - total_money

theorem baseball_card_purchase :
  additional_money_needed = 49 := by
  sorry

end baseball_card_purchase_l2607_260700


namespace cory_needs_78_dollars_l2607_260722

/-- The amount of additional money Cory needs to buy two packs of candies -/
def additional_money_needed (initial_money : ℚ) (candy_pack_cost : ℚ) (num_packs : ℕ) : ℚ :=
  candy_pack_cost * num_packs - initial_money

/-- Theorem stating that Cory needs $78 more to buy two packs of candies -/
theorem cory_needs_78_dollars :
  additional_money_needed 20 49 2 = 78 := by
  sorry

end cory_needs_78_dollars_l2607_260722


namespace article_price_proof_l2607_260723

/-- The normal price of an article before discounts -/
def normal_price : ℝ := 150

/-- The final price after discounts -/
def final_price : ℝ := 108

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

theorem article_price_proof :
  normal_price * (1 - discount1) * (1 - discount2) = final_price := by
  sorry

end article_price_proof_l2607_260723


namespace evaluate_expression_l2607_260703

theorem evaluate_expression (b : ℕ) (h : b = 4) : b^3 * b^4 * b^2 = 262144 := by
  sorry

end evaluate_expression_l2607_260703


namespace intersection_points_count_l2607_260710

/-- A line in a 2D plane, represented by coefficients a, b, and c in the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determine if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Determine if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2)

/-- The three lines given in the problem -/
def line1 : Line := ⟨2, -3, 4⟩
def line2 : Line := ⟨3, 4, 6⟩
def line3 : Line := ⟨6, -9, 8⟩

/-- The theorem to be proved -/
theorem intersection_points_count :
  (intersect line1 line2 ∧ intersect line2 line3 ∧ parallel line1 line3) ∧
  (∃! p : ℝ × ℝ, p.1 * line1.a + p.2 * line1.b = line1.c ∧ p.1 * line2.a + p.2 * line2.b = line2.c) ∧
  (∃! p : ℝ × ℝ, p.1 * line2.a + p.2 * line2.b = line2.c ∧ p.1 * line3.a + p.2 * line3.b = line3.c) :=
by sorry

end intersection_points_count_l2607_260710


namespace no_double_application_function_l2607_260783

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end no_double_application_function_l2607_260783


namespace gcd_sequence_is_one_l2607_260704

theorem gcd_sequence_is_one (n : ℕ) : 
  Nat.gcd ((7^n - 1) / 6) ((7^(n+1) - 1) / 6) = 1 := by
  sorry

end gcd_sequence_is_one_l2607_260704


namespace deposit_calculation_l2607_260762

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) :
  remaining_amount = 1350 ∧ deposit_percentage = 0.1 →
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 150 := by
sorry

end deposit_calculation_l2607_260762


namespace arithmetic_sequence_properties_l2607_260711

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.a 2 + seq.a 6 = 2)
    (h2 : seq.S 9 = -18) :
    (∀ n, seq.a n = 13 - 3*n) ∧
    (∀ n, |seq.S n| ≥ |seq.S 8|) ∧
    (|seq.S 8| = 4) := by
  sorry


end arithmetic_sequence_properties_l2607_260711


namespace power_outage_duration_is_three_l2607_260778

/-- The duration of the power outage in hours -/
def power_outage_duration : ℝ := 3

/-- The temperature rise rate during the power outage in degrees per hour -/
def temperature_rise_rate : ℝ := 8

/-- The temperature decrease rate when the air conditioner is on in degrees per hour -/
def temperature_decrease_rate : ℝ := 4

/-- The time taken by the air conditioner to restore the temperature in hours -/
def air_conditioner_duration : ℝ := 6

/-- Theorem stating that the power outage duration is 3 hours -/
theorem power_outage_duration_is_three :
  power_outage_duration = temperature_rise_rate⁻¹ * temperature_decrease_rate * air_conditioner_duration :=
by sorry

end power_outage_duration_is_three_l2607_260778


namespace amanda_to_kimberly_distance_l2607_260743

/-- The distance between two houses given a constant speed and time -/
def distance_between_houses (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: Amanda's house is 6 miles away from Kimberly's house -/
theorem amanda_to_kimberly_distance :
  distance_between_houses 2 3 = 6 := by
  sorry

end amanda_to_kimberly_distance_l2607_260743


namespace not_prime_sum_product_l2607_260785

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : d < c ∧ c < b ∧ b < a)
  (h_eq : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬ Nat.Prime (a * b + c * d) :=
by sorry

end not_prime_sum_product_l2607_260785


namespace dave_guitar_strings_l2607_260794

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 4

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 24

/-- The total number of guitar strings Dave needs to replace -/
def total_strings : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings = 576 := by sorry

end dave_guitar_strings_l2607_260794


namespace geometric_sequence_formula_l2607_260755

/-- A geometric sequence with given conditions -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 6) 
  (h_a5 : a 5 = 162) : 
  ∀ n : ℕ, a n = 2 * 3^(n - 1) :=
sorry

end geometric_sequence_formula_l2607_260755


namespace max_distinct_ten_blocks_l2607_260725

/-- Represents a binary string of length 10^4 -/
def BinaryString := Fin 10000 → Bool

/-- A k-block is a contiguous substring of length k -/
def kBlock (s : BinaryString) (start : Fin 10000) (k : Nat) : Fin k → Bool :=
  fun i => s ⟨start + i, sorry⟩

/-- Two k-blocks are identical if all their corresponding elements are equal -/
def kBlocksEqual (b1 b2 : Fin k → Bool) : Prop :=
  ∀ i : Fin k, b1 i = b2 i

/-- Count the number of distinct 3-blocks in a binary string -/
def distinctThreeBlocks (s : BinaryString) : Nat :=
  sorry

/-- Count the number of distinct 10-blocks in a binary string -/
def distinctTenBlocks (s : BinaryString) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem max_distinct_ten_blocks :
  ∀ s : BinaryString,
    distinctThreeBlocks s ≤ 7 →
    distinctTenBlocks s ≤ 504 :=
  sorry

end max_distinct_ten_blocks_l2607_260725


namespace valid_triples_l2607_260752

def is_valid_triple (p x y : ℕ) : Prop :=
  Nat.Prime p ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  ∃ a : ℕ, x^(p-1) + y = p^a ∧ 
  ∃ b : ℕ, x + y^(p-1) = p^b

def is_valid_triple_for_two (n i : ℕ) : Prop :=
  n > 0 ∧ n < 2^i

theorem valid_triples :
  ∀ p x y : ℕ, is_valid_triple p x y →
    ((p = 3 ∧ ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2))) ∨
     (p = 2 ∧ ∃ n i : ℕ, is_valid_triple_for_two n i ∧ x = n ∧ y = 2^i - n)) :=
sorry

end valid_triples_l2607_260752


namespace greatest_common_multiple_under_120_l2607_260712

theorem greatest_common_multiple_under_120 : ∃ (n : ℕ), n = 90 ∧ 
  (∀ m : ℕ, m < 120 → m % 10 = 0 → m % 15 = 0 → m ≤ n) :=
by sorry

end greatest_common_multiple_under_120_l2607_260712


namespace interactive_lines_count_l2607_260763

/-- Represents a four-digit number M with specific digit placement. -/
structure FourDigitNumber where
  a : ℕ  -- Thousands place
  b : ℕ  -- Hundreds place
  c : ℕ  -- Ones place
  h1 : c ≠ 0
  h2 : a < 10 ∧ b < 10 ∧ c < 10

/-- Calculates the value of M given its digit representation. -/
def M (n : FourDigitNumber) : ℕ :=
  1000 * n.a + 100 * n.b + 10 + n.c

/-- Calculates the value of N by moving the ones digit to the front. -/
def N (n : FourDigitNumber) : ℕ :=
  1000 * n.c + 100 * n.a + 10 * n.b + 1

/-- Defines the function F(M) = (M + N) / 11. -/
def F (n : FourDigitNumber) : ℚ :=
  (M n + N n : ℚ) / 11

/-- Predicate for the interactive line condition. -/
def IsInteractiveLine (n : FourDigitNumber) : Prop :=
  n.c = n.a + n.b

/-- The main theorem stating the number of interactive lines satisfying the condition. -/
theorem interactive_lines_count :
  (∃ (S : Finset FourDigitNumber),
    S.card = 8 ∧
    (∀ n ∈ S, IsInteractiveLine n ∧ ∃ k : ℕ, F n = 6 * k) ∧
    (∀ n : FourDigitNumber, IsInteractiveLine n → (∃ k : ℕ, F n = 6 * k) → n ∈ S)) :=
  sorry


end interactive_lines_count_l2607_260763


namespace remainder_theorem_l2607_260764

def dividend (b x : ℝ) : ℝ := 12 * x^3 - 9 * x^2 + b * x + 8
def divisor (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

theorem remainder_theorem (b : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, dividend b x = divisor x * q x + 10) ↔ b = -31/3 := by
  sorry

end remainder_theorem_l2607_260764


namespace alien_minerals_count_l2607_260727

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The number of minerals collected by the alien --/
def alienMinerals : ℕ := base7ToBase10 3 2 1

theorem alien_minerals_count :
  alienMinerals = 162 := by sorry

end alien_minerals_count_l2607_260727


namespace double_acute_angle_range_l2607_260793

theorem double_acute_angle_range (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  0 < 2 * α ∧ 2 * α < Real.pi := by
  sorry

end double_acute_angle_range_l2607_260793


namespace deepak_present_age_l2607_260791

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age. -/
theorem deepak_present_age 
  (ratio_rahul : ℕ) 
  (ratio_deepak : ℕ) 
  (rahul_future_age : ℕ) 
  (years_to_future : ℕ) :
  ratio_rahul = 4 →
  ratio_deepak = 3 →
  rahul_future_age = 26 →
  years_to_future = 10 →
  ∃ (x : ℕ), 
    ratio_rahul * x + years_to_future = rahul_future_age ∧
    ratio_deepak * x = 12 :=
by sorry

end deepak_present_age_l2607_260791


namespace geometric_sequence_problem_l2607_260786

theorem geometric_sequence_problem (a : ℝ) (h : a > 0) :
  let r : ℝ := 1/2
  let n : ℕ := 6
  let sum : ℝ := a * (1 - r^n) / (1 - r)
  sum = 189 → a * r = 48 := by sorry

end geometric_sequence_problem_l2607_260786


namespace dispatch_plans_count_l2607_260731

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students --/
def total_students : ℕ := 6

/-- The number of students participating in the campaign --/
def participating_students : ℕ := 4

/-- The number of students participating on Sunday --/
def sunday_students : ℕ := 2

/-- The number of students participating on Friday --/
def friday_students : ℕ := 1

/-- The number of students participating on Saturday --/
def saturday_students : ℕ := 1

theorem dispatch_plans_count :
  (choose total_students sunday_students) *
  (choose (total_students - sunday_students) friday_students) *
  (choose (total_students - sunday_students - friday_students) saturday_students) = 180 := by
  sorry

end dispatch_plans_count_l2607_260731


namespace arithmetic_sequence_2023rd_term_l2607_260705

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2023rd_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p 6 2 = p + 6)
  (h2 : arithmetic_sequence p 6 3 = 4*p - q)
  (h3 : arithmetic_sequence p 6 4 = 4*p + q) :
  arithmetic_sequence p 6 2023 = 12137 := by
sorry

end arithmetic_sequence_2023rd_term_l2607_260705


namespace decimal_to_binary_38_l2607_260719

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
sorry

end decimal_to_binary_38_l2607_260719


namespace perfect_square_condition_l2607_260717

/-- A polynomial of the form ax^2 + bx + c is a perfect square trinomial if and only if
    there exist real numbers p and q such that ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem stating the condition for the given polynomial to be a perfect square trinomial -/
theorem perfect_square_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (-(k-1)) 9 ↔ k = 13 ∨ k = -11 := by
  sorry


end perfect_square_condition_l2607_260717


namespace max_songs_is_56_l2607_260753

/-- Calculates the maximum number of songs that can be played given the specified conditions -/
def max_songs_played (short_songs : ℕ) (long_songs : ℕ) (short_duration : ℕ) (long_duration : ℕ) (total_time : ℕ) : ℕ :=
  let time_for_short := min (short_songs * short_duration) total_time
  let remaining_time := total_time - time_for_short
  let short_count := time_for_short / short_duration
  let long_count := remaining_time / long_duration
  short_count + long_count

/-- Theorem stating that the maximum number of songs that can be played is 56 -/
theorem max_songs_is_56 : 
  max_songs_played 50 50 3 5 (3 * 60) = 56 := by
  sorry

end max_songs_is_56_l2607_260753


namespace negation_of_no_red_cards_negation_equivalent_to_some_red_cards_l2607_260766

-- Define the universe of cards
variable (U : Type)

-- Define the property of being a red card
variable (red : U → Prop)

-- Define the property of being in the deck
variable (in_deck : U → Prop)

-- Statement to be proven
theorem negation_of_no_red_cards (h : ¬∃ x, red x ∧ in_deck x) :
  ¬∀ x, red x → ¬in_deck x :=
sorry

-- Proof that the negation is equivalent to "Some red cards are in this deck"
theorem negation_equivalent_to_some_red_cards :
  (¬∀ x, red x → ¬in_deck x) ↔ (∃ x, red x ∧ in_deck x) :=
sorry

end negation_of_no_red_cards_negation_equivalent_to_some_red_cards_l2607_260766


namespace line_graph_most_suitable_l2607_260750

/-- Represents types of graphs --/
inductive GraphType
  | Bar
  | Pie
  | Line

/-- Represents a geographical direction --/
inductive Direction
  | West
  | East

/-- Represents the characteristics of terrain elevation --/
structure TerrainElevation where
  higher : Direction
  lower : Direction

/-- Represents the requirement for visual representation --/
structure VisualRepresentation where
  showChanges : Bool
  alongLatitude : Bool

/-- Determines the most suitable graph type for representing elevation changes --/
def mostSuitableGraphType (terrain : TerrainElevation) (requirement : VisualRepresentation) : GraphType :=
  sorry

/-- Theorem stating that a line graph is the most suitable for the given conditions --/
theorem line_graph_most_suitable 
  (terrain : TerrainElevation)
  (requirement : VisualRepresentation)
  (h1 : terrain.higher = Direction.West)
  (h2 : terrain.lower = Direction.East)
  (h3 : requirement.showChanges = true)
  (h4 : requirement.alongLatitude = true) :
  mostSuitableGraphType terrain requirement = GraphType.Line :=
  sorry

end line_graph_most_suitable_l2607_260750


namespace miju_handshakes_l2607_260702

/-- Calculate the total number of handshakes in a group where each person shakes hands with every other person exactly once. -/
def totalHandshakes (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The problem statement -/
theorem miju_handshakes :
  totalHandshakes 12 = 66 := by
  sorry

end miju_handshakes_l2607_260702


namespace cylinder_height_l2607_260726

/-- The height of a cylinder given its base perimeter and side surface diagonal --/
theorem cylinder_height (base_perimeter : ℝ) (diagonal : ℝ) (h : base_perimeter = 6 ∧ diagonal = 10) :
  ∃ (height : ℝ), height = 8 ∧ height ^ 2 + base_perimeter ^ 2 = diagonal ^ 2 := by
  sorry

end cylinder_height_l2607_260726


namespace allison_wins_prob_l2607_260756

/-- Represents a die with a fixed number of faces -/
structure Die where
  faces : List Nat

/-- Allison's die always shows 5 -/
def allison_die : Die := ⟨[5, 5, 5, 5, 5, 5]⟩

/-- Brian's die has faces numbered 1, 2, 3, 4, 4, 5, 5, and 6 -/
def brian_die : Die := ⟨[1, 2, 3, 4, 4, 5, 5, 6]⟩

/-- Noah's die has faces numbered 2, 2, 6, 6, 3, 3, 7, and 7 -/
def noah_die : Die := ⟨[2, 2, 6, 6, 3, 3, 7, 7]⟩

/-- Calculate the probability of rolling less than a given number on a die -/
def prob_less_than (d : Die) (n : Nat) : Rat :=
  (d.faces.filter (· < n)).length / d.faces.length

/-- The probability that Allison's roll is greater than both Brian's and Noah's -/
theorem allison_wins_prob : 
  prob_less_than brian_die 5 * prob_less_than noah_die 5 = 5 / 16 := by
  sorry


end allison_wins_prob_l2607_260756


namespace set_size_from_averages_l2607_260740

theorem set_size_from_averages (S : Finset ℝ) (sum : ℝ) (n : ℕ) :
  sum = S.sum (λ x => x) →
  n = S.card →
  sum / n = 6.2 →
  (sum + 7) / n = 6.9 →
  n = 10 := by
  sorry

end set_size_from_averages_l2607_260740


namespace triangle_vector_intersection_l2607_260774

/-- Given a triangle XYZ with points M, N, and Q satisfying specific conditions,
    prove that Q can be expressed as a linear combination of X, Y, and Z with specific coefficients. -/
theorem triangle_vector_intersection (X Y Z M N Q : ℝ × ℝ) : 
  (∃ (k : ℝ), M = k • Z + (1 - k) • Y ∧ k = 1/5) →  -- M lies on YZ extended
  (∃ (l : ℝ), N = l • X + (1 - l) • Z ∧ l = 3/5) →  -- N lies on XZ
  (∃ (s t : ℝ), Q = s • Y + (1 - s) • N ∧ Q = t • X + (1 - t) • M) →  -- Q is intersection of YN and XM
  Q = (12/23) • X + (3/23) • Y + (8/23) • Z :=
by sorry

end triangle_vector_intersection_l2607_260774


namespace largest_prime_factors_difference_l2607_260776

def number : Nat := 219257

theorem largest_prime_factors_difference (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ number ∧ q ∣ number ∧
  ∀ r, Nat.Prime r → r ∣ number → r ≤ p ∧ r ≤ q →
  p - q = 144 := by
  sorry

end largest_prime_factors_difference_l2607_260776


namespace x_squared_minus_y_squared_equals_27_l2607_260789

theorem x_squared_minus_y_squared_equals_27
  (x y : ℝ)
  (h1 : y + 6 = (x - 3)^2)
  (h2 : x + 6 = (y - 3)^2)
  (h3 : x ≠ y) :
  x^2 - y^2 = 27 := by
sorry

end x_squared_minus_y_squared_equals_27_l2607_260789


namespace hair_extension_ratio_l2607_260757

theorem hair_extension_ratio : 
  let initial_length : ℕ := 18
  let final_length : ℕ := 36
  (final_length : ℚ) / (initial_length : ℚ) = 2 := by
  sorry

end hair_extension_ratio_l2607_260757


namespace triangle_parallelogram_altitude_l2607_260716

theorem triangle_parallelogram_altitude (b h_t h_p : ℝ) : 
  b > 0 →  -- Ensure base is positive
  h_t = 200 →  -- Given altitude of triangle
  (1 / 2) * b * h_t = b * h_p →  -- Equal areas
  h_p = 100 := by
sorry

end triangle_parallelogram_altitude_l2607_260716


namespace days_without_class_total_course_days_course_duration_proof_l2607_260739

/- Define the parameters of the problem -/
def total_hours : ℕ := 30
def class_duration : ℕ := 1
def afternoons_without_class : ℕ := 20
def mornings_without_class : ℕ := 18

/- Define the theorems to be proved -/
theorem days_without_class : ℕ := by sorry

theorem total_course_days : ℕ := by sorry

/- Main theorem combining both results -/
theorem course_duration_proof :
  (days_without_class = 4) ∧ (total_course_days = 34) := by
  sorry

end days_without_class_total_course_days_course_duration_proof_l2607_260739


namespace petya_spent_less_than_5000_l2607_260769

/-- Represents the purchase of a book -/
inductive Purchase
  | Expensive (cost : ℕ)
  | Cheap (cost : ℕ)

/-- Represents Petya's shopping process -/
structure ShoppingProcess where
  initial_money : ℕ
  purchases : List Purchase
  final_coins : ℕ

/-- Checks if a shopping process is valid according to the problem conditions -/
def is_valid_process (p : ShoppingProcess) : Prop :=
  p.initial_money % 100 = 0 ∧
  (∀ purchase ∈ p.purchases, match purchase with
    | Purchase.Expensive cost => cost ≥ 100
    | Purchase.Cheap cost => cost < 100
  ) ∧
  p.final_coins < 100 ∧
  2 * (p.initial_money - p.final_coins) = p.initial_money

/-- Calculates the total amount spent on books -/
def total_spent (p : ShoppingProcess) : ℕ :=
  p.initial_money - p.final_coins

/-- Theorem stating that Petya could not have spent at least 5000 rubles on books -/
theorem petya_spent_less_than_5000 (p : ShoppingProcess) :
  is_valid_process p → total_spent p < 5000 := by
  sorry

end petya_spent_less_than_5000_l2607_260769


namespace quadrilateral_property_implication_l2607_260795

-- Define a quadrilateral type
structure Quadrilateral :=
  (A B C D : Point)

-- Define the three properties
def diagonals_perpendicular (q : Quadrilateral) : Prop := sorry

def inscribed_in_circle (q : Quadrilateral) : Prop := sorry

def perpendicular_through_intersection (q : Quadrilateral) : Prop := sorry

-- Main theorem
theorem quadrilateral_property_implication (q : Quadrilateral) :
  (diagonals_perpendicular q ∧ inscribed_in_circle q) ∨
  (diagonals_perpendicular q ∧ perpendicular_through_intersection q) ∨
  (inscribed_in_circle q ∧ perpendicular_through_intersection q) →
  diagonals_perpendicular q ∧ inscribed_in_circle q ∧ perpendicular_through_intersection q :=
by sorry

end quadrilateral_property_implication_l2607_260795


namespace sphere_volume_from_cross_section_l2607_260772

/-- Given a sphere with a circular cross-section of radius 4 and the distance
    from the sphere's center to the center of the cross-section is 3,
    prove that the volume of the sphere is (500/3)π. -/
theorem sphere_volume_from_cross_section (r : ℝ) (h : ℝ) :
  r^2 = 4^2 + 3^2 →
  (4 / 3) * π * r^3 = (500 / 3) * π := by
sorry

end sphere_volume_from_cross_section_l2607_260772


namespace final_pressure_is_three_l2607_260797

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
structure GasState where
  pressure : ℝ
  volume : ℝ
  constant : ℝ
  h : pressure * volume = constant

/-- The initial state of the hydrogen gas -/
def initial_state : GasState :=
  { pressure := 6
  , volume := 3
  , constant := 18
  , h := by sorry }

/-- The final state of the hydrogen gas after transfer -/
def final_state : GasState :=
  { pressure := 3
  , volume := 6
  , constant := 18
  , h := by sorry }

/-- Theorem stating that the final pressure is 3 kPa -/
theorem final_pressure_is_three :
  final_state.pressure = 3 :=
by sorry

end final_pressure_is_three_l2607_260797


namespace element_in_set_l2607_260714

theorem element_in_set (a b : Type) : a ∈ ({a, b} : Set Type) := by
  sorry

end element_in_set_l2607_260714


namespace equation_solution_l2607_260706

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ (x = 8) := by
  sorry

end equation_solution_l2607_260706


namespace A_B_mutually_exclusive_A_C_mutually_exclusive_C_D_complementary_l2607_260748

-- Define the sample space for a six-sided die
def Ω : Type := Fin 6

-- Define the probability measure
variable (P : Ω → ℝ)

-- Assume the die is fair
axiom fair_die : ∀ x : Ω, P x = 1 / 6

-- Define events
def A (x : Ω) : Prop := x.val + 1 = 4
def B (x : Ω) : Prop := x.val % 2 = 0
def C (x : Ω) : Prop := x.val + 1 < 4
def D (x : Ω) : Prop := x.val + 1 > 3

-- Theorem statements
theorem A_B_mutually_exclusive : ∀ x : Ω, ¬(A x ∧ B x) := by sorry

theorem A_C_mutually_exclusive : ∀ x : Ω, ¬(A x ∧ C x) := by sorry

theorem C_D_complementary : ∀ x : Ω, C x ↔ ¬(D x) := by sorry

end A_B_mutually_exclusive_A_C_mutually_exclusive_C_D_complementary_l2607_260748


namespace website_earnings_theorem_l2607_260735

/-- Calculates the earnings for a website over a week given the following conditions:
  - The website gets a fixed number of visitors per day for the first 6 days
  - On the 7th day, it gets twice as many visitors as the previous 6 days combined
  - There is a fixed earning per visit -/
def websiteEarnings (dailyVisitors : ℕ) (earningsPerVisit : ℚ) : ℚ :=
  let firstSixDaysVisits : ℕ := 6 * dailyVisitors
  let seventhDayVisits : ℕ := 2 * firstSixDaysVisits
  let totalVisits : ℕ := firstSixDaysVisits + seventhDayVisits
  (totalVisits : ℚ) * earningsPerVisit

/-- Theorem stating that under the given conditions, the website earnings for the week are $18 -/
theorem website_earnings_theorem :
  websiteEarnings 100 (1 / 100) = 18 := by
  sorry


end website_earnings_theorem_l2607_260735


namespace orange_basket_problem_l2607_260788

/-- 
Given:
- When 2 oranges are put in each basket, 4 oranges are left over.
- When 5 oranges are put in each basket, 1 basket is left over.

Prove that the number of baskets is 3 and the number of oranges is 10.
-/
theorem orange_basket_problem (b o : ℕ) 
  (h1 : 2 * b + 4 = o) 
  (h2 : 5 * (b - 1) = o) : 
  b = 3 ∧ o = 10 := by
  sorry


end orange_basket_problem_l2607_260788


namespace circle_equation_through_points_l2607_260744

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) -/
theorem circle_equation_through_points :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 4*x - 6*y = 0) ↔
  ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end circle_equation_through_points_l2607_260744


namespace billy_video_count_l2607_260790

theorem billy_video_count 
  (suggestions_per_round : ℕ) 
  (num_rounds : ℕ) 
  (final_pick : ℕ) :
  suggestions_per_round = 15 →
  num_rounds = 5 →
  final_pick = 5 →
  suggestions_per_round * num_rounds - (suggestions_per_round - final_pick) = 65 :=
by
  sorry

end billy_video_count_l2607_260790


namespace sum_of_coefficients_l2607_260760

theorem sum_of_coefficients (x : ℝ) :
  ∃ (A B C D E : ℝ),
    125 * x^3 + 64 = (A * x + B) * (C * x^2 + D * x + E) ∧
    A + B + C + D + E = 30 := by
  sorry

end sum_of_coefficients_l2607_260760


namespace perpendicular_vectors_acute_angle_vectors_l2607_260742

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

/-- Theorem for part 1 -/
theorem perpendicular_vectors (m : ℝ) : 
  (((1/2 : ℝ) • a.1 + b.1, (1/2 : ℝ) • a.2 + b.2) • (a.1 + m * b.1, a.2 + m * b.2) = 0) ↔ 
  (m = -5/12) :=
sorry

/-- Theorem for part 2 -/
theorem acute_angle_vectors (m : ℝ) :
  (((1/2 : ℝ) • a.1 + b.1, (1/2 : ℝ) • a.2 + b.2) • (a.1 + m * b.1, a.2 + m * b.2) > 0 ∧
   ((1/2 : ℝ) • a.1 + b.1) / ((1/2 : ℝ) • a.2 + b.2) ≠ (a.1 + m * b.1) / (a.2 + m * b.2)) ↔
  (m > -5/12 ∧ m ≠ 2) :=
sorry

end perpendicular_vectors_acute_angle_vectors_l2607_260742


namespace floor_abs_neg_57_6_l2607_260747

theorem floor_abs_neg_57_6 : ⌊|(-57.6 : ℝ)|⌋ = 57 := by sorry

end floor_abs_neg_57_6_l2607_260747


namespace furniture_making_l2607_260799

theorem furniture_making (total_wood pieces_per_table pieces_per_chair chairs_made : ℕ) 
  (h1 : total_wood = 672)
  (h2 : pieces_per_table = 12)
  (h3 : pieces_per_chair = 8)
  (h4 : chairs_made = 48) :
  (total_wood - chairs_made * pieces_per_chair) / pieces_per_table = 24 := by
  sorry

end furniture_making_l2607_260799


namespace vals_money_value_is_38_80_l2607_260770

/-- Calculates the total value of Val's money in USD -/
def valsMoneyValue (initialNickels : ℕ) (dimesToNickelsRatio : ℕ) (quartersToDimesRatio : ℕ) 
  (newNickelsMultiplier : ℕ) (canadianNickelRatio : ℚ) (exchangeRate : ℚ) : ℚ :=
  let initialDimes := initialNickels * dimesToNickelsRatio
  let initialQuarters := initialDimes * quartersToDimesRatio
  let newNickels := initialNickels * newNickelsMultiplier
  let canadianNickels := (newNickels : ℚ) * canadianNickelRatio
  let usNickels := (newNickels : ℚ) - canadianNickels
  let initialValue := (initialNickels : ℚ) * (5 / 100) + (initialDimes : ℚ) * (10 / 100) + (initialQuarters : ℚ) * (25 / 100)
  let newUsNickelsValue := usNickels * (5 / 100)
  let canadianNickelsValue := canadianNickels * (5 / 100) * exchangeRate
  initialValue + newUsNickelsValue + canadianNickelsValue

/-- Theorem stating that Val's money value is $38.80 given the problem conditions -/
theorem vals_money_value_is_38_80 :
  valsMoneyValue 20 3 2 2 (1/2) (4/5) = 388/10 := by
  sorry

end vals_money_value_is_38_80_l2607_260770


namespace johns_house_nails_l2607_260715

/-- Calculates the total number of nails needed for a house wall -/
def total_nails (large_planks : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  large_planks * nails_per_plank + additional_nails

/-- Proves that John needs 229 nails for the house wall -/
theorem johns_house_nails :
  total_nails 13 17 8 = 229 := by
  sorry

end johns_house_nails_l2607_260715


namespace triangle_properties_l2607_260796

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions and the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.c = abc.a)
  (h2 : abc.c = Real.sqrt 3)
  (h3 : Real.sin abc.B ^ 2 = 2 * Real.sin abc.A * Real.sin abc.C) :
  Real.cos abc.B = 0 ∧ (1/2 * abc.a * abc.c * Real.sin abc.B = 3/2) := by
  sorry

end triangle_properties_l2607_260796
