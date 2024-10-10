import Mathlib

namespace absolute_value_simplification_l4116_411616

theorem absolute_value_simplification : |(-5^2 + 7 - 3)| = 21 := by
  sorry

end absolute_value_simplification_l4116_411616


namespace intersection_of_M_and_N_l4116_411605

def I : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_of_M_and_N : M ∩ N = {0} := by sorry

end intersection_of_M_and_N_l4116_411605


namespace geometric_progression_with_means_l4116_411622

theorem geometric_progression_with_means
  (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  let q := (b / a) ^ (1 / (n + 1 : ℝ))
  ∀ k : ℕ, ∃ r : ℝ, a * q ^ k = a * (b / a) ^ (k / (n + 1 : ℝ)) :=
by sorry

end geometric_progression_with_means_l4116_411622


namespace A_and_D_mutually_exclusive_but_not_complementary_l4116_411626

-- Define the sample space for a fair six-sided die
def DieOutcome := Fin 6

-- Define the events
def event_A (n : DieOutcome) : Prop := n.val % 2 = 1
def event_B (n : DieOutcome) : Prop := n.val % 2 = 0
def event_C (n : DieOutcome) : Prop := n.val % 2 = 0
def event_D (n : DieOutcome) : Prop := n.val = 2 ∨ n.val = 4

-- Define mutual exclusivity
def mutually_exclusive (e1 e2 : DieOutcome → Prop) : Prop :=
  ∀ n : DieOutcome, ¬(e1 n ∧ e2 n)

-- Define complementary events
def complementary (e1 e2 : DieOutcome → Prop) : Prop :=
  ∀ n : DieOutcome, e1 n ↔ ¬e2 n

-- Theorem to prove
theorem A_and_D_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_A event_D ∧ ¬complementary event_A event_D :=
sorry

end A_and_D_mutually_exclusive_but_not_complementary_l4116_411626


namespace vertex_of_given_function_l4116_411666

/-- A quadratic function of the form y = a(x - h)^2 + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The vertex of a quadratic function -/
def vertex (f : QuadraticFunction) : ℝ × ℝ := (f.h, f.k)

/-- The given quadratic function y = -2(x+1)^2 + 5 -/
def given_function : QuadraticFunction := ⟨-2, -1, 5⟩

theorem vertex_of_given_function :
  vertex given_function = (-1, 5) := by sorry

end vertex_of_given_function_l4116_411666


namespace money_sharing_problem_l4116_411609

theorem money_sharing_problem (amanda_ratio ben_ratio carlos_ratio : ℕ) 
  (ben_share : ℕ) (total : ℕ) : 
  amanda_ratio = 3 → 
  ben_ratio = 5 → 
  carlos_ratio = 8 → 
  ben_share = 25 → 
  total = amanda_ratio * (ben_share / ben_ratio) + 
          ben_share + 
          carlos_ratio * (ben_share / ben_ratio) → 
  total = 80 := by
sorry

end money_sharing_problem_l4116_411609


namespace arithmetic_evaluation_l4116_411696

theorem arithmetic_evaluation : 6 + 4 / 2 = 8 := by
  sorry

end arithmetic_evaluation_l4116_411696


namespace min_value_x_one_minus_y_l4116_411634

theorem min_value_x_one_minus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 4 * x^2 + 4 * x * y + y^2 + 2 * x + y - 6 = 0) :
  ∀ z : ℝ, z > 0 → ∀ w : ℝ, w > 0 →
  4 * z^2 + 4 * z * w + w^2 + 2 * z + w - 6 = 0 →
  x * (1 - y) ≤ z * (1 - w) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  4 * a^2 + 4 * a * b + b^2 + 2 * a + b - 6 = 0 ∧
  a * (1 - b) = -1/8 :=
sorry

end min_value_x_one_minus_y_l4116_411634


namespace arithmetic_calculations_l4116_411658

theorem arithmetic_calculations : 
  (2 - 7 * (-3) + 10 + (-2) = 31) ∧ 
  (-1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14) := by
sorry

end arithmetic_calculations_l4116_411658


namespace remainder_3_pow_2003_mod_13_l4116_411613

theorem remainder_3_pow_2003_mod_13 :
  ∃ k : ℤ, 3^2003 = 13 * k + 9 :=
by
  sorry

end remainder_3_pow_2003_mod_13_l4116_411613


namespace complement_connected_if_not_connected_l4116_411660

-- Define a graph
def Graph := Type

-- Define the property of being connected
def is_connected (G : Graph) : Prop := sorry

-- Define the complement of a graph
def complement (G : Graph) : Graph := sorry

-- Theorem statement
theorem complement_connected_if_not_connected (G : Graph) :
  ¬(is_connected G) → is_connected (complement G) := by sorry

end complement_connected_if_not_connected_l4116_411660


namespace pat_shark_photo_profit_l4116_411685

/-- Calculates the expected profit for Pat's shark photo hunting trip. -/
theorem pat_shark_photo_profit :
  let photo_earnings : ℕ → ℚ := λ n => 15 * n
  let sharks_per_hour : ℕ := 6
  let fuel_cost_per_hour : ℚ := 50
  let hunting_hours : ℕ := 5
  let total_sharks : ℕ := sharks_per_hour * hunting_hours
  let total_earnings : ℚ := photo_earnings total_sharks
  let total_fuel_cost : ℚ := fuel_cost_per_hour * hunting_hours
  let profit : ℚ := total_earnings - total_fuel_cost
  profit = 200 := by
sorry


end pat_shark_photo_profit_l4116_411685


namespace two_digit_number_is_30_l4116_411684

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := { n : ℕ × ℕ // n.1 < 10 ∧ n.2 < 10 }

/-- Converts a two-digit number to its decimal representation -/
def to_decimal (n : TwoDigitNumber) : ℚ :=
  n.val.1 * 10 + n.val.2

/-- Represents a repeating decimal of the form 2.xy̅ -/
def repeating_decimal (n : TwoDigitNumber) : ℚ :=
  2 + (to_decimal n) / 99

/-- The main theorem stating that the two-digit number satisfying the equation is 30 -/
theorem two_digit_number_is_30 :
  ∃ (n : TwoDigitNumber), 
    75 * (repeating_decimal n - (2 + (to_decimal n) / 100)) = 2 ∧
    to_decimal n = 30 := by
  sorry

end two_digit_number_is_30_l4116_411684


namespace problem_solution_l4116_411631

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c + 4 - d) → d = 17/4 := by
  sorry

end problem_solution_l4116_411631


namespace geometric_sequence_product_l4116_411625

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a_4 * a_6 = 5, prove that a_2 * a_3 * a_7 * a_8 = 25 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : IsGeometricSequence a) 
    (h_prod : a 4 * a 6 = 5) : a 2 * a 3 * a 7 * a 8 = 25 := by
  sorry

end geometric_sequence_product_l4116_411625


namespace irrational_among_options_l4116_411654

theorem irrational_among_options : 
  (¬ (∃ (a b : ℤ), -Real.sqrt 3 = (a : ℚ) / (b : ℚ) ∧ b ≠ 0)) ∧
  (∃ (a b : ℤ), (-2 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) ∧
  (∃ (a b : ℤ), (0.1010 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) ∧
  (∃ (a b : ℤ), (1/3 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
by sorry

end irrational_among_options_l4116_411654


namespace red_other_side_probability_l4116_411619

structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

def box : Finset Card := sorry

axiom box_size : box.card = 8

axiom black_both_sides : (box.filter (fun c => !c.side1 ∧ !c.side2)).card = 4

axiom black_red_sides : (box.filter (fun c => (c.side1 ∧ !c.side2) ∨ (!c.side1 ∧ c.side2))).card = 2

axiom red_both_sides : (box.filter (fun c => c.side1 ∧ c.side2)).card = 2

def observe_red (c : Card) : Bool := c.side1 ∨ c.side2

def other_side_red (c : Card) : Bool := c.side1 ∧ c.side2

theorem red_other_side_probability :
  (box.filter (fun c => other_side_red c)).card / (box.filter (fun c => observe_red c)).card = 2 / 3 := by
  sorry

end red_other_side_probability_l4116_411619


namespace y_value_proof_l4116_411603

theorem y_value_proof (y : ℝ) (h : 9 / (y^3) = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end y_value_proof_l4116_411603


namespace trigonometric_equation_solution_l4116_411642

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 
    4 * sin x * cos (π/2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3*π/2 - x) * cos (π + x) = 1 ↔ 
    (∃ k : ℤ, x = arctan (1/3) + π * k) ∨ (∃ n : ℤ, x = π/4 + π * n) :=
by sorry

end trigonometric_equation_solution_l4116_411642


namespace symmetric_point_coordinates_l4116_411628

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to x-axis -/
def symmetric_x (A B : Point) : Prop :=
  B.x = A.x ∧ B.y = -A.y

theorem symmetric_point_coordinates :
  let A : Point := ⟨2, -1⟩
  ∀ B : Point, symmetric_x A B → B = ⟨2, 1⟩ := by
  sorry

end symmetric_point_coordinates_l4116_411628


namespace calendar_sum_property_l4116_411652

/-- Represents a monthly calendar with dates behind letters --/
structure Calendar where
  x : ℕ  -- The date behind C
  dateA : ℕ := x + 1
  dateB : ℕ := x + 13
  dateP : ℕ := x + 14
  dateQ : ℕ
  dateR : ℕ
  dateS : ℕ
  dateT : ℕ

/-- The letter P is the only one that satisfies the condition --/
theorem calendar_sum_property (cal : Calendar) :
  (cal.x + cal.dateP = cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateQ ≠ cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateR ≠ cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateS ≠ cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateT ≠ cal.dateA + cal.dateB) :=
by sorry

end calendar_sum_property_l4116_411652


namespace combined_shoe_size_l4116_411682

theorem combined_shoe_size (jasmine_size : ℝ) (alexa_size : ℝ) (clara_size : ℝ) (molly_shoe_size : ℝ) (molly_sandal_size : ℝ) :
  jasmine_size = 7 →
  alexa_size = 2 * jasmine_size →
  clara_size = 3 * jasmine_size →
  molly_shoe_size = 1.5 * jasmine_size →
  molly_sandal_size = molly_shoe_size - 0.5 →
  jasmine_size + alexa_size + clara_size + molly_shoe_size + molly_sandal_size = 62.5 := by
  sorry

end combined_shoe_size_l4116_411682


namespace wardrobe_probability_l4116_411620

def num_shirts : ℕ := 5
def num_shorts : ℕ := 6
def num_socks : ℕ := 7
def num_selected : ℕ := 4

def total_articles : ℕ := num_shirts + num_shorts + num_socks

theorem wardrobe_probability :
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 1 * Nat.choose num_socks 1) /
  (Nat.choose total_articles num_selected) = 7 / 51 :=
by sorry

end wardrobe_probability_l4116_411620


namespace dodecagon_diagonals_l4116_411676

/-- Number of diagonals in a convex polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l4116_411676


namespace season_games_calculation_l4116_411699

/-- Represents the number of games played by a team in a season -/
def total_games : ℕ := 125

/-- Represents the number of games in the first part of the season -/
def first_games : ℕ := 100

/-- Represents the win percentage for the first part of the season -/
def first_win_percentage : ℚ := 75 / 100

/-- Represents the win percentage for the remaining games -/
def remaining_win_percentage : ℚ := 50 / 100

/-- Represents the overall win percentage for the entire season -/
def overall_win_percentage : ℚ := 70 / 100

theorem season_games_calculation :
  let remaining_games := total_games - first_games
  (first_win_percentage * first_games + remaining_win_percentage * remaining_games) / total_games = overall_win_percentage :=
by sorry

end season_games_calculation_l4116_411699


namespace paper_strip_division_l4116_411636

theorem paper_strip_division (total_fraction : ℚ) (num_books : ℕ) : 
  total_fraction = 5/8 ∧ num_books = 5 → 
  total_fraction / num_books = 1/8 := by
  sorry

end paper_strip_division_l4116_411636


namespace expression_simplification_l4116_411639

theorem expression_simplification (x : ℝ) : 
  x - 3*(1+x) + 4*(1-x)^2 - 5*(1+3*x) = 4*x^2 - 25*x - 4 := by
sorry

end expression_simplification_l4116_411639


namespace no_integer_solution_2016_equation_l4116_411611

theorem no_integer_solution_2016_equation :
  ¬∃ (x y z : ℤ), (2016 : ℚ) = (x^2 + y^2 + z^2 : ℚ) / (x*y + y*z + z*x : ℚ) :=
by sorry

end no_integer_solution_2016_equation_l4116_411611


namespace remainder_theorem_l4116_411607

theorem remainder_theorem (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end remainder_theorem_l4116_411607


namespace complex_expression_equals_half_l4116_411691

theorem complex_expression_equals_half :
  |2 - Real.sqrt 2| - Real.sqrt (1/12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1/2 := by
  sorry

end complex_expression_equals_half_l4116_411691


namespace lisa_photos_l4116_411606

/-- The number of photos Lisa took this weekend -/
def total_photos (animal_photos flower_photos scenery_photos : ℕ) : ℕ :=
  animal_photos + flower_photos + scenery_photos

theorem lisa_photos : 
  ∀ (animal_photos flower_photos scenery_photos : ℕ),
    animal_photos = 10 →
    flower_photos = 3 * animal_photos →
    scenery_photos = flower_photos - 10 →
    total_photos animal_photos flower_photos scenery_photos = 60 := by
  sorry

#check lisa_photos

end lisa_photos_l4116_411606


namespace constant_k_value_l4116_411614

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, 
  -x^2 - (k + 12)*x - 8 = -(x - 2)*(x - 4) ↔ k = -18 :=
by
  sorry

end constant_k_value_l4116_411614


namespace alloy_mixture_l4116_411651

/-- The amount of alloy B mixed with alloy A -/
def amount_alloy_B : ℝ := 180

/-- The amount of alloy A -/
def amount_alloy_A : ℝ := 120

/-- The ratio of lead to tin in alloy A -/
def ratio_A : ℚ := 2 / 3

/-- The ratio of tin to copper in alloy B -/
def ratio_B : ℚ := 3 / 5

/-- The amount of tin in the new alloy -/
def amount_tin_new : ℝ := 139.5

theorem alloy_mixture :
  amount_alloy_B = 180 ∧
  (ratio_A * amount_alloy_A + ratio_B * amount_alloy_B) / (1 + ratio_A) = amount_tin_new :=
by sorry

end alloy_mixture_l4116_411651


namespace only_molality_can_be_calculated_l4116_411640

-- Define the given quantities
variable (mass_solute : ℝ)
variable (mass_solvent : ℝ)
variable (molar_mass_solute : ℝ)
variable (molar_mass_solvent : ℝ)

-- Define the quantitative descriptions
def can_calculate_molarity (mass_solute molar_mass_solute mass_solvent : ℝ) : Prop :=
  ∃ (volume_solution : ℝ), volume_solution > 0

def can_calculate_molality (mass_solute molar_mass_solute mass_solvent : ℝ) : Prop :=
  mass_solvent > 0 ∧ molar_mass_solute > 0

def can_calculate_density (mass_solute mass_solvent : ℝ) : Prop :=
  ∃ (volume_solution : ℝ), volume_solution > 0

-- Theorem statement
theorem only_molality_can_be_calculated
  (mass_solute mass_solvent molar_mass_solute molar_mass_solvent : ℝ) :
  can_calculate_molality mass_solute molar_mass_solute mass_solvent ∧
  ¬can_calculate_molarity mass_solute molar_mass_solute mass_solvent ∧
  ¬can_calculate_density mass_solute mass_solvent :=
sorry

end only_molality_can_be_calculated_l4116_411640


namespace time_to_eat_half_l4116_411633

/-- Represents the eating rate of a bird in terms of fraction of nuts eaten per hour -/
structure BirdRate where
  fraction : ℚ
  hours : ℚ

/-- Calculates the rate at which a bird eats nuts per hour -/
def eatRate (br : BirdRate) : ℚ :=
  br.fraction / br.hours

/-- Represents the rates of the three birds -/
structure BirdRates where
  crow : BirdRate
  sparrow : BirdRate
  parrot : BirdRate

/-- Calculates the combined eating rate of all three birds -/
def combinedRate (rates : BirdRates) : ℚ :=
  eatRate rates.crow + eatRate rates.sparrow + eatRate rates.parrot

/-- The main theorem stating the time taken to eat half the nuts -/
theorem time_to_eat_half (rates : BirdRates) 
  (h_crow : rates.crow = ⟨1/5, 4⟩) 
  (h_sparrow : rates.sparrow = ⟨1/3, 6⟩)
  (h_parrot : rates.parrot = ⟨1/4, 8⟩) : 
  (1/2) / combinedRate rates = 2880 / 788 := by
  sorry

end time_to_eat_half_l4116_411633


namespace sum_of_common_ratios_is_two_l4116_411694

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 2 if the difference of their
    third terms equals 3 times the difference of their second terms minus their first term. -/
theorem sum_of_common_ratios_is_two
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) (hk : k ≠ 0) :
  (k * p^2 - k * r^2 = 3 * (k * p - k * r) - k) →
  p + r = 2 :=
sorry

end sum_of_common_ratios_is_two_l4116_411694


namespace closest_integer_to_cube_root_80_l4116_411683

theorem closest_integer_to_cube_root_80 : 
  ∀ n : ℤ, |n - (80 : ℝ)^(1/3)| ≥ |4 - (80 : ℝ)^(1/3)| := by sorry

end closest_integer_to_cube_root_80_l4116_411683


namespace waiting_by_tree_only_random_l4116_411687

/-- Represents an idiom --/
inductive Idiom
  | CatchingTurtleInJar
  | WaitingByTreeForRabbit
  | RisingTideLiftAllBoats
  | FishingForMoonInWater

/-- Predicate to determine if an idiom describes a random event --/
def is_random_event (i : Idiom) : Prop :=
  match i with
  | Idiom.WaitingByTreeForRabbit => true
  | _ => false

/-- Theorem stating that "Waiting by a tree for a rabbit" is the only idiom
    among the given options that describes a random event --/
theorem waiting_by_tree_only_random :
  ∀ (i : Idiom), is_random_event i ↔ i = Idiom.WaitingByTreeForRabbit :=
by sorry

end waiting_by_tree_only_random_l4116_411687


namespace right_triangle_tan_b_l4116_411629

theorem right_triangle_tan_b (A B C : ℝ) (h1 : C = π / 2) (h2 : Real.cos A = 3 / 5) : 
  Real.tan B = 3 / 4 := by
  sorry

end right_triangle_tan_b_l4116_411629


namespace moving_circle_trajectory_l4116_411602

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4
def circle_N (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 100

-- Define the property of being externally tangent
def externally_tangent (x y R : ℝ) : Prop :=
  ∃ (x_m y_m : ℝ), circle_M x_m y_m ∧ (x - x_m)^2 + (y - y_m)^2 = (R + 2)^2

-- Define the property of being internally tangent
def internally_tangent (x y R : ℝ) : Prop :=
  ∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = (10 - R)^2

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y R : ℝ),
    externally_tangent x y R →
    internally_tangent x y R →
    x^2 / 36 + y^2 / 27 = 1 :=
by
  sorry

end moving_circle_trajectory_l4116_411602


namespace circle_locus_is_spherical_triangle_l4116_411648

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  vertex : Point3D
  isRightAngled : Bool

/-- The locus of circle centers touching the faces of a right-angled trihedral angle -/
def circleLocus (t : TrihedralAngle) (r : ℝ) : Set Point3D :=
  {p : Point3D | ∃ (c : Circle3D), c.radius = r ∧ 
    c.center = p ∧ 
    (c.center.x ≤ r ∧ c.center.y ≤ r ∧ c.center.z ≤ r) ∧
    (c.center.x ≥ 0 ∧ c.center.y ≥ 0 ∧ c.center.z ≥ 0) ∧
    (c.center.x ^ 2 + c.center.y ^ 2 + c.center.z ^ 2 = 2 * r ^ 2)}

theorem circle_locus_is_spherical_triangle (t : TrihedralAngle) (r : ℝ) 
  (h : t.isRightAngled = true) :
  circleLocus t r = {p : Point3D | 
    p.x ^ 2 + p.y ^ 2 + p.z ^ 2 = 2 * r ^ 2 ∧
    p.x ≤ r ∧ p.y ≤ r ∧ p.z ≤ r ∧
    p.x ≥ 0 ∧ p.y ≥ 0 ∧ p.z ≥ 0} := by
  sorry

end circle_locus_is_spherical_triangle_l4116_411648


namespace prob_fifth_six_given_two_sixes_l4116_411615

/-- Represents a six-sided die -/
inductive Die
| Fair
| Biased

/-- Probability of rolling a six for a given die -/
def prob_six (d : Die) : ℚ :=
  match d with
  | Die.Fair => 1/6
  | Die.Biased => 1/2

/-- Probability of rolling a number other than six for a given die -/
def prob_not_six (d : Die) : ℚ :=
  match d with
  | Die.Fair => 5/6
  | Die.Biased => 1/10

/-- Probability of rolling at least two sixes in four rolls for a given die -/
def prob_at_least_two_sixes (d : Die) : ℚ :=
  match d with
  | Die.Fair => 11/1296
  | Die.Biased => 11/16

/-- The main theorem -/
theorem prob_fifth_six_given_two_sixes (d : Die) : 
  (prob_at_least_two_sixes Die.Fair + prob_at_least_two_sixes Die.Biased) *
  (prob_six d * prob_at_least_two_sixes d) / 
  (prob_at_least_two_sixes Die.Fair + prob_at_least_two_sixes Die.Biased) = 325/656 :=
sorry

end prob_fifth_six_given_two_sixes_l4116_411615


namespace no_real_solution_for_equation_l4116_411653

theorem no_real_solution_for_equation :
  ¬ ∃ x : ℝ, (Real.sqrt (4 * x + 2) + 1) / Real.sqrt (8 * x + 10) = 2 / Real.sqrt 5 := by
  sorry

end no_real_solution_for_equation_l4116_411653


namespace descending_order_XYZ_l4116_411675

theorem descending_order_XYZ : ∀ (X Y Z : ℝ),
  X = 0.6 * 0.5 + 0.4 →
  Y = 0.6 * 0.5 / 0.4 →
  Z = 0.6 * 0.5 * 0.4 →
  Y > X ∧ X > Z :=
by
  sorry

end descending_order_XYZ_l4116_411675


namespace problem_solution_l4116_411621

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

theorem problem_solution (k : ℝ) (h : k > 0) :
  (∀ x ∈ Set.Ioo 0 4, f_deriv k x < 0) → k = 1/3 ∧
  (∀ x ∈ Set.Ioo 0 4, f_deriv k x ≤ 0) ↔ 0 < k ∧ k ≤ 1/3 :=
sorry

end problem_solution_l4116_411621


namespace convex_polyhedron_same_sided_faces_l4116_411671

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face

/-- Theorem: Every convex polyhedron has at least two faces with the same number of sides -/
theorem convex_polyhedron_same_sided_faces (P : ConvexPolyhedron) :
  ∃ (f₁ f₂ : Face), f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.sides = f₂.sides :=
sorry

end convex_polyhedron_same_sided_faces_l4116_411671


namespace inequality_proof_l4116_411638

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_eq : 1 + a + b + c = 2 * a * b * c) : 
  (a * b) / (1 + a + b) + (b * c) / (1 + b + c) + (c * a) / (1 + c + a) ≥ 3 / 2 := by
  sorry

end inequality_proof_l4116_411638


namespace line_not_in_second_quadrant_l4116_411612

-- Define the line Ax + By + C = 0
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (l : Line) : Prop :=
  l.A * l.C < 0 ∧ l.B * l.C > 0

-- Define the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant (l : Line) 
  (h : satisfies_conditions l) :
  ¬∃ (x y : ℝ), l.A * x + l.B * y + l.C = 0 ∧ in_second_quadrant x y :=
by sorry

end line_not_in_second_quadrant_l4116_411612


namespace volume_surface_area_ratio_l4116_411608

/-- Represents a shape created by joining unit cubes -/
structure CubeShape where
  /-- The number of unit cubes in the shape -/
  num_cubes : ℕ
  /-- The number of cubes surrounding the center cube -/
  surrounding_cubes : ℕ
  /-- Whether there's an additional cube on top -/
  has_top_cube : Bool

/-- Calculates the volume of the shape -/
def volume (shape : CubeShape) : ℕ := shape.num_cubes

/-- Calculates the surface area of the shape -/
def surface_area (shape : CubeShape) : ℕ :=
  shape.surrounding_cubes * 4 + (if shape.has_top_cube then 5 else 0)

/-- The specific shape described in the problem -/
def problem_shape : CubeShape :=
  { num_cubes := 8
  , surrounding_cubes := 6
  , has_top_cube := true }

theorem volume_surface_area_ratio :
  (volume problem_shape : ℚ) / (surface_area problem_shape : ℚ) = 8 / 29 := by sorry

end volume_surface_area_ratio_l4116_411608


namespace funfair_tickets_l4116_411627

theorem funfair_tickets (total_rolls : ℕ) (fourth_grade_percent : ℚ) 
  (fifth_grade_percent : ℚ) (sixth_grade_bought : ℕ) (tickets_left : ℕ) :
  total_rolls = 30 →
  fourth_grade_percent = 30 / 100 →
  fifth_grade_percent = 50 / 100 →
  sixth_grade_bought = 100 →
  tickets_left = 950 →
  ∃ (tickets_per_roll : ℕ),
    tickets_per_roll * total_rolls * (1 - fourth_grade_percent) * (1 - fifth_grade_percent) - sixth_grade_bought = tickets_left ∧
    tickets_per_roll = 100 := by
  sorry

end funfair_tickets_l4116_411627


namespace total_players_is_fifty_l4116_411624

/-- The number of cricket players -/
def cricket_players : ℕ := 12

/-- The number of hockey players -/
def hockey_players : ℕ := 17

/-- The number of football players -/
def football_players : ℕ := 11

/-- The number of softball players -/
def softball_players : ℕ := 10

/-- The total number of players on the ground -/
def total_players : ℕ := cricket_players + hockey_players + football_players + softball_players

/-- Theorem stating that the total number of players is 50 -/
theorem total_players_is_fifty : total_players = 50 := by
  sorry

end total_players_is_fifty_l4116_411624


namespace inequality_solution_l4116_411601

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 8) ∨ 10 < x) :=
by sorry

end inequality_solution_l4116_411601


namespace pedestrian_speed_ratio_l4116_411688

/-- Two pedestrians depart simultaneously from point A in the same direction.
    The first pedestrian meets a tourist 20 minutes after leaving point A.
    The second pedestrian meets the tourist 5 minutes after the first pedestrian.
    The tourist arrives at point A 10 minutes after the second meeting. -/
theorem pedestrian_speed_ratio 
  (v₁ : ℝ) -- speed of the first pedestrian
  (v₂ : ℝ) -- speed of the second pedestrian
  (v : ℝ)  -- speed of the tourist
  (h₁ : v₁ > 0)
  (h₂ : v₂ > 0)
  (h₃ : v > 0)
  (h₄ : (1/3) * v₁ = (1/4) * v) -- first meeting point equation
  (h₅ : (5/12) * v₂ = (1/6) * v) -- second meeting point equation
  : v₁ / v₂ = 15 / 8 := by
  sorry


end pedestrian_speed_ratio_l4116_411688


namespace first_project_breadth_l4116_411679

/-- Represents a digging project with depth, length, breadth, and duration -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ
  duration : ℝ

/-- The volume of a digging project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project with unknown breadth -/
def project1 (b : ℝ) : DiggingProject := {
  depth := 100,
  length := 25,
  breadth := b,
  duration := 12
}

/-- The second digging project -/
def project2 : DiggingProject := {
  depth := 75,
  length := 20,
  breadth := 50,
  duration := 12
}

/-- The theorem stating that the breadth of the first project is 30 meters -/
theorem first_project_breadth :
  ∃ b : ℝ, volume (project1 b) = volume project2 ∧ b = 30 := by
  sorry


end first_project_breadth_l4116_411679


namespace fraction_sum_equals_one_l4116_411649

theorem fraction_sum_equals_one (m : ℝ) (h : m ≠ 1) :
  m / (m - 1) + 1 / (1 - m) = 1 := by
  sorry

end fraction_sum_equals_one_l4116_411649


namespace system_of_equations_sum_l4116_411655

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 13 * x + b * y + c * z = 0)
  (eq2 : a * x + 23 * y + c * z = 0)
  (eq3 : a * x + b * y + 42 * z = 0)
  (ha : a ≠ 13)
  (hx : x ≠ 0) :
  13 / (a - 13) + 23 / (b - 23) + 42 / (c - 42) = -2 := by
  sorry

end system_of_equations_sum_l4116_411655


namespace rectangle_area_ratio_l4116_411689

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a / c = b / d = 3 / 5, then the ratio of their areas is 9:25 -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 3 / 5) (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 := by
  sorry

end rectangle_area_ratio_l4116_411689


namespace tan_cos_eq_sin_minus_m_sin_l4116_411674

theorem tan_cos_eq_sin_minus_m_sin (m : ℝ) : 
  Real.tan (π / 12) * Real.cos (5 * π / 12) = Real.sin (5 * π / 12) - m * Real.sin (π / 12) → 
  m = 2 * Real.sqrt 3 := by
sorry

end tan_cos_eq_sin_minus_m_sin_l4116_411674


namespace cube_volume_surface_area_l4116_411635

theorem cube_volume_surface_area (y : ℝ) (h1 : y > 0) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*y ∧ 6*s^2 = 6*y) → y = 64 := by
  sorry

end cube_volume_surface_area_l4116_411635


namespace sally_monday_shirts_l4116_411645

def shirts_sewn_tuesday : ℕ := 3
def shirts_sewn_wednesday : ℕ := 2
def buttons_per_shirt : ℕ := 5
def total_buttons_needed : ℕ := 45

theorem sally_monday_shirts :
  ∃ (monday_shirts : ℕ),
    monday_shirts + shirts_sewn_tuesday + shirts_sewn_wednesday = 
    total_buttons_needed / buttons_per_shirt ∧
    monday_shirts = 4 := by
  sorry

end sally_monday_shirts_l4116_411645


namespace problem_statement_l4116_411662

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ 1/x + y < 1/a + b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → 1/x + y ≥ 4) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → x/y ≤ 1/4) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ x/y = 1/4) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → 1/2 * y - x ≥ Real.sqrt 2 - 1) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ 1/2 * y - x = Real.sqrt 2 - 1) :=
by sorry

end problem_statement_l4116_411662


namespace common_tangent_lower_bound_l4116_411659

/-- Given two curves C₁: y = ax² (a > 0) and C₂: y = e^x, if they have a common tangent line,
    then a ≥ e²/4 -/
theorem common_tangent_lower_bound (a : ℝ) (h_pos : a > 0) :
  (∃ x₁ x₂ : ℝ, (2 * a * x₁ = Real.exp x₂) ∧ 
                (a * x₁^2 - Real.exp x₂ = 2 * a * x₁ * (x₁ - x₂))) →
  a ≥ Real.exp 2 / 4 := by
  sorry


end common_tangent_lower_bound_l4116_411659


namespace rhombus_area_l4116_411610

/-- The area of a rhombus with diagonals measuring 9 cm and 14 cm is 63 square centimeters. -/
theorem rhombus_area (d1 d2 area : ℝ) : 
  d1 = 9 → d2 = 14 → area = (d1 * d2) / 2 → area = 63 := by sorry

end rhombus_area_l4116_411610


namespace atlantic_charge_proof_l4116_411669

/-- The base rate for United Telephone in dollars -/
def united_base_rate : ℚ := 9

/-- The additional charge per minute for United Telephone in dollars -/
def united_per_minute : ℚ := 1/4

/-- The base rate for Atlantic Call in dollars -/
def atlantic_base_rate : ℚ := 12

/-- The number of minutes for which the bills are equal -/
def equal_minutes : ℕ := 60

/-- The additional charge per minute for Atlantic Call in dollars -/
def atlantic_per_minute : ℚ := 1/5

theorem atlantic_charge_proof :
  united_base_rate + united_per_minute * equal_minutes =
  atlantic_base_rate + atlantic_per_minute * equal_minutes :=
sorry

end atlantic_charge_proof_l4116_411669


namespace intersection_of_M_and_N_l4116_411677

def M : Set ℤ := {1, 2, 3, 4, 5, 6}

def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 2, 3, 4} := by sorry

end intersection_of_M_and_N_l4116_411677


namespace cube_angles_l4116_411647

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Calculates the angle between two skew lines in a cube -/
def angle_between_skew_lines (c : Cube) (l1 l2 : Fin 2 → Fin 8) : ℝ :=
  sorry

/-- Calculates the angle between a line and a plane in a cube -/
def angle_between_line_and_plane (c : Cube) (l : Fin 2 → Fin 8) (p : Fin 4 → Fin 8) : ℝ :=
  sorry

/-- Theorem stating the angles in a cube -/
theorem cube_angles (c : Cube) : 
  angle_between_skew_lines c ![7, 1] ![0, 2] = 60 ∧ 
  angle_between_line_and_plane c ![7, 1] ![7, 5, 2, 3] = 30 :=
sorry

end cube_angles_l4116_411647


namespace ellie_and_hank_weight_l4116_411665

/-- The weights of Ellie, Frank, Gina, and Hank satisfy the given conditions
    and Ellie and Hank weigh 355 pounds together. -/
theorem ellie_and_hank_weight (e f g h : ℝ) 
    (ef_sum : e + f = 310)
    (fg_sum : f + g = 280)
    (gh_sum : g + h = 325)
    (g_minus_h : g = h + 10) :
  e + h = 355 := by
  sorry

end ellie_and_hank_weight_l4116_411665


namespace first_number_value_l4116_411695

theorem first_number_value (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 →
  (b + c + d) / 3 = 15 →
  d = 18 →
  a = 33 := by
sorry

end first_number_value_l4116_411695


namespace tangent_lines_proof_l4116_411656

-- Define the curves
def f (x : ℝ) : ℝ := x^3 + x^2 + 1
def g (x : ℝ) : ℝ := x^2

-- Define the points
def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (3, 5)

-- Define the tangent line equations
def tangent_line1 (x y : ℝ) : Prop := x - y + 2 = 0
def tangent_line2 (x y : ℝ) : Prop := 2*x - y - 1 = 0
def tangent_line3 (x y : ℝ) : Prop := 10*x - y - 25 = 0

theorem tangent_lines_proof :
  (∀ x y : ℝ, y = f x → (x, y) = P1 → tangent_line1 x y) ∧
  (∀ x y : ℝ, y = g x → (x, y) = P2 → (tangent_line2 x y ∨ tangent_line3 x y)) :=
sorry

end tangent_lines_proof_l4116_411656


namespace batsman_average_increase_l4116_411698

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the average runs per innings -/
def calculateAverage (totalRuns : ℕ) (innings : ℕ) : ℚ :=
  (totalRuns : ℚ) / (innings : ℚ)

theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 11 →
    let newTotalRuns := b.totalRuns + 55
    let newInnings := b.innings + 1
    let newAverage := calculateAverage newTotalRuns newInnings
    newAverage = 44 →
    newAverage - b.average = 1 := by
  sorry

#check batsman_average_increase

end batsman_average_increase_l4116_411698


namespace career_preference_degrees_l4116_411646

/-- Represents the ratio of male to female students in a class -/
structure GenderRatio where
  male : ℕ
  female : ℕ

/-- Represents the number of students preferring a career -/
structure CareerPreference where
  male : ℕ
  female : ℕ

/-- Calculates the degrees in a circle graph for a career preference -/
def degreesForPreference (ratio : GenderRatio) (pref : CareerPreference) : ℚ :=
  360 * (pref.male + pref.female : ℚ) / (ratio.male + ratio.female : ℚ)

theorem career_preference_degrees 
  (ratio : GenderRatio) 
  (pref : CareerPreference) : 
  ratio.male = 2 ∧ ratio.female = 3 ∧ pref.male = 1 ∧ pref.female = 1 → 
  degreesForPreference ratio pref = 144 := by
  sorry

end career_preference_degrees_l4116_411646


namespace parabola_vertex_l4116_411641

/-- The vertex of the parabola y = -2x^2 + 3 is (0, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * x^2 + 3 → (0, 3) = (x, y) := by
  sorry

end parabola_vertex_l4116_411641


namespace abc_books_sold_l4116_411670

theorem abc_books_sold (top_price : ℕ) (abc_price : ℕ) (top_sold : ℕ) (earnings_diff : ℕ) :
  top_price = 8 →
  abc_price = 23 →
  top_sold = 13 →
  earnings_diff = 12 →
  ∃ (abc_sold : ℕ), abc_sold * abc_price = top_sold * top_price - earnings_diff :=
by
  sorry

end abc_books_sold_l4116_411670


namespace combined_mean_of_two_sets_l4116_411637

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 4 →
  set1_mean = 10 →
  set2_count = 8 →
  set2_mean = 21 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 52 / 3 := by
  sorry

end combined_mean_of_two_sets_l4116_411637


namespace arithmetic_sequence_common_difference_l4116_411618

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_common_difference :
  ∀ (d : ℚ), 
    (arithmetic_sequence 1 d 0 = 1) →
    (sum_arithmetic_sequence 1 d 5 = 20) →
    d = 3/2 := by
  sorry

end arithmetic_sequence_common_difference_l4116_411618


namespace arccos_one_half_l4116_411697

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_l4116_411697


namespace digit_sum_puzzle_l4116_411678

theorem digit_sum_puzzle :
  ∀ (B E D H : ℕ),
    B < 10 → E < 10 → D < 10 → H < 10 →
    B ≠ E → B ≠ D → B ≠ H → E ≠ D → E ≠ H → D ≠ H →
    (10 * B + E) * (10 * D + E) = 111 * H →
    E + B + D + H = 17 := by
  sorry

end digit_sum_puzzle_l4116_411678


namespace lines_parallel_iff_a_eq_neg_one_l4116_411668

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l₁: x + ay + 3 = 0 -/
def l1 (a : ℝ) : Line :=
  { a := 1, b := a, c := 3 }

/-- The second line l₂: (a-2)x + 3y + a = 0 -/
def l2 (a : ℝ) : Line :=
  { a := a - 2, b := 3, c := a }

/-- Theorem: The lines l₁ and l₂ are parallel if and only if a = -1 -/
theorem lines_parallel_iff_a_eq_neg_one :
  ∀ a : ℝ, are_parallel (l1 a) (l2 a) ↔ a = -1 := by
  sorry

end lines_parallel_iff_a_eq_neg_one_l4116_411668


namespace total_spots_l4116_411664

/-- The number of spots on each dog -/
structure DogSpots where
  rover : ℕ
  cisco : ℕ
  granger : ℕ
  sparky : ℕ
  bella : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (d : DogSpots) : Prop :=
  d.rover = 46 ∧
  d.cisco = d.rover / 2 - 5 ∧
  d.granger = 5 * d.cisco ∧
  d.sparky = 3 * d.rover ∧
  d.bella = 2 * (d.granger + d.sparky)

/-- The theorem to be proven -/
theorem total_spots (d : DogSpots) (h : satisfiesConditions d) : 
  d.granger + d.cisco + d.sparky + d.bella = 702 := by
  sorry

end total_spots_l4116_411664


namespace house_area_proof_l4116_411600

def house_painting_problem (price_per_sqft : ℝ) (total_cost : ℝ) : Prop :=
  price_per_sqft > 0 ∧ total_cost > 0 ∧ (total_cost / price_per_sqft = 88)

theorem house_area_proof :
  house_painting_problem 20 1760 :=
sorry

end house_area_proof_l4116_411600


namespace distinct_lunches_l4116_411690

/-- The number of main course options available --/
def main_course_options : ℕ := 4

/-- The number of beverage options available --/
def beverage_options : ℕ := 3

/-- The number of snack options available --/
def snack_options : ℕ := 2

/-- The total number of distinct possible lunches --/
def total_lunches : ℕ := main_course_options * beverage_options * snack_options

/-- Theorem stating that the total number of distinct possible lunches is 24 --/
theorem distinct_lunches : total_lunches = 24 := by
  sorry

end distinct_lunches_l4116_411690


namespace find_number_l4116_411667

theorem find_number : ∃ x : ℝ, 3 * x + 3 * 14 + 3 * 18 + 11 = 152 ∧ x = 15 := by
  sorry

end find_number_l4116_411667


namespace race_probability_l4116_411661

theorem race_probability (total_cars : ℕ) (prob_X prob_Z prob_total : ℝ) : 
  total_cars = 12 →
  prob_X = 1/6 →
  prob_Z = 1/8 →
  prob_total = 0.39166666666666666 →
  ∃ (prob_Y : ℝ), prob_Y = prob_total - prob_X - prob_Z ∧ prob_Y = 0.1 :=
by sorry

end race_probability_l4116_411661


namespace saturday_price_calculation_l4116_411692

theorem saturday_price_calculation (original_price : ℝ) 
  (h1 : original_price = 180) 
  (sale_discount : ℝ) (h2 : sale_discount = 0.5)
  (saturday_discount : ℝ) (h3 : saturday_discount = 0.2) : 
  original_price * (1 - sale_discount) * (1 - saturday_discount) = 72 := by
  sorry

end saturday_price_calculation_l4116_411692


namespace one_rhythm_for_specific_phrase_l4116_411693

/-- Represents the duration of a note in terms of fractions of a measure -/
structure NoteDuration where
  numerator : ℕ
  denominator : ℕ+

/-- Represents a musical phrase -/
structure MusicalPhrase where
  measures : ℕ
  note_duration : NoteDuration
  no_rests : Bool

/-- Counts the number of different rhythms possible for a given musical phrase -/
def count_rhythms (phrase : MusicalPhrase) : ℕ :=
  sorry

/-- Theorem stating that a 2-measure phrase with notes lasting 1/8 of 1/4 of a measure and no rests has only one possible rhythm -/
theorem one_rhythm_for_specific_phrase :
  ∀ (phrase : MusicalPhrase),
    phrase.measures = 2 ∧
    phrase.note_duration = { numerator := 1, denominator := 32 } ∧
    phrase.no_rests = true →
    count_rhythms phrase = 1 :=
  sorry

end one_rhythm_for_specific_phrase_l4116_411693


namespace ratio_problem_l4116_411672

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.25 * a) (h5 : m = b - 0.2 * b) : m / x = 4 / 5 := by
  sorry

end ratio_problem_l4116_411672


namespace flour_calculation_l4116_411680

/-- Calculates the required cups of flour given the original recipe ratio, scaling factor, and amount of butter used. -/
def required_flour (original_butter original_flour scaling_factor butter_used : ℚ) : ℚ :=
  (butter_used / original_butter) * scaling_factor * original_flour

/-- Proves that given the specified conditions, the required amount of flour is 30 cups. -/
theorem flour_calculation (original_butter original_flour scaling_factor butter_used : ℚ) 
  (h1 : original_butter = 2)
  (h2 : original_flour = 5)
  (h3 : scaling_factor = 4)
  (h4 : butter_used = 12) :
  required_flour original_butter original_flour scaling_factor butter_used = 30 := by
sorry

#eval required_flour 2 5 4 12

end flour_calculation_l4116_411680


namespace james_and_louise_age_sum_james_and_louise_age_sum_is_correct_l4116_411617

/-- The sum of James and Louise's current ages given the conditions -/
theorem james_and_louise_age_sum : ℝ :=
  let james_age : ℝ := sorry
  let louise_age : ℝ := sorry

  -- James is eight years older than Louise
  have h1 : james_age = louise_age + 8 := by sorry

  -- Ten years from now, James will be five times as old as Louise was five years ago
  have h2 : james_age + 10 = 5 * (louise_age - 5) := by sorry

  -- The sum of their current ages
  have h3 : james_age + louise_age = 29.5 := by sorry

  29.5

theorem james_and_louise_age_sum_is_correct : james_and_louise_age_sum = 29.5 := by sorry

end james_and_louise_age_sum_james_and_louise_age_sum_is_correct_l4116_411617


namespace population_change_theorem_l4116_411686

/-- Calculates the population after three years of changes --/
def population_after_three_years (initial_population : ℕ) : ℕ :=
  let year1 := (initial_population * 80) / 100
  let year2_increase := (year1 * 110) / 100
  let year2 := (year2_increase * 95) / 100
  let year3_increase := (year2 * 108) / 100
  (year3_increase * 75) / 100

/-- Theorem stating that the population after three years of changes is 10157 --/
theorem population_change_theorem :
  population_after_three_years 15000 = 10157 := by
  sorry

end population_change_theorem_l4116_411686


namespace square_exterior_points_distance_l4116_411630

/-- Given a square ABCD with side length 13 and exterior points E and F,
    prove that EF² = 578 when BE = DF = 5 and AE = CF = 12 -/
theorem square_exterior_points_distance (A B C D E F : ℝ × ℝ) : 
  let side_length : ℝ := 13
  -- Square ABCD
  A = (0, side_length) ∧ 
  B = (side_length, side_length) ∧ 
  C = (side_length, 0) ∧ 
  D = (0, 0) ∧
  -- Exterior points E and F
  dist B E = 5 ∧
  dist D F = 5 ∧
  dist A E = 12 ∧
  dist C F = 12
  →
  dist E F ^ 2 = 578 := by
sorry


end square_exterior_points_distance_l4116_411630


namespace additional_three_pointers_l4116_411623

def points_to_tie : ℕ := 17
def points_over_record : ℕ := 5
def old_record : ℕ := 257
def free_throws : ℕ := 5
def regular_baskets : ℕ := 4
def normal_three_pointers : ℕ := 2

def points_per_free_throw : ℕ := 1
def points_per_regular_basket : ℕ := 2
def points_per_three_pointer : ℕ := 3

def total_points_final_game : ℕ := points_to_tie + points_over_record
def points_from_free_throws : ℕ := free_throws * points_per_free_throw
def points_from_regular_baskets : ℕ := regular_baskets * points_per_regular_basket
def points_from_three_pointers : ℕ := total_points_final_game - points_from_free_throws - points_from_regular_baskets

theorem additional_three_pointers (
  h1 : points_from_three_pointers % points_per_three_pointer = 0
) : (points_from_three_pointers / points_per_three_pointer) - normal_three_pointers = 1 := by
  sorry

end additional_three_pointers_l4116_411623


namespace banana_slices_per_yogurt_l4116_411644

/-- Given that one banana yields 10 slices, 5 yogurts need to be made, and 4 bananas are bought,
    prove that 8 banana slices are needed for each yogurt. -/
theorem banana_slices_per_yogurt :
  let slices_per_banana : ℕ := 10
  let yogurts_to_make : ℕ := 5
  let bananas_bought : ℕ := 4
  let total_slices : ℕ := slices_per_banana * bananas_bought
  let slices_per_yogurt : ℕ := total_slices / yogurts_to_make
  slices_per_yogurt = 8 := by sorry

end banana_slices_per_yogurt_l4116_411644


namespace car_speed_second_hour_l4116_411681

/-- Proves that given a car traveling for two hours with an average speed of 45 km/h
    and a speed of 60 km/h in the first hour, the speed in the second hour must be 30 km/h. -/
theorem car_speed_second_hour
  (average_speed : ℝ)
  (first_hour_speed : ℝ)
  (total_time : ℝ)
  (h_average_speed : average_speed = 45)
  (h_first_hour_speed : first_hour_speed = 60)
  (h_total_time : total_time = 2)
  : (2 * average_speed - first_hour_speed = 30) :=
by sorry

end car_speed_second_hour_l4116_411681


namespace gcd_390_455_l4116_411632

theorem gcd_390_455 : Nat.gcd 390 455 = 65 := by sorry

end gcd_390_455_l4116_411632


namespace min_value_expression_min_value_achievable_l4116_411650

theorem min_value_expression (x : ℝ) (h : x > 0) :
  x^2 / x + 2 + 5 / x ≥ 2 * Real.sqrt 5 + 2 :=
sorry

theorem min_value_achievable :
  ∃ (x : ℝ), x > 0 ∧ x^2 / x + 2 + 5 / x = 2 * Real.sqrt 5 + 2 :=
sorry

end min_value_expression_min_value_achievable_l4116_411650


namespace abc_xyz_inequality_l4116_411673

theorem abc_xyz_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a*x + b*y + c*z ≥ 0 := by
  sorry

end abc_xyz_inequality_l4116_411673


namespace die_throw_probability_l4116_411663

/-- Represents a fair six-sided die throw -/
def DieFace := Fin 6

/-- The probability of a specific outcome when throwing a fair die three times -/
def prob_single_outcome : ℚ := 1 / 216

/-- Checks if a + bi is a root of x^2 - 2x + c = 0 -/
def is_root (a b c : ℕ) : Prop :=
  a = 1 ∧ c = b^2 + 1

/-- The number of favorable outcomes -/
def favorable_outcomes : ℕ := 2

theorem die_throw_probability :
  (favorable_outcomes : ℚ) * prob_single_outcome = 1 / 108 := by
  sorry


end die_throw_probability_l4116_411663


namespace number_of_divisors_36_l4116_411643

/-- The number of positive divisors of 36 is 9. -/
theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end number_of_divisors_36_l4116_411643


namespace crescent_area_equals_rectangle_area_l4116_411604

theorem crescent_area_equals_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_area := 4 * a * b
  let circle_area := π * (a^2 + b^2)
  let semicircles_area := π * a^2 + π * b^2
  let crescent_area := semicircles_area + rectangle_area - circle_area
  crescent_area = rectangle_area := by
  sorry

end crescent_area_equals_rectangle_area_l4116_411604


namespace derivative_f_at_zero_l4116_411657

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x^2 * Real.cos (4 / (3 * x)) + x^2 / 2 else 0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end derivative_f_at_zero_l4116_411657
