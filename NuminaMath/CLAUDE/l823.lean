import Mathlib

namespace triangle_area_from_altitudes_l823_82341

/-- The area of a triangle given its three altitudes -/
theorem triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (h₃_pos : h₃ > 0) :
  ∃ S : ℝ, S > 0 ∧ S = Real.sqrt ((1/h₁ + 1/h₂ + 1/h₃) * (-1/h₁ + 1/h₂ + 1/h₃) * (1/h₁ - 1/h₂ + 1/h₃) * (1/h₁ + 1/h₂ - 1/h₃)) :=
by sorry

end triangle_area_from_altitudes_l823_82341


namespace spectators_count_l823_82326

/-- The number of wristbands given to each spectator -/
def wristbands_per_person : ℕ := 2

/-- The total number of wristbands distributed -/
def total_wristbands : ℕ := 290

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 145 := by
  sorry

end spectators_count_l823_82326


namespace binomial_max_and_expectation_l823_82330

/-- The probability mass function for a binomial distribution with 20 trials and 2 successes -/
def f (p : ℝ) : ℝ := 190 * p^2 * (1 - p)^18

/-- The value of p that maximizes f(p) -/
def p₀ : ℝ := 0.1

/-- The number of items in a box -/
def box_size : ℕ := 200

/-- The number of items initially inspected -/
def initial_inspection : ℕ := 20

/-- The cost of inspecting one item -/
def inspection_cost : ℝ := 2

/-- The compensation fee for one defective item -/
def compensation_fee : ℝ := 25

/-- The expected number of defective items in the remaining items after initial inspection -/
def expected_defective : ℝ := 18

theorem binomial_max_and_expectation :
  (∀ p, p > 0 ∧ p < 1 → f p ≤ f p₀) ∧
  expected_defective = (box_size - initial_inspection : ℝ) * p₀ := by sorry

end binomial_max_and_expectation_l823_82330


namespace ant_travel_distance_l823_82343

/-- The number of nodes on the bamboo -/
def num_nodes : ℕ := 30

/-- The height of the first node in feet -/
def first_node_height : ℝ := 0.5

/-- The increase in height between consecutive nodes in feet -/
def node_height_diff : ℝ := 0.03

/-- The circumference of the first circle in feet -/
def first_circle_circumference : ℝ := 1.3

/-- The decrease in circumference between consecutive circles in feet -/
def circle_circumference_diff : ℝ := 0.013

/-- The total distance traveled by the ant in feet -/
def total_distance : ℝ := 61.395

/-- Theorem stating the total distance traveled by the ant -/
theorem ant_travel_distance :
  (num_nodes : ℝ) * first_node_height + 
  (num_nodes * (num_nodes - 1) / 2) * node_height_diff +
  (num_nodes : ℝ) * first_circle_circumference - 
  (num_nodes * (num_nodes - 1) / 2) * circle_circumference_diff = 
  total_distance :=
sorry

end ant_travel_distance_l823_82343


namespace hand_towels_per_set_l823_82328

/-- The number of hand towels in a set -/
def h : ℕ := sorry

/-- The number of bath towels in a set -/
def bath_towels_per_set : ℕ := 6

/-- The smallest number of each type of towel sold -/
def min_towels_sold : ℕ := 102

theorem hand_towels_per_set :
  (∃ (n : ℕ), h * n = bath_towels_per_set * n ∧ h * n = min_towels_sold) →
  h = 17 := by sorry

end hand_towels_per_set_l823_82328


namespace river_speed_is_three_l823_82344

/-- Represents a ship with its upstream speed -/
structure Ship where
  speed : ℝ

/-- Represents the rescue scenario -/
structure RescueScenario where
  ships : List Ship
  timeToTurn : ℝ
  distanceToRescue : ℝ
  riverSpeed : ℝ

/-- Theorem: Given the conditions, the river speed is 3 km/h -/
theorem river_speed_is_three (scenario : RescueScenario) :
  scenario.ships = [Ship.mk 4, Ship.mk 6, Ship.mk 10] →
  scenario.timeToTurn = 1 →
  scenario.distanceToRescue = 6 →
  scenario.riverSpeed = 3 := by
  sorry


end river_speed_is_three_l823_82344


namespace trains_meet_at_1108_l823_82369

/-- Represents a train with its departure time and speed -/
structure Train where
  departureTime : Nat  -- minutes since midnight
  speed : Nat          -- km/h

/-- Represents a station with its distance from Station A -/
structure Station where
  distanceFromA : Nat  -- km

def stationA : Station := { distanceFromA := 0 }
def stationB : Station := { distanceFromA := 300 }
def stationC : Station := { distanceFromA := 150 }

def trainA : Train := { departureTime := 9 * 60 + 45, speed := 60 }
def trainB : Train := { departureTime := 10 * 60, speed := 80 }

def stopTime : Nat := 10  -- minutes

/-- Calculates the meeting time of two trains given the conditions -/
def calculateMeetingTime (trainA trainB : Train) (stationA stationB stationC : Station) (stopTime : Nat) : Nat :=
  sorry  -- Proof to be implemented

theorem trains_meet_at_1108 :
  calculateMeetingTime trainA trainB stationA stationB stationC stopTime = 11 * 60 + 8 := by
  sorry  -- Proof to be implemented

end trains_meet_at_1108_l823_82369


namespace circle_center_coordinates_l823_82384

/-- The center of a circle satisfying given conditions -/
theorem circle_center_coordinates : ∃ (x y : ℝ),
  (x - 2 * y = 0) ∧
  (3 * x + 4 * y = 10) ∧
  (x = 2 ∧ y = 1) := by
  sorry

end circle_center_coordinates_l823_82384


namespace instantaneous_rate_of_change_at_e_l823_82386

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem instantaneous_rate_of_change_at_e :
  deriv f e = 0 := by sorry

end instantaneous_rate_of_change_at_e_l823_82386


namespace infinite_series_sum_l823_82333

/-- The sum of the infinite series $\sum_{k = 1}^\infty \frac{k^3}{3^k}$ is equal to $\frac{39}{8}$. -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k^3 : ℝ) / 3^k) = 39 / 8 := by sorry

end infinite_series_sum_l823_82333


namespace continuity_at_3_l823_82319

def f (x : ℝ) := -2 * x^2 - 4

theorem continuity_at_3 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end continuity_at_3_l823_82319


namespace impossible_shot_l823_82306

-- Define the elliptical billiard table
structure EllipticalTable where
  foci : Pointℝ × Pointℝ

-- Define the balls
structure Ball where
  position : Pointℝ

-- Define the properties of the problem
def is_on_edge (table : EllipticalTable) (ball : Ball) : Prop := sorry

def is_on_focal_segment (table : EllipticalTable) (ball : Ball) : Prop := sorry

def bounces_and_hits (table : EllipticalTable) (ball_A ball_B : Ball) : Prop := sorry

def crosses_focal_segment_before_bounce (table : EllipticalTable) (ball : Ball) : Prop := sorry

-- State the theorem
theorem impossible_shot (table : EllipticalTable) (ball_A ball_B : Ball) :
  is_on_edge table ball_A ∧
  is_on_focal_segment table ball_B ∧
  bounces_and_hits table ball_A ball_B ∧
  ¬crosses_focal_segment_before_bounce table ball_A →
  False := by sorry

end impossible_shot_l823_82306


namespace fraction_decomposition_l823_82361

theorem fraction_decomposition (n : ℕ) (h : n ≥ 3) :
  (2 : ℚ) / (2 * n - 1) = 1 / n + 1 / (n * (2 * n - 1)) := by
  sorry

#check fraction_decomposition

end fraction_decomposition_l823_82361


namespace height_increase_l823_82398

/-- If a person's height increases by 5% to reach 147 cm, their original height was 140 cm. -/
theorem height_increase (original_height : ℝ) : 
  original_height * 1.05 = 147 → original_height = 140 :=
by sorry

end height_increase_l823_82398


namespace arithmetic_mean_of_special_set_l823_82394

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := List.replicate (n - 2) 2 ++ [1 - 2 / n, 1 - 2 / n]
  (List.sum set) / n = 2 - 2 / n - 4 / n^2 := by
  sorry

end arithmetic_mean_of_special_set_l823_82394


namespace eighth_triangular_number_l823_82393

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 8th triangular number is 36 -/
theorem eighth_triangular_number : triangular_number 8 = 36 := by
  sorry

end eighth_triangular_number_l823_82393


namespace fifth_grade_students_l823_82397

theorem fifth_grade_students (total_boys : ℕ) (soccer_players : ℕ) (boys_soccer_percentage : ℚ) (girls_not_soccer : ℕ) :
  total_boys = 296 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  girls_not_soccer = 89 →
  ∃ (total_students : ℕ), total_students = 420 :=
by
  sorry

end fifth_grade_students_l823_82397


namespace rationalize_denominator_l823_82340

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end rationalize_denominator_l823_82340


namespace x_range_l823_82387

theorem x_range (x : ℝ) : (1 / x < 4 ∧ 1 / x > -2) → (x < -1/2 ∨ x > 1/4) := by
  sorry

end x_range_l823_82387


namespace odd_square_plus_n_times_odd_plus_one_parity_l823_82350

theorem odd_square_plus_n_times_odd_plus_one_parity (o n : ℤ) 
  (ho : ∃ k : ℤ, o = 2 * k + 1) :
  Odd (o^2 + n*o + 1) ↔ Odd n :=
sorry

end odd_square_plus_n_times_odd_plus_one_parity_l823_82350


namespace probability_six_heads_ten_coins_l823_82331

def num_coins : ℕ := 10
def num_heads : ℕ := 6

theorem probability_six_heads_ten_coins :
  (Nat.choose num_coins num_heads : ℚ) / (2 ^ num_coins) = 210 / 1024 := by
  sorry

end probability_six_heads_ten_coins_l823_82331


namespace problem_solution_l823_82320

/-- An increasing linear function on ℝ -/
def IncreasingLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ ∀ x, f x = a * x + b

theorem problem_solution (f g : ℝ → ℝ) (m : ℝ) :
  IncreasingLinearFunction f →
  (∀ x, g x = f x * (x + m)) →
  (∀ x, f (f x) = 16 * x + 5) →
  (∃ M, M = 13 ∧ ∀ x ∈ Set.Icc 1 3, g x ≤ M) →
  (∀ x, f x = 4 * x + 1) ∧ m = -2 := by
  sorry

end problem_solution_l823_82320


namespace complex_equation_solution_l823_82315

theorem complex_equation_solution (i : ℂ) (z : ℂ) (h1 : i * i = -1) (h2 : i * z = 4 + 3 * i) : z = 3 - 4 * i := by
  sorry

end complex_equation_solution_l823_82315


namespace least_sum_of_exponents_l823_82308

/-- The sum of distinct powers of 2 that equals 72 -/
def sum_of_powers (a b c : ℕ) : Prop :=
  2^a + 2^b + 2^c = 72 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The least sum of exponents when expressing 72 as a sum of at least three distinct powers of 2 -/
theorem least_sum_of_exponents :
  ∃ (a b c : ℕ), sum_of_powers a b c ∧
    ∀ (x y z : ℕ), sum_of_powers x y z → a + b + c ≤ x + y + z ∧ a + b + c = 9 :=
sorry

end least_sum_of_exponents_l823_82308


namespace cube_root_243_equals_3_to_5_thirds_l823_82359

theorem cube_root_243_equals_3_to_5_thirds : 
  (243 : ℝ) = 3^5 → (243 : ℝ)^(1/3) = 3^(5/3) := by sorry

end cube_root_243_equals_3_to_5_thirds_l823_82359


namespace passengers_proportion_ge_cars_proportion_passengers_proportion_not_lt_cars_proportion_l823_82377

/-- Represents the distribution of passenger cars -/
structure CarDistribution where
  total : ℕ
  overcrowded : ℕ
  passengers : ℕ
  passengers_overcrowded : ℕ

/-- Definition of an overcrowded car (60 or more passengers) -/
def is_overcrowded (passengers : ℕ) : Prop := passengers ≥ 60

/-- The proportion of overcrowded cars -/
def proportion_overcrowded (d : CarDistribution) : ℚ :=
  d.overcrowded / d.total

/-- The proportion of passengers in overcrowded cars -/
def proportion_passengers_overcrowded (d : CarDistribution) : ℚ :=
  d.passengers_overcrowded / d.passengers

/-- Theorem: The proportion of passengers in overcrowded cars is always
    greater than or equal to the proportion of overcrowded cars -/
theorem passengers_proportion_ge_cars_proportion (d : CarDistribution) :
  proportion_passengers_overcrowded d ≥ proportion_overcrowded d := by
  sorry

/-- Corollary: The proportion of passengers in overcrowded cars cannot be
    less than the proportion of overcrowded cars -/
theorem passengers_proportion_not_lt_cars_proportion (d : CarDistribution) :
  ¬(proportion_passengers_overcrowded d < proportion_overcrowded d) := by
  sorry

end passengers_proportion_ge_cars_proportion_passengers_proportion_not_lt_cars_proportion_l823_82377


namespace min_distance_parabola_circle_l823_82396

theorem min_distance_parabola_circle : 
  let parabola := {P : ℝ × ℝ | P.2^2 = P.1}
  let circle := {Q : ℝ × ℝ | (Q.1 - 3)^2 + Q.2^2 = 1}
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ parabola ∧ Q ∈ circle ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ), P' ∈ parabola → Q' ∈ circle →
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (Real.sqrt 11 - 2) / 2 :=
by sorry

end min_distance_parabola_circle_l823_82396


namespace exists_maximal_element_l823_82346

/-- A family of subsets of ℕ satisfying the chain condition -/
structure ChainFamily where
  C : Set (Set ℕ)
  chain_condition : ∀ (chain : ℕ → Set ℕ), (∀ n m, n ≤ m → chain n ⊆ chain m) →
    (∀ n, chain n ∈ C) → ∃ S ∈ C, ∀ n, chain n ⊆ S

/-- The existence of a maximal element in a chain family -/
theorem exists_maximal_element (F : ChainFamily) :
  ∃ S ∈ F.C, ∀ T ∈ F.C, S ⊆ T → S = T := by sorry

end exists_maximal_element_l823_82346


namespace lasagna_ratio_is_two_to_one_l823_82353

/-- Represents the ratio of noodles to beef in Tom's lasagna recipe -/
def lasagna_ratio (beef_amount : ℕ) (initial_noodles : ℕ) (package_weight : ℕ) (packages_needed : ℕ) : ℚ :=
  let total_noodles := initial_noodles + package_weight * packages_needed
  (total_noodles : ℚ) / beef_amount

/-- The ratio of noodles to beef in Tom's lasagna recipe is 2:1 -/
theorem lasagna_ratio_is_two_to_one :
  lasagna_ratio 10 4 2 8 = 2 := by
  sorry

end lasagna_ratio_is_two_to_one_l823_82353


namespace inequality_equivalence_l823_82314

theorem inequality_equivalence (y : ℝ) :
  3/40 + |y - 17/80| < 1/8 ↔ 13/80 < y ∧ y < 21/80 := by sorry

end inequality_equivalence_l823_82314


namespace a_gt_b_necessary_not_sufficient_l823_82368

-- Define the curve C
def CurveC (a b x y : ℝ) : Prop := x^2 / a + y^2 / b = 1

-- Define what it means for C to be an ellipse with foci on the x-axis
def IsEllipseOnXAxis (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ ∀ x y, CurveC a b x y → x^2 + y^2 < a^2

-- Theorem stating that a > b is a necessary but not sufficient condition
theorem a_gt_b_necessary_not_sufficient :
  (∀ a b : ℝ, IsEllipseOnXAxis a b → a > b) ∧
  ¬(∀ a b : ℝ, a > b → IsEllipseOnXAxis a b) := by
  sorry

end a_gt_b_necessary_not_sufficient_l823_82368


namespace sqrt_equation_solution_l823_82372

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 10 → y = 68 / 3 := by
  sorry

end sqrt_equation_solution_l823_82372


namespace rain_probability_tel_aviv_l823_82311

theorem rain_probability_tel_aviv (p : ℝ) (n k : ℕ) (h_p : p = 1/2) (h_n : n = 6) (h_k : k = 4) :
  (n.choose k) * p^k * (1-p)^(n-k) = 15/64 := by
  sorry

end rain_probability_tel_aviv_l823_82311


namespace girls_on_playground_l823_82303

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 62) 
  (h2 : boys = 27) : 
  total_children - boys = 35 := by
sorry

end girls_on_playground_l823_82303


namespace angle_terminal_side_point_l823_82363

/-- Given an angle α whose terminal side passes through the point P(4a,-3a) where a < 0,
    prove that 2sin α + cos α = 2/5 -/
theorem angle_terminal_side_point (a : ℝ) (α : ℝ) (h : a < 0) :
  let x : ℝ := 4 * a
  let y : ℝ := -3 * a
  let r : ℝ := Real.sqrt (x^2 + y^2)
  2 * (y / r) + (x / r) = 2 / 5 := by
  sorry

end angle_terminal_side_point_l823_82363


namespace furniture_shop_cost_price_l823_82345

/-- Given a markup percentage and a selling price, calculates the cost price -/
def calculate_cost_price (markup_percentage : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price / (1 + markup_percentage)

/-- Proves that for a 25% markup and selling price of 4800, the cost price is 3840 -/
theorem furniture_shop_cost_price :
  calculate_cost_price (25 / 100) 4800 = 3840 := by
  sorry

end furniture_shop_cost_price_l823_82345


namespace major_axis_length_l823_82357

/-- Represents an ellipse formed by a plane intersecting a right circular cylinder -/
structure CylinderEllipse where
  cylinder_radius : ℝ
  major_axis : ℝ
  minor_axis : ℝ

/-- The theorem stating the length of the major axis given the conditions -/
theorem major_axis_length (e : CylinderEllipse) 
  (h1 : e.cylinder_radius = 2)
  (h2 : e.minor_axis = 2 * e.cylinder_radius)
  (h3 : e.major_axis = e.minor_axis * 1.6) :
  e.major_axis = 6.4 := by
  sorry

end major_axis_length_l823_82357


namespace absolute_value_complex_power_l823_82300

theorem absolute_value_complex_power : 
  Complex.abs ((5 : ℂ) + (Complex.I * Real.sqrt 11)) ^ 4 = 1296 := by
  sorry

end absolute_value_complex_power_l823_82300


namespace exam_average_score_l823_82332

theorem exam_average_score (max_score : ℕ) (amar_percent bhavan_percent chetan_percent deepak_percent : ℚ) :
  max_score = 1100 →
  amar_percent = 64 / 100 →
  bhavan_percent = 36 / 100 →
  chetan_percent = 44 / 100 →
  deepak_percent = 52 / 100 →
  let amar_score := (amar_percent * max_score : ℚ).floor
  let bhavan_score := (bhavan_percent * max_score : ℚ).floor
  let chetan_score := (chetan_percent * max_score : ℚ).floor
  let deepak_score := (deepak_percent * max_score : ℚ).floor
  let total_score := amar_score + bhavan_score + chetan_score + deepak_score
  (total_score / 4 : ℚ).floor = 539 := by
  sorry

#eval (64 / 100 : ℚ) * 1100  -- Expected output: 704
#eval (36 / 100 : ℚ) * 1100  -- Expected output: 396
#eval (44 / 100 : ℚ) * 1100  -- Expected output: 484
#eval (52 / 100 : ℚ) * 1100  -- Expected output: 572
#eval ((704 + 396 + 484 + 572) / 4 : ℚ)  -- Expected output: 539

end exam_average_score_l823_82332


namespace exists_left_absorbing_l823_82371

variable {S : Type}
variable (star : S → S → S)

axiom commutative : ∀ a b : S, star a b = star b a
axiom associative : ∀ a b c : S, star (star a b) c = star a (star b c)
axiom exists_idempotent : ∃ a : S, star a a = a

theorem exists_left_absorbing : ∃ a : S, ∀ b : S, star a b = a := by
  sorry

end exists_left_absorbing_l823_82371


namespace no_valid_pairs_l823_82381

theorem no_valid_pairs : 
  ¬∃ (a b x y : ℤ), 
    (a * x + b * y = 3) ∧ 
    (x^2 + y^2 = 85) ∧ 
    (3 * a - 5 * b = 0) := by
  sorry

end no_valid_pairs_l823_82381


namespace train_length_l823_82388

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) (h1 : speed_kmh = 90) (h2 : cross_time_s = 10) :
  speed_kmh * (1000 / 3600) * cross_time_s = 250 := by
  sorry

#check train_length

end train_length_l823_82388


namespace cost_effective_purchase_anton_offer_is_best_l823_82347

/-- Represents a shareholder in the company -/
structure Shareholder where
  name : String
  shares : Nat
  sellPrice : Rat

/-- Represents the company and its shareholders -/
structure Company where
  totalShares : Nat
  sharePrice : Nat
  shareholders : List Shareholder

/-- Calculates the cost of buying shares from a shareholder -/
def buyCost (shareholder : Shareholder) : Rat :=
  shareholder.shares * shareholder.sellPrice

/-- Checks if a shareholder has enough shares to be the largest -/
def isLargestShareholder (company : Company) (shares : Nat) : Prop :=
  ∀ s : Shareholder, s ∈ company.shareholders → shares > s.shares

/-- The main theorem to prove -/
theorem cost_effective_purchase (company : Company) : Prop :=
  let arina : Shareholder := { name := "Arina", shares := 90001, sellPrice := 10 }
  let anton : Shareholder := { name := "Anton", shares := 15000, sellPrice := 14 }
  let arinaNewShares := arina.shares + anton.shares
  isLargestShareholder company arinaNewShares ∧
  ∀ s : Shareholder, s ∈ company.shareholders → s.name ≠ "Arina" →
    buyCost anton ≤ buyCost s ∨ ¬(isLargestShareholder company (arina.shares + s.shares))

/-- The company instance with given conditions -/
def jscCompany : Company := {
  totalShares := 300000,
  sharePrice := 10,
  shareholders := [
    { name := "Arina", shares := 90001, sellPrice := 10 },
    { name := "Maxim", shares := 104999, sellPrice := 11 },
    { name := "Inga", shares := 30000, sellPrice := 12.5 },
    { name := "Yuri", shares := 30000, sellPrice := 11.5 },
    { name := "Yulia", shares := 30000, sellPrice := 13 },
    { name := "Anton", shares := 15000, sellPrice := 14 }
  ]
}

/-- The main theorem applied to our specific company -/
theorem anton_offer_is_best : cost_effective_purchase jscCompany := by
  sorry


end cost_effective_purchase_anton_offer_is_best_l823_82347


namespace equation_simplification_l823_82321

theorem equation_simplification (x : ℝ) :
  x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2 ↔ 10 * x / 3 = 1 + (12 - 3 * x) / 2 :=
by sorry

end equation_simplification_l823_82321


namespace four_inch_cube_value_l823_82313

/-- Represents the properties of a gold cube -/
structure GoldCube where
  edge : ℝ  -- Edge length in inches
  weight : ℝ  -- Weight in pounds
  value : ℝ  -- Value in dollars

/-- The properties of gold cubes are directly proportional to their volume -/
axiom prop_proportional_to_volume (c1 c2 : GoldCube) :
  c2.weight = c1.weight * (c2.edge / c1.edge)^3 ∧
  c2.value = c1.value * (c2.edge / c1.edge)^3

/-- Given information about a one-inch gold cube -/
def one_inch_cube : GoldCube :=
  { edge := 1
  , weight := 0.5
  , value := 1000 }

/-- Theorem: A four-inch cube of gold is worth $64000 -/
theorem four_inch_cube_value :
  ∃ (c : GoldCube), c.edge = 4 ∧ c.value = 64000 :=
sorry

end four_inch_cube_value_l823_82313


namespace sum_of_digits_888_base8_l823_82349

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits of the base 8 representation of 888₁₀ is 13 -/
theorem sum_of_digits_888_base8 : sumDigits (toBase8 888) = 13 := by
  sorry

end sum_of_digits_888_base8_l823_82349


namespace range_of_f_l823_82317

def f (x : ℝ) : ℝ := x^2 + 1

theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end range_of_f_l823_82317


namespace max_value_of_expression_l823_82307

theorem max_value_of_expression (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → w + x + y + z = 200 → 
  w * z + x * y + z * x ≤ 7500 :=
by
  sorry

end max_value_of_expression_l823_82307


namespace yang_hui_theorem_l823_82354

theorem yang_hui_theorem (a b : ℝ) 
  (sum : a + b = 3)
  (product : a * b = 1)
  (sum_squares : a^2 + b^2 = 7)
  (sum_cubes : a^3 + b^3 = 18)
  (sum_fourth_powers : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 := by sorry

end yang_hui_theorem_l823_82354


namespace sum_of_integers_with_lcm_gcd_l823_82334

theorem sum_of_integers_with_lcm_gcd (m n : ℕ) : 
  m > 50 → 
  n > 50 → 
  Nat.lcm m n = 480 → 
  Nat.gcd m n = 12 → 
  m + n = 156 := by
sorry

end sum_of_integers_with_lcm_gcd_l823_82334


namespace point_relationship_l823_82324

theorem point_relationship (b y₁ y₂ : ℝ) 
  (h1 : y₁ = -(-2) + b) 
  (h2 : y₂ = -(3) + b) : 
  y₁ > y₂ := by
sorry

end point_relationship_l823_82324


namespace area_of_rotated_squares_l823_82385

/-- Represents a square sheet of paper -/
structure Square :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping squares -/
structure OverlappingSquares :=
  (base : Square)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)

/-- Calculates the area of the 24-sided polygon formed by the overlapping squares -/
def area_of_polygon (config : OverlappingSquares) : ℝ :=
  sorry

theorem area_of_rotated_squares :
  let config := OverlappingSquares.mk (Square.mk 8) (20 * π / 180) (45 * π / 180)
  area_of_polygon config = 192 := by
  sorry

end area_of_rotated_squares_l823_82385


namespace smallest_prime_after_seven_nonprimes_l823_82304

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ i : ℕ, i < count → ¬(is_prime (start + i))

theorem smallest_prime_after_seven_nonprimes :
  (is_prime 97) ∧
  (consecutive_nonprimes 90 7) ∧
  (∀ p : ℕ, p < 97 → is_prime p → ¬(∃ start : ℕ, start < p ∧ consecutive_nonprimes start 7)) :=
by sorry

end smallest_prime_after_seven_nonprimes_l823_82304


namespace arithmetic_sequence_property_l823_82370

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 240) :
  a 9 - (1/3) * a 11 = 32 := by
  sorry

end arithmetic_sequence_property_l823_82370


namespace parabola_directrix_l823_82383

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 2*x + 1) / 8

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -2

/-- Theorem: The directrix of the given parabola is y = -2 -/
theorem parabola_directrix : ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.1 - x)^2 + (p.2 - d)^2) :=
by sorry

end parabola_directrix_l823_82383


namespace distance_between_foci_l823_82366

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

-- Define the foci
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 29 := by sorry

end distance_between_foci_l823_82366


namespace calculation_proof_l823_82310

theorem calculation_proof : 99 * (5/8) - 0.625 * 68 + 6.25 * 0.1 = 20 := by
  sorry

end calculation_proof_l823_82310


namespace convergence_condition_l823_82323

/-- The iteration function for calculating 1/a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (2 - a * x)

/-- The sequence generated by the iteration -/
def iterSeq (a : ℝ) (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f a (iterSeq a x₀ n)

theorem convergence_condition (a : ℝ) (x₀ : ℝ) (h : a > 0) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |iterSeq a x₀ n - 1/a| < ε) ↔ (0 < x₀ ∧ x₀ < 2/a) :=
sorry

end convergence_condition_l823_82323


namespace bakery_storage_ratio_l823_82322

/-- Given the conditions of a bakery's storage room, prove that the ratio of sugar to flour is 1 to 1. -/
theorem bakery_storage_ratio : ∀ (sugar flour baking_soda : ℝ),
  sugar = 2400 →
  flour = 10 * baking_soda →
  flour = 8 * (baking_soda + 60) →
  sugar / flour = 1 := by
  sorry

end bakery_storage_ratio_l823_82322


namespace lunch_sales_calculation_l823_82302

/-- Represents the number of hot dogs served by a restaurant -/
structure HotDogSales where
  total : ℕ
  dinner : ℕ
  lunch : ℕ

/-- Given the total number of hot dogs sold and the number sold during dinner,
    calculate the number of hot dogs sold during lunch -/
def lunchSales (sales : HotDogSales) : ℕ :=
  sales.total - sales.dinner

theorem lunch_sales_calculation (sales : HotDogSales) 
  (h1 : sales.total = 11)
  (h2 : sales.dinner = 2) :
  lunchSales sales = 9 := by
sorry

end lunch_sales_calculation_l823_82302


namespace order_total_price_l823_82352

/-- Calculate the total price of an order given the number of ice-cream bars, number of sundaes,
    price per ice-cream bar, and price per sundae. -/
def total_price (ice_cream_bars : ℕ) (sundaes : ℕ) (price_ice_cream : ℚ) (price_sundae : ℚ) : ℚ :=
  ice_cream_bars * price_ice_cream + sundaes * price_sundae

/-- Theorem stating that the total price of the order is $200 given the specific quantities and prices. -/
theorem order_total_price :
  total_price 225 125 (60/100) (52/100) = 200 := by
  sorry

end order_total_price_l823_82352


namespace fibonacci_divisibility_l823_82380

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisibility (m : ℕ) :
  ∃ k : ℕ, m ∣ (fibonacci k)^4 - (fibonacci k) - 2 := by
  sorry

end fibonacci_divisibility_l823_82380


namespace sequence_sum_equals_9972_l823_82360

def otimes (m n : ℕ) : ℤ := m * m - n * n

def sequence_sum : ℤ :=
  otimes 2 4 - otimes 4 6 - otimes 6 8 - otimes 8 10 - otimes 10 12 - otimes 12 14 - 
  otimes 14 16 - otimes 16 18 - otimes 18 20 - otimes 20 22 - otimes 22 24 - 
  otimes 24 26 - otimes 26 28 - otimes 28 30 - otimes 30 32 - otimes 32 34 - 
  otimes 34 36 - otimes 36 38 - otimes 38 40 - otimes 40 42 - otimes 42 44 - 
  otimes 44 46 - otimes 46 48 - otimes 48 50 - otimes 50 52 - otimes 52 54 - 
  otimes 54 56 - otimes 56 58 - otimes 58 60 - otimes 60 62 - otimes 62 64 - 
  otimes 64 66 - otimes 66 68 - otimes 68 70 - otimes 70 72 - otimes 72 74 - 
  otimes 74 76 - otimes 76 78 - otimes 78 80 - otimes 80 82 - otimes 82 84 - 
  otimes 84 86 - otimes 86 88 - otimes 88 90 - otimes 90 92 - otimes 92 94 - 
  otimes 94 96 - otimes 96 98 - otimes 98 100

theorem sequence_sum_equals_9972 : sequence_sum = 9972 := by
  sorry

end sequence_sum_equals_9972_l823_82360


namespace college_students_count_l823_82358

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 135) :
  boys + girls = 351 := by
  sorry

end college_students_count_l823_82358


namespace hyperbola_eccentricity_l823_82338

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (p a b m n : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  let parabola := fun x y => y^2 = 2*p*x
  let hyperbola := fun x y => x^2/a^2 - y^2/b^2 = 1
  let focus : ℝ × ℝ := (p/2, 0)
  let A : ℝ × ℝ := (p/2, p)
  let B : ℝ × ℝ := (p/2, -p)
  let M : ℝ × ℝ := (p/2, b^2/a)
  (∀ x y, parabola x y → hyperbola x y → (x = p/2 ∧ y = 0)) →
  (m + n = 1) →
  (m - n = b^2/(a*p)) →
  (m * n = 1/8) →
  let e := c/a
  let c := Real.sqrt (a^2 + b^2)
  e = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end hyperbola_eccentricity_l823_82338


namespace samuel_food_drinks_spending_l823_82316

def total_budget : ℕ := 20
def ticket_cost : ℕ := 14
def kevin_drinks : ℕ := 2
def kevin_food : ℕ := 4

theorem samuel_food_drinks_spending :
  ∀ (samuel_food_drinks : ℕ),
    samuel_food_drinks = total_budget - ticket_cost →
    kevin_drinks + kevin_food + ticket_cost = total_budget →
    samuel_food_drinks = 6 := by
  sorry

end samuel_food_drinks_spending_l823_82316


namespace min_participants_correct_l823_82373

/-- Represents a participant in the race -/
inductive Participant
| Andrei
| Dima
| Lenya
| Other

/-- Represents the race results -/
def RaceResult := List Participant

/-- Checks if the race result satisfies the given conditions -/
def satisfiesConditions (result : RaceResult) : Prop :=
  let n := result.length
  ∃ (a d l : Nat),
    a + 1 + 2 * a = n ∧
    d + 1 + 3 * d = n ∧
    l + 1 + 4 * l = n ∧
    a ≠ d ∧ a ≠ l ∧ d ≠ l

/-- The minimum number of participants in the race -/
def minParticipants : Nat := 61

theorem min_participants_correct :
  ∃ (result : RaceResult),
    result.length = minParticipants ∧
    satisfiesConditions result ∧
    ∀ (result' : RaceResult),
      satisfiesConditions result' →
      result'.length ≥ minParticipants :=
sorry

end min_participants_correct_l823_82373


namespace max_profit_at_70_optimal_selling_price_l823_82395

def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 500
def price_increment : ℕ := 1
def sales_volume_decrement : ℕ := 10

def profit (x : ℕ) : ℤ :=
  (initial_sales_volume - sales_volume_decrement * x) * (initial_selling_price + x) -
  (initial_sales_volume - sales_volume_decrement * x) * purchase_price

theorem max_profit_at_70 :
  ∀ x : ℕ, x ≤ 50 → profit x ≤ profit 20 := by sorry

theorem optimal_selling_price :
  ∃ x : ℕ, x ≤ 50 ∧ ∀ y : ℕ, y ≤ 50 → profit y ≤ profit x :=
by
  use 20
  sorry

#eval initial_selling_price + 20

end max_profit_at_70_optimal_selling_price_l823_82395


namespace driver_speed_problem_l823_82335

theorem driver_speed_problem (v : ℝ) : 
  (v * 1 = (v + 18) * (2/3)) → v = 36 :=
by sorry

end driver_speed_problem_l823_82335


namespace fish_market_customers_l823_82391

theorem fish_market_customers (num_tuna : ℕ) (tuna_weight : ℕ) (customer_want : ℕ) (unserved : ℕ) : 
  num_tuna = 10 → 
  tuna_weight = 200 → 
  customer_want = 25 → 
  unserved = 20 → 
  (num_tuna * tuna_weight) / customer_want + unserved = 100 := by
sorry

end fish_market_customers_l823_82391


namespace combined_net_earnings_proof_l823_82376

def connor_hourly_rate : ℝ := 7.20
def connor_hours : ℝ := 8
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def emily_hours : ℝ := 10
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate + connor_hourly_rate
def sarah_hours : ℝ := connor_hours

def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

def connor_gross_earnings : ℝ := connor_hourly_rate * connor_hours
def emily_gross_earnings : ℝ := emily_hourly_rate * emily_hours
def sarah_gross_earnings : ℝ := sarah_hourly_rate * sarah_hours

def connor_net_earnings : ℝ := connor_gross_earnings * (1 - connor_deduction_rate)
def emily_net_earnings : ℝ := emily_gross_earnings * (1 - emily_deduction_rate)
def sarah_net_earnings : ℝ := sarah_gross_earnings * (1 - sarah_deduction_rate)

def combined_net_earnings : ℝ := connor_net_earnings + emily_net_earnings + sarah_net_earnings

theorem combined_net_earnings_proof : combined_net_earnings = 498.24 := by
  sorry

end combined_net_earnings_proof_l823_82376


namespace kates_hair_length_l823_82329

theorem kates_hair_length (logan_hair emily_hair kate_hair : ℝ) : 
  logan_hair = 20 →
  emily_hair = logan_hair + 6 →
  kate_hair = emily_hair / 2 →
  kate_hair = 13 :=
by
  sorry

end kates_hair_length_l823_82329


namespace cube_volume_relation_l823_82351

theorem cube_volume_relation (V : ℝ) : 
  (∃ (s : ℝ), V = s^3 ∧ 512 = (2*s)^3) → V = 64 := by
  sorry

end cube_volume_relation_l823_82351


namespace chord_length_l823_82312

/-- The length of the chord intercepted by a line on a circle -/
theorem chord_length (x y : ℝ) : 
  let circle := {(x, y) | x^2 + y^2 - 2*x - 4*y = 0}
  let line := {(x, y) | x + 2*y - 5 + Real.sqrt 5 = 0}
  let chord := circle ∩ line
  (∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 4) :=
by
  sorry


end chord_length_l823_82312


namespace system_solution_l823_82356

/-- Given a system of equations and a partial solution, prove the complete solution -/
theorem system_solution (a : ℝ) :
  (∃ x y : ℝ, 2*x + y = a ∧ 2*x - y = 12 ∧ x = 5) →
  (∃ x y : ℝ, 2*x + y = a ∧ 2*x - y = 12 ∧ x = 5 ∧ y = -2 ∧ a = 8) :=
by sorry

end system_solution_l823_82356


namespace prob_queens_or_jacks_l823_82325

/-- The probability of drawing either all three queens or at least 2 jacks from 3 cards in a standard deck -/
theorem prob_queens_or_jacks (total_cards : Nat) (num_queens : Nat) (num_jacks : Nat) 
  (h1 : total_cards = 52)
  (h2 : num_queens = 4)
  (h3 : num_jacks = 4) : 
  (Nat.choose num_queens 3) / (Nat.choose total_cards 3) + 
  (Nat.choose num_jacks 2 * (total_cards - num_jacks) + Nat.choose num_jacks 3) / (Nat.choose total_cards 3) = 290 / 5525 := by
  sorry

end prob_queens_or_jacks_l823_82325


namespace interior_angles_increase_l823_82342

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: If a convex polygon with n sides has a sum of interior angles of 3240 degrees,
    then a convex polygon with n + 3 sides has a sum of interior angles of 3780 degrees. -/
theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 3240 → sum_interior_angles (n + 3) = 3780 := by
  sorry

end interior_angles_increase_l823_82342


namespace arithmetic_sequence_formula_l823_82336

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (n : ℕ) : 
  (a 1 = 1) → 
  (a 2 = 3) → 
  (a 3 = 5) → 
  (a 4 = 7) → 
  (a 5 = 9) → 
  (∀ k : ℕ, a (k + 1) - a k = 2) → 
  a n = 2 * n - 1 := by
sorry

end arithmetic_sequence_formula_l823_82336


namespace value_of_r_l823_82318

theorem value_of_r (a b m p r : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  r = 49/6 := by
sorry

end value_of_r_l823_82318


namespace quadratic_equations_solutions_l823_82392

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  (∃ x : ℝ, 2*x^2 - 7*x + 5 = 0 ↔ x = 5/2 ∨ x = 1) ∧
  (∃ x : ℝ, (x + 3)^2 - 2*(x + 3) = 0 ↔ x = -3 ∨ x = -1) :=
by sorry

end quadratic_equations_solutions_l823_82392


namespace cubes_with_three_painted_faces_l823_82389

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_outside : Bool

/-- Represents a smaller cube cut from a larger cube -/
structure SmallCube where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Function to count the number of painted faces of a small cube -/
def count_painted_faces (c : Cube 4) (sc : SmallCube) : ℕ :=
  sorry

/-- Function to count the number of small cubes with at least three painted faces -/
def count_cubes_with_three_painted_faces (c : Cube 4) : ℕ :=
  sorry

/-- Theorem: In a 4x4x4 cube that is fully painted on the outside and then cut into 1x1x1 cubes,
    the number of 1x1x1 cubes with at least three faces painted is equal to 8 -/
theorem cubes_with_three_painted_faces (c : Cube 4) (h : c.painted_outside = true) :
  count_cubes_with_three_painted_faces c = 8 :=
by sorry

end cubes_with_three_painted_faces_l823_82389


namespace power_58_digits_l823_82364

theorem power_58_digits (n : ℤ) :
  ¬ (10^63 ≤ n^58 ∧ n^58 < 10^64) ∧
  ∀ k : ℕ, k ≤ 81 → ¬ (10^(k-1) ≤ n^58 ∧ n^58 < 10^k) ∧
  ∃ m : ℤ, 10^81 ≤ m^58 ∧ m^58 < 10^82 :=
by sorry

end power_58_digits_l823_82364


namespace cow_husk_consumption_l823_82339

/-- Given that 34 cows eat 34 bags of husk in 34 days, prove that one cow will eat one bag of husk in 34 days. -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 34 ∧ bags = 34 ∧ days = 34) :
  (1 : ℕ) * days = 34 := by
  sorry

end cow_husk_consumption_l823_82339


namespace sum_is_composite_l823_82390

theorem sum_is_composite (a b c d : ℕ) (h : a^2 + b^2 = c^2 + d^2) :
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ a + b + c + d = k * m :=
by sorry

end sum_is_composite_l823_82390


namespace wrench_sales_profit_l823_82374

theorem wrench_sales_profit (selling_price : ℝ) : 
  selling_price > 0 →
  let profit_percent : ℝ := 0.25
  let loss_percent : ℝ := 0.15
  let cost_price1 : ℝ := selling_price / (1 + profit_percent)
  let cost_price2 : ℝ := selling_price / (1 - loss_percent)
  let total_cost : ℝ := cost_price1 + cost_price2
  let total_revenue : ℝ := 2 * selling_price
  let net_gain : ℝ := total_revenue - total_cost
  net_gain / selling_price = 0.028 :=
by sorry

end wrench_sales_profit_l823_82374


namespace min_sum_floor_l823_82309

theorem min_sum_floor (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a + b) / c⌋ + ⌊(b + c) / d⌋ + ⌊(c + a) / b⌋ + ⌊(d + a) / c⌋ ≥ 5 := by
  sorry

end min_sum_floor_l823_82309


namespace trader_theorem_l823_82355

def trader_problem (profit goal donations : ℕ) : Prop :=
  let half_profit := profit / 2
  let total_available := half_profit + donations
  total_available - goal = 180

theorem trader_theorem : trader_problem 960 610 310 := by
  sorry

end trader_theorem_l823_82355


namespace polynomial_division_remainder_l823_82301

theorem polynomial_division_remainder : ∀ (x : ℝ), ∃ (q : ℝ), 2*x^2 - 17*x + 47 = (x - 5) * q + 12 := by
  sorry

end polynomial_division_remainder_l823_82301


namespace min_socks_for_fifteen_pairs_l823_82379

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : Nat)
  (green : Nat)
  (blue : Nat)
  (black : Nat)

/-- The minimum number of socks needed to ensure at least n pairs -/
def minSocksForPairs (drawer : SockDrawer) (n : Nat) : Nat :=
  sorry

theorem min_socks_for_fifteen_pairs :
  let drawer := SockDrawer.mk 120 100 70 50
  minSocksForPairs drawer 15 = 33 := by
  sorry

end min_socks_for_fifteen_pairs_l823_82379


namespace jordan_rectangle_length_l823_82382

/-- Given two rectangles with equal areas, where one rectangle measures 15 inches by 20 inches
    and the other has a width of 50 inches, prove that the length of the second rectangle is 6 inches. -/
theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℝ)
    (carol_area jordan_area : ℝ) (h1 : carol_length = 15)
    (h2 : carol_width = 20) (h3 : jordan_width = 50)
    (h4 : carol_area = carol_length * carol_width)
    (h5 : jordan_area = jordan_width * 6)
    (h6 : carol_area = jordan_area) : 6 = jordan_area / jordan_width := by
  sorry

end jordan_rectangle_length_l823_82382


namespace size_relationship_l823_82367

theorem size_relationship (a b : ℚ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hab : |a| > |b|) : 
  -a < -b ∧ -b < b ∧ b < a := by
  sorry

end size_relationship_l823_82367


namespace largest_number_l823_82365

theorem largest_number (a b c d e : ℝ) 
  (ha : a = 0.993) 
  (hb : b = 0.9899) 
  (hc : c = 0.990) 
  (hd : d = 0.989) 
  (he : e = 0.9909) : 
  a > b ∧ a > c ∧ a > d ∧ a > e :=
sorry

end largest_number_l823_82365


namespace emily_spent_234_l823_82375

/-- The cost of Charlie's purchase of 4 burgers and 3 sodas -/
def charlie_cost : ℝ := 4.40

/-- The cost of Dana's purchase of 3 burgers and 4 sodas -/
def dana_cost : ℝ := 3.80

/-- The number of burgers in Charlie's purchase -/
def charlie_burgers : ℕ := 4

/-- The number of sodas in Charlie's purchase -/
def charlie_sodas : ℕ := 3

/-- The number of burgers in Dana's purchase -/
def dana_burgers : ℕ := 3

/-- The number of sodas in Dana's purchase -/
def dana_sodas : ℕ := 4

/-- The number of burgers in Emily's purchase -/
def emily_burgers : ℕ := 2

/-- The number of sodas in Emily's purchase -/
def emily_sodas : ℕ := 1

/-- The cost of a single burger -/
noncomputable def burger_cost : ℝ := 
  (charlie_cost * dana_sodas - dana_cost * charlie_sodas) / 
  (charlie_burgers * dana_sodas - dana_burgers * charlie_sodas)

/-- The cost of a single soda -/
noncomputable def soda_cost : ℝ := 
  (charlie_cost * dana_burgers - dana_cost * charlie_burgers) / 
  (charlie_sodas * dana_burgers - dana_sodas * charlie_burgers)

/-- Emily's total cost -/
noncomputable def emily_cost : ℝ := emily_burgers * burger_cost + emily_sodas * soda_cost

theorem emily_spent_234 : ∃ ε > 0, |emily_cost - 2.34| < ε :=
sorry

end emily_spent_234_l823_82375


namespace tina_money_left_is_40_l823_82348

/-- Calculates the amount of money Tina has left after savings and expenses -/
def tina_money_left (june_savings july_savings august_savings book_expense shoe_expense : ℕ) : ℕ :=
  (june_savings + july_savings + august_savings) - (book_expense + shoe_expense)

/-- Theorem stating that Tina has $40 left given her savings and expenses -/
theorem tina_money_left_is_40 :
  tina_money_left 27 14 21 5 17 = 40 := by
  sorry

#eval tina_money_left 27 14 21 5 17

end tina_money_left_is_40_l823_82348


namespace pencil_sales_l823_82337

/-- The number of pencils initially sold for a rupee -/
def N : ℕ := 20

/-- The cost price of one pencil -/
def C : ℚ := 1 / 13

/-- Theorem stating that N pencils sold for a rupee results in a 35% loss
    and 10 pencils sold for a rupee results in a 30% gain -/
theorem pencil_sales (N : ℕ) (C : ℚ) :
  (N : ℚ) * (0.65 * C) = 1 ∧ 10 * (1.3 * C) = 1 → N = 20 :=
by sorry

end pencil_sales_l823_82337


namespace repetend_of_four_seventeenths_l823_82305

/-- The decimal representation of 4/17 has a repeating block of 235294117647 -/
theorem repetend_of_four_seventeenths : 
  ∃ (n : ℕ), (4 : ℚ) / 17 = (n : ℚ) / 999999999999 ∧ n = 235294117647 := by
  sorry

end repetend_of_four_seventeenths_l823_82305


namespace multiple_of_z_l823_82378

theorem multiple_of_z (x y z k : ℕ+) : 
  (3 * x.val = 4 * y.val) → 
  (3 * x.val = k * z.val) → 
  (x.val - y.val + z.val = 19) → 
  (∀ (x' y' z' : ℕ+), 3 * x'.val = 4 * y'.val → 3 * x'.val = k * z'.val → x'.val - y'.val + z'.val ≥ 19) →
  k = 12 := by
sorry

end multiple_of_z_l823_82378


namespace black_card_probability_l823_82327

theorem black_card_probability (total_cards : ℕ) (black_cards : ℕ) 
  (h_total : total_cards = 52) 
  (h_black : black_cards = 17) : 
  (black_cards * (black_cards - 1) * (black_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 40 / 1301 := by
  sorry

end black_card_probability_l823_82327


namespace oliver_shelves_l823_82362

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((total_books - books_taken) + (books_per_shelf - 1)) / books_per_shelf

theorem oliver_shelves :
  shelves_needed 46 10 4 = 9 := by
  sorry

end oliver_shelves_l823_82362


namespace both_knights_l823_82399

-- Define the Person type
inductive Person : Type
| A : Person
| B : Person

-- Define the property of being a knight
def is_knight (p : Person) : Prop := sorry

-- Define A's statement
def A_statement : Prop :=
  ¬(is_knight Person.A) ∨ is_knight Person.B

-- Theorem: If A's statement is true, then both A and B are knights
theorem both_knights (h : A_statement) :
  is_knight Person.A ∧ is_knight Person.B := by
  sorry

end both_knights_l823_82399
