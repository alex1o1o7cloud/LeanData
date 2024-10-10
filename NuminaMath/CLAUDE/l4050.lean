import Mathlib

namespace investment_change_l4050_405016

theorem investment_change (initial_investment : ℝ) 
  (loss_rate1 loss_rate3 gain_rate2 : ℝ) : 
  initial_investment = 200 →
  loss_rate1 = 0.1 →
  gain_rate2 = 0.15 →
  loss_rate3 = 0.05 →
  let year1 := initial_investment * (1 - loss_rate1)
  let year2 := year1 * (1 + gain_rate2)
  let year3 := year2 * (1 - loss_rate3)
  let percent_change := (year3 - initial_investment) / initial_investment * 100
  ∃ ε > 0, |percent_change + 1.68| < ε :=
by sorry

end investment_change_l4050_405016


namespace harriet_miles_run_l4050_405017

/-- Proves that given four runners who ran a combined total of 195 miles,
    with one runner running 51 miles and the other three runners running equal distances,
    each of the other three runners ran 48 miles. -/
theorem harriet_miles_run (total_miles : ℕ) (katarina_miles : ℕ) (other_runners : ℕ) :
  total_miles = 195 →
  katarina_miles = 51 →
  other_runners = 3 →
  ∃ (harriet_miles : ℕ),
    harriet_miles * other_runners = total_miles - katarina_miles ∧
    harriet_miles = 48 := by
  sorry

end harriet_miles_run_l4050_405017


namespace ralphs_initial_cards_l4050_405053

theorem ralphs_initial_cards (cards_from_father cards_after : ℕ) :
  cards_from_father = 8 →
  cards_after = 12 →
  cards_after - cards_from_father = 4 :=
by
  sorry

end ralphs_initial_cards_l4050_405053


namespace third_quiz_score_is_92_l4050_405088

/-- Given the average score of three quizzes and the average score of the first two quizzes,
    calculates the score of the third quiz. -/
def third_quiz_score (avg_three : ℚ) (avg_two : ℚ) : ℚ :=
  3 * avg_three - 2 * avg_two

/-- Theorem stating that given the specific average scores,
    the third quiz score is 92. -/
theorem third_quiz_score_is_92 :
  third_quiz_score 94 95 = 92 := by
  sorry

end third_quiz_score_is_92_l4050_405088


namespace sum_equals_eight_l4050_405030

theorem sum_equals_eight (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq1 : b * (a + b + c) + a * c ≥ 16)
  (h_ineq2 : a + 2 * b + c ≤ 8) : 
  a + 2 * b + c = 8 := by
  sorry

end sum_equals_eight_l4050_405030


namespace triangle_side_length_l4050_405039

theorem triangle_side_length (a b : ℝ) (C : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : C = 2 * π / 3) :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  c = 7 := by sorry

end triangle_side_length_l4050_405039


namespace four_inequalities_true_l4050_405058

theorem four_inequalities_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x < a) (hyb : y < b) 
  (hxneg : x < 0) (hyneg : y < 0)
  (hapos : a > 0) (hbpos : b > 0) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
by sorry

end four_inequalities_true_l4050_405058


namespace even_cube_plus_20_divisible_by_48_l4050_405075

theorem even_cube_plus_20_divisible_by_48 (k : ℤ) : 
  ∃ (n : ℤ), 8 * k * (k^2 + 5) = 48 * n := by
  sorry

end even_cube_plus_20_divisible_by_48_l4050_405075


namespace ratio_of_percentages_l4050_405059

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.75 * P) : 
  M / N = 2 / 15 := by
  sorry

end ratio_of_percentages_l4050_405059


namespace intersection_M_N_l4050_405036

def M : Set ℝ := {x | x^2 + x - 2 = 0}
def N : Set ℝ := {x | x < 0}

theorem intersection_M_N : M ∩ N = {-2} := by sorry

end intersection_M_N_l4050_405036


namespace six_point_five_minutes_in_seconds_l4050_405018

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℝ) : ℝ := minutes * 60

/-- Theorem stating that 6.5 minutes equals 390 seconds -/
theorem six_point_five_minutes_in_seconds : 
  minutes_to_seconds 6.5 = 390 := by sorry

end six_point_five_minutes_in_seconds_l4050_405018


namespace paint_cost_theorem_l4050_405080

-- Define the paint properties
structure Paint where
  cost : Float
  coverage : Float

-- Define the cuboid dimensions
def cuboid_length : Float := 12
def cuboid_width : Float := 15
def cuboid_height : Float := 20

-- Define the paints
def paint_A : Paint := { cost := 3.20, coverage := 60 }
def paint_B : Paint := { cost := 5.50, coverage := 55 }
def paint_C : Paint := { cost := 4.00, coverage := 50 }

-- Calculate the areas of the faces
def largest_face_area : Float := 2 * cuboid_width * cuboid_height
def middle_face_area : Float := 2 * cuboid_length * cuboid_height
def smallest_face_area : Float := 2 * cuboid_length * cuboid_width

-- Calculate the number of quarts needed for each paint
def quarts_A : Float := Float.ceil (largest_face_area / paint_A.coverage)
def quarts_B : Float := Float.ceil (middle_face_area / paint_B.coverage)
def quarts_C : Float := Float.ceil (smallest_face_area / paint_C.coverage)

-- Calculate the total cost
def total_cost : Float := quarts_A * paint_A.cost + quarts_B * paint_B.cost + quarts_C * paint_C.cost

-- Theorem to prove
theorem paint_cost_theorem : total_cost = 113.50 := by
  sorry

end paint_cost_theorem_l4050_405080


namespace spheres_radius_is_half_l4050_405015

/-- A cube with side length 2 containing eight congruent spheres --/
structure SpheresInCube where
  -- The side length of the cube
  cube_side : ℝ
  -- The radius of each sphere
  sphere_radius : ℝ
  -- The number of spheres
  num_spheres : ℕ
  -- Condition that the cube side length is 2
  cube_side_is_two : cube_side = 2
  -- Condition that there are 8 spheres
  eight_spheres : num_spheres = 8
  -- Condition that spheres are tangent to three faces and neighboring spheres
  spheres_tangent : True  -- This is a simplification, as we can't easily express this geometric condition

/-- Theorem stating that the radius of each sphere is 1/2 --/
theorem spheres_radius_is_half (s : SpheresInCube) : s.sphere_radius = 1/2 := by
  sorry

end spheres_radius_is_half_l4050_405015


namespace parabola_property_l4050_405063

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix l
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define the perpendicular condition
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  directrix A.1 ∧ (P.2 = A.2)

-- Define the slope condition for AF
def slope_AF_is_neg_one (A : ℝ × ℝ) : Prop :=
  (A.2 - focus.2) / (A.1 - focus.1) = -1

theorem parabola_property :
  ∀ P : ℝ × ℝ,
  point_on_parabola P →
  ∃ A : ℝ × ℝ,
  perpendicular_to_directrix P A ∧
  slope_AF_is_neg_one A →
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 4 :=
sorry

end parabola_property_l4050_405063


namespace fraction_simplification_complex_fraction_simplification_l4050_405081

-- Problem 1
theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (a + b)) = (a^2 + b^2) / (a^2 - b^2) :=
sorry

-- Problem 2
theorem complex_fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ((x - 2) / (x - 1)) / ((x^2 - 4*x + 4) / (x^2 - 1)) + ((1 - x) / (x - 2)) = 2 / (x - 2) :=
sorry

end fraction_simplification_complex_fraction_simplification_l4050_405081


namespace hyperbolas_same_asymptotes_l4050_405047

/-- Given two hyperbolas with equations x^2/16 - y^2/25 = 1 and y^2/50 - x^2/M = 1,
    if they have the same asymptotes, then M = 32 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ y^2/50 - x^2/M = 1) →
  M = 32 := by
  sorry

end hyperbolas_same_asymptotes_l4050_405047


namespace regular_polygon_sides_l4050_405082

-- Define the properties of the polygon
def perimeter : ℝ := 150
def side_length : ℝ := 15

-- Theorem statement
theorem regular_polygon_sides : 
  perimeter / side_length = 10 := by
  sorry

end regular_polygon_sides_l4050_405082


namespace cross_section_perimeter_bounds_l4050_405028

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A triangular cross-section through a vertex of a regular tetrahedron -/
structure TriangularCrossSection (a : ℝ) (t : RegularTetrahedron a) where
  perimeter : ℝ

/-- The perimeter of any triangular cross-section through a vertex of a regular tetrahedron
    with edge length a satisfies 2a < P ≤ 3a -/
theorem cross_section_perimeter_bounds (a : ℝ) (t : RegularTetrahedron a) 
  (s : TriangularCrossSection a t) : 2 * a < s.perimeter ∧ s.perimeter ≤ 3 * a := by
  sorry


end cross_section_perimeter_bounds_l4050_405028


namespace negation_equivalence_l4050_405087

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Adult : U → Prop)
variable (GoodCook : U → Prop)

-- Define the statements
def AllAdultsAreGoodCooks : Prop := ∀ x, Adult x → GoodCook x
def AtLeastOneAdultIsBadCook : Prop := ∃ x, Adult x ∧ ¬GoodCook x

-- Theorem statement
theorem negation_equivalence : 
  AtLeastOneAdultIsBadCook U Adult GoodCook ↔ ¬(AllAdultsAreGoodCooks U Adult GoodCook) :=
sorry

end negation_equivalence_l4050_405087


namespace lcm_from_product_and_hcf_l4050_405040

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h1 : a * b = 62216)
  (h2 : Nat.gcd a b = 22) : 
  Nat.lcm a b = 2828 := by
sorry

end lcm_from_product_and_hcf_l4050_405040


namespace value_of_y_l4050_405006

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 14 ∧ y = 98 / 3 := by
  sorry

end value_of_y_l4050_405006


namespace min_value_theorem_l4050_405098

theorem min_value_theorem (x : ℝ) (h : x > 6) :
  x^2 / (x - 6) ≥ 18 ∧ (x^2 / (x - 6) = 18 ↔ x = 12) := by
  sorry

end min_value_theorem_l4050_405098


namespace grandmas_salad_l4050_405072

/-- The number of mushrooms Grandma put on her salad -/
def mushrooms : ℕ := sorry

/-- The number of cherry tomatoes Grandma put on her salad -/
def cherry_tomatoes : ℕ := 2 * mushrooms

/-- The number of pickles Grandma put on her salad -/
def pickles : ℕ := 4 * cherry_tomatoes

/-- The total number of bacon bits Grandma put on her salad -/
def bacon_bits : ℕ := 4 * pickles

/-- The number of red bacon bits Grandma put on her salad -/
def red_bacon_bits : ℕ := 32

theorem grandmas_salad : mushrooms = 3 := by
  sorry

end grandmas_salad_l4050_405072


namespace min_value_expression_l4050_405000

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 := by
sorry

end min_value_expression_l4050_405000


namespace correct_operations_l4050_405010

theorem correct_operations (a b : ℝ) : 
  (2 * a * (3 * b) = 6 * a * b) ∧ ((-a^3)^2 = a^6) := by
  sorry

end correct_operations_l4050_405010


namespace profit_sharing_ratio_l4050_405066

/-- Represents the capital contribution and duration for a business partner -/
structure Partner where
  capital : ℕ
  duration : ℕ

/-- Calculates the effective capital contribution of a partner -/
def effectiveCapital (p : Partner) : ℕ := p.capital * p.duration

/-- Represents the business scenario with two partners -/
structure Business where
  partnerA : Partner
  partnerB : Partner

/-- The given business scenario -/
def givenBusiness : Business :=
  { partnerA := { capital := 3500, duration := 12 }
  , partnerB := { capital := 21000, duration := 3 }
  }

/-- Theorem stating that the profit sharing ratio is 2:3 for the given business -/
theorem profit_sharing_ratio (b : Business := givenBusiness) :
  (effectiveCapital b.partnerA) * 3 = (effectiveCapital b.partnerB) * 2 := by
  sorry

end profit_sharing_ratio_l4050_405066


namespace inequality_proof_l4050_405019

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := by
  sorry

end inequality_proof_l4050_405019


namespace function_machine_output_l4050_405054

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 25 then
    step1 - 7
  else
    (step1 + 3) * 2

theorem function_machine_output : function_machine 12 = 78 := by
  sorry

end function_machine_output_l4050_405054


namespace james_has_more_balloons_l4050_405078

/-- James has 1222 balloons -/
def james_balloons : ℕ := 1222

/-- Amy has 513 balloons -/
def amy_balloons : ℕ := 513

/-- The difference in balloon count between James and Amy -/
def balloon_difference : ℕ := james_balloons - amy_balloons

/-- Theorem stating that James has 709 more balloons than Amy -/
theorem james_has_more_balloons : balloon_difference = 709 := by
  sorry

end james_has_more_balloons_l4050_405078


namespace viewer_increase_l4050_405093

/-- The number of people who watched the second baseball game -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first baseball game -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The number of people who watched the third baseball game -/
def third_game_viewers : ℕ := second_game_viewers + 15

/-- The total number of people who watched the games last week -/
def last_week_viewers : ℕ := 200

/-- The total number of people who watched the games this week -/
def this_week_viewers : ℕ := first_game_viewers + second_game_viewers + third_game_viewers

theorem viewer_increase :
  this_week_viewers - last_week_viewers = 35 := by
  sorry

end viewer_increase_l4050_405093


namespace magnitude_of_sum_l4050_405020

/-- Given two vectors a and b in ℝ², prove that under certain conditions, 
    the magnitude of a + 2b is √29. -/
theorem magnitude_of_sum (a b : ℝ × ℝ) : 
  (a.1 = 4 ∧ a.2 = 3) → -- a = (4, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → -- a ⟂ b (dot product is 0)
  (b.1^2 + b.2^2 = 1) → -- |b| = 1
  ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2 = 29) := by
sorry

end magnitude_of_sum_l4050_405020


namespace fib_mod_5_periodic_fib_10_mod_5_fib_50_mod_5_l4050_405069

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_mod_5_periodic (n : ℕ) : fib n % 5 = fib (n % 20) % 5 := sorry

theorem fib_10_mod_5 : fib 10 % 5 = 0 := sorry

theorem fib_50_mod_5 : fib 50 % 5 = 0 := by
  sorry

end fib_mod_5_periodic_fib_10_mod_5_fib_50_mod_5_l4050_405069


namespace dress_design_count_l4050_405060

/-- The number of fabric colors available -/
def num_colors : ℕ := 3

/-- The number of fabric types available -/
def num_fabric_types : ℕ := 4

/-- The number of patterns available -/
def num_patterns : ℕ := 3

/-- Each dress design requires exactly one color, one fabric type, and one pattern -/
axiom dress_design_requirements : True

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_fabric_types * num_patterns

theorem dress_design_count : total_designs = 36 := by
  sorry

end dress_design_count_l4050_405060


namespace givenPoint_on_y_axis_l4050_405055

/-- A point in the Cartesian coordinate system. -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on the y-axis. -/
def OnYAxis (p : CartesianPoint) : Prop :=
  p.x = 0

/-- The given point (0, -1) in the Cartesian coordinate system. -/
def givenPoint : CartesianPoint :=
  ⟨0, -1⟩

/-- Theorem stating that the given point lies on the y-axis. -/
theorem givenPoint_on_y_axis : OnYAxis givenPoint := by
  sorry

end givenPoint_on_y_axis_l4050_405055


namespace batsman_average_after_17th_inning_l4050_405086

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an additional inning -/
def newAverage (bp : BatsmanPerformance) (newRuns : Nat) : Rat :=
  (bp.totalRuns + newRuns) / (bp.innings + 1)

/-- Theorem: Given the conditions, prove that the batsman's average after the 17th inning is 39 -/
theorem batsman_average_after_17th_inning 
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 16)
  (h2 : newAverage bp 87 = bp.average + 3)
  : newAverage bp 87 = 39 := by
  sorry

#check batsman_average_after_17th_inning

end batsman_average_after_17th_inning_l4050_405086


namespace sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l4050_405013

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b :=
by sorry

theorem sqrt_three_times_sqrt_two_equals_sqrt_six : 
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 :=
by sorry

end sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l4050_405013


namespace twelfth_term_of_arithmetic_progression_l4050_405048

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- Theorem: The 12th term of an arithmetic progression with first term 2 and common difference 8 is 90 -/
theorem twelfth_term_of_arithmetic_progression :
  arithmeticProgressionTerm 2 8 12 = 90 := by
  sorry

end twelfth_term_of_arithmetic_progression_l4050_405048


namespace line_parameterization_l4050_405023

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 10t - 12), 
    prove that g(t) = 5t + 14 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y t : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 10*t - 12) → 
  (∀ t : ℝ, g t = 5*t + 14) := by
sorry

end line_parameterization_l4050_405023


namespace factory_production_time_l4050_405014

/-- The number of dolls produced by the factory -/
def num_dolls : ℕ := 12000

/-- The number of shoes per doll -/
def shoes_per_doll : ℕ := 2

/-- The number of bags per doll -/
def bags_per_doll : ℕ := 3

/-- The number of cosmetics sets per doll -/
def cosmetics_per_doll : ℕ := 1

/-- The number of hats per doll -/
def hats_per_doll : ℕ := 5

/-- The time in seconds to make one doll -/
def time_per_doll : ℕ := 45

/-- The time in seconds to make one accessory -/
def time_per_accessory : ℕ := 10

/-- The total combined machine operation time for manufacturing all dolls and accessories -/
def total_time : ℕ := 1860000

theorem factory_production_time : 
  num_dolls * time_per_doll + 
  num_dolls * (shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll) * time_per_accessory = 
  total_time := by sorry

end factory_production_time_l4050_405014


namespace freds_remaining_balloons_l4050_405027

/-- The number of green balloons Fred has after giving some away -/
def remaining_balloons (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Fred's remaining balloons equals the difference between initial and given away -/
theorem freds_remaining_balloons :
  remaining_balloons 709 221 = 488 := by
  sorry

end freds_remaining_balloons_l4050_405027


namespace sum_evaluation_l4050_405001

theorem sum_evaluation : 
  (4 : ℚ) / 3 + 7 / 6 + 13 / 12 + 25 / 24 + 49 / 48 + 97 / 96 - 8 = -43 / 32 := by
  sorry

end sum_evaluation_l4050_405001


namespace course_selection_schemes_l4050_405064

/-- The number of elective courses in physical education -/
def pe_courses : ℕ := 4

/-- The number of elective courses in art -/
def art_courses : ℕ := 4

/-- The minimum number of courses a student can choose -/
def min_courses : ℕ := 2

/-- The maximum number of courses a student can choose -/
def max_courses : ℕ := 3

/-- The minimum number of courses a student must choose from each category -/
def min_per_category : ℕ := 1

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := 64

/-- Theorem stating that the total number of different course selection schemes is 64 -/
theorem course_selection_schemes :
  (pe_courses = 4) →
  (art_courses = 4) →
  (min_courses = 2) →
  (max_courses = 3) →
  (min_per_category = 1) →
  (total_schemes = 64) :=
by sorry

end course_selection_schemes_l4050_405064


namespace prime_condition_characterization_l4050_405005

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The main theorem statement -/
theorem prime_condition_characterization (n : ℕ) :
  (n > 0 ∧ ∀ k : ℕ, k < n → is_prime (4 * k^2 + n)) ↔ (n = 3 ∨ n = 7) :=
sorry

end prime_condition_characterization_l4050_405005


namespace equation_solution_l4050_405045

theorem equation_solution (x : ℝ) : 
  1 - 6/x + 9/x^2 - 4/x^3 = 0 → (3/x = 3 ∨ 3/x = 3/4) :=
by
  sorry

end equation_solution_l4050_405045


namespace mike_limes_l4050_405021

-- Define the given conditions
def total_limes : ℕ := 57
def alyssa_limes : ℕ := 25

-- State the theorem
theorem mike_limes : total_limes - alyssa_limes = 32 := by
  sorry

end mike_limes_l4050_405021


namespace betty_blue_beads_l4050_405094

/-- Given a ratio of red to blue beads and a number of red beads, calculate the number of blue beads -/
def calculate_blue_beads (red_ratio : ℕ) (blue_ratio : ℕ) (total_red : ℕ) : ℕ :=
  (total_red / red_ratio) * blue_ratio

/-- Theorem: Given Betty's bead ratio and total red beads, prove she has 20 blue beads -/
theorem betty_blue_beads :
  let red_ratio : ℕ := 3
  let blue_ratio : ℕ := 2
  let total_red : ℕ := 30
  calculate_blue_beads red_ratio blue_ratio total_red = 20 := by
  sorry

#eval calculate_blue_beads 3 2 30

end betty_blue_beads_l4050_405094


namespace arithmetic_sequence_properties_l4050_405062

def a (n : ℕ+) : ℤ := 2^n.val - (-1)^n.val

theorem arithmetic_sequence_properties :
  (∃ n : ℕ+, a n + a (n + 2) = 2 * a (n + 1) ∧ n = 2) ∧
  (∃ n₂ n₃ : ℕ+, n₂ < n₃ ∧ a 1 + a n₃ = 2 * a n₂ ∧ n₃ - n₂ = 1) ∧
  (∀ t : ℕ+, t > 3 →
    ¬∃ (n : Fin t → ℕ+), (∀ i j : Fin t, i < j → n i < n j) ∧
      (∀ i : Fin (t - 2), 2 * a (n (i + 1)) = a (n i) + a (n (i + 2)))) :=
by sorry

end arithmetic_sequence_properties_l4050_405062


namespace rich_book_pages_left_to_read_l4050_405073

/-- Given a book with a total number of pages, the number of pages already read,
    and the number of pages to be skipped, calculate the number of pages left to read. -/
def pages_left_to_read (total_pages read_pages skipped_pages : ℕ) : ℕ :=
  total_pages - (read_pages + skipped_pages)

/-- Theorem stating that for a 372-page book with 125 pages read and 16 pages skipped,
    there are 231 pages left to read. -/
theorem rich_book_pages_left_to_read :
  pages_left_to_read 372 125 16 = 231 := by
  sorry

end rich_book_pages_left_to_read_l4050_405073


namespace profit_share_difference_is_1000_l4050_405032

/-- Represents the profit share calculation for a business partnership --/
structure BusinessPartnership where
  investment_a : ℕ
  investment_b : ℕ
  investment_c : ℕ
  profit_share_b : ℕ

/-- Calculates the difference between profit shares of partners C and A --/
def profit_share_difference (bp : BusinessPartnership) : ℕ :=
  let total_investment := bp.investment_a + bp.investment_b + bp.investment_c
  let total_profit := bp.profit_share_b * total_investment / bp.investment_b
  let share_a := total_profit * bp.investment_a / total_investment
  let share_c := total_profit * bp.investment_c / total_investment
  share_c - share_a

/-- Theorem stating that for the given investments and B's profit share, 
    the difference between C's and A's profit shares is 1000 --/
theorem profit_share_difference_is_1000 :
  profit_share_difference ⟨8000, 10000, 12000, 2500⟩ = 1000 := by
  sorry

end profit_share_difference_is_1000_l4050_405032


namespace equiangular_iff_rectangle_l4050_405044

-- Define a quadrilateral
class Quadrilateral :=
(angles : Fin 4 → ℝ)

-- Define an equiangular quadrilateral
def is_equiangular (q : Quadrilateral) : Prop :=
∀ i j : Fin 4, q.angles i = q.angles j

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
∀ i : Fin 4, q.angles i = 90

-- Theorem statement
theorem equiangular_iff_rectangle (q : Quadrilateral) : 
  is_equiangular q ↔ is_rectangle q :=
sorry

end equiangular_iff_rectangle_l4050_405044


namespace cylinder_radius_problem_l4050_405033

theorem cylinder_radius_problem (r : ℝ) :
  (r > 0) →
  (5 * π * (r + 4)^2 = 15 * π * r^2) →
  (r = 2 + 2 * Real.sqrt 3) :=
by sorry

end cylinder_radius_problem_l4050_405033


namespace fuel_consumption_rate_l4050_405003

/-- Given a plane with a certain amount of fuel and remaining flight time,
    calculate the rate of fuel consumption per hour. -/
theorem fuel_consumption_rate (fuel_left : ℝ) (time_left : ℝ) :
  fuel_left = 6.3333 →
  time_left = 0.6667 →
  ∃ (rate : ℝ), abs (rate - (fuel_left / time_left)) < 0.01 ∧ abs (rate - 9.5) < 0.01 :=
by sorry

end fuel_consumption_rate_l4050_405003


namespace sqrt_x_minus_2_meaningful_l4050_405022

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_meaningful_l4050_405022


namespace intersection_A_B_l4050_405049

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}

def B : Set ℝ := {-2, -1, 1, 2, 4}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end intersection_A_B_l4050_405049


namespace equation_solution_l4050_405034

theorem equation_solution : ∃! x : ℚ, 
  (1 : ℚ) / ((x + 12)^2) + (1 : ℚ) / ((x + 8)^2) = 
  (1 : ℚ) / ((x + 13)^2) + (1 : ℚ) / ((x + 7)^2) ∧ 
  x = -15/2 := by
  sorry

end equation_solution_l4050_405034


namespace expansion_terms_product_l4050_405052

/-- The number of terms in the expansion of a product of two polynomials -/
def expansion_terms (n m : ℕ) : ℕ := n * m

theorem expansion_terms_product (n m : ℕ) (h1 : n = 3) (h2 : m = 5) :
  expansion_terms n m = 15 := by
  sorry

#check expansion_terms_product

end expansion_terms_product_l4050_405052


namespace nuts_mixed_with_raisins_l4050_405037

/-- The number of pounds of nuts mixed with raisins -/
def pounds_of_nuts : ℝ := 4

/-- The number of pounds of raisins -/
def pounds_of_raisins : ℝ := 5

/-- The cost ratio of nuts to raisins -/
def cost_ratio : ℝ := 3

/-- The fraction of the total cost that the raisins represent -/
def raisin_cost_fraction : ℝ := 0.29411764705882354

/-- Proves that the number of pounds of nuts mixed with 5 pounds of raisins is 4 -/
theorem nuts_mixed_with_raisins :
  let r := 1  -- Cost of 1 pound of raisins (arbitrary unit)
  let n := cost_ratio * r  -- Cost of 1 pound of nuts
  pounds_of_nuts * n / (pounds_of_nuts * n + pounds_of_raisins * r) = 1 - raisin_cost_fraction :=
by sorry

end nuts_mixed_with_raisins_l4050_405037


namespace tangent_circles_area_sum_l4050_405068

/-- A right triangle with sides 6, 8, and 10, where each vertex is the center of a circle
    and each circle is externally tangent to the other two. -/
structure TangentCirclesTriangle where
  -- The sides of the triangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  -- The radii of the circles
  radius1 : ℝ
  radius2 : ℝ
  radius3 : ℝ
  -- Conditions
  is_right_triangle : side1^2 + side2^2 = side3^2
  side_lengths : side1 = 6 ∧ side2 = 8 ∧ side3 = 10
  tangency1 : radius1 + radius2 = side3
  tangency2 : radius2 + radius3 = side1
  tangency3 : radius1 + radius3 = side2

/-- The sum of the areas of the circles in a TangentCirclesTriangle is 56π. -/
theorem tangent_circles_area_sum (t : TangentCirclesTriangle) :
  π * (t.radius1^2 + t.radius2^2 + t.radius3^2) = 56 * π := by
  sorry

end tangent_circles_area_sum_l4050_405068


namespace fermat_prime_sum_l4050_405012

theorem fermat_prime_sum (n : ℕ) (p : ℕ) (hn : Odd n) (hn1 : n > 1) (hp : Prime p) :
  ¬ ∃ (x y z : ℤ), x^n + y^n = z^n ∧ x + y = p := by
  sorry

end fermat_prime_sum_l4050_405012


namespace hyperbola_eccentricity_l4050_405041

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = k is sqrt(2) -/
theorem hyperbola_eccentricity (k : ℝ) (h : k > 0) :
  let e := Real.sqrt (1 + (Real.sqrt k / Real.sqrt k)^2)
  e = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_l4050_405041


namespace product_equals_result_l4050_405085

theorem product_equals_result : ∃ x : ℝ, 469158 * x = 4691110842 ∧ x = 10000.2 := by
  sorry

end product_equals_result_l4050_405085


namespace ab_value_l4050_405083

theorem ab_value (a b : ℕ+) (h : a^2 + 3*b = 33) : a*b = 24 := by
  sorry

end ab_value_l4050_405083


namespace lisa_marble_distribution_l4050_405026

/-- The minimum number of additional marbles needed -/
def additional_marbles_needed (friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := friends * (friends + 1) / 2
  max (required_marbles - initial_marbles) 0

/-- Theorem stating the solution to Lisa's marble distribution problem -/
theorem lisa_marble_distribution (friends : ℕ) (initial_marbles : ℕ)
    (h1 : friends = 12)
    (h2 : initial_marbles = 50) :
    additional_marbles_needed friends initial_marbles = 28 := by
  sorry

end lisa_marble_distribution_l4050_405026


namespace intersection_distance_implies_k_range_l4050_405095

/-- Given a line y = kx and a circle (x-2)^2 + (y+1)^2 = 4,
    if the distance between their intersection points is at least 2√3,
    then -4/3 ≤ k ≤ 0 -/
theorem intersection_distance_implies_k_range (k : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.2 = k * A.1 ∧ B.2 = k * B.1) ∧
    ((A.1 - 2)^2 + (A.2 + 1)^2 = 4 ∧ (B.1 - 2)^2 + (B.2 + 1)^2 = 4) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ 12)) →
  -4/3 ≤ k ∧ k ≤ 0 := by
  sorry

end intersection_distance_implies_k_range_l4050_405095


namespace sequence_term_proof_l4050_405070

/-- Given a sequence where the sum of the first n terms is 5n + 2n^2,
    this function represents the rth term of the sequence. -/
def sequence_term (r : ℕ) : ℕ := 4 * r + 3

/-- The sum of the first n terms of the sequence. -/
def sequence_sum (n : ℕ) : ℕ := 5 * n + 2 * n^2

/-- Theorem stating that the rth term of the sequence is 4r + 3,
    given that the sum of the first n terms is 5n + 2n^2 for all n. -/
theorem sequence_term_proof (r : ℕ) : 
  sequence_term r = sequence_sum r - sequence_sum (r - 1) :=
sorry

end sequence_term_proof_l4050_405070


namespace stratified_sampling_major_c_l4050_405025

theorem stratified_sampling_major_c (total_students : ℕ) (sample_size : ℕ) 
  (major_a_students : ℕ) (major_b_students : ℕ) : 
  total_students = 1200 →
  sample_size = 120 →
  major_a_students = 380 →
  major_b_students = 420 →
  (total_students - major_a_students - major_b_students) * sample_size / total_students = 40 := by
  sorry

end stratified_sampling_major_c_l4050_405025


namespace problem_1_problem_2_l4050_405043

-- Problem 1
theorem problem_1 : (Real.sqrt 48 - (1/4) * Real.sqrt 6) / (-(1/9) * Real.sqrt 27) = -12 + (3/4) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = (1/2) * (Real.sqrt 3 + 1)) (hy : y = (1/2) * (1 - Real.sqrt 3)) :
  x^2 + y^2 - 2*x*y = 3 := by
  sorry

end problem_1_problem_2_l4050_405043


namespace sqrt_equation_solution_l4050_405077

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (3 + 4 * x) = 7 ∧ x = 11.5) :=
by sorry

end sqrt_equation_solution_l4050_405077


namespace binomial_expansion_coefficient_l4050_405091

/-- The coefficient of x^3 in the expansion of (1+ax)^5 -/
def coefficient_x3 (a : ℝ) : ℝ := 10 * a^3

theorem binomial_expansion_coefficient (a : ℝ) :
  coefficient_x3 a = -80 → a = -2 := by
  sorry

end binomial_expansion_coefficient_l4050_405091


namespace left_handed_classical_music_lovers_l4050_405096

theorem left_handed_classical_music_lovers (total : ℕ) (left_handed : ℕ) (classical_music_lovers : ℕ) (right_handed_non_lovers : ℕ) :
  total = 30 →
  left_handed = 12 →
  classical_music_lovers = 20 →
  right_handed_non_lovers = 3 →
  ∃ (x : ℕ), x = 5 ∧ 
    x + (left_handed - x) + (classical_music_lovers - x) + right_handed_non_lovers = total :=
by sorry

end left_handed_classical_music_lovers_l4050_405096


namespace jills_age_l4050_405065

theorem jills_age (henry_age jill_age : ℕ) : 
  henry_age + jill_age = 43 →
  henry_age - 5 = 2 * (jill_age - 5) →
  jill_age = 16 := by
sorry

end jills_age_l4050_405065


namespace circle_equation_proof_line_equation_proof_l4050_405051

-- Define the points
def A : ℝ × ℝ := (-4, -3)
def B : ℝ × ℝ := (2, 9)
def P : ℝ × ℝ := (0, 2)

-- Define the circle C with AB as its diameter
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 45}

-- Define the line l₀
def l₀ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 2 = 0}

theorem circle_equation_proof :
  C = {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 45} :=
sorry

theorem line_equation_proof :
  l₀ = {p : ℝ × ℝ | p.1 - p.2 + 2 = 0} :=
sorry

end circle_equation_proof_line_equation_proof_l4050_405051


namespace smallest_n_exceeding_1500_l4050_405007

def exterior_sum (n : ℕ) : ℕ := 8 + 24 * (n - 2) + 12 * (n - 2)^2

theorem smallest_n_exceeding_1500 :
  ∀ n : ℕ, n ≥ 13 ↔ exterior_sum n > 1500 :=
by sorry

end smallest_n_exceeding_1500_l4050_405007


namespace valid_speaking_orders_eq_1080_l4050_405038

-- Define the number of candidates
def num_candidates : ℕ := 8

-- Define the number of speakers to be selected
def num_speakers : ℕ := 4

-- Define a function to calculate the number of valid speaking orders
def valid_speaking_orders : ℕ :=
  -- Number of orders where only one of A or B participates
  (Nat.choose 2 1) * (Nat.choose 6 3) * (Nat.factorial 4) +
  -- Number of orders where both A and B participate with one person between them
  (Nat.choose 2 2) * (Nat.choose 6 2) * (Nat.choose 2 1) * (Nat.factorial 2) * (Nat.factorial 2)

-- Theorem stating that the number of valid speaking orders is 1080
theorem valid_speaking_orders_eq_1080 : valid_speaking_orders = 1080 := by
  sorry

end valid_speaking_orders_eq_1080_l4050_405038


namespace a_value_theorem_l4050_405008

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x + 1

theorem a_value_theorem (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((a * Real.log x) + a) x) →
  HasDerivAt (f a) 2 1 →
  a = 2 := by
  sorry

end a_value_theorem_l4050_405008


namespace desired_annual_profit_l4050_405046

def annual_fixed_costs : ℕ := 50200000
def average_cost_per_vehicle : ℕ := 5000
def forecasted_sales : ℕ := 20000
def selling_price_per_car : ℕ := 9035

theorem desired_annual_profit :
  (selling_price_per_car * forecasted_sales) - 
  (annual_fixed_costs + average_cost_per_vehicle * forecasted_sales) = 30500000 := by
  sorry

end desired_annual_profit_l4050_405046


namespace hannah_fair_money_l4050_405050

theorem hannah_fair_money (initial_money : ℝ) : 
  (initial_money / 2 + 5 + 10 = initial_money) → initial_money = 30 := by
  sorry

end hannah_fair_money_l4050_405050


namespace missing_number_is_34_l4050_405089

theorem missing_number_is_34 : 
  ∃ x : ℝ, ((306 / x) * 15 + 270 = 405) ∧ (x = 34) :=
by sorry

end missing_number_is_34_l4050_405089


namespace age_difference_l4050_405071

theorem age_difference (a b c : ℕ) : 
  b = 12 →
  b = 2 * c →
  a + b + c = 32 →
  a = b + 2 :=
by sorry

end age_difference_l4050_405071


namespace fourth_black_ball_probability_l4050_405092

/-- Represents a box of colored balls -/
structure ColoredBallBox where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball on any draw -/
def prob_black_ball (box : ColoredBallBox) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The box described in the problem -/
def problem_box : ColoredBallBox :=
  { red_balls := 3, black_balls := 4 }

theorem fourth_black_ball_probability :
  prob_black_ball problem_box = 4 / 7 := by
  sorry

end fourth_black_ball_probability_l4050_405092


namespace max_value_theorem_l4050_405029

theorem max_value_theorem (a b c : ℝ) (h : a^2 + b^2 + c^2 ≤ 8) :
  4 * (a^3 + b^3 + c^3) - (a^4 + b^4 + c^4) ≤ 48 := by
  sorry

end max_value_theorem_l4050_405029


namespace total_fish_count_l4050_405097

/-- The number of fish in three tanks given specific conditions -/
def total_fish (goldfish1 guppies1 : ℕ) : ℕ :=
  let tank1 := goldfish1 + guppies1
  let tank2 := 2 * goldfish1 + 3 * guppies1
  let tank3 := 3 * goldfish1 + 2 * guppies1
  tank1 + tank2 + tank3

/-- Theorem stating that the total number of fish is 162 given the specific conditions -/
theorem total_fish_count : total_fish 15 12 = 162 := by
  sorry

end total_fish_count_l4050_405097


namespace time_to_reach_room_l4050_405035

theorem time_to_reach_room (total_time gate_time building_time : ℕ) 
  (h1 : total_time = 30)
  (h2 : gate_time = 15)
  (h3 : building_time = 6) :
  total_time - (gate_time + building_time) = 9 := by
  sorry

end time_to_reach_room_l4050_405035


namespace A_is_half_of_B_l4050_405042

def A : ℕ → ℕ
| 0 => 0
| (n + 1) => A n + (n + 1) * (2023 - n)

def B : ℕ → ℕ
| 0 => 0
| (n + 1) => B n + (n + 1) * (2024 - n)

theorem A_is_half_of_B : A 2022 = (B 2022) / 2 := by
  sorry

end A_is_half_of_B_l4050_405042


namespace base_conversion_1729_to_base_7_l4050_405061

theorem base_conversion_1729_to_base_7 :
  (5 * 7^3 + 0 * 7^2 + 2 * 7^1 + 0 * 7^0 : ℕ) = 1729 := by
  sorry

end base_conversion_1729_to_base_7_l4050_405061


namespace diamond_calculation_l4050_405074

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 3 4) 5
  let y := diamond 3 (diamond 4 5)
  x - y = -71 / 380 := by sorry

end diamond_calculation_l4050_405074


namespace pencils_bought_is_three_l4050_405024

/-- Calculates the number of pencils bought given the total paid, cost per pencil, cost of glue, and change received. -/
def number_of_pencils (total_paid change cost_per_pencil cost_of_glue : ℕ) : ℕ :=
  ((total_paid - change - cost_of_glue) / cost_per_pencil)

/-- Proves that the number of pencils bought is 3 under the given conditions. -/
theorem pencils_bought_is_three :
  number_of_pencils 1000 100 210 270 = 3 := by
  sorry

#eval number_of_pencils 1000 100 210 270

end pencils_bought_is_three_l4050_405024


namespace range_of_f_l4050_405099

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x < 5 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -4 ≤ y ∧ y < 5 } := by sorry

end range_of_f_l4050_405099


namespace vector_difference_magnitude_l4050_405076

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (a b : V)

-- State the theorem
theorem vector_difference_magnitude 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = 1) : 
  ‖a - b‖ = Real.sqrt 3 := by
  sorry

end vector_difference_magnitude_l4050_405076


namespace oak_grove_library_books_l4050_405031

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The total number of books in all Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := total_books - public_library_books

theorem oak_grove_library_books : school_library_books = 5106 := by
  sorry

end oak_grove_library_books_l4050_405031


namespace backpack_profit_equation_l4050_405002

/-- Represents the profit calculation for a backpack sale -/
theorem backpack_profit_equation (x : ℝ) : 
  (1 + 0.5) * x * 0.8 - x = 8 ↔ 
  (x > 0 ∧ 
   (1 + 0.5) * x * 0.8 = x + 8) :=
by sorry

#check backpack_profit_equation

end backpack_profit_equation_l4050_405002


namespace logarithm_properties_l4050_405067

-- Define the logarithm function for any base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define lg as log base 10
noncomputable def lg (x : ℝ) : ℝ := log 10 x

theorem logarithm_properties :
  (lg 2 + lg 5 = 1) ∧ (log 3 9 = 2) := by sorry

end logarithm_properties_l4050_405067


namespace function_inequality_l4050_405084

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) (a : ℝ) (ha : a > 0) : 
  f a < Real.exp a * f 0 := by
  sorry

end function_inequality_l4050_405084


namespace root_sum_product_reciprocal_sum_l4050_405011

theorem root_sum_product_reciprocal_sum (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (x₁ x₂ x₃ : ℝ) (hroots : ∀ x, α * x^3 - α * x^2 + β * x + β = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) :
  (x₁ + x₂ + x₃) * (1/x₁ + 1/x₂ + 1/x₃) = -1 := by
  sorry

end root_sum_product_reciprocal_sum_l4050_405011


namespace code_cracking_probability_l4050_405056

theorem code_cracking_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
sorry

end code_cracking_probability_l4050_405056


namespace objects_meet_time_l4050_405004

/-- Two objects moving towards each other meet after 10 seconds -/
theorem objects_meet_time : ∃ t : ℝ, t = 10 ∧ 
  390 = 3 * t^2 + 0.012 * (t - 5) := by sorry

end objects_meet_time_l4050_405004


namespace teacher_grading_problem_l4050_405079

def remaining_problems (problems_per_worksheet : ℕ) (total_worksheets : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem teacher_grading_problem :
  let problems_per_worksheet : ℕ := 3
  let total_worksheets : ℕ := 15
  let graded_worksheets : ℕ := 7
  remaining_problems problems_per_worksheet total_worksheets graded_worksheets = 24 := by
sorry

end teacher_grading_problem_l4050_405079


namespace hyperbola_eccentricity_l4050_405057

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ := λ x => Real.sqrt x
  let h : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  ∃ x₀ y₀ : ℝ,
    x₀ ≥ 0 ∧
    y₀ = f x₀ ∧
    h x₀ y₀ ∧
    (λ x => (f x₀ - 0) / (x₀ - (-1)) * (x - x₀) + f x₀) (-1) = 0 →
    (Real.sqrt (a^2 + b^2)) / a = (Real.sqrt 5 + 1) / 2 :=
sorry

end hyperbola_eccentricity_l4050_405057


namespace square_side_length_from_rectangle_l4050_405009

/-- The side length of a square with an area 7 times larger than a rectangle with length 400 feet and width 300 feet is approximately 916.515 feet. -/
theorem square_side_length_from_rectangle (ε : ℝ) (h : ε > 0) : ∃ (s : ℝ), 
  abs (s - Real.sqrt (7 * 400 * 300)) < ε ∧ 
  s^2 = 7 * 400 * 300 := by
  sorry

end square_side_length_from_rectangle_l4050_405009


namespace negation_equivalence_l4050_405090

theorem negation_equivalence (m : ℝ) :
  (¬ ∃ (x : ℤ), x^2 + x + m < 0) ↔ (∀ (x : ℝ), x^2 + x + m ≥ 0) :=
by sorry

end negation_equivalence_l4050_405090
