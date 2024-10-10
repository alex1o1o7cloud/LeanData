import Mathlib

namespace balloons_in_park_l3430_343003

/-- The number of balloons Allan and Jake had in the park -/
def total_balloons (allan_initial : ℕ) (jake_balloons : ℕ) (allan_bought : ℕ) : ℕ :=
  (allan_initial + allan_bought) + jake_balloons

/-- Theorem stating the total number of balloons Allan and Jake had in the park -/
theorem balloons_in_park : total_balloons 3 5 2 = 10 := by
  sorry

end balloons_in_park_l3430_343003


namespace triangle_angle_ratio_l3430_343024

theorem triangle_angle_ratio (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle
  A = 20 →           -- Smallest angle
  B = 3 * A →        -- Middle angle is 3 times the smallest
  A ≤ B →            -- B is larger than or equal to A
  B ≤ C →            -- C is the largest angle
  C / A = 5 :=       -- Ratio of largest to smallest is 5:1
by sorry

end triangle_angle_ratio_l3430_343024


namespace jacob_jogging_distance_l3430_343089

/-- Calculates the total distance jogged given a constant speed and total jogging time -/
def total_distance_jogged (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that jogging at 4 miles per hour for 3 hours results in a total distance of 12 miles -/
theorem jacob_jogging_distance :
  let speed : ℝ := 4
  let time : ℝ := 3
  total_distance_jogged speed time = 12 := by
  sorry

end jacob_jogging_distance_l3430_343089


namespace fraction_sum_squared_l3430_343015

theorem fraction_sum_squared : 
  (2/10 + 3/100 + 5/1000 + 7/10000)^2 = 0.05555649 := by
  sorry

end fraction_sum_squared_l3430_343015


namespace max_value_theorem_l3430_343067

open Real

noncomputable def e : ℝ := Real.exp 1

theorem max_value_theorem (a b : ℝ) :
  (∀ x : ℝ, (e - a) * (Real.exp x) + x + b + 1 ≤ 0) →
  (b + 1) / a ≤ 1 / e :=
by sorry

end max_value_theorem_l3430_343067


namespace power_fraction_simplification_l3430_343008

theorem power_fraction_simplification :
  (2^2023 + 2^2019) / (2^2023 - 2^2019) = 17 / 15 := by
  sorry

end power_fraction_simplification_l3430_343008


namespace hyperbola_circle_relation_l3430_343033

-- Define the hyperbola
def is_hyperbola (x y : ℝ) : Prop := y^2 - x^2/3 = 1

-- Define a focus of the hyperbola
def is_focus (x y : ℝ) : Prop := x = 0 ∧ (y = 2 ∨ y = -2)

-- Define the eccentricity of the hyperbola
def eccentricity : ℝ := 2

-- Define the circle
def is_circle (x y : ℝ) : Prop := x^2 + (y-2)^2 = 4

-- Theorem statement
theorem hyperbola_circle_relation :
  ∀ (x y cx cy : ℝ),
  is_hyperbola x y →
  is_focus cx cy →
  is_circle (x - cx) (y - cy) :=
sorry

end hyperbola_circle_relation_l3430_343033


namespace trigonometric_equation_solution_l3430_343027

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.cos (x/2) * Real.cos (3*x/2) - Real.sin x * Real.sin (3*x) - Real.sin (2*x) * Real.sin (3*x) = 0) →
  ∃ k : ℤ, x = π/9 * (2*k + 1) := by
sorry

end trigonometric_equation_solution_l3430_343027


namespace collinear_vectors_t_value_l3430_343094

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem collinear_vectors_t_value 
  (h_non_collinear : ¬ ∃ (k : ℝ), a = k • b) 
  (h_collinear : ∃ (k : ℝ), a - t • b = k • (2 • a + b)) : 
  t = -1/2 := by
  sorry

end collinear_vectors_t_value_l3430_343094


namespace fraction_multiplication_equality_l3430_343060

theorem fraction_multiplication_equality : 
  (5 / 8 : ℚ)^2 * (3 / 4 : ℚ)^2 * (2 / 3 : ℚ) = 75 / 512 := by
  sorry

end fraction_multiplication_equality_l3430_343060


namespace star_diameter_scientific_notation_l3430_343032

/-- Represents the diameter of the star in meters -/
def star_diameter : ℝ := 16600000000

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 1.66

/-- Represents the exponent in scientific notation -/
def exponent : ℕ := 10

/-- Theorem stating that the star's diameter is correctly expressed in scientific notation -/
theorem star_diameter_scientific_notation : 
  star_diameter = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end star_diameter_scientific_notation_l3430_343032


namespace alcohol_mixture_proof_l3430_343031

/-- Proves that mixing 200 mL of 10% alcohol solution with 50 mL of 30% alcohol solution 
    results in a 14% alcohol solution -/
theorem alcohol_mixture_proof (x_vol : ℝ) (y_vol : ℝ) (x_conc : ℝ) (y_conc : ℝ) 
    (mix_conc : ℝ) (h1 : x_vol = 200) (h2 : y_vol = 50) (h3 : x_conc = 0.1) 
    (h4 : y_conc = 0.3) (h5 : mix_conc = 0.14) : 
    (x_vol * x_conc + y_vol * y_conc) / (x_vol + y_vol) = mix_conc := by
  sorry

#check alcohol_mixture_proof

end alcohol_mixture_proof_l3430_343031


namespace four_similar_triangle_solutions_l3430_343092

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Define a line
structure Line :=
  (m b : ℝ)

-- Function to check if a point is on a side of a triangle
def isPointOnSide (T : Triangle) (P : Point) : Prop :=
  sorry

-- Function to check if two triangles are similar
def areSimilarTriangles (T1 T2 : Triangle) : Prop :=
  sorry

-- Function to check if a line intersects a triangle
def lineIntersectsTriangle (L : Line) (T : Triangle) : Prop :=
  sorry

-- Function to get the triangle cut off by a line
def getCutOffTriangle (T : Triangle) (L : Line) : Triangle :=
  sorry

-- The main theorem
theorem four_similar_triangle_solutions 
  (T : Triangle) (P : Point) (h : isPointOnSide T P) :
  ∃ (L1 L2 L3 L4 : Line),
    (L1 ≠ L2 ∧ L1 ≠ L3 ∧ L1 ≠ L4 ∧ L2 ≠ L3 ∧ L2 ≠ L4 ∧ L3 ≠ L4) ∧
    (∀ (L : Line), 
      (lineIntersectsTriangle L T ∧ areSimilarTriangles (getCutOffTriangle T L) T) →
      (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4)) :=
sorry

end four_similar_triangle_solutions_l3430_343092


namespace lcm_of_45_and_200_l3430_343080

theorem lcm_of_45_and_200 : Nat.lcm 45 200 = 1800 := by
  sorry

end lcm_of_45_and_200_l3430_343080


namespace olympiad_problem_distribution_l3430_343071

theorem olympiad_problem_distribution (n : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : n = 30) 
  (h2 : m = 40) 
  (h3 : k = 5) 
  (h4 : ∃ (x y z q r : ℕ), 
    x + y + z + q + r = n ∧ 
    x + 2*y + 3*z + 4*q + 5*r = m ∧ 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ q > 0 ∧ r > 0) :
  ∃ (x : ℕ), x = 26 ∧ 
    ∃ (y z q r : ℕ), 
      x + y + z + q + r = n ∧ 
      x + 2*y + 3*z + 4*q + 5*r = m ∧
      y = 1 ∧ z = 1 ∧ q = 1 ∧ r = 1 := by
  sorry

end olympiad_problem_distribution_l3430_343071


namespace complex_fraction_evaluation_l3430_343075

theorem complex_fraction_evaluation : 
  (1 : ℚ) / (1 - 1 / (3 + 1 / 4)) = 13 / 9 := by sorry

end complex_fraction_evaluation_l3430_343075


namespace sqrt5_parts_sqrt2_plus_1_parts_sqrt3_plus_2_parts_l3430_343002

-- Define the irrational numbers
axiom sqrt2 : ℝ
axiom sqrt3 : ℝ
axiom sqrt5 : ℝ

-- Define the properties of these irrational numbers
axiom sqrt2_irrational : Irrational sqrt2
axiom sqrt3_irrational : Irrational sqrt3
axiom sqrt5_irrational : Irrational sqrt5

axiom sqrt2_bounds : 1 < sqrt2 ∧ sqrt2 < 2
axiom sqrt3_bounds : 1 < sqrt3 ∧ sqrt3 < 2
axiom sqrt5_bounds : 2 < sqrt5 ∧ sqrt5 < 3

-- Define the integer and decimal part functions
def intPart (x : ℝ) : ℤ := sorry
def decPart (x : ℝ) : ℝ := sorry

-- Theorem statements
theorem sqrt5_parts : intPart sqrt5 = 2 ∧ decPart sqrt5 = sqrt5 - 2 := by sorry

theorem sqrt2_plus_1_parts : intPart (1 + sqrt2) = 2 ∧ decPart (1 + sqrt2) = sqrt2 - 1 := by sorry

theorem sqrt3_plus_2_parts :
  let x := intPart (2 + sqrt3)
  let y := decPart (2 + sqrt3)
  x - sqrt3 * y = sqrt3 := by sorry

end sqrt5_parts_sqrt2_plus_1_parts_sqrt3_plus_2_parts_l3430_343002


namespace statements_correctness_l3430_343049

theorem statements_correctness :
  (∃ a b : ℝ, a > b ∧ 1/a > 1/b ∧ a*b ≤ 0) ∧
  (∀ a b c : ℝ, a > b ∧ b > 0 ∧ c < 0 → c/a > c/b) ∧
  (∃ a b : ℝ, a < b ∧ b < 0 ∧ a^2 ≥ b^2) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → a*c < b*d) :=
by sorry

end statements_correctness_l3430_343049


namespace right_triangle_circle_ratio_l3430_343084

theorem right_triangle_circle_ratio (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let R := c / 2
  let r := (a + b - c) / 2
  R / r = 5 / 2 := by sorry

end right_triangle_circle_ratio_l3430_343084


namespace ned_good_games_l3430_343099

/-- The number of good games Ned ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - non_working_games

/-- Proof that Ned ended up with 3 good games -/
theorem ned_good_games :
  good_games 50 27 74 = 3 := by
  sorry

end ned_good_games_l3430_343099


namespace trig_inequality_l3430_343054

theorem trig_inequality (θ : Real) (h : π < θ ∧ θ < 5 * π / 4) :
  Real.cos θ < Real.sin θ ∧ Real.sin θ < Real.tan θ := by
  sorry

end trig_inequality_l3430_343054


namespace x_value_l3430_343005

def A : Set ℝ := {1, 2, 3}
def B (x : ℝ) : Set ℝ := {1, x}

theorem x_value (x : ℝ) : A ∪ B x = A → x = 2 ∨ x = 3 := by
  sorry

end x_value_l3430_343005


namespace veridux_female_managers_l3430_343062

/-- Calculates the number of female managers given the total number of employees,
    female employees, total managers, and male associates. -/
def female_managers (total_employees : ℕ) (female_employees : ℕ) (total_managers : ℕ) (male_associates : ℕ) : ℕ :=
  total_managers - (total_employees - female_employees - male_associates)

/-- Theorem stating that given the conditions from the problem, 
    the number of female managers is 40. -/
theorem veridux_female_managers :
  female_managers 250 90 40 160 = 40 := by
  sorry

#eval female_managers 250 90 40 160

end veridux_female_managers_l3430_343062


namespace plane_equation_proof_l3430_343000

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  coeff_coprime : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if two planes are parallel -/
def planes_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = 4 ∧ given_plane.B = -2 ∧ given_plane.C = 6 ∧ given_plane.D = 14 →
  point.x = 2 ∧ point.y = -1 ∧ point.z = 3 →
  ∃ (result_plane : Plane),
    point_on_plane point result_plane ∧
    planes_parallel result_plane given_plane ∧
    result_plane.A = 2 ∧ result_plane.B = -1 ∧ result_plane.C = 3 ∧ result_plane.D = -14 :=
by sorry

end plane_equation_proof_l3430_343000


namespace score_difference_l3430_343079

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℝ
  physics : ℝ
  chemistry : ℝ

/-- The problem statement -/
theorem score_difference (s : Scores) 
  (h1 : s.math + s.physics = 20)
  (h2 : (s.math + s.chemistry) / 2 = 20)
  (h3 : s.chemistry > s.physics) :
  s.chemistry - s.physics = 20 := by
  sorry

end score_difference_l3430_343079


namespace tea_party_wait_time_l3430_343096

/-- Mad Hatter's clock speed relative to real time -/
def mad_hatter_clock_speed : ℚ := 5/4

/-- March Hare's clock speed relative to real time -/
def march_hare_clock_speed : ℚ := 5/6

/-- The agreed meeting time on their clocks (in hours after noon) -/
def meeting_time : ℚ := 5

/-- Calculate the real time when someone arrives based on their clock speed -/
def real_arrival_time (clock_speed : ℚ) : ℚ :=
  meeting_time / clock_speed

theorem tea_party_wait_time :
  real_arrival_time march_hare_clock_speed - real_arrival_time mad_hatter_clock_speed = 2 := by
  sorry

end tea_party_wait_time_l3430_343096


namespace cinematic_academy_members_l3430_343047

/-- The minimum fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def min_fraction : ℚ := 1 / 4

/-- The smallest number of top-10 lists a film can appear on and still be considered -/
def min_lists : ℚ := 198.75

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 795

/-- Theorem stating that the number of members in the Cinematic Academy is 795 -/
theorem cinematic_academy_members :
  academy_members = ⌈(min_lists / min_fraction : ℚ)⌉ := by
  sorry

end cinematic_academy_members_l3430_343047


namespace sunflower_seed_distribution_l3430_343006

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) :
  total_seeds = 54 →
  num_cans = 9 →
  total_seeds = num_cans * seeds_per_can →
  seeds_per_can = 6 :=
by
  sorry

end sunflower_seed_distribution_l3430_343006


namespace log_inequality_l3430_343052

theorem log_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c) :
  Real.log c / Real.log a > Real.log c / Real.log b :=
by sorry

end log_inequality_l3430_343052


namespace tv_discount_percentage_l3430_343030

def original_price : ℚ := 480
def first_installment : ℚ := 150
def num_monthly_installments : ℕ := 3
def monthly_installment : ℚ := 102

def total_payment : ℚ := first_installment + (monthly_installment * num_monthly_installments)
def discount : ℚ := original_price - total_payment
def discount_percentage : ℚ := (discount / original_price) * 100

theorem tv_discount_percentage :
  discount_percentage = 5 := by sorry

end tv_discount_percentage_l3430_343030


namespace fruit_drink_composition_l3430_343082

-- Define the composition of the fruit drink
def orange_percent : ℝ := 25
def watermelon_percent : ℝ := 40
def grape_ounces : ℝ := 70

-- Define the total volume of the drink
def total_volume : ℝ := 200

-- Theorem statement
theorem fruit_drink_composition :
  orange_percent + watermelon_percent + (grape_ounces / total_volume * 100) = 100 ∧
  grape_ounces / (grape_ounces / total_volume * 100) * 100 = total_volume :=
by sorry

end fruit_drink_composition_l3430_343082


namespace peaches_sold_to_relatives_l3430_343004

theorem peaches_sold_to_relatives (total_peaches : ℕ) 
                                  (peaches_to_friends : ℕ) 
                                  (price_to_friends : ℚ)
                                  (price_to_relatives : ℚ)
                                  (peaches_kept : ℕ)
                                  (total_sold : ℕ)
                                  (total_earnings : ℚ) :
  total_peaches = 15 →
  peaches_to_friends = 10 →
  price_to_friends = 2 →
  price_to_relatives = 5/4 →
  peaches_kept = 1 →
  total_sold = 14 →
  total_earnings = 25 →
  total_peaches = peaches_to_friends + (total_sold - peaches_to_friends) + peaches_kept →
  total_earnings = peaches_to_friends * price_to_friends + 
                   (total_sold - peaches_to_friends) * price_to_relatives →
  (total_sold - peaches_to_friends) = 4 := by
sorry

end peaches_sold_to_relatives_l3430_343004


namespace roses_handed_out_l3430_343029

theorem roses_handed_out (total : ℕ) (left : ℕ) (handed_out : ℕ) : 
  total = 29 → left = 12 → handed_out = total - left → handed_out = 17 := by
  sorry

end roses_handed_out_l3430_343029


namespace days_to_complete_paper_l3430_343069

-- Define the paper length and writing rate
def paper_length : ℕ := 63
def pages_per_day : ℕ := 21

-- Theorem statement
theorem days_to_complete_paper : 
  (paper_length / pages_per_day : ℕ) = 3 := by
  sorry

end days_to_complete_paper_l3430_343069


namespace third_month_sale_is_10389_l3430_343009

/-- Calculates the sale in the third month given the sales for other months and the average -/
def third_month_sale (sale1 sale2 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the sale in the third month is 10389 given the conditions -/
theorem third_month_sale_is_10389 :
  third_month_sale 4000 6524 7230 6000 12557 7000 = 10389 := by
  sorry

end third_month_sale_is_10389_l3430_343009


namespace fraction_simplification_complex_fraction_simplification_l3430_343077

theorem fraction_simplification (x y : ℝ) (h : 2 * x ≠ y) :
  (3 * x) / (2 * x - y) - (x + y) / (2 * x - y) = 1 := by sorry

theorem complex_fraction_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x^2 - 5*x) / (x + 2) / ((x - 5) / (x^2 - 4)) = x^2 - 2*x := by sorry

end fraction_simplification_complex_fraction_simplification_l3430_343077


namespace investment_interest_rate_l3430_343044

theorem investment_interest_rate (total_investment : ℝ) (first_part : ℝ) (first_rate : ℝ) (total_interest : ℝ) : 
  total_investment = 3600 →
  first_part = 1800 →
  first_rate = 3 →
  total_interest = 144 →
  (first_part * first_rate / 100 + (total_investment - first_part) * 5 / 100 = total_interest) :=
by sorry

end investment_interest_rate_l3430_343044


namespace expand_product_l3430_343012

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := by
  sorry

end expand_product_l3430_343012


namespace random_sampling_appropriate_for_air_quality_l3430_343016

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| RandomSampling

/-- Represents a scenario for which a survey method is chosen -/
inductive Scenario
| LightBulbLifespan
| FoodPreservatives
| SpaceEquipmentQuality
| AirQuality

/-- Determines if a survey method is appropriate for a given scenario -/
def isAppropriate (method : SurveyMethod) (scenario : Scenario) : Prop :=
  match scenario with
  | Scenario.LightBulbLifespan => method = SurveyMethod.RandomSampling
  | Scenario.FoodPreservatives => method = SurveyMethod.RandomSampling
  | Scenario.SpaceEquipmentQuality => method = SurveyMethod.Comprehensive
  | Scenario.AirQuality => method = SurveyMethod.RandomSampling

/-- Theorem stating that random sampling is appropriate for air quality measurement -/
theorem random_sampling_appropriate_for_air_quality :
  isAppropriate SurveyMethod.RandomSampling Scenario.AirQuality :=
by
  sorry

#check random_sampling_appropriate_for_air_quality

end random_sampling_appropriate_for_air_quality_l3430_343016


namespace exists_increasing_sequence_with_gcd_property_l3430_343018

theorem exists_increasing_sequence_with_gcd_property :
  ∃ (a : ℕ → ℕ), 
    (∀ n : ℕ, a n < a (n + 1)) ∧ 
    (∀ i j : ℕ, i ≠ j → Nat.gcd (i * a j) (j * a i) = Nat.gcd i j) := by
  sorry

end exists_increasing_sequence_with_gcd_property_l3430_343018


namespace distance_formula_l3430_343036

/-- The distance between two points on a real number line -/
def distance (a b : ℝ) : ℝ := |b - a|

/-- Theorem: The distance between two points A and B with coordinates a and b is |b - a| -/
theorem distance_formula (a b : ℝ) : distance a b = |b - a| := by sorry

end distance_formula_l3430_343036


namespace revenue_decrease_percentage_l3430_343001

def old_revenue : ℝ := 85.0
def new_revenue : ℝ := 48.0

theorem revenue_decrease_percentage :
  abs (((old_revenue - new_revenue) / old_revenue) * 100 - 43.53) < 0.01 := by
  sorry

end revenue_decrease_percentage_l3430_343001


namespace apple_cost_is_21_cents_l3430_343041

/-- The cost of an apple and an orange satisfy the given conditions -/
def apple_orange_cost (apple_cost orange_cost : ℚ) : Prop :=
  6 * apple_cost + 3 * orange_cost = 177/100 ∧
  2 * apple_cost + 5 * orange_cost = 127/100

/-- The cost of an apple is 0.21 dollars -/
theorem apple_cost_is_21_cents :
  ∃ (orange_cost : ℚ), apple_orange_cost (21/100) orange_cost := by
  sorry

end apple_cost_is_21_cents_l3430_343041


namespace subtraction_of_decimals_l3430_343056

theorem subtraction_of_decimals : (3.156 : ℝ) - (1.029 : ℝ) = 2.127 := by sorry

end subtraction_of_decimals_l3430_343056


namespace pills_in_week_l3430_343086

/-- Calculates the number of pills taken in a week given the interval between pills in hours -/
def pills_per_week (hours_between_pills : ℕ) : ℕ :=
  let hours_per_day : ℕ := 24
  let days_per_week : ℕ := 7
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem: A person who takes a pill every 6 hours will take 28 pills in a week -/
theorem pills_in_week : pills_per_week 6 = 28 := by
  sorry

end pills_in_week_l3430_343086


namespace rice_distribution_difference_l3430_343043

/-- Given a total amount of rice and the fraction kept by Mr. Llesis,
    calculate how much more rice Mr. Llesis keeps compared to Mr. Everest. -/
def rice_difference (total : ℚ) (llesis_fraction : ℚ) : ℚ :=
  let llesis_amount := total * llesis_fraction
  let everest_amount := total - llesis_amount
  llesis_amount - everest_amount

/-- Theorem stating that given 50 kg of rice, if Mr. Llesis keeps 7/10 of it,
    he will have 20 kg more than Mr. Everest. -/
theorem rice_distribution_difference :
  rice_difference 50 (7/10) = 20 := by
  sorry

end rice_distribution_difference_l3430_343043


namespace exists_line_with_specified_length_l3430_343025

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the intersection points
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the circles
def S₁ : Circle := sorry
def S₂ : Circle := sorry

-- Specify that the circles intersect at A and B
axiom intersect_at_A : A ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2} ∩
                           {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}
axiom intersect_at_B : B ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2} ∩
                           {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}

-- Define a line passing through point A
def line_through_A (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 - A.2 = m * (p.1 - A.1)}

-- Define the segment of a line contained within both circles
def segment_in_circles (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  l ∩ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 ≤ S₁.radius^2} ∩
       {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 ≤ S₂.radius^2}

-- Define the length of a segment
def segment_length (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem exists_line_with_specified_length (length : ℝ) :
  ∃ m : ℝ, segment_length (segment_in_circles (line_through_A m)) = length :=
sorry

end exists_line_with_specified_length_l3430_343025


namespace exists_same_color_rectangle_l3430_343087

/-- A color represented as an enumeration -/
inductive Color
  | Red
  | Green
  | Blue

/-- A grid coloring is a function from grid coordinates to colors -/
def GridColoring := (Fin 4 × Fin 82) → Color

/-- A rectangle is represented by four points in the grid -/
structure Rectangle :=
  (p1 p2 p3 p4 : Fin 4 × Fin 82)

/-- Predicate to check if all vertices of a rectangle have the same color -/
def SameColorRectangle (coloring : GridColoring) (rect : Rectangle) : Prop :=
  coloring rect.p1 = coloring rect.p2 ∧
  coloring rect.p1 = coloring rect.p3 ∧
  coloring rect.p1 = coloring rect.p4

/-- Main theorem: There exists a rectangle with vertices of the same color in any 4x82 grid coloring -/
theorem exists_same_color_rectangle (coloring : GridColoring) :
  ∃ (rect : Rectangle), SameColorRectangle coloring rect := by
  sorry


end exists_same_color_rectangle_l3430_343087


namespace abs_eq_iff_eq_l3430_343037

theorem abs_eq_iff_eq (x y : ℝ) : 
  (|x| = |y| → x = y) ↔ False ∧ 
  (x = y → |x| = |y|) :=
sorry

end abs_eq_iff_eq_l3430_343037


namespace books_total_is_140_l3430_343098

/-- The number of books Beatrix has -/
def beatrix_books : ℕ := 30

/-- The number of books Alannah has -/
def alannah_books : ℕ := beatrix_books + 20

/-- The number of books Queen has -/
def queen_books : ℕ := alannah_books + alannah_books / 5

/-- The total number of books all three have together -/
def total_books : ℕ := beatrix_books + alannah_books + queen_books

theorem books_total_is_140 : total_books = 140 := by
  sorry

end books_total_is_140_l3430_343098


namespace oscar_swag_bag_value_l3430_343073

/-- The total value of a swag bag with specified items -/
def swag_bag_value (earring_cost : ℕ) (iphone_cost : ℕ) (scarf_cost : ℕ) : ℕ :=
  2 * earring_cost + iphone_cost + 4 * scarf_cost

/-- Theorem: The total value of the Oscar swag bag is $20,000 -/
theorem oscar_swag_bag_value :
  swag_bag_value 6000 2000 1500 = 20000 := by
  sorry

end oscar_swag_bag_value_l3430_343073


namespace third_term_coefficient_binomial_expansion_l3430_343039

theorem third_term_coefficient_binomial_expansion :
  let a := x
  let b := -1 / (2 * x)
  let n := 6
  let k := 2  -- Third term corresponds to k = 2
  (Nat.choose n k : ℚ) * a^(n - k) * b^k = 15/4 := by
  sorry

end third_term_coefficient_binomial_expansion_l3430_343039


namespace teacher_selection_problem_l3430_343091

/-- The number of ways to select k items from n items --/
def permutation (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- The number of valid selections of teachers --/
def validSelections (totalTeachers maleTeachers femaleTeachers selectCount : ℕ) : ℕ :=
  permutation totalTeachers selectCount - 
  (permutation maleTeachers selectCount + permutation femaleTeachers selectCount)

theorem teacher_selection_problem :
  validSelections 9 5 4 3 = 420 := by
  sorry

end teacher_selection_problem_l3430_343091


namespace initial_student_count_l3430_343020

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 15 →
  new_avg = 14.4 →
  new_student_weight = 3 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 :=
by
  sorry

end initial_student_count_l3430_343020


namespace hiker_catchup_time_l3430_343022

/-- Proves that a hiker catches up to a motorcyclist in 48 minutes under given conditions -/
theorem hiker_catchup_time (hiker_speed : ℝ) (motorcyclist_speed : ℝ) (stop_time : ℝ) : 
  hiker_speed = 6 →
  motorcyclist_speed = 30 →
  stop_time = 12 / 60 →
  (motorcyclist_speed * stop_time - hiker_speed * stop_time) / hiker_speed * 60 = 48 := by
sorry

end hiker_catchup_time_l3430_343022


namespace median_equal_mean_l3430_343010

def set_elements (n : ℝ) : List ℝ := [n, n+4, n+7, n+10, n+14]

theorem median_equal_mean (n : ℝ) (h : n + 7 = 14) : 
  (List.sum (set_elements n)) / (List.length (set_elements n)) = 14 := by
  sorry

end median_equal_mean_l3430_343010


namespace kennel_dogs_l3430_343053

theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 3 / 4 →
  cats = dogs - 8 →
  dogs = 32 := by
sorry

end kennel_dogs_l3430_343053


namespace train_length_l3430_343040

/-- Given a train that crosses an electric pole in 2.5 seconds at a speed of 144 km/hr,
    prove that its length is 100 meters. -/
theorem train_length (crossing_time : Real) (speed_kmh : Real) (length : Real) : 
  crossing_time = 2.5 →
  speed_kmh = 144 →
  length = speed_kmh * (1000 / 3600) * crossing_time →
  length = 100 := by
  sorry

#check train_length

end train_length_l3430_343040


namespace suresh_work_time_l3430_343090

theorem suresh_work_time (S : ℝ) (h1 : S > 0) : 
  (∃ (ashutosh_time : ℝ), 
    ashutosh_time = 35 ∧ 
    (9 / S) + (14 / ashutosh_time) = 1) → 
  S = 15 := by
sorry

end suresh_work_time_l3430_343090


namespace lcm_of_36_and_176_l3430_343051

theorem lcm_of_36_and_176 :
  let a : ℕ := 36
  let b : ℕ := 176
  let hcf : ℕ := 16
  Nat.gcd a b = hcf →
  Nat.lcm a b = 396 := by
sorry

end lcm_of_36_and_176_l3430_343051


namespace geometric_sequence_statements_l3430_343021

def geometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def increasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

theorem geometric_sequence_statements (a : ℕ → ℝ) (q : ℝ) 
  (h : geometricSequence a q) : 
  (¬(q > 1 → increasingSequence a) ∧
   ¬(increasingSequence a → q > 1) ∧
   ¬(q ≤ 1 → ¬increasingSequence a) ∧
   ¬(¬increasingSequence a → q ≤ 1)) := by
  sorry

end geometric_sequence_statements_l3430_343021


namespace seashells_count_l3430_343014

theorem seashells_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end seashells_count_l3430_343014


namespace f_is_quadratic_l3430_343070

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l3430_343070


namespace pond_length_l3430_343058

/-- Given a rectangular pond with width 10 m, depth 5 m, and volume of extracted soil 1000 cubic meters, the length of the pond is 20 m. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) (length : ℝ) : 
  width = 10 → depth = 5 → volume = 1000 → volume = length * width * depth → length = 20 := by
  sorry

end pond_length_l3430_343058


namespace max_candies_ben_l3430_343064

/-- The maximum number of candies Ben can eat -/
theorem max_candies_ben (total : ℕ) (h_total : total = 30) : ∃ (b : ℕ), b ≤ 6 ∧ 
  ∀ (k : ℕ+) (b' : ℕ), b' + 2 * b' + k * b' = total → b' ≤ b :=
sorry

end max_candies_ben_l3430_343064


namespace ceiling_floor_sum_l3430_343066

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l3430_343066


namespace eight_digit_integers_count_l3430_343048

/-- The number of choices for the first digit -/
def first_digit_choices : ℕ := 9

/-- The number of choices for each of the remaining seven digits -/
def remaining_digit_choices : ℕ := 5

/-- The number of remaining digits -/
def remaining_digits : ℕ := 7

/-- The total number of different 8-digit positive integers under the given conditions -/
def total_combinations : ℕ := first_digit_choices * remaining_digit_choices ^ remaining_digits

theorem eight_digit_integers_count : total_combinations = 703125 := by
  sorry

end eight_digit_integers_count_l3430_343048


namespace missing_digits_sum_l3430_343057

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- The addition problem structure -/
structure AdditionProblem where
  d1 : Digit  -- First missing digit
  d2 : Digit  -- Second missing digit

/-- The addition problem is valid -/
def isValidAddition (p : AdditionProblem) : Prop :=
  708 + 10 * p.d1.val + 2182 = 86301 + 100 * p.d2.val

/-- The theorem to be proved -/
theorem missing_digits_sum (p : AdditionProblem) 
  (h : isValidAddition p) : p.d1.val + p.d2.val = 7 := by
  sorry

#check missing_digits_sum

end missing_digits_sum_l3430_343057


namespace exponent_rule_l3430_343081

theorem exponent_rule (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_rule_l3430_343081


namespace two_digit_number_property_l3430_343074

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_valid : 1 ≤ tens ∧ tens ≤ 9
  units_valid : units ≤ 9

/-- The value of a two-digit number -/
def value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The reverse of a two-digit number -/
def reverse (n : TwoDigitNumber) : Nat :=
  10 * n.units + n.tens

/-- The sum of digits of a two-digit number -/
def digitSum (n : TwoDigitNumber) : Nat :=
  n.tens + n.units

theorem two_digit_number_property (n : TwoDigitNumber) :
  value n - reverse n = 7 * digitSum n →
  value n + reverse n = 99 := by
  sorry

end two_digit_number_property_l3430_343074


namespace negation_of_existence_is_forall_l3430_343059

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end negation_of_existence_is_forall_l3430_343059


namespace square_area_equal_perimeter_triangle_l3430_343013

theorem square_area_equal_perimeter_triangle (a b c : Real) (h1 : a = 7.5) (h2 : b = 5.3) (h3 : c = 11.2) :
  let triangle_perimeter := a + b + c
  let square_side := triangle_perimeter / 4
  square_side ^ 2 = 36 := by sorry

end square_area_equal_perimeter_triangle_l3430_343013


namespace point_coordinates_wrt_origin_l3430_343078

/-- In a plane rectangular coordinate system, the coordinates of a point
    with respect to the origin are equal to its given coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let A : ℝ × ℝ := (x, y)
  A = (x, y) := by sorry

end point_coordinates_wrt_origin_l3430_343078


namespace product_inequality_l3430_343061

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end product_inequality_l3430_343061


namespace f_inequality_l3430_343017

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_inequality : f (π/3) > f 1 ∧ f 1 > f (-π/4) := by
  sorry

end f_inequality_l3430_343017


namespace pistachio_problem_l3430_343093

theorem pistachio_problem (total : ℕ) (shell_percent : ℚ) (open_percent : ℚ) 
  (h1 : total = 80)
  (h2 : shell_percent = 95 / 100)
  (h3 : open_percent = 75 / 100) :
  ⌊(shell_percent * total : ℚ) * open_percent⌋ = 57 := by
sorry

#eval ⌊(95 / 100 : ℚ) * 80 * (75 / 100 : ℚ)⌋

end pistachio_problem_l3430_343093


namespace triangle_example_1_triangle_example_2_l3430_343085

-- Define the new operation ▲
def triangle (m n : ℤ) : ℤ := m - n + m * n

-- Theorem statements
theorem triangle_example_1 : triangle 3 (-4) = -5 := by sorry

theorem triangle_example_2 : triangle (-6) (triangle 2 (-3)) = 1 := by sorry

end triangle_example_1_triangle_example_2_l3430_343085


namespace inequality_solution_l3430_343046

theorem inequality_solution (x y : ℝ) :
  x + y^2 + Real.sqrt (x - y^2 - 1) ≤ 1 ∧
  x - y^2 - 1 ≥ 0 →
  x = 1 ∧ y = 0 := by
sorry

end inequality_solution_l3430_343046


namespace max_performances_l3430_343019

theorem max_performances (n : ℕ) : 
  (∃ (performances : Fin n → Finset (Fin 12)),
    (∀ i : Fin n, (performances i).card = 6) ∧ 
    (∀ i j : Fin n, i ≠ j → (performances i ∩ performances j).card ≤ 2)) →
  n ≤ 4 :=
by sorry

end max_performances_l3430_343019


namespace triangle_side_length_l3430_343011

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively.
    Prove that if c = 10, A = 45°, and C = 30°, then b = 5(√6 + √2). -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  c = 10 → A = π/4 → C = π/6 → b = 5 * (Real.sqrt 6 + Real.sqrt 2) :=
by sorry

end triangle_side_length_l3430_343011


namespace equation_solution_l3430_343042

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 :=
by
  use -13/4
  sorry

end equation_solution_l3430_343042


namespace proposition_truth_l3430_343097

theorem proposition_truth (x y : ℝ) : x + y ≥ 5 → x ≥ 3 ∨ y ≥ 2 := by
  sorry

end proposition_truth_l3430_343097


namespace mincheol_midterm_average_l3430_343088

/-- Calculates the average of three exam scores -/
def midterm_average (math_score korean_score english_score : ℕ) : ℚ :=
  (math_score + korean_score + english_score : ℚ) / 3

/-- Theorem: Mincheol's midterm average is 80 points -/
theorem mincheol_midterm_average : 
  midterm_average 70 80 90 = 80 := by
  sorry

end mincheol_midterm_average_l3430_343088


namespace netGainDifference_l3430_343068

/-- Represents a job candidate with their associated costs and revenue --/
structure Candidate where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for a candidate --/
def netGain (c : Candidate) : ℕ :=
  c.revenue - c.salary - (c.trainingMonths * c.trainingCostPerMonth) - (c.salary * c.hiringBonusPercent / 100)

/-- The two candidates as described in the problem --/
def candidate1 : Candidate :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

def candidate2 : Candidate :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two candidates --/
theorem netGainDifference : netGain candidate1 - netGain candidate2 = 850 := by
  sorry

end netGainDifference_l3430_343068


namespace employee_payment_percentage_l3430_343034

theorem employee_payment_percentage (total_payment : ℝ) (b_payment : ℝ) :
  total_payment = 450 ∧ b_payment = 180 →
  (total_payment - b_payment) / b_payment * 100 = 150 := by
sorry

end employee_payment_percentage_l3430_343034


namespace zoo_population_after_changes_l3430_343038

/-- Represents the population of animals in a zoo --/
structure ZooPopulation where
  foxes : ℕ
  rabbits : ℕ

/-- Calculates the ratio of foxes to rabbits --/
def ratio (pop : ZooPopulation) : ℚ :=
  pop.foxes / pop.rabbits

theorem zoo_population_after_changes 
  (initial : ZooPopulation)
  (h1 : ratio initial = 2 / 3)
  (h2 : ratio { foxes := initial.foxes - 10, rabbits := initial.rabbits / 2 } = 13 / 10) :
  initial.foxes - 10 + initial.rabbits / 2 = 690 := by
  sorry


end zoo_population_after_changes_l3430_343038


namespace base10_to_base8_2357_l3430_343023

-- Define a function to convert a base 10 number to base 8
def toBase8 (n : ℕ) : List ℕ :=
  sorry

-- Theorem stating that 2357 in base 10 is equal to 4445 in base 8
theorem base10_to_base8_2357 :
  toBase8 2357 = [4, 4, 4, 5] :=
sorry

end base10_to_base8_2357_l3430_343023


namespace square_sum_zero_implies_both_zero_l3430_343055

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l3430_343055


namespace ratio_equality_l3430_343045

theorem ratio_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) : 
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := by
  sorry

end ratio_equality_l3430_343045


namespace triangle_DEF_circles_l3430_343065

/-- Triangle DEF with side lengths -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ

/-- The inscribed circle of a triangle -/
def inscribedCircleDiameter (t : Triangle) : ℝ := sorry

/-- The circumscribed circle of a triangle -/
def circumscribedCircleRadius (t : Triangle) : ℝ := sorry

/-- Main theorem about triangle DEF -/
theorem triangle_DEF_circles :
  let t : Triangle := { DE := 13, DF := 8, EF := 9 }
  inscribedCircleDiameter t = 2 * Real.sqrt 14 ∧
  circumscribedCircleRadius t = (39 * Real.sqrt 14) / 35 := by sorry

end triangle_DEF_circles_l3430_343065


namespace seashells_given_correct_l3430_343072

/-- Calculates the number of seashells given away -/
def seashells_given (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Proves that the number of seashells given away is correct -/
theorem seashells_given_correct (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_given initial_seashells current_seashells = initial_seashells - current_seashells :=
by
  sorry

#eval seashells_given 49 36

end seashells_given_correct_l3430_343072


namespace hours_worked_on_second_job_l3430_343063

/-- Calculates the number of hours worked on the second job given the total earnings and other job details -/
theorem hours_worked_on_second_job
  (hourly_rate_1 hourly_rate_2 hourly_rate_3 : ℚ)
  (hours_1 hours_3 : ℚ)
  (days : ℚ)
  (total_earnings : ℚ)
  (h1 : hourly_rate_1 = 7)
  (h2 : hourly_rate_2 = 10)
  (h3 : hourly_rate_3 = 12)
  (h4 : hours_1 = 3)
  (h5 : hours_3 = 4)
  (h6 : days = 5)
  (h7 : total_earnings = 445)
  : ∃ hours_2 : ℚ, hours_2 = 2 ∧ 
    days * (hourly_rate_1 * hours_1 + hourly_rate_2 * hours_2 + hourly_rate_3 * hours_3) = total_earnings :=
by sorry

end hours_worked_on_second_job_l3430_343063


namespace f_monotone_increasing_l3430_343083

theorem f_monotone_increasing (k : ℝ) (h_k : k ≥ 0) :
  ∀ x ≥ Real.sqrt (2 * k + 1), HasDerivAt (λ x => x + (2 * k + 1) / x) ((x^2 - (2 * k + 1)) / x^2) x ∧
  (x^2 - (2 * k + 1)) / x^2 ≥ 0 := by
  sorry

end f_monotone_increasing_l3430_343083


namespace simplify_polynomial_l3430_343026

theorem simplify_polynomial (b : ℝ) : (1 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 360 * b^10 := by
  sorry

end simplify_polynomial_l3430_343026


namespace intersection_points_of_cubic_l3430_343007

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem intersection_points_of_cubic (c : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ + c = 0 ∧ f x₂ + c = 0 ∧
    ∀ x, f x + c = 0 → x = x₁ ∨ x = x₂) ↔ c = -2 ∨ c = 2 :=
sorry

end intersection_points_of_cubic_l3430_343007


namespace eliza_ironing_time_l3430_343050

theorem eliza_ironing_time :
  ∀ (blouse_time : ℝ),
    (blouse_time > 0) →
    (120 / blouse_time + 180 / 20 = 17) →
    blouse_time = 15 := by
  sorry

end eliza_ironing_time_l3430_343050


namespace math_club_team_selection_l3430_343076

def total_boys : ℕ := 9
def total_girls : ℕ := 10
def experienced_boys : ℕ := 4
def team_size : ℕ := 7
def required_boys : ℕ := 4
def required_girls : ℕ := 3
def required_experienced_boys : ℕ := 2

theorem math_club_team_selection :
  (Nat.choose experienced_boys required_experienced_boys) *
  (Nat.choose (total_boys - experienced_boys) (required_boys - required_experienced_boys)) *
  (Nat.choose total_girls required_girls) = 7200 :=
sorry

end math_club_team_selection_l3430_343076


namespace fraction_subtraction_proof_l3430_343028

theorem fraction_subtraction_proof : 
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by sorry

end fraction_subtraction_proof_l3430_343028


namespace cubic_one_real_root_l3430_343035

theorem cubic_one_real_root (a b : ℝ) : 
  (∃! x : ℝ, x^3 - a*x + b = 0) ↔ 
  ((a = 0 ∧ b = 2) ∨ (a = -3 ∧ b = 2) ∨ (a = 3 ∧ b = -3)) :=
sorry

end cubic_one_real_root_l3430_343035


namespace log_equation_implies_x_greater_than_two_l3430_343095

theorem log_equation_implies_x_greater_than_two (x : ℝ) :
  Real.log (x^2 + 5*x + 6) = Real.log ((x+1)*(x+4)) + Real.log (x-2) →
  x > 2 :=
by sorry

end log_equation_implies_x_greater_than_two_l3430_343095
