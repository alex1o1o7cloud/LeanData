import Mathlib

namespace triangle_trigonometric_expression_l1020_102054

theorem triangle_trigonometric_expression (X Y Z : ℝ) : 
  (13 : ℝ) ^ 2 = X ^ 2 + Y ^ 2 - 2 * X * Y * Real.cos Z →
  (14 : ℝ) ^ 2 = X ^ 2 + Z ^ 2 - 2 * X * Z * Real.cos Y →
  (15 : ℝ) ^ 2 = Y ^ 2 + Z ^ 2 - 2 * Y * Z * Real.cos X →
  (Real.cos ((X - Y) / 2) / Real.sin (Z / 2)) - (Real.sin ((X - Y) / 2) / Real.cos (Z / 2)) = 28 / 13 := by
  sorry


end triangle_trigonometric_expression_l1020_102054


namespace equal_charge_at_20_minutes_l1020_102095

/-- United Telephone's base rate -/
def united_base : ℝ := 11

/-- United Telephone's per-minute rate -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℝ := 20

theorem equal_charge_at_20_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
sorry

end equal_charge_at_20_minutes_l1020_102095


namespace systematic_sampling_first_number_l1020_102084

/-- Systematic sampling problem -/
theorem systematic_sampling_first_number
  (population : ℕ)
  (sample_size : ℕ)
  (eighteenth_sample : ℕ)
  (h1 : population = 1000)
  (h2 : sample_size = 40)
  (h3 : eighteenth_sample = 443)
  (h4 : sample_size > 0)
  (h5 : population ≥ sample_size) :
  ∃ (first_sample : ℕ),
    first_sample + 17 * (population / sample_size) = eighteenth_sample ∧
    first_sample = 18 :=
by sorry

end systematic_sampling_first_number_l1020_102084


namespace second_carpenter_proof_l1020_102049

/-- The time taken by the second carpenter to complete the job alone -/
def second_carpenter_time : ℚ :=
  10 / 3

theorem second_carpenter_proof (first_carpenter_time : ℚ) 
  (first_carpenter_initial_work : ℚ) (combined_work_time : ℚ) :
  first_carpenter_time = 5 →
  first_carpenter_initial_work = 1 →
  combined_work_time = 2 →
  second_carpenter_time = 10 / 3 :=
by
  sorry

#eval second_carpenter_time

end second_carpenter_proof_l1020_102049


namespace different_color_probability_l1020_102011

/-- Given 6 cards with 3 red and 3 yellow, the probability of drawing 2 cards of different colors is 3/5 -/
theorem different_color_probability (total_cards : Nat) (red_cards : Nat) (yellow_cards : Nat) :
  total_cards = 6 →
  red_cards = 3 →
  yellow_cards = 3 →
  (Nat.choose red_cards 1 * Nat.choose yellow_cards 1 : Rat) / Nat.choose total_cards 2 = 3/5 := by
  sorry

end different_color_probability_l1020_102011


namespace problem_solution_l1020_102013

theorem problem_solution : (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + 0.009 + (0.06 : ℝ)^2 = 0.006375 := by
  sorry

end problem_solution_l1020_102013


namespace parallelogram_area_l1020_102006

/-- A parallelogram with base 10 and altitude twice the base has area 200. -/
theorem parallelogram_area (base : ℝ) (altitude : ℝ) : 
  base = 10 → altitude = 2 * base → base * altitude = 200 := by
  sorry

end parallelogram_area_l1020_102006


namespace continuous_stripe_probability_l1020_102098

/-- Represents the type of stripe on a cube face -/
inductive StripeType
| Solid
| Dashed

/-- Represents the orientation of a stripe on a cube face -/
inductive StripeOrientation
| Horizontal
| Vertical

/-- Represents a single face configuration -/
structure FaceConfig where
  stripeType : StripeType
  orientation : StripeOrientation

/-- Represents a complete cube configuration -/
structure CubeConfig where
  faces : Fin 6 → FaceConfig

/-- Determines if a given cube configuration has a continuous stripe -/
def hasContinuousStripe (config : CubeConfig) : Bool := sorry

/-- The total number of possible cube configurations -/
def totalConfigurations : Nat := 4^6

/-- The number of configurations with a continuous stripe -/
def continuousStripeConfigurations : Nat := 3 * 16

/-- The probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  (continuousStripeConfigurations : ℚ) / totalConfigurations = 3 / 256 := by sorry

end continuous_stripe_probability_l1020_102098


namespace ayen_jog_time_l1020_102060

/-- The number of minutes Ayen usually jogs every day during weekdays -/
def usual_jog_time : ℕ := sorry

/-- The total time Ayen jogged this week in minutes -/
def total_jog_time : ℕ := 180

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The extra minutes Ayen jogged on Tuesday -/
def tuesday_extra : ℕ := 5

/-- The extra minutes Ayen jogged on Friday -/
def friday_extra : ℕ := 25

theorem ayen_jog_time : 
  usual_jog_time * weekdays + tuesday_extra + friday_extra = total_jog_time ∧
  usual_jog_time = 30 := by sorry

end ayen_jog_time_l1020_102060


namespace midpoint_figure_area_l1020_102086

/-- The area of a figure in a 6x6 grid formed by connecting midpoints to the center -/
theorem midpoint_figure_area : 
  ∀ (grid_size : ℕ) (center_square_area corner_triangle_area : ℝ),
  grid_size = 6 →
  center_square_area = 4.5 →
  corner_triangle_area = 4.5 →
  center_square_area + 4 * corner_triangle_area = 22.5 :=
by sorry

end midpoint_figure_area_l1020_102086


namespace fourteen_percent_of_seven_hundred_is_ninety_eight_l1020_102093

theorem fourteen_percent_of_seven_hundred_is_ninety_eight :
  ∀ x : ℝ, (14 / 100) * x = 98 → x = 700 := by
  sorry

end fourteen_percent_of_seven_hundred_is_ninety_eight_l1020_102093


namespace num_perfect_square_factors_is_440_l1020_102020

/-- The number of positive perfect square factors of (2^14)(3^9)(5^20) -/
def num_perfect_square_factors : ℕ :=
  (Finset.range 8).card * (Finset.range 5).card * (Finset.range 11).card

/-- Theorem stating that the number of positive perfect square factors of (2^14)(3^9)(5^20) is 440 -/
theorem num_perfect_square_factors_is_440 : num_perfect_square_factors = 440 := by
  sorry

end num_perfect_square_factors_is_440_l1020_102020


namespace flagpole_length_correct_flagpole_length_is_60_l1020_102045

/-- The length of the flagpole in feet. -/
def flagpole_length : ℝ := 60

/-- The total distance the flag moves up and down the pole in feet. -/
def total_flag_movement : ℝ := 180

/-- Theorem stating that the flagpole length is correct given the total flag movement. -/
theorem flagpole_length_correct :
  flagpole_length * 3 = total_flag_movement :=
by sorry

/-- Theorem proving that the flagpole length is 60 feet. -/
theorem flagpole_length_is_60 :
  flagpole_length = 60 :=
by sorry

end flagpole_length_correct_flagpole_length_is_60_l1020_102045


namespace domain_implies_k_range_inequality_solution_set_l1020_102099

-- Problem I
theorem domain_implies_k_range (f : ℝ → ℝ) (h : ∀ x, ∃ y, f x = y) :
  (∀ x, f x = Real.sqrt (x^2 - x * k - k)) → k ∈ Set.Icc (-4) 0 := by sorry

-- Problem II
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | (x - a) * (x + a - 1) > 0} =
    if a = 1/2 then
      {x : ℝ | x ≠ 1/2}
    else if a < 1/2 then
      {x : ℝ | x > 1 - a ∨ x < a}
    else
      {x : ℝ | x > a ∨ x < 1 - a} := by sorry

end domain_implies_k_range_inequality_solution_set_l1020_102099


namespace average_of_first_two_l1020_102044

theorem average_of_first_two (total_avg : ℝ) (second_set_avg : ℝ) (third_set_avg : ℝ)
  (h1 : total_avg = 2.5)
  (h2 : second_set_avg = 1.4)
  (h3 : third_set_avg = 5) :
  let total_sum := 6 * total_avg
  let second_set_sum := 2 * second_set_avg
  let third_set_sum := 2 * third_set_avg
  let first_set_sum := total_sum - second_set_sum - third_set_sum
  first_set_sum / 2 = 1.1 := by
sorry

end average_of_first_two_l1020_102044


namespace total_peanuts_l1020_102041

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def marcos_peanuts : ℕ := kenya_peanuts + 37

theorem total_peanuts : jose_peanuts + kenya_peanuts + marcos_peanuts = 388 := by
  sorry

end total_peanuts_l1020_102041


namespace total_shells_count_l1020_102010

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_count : morning_shells + afternoon_shells = 616 := by
  sorry

end total_shells_count_l1020_102010


namespace nested_expression_equals_4094_l1020_102030

def nested_expression : ℕ := 2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2))))))))))

theorem nested_expression_equals_4094 : nested_expression = 4094 := by
  sorry

end nested_expression_equals_4094_l1020_102030


namespace robies_boxes_given_away_l1020_102057

/-- Given information about Robie's hockey cards and boxes, prove the number of boxes he gave away. -/
theorem robies_boxes_given_away
  (total_cards : ℕ)
  (cards_per_box : ℕ)
  (cards_not_in_box : ℕ)
  (boxes_with_robie : ℕ)
  (h1 : total_cards = 75)
  (h2 : cards_per_box = 10)
  (h3 : cards_not_in_box = 5)
  (h4 : boxes_with_robie = 5) :
  total_cards / cards_per_box - boxes_with_robie = 2 :=
by sorry

end robies_boxes_given_away_l1020_102057


namespace candies_distribution_l1020_102092

def candies_a : ℕ := 17
def candies_b : ℕ := 19
def num_people : ℕ := 9

theorem candies_distribution :
  (candies_a + candies_b) / num_people = 4 := by
  sorry

end candies_distribution_l1020_102092


namespace no_real_roots_quadratic_l1020_102082

theorem no_real_roots_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end no_real_roots_quadratic_l1020_102082


namespace inequality_implication_l1020_102066

theorem inequality_implication (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2) → 
  a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a) := by
  sorry

end inequality_implication_l1020_102066


namespace simplify_fraction_l1020_102059

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  8 * x * y^2 / (6 * x^2 * y) = 2 := by sorry

end simplify_fraction_l1020_102059


namespace sum_of_coefficients_l1020_102083

def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

theorem sum_of_coefficients : f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  sorry

end sum_of_coefficients_l1020_102083


namespace least_integer_square_72_more_than_double_l1020_102073

theorem least_integer_square_72_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 72 ∧ ∀ y : ℤ, y^2 = 2*y + 72 → x ≤ y :=
sorry

end least_integer_square_72_more_than_double_l1020_102073


namespace star_value_proof_l1020_102036

theorem star_value_proof (star : ℝ) : 
  45 - (28 - (37 - (15 - star^2))) = 59 → star = 2 * Real.sqrt 5 := by
sorry

end star_value_proof_l1020_102036


namespace arithmetic_sequence_log_property_l1020_102017

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the arithmetic sequence property
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

-- Define the theorem
theorem arithmetic_sequence_log_property
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : is_arithmetic_sequence (log (a^2 * b^6)) (log (a^4 * b^11)) (log (a^7 * b^14)))
  (h4 : ∃ m : ℕ, (log (b^m)) = (log (a^2 * b^6)) + 7 * ((log (a^4 * b^11)) - (log (a^2 * b^6))))
  : ∃ m : ℕ, m = 73 :=
sorry

end arithmetic_sequence_log_property_l1020_102017


namespace function_range_condition_l1020_102091

open Real

/-- Given a function f(x) = ax - ln x - 1, prove that there exists x₀ ∈ (0,e] 
    such that f(x₀) < 0 if and only if a ∈ (-∞, 1). -/
theorem function_range_condition (a : ℝ) : 
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ ℯ ∧ a * x₀ - log x₀ - 1 < 0) ↔ a < 1 :=
by sorry

end function_range_condition_l1020_102091


namespace total_time_is_186_l1020_102027

def total_time (mac_download : ℕ) (windows_multiplier : ℕ)
  (ny_audio_glitch_count ny_audio_glitch_duration : ℕ)
  (ny_video_glitch_count ny_video_glitch_duration : ℕ)
  (ny_unglitched_multiplier : ℕ)
  (berlin_audio_glitch_count berlin_audio_glitch_duration : ℕ)
  (berlin_video_glitch_count berlin_video_glitch_duration : ℕ)
  (berlin_unglitched_multiplier : ℕ) : ℕ :=
  let windows_download := mac_download * windows_multiplier
  let total_download := mac_download + windows_download

  let ny_audio_glitch := ny_audio_glitch_count * ny_audio_glitch_duration
  let ny_video_glitch := ny_video_glitch_count * ny_video_glitch_duration
  let ny_total_glitch := ny_audio_glitch + ny_video_glitch
  let ny_unglitched := ny_total_glitch * ny_unglitched_multiplier
  let ny_total := ny_total_glitch + ny_unglitched

  let berlin_audio_glitch := berlin_audio_glitch_count * berlin_audio_glitch_duration
  let berlin_video_glitch := berlin_video_glitch_count * berlin_video_glitch_duration
  let berlin_total_glitch := berlin_audio_glitch + berlin_video_glitch
  let berlin_unglitched := berlin_total_glitch * berlin_unglitched_multiplier
  let berlin_total := berlin_total_glitch + berlin_unglitched

  total_download + ny_total + berlin_total

theorem total_time_is_186 :
  total_time 10 3 2 6 1 8 3 3 4 2 5 2 = 186 := by sorry

end total_time_is_186_l1020_102027


namespace units_digit_problem_l1020_102047

def geometric_sum (a r : ℕ) (n : ℕ) : ℕ := 
  a * (r^(n+1) - 1) / (r - 1)

theorem units_digit_problem : 
  (2 * geometric_sum 1 3 9) % 10 = 6 := by
  sorry

end units_digit_problem_l1020_102047


namespace no_solution_to_inequality_l1020_102046

theorem no_solution_to_inequality :
  ¬∃ x : ℝ, (4 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 5) := by
sorry

end no_solution_to_inequality_l1020_102046


namespace constant_term_expansion_l1020_102012

theorem constant_term_expansion : 
  let p₁ : Polynomial ℤ := X^4 + 2*X^2 + 7
  let p₂ : Polynomial ℤ := 2*X^5 + 3*X^3 + 25
  (p₁ * p₂).coeff 0 = 175 := by sorry

end constant_term_expansion_l1020_102012


namespace compound_composition_l1020_102071

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of aluminum atoms in the compound -/
def num_Al : ℕ := 1

/-- The number of fluorine atoms in the compound -/
def num_F : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 84

theorem compound_composition :
  (num_Al : ℝ) * atomic_weight_Al + (num_F : ℝ) * atomic_weight_F = molecular_weight := by
  sorry

end compound_composition_l1020_102071


namespace starting_number_proof_l1020_102004

theorem starting_number_proof : ∃ (n : ℕ), 
  n = 220 ∧ 
  n < 580 ∧ 
  (∃ (m : ℕ), m = 6 ∧ 
    (∀ k : ℕ, n ≤ k ∧ k ≤ 580 → (k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0) ↔ k ∈ Finset.range (m + 1) ∧ k ≠ n)) ∧
  (∀ n' : ℕ, n < n' → n' < 580 → 
    ¬(∃ (m : ℕ), m = 6 ∧ 
      (∀ k : ℕ, n' ≤ k ∧ k ≤ 580 → (k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0) ↔ k ∈ Finset.range (m + 1) ∧ k ≠ n'))) :=
by sorry

end starting_number_proof_l1020_102004


namespace altitude_intersection_property_l1020_102094

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Finds the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Theorem: In an acute triangle ABC with altitudes AP and BQ intersecting at H,
    if HP = 7 and HQ = 3, then (BP)(PC) - (AQ)(QC) = 40 -/
theorem altitude_intersection_property (t : Triangle) (P Q H : Point) :
  isAcute t →
  isPerpendicular t.A P t.B t.C →
  isPerpendicular t.B Q t.A t.C →
  H = lineIntersection t.A P t.B Q →
  distance H P = 7 →
  distance H Q = 3 →
  distance t.B P * distance P t.C - distance t.A Q * distance Q t.C = 40 := by
  sorry

end altitude_intersection_property_l1020_102094


namespace fraction_inequality_solution_set_l1020_102023

theorem fraction_inequality_solution_set :
  {x : ℝ | (2*x + 1) / (x - 3) ≤ 0} = {x : ℝ | -1/2 ≤ x ∧ x < 3} := by sorry

end fraction_inequality_solution_set_l1020_102023


namespace cube_sum_problem_l1020_102015

theorem cube_sum_problem (x y : ℝ) 
  (h1 : 1/x + 1/y = 4)
  (h2 : x*y + x^2 + y^2 = 17) :
  x^3 + y^3 = 52 := by
  sorry

end cube_sum_problem_l1020_102015


namespace students_playing_football_l1020_102040

theorem students_playing_football 
  (total : ℕ) 
  (cricket : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (h1 : total = 470) 
  (h2 : cricket = 175) 
  (h3 : neither = 50) 
  (h4 : both = 80) : 
  total - neither - cricket + both = 325 := by
  sorry

end students_playing_football_l1020_102040


namespace area_perimeter_ratio_inequality_l1020_102061

/-- A convex polygon -/
structure ConvexPolygon where
  area : ℝ
  perimeter : ℝ

/-- X is contained within Y -/
def isContainedIn (X Y : ConvexPolygon) : Prop := sorry

theorem area_perimeter_ratio_inequality {X Y : ConvexPolygon} 
  (h : isContainedIn X Y) :
  X.area / X.perimeter < 2 * Y.area / Y.perimeter := by
  sorry

end area_perimeter_ratio_inequality_l1020_102061


namespace earnings_ratio_l1020_102019

theorem earnings_ratio (total_earnings lottie_earnings jerusha_earnings : ℕ)
  (h1 : total_earnings = 85)
  (h2 : jerusha_earnings = 68)
  (h3 : total_earnings = lottie_earnings + jerusha_earnings)
  (h4 : ∃ k : ℕ, jerusha_earnings = k * lottie_earnings) :
  jerusha_earnings = 4 * lottie_earnings :=
by sorry

end earnings_ratio_l1020_102019


namespace pictures_per_album_l1020_102008

/-- Given pictures from a phone and camera, prove the number of pictures in each album when equally distributed. -/
theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 2) 
  (h2 : camera_pics = 4) 
  (h3 : num_albums = 3) 
  (h4 : num_albums > 0) : 
  (phone_pics + camera_pics) / num_albums = 2 := by
sorry

end pictures_per_album_l1020_102008


namespace highway_length_l1020_102078

theorem highway_length (speed1 speed2 time : ℝ) (h1 : speed1 = 13) (h2 : speed2 = 17) (h3 : time = 2) :
  (speed1 + speed2) * time = 60 := by
  sorry

end highway_length_l1020_102078


namespace octahedron_containment_l1020_102024

-- Define the plane equation
def plane_equation (x y z : ℚ) (n : ℤ) : Prop :=
  (x + y + z = n) ∨ (x + y - z = n) ∨ (x - y + z = n) ∨ (x - y - z = n)

-- Define a point not on any plane
def not_on_planes (x y z : ℚ) : Prop :=
  ∀ n : ℤ, ¬ plane_equation x y z n

-- Define a point inside an octahedron
def inside_octahedron (x y z : ℚ) : Prop :=
  ∃ n : ℤ, 
    n < x + y + z ∧ x + y + z < n + 1 ∧
    n < x + y - z ∧ x + y - z < n + 1 ∧
    n < x - y + z ∧ x - y + z < n + 1 ∧
    n < -x + y + z ∧ -x + y + z < n + 1

-- The main theorem
theorem octahedron_containment (x₀ y₀ z₀ : ℚ) 
  (h : not_on_planes x₀ y₀ z₀) :
  ∃ k : ℕ, inside_octahedron (k * x₀) (k * y₀) (k * z₀) := by
  sorry

end octahedron_containment_l1020_102024


namespace melanie_dimes_proof_l1020_102068

def final_dimes (initial : ℕ) (received : ℕ) (given_away : ℕ) : ℕ :=
  initial + received - given_away

theorem melanie_dimes_proof :
  final_dimes 7 8 4 = 11 := by
  sorry

end melanie_dimes_proof_l1020_102068


namespace contiguous_substring_divisible_by_2011_l1020_102032

def isContiguousSubstring (s t : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), t = (s / 10^k) % 10^m

theorem contiguous_substring_divisible_by_2011 :
  ∃ (N : ℕ), ∀ (a : ℕ), a > N →
    ∃ (s : ℕ), isContiguousSubstring a s ∧ s % 2011 = 0 := by
  sorry

end contiguous_substring_divisible_by_2011_l1020_102032


namespace power_two_mod_seven_l1020_102067

theorem power_two_mod_seven : (2^200 - 3) % 7 = 1 := by
  sorry

end power_two_mod_seven_l1020_102067


namespace curler_count_l1020_102072

theorem curler_count (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ) : 
  total = 16 →
  pink = total / 4 →
  blue = 2 * pink →
  green = total - (pink + blue) →
  green = 4 := by
  sorry

end curler_count_l1020_102072


namespace interest_discount_sum_l1020_102088

/-- Given a sum, rate, and time, if the simple interest is 85 and the true discount is 80, then the sum is 1360 -/
theorem interest_discount_sum (P r t : ℝ) : 
  (P * r * t / 100 = 85) → 
  (P * r * t / (100 + r * t) = 80) → 
  P = 1360 := by
  sorry

end interest_discount_sum_l1020_102088


namespace square_adjacent_to_multiple_of_five_l1020_102077

theorem square_adjacent_to_multiple_of_five (n : ℤ) (h : ¬ 5 ∣ n) :
  ∃ k : ℤ, n^2 = 5*k + 1 ∨ n^2 = 5*k - 1 := by
  sorry

end square_adjacent_to_multiple_of_five_l1020_102077


namespace income_distribution_l1020_102048

theorem income_distribution (income : ℝ) (h1 : income = 800000) : 
  let children_share := 0.2 * income * 3
  let wife_share := 0.3 * income
  let family_distribution := children_share + wife_share
  let remaining_after_family := income - family_distribution
  let orphan_donation := 0.05 * remaining_after_family
  let final_amount := remaining_after_family - orphan_donation
  final_amount = 76000 :=
by sorry

end income_distribution_l1020_102048


namespace smallest_valid_integers_difference_l1020_102043

def is_valid_integer (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 1

theorem smallest_valid_integers_difference : 
  ∃ n₁ n₂ : ℕ, 
    is_valid_integer n₁ ∧
    is_valid_integer n₂ ∧
    n₁ < n₂ ∧
    (∀ m : ℕ, is_valid_integer m → m ≥ n₁) ∧
    (∀ m : ℕ, is_valid_integer m ∧ m ≠ n₁ → m ≥ n₂) ∧
    n₂ - n₁ = 27720 :=
by sorry

end smallest_valid_integers_difference_l1020_102043


namespace christine_and_siri_money_l1020_102074

theorem christine_and_siri_money (christine_money siri_money : ℚ) : 
  christine_money = 20.5 → 
  christine_money = siri_money + 20 → 
  christine_money + siri_money = 21 := by
sorry

end christine_and_siri_money_l1020_102074


namespace quadratic_function_properties_l1020_102070

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  f : ℝ → ℝ
  passesA : f 1 = 0
  passesB : f (-3) = 0
  passesC : f 0 = -3

/-- The theorem stating properties of the quadratic function -/
theorem quadratic_function_properties (qf : QuadraticFunction) :
  (∃ a b c : ℝ, ∀ x, qf.f x = a * x^2 + b * x + c) →
  (∀ x, qf.f x = x^2 + 2*x - 3) ∧
  (qf.f (-1) = -4 ∧ ∀ x, qf.f x ≥ qf.f (-1)) := by
  sorry

end quadratic_function_properties_l1020_102070


namespace square_difference_plus_six_b_l1020_102035

theorem square_difference_plus_six_b (a b : ℝ) (h : a + b = 3) : 
  a^2 - b^2 + 6*b = 9 := by
  sorry

end square_difference_plus_six_b_l1020_102035


namespace B_power_15_minus_3_power_14_l1020_102022

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 8192; 0, -8192] := by sorry

end B_power_15_minus_3_power_14_l1020_102022


namespace symmetry_implies_phi_value_l1020_102018

/-- Given a function f and its translation g, proves that if g is symmetric about π/2, then φ = π/2 -/
theorem symmetry_implies_phi_value 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (φ : ℝ) 
  (h1 : 0 < φ ∧ φ < π)
  (h2 : ∀ x, f x = Real.cos (2 * x + φ))
  (h3 : ∀ x, g x = f (x - π/4))
  (h4 : ∀ x, g x = g (π - x)) : 
  φ = π/2 := by
  sorry

end symmetry_implies_phi_value_l1020_102018


namespace four_valid_configurations_l1020_102080

/-- Represents a square piece -/
inductive Square
| A | B | C | D | E | F | G | H

/-- Represents the F-shaped figure -/
structure FShape :=
  (squares : Fin 6 → Unit)

/-- Represents a topless rectangular box -/
structure ToplessBox :=
  (base : Unit)
  (sides : Fin 4 → Unit)

/-- Function to check if a square can be combined with the F-shape to form a topless box -/
def canFormBox (s : Square) (f : FShape) : Prop :=
  ∃ (box : ToplessBox), true  -- Placeholder, actual implementation would be more complex

/-- The main theorem stating that exactly 4 squares can form a topless box with the F-shape -/
theorem four_valid_configurations (squares : Fin 8 → Square) (f : FShape) :
  (∃! (validSquares : Finset Square), 
    validSquares.card = 4 ∧ 
    ∀ s, s ∈ validSquares ↔ canFormBox s f) :=
sorry

end four_valid_configurations_l1020_102080


namespace one_weighing_sufficient_l1020_102087

/-- Represents the types of balls -/
inductive BallType
| Aluminum
| Duralumin

/-- Represents a collection of balls -/
structure BallCollection where
  aluminum : ℕ
  duralumin : ℕ

/-- The mass of a ball collection -/
def mass (bc : BallCollection) : ℚ :=
  10 * bc.aluminum + 99/10 * bc.duralumin

theorem one_weighing_sufficient :
  ∃ (group1 group2 : BallCollection),
    group1.aluminum + group1.duralumin = 1000 ∧
    group2.aluminum + group2.duralumin = 1000 ∧
    group1.aluminum + group2.aluminum = 1000 ∧
    group1.duralumin + group2.duralumin = 1000 ∧
    mass group1 ≠ mass group2 :=
sorry

end one_weighing_sufficient_l1020_102087


namespace fraction_denominator_l1020_102039

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / 5 + (3 * y) / x = 0.7 * y) : x = 10 := by
  sorry

end fraction_denominator_l1020_102039


namespace pentagon_reconstruction_l1020_102079

-- Define the pentagon and its extended points
variable (A B C D E A' B' C' D' E' : ℝ × ℝ)

-- Define the conditions of the pentagon
axiom extend_A : A' - B = B - A
axiom extend_B : B' - C = C - B
axiom extend_C : C' - D = D - C
axiom extend_D : D' - E = E - D
axiom extend_E : E' - A = A - E

-- Define the theorem
theorem pentagon_reconstruction :
  E = (1/31 : ℝ) • A' + (1/31 : ℝ) • B' + (2/31 : ℝ) • C' + (4/31 : ℝ) • D' + (8/31 : ℝ) • E' :=
sorry

end pentagon_reconstruction_l1020_102079


namespace garden_border_material_l1020_102058

/-- The amount of material needed for a decorative border around a circular garden -/
theorem garden_border_material (garden_area : Real) (pi_estimate : Real) (extra_material : Real) : 
  garden_area = 50.24 → pi_estimate = 3.14 → extra_material = 4 →
  2 * pi_estimate * (garden_area / pi_estimate).sqrt + extra_material = 29.12 := by
sorry

end garden_border_material_l1020_102058


namespace students_like_both_correct_l1020_102038

/-- The number of students who like both apple pie and chocolate cake -/
def students_like_both (total : ℕ) (apple : ℕ) (chocolate : ℕ) (pumpkin : ℕ) (none : ℕ) : ℕ := 
  apple + chocolate - (total - none)

theorem students_like_both_correct (total : ℕ) (apple : ℕ) (chocolate : ℕ) (pumpkin : ℕ) (none : ℕ) 
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 20)
  (h4 : pumpkin = 17)
  (h5 : none = 15) :
  students_like_both total apple chocolate pumpkin none = 7 := by
  sorry

#eval students_like_both 50 22 20 17 15

end students_like_both_correct_l1020_102038


namespace solution_in_interval_l1020_102002

open Real

/-- A monotonically increasing function on (0, +∞) satisfying f[f(x) - ln x] = 1 -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (∀ x, 0 < x → f (f x - log x) = 1)

/-- The solution to f(x) - f'(x) = 1 lies in (1, 2) -/
theorem solution_in_interval (f : ℝ → ℝ) (hf : MonotonicFunction f) :
  ∃ x, 1 < x ∧ x < 2 ∧ f x - (deriv f) x = 1 :=
sorry

end solution_in_interval_l1020_102002


namespace no_three_digit_perfect_square_difference_l1020_102005

theorem no_three_digit_perfect_square_difference :
  ¬ ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    ∃ (k : ℕ), (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = k^2 :=
by sorry

end no_three_digit_perfect_square_difference_l1020_102005


namespace negation_of_all_not_divisible_by_two_are_odd_l1020_102016

theorem negation_of_all_not_divisible_by_two_are_odd :
  (¬ ∀ n : ℤ, ¬(2 ∣ n) → Odd n) ↔ (∃ n : ℤ, ¬(2 ∣ n) ∧ ¬(Odd n)) :=
by sorry

end negation_of_all_not_divisible_by_two_are_odd_l1020_102016


namespace sufficient_not_necessary_l1020_102029

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧
  (∃ x, 1 / x < 1 ∧ x ≤ 1) :=
by sorry

end sufficient_not_necessary_l1020_102029


namespace painter_completion_time_l1020_102053

-- Define the start time
def start_time : Nat := 9

-- Define the quarter completion time
def quarter_time : Nat := 12

-- Define the time taken for quarter completion
def quarter_duration : Nat := quarter_time - start_time

-- Define the total duration
def total_duration : Nat := 4 * quarter_duration

-- Define the completion time
def completion_time : Nat := start_time + total_duration

-- Theorem statement
theorem painter_completion_time :
  start_time = 9 →
  quarter_time = 12 →
  completion_time = 21 :=
by
  sorry

end painter_completion_time_l1020_102053


namespace a_5_value_l1020_102056

def geometric_sequence_with_ratio_difference (a : ℕ → ℝ) (k : ℝ) :=
  ∀ n, a (n + 2) / a (n + 1) - a (n + 1) / a n = k

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence_with_ratio_difference a 2 →
  a 1 = 1 →
  a 2 = 2 →
  a 5 = 384 := by
sorry

end a_5_value_l1020_102056


namespace find_y_l1020_102075

theorem find_y (x z : ℤ) (y : ℚ) 
  (h1 : x = -2) 
  (h2 : z = 4) 
  (h3 : x^2 * y * z - x * y * z^2 = 48) : 
  y = 1 := by
  sorry

end find_y_l1020_102075


namespace parabola_focus_l1020_102063

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -1/8 * x^2

-- Define symmetry about y-axis
def symmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the focus of a parabola
def focus (f h : ℝ) : Prop :=
  ∀ x y, parabola x y → (x - 0)^2 + (y - h)^2 = (y + 2)^2

-- Theorem statement
theorem parabola_focus :
  symmetricAboutYAxis (λ x => -1/8 * x^2) →
  focus 0 (-2) :=
sorry

end parabola_focus_l1020_102063


namespace triangle_properties_l1020_102081

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x - Real.sqrt 3 * Real.sin (2 * x)

theorem triangle_properties :
  ∀ (A B C : ℝ) (a b c : ℝ),
  f A = -1 →
  a = Real.sqrt 7 →
  ∃ (m n : ℝ × ℝ), m = (3, Real.sin B) ∧ n = (2, Real.sin C) ∧ ∃ (k : ℝ), m = k • n →
  A = π / 3 ∧
  b = 3 ∧
  c = 2 ∧
  (1 / 2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_properties_l1020_102081


namespace least_five_digit_square_cube_l1020_102025

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n ≤ 99999) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, (10000 ≤ m ∧ m < n) → ¬(∃ x : ℕ, m = x^2) ∨ ¬(∃ y : ℕ, m = y^3)) ∧
  n = 15625 := by
sorry

end least_five_digit_square_cube_l1020_102025


namespace water_displacement_cube_in_cylinder_l1020_102051

theorem water_displacement_cube_in_cylinder (cube_side : ℝ) (cylinder_radius : ℝ) 
  (h_cube : cube_side = 12) (h_cylinder : cylinder_radius = 6) : ∃ v : ℝ, v^2 = 4374 :=
by
  sorry

end water_displacement_cube_in_cylinder_l1020_102051


namespace snake_paint_theorem_l1020_102009

/-- The amount of paint needed for a single cube -/
def paint_per_cube : ℕ := 60

/-- The number of cubes in the snake -/
def total_cubes : ℕ := 2016

/-- The number of cubes in one segment of the snake -/
def cubes_per_segment : ℕ := 6

/-- The amount of paint needed for one segment -/
def paint_per_segment : ℕ := 240

/-- The amount of extra paint needed for the ends of the snake -/
def extra_paint_for_ends : ℕ := 20

/-- Theorem stating the total amount of paint needed for the snake -/
theorem snake_paint_theorem :
  let segments := total_cubes / cubes_per_segment
  let paint_for_segments := segments * paint_per_segment
  paint_for_segments + extra_paint_for_ends = 80660 := by
  sorry

end snake_paint_theorem_l1020_102009


namespace batsman_average_after_11th_inning_l1020_102065

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalScore : ℕ
  averageScore : ℚ

/-- Calculates the new average score after an inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalScore + newInningScore : ℚ) / (stats.inningsPlayed + 1 : ℚ)

/-- Theorem: Given the conditions, the batsman's average after the 11th inning is 45 -/
theorem batsman_average_after_11th_inning
  (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 10)
  (h2 : newAverage stats 95 = stats.averageScore + 5) :
  newAverage stats 95 = 45 := by
  sorry

#check batsman_average_after_11th_inning

end batsman_average_after_11th_inning_l1020_102065


namespace lunch_with_tip_l1020_102037

/-- Calculate the total amount spent on lunch including tip -/
theorem lunch_with_tip (lunch_cost : ℝ) (tip_percentage : ℝ) :
  lunch_cost = 50.20 →
  tip_percentage = 20 →
  lunch_cost * (1 + tip_percentage / 100) = 60.24 := by
  sorry

end lunch_with_tip_l1020_102037


namespace gravel_pile_volume_l1020_102097

/-- The volume of a conical pile of gravel -/
theorem gravel_pile_volume (diameter : Real) (height_ratio : Real) : 
  diameter = 10 →
  height_ratio = 0.6 →
  let height := height_ratio * diameter
  let radius := diameter / 2
  let volume := (1 / 3) * Real.pi * radius^2 * height
  volume = 50 * Real.pi :=
by sorry

end gravel_pile_volume_l1020_102097


namespace system_solution_l1020_102042

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3 / 4 := by
  sorry

end system_solution_l1020_102042


namespace money_distribution_l1020_102076

/-- Represents the distribution of money among x, y, and z -/
structure Distribution where
  x : ℚ  -- Amount x gets in rupees
  y : ℚ  -- Amount y gets in rupees
  z : ℚ  -- Amount z gets in rupees

/-- The problem statement and conditions -/
theorem money_distribution (d : Distribution) : 
  -- For each rupee x gets, z gets 30 paisa
  d.z = 0.3 * d.x →
  -- The share of y is Rs. 27
  d.y = 27 →
  -- The total amount is Rs. 105
  d.x + d.y + d.z = 105 →
  -- Prove that y gets 45 paisa for each rupee x gets
  d.y / d.x = 0.45 := by
sorry

end money_distribution_l1020_102076


namespace expression_equality_l1020_102033

theorem expression_equality : 200 * (200 - 5) - (200 * 200 - 5) = -995 := by
  sorry

end expression_equality_l1020_102033


namespace train_speed_calculation_l1020_102007

/-- Theorem: Train Speed Calculation
Given a train of length 120 meters crossing a bridge of length 240 meters in 3 minutes,
prove that the speed of the train is 2 m/s. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 240 →
  crossing_time = 3 * 60 →
  (train_length + bridge_length) / crossing_time = 2 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1020_102007


namespace fraction_addition_l1020_102096

theorem fraction_addition (x P Q : ℚ) : 
  (8 * x^2 - 9 * x + 20) / (4 * x^3 - 5 * x^2 - 26 * x + 24) = 
  P / (2 * x^2 - 5 * x + 3) + Q / (2 * x - 3) →
  P = 4/9 ∧ Q = 68/9 := by
sorry

end fraction_addition_l1020_102096


namespace complement_of_P_l1020_102050

def U : Set ℝ := Set.univ

def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_of_P : 
  (Set.univ \ P) = {x | x < -1 ∨ x > 1} := by sorry

end complement_of_P_l1020_102050


namespace unique_function_satisfying_conditions_l1020_102055

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x > 0, f x > 0}

/-- The property f(x f(y)) = y f(x) for all positive real x, y -/
def HasFunctionalProperty (f : PositiveRealFunction) :=
  ∀ x y, x > 0 → y > 0 → f.val (x * f.val y) = y * f.val x

/-- The property that f(x) → 0 as x → +∞ -/
def TendsToZeroAtInfinity (f : PositiveRealFunction) :=
  ∀ ε > 0, ∃ M, ∀ x > M, f.val x < ε

/-- The main theorem -/
theorem unique_function_satisfying_conditions
  (f : PositiveRealFunction)
  (h1 : HasFunctionalProperty f)
  (h2 : TendsToZeroAtInfinity f) :
  ∀ x > 0, f.val x = 1 / x :=
sorry

end unique_function_satisfying_conditions_l1020_102055


namespace sum_of_T_l1020_102089

/-- The sum of the geometric series for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ := 18 / (1 - r)

/-- Theorem: Sum of T(b) and T(-b) equals 337.5 -/
theorem sum_of_T (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3024) :
  T b + T (-b) = 337.5 := by
  sorry

end sum_of_T_l1020_102089


namespace hockey_league_games_l1020_102052

/-- Calculate the number of games in a hockey league season -/
theorem hockey_league_games (num_teams : ℕ) (face_times : ℕ) : 
  num_teams = 18 → face_times = 10 → 
  (num_teams * (num_teams - 1) / 2) * face_times = 1530 :=
by sorry

end hockey_league_games_l1020_102052


namespace color_fractions_l1020_102031

-- Define the color type
inductive Color
  | Red
  | Blue

-- Define the coloring function
def color : ℚ → Color := sorry

-- Define the coloring rules
axiom color_one : color 1 = Color.Red
axiom color_diff_one (x : ℚ) : color (x + 1) ≠ color x
axiom color_reciprocal (x : ℚ) (h : x ≠ 1) : color (1 / x) ≠ color x

-- State the theorem
theorem color_fractions :
  color (2013 / 2014) = Color.Red ∧ color (2 / 7) = Color.Blue :=
sorry

end color_fractions_l1020_102031


namespace mayoral_election_votes_l1020_102000

theorem mayoral_election_votes (Z Y X : ℕ) : 
  Z = 25000 → 
  Y = Z - (2/5 : ℚ) * Z →
  X = Y + (1/2 : ℚ) * Y →
  X = 22500 := by
  sorry

end mayoral_election_votes_l1020_102000


namespace octal_sum_theorem_l1020_102026

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The main theorem stating that the sum of 642₈ and 157₈ in base 8 is 1021₈ -/
theorem octal_sum_theorem :
  decimal_to_octal (octal_to_decimal [6, 4, 2] + octal_to_decimal [1, 5, 7]) = [1, 0, 2, 1] :=
sorry

end octal_sum_theorem_l1020_102026


namespace sum_after_removal_l1020_102090

theorem sum_after_removal (a b c d e f : ℚ) : 
  a = 1/3 → b = 1/6 → c = 1/9 → d = 1/12 → e = 1/15 → f = 1/18 →
  a + b + c + f = 3/4 := by
  sorry

end sum_after_removal_l1020_102090


namespace two_zeros_implies_a_is_inverse_e_l1020_102001

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x + x else a*x - Real.log x

theorem two_zeros_implies_a_is_inverse_e (a : ℝ) (h_a_pos : a > 0) :
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  a = Real.exp (-1) :=
sorry

end two_zeros_implies_a_is_inverse_e_l1020_102001


namespace group_formation_count_l1020_102003

def total_people : ℕ := 7
def group_size_1 : ℕ := 3
def group_size_2 : ℕ := 4

theorem group_formation_count :
  Nat.choose total_people group_size_1 = 35 :=
by sorry

end group_formation_count_l1020_102003


namespace f_symmetry_l1020_102069

/-- A function f(x) = x^5 + ax^3 + bx - 8 for some real a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

/-- Theorem: If f(-2) = 10, then f(2) = -26 -/
theorem f_symmetry (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end f_symmetry_l1020_102069


namespace unique_quadratic_solution_l1020_102062

/-- Represents a quadratic equation ax² - 6x + c = 0 with exactly one solution -/
structure UniqueQuadratic where
  a : ℝ
  c : ℝ
  has_unique_solution : ∃! x, a * x^2 - 6 * x + c = 0

theorem unique_quadratic_solution (q : UniqueQuadratic)
  (sum_eq_12 : q.a + q.c = 12)
  (a_lt_c : q.a < q.c) :
  q.a = 6 - 3 * Real.sqrt 3 ∧ q.c = 6 + 3 * Real.sqrt 3 :=
sorry

end unique_quadratic_solution_l1020_102062


namespace a_investment_is_6300_l1020_102028

/-- Represents the investment and profit scenario of a partnership business -/
structure BusinessPartnership where
  /-- A's investment amount -/
  a_investment : ℝ
  /-- B's investment amount -/
  b_investment : ℝ
  /-- C's investment amount -/
  c_investment : ℝ
  /-- Total profit -/
  total_profit : ℝ
  /-- A's share of the profit -/
  a_profit : ℝ

/-- Theorem stating that given the conditions, A's investment is 6300 -/
theorem a_investment_is_6300 (bp : BusinessPartnership)
  (h1 : bp.b_investment = 4200)
  (h2 : bp.c_investment = 10500)
  (h3 : bp.total_profit = 12700)
  (h4 : bp.a_profit = 3810)
  (h5 : bp.a_profit / bp.total_profit = bp.a_investment / (bp.a_investment + bp.b_investment + bp.c_investment)) :
  bp.a_investment = 6300 := by
  sorry

end a_investment_is_6300_l1020_102028


namespace yolka_probability_l1020_102064

/-- Represents the time in minutes after 15:00 -/
def Time := Fin 60

/-- The waiting time for Vasya in minutes -/
def vasyaWaitTime : ℕ := 15

/-- The waiting time for Boris in minutes -/
def borisWaitTime : ℕ := 10

/-- The probability that Anya arrives last -/
def probAnyaLast : ℚ := 1/3

/-- The area in the time square where Boris and Vasya meet -/
def meetingArea : ℕ := 3500

/-- The total area of the time square -/
def totalArea : ℕ := 3600

/-- The probability that all three go to Yolka together -/
def probAllTogether : ℚ := probAnyaLast * (meetingArea / totalArea)

theorem yolka_probability :
  probAllTogether = 1/3 * (3500/3600) :=
sorry

end yolka_probability_l1020_102064


namespace unread_pages_after_two_weeks_l1020_102014

theorem unread_pages_after_two_weeks (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) (unread_pages : ℕ) : 
  total_pages = 200 →
  pages_per_day = 12 →
  days = 14 →
  unread_pages = total_pages - (pages_per_day * days) →
  unread_pages = 32 := by
sorry

end unread_pages_after_two_weeks_l1020_102014


namespace tangent_line_intersection_l1020_102034

theorem tangent_line_intersection (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x₀ = y₀ →
  (∀ x, f x = x^3 + 11) →
  x₀ = 1 →
  y₀ = 12 →
  ∃ m : ℝ, ∀ x y, y - y₀ = m * (x - x₀) →
    y = 0 →
    x = -3 :=
by sorry

end tangent_line_intersection_l1020_102034


namespace f_composition_negative_two_l1020_102085

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_composition_negative_two (f : ℝ → ℝ) :
  (∀ x, x ≥ 0 → f x = 1 - Real.sqrt x) →
  (∀ x, x < 0 → f x = 2^x) →
  f (f (-2)) = 1/2 := by
  sorry

end f_composition_negative_two_l1020_102085


namespace function_inequality_l1020_102021

/-- Given f(x) = e^(2x) - ax, for all x > 0, if f(x) > ax^2 + 1, then a ≤ 2 -/
theorem function_inequality (a : ℝ) : 
  (∀ x > 0, Real.exp (2 * x) - a * x > a * x^2 + 1) → a ≤ 2 := by
  sorry

end function_inequality_l1020_102021
