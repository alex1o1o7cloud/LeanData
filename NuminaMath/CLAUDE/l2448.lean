import Mathlib

namespace inequality_solution_l2448_244841

theorem inequality_solution (x : ℕ) : 
  (x + 3 : ℚ) / (x^2 - 4) - 1 / (x + 2) < 2 * x / (2 * x - x^2) ↔ x = 1 :=
by sorry

end inequality_solution_l2448_244841


namespace quadratic_function_proof_l2448_244845

theorem quadratic_function_proof (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = -2 ∨ x = 4) →
  (∃ x, ∀ y, a * y^2 + b * y + c ≤ a * x^2 + b * x + c) →
  (∃ x, a * x^2 + b * x + c = 9) →
  (∀ x, a * x^2 + b * x + c = -x^2 + 2*x + 8) :=
by sorry

end quadratic_function_proof_l2448_244845


namespace complex_number_in_third_quadrant_l2448_244800

theorem complex_number_in_third_quadrant :
  let z : ℂ := -Complex.I / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end complex_number_in_third_quadrant_l2448_244800


namespace smallest_perfect_square_divisible_by_2_3_5_l2448_244834

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 2 = 0 → n % 3 = 0 → n % 5 = 0 → n ≥ 225 := by
  sorry

end smallest_perfect_square_divisible_by_2_3_5_l2448_244834


namespace donnas_truck_dryers_l2448_244802

/-- Calculates the number of dryers on Donna's truck given the weight constraints --/
theorem donnas_truck_dryers :
  let bridge_limit : ℕ := 20000
  let empty_truck_weight : ℕ := 12000
  let num_soda_crates : ℕ := 20
  let soda_crate_weight : ℕ := 50
  let dryer_weight : ℕ := 3000
  let loaded_truck_weight : ℕ := 24000
  let soda_weight : ℕ := num_soda_crates * soda_crate_weight
  let produce_weight : ℕ := 2 * soda_weight
  let truck_with_soda_produce : ℕ := empty_truck_weight + soda_weight + produce_weight
  let dryers_weight : ℕ := loaded_truck_weight - truck_with_soda_produce
  let num_dryers : ℕ := dryers_weight / dryer_weight
  num_dryers = 3 :=
by
  sorry


end donnas_truck_dryers_l2448_244802


namespace adjusted_target_heart_rate_for_30_year_old_l2448_244853

/-- Calculates the adjusted target heart rate for a runner --/
def adjustedTargetHeartRate (age : ℕ) : ℕ :=
  let maxHeartRate : ℕ := 220 - age
  let initialTargetRate : ℚ := 0.7 * maxHeartRate
  let adjustment : ℚ := 0.1 * initialTargetRate
  let adjustedRate : ℚ := initialTargetRate + adjustment
  (adjustedRate + 0.5).floor.toNat

/-- Theorem stating that for a 30-year-old runner, the adjusted target heart rate is 146 bpm --/
theorem adjusted_target_heart_rate_for_30_year_old :
  adjustedTargetHeartRate 30 = 146 := by
  sorry

#eval adjustedTargetHeartRate 30

end adjusted_target_heart_rate_for_30_year_old_l2448_244853


namespace marching_band_ratio_l2448_244854

theorem marching_band_ratio (total_students : ℕ) (marching_band_fraction : ℚ) 
  (brass_to_saxophone : ℚ) (saxophone_to_alto : ℚ) (alto_players : ℕ) :
  total_students = 600 →
  marching_band_fraction = 1 / 5 →
  brass_to_saxophone = 1 / 5 →
  saxophone_to_alto = 1 / 3 →
  alto_players = 4 →
  (↑alto_players / (marching_band_fraction * saxophone_to_alto * brass_to_saxophone)) / 
  (marching_band_fraction * ↑total_students) = 1 / 2 := by
  sorry

end marching_band_ratio_l2448_244854


namespace negation_of_proposition_l2448_244836

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 1 → x - 1 ≥ Real.log x) ↔ (∃ x : ℝ, x > 1 ∧ x - 1 < Real.log x) := by
  sorry

end negation_of_proposition_l2448_244836


namespace complex_magnitude_l2448_244840

theorem complex_magnitude (z : ℂ) (h1 : z.im = 2) (h2 : (z^2 + 3).re = 0) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l2448_244840


namespace new_person_weight_l2448_244875

theorem new_person_weight (n : ℕ) (initial_avg weight_replaced increase : ℝ) :
  n = 8 →
  initial_avg = 57 →
  weight_replaced = 55 →
  increase = 1.5 →
  (n * initial_avg + (weight_replaced + increase * n) - weight_replaced) / n = initial_avg + increase →
  weight_replaced + increase * n = 67 :=
by sorry

end new_person_weight_l2448_244875


namespace mode_median_constant_l2448_244890

/-- Represents the age distribution of a club --/
structure AgeDistribution where
  age13 : ℕ
  age14 : ℕ
  age15 : ℕ
  age16 : ℕ
  age17 : ℕ
  total : ℕ
  sum_eq_total : age13 + age14 + age15 + age16 + age17 = total

/-- The age distribution of the club --/
def clubDistribution (x : ℕ) : AgeDistribution where
  age13 := 5
  age14 := 12
  age15 := x
  age16 := 11 - x
  age17 := 2
  total := 30
  sum_eq_total := by sorry

/-- The mode of the age distribution --/
def mode (d : AgeDistribution) : ℕ := 
  max d.age13 (max d.age14 (max d.age15 (max d.age16 d.age17)))

/-- The median of the age distribution --/
def median (d : AgeDistribution) : ℚ := 14

theorem mode_median_constant (x : ℕ) : 
  mode (clubDistribution x) = 14 ∧ median (clubDistribution x) = 14 := by sorry

end mode_median_constant_l2448_244890


namespace stating_carlas_counting_problem_l2448_244809

/-- 
Theorem stating that there exists a positive integer solution for the number of tiles and books
that satisfies the equation from Carla's counting problem.
-/
theorem carlas_counting_problem :
  ∃ (T B : ℕ), T > 0 ∧ B > 0 ∧ 2 * T + 3 * B = 301 := by
  sorry

end stating_carlas_counting_problem_l2448_244809


namespace special_line_equation_l2448_244810

/-- A line passing through (-2, 2) forming a triangle with area 1 with the coordinate axes -/
structure SpecialLine where
  /-- Slope of the line -/
  k : ℝ
  /-- The line passes through (-2, 2) -/
  passes_through : 2 = k * (-2) + 2
  /-- The area of the triangle formed with the axes is 1 -/
  triangle_area : |4 + 2/k + 2*k| = 1

/-- The equation of a SpecialLine is either x + 2y - 2 = 0 or 2x + y + 2 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.k = -1/2 ∧ ∀ x y, x + 2*y - 2 = 0 ↔ y = l.k * x + 2) ∨
  (l.k = -2 ∧ ∀ x y, 2*x + y + 2 = 0 ↔ y = l.k * x + 2) :=
sorry

end special_line_equation_l2448_244810


namespace tan_quadruple_angle_l2448_244816

theorem tan_quadruple_angle (θ : Real) (h : Real.tan θ = 3) : 
  Real.tan (4 * θ) = -24 / 7 := by
  sorry

end tan_quadruple_angle_l2448_244816


namespace expand_and_simplify_l2448_244850

theorem expand_and_simplify (x : ℝ) : (17*x - 9) * 3*x = 51*x^2 - 27*x := by
  sorry

end expand_and_simplify_l2448_244850


namespace committee_selection_ways_l2448_244831

theorem committee_selection_ways (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 5) :
  Nat.choose n k = 118755 := by
  sorry

end committee_selection_ways_l2448_244831


namespace parallelogram_side_length_l2448_244847

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h : side1 = 3 * s) 
  (h' : side2 = s) 
  (h'' : angle = π / 3) 
  (h''' : area = 9 * Real.sqrt 3) 
  (h'''' : area = side1 * side2 * Real.sin angle) : 
  s = Real.sqrt 6 := by
sorry

end parallelogram_side_length_l2448_244847


namespace sum_of_radii_is_eight_l2448_244884

/-- A circle with center C that is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (3,0) with radius 1 -/
def CircleC (r : ℝ) : Prop :=
  (∃ C : ℝ × ℝ, C.1 = r ∧ C.2 = r) ∧  -- Center of circle C is at (r,r)
  ((r - 3)^2 + r^2 = (r + 1)^2)  -- External tangency condition

/-- The theorem stating that the sum of all possible radii of CircleC is 8 -/
theorem sum_of_radii_is_eight :
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ CircleC r₁ ∧ CircleC r₂ ∧ r₁ + r₂ = 8 :=
sorry

end sum_of_radii_is_eight_l2448_244884


namespace complement_union_theorem_l2448_244876

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end complement_union_theorem_l2448_244876


namespace cindy_calculation_l2448_244880

theorem cindy_calculation (x : ℝ) : 
  ((x - 10) / 5 = 40) → ((x - 4) / 10 = 20.6) := by
  sorry

end cindy_calculation_l2448_244880


namespace mangoes_per_box_l2448_244839

/-- Given a total of 4320 mangoes distributed equally among 36 boxes,
    prove that there are 10 dozens of mangoes in each box. -/
theorem mangoes_per_box (total_mangoes : Nat) (num_boxes : Nat) 
    (h1 : total_mangoes = 4320) (h2 : num_boxes = 36) :
    (total_mangoes / (12 * num_boxes) : Nat) = 10 := by
  sorry

end mangoes_per_box_l2448_244839


namespace sum_of_coefficients_zero_l2448_244852

theorem sum_of_coefficients_zero (x y : ℝ) : 
  (fun x y => (3 * x^2 - 5 * x * y + 2 * y^2)^5) 1 1 = 0 := by sorry

end sum_of_coefficients_zero_l2448_244852


namespace roots_real_implies_ab_nonpositive_l2448_244882

/-- The polynomial x^4 + ax^3 + bx + c has all real roots -/
def has_all_real_roots (a b c : ℝ) : Prop :=
  ∀ x : ℂ, x^4 + a*x^3 + b*x + c = 0 → x.im = 0

/-- If all roots of the polynomial x^4 + ax^3 + bx + c are real numbers, then ab ≤ 0 -/
theorem roots_real_implies_ab_nonpositive (a b c : ℝ) :
  has_all_real_roots a b c → a * b ≤ 0 := by
  sorry


end roots_real_implies_ab_nonpositive_l2448_244882


namespace negation_equivalence_l2448_244806

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end negation_equivalence_l2448_244806


namespace weight_loss_challenge_l2448_244886

theorem weight_loss_challenge (W : ℝ) (x : ℝ) (h : x > 0) :
  W * (1 - x / 100 + 2 / 100) = W * (100 - 12.28) / 100 →
  x = 14.28 := by
sorry

end weight_loss_challenge_l2448_244886


namespace theater_seat_count_l2448_244838

/-- Calculates the total number of seats in a theater with the given configuration. -/
def theater_seats (total_rows : ℕ) (odd_row_seats : ℕ) (even_row_seats : ℕ) : ℕ :=
  let odd_rows := (total_rows + 1) / 2
  let even_rows := total_rows / 2
  odd_rows * odd_row_seats + even_rows * even_row_seats

/-- Theorem stating that a theater with 11 rows, where odd rows have 15 seats
    and even rows have 16 seats, has a total of 170 seats. -/
theorem theater_seat_count :
  theater_seats 11 15 16 = 170 := by
  sorry

#eval theater_seats 11 15 16

end theater_seat_count_l2448_244838


namespace election_results_l2448_244887

theorem election_results (total_votes : ℕ) 
  (votes_A votes_B votes_C : ℕ) : 
  votes_A = (35 : ℕ) * total_votes / 100 →
  votes_B = votes_A + 1800 →
  votes_C = votes_A / 2 →
  total_votes = votes_A + votes_B + votes_C →
  total_votes = 14400 ∧
  (votes_A : ℚ) / total_votes = 35 / 100 ∧
  (votes_B : ℚ) / total_votes = 475 / 1000 ∧
  (votes_C : ℚ) / total_votes = 175 / 1000 :=
by
  sorry

#check election_results

end election_results_l2448_244887


namespace lines_not_form_triangle_l2448_244837

/-- A line in the xy-plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The three lines in the problem -/
def l1 (m : ℝ) : Line := ⟨3, m, -1⟩
def l2 : Line := ⟨3, -2, -5⟩
def l3 : Line := ⟨6, 1, -5⟩

/-- Theorem stating the conditions under which the three lines cannot form a triangle -/
theorem lines_not_form_triangle (m : ℝ) : 
  (¬(∃ x y : ℝ, 3*x + m*y - 1 = 0 ∧ 3*x - 2*y - 5 = 0 ∧ 6*x + y - 5 = 0)) ↔ 
  (m = -2 ∨ m = 1/2) :=
sorry

end lines_not_form_triangle_l2448_244837


namespace elvis_song_writing_time_l2448_244817

/-- Proves that Elvis spent 15 minutes writing each song given the conditions of his album production. -/
theorem elvis_song_writing_time :
  let total_songs : ℕ := 10
  let total_studio_time : ℕ := 5 * 60  -- in minutes
  let recording_time_per_song : ℕ := 12
  let editing_time_all_songs : ℕ := 30
  let total_recording_time := total_songs * recording_time_per_song
  let remaining_time := total_studio_time - total_recording_time - editing_time_all_songs
  let writing_time_per_song := remaining_time / total_songs
  writing_time_per_song = 15 := by
    sorry

#check elvis_song_writing_time

end elvis_song_writing_time_l2448_244817


namespace trajectory_of_P_l2448_244832

-- Define the coordinate system
variable (O : ℝ × ℝ)  -- Origin
variable (A B P : ℝ → ℝ × ℝ)  -- Points as functions of time

-- Define the conditions
axiom origin : O = (0, 0)
axiom A_on_x_axis : ∀ t, (A t).2 = 0
axiom B_on_y_axis : ∀ t, (B t).1 = 0
axiom AB_length : ∀ t, Real.sqrt ((A t).1^2 + (B t).2^2) = 3
axiom P_position : ∀ t, P t = (2/3 • A t) + (1/3 • B t)

-- State the theorem
theorem trajectory_of_P :
  ∀ t, (P t).1^2 / 4 + (P t).2^2 = 1 :=
sorry

end trajectory_of_P_l2448_244832


namespace unique_function_satisfying_inequality_l2448_244835

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x * z) + f (y * z) - f x * f y * f z ≥ 1

theorem unique_function_satisfying_inequality :
  ∃! f : ℝ → ℝ, satisfies_inequality f ∧ ∀ x : ℝ, f x = 1 :=
sorry

end unique_function_satisfying_inequality_l2448_244835


namespace negation_equivalence_l2448_244894

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 5^x + Real.sin x ≤ 0) ↔ (∀ x : ℝ, 5^x + Real.sin x > 0) := by sorry

end negation_equivalence_l2448_244894


namespace product_equals_square_l2448_244825

theorem product_equals_square : 500 * 2019 * 0.0505 * 20 = (2019 : ℝ)^2 := by
  sorry

end product_equals_square_l2448_244825


namespace complex_simplification_l2448_244872

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The property of the imaginary unit -/
axiom i_squared : i * i = -1

/-- The theorem to prove -/
theorem complex_simplification :
  3 * (2 - 2 * i) + 2 * i * (3 + i) = (4 : ℂ) := by sorry

end complex_simplification_l2448_244872


namespace correct_factorization_l2448_244823

theorem correct_factorization (x y : ℝ) : x * (x - y) - y * (x - y) = (x - y)^2 := by
  sorry

end correct_factorization_l2448_244823


namespace inverse_z_minus_z_inv_l2448_244898

/-- Given a complex number z = 1 + i where i² = -1, prove that (z - z⁻¹)⁻¹ = (1 - 3i) / 5 -/
theorem inverse_z_minus_z_inv (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 1 + i
  (z - z⁻¹)⁻¹ = (1 - 3*i) / 5 := by
  sorry

end inverse_z_minus_z_inv_l2448_244898


namespace outfit_count_l2448_244864

/-- Represents the colors available for clothing items -/
inductive Color
| Red | Black | Blue | Gray | Green | Purple | White

/-- Represents a clothing item -/
structure ClothingItem :=
  (color : Color)

/-- Represents an outfit -/
structure Outfit :=
  (shirt : ClothingItem)
  (pants : ClothingItem)
  (hat : ClothingItem)

def is_monochrome (outfit : Outfit) : Prop :=
  outfit.shirt.color = outfit.pants.color ∧ outfit.shirt.color = outfit.hat.color

def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_hats : Nat := 7
def num_pants_colors : Nat := 5
def num_shirt_hat_colors : Nat := 7

theorem outfit_count :
  let total_outfits := num_shirts * num_pants * num_hats
  let monochrome_outfits := num_pants_colors
  (total_outfits - monochrome_outfits : Nat) = 275 := by sorry

end outfit_count_l2448_244864


namespace sum_less_than_six_for_735_l2448_244821

def is_less_than_six (n : ℕ) : Bool :=
  n < 6

def sum_less_than_six (cards : List ℕ) : ℕ :=
  (cards.filter is_less_than_six).sum

theorem sum_less_than_six_for_735 : 
  ∃ (cards : List ℕ), 
    cards.length = 3 ∧ 
    (∀ n ∈ cards, 1 ≤ n ∧ n ≤ 9) ∧
    cards.foldl (λ acc d => acc * 10 + d) 0 = 735 ∧
    sum_less_than_six cards = 8 :=
by
  sorry

end sum_less_than_six_for_735_l2448_244821


namespace fraction_doubles_l2448_244855

theorem fraction_doubles (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2*x)*(2*y) / ((2*x) + (2*y)) = 2 * (x*y / (x + y)) := by
  sorry

end fraction_doubles_l2448_244855


namespace bluray_movies_returned_l2448_244827

/-- Represents the number of movies returned -/
def movies_returned (initial_dvd : ℕ) (initial_bluray : ℕ) (final_dvd : ℕ) (final_bluray : ℕ) : ℕ :=
  initial_bluray - final_bluray

theorem bluray_movies_returned :
  ∀ (initial_dvd initial_bluray final_dvd final_bluray : ℕ),
    initial_dvd + initial_bluray = 378 →
    initial_dvd * 4 = initial_bluray * 17 →
    final_dvd * 2 = final_bluray * 9 →
    final_dvd = initial_dvd →
    movies_returned initial_dvd initial_bluray final_dvd final_bluray = 4 := by
  sorry

end bluray_movies_returned_l2448_244827


namespace female_officers_count_l2448_244892

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percentage : ℚ) 
  (h1 : total_on_duty = 144)
  (h2 : female_on_duty_percentage = 18 / 100)
  (h3 : (total_on_duty / 2 : ℚ) = female_on_duty_percentage * female_total) :
  female_total = 400 :=
by
  sorry

end female_officers_count_l2448_244892


namespace candy_boxes_l2448_244826

theorem candy_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 500) (h2 : total_pieces = 3000) :
  total_pieces / pieces_per_box = 6 := by
sorry

end candy_boxes_l2448_244826


namespace correct_reasoning_statements_l2448_244889

/-- Represents different types of reasoning -/
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
  | PartToWhole
  | GeneralToGeneral
  | GeneralToSpecific
  | SpecificToGeneral
  | SpecificToSpecific

/-- Defines the correct reasoning direction for each reasoning type -/
def correct_reasoning (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

/-- Theorem stating the correct reasoning directions for each type -/
theorem correct_reasoning_statements :
  (correct_reasoning ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (correct_reasoning ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (correct_reasoning ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end correct_reasoning_statements_l2448_244889


namespace abs_sum_inequality_positive_reals_inequality_l2448_244846

-- Problem 1
theorem abs_sum_inequality (x : ℝ) :
  |x - 1| + |x + 1| ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 := by sorry

-- Problem 2
theorem positive_reals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ a + b + c := by sorry

end abs_sum_inequality_positive_reals_inequality_l2448_244846


namespace largest_number_with_conditions_l2448_244888

def is_valid_digit (d : ℕ) : Prop := d = 4 ∨ d = 5

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def all_digits_valid (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_conditions :
  ∀ n : ℕ,
    all_digits_valid n →
    digit_sum n = 17 →
    n ≤ 5444 :=
sorry

end largest_number_with_conditions_l2448_244888


namespace curve_tangent_l2448_244819

/-- Given a curve C defined by x = √2 cos(φ) and y = sin(φ), prove that for a point M on C,
    if the angle between OM and the positive x-axis is π/3, then tan(φ) = √6. -/
theorem curve_tangent (φ : ℝ) : 
  let M : ℝ × ℝ := (Real.sqrt 2 * Real.cos φ, Real.sin φ)
  (M.2 / M.1 = Real.tan (π / 3)) → Real.tan φ = Real.sqrt 6 := by
  sorry

end curve_tangent_l2448_244819


namespace b_completes_in_12_days_l2448_244824

/-- The number of days B takes to complete the remaining work after A works for 5 days -/
def days_B_completes_work (a_rate b_rate : ℚ) (a_days : ℕ) : ℚ :=
  (1 - a_rate * a_days) / b_rate

theorem b_completes_in_12_days :
  let a_rate : ℚ := 1 / 15
  let b_rate : ℚ := 1 / 18
  let a_days : ℕ := 5
  days_B_completes_work a_rate b_rate a_days = 12 := by
sorry

end b_completes_in_12_days_l2448_244824


namespace flight_speed_l2448_244807

/-- Given a flight distance and time, calculate the speed -/
theorem flight_speed (distance : ℝ) (time : ℝ) (h1 : distance = 256) (h2 : time = 8) :
  distance / time = 32 := by
  sorry

end flight_speed_l2448_244807


namespace probability_receive_one_l2448_244883

/-- Probability of receiving a signal as 1 in a digital communication system with given error rates --/
theorem probability_receive_one (p_receive_zero_given_send_zero : ℝ)
                                (p_receive_one_given_send_zero : ℝ)
                                (p_receive_one_given_send_one : ℝ)
                                (p_receive_zero_given_send_one : ℝ)
                                (p_send_zero : ℝ)
                                (p_send_one : ℝ)
                                (h1 : p_receive_zero_given_send_zero = 0.9)
                                (h2 : p_receive_one_given_send_zero = 0.1)
                                (h3 : p_receive_one_given_send_one = 0.95)
                                (h4 : p_receive_zero_given_send_one = 0.05)
                                (h5 : p_send_zero = 0.5)
                                (h6 : p_send_one = 0.5) :
  p_send_zero * p_receive_one_given_send_zero + p_send_one * p_receive_one_given_send_one = 0.525 :=
by sorry

end probability_receive_one_l2448_244883


namespace salem_poem_word_count_l2448_244868

/-- Represents a poem with a specific structure -/
structure Poem where
  stanzas : Nat
  lines_per_stanza : Nat
  words_per_line : Nat

/-- Calculates the total number of words in a poem -/
def total_words (p : Poem) : Nat :=
  p.stanzas * p.lines_per_stanza * p.words_per_line

/-- Theorem: A poem with 35 stanzas, 15 lines per stanza, and 12 words per line has 6300 words -/
theorem salem_poem_word_count :
  let p : Poem := { stanzas := 35, lines_per_stanza := 15, words_per_line := 12 }
  total_words p = 6300 := by
  sorry

#eval total_words { stanzas := 35, lines_per_stanza := 15, words_per_line := 12 }

end salem_poem_word_count_l2448_244868


namespace coin_exchange_terminates_l2448_244814

-- Define the Dwarf type
structure Dwarf where
  id : Nat
  coins : Nat
  acquaintances : List Nat

-- Define the Clan type
def Clan := List Dwarf

-- Function to represent a single day's coin exchange
def exchangeCoins (clan : Clan) : Clan :=
  sorry

-- Theorem statement
theorem coin_exchange_terminates (initialClan : Clan) :
  ∃ n : Nat, ∀ m : Nat, m ≥ n → exchangeCoins^[m] initialClan = exchangeCoins^[n] initialClan :=
sorry

end coin_exchange_terminates_l2448_244814


namespace lemon_bag_mass_l2448_244858

theorem lemon_bag_mass (max_load : ℕ) (num_bags : ℕ) (remaining_capacity : ℕ) 
  (h1 : max_load = 900)
  (h2 : num_bags = 100)
  (h3 : remaining_capacity = 100) :
  (max_load - remaining_capacity) / num_bags = 8 := by
  sorry

end lemon_bag_mass_l2448_244858


namespace correct_linear_regression_l2448_244870

-- Define the variables and constants
variable (x y : ℝ)
def x_mean : ℝ := 2.5
def y_mean : ℝ := 3.5

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 0.4 * x + 2.5

-- State the theorem
theorem correct_linear_regression :
  (∃ r : ℝ, r > 0 ∧ (∀ x y : ℝ, y - y_mean = r * (x - x_mean))) →  -- Positive correlation
  (linear_regression x_mean = y_mean) →                           -- Passes through (x̄, ȳ)
  (∀ x : ℝ, linear_regression x = 0.4 * x + 2.5) :=               -- The equation is correct
by sorry

end correct_linear_regression_l2448_244870


namespace difference_of_squares_l2448_244804

theorem difference_of_squares (a b : ℝ) : (3*a + b) * (3*a - b) = 9*a^2 - b^2 := by
  sorry

end difference_of_squares_l2448_244804


namespace david_homework_hours_l2448_244801

/-- Calculates the weekly homework hours for a course -/
def weekly_homework_hours (total_weeks : ℕ) (class_hours_per_week : ℕ) (total_course_hours : ℕ) : ℕ :=
  (total_course_hours - (total_weeks * class_hours_per_week)) / total_weeks

theorem david_homework_hours :
  let total_weeks : ℕ := 24
  let three_hour_classes : ℕ := 2
  let four_hour_classes : ℕ := 1
  let class_hours_per_week : ℕ := three_hour_classes * 3 + four_hour_classes * 4
  let total_course_hours : ℕ := 336
  weekly_homework_hours total_weeks class_hours_per_week total_course_hours = 4 := by
  sorry

end david_homework_hours_l2448_244801


namespace least_frood_drop_beats_eat_l2448_244830

def frood_drop_score (n : ℕ) : ℕ := n * (n + 1) / 2
def frood_eat_score (n : ℕ) : ℕ := 15 * n

theorem least_frood_drop_beats_eat :
  ∀ k : ℕ, k < 30 → frood_drop_score k ≤ frood_eat_score k ∧
  frood_drop_score 30 > frood_eat_score 30 :=
sorry

end least_frood_drop_beats_eat_l2448_244830


namespace infinite_partition_numbers_l2448_244866

theorem infinite_partition_numbers : ∃ (f : ℕ → ℕ), Infinite {n : ℕ | ∃ k, n = f k ∧ n % 4 = 1 ∧ (3 * n * (3 * n + 1) / 2) % (6 * n) = 0} :=
sorry

end infinite_partition_numbers_l2448_244866


namespace melinda_paid_759_l2448_244818

-- Define the cost of items
def doughnut_cost : ℚ := 0.45
def coffee_cost : ℚ := (4.91 - 3 * doughnut_cost) / 4

-- Define Melinda's purchase
def melinda_doughnuts : ℕ := 5
def melinda_coffees : ℕ := 6

-- Define Melinda's total cost
def melinda_total_cost : ℚ := melinda_doughnuts * doughnut_cost + melinda_coffees * coffee_cost

-- Theorem to prove
theorem melinda_paid_759 : melinda_total_cost = 7.59 := by
  sorry

end melinda_paid_759_l2448_244818


namespace discount_profit_calculation_l2448_244891

/-- Calculates the profit percentage with discount given the discount rate and profit without discount -/
def profit_with_discount (discount : ℝ) (profit_without_discount : ℝ) : ℝ :=
  let marked_price := 1 + profit_without_discount
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that with a 5% discount and 30% profit without discount, 
    the profit with discount is 23.5% -/
theorem discount_profit_calculation :
  profit_with_discount 0.05 0.30 = 23.5 := by
  sorry

end discount_profit_calculation_l2448_244891


namespace equal_utility_implies_u_equals_four_l2448_244896

def sunday_utility (u : ℝ) : ℝ := 2 * u * (10 - 2 * u)
def monday_utility (u : ℝ) : ℝ := 2 * (4 - 2 * u) * (2 * u + 4)

theorem equal_utility_implies_u_equals_four :
  ∀ u : ℝ, sunday_utility u = monday_utility u → u = 4 :=
by sorry

end equal_utility_implies_u_equals_four_l2448_244896


namespace log_equation_holds_l2448_244893

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 3) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 3 := by
  sorry

end log_equation_holds_l2448_244893


namespace savings_theorem_l2448_244885

/-- Represents the savings and interest calculations for Dick and Jane --/
structure Savings where
  dick_1989 : ℝ
  jane_1989 : ℝ
  dick_increase_rate : ℝ
  interest_rate : ℝ

/-- Calculates the total savings of Dick and Jane in 1990 --/
def total_savings_1990 (s : Savings) : ℝ :=
  (s.dick_1989 * (1 + s.dick_increase_rate) + s.dick_1989) * (1 + s.interest_rate) +
  s.jane_1989 * (1 + s.interest_rate)

/-- Calculates the percent change in Jane's savings from 1989 to 1990 --/
def jane_savings_percent_change (s : Savings) : ℝ := 0

/-- Theorem stating the total savings in 1990 and Jane's savings percent change --/
theorem savings_theorem (s : Savings) 
  (h1 : s.dick_1989 = 5000)
  (h2 : s.jane_1989 = 3000)
  (h3 : s.dick_increase_rate = 0.1)
  (h4 : s.interest_rate = 0.03) :
  total_savings_1990 s = 8740 ∧ jane_savings_percent_change s = 0 := by
  sorry

end savings_theorem_l2448_244885


namespace incorrect_observation_value_l2448_244812

theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 36) 
  (h3 : new_mean = 36.5) 
  (h4 : correct_value = 45) :
  ∃ (incorrect_value : ℝ), 
    (n : ℝ) * original_mean = (n : ℝ) * new_mean - correct_value + incorrect_value ∧ 
    incorrect_value = 20 := by
  sorry

end incorrect_observation_value_l2448_244812


namespace points_in_different_half_spaces_l2448_244803

/-- A plane in 3D space defined by the equation ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Determine if two points are on opposite sides of a plane --/
def oppositeHalfSpaces (plane : Plane) (p1 p2 : Point3D) : Prop :=
  (plane.a * p1.x + plane.b * p1.y + plane.c * p1.z + plane.d) *
  (plane.a * p2.x + plane.b * p2.y + plane.c * p2.z + plane.d) < 0

theorem points_in_different_half_spaces :
  let plane := Plane.mk 1 2 3 0
  let point1 := Point3D.mk 1 2 (-2)
  let point2 := Point3D.mk 2 1 (-1)
  oppositeHalfSpaces plane point1 point2 := by
  sorry


end points_in_different_half_spaces_l2448_244803


namespace opposite_signs_sum_and_max_difference_l2448_244815

theorem opposite_signs_sum_and_max_difference (m n : ℤ) : 
  (|m| = 1 ∧ |n| = 4) → 
  ((m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = -3 ∨ m + n = 3)) ∧
  (∀ (a b : ℤ), |a| = 1 ∧ |b| = 4 → m - n ≥ a - b) :=
by sorry

end opposite_signs_sum_and_max_difference_l2448_244815


namespace simplify_expression_l2448_244811

theorem simplify_expression (a c d x y z : ℝ) (h : cx + dz ≠ 0) :
  (c*x*(a^3*x^3 + 3*a^3*y^3 + c^3*z^3) + d*z*(a^3*x^3 + 3*c^3*x^3 + c^3*z^3)) / (c*x + d*z) =
  a^3*x^3 + c^3*z^3 + (3*c*x*a^3*y^3)/(c*x + d*z) + (3*d*z*c^3*x^3)/(c*x + d*z) := by
sorry

end simplify_expression_l2448_244811


namespace min_value_reciprocal_sum_l2448_244849

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1) :
  1 / a + 1 / b ≥ 9 := by
  sorry

end min_value_reciprocal_sum_l2448_244849


namespace sqrt_49_is_7_l2448_244863

theorem sqrt_49_is_7 : Real.sqrt 49 = 7 := by
  sorry

end sqrt_49_is_7_l2448_244863


namespace quadratic_function_value_l2448_244856

theorem quadratic_function_value (a b x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∃ y₁ y₂ : ℝ, y₁ = a * x₁^2 + b * x₁ + 2009 ∧ 
                y₂ = a * x₂^2 + b * x₂ + 2009 ∧ 
                y₁ = 2012 ∧ 
                y₂ = 2012) → 
  a * (x₁ + x₂)^2 + b * (x₁ + x₂) + 2009 = 2009 :=
by sorry

end quadratic_function_value_l2448_244856


namespace right_triangle_area_l2448_244848

theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) :
  h = 8 * Real.sqrt 2 →
  α = 45 * π / 180 →
  A = (h^2 / 4) →
  A = 32 :=
by
  sorry

#check right_triangle_area

end right_triangle_area_l2448_244848


namespace f_value_theorem_l2448_244877

def is_prime (p : ℕ) : Prop := ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def f_property (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem f_value_theorem (f : ℕ → ℝ) (h1 : f_property f) 
  (h2 : f (2^2007) + f (3^2008) + f (5^2009) = 2006) :
  f (2007^2) + f (2008^3) + f (2009^5) = 9 := by
  sorry

end f_value_theorem_l2448_244877


namespace garden_length_l2448_244857

/-- Proves that a rectangular garden with perimeter 80 meters and width 15 meters has a length of 25 meters -/
theorem garden_length (perimeter width : ℝ) (h1 : perimeter = 80) (h2 : width = 15) :
  let length := (perimeter / 2) - width
  length = 25 := by
sorry

end garden_length_l2448_244857


namespace trader_profit_double_price_l2448_244860

theorem trader_profit_double_price (cost : ℝ) (initial_profit_percent : ℝ) 
  (h1 : initial_profit_percent = 40) : 
  let initial_price := cost * (1 + initial_profit_percent / 100)
  let new_price := 2 * initial_price
  let new_profit := new_price - cost
  new_profit / cost * 100 = 180 := by
sorry

end trader_profit_double_price_l2448_244860


namespace tricia_age_is_five_l2448_244851

-- Define the ages as natural numbers
def tricia_age : ℕ := 5
def amilia_age : ℕ := 3 * tricia_age
def yorick_age : ℕ := 4 * amilia_age
def eugene_age : ℕ := yorick_age / 2
def khloe_age : ℕ := eugene_age / 3
def rupert_age : ℕ := khloe_age + 10
def vincent_age : ℕ := 22

-- State the theorem
theorem tricia_age_is_five :
  tricia_age = 5 ∧
  amilia_age = 3 * tricia_age ∧
  yorick_age = 4 * amilia_age ∧
  yorick_age = 2 * eugene_age ∧
  khloe_age = eugene_age / 3 ∧
  rupert_age = khloe_age + 10 ∧
  vincent_age = 22 ∧
  rupert_age < vincent_age →
  tricia_age = 5 := by
sorry

end tricia_age_is_five_l2448_244851


namespace problem_statement_l2448_244867

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → a^2005 + b^2005 = -1 := by
sorry

end problem_statement_l2448_244867


namespace series_sum_equals_35_over_13_l2448_244873

/-- Definition of the sequence G_n -/
def G : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => G (n + 1) + 2 * G n

/-- The sum of the series -/
noncomputable def seriesSum : ℚ := ∑' n, G n / 5^n

/-- Theorem stating that the sum of the series equals 35/13 -/
theorem series_sum_equals_35_over_13 : seriesSum = 35/13 := by
  sorry

end series_sum_equals_35_over_13_l2448_244873


namespace belts_count_l2448_244869

/-- The number of ties in the store -/
def ties : ℕ := 34

/-- The number of black shirts in the store -/
def black_shirts : ℕ := 63

/-- The number of white shirts in the store -/
def white_shirts : ℕ := 42

/-- The number of jeans in the store -/
def jeans : ℕ := (2 * (black_shirts + white_shirts)) / 3

/-- The number of scarves in the store -/
def scarves (belts : ℕ) : ℕ := (ties + belts) / 2

/-- The relationship between jeans and scarves -/
def jeans_scarves_relation (belts : ℕ) : Prop :=
  jeans = scarves belts + 33

theorem belts_count : ∃ (belts : ℕ), jeans_scarves_relation belts ∧ belts = 40 :=
sorry

end belts_count_l2448_244869


namespace geometric_sequence_third_term_l2448_244822

/-- A geometric sequence with given first and fourth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1)
  (h_fourth : a 4 = 27) :
  a 3 = 9 := by
sorry

end geometric_sequence_third_term_l2448_244822


namespace product_units_digit_base_6_l2448_244829

-- Define the base-10 numbers
def a : ℕ := 217
def b : ℕ := 45

-- Define the base of the target representation
def base : ℕ := 6

-- Theorem statement
theorem product_units_digit_base_6 :
  (a * b) % base = 3 := by
  sorry

end product_units_digit_base_6_l2448_244829


namespace complex_equation_solution_l2448_244861

/-- Given a complex number in the form (2-mi)/(1+2i) = A+Bi, where m, A, and B are real numbers,
    if A + B = 0, then m = 2 -/
theorem complex_equation_solution (m A B : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - m * Complex.I) / (1 + 2 * Complex.I) = A + B * Complex.I →
  A + B = 0 →
  m = 2 := by
  sorry

end complex_equation_solution_l2448_244861


namespace production_decrease_l2448_244820

theorem production_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - x / 100) = 0.49 → x = 30 := by
  sorry

end production_decrease_l2448_244820


namespace units_digit_of_7_power_2023_l2448_244862

theorem units_digit_of_7_power_2023 : (7^2023 : ℕ) % 10 = 3 := by
  sorry

end units_digit_of_7_power_2023_l2448_244862


namespace probability_all_white_balls_l2448_244899

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 6435 :=
sorry

end probability_all_white_balls_l2448_244899


namespace right_triangle_hypotenuse_l2448_244805

/-- A right triangle with perimeter 40 and area 30 has a hypotenuse of length 18.5 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a + b + c = 40 →   -- Perimeter condition
  a * b / 2 = 30 →   -- Area condition
  c = 18.5 := by
    sorry

end right_triangle_hypotenuse_l2448_244805


namespace parabola_hyperbola_focus_l2448_244881

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/6 - y^2/3 = 1

-- Define the right focus of the hyperbola
def right_focus_hyperbola (x y : ℝ) : Prop := x = 3 ∧ y = 0

-- Theorem statement
theorem parabola_hyperbola_focus (p : ℝ) : 
  (∃ x y : ℝ, parabola p x y ∧ right_focus_hyperbola x y) → p = 6 := by
  sorry

end parabola_hyperbola_focus_l2448_244881


namespace unique_solution_l2448_244843

/-- A function from positive reals to positive reals -/
def PositiveFunction := {f : ℝ → ℝ // ∀ x, x > 0 → f x > 0}

/-- The functional equation -/
def SatisfiesEquation (f : PositiveFunction) (c : ℝ) : Prop :=
  c > 0 ∧ ∀ x y, x > 0 → y > 0 → f.val ((c + 1) * x + f.val y) = f.val (x + 2 * y) + 2 * c * x

/-- The theorem statement -/
theorem unique_solution (f : PositiveFunction) (c : ℝ) 
  (h : SatisfiesEquation f c) : 
  ∀ x, x > 0 → f.val x = 2 * x :=
by sorry

end unique_solution_l2448_244843


namespace geometric_progression_first_term_l2448_244879

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 8) 
  (h2 : sum_first_two = 5) :
  ∃ a : ℝ, (a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6)) ∧ 
    (∃ r : ℝ, a / (1 - r) = S ∧ a + a * r = sum_first_two) :=
by sorry

end geometric_progression_first_term_l2448_244879


namespace disinfectant_sales_analysis_l2448_244874

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

/-- Represents the daily profit as a function of selling price -/
def profit (x : ℤ) : ℤ := (x - 8) * sales_quantity x

theorem disinfectant_sales_analysis 
  (h1 : ∀ x : ℤ, 8 ≤ x → x ≤ 15 → sales_quantity x = -5 * x + 150)
  (h2 : sales_quantity 9 = 105)
  (h3 : sales_quantity 11 = 95)
  (h4 : sales_quantity 13 = 85) :
  (∀ x : ℤ, 8 ≤ x → x ≤ 15 → sales_quantity x = -5 * x + 150) ∧
  (profit 13 = 425) ∧
  (∀ x : ℤ, 8 ≤ x → x ≤ 15 → profit x ≤ 525) ∧
  (profit 15 = 525) := by
sorry


end disinfectant_sales_analysis_l2448_244874


namespace rugby_team_size_l2448_244859

theorem rugby_team_size (initial_avg : ℝ) (new_player_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 180 →
  new_player_weight = 210 →
  new_avg = 181.42857142857142 →
  ∃ n : ℕ, (n : ℝ) * initial_avg + new_player_weight = (n + 1 : ℝ) * new_avg ∧ n = 20 :=
by sorry

end rugby_team_size_l2448_244859


namespace find_value_of_A_l2448_244895

theorem find_value_of_A : ∃ A : ℚ, 
  (∃ B : ℚ, B - A = 0.99 ∧ B = 10 * A) → A = 0.11 := by
  sorry

end find_value_of_A_l2448_244895


namespace chess_tournament_games_l2448_244833

/-- Calculate the number of games in a chess tournament -/
theorem chess_tournament_games (n : ℕ) (h : n = 20) : n * (n - 1) = 760 := by
  sorry

#check chess_tournament_games

end chess_tournament_games_l2448_244833


namespace find_a_plus_c_l2448_244813

theorem find_a_plus_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 6)
  (h3 : b * d = 5) :
  a + c = 7 := by
sorry

end find_a_plus_c_l2448_244813


namespace inverse_function_inequality_l2448_244808

open Set
open Function
open Real

noncomputable def f (x : ℝ) := -x * abs x

theorem inverse_function_inequality (h : Bijective f) 
  (h2 : ∀ x ∈ Icc (-2 : ℝ) 2, (invFun f) (x^2 + m) < f x) : 
  m > 12 := by sorry

end inverse_function_inequality_l2448_244808


namespace erased_odd_number_l2448_244842

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := n^2

/-- The sequence of odd numbers -/
def odd_sequence (n : ℕ) : ℕ := 2*n - 1

theorem erased_odd_number :
  ∃ (n : ℕ) (k : ℕ), k < n ∧ sum_odd_numbers n - odd_sequence k = 1998 →
  odd_sequence k = 27 :=
sorry

end erased_odd_number_l2448_244842


namespace custom_op_three_four_l2448_244865

/-- Custom binary operation * -/
def custom_op (a b : ℝ) : ℝ := 4*a + 5*b - a^2*b

/-- Theorem stating that 3 * 4 = -4 under the custom operation -/
theorem custom_op_three_four : custom_op 3 4 = -4 := by
  sorry

end custom_op_three_four_l2448_244865


namespace unique_magnitude_of_quadratic_root_l2448_244844

theorem unique_magnitude_of_quadratic_root : ∃! m : ℝ, ∃ z : ℂ, z^2 - 6*z + 25 = 0 ∧ Complex.abs z = m := by
  sorry

end unique_magnitude_of_quadratic_root_l2448_244844


namespace arithmetic_calculation_l2448_244878

theorem arithmetic_calculation : 8 / 2 + (-3) * 4 - (-10) + 6 * (-2) = -10 := by
  sorry

end arithmetic_calculation_l2448_244878


namespace twentieth_term_of_sequence_l2448_244897

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_sequence :
  let a₁ := 2
  let a₂ := 7
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 20 = 97 := by sorry

end twentieth_term_of_sequence_l2448_244897


namespace floor_ceiling_sum_seven_l2448_244871

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + ⌈x⌉ = 7 ↔ 3 < x ∧ x < 4 :=
sorry

end floor_ceiling_sum_seven_l2448_244871


namespace trajectory_of_midpoint_l2448_244828

/-- The trajectory of point M given point P on a curve and M as the midpoint of OP -/
theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) : 
  (2 * x^2 - y^2 = 1) →  -- P is on the curve
  (x₀ = x / 2) →         -- M is the midpoint of OP (x-coordinate)
  (y₀ = y / 2) →         -- M is the midpoint of OP (y-coordinate)
  (8 * x₀^2 - 4 * y₀^2 = 1) := by
sorry

end trajectory_of_midpoint_l2448_244828
