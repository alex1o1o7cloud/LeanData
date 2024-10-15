import Mathlib

namespace NUMINAMATH_CALUDE_ivan_revival_time_l3726_372615

/-- Represents the scenario of Wolf, Ivan, and Raven --/
structure Scenario where
  distance : ℝ
  wolf_speed : ℝ
  water_needed : ℝ
  water_flow_rate : ℝ
  raven_speed : ℝ
  water_spill_rate : ℝ

/-- Checks if Ivan can be revived within the given time --/
def can_revive (s : Scenario) (time : ℝ) : Prop :=
  let wolf_travel_time := s.distance / s.wolf_speed
  let water_collect_time := s.water_needed / s.water_flow_rate
  let total_time := wolf_travel_time + water_collect_time
  let raven_travel_distance := s.distance / 2
  let raven_travel_time := raven_travel_distance / s.raven_speed
  let water_lost := raven_travel_time * s.water_spill_rate
  let water_remaining := s.water_needed - water_lost
  
  time ≥ total_time ∧ water_remaining > 0

/-- The main theorem to prove --/
theorem ivan_revival_time (s : Scenario) :
  s.distance = 20 ∧
  s.wolf_speed = 3 ∧
  s.water_needed = 1 ∧
  s.water_flow_rate = 0.5 ∧
  s.raven_speed = 6 ∧
  s.water_spill_rate = 0.25 →
  can_revive s 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ivan_revival_time_l3726_372615


namespace NUMINAMATH_CALUDE_max_price_changes_l3726_372616

/-- Represents the price of the souvenir after n changes -/
def price (initial : ℕ) (x : ℚ) (n : ℕ) : ℚ :=
  initial * ((1 - x/100)^n * (1 + x/100)^n)

/-- The problem statement -/
theorem max_price_changes (initial : ℕ) (x : ℚ) : 
  initial = 10000 →
  0 < x →
  x < 100 →
  (∃ n : ℕ, ¬(price initial x n).isInt ∧ (price initial x (n-1)).isInt) →
  (∃ max_changes : ℕ, 
    (∀ n : ℕ, n ≤ max_changes → (price initial x n).isInt) ∧
    ¬(price initial x (max_changes + 1)).isInt ∧
    max_changes = 5) :=
sorry

end NUMINAMATH_CALUDE_max_price_changes_l3726_372616


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l3726_372679

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l3726_372679


namespace NUMINAMATH_CALUDE_number_divided_by_five_l3726_372691

theorem number_divided_by_five (x : ℝ) : x - 5 = 35 → x / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_five_l3726_372691


namespace NUMINAMATH_CALUDE_cabbages_on_plot_l3726_372686

/-- Calculates the total number of cabbages that can be planted on a rectangular plot. -/
def total_cabbages (length width density : ℕ) : ℕ :=
  length * width * density

/-- Theorem stating the total number of cabbages on the given plot. -/
theorem cabbages_on_plot :
  total_cabbages 16 12 9 = 1728 := by
  sorry

#eval total_cabbages 16 12 9

end NUMINAMATH_CALUDE_cabbages_on_plot_l3726_372686


namespace NUMINAMATH_CALUDE_contradiction_proof_l3726_372675

theorem contradiction_proof (x : ℝ) : (x^2 - 1 = 0) → (x = -1 ∨ x = 1) := by
  contrapose
  intro h
  have h1 : x ≠ -1 ∧ x ≠ 1 := by
    push_neg at h
    exact h
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l3726_372675


namespace NUMINAMATH_CALUDE_balloon_difference_is_two_l3726_372638

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 3

/-- The difference in the number of balloons between Allan and Jake -/
def balloon_difference : ℕ := allan_balloons - jake_balloons

theorem balloon_difference_is_two : balloon_difference = 2 := by sorry

end NUMINAMATH_CALUDE_balloon_difference_is_two_l3726_372638


namespace NUMINAMATH_CALUDE_current_speed_l3726_372635

/-- The speed of the current given boat speeds upstream and downstream -/
theorem current_speed (upstream_speed downstream_speed : ℝ) : 
  upstream_speed = 1 / (20 / 60) →
  downstream_speed = 1 / (15 / 60) →
  (downstream_speed - upstream_speed) / 2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l3726_372635


namespace NUMINAMATH_CALUDE_max_students_on_field_trip_l3726_372640

def budget : ℕ := 350
def bus_rental : ℕ := 100
def admission_cost : ℕ := 10

theorem max_students_on_field_trip : 
  (budget - bus_rental) / admission_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_students_on_field_trip_l3726_372640


namespace NUMINAMATH_CALUDE_bearded_male_percentage_is_40_percent_l3726_372682

/-- Represents the data for Scrabble champions over a period of years -/
structure ScrabbleChampionData where
  total_years : ℕ
  women_percentage : ℚ
  champions_per_year : ℕ
  bearded_men : ℕ

/-- Calculates the percentage of male Scrabble champions with beards -/
def bearded_male_percentage (data : ScrabbleChampionData) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, 
    the percentage of male Scrabble champions with beards is 40% -/
theorem bearded_male_percentage_is_40_percent 
  (data : ScrabbleChampionData)
  (h1 : data.total_years = 25)
  (h2 : data.women_percentage = 60 / 100)
  (h3 : data.champions_per_year = 1)
  (h4 : data.bearded_men = 4) :
  bearded_male_percentage data = 40 / 100 :=
sorry

end NUMINAMATH_CALUDE_bearded_male_percentage_is_40_percent_l3726_372682


namespace NUMINAMATH_CALUDE_light_travel_distance_l3726_372608

/-- The distance light travels in one year, in miles -/
def light_year_distance : ℕ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℕ := 200

/-- Theorem stating that light travels 1174 × 10^12 miles in 200 years -/
theorem light_travel_distance : 
  (light_year_distance * years : ℚ) = 1174 * (10^12 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l3726_372608


namespace NUMINAMATH_CALUDE_expression_values_l3726_372658

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d)
  expr = 5 ∨ expr = 1 ∨ expr = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3726_372658


namespace NUMINAMATH_CALUDE_millet_in_brand_b_l3726_372681

/-- Represents the composition of a bird seed brand -/
structure BirdSeed where
  millet : ℝ
  other : ℝ
  composition_sum : millet + other = 1

/-- Represents a mix of two bird seed brands -/
structure BirdSeedMix where
  brandA : BirdSeed
  brandB : BirdSeed
  proportionA : ℝ
  proportionB : ℝ
  mix_sum : proportionA + proportionB = 1

/-- Theorem stating the millet percentage in Brand B given the conditions -/
theorem millet_in_brand_b 
  (mix : BirdSeedMix)
  (brandA_millet : mix.brandA.millet = 0.6)
  (mix_proportionA : mix.proportionA = 0.6)
  (mix_millet : mix.proportionA * mix.brandA.millet + mix.proportionB * mix.brandB.millet = 0.5) :
  mix.brandB.millet = 0.35 := by
  sorry


end NUMINAMATH_CALUDE_millet_in_brand_b_l3726_372681


namespace NUMINAMATH_CALUDE_prob_at_least_one_black_is_five_sixths_l3726_372678

/-- The number of white balls in the pouch -/
def num_white_balls : ℕ := 2

/-- The number of black balls in the pouch -/
def num_black_balls : ℕ := 2

/-- The total number of balls in the pouch -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- The number of balls drawn from the pouch -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 5/6

theorem prob_at_least_one_black_is_five_sixths :
  prob_at_least_one_black = 1 - (num_white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_black_is_five_sixths_l3726_372678


namespace NUMINAMATH_CALUDE_attic_boxes_count_l3726_372605

/-- Represents the problem of arranging teacups in an attic --/
def TeacupArrangement (B : ℕ) : Prop :=
  let boxes_without_pans := B - 6
  let boxes_with_teacups := boxes_without_pans / 2
  let cups_per_box := 5 * 4
  let broken_cups := 2 * boxes_with_teacups
  let original_cups := cups_per_box * boxes_with_teacups
  original_cups = 180 + broken_cups

/-- Theorem stating that there are 26 boxes in the attic --/
theorem attic_boxes_count : ∃ B : ℕ, TeacupArrangement B ∧ B = 26 := by
  sorry

end NUMINAMATH_CALUDE_attic_boxes_count_l3726_372605


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l3726_372661

/-- The rate of drawing barbed wire per meter given a square field's area, gate widths, and total cost --/
theorem barbed_wire_rate (field_area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : 
  field_area = 3136 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 2331 →
  (total_cost / (4 * Real.sqrt field_area - num_gates * gate_width)) = 10.5 := by
  sorry

#check barbed_wire_rate

end NUMINAMATH_CALUDE_barbed_wire_rate_l3726_372661


namespace NUMINAMATH_CALUDE_aquarium_visitors_l3726_372644

theorem aquarium_visitors (total : ℕ) (ill_percent : ℚ) (not_ill : ℕ) 
  (h1 : ill_percent = 40 / 100)
  (h2 : not_ill = 300)
  (h3 : (1 - ill_percent) * total = not_ill) : 
  total = 500 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l3726_372644


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3726_372627

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3726_372627


namespace NUMINAMATH_CALUDE_marks_deposit_is_88_l3726_372677

-- Define Mark's deposit
def mark_deposit : ℕ := 88

-- Define Bryan's deposit in terms of Mark's
def bryan_deposit : ℕ := 5 * mark_deposit - 40

-- Theorem to prove
theorem marks_deposit_is_88 : mark_deposit = 88 := by
  sorry

end NUMINAMATH_CALUDE_marks_deposit_is_88_l3726_372677


namespace NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l3726_372603

def are_in_ratio (a b c : ℕ) (x y z : ℕ) : Prop :=
  ∃ (k : ℕ), a = k * x ∧ b = k * y ∧ c = k * z

theorem lcm_of_numbers_in_ratio (a b c : ℕ) 
  (h_ratio : are_in_ratio a b c 5 7 9)
  (h_hcf : Nat.gcd a (Nat.gcd b c) = 11) :
  Nat.lcm a (Nat.lcm b c) = 99 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l3726_372603


namespace NUMINAMATH_CALUDE_last_remaining_number_l3726_372600

/-- Represents the marking process on a list of numbers -/
def MarkingProcess (n : ℕ) (skip : ℕ) (l : List ℕ) : List ℕ :=
  sorry

/-- Represents a single pass of the marking process -/
def SinglePass (n : ℕ) (skip : ℕ) (l : List ℕ) : List ℕ :=
  sorry

/-- Represents the entire process of marking and skipping -/
def FullProcess (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the last remaining number is 21 -/
theorem last_remaining_number : FullProcess 50 = 21 :=
  sorry

end NUMINAMATH_CALUDE_last_remaining_number_l3726_372600


namespace NUMINAMATH_CALUDE_class_average_weight_l3726_372613

theorem class_average_weight (num_boys : ℕ) (num_girls : ℕ) 
  (avg_weight_boys : ℝ) (avg_weight_girls : ℝ) :
  num_boys = 5 →
  num_girls = 3 →
  avg_weight_boys = 60 →
  avg_weight_girls = 50 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l3726_372613


namespace NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l3726_372628

/-- A parallelogram with vertices at (8,35), (8,90), (25,125), and (25,70) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (8, 35)
  v2 : ℝ × ℝ := (8, 90)
  v3 : ℝ × ℝ := (25, 125)
  v4 : ℝ × ℝ := (25, 70)

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- The line cuts the parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop := sorry

theorem parallelogram_bisecting_line_slope (p : Parallelogram) (l : Line) :
  cuts_into_congruent_polygons p l → l.slope = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l3726_372628


namespace NUMINAMATH_CALUDE_pole_area_after_cuts_l3726_372642

/-- The area of a rectangular pole after two cuts -/
theorem pole_area_after_cuts (original_length original_width : ℝ)
  (length_cut_percentage width_cut_percentage : ℝ) :
  original_length = 20 →
  original_width = 2 →
  length_cut_percentage = 0.3 →
  width_cut_percentage = 0.25 →
  let new_length := original_length * (1 - length_cut_percentage)
  let new_width := original_width * (1 - width_cut_percentage)
  new_length * new_width = 21 := by
  sorry

end NUMINAMATH_CALUDE_pole_area_after_cuts_l3726_372642


namespace NUMINAMATH_CALUDE_exponentiation_puzzle_l3726_372636

theorem exponentiation_puzzle : 3^(1^(0^2)) - ((3^1)^0)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponentiation_puzzle_l3726_372636


namespace NUMINAMATH_CALUDE_lincoln_high_school_groups_l3726_372653

/-- Represents the number of students in various groups at Lincoln High School -/
structure SchoolGroups where
  total : ℕ
  band : ℕ
  chorus : ℕ
  drama : ℕ
  band_chorus_drama : ℕ

/-- Calculates the number of students in both band and chorus but not in drama -/
def students_in_band_and_chorus_not_drama (g : SchoolGroups) : ℕ :=
  g.band + g.chorus - (g.band_chorus_drama - g.drama)

/-- Theorem stating the number of students in both band and chorus but not in drama -/
theorem lincoln_high_school_groups (g : SchoolGroups) 
  (h1 : g.total = 300)
  (h2 : g.band = 80)
  (h3 : g.chorus = 120)
  (h4 : g.drama = 50)
  (h5 : g.band_chorus_drama = 200) :
  students_in_band_and_chorus_not_drama g = 50 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_high_school_groups_l3726_372653


namespace NUMINAMATH_CALUDE_becky_eddie_age_ratio_l3726_372698

/-- Given the ages of Eddie, Irene, and the relationship between Irene and Becky's ages,
    prove that the ratio of Becky's age to Eddie's age is 1:4. -/
theorem becky_eddie_age_ratio 
  (eddie_age : ℕ) 
  (irene_age : ℕ) 
  (becky_age : ℕ) 
  (h1 : eddie_age = 92) 
  (h2 : irene_age = 46) 
  (h3 : irene_age = 2 * becky_age) : 
  becky_age * 4 = eddie_age := by
  sorry

#check becky_eddie_age_ratio

end NUMINAMATH_CALUDE_becky_eddie_age_ratio_l3726_372698


namespace NUMINAMATH_CALUDE_mary_berry_spending_l3726_372643

theorem mary_berry_spending (total apples peaches : ℝ) (h1 : total = 34.72) (h2 : apples = 14.33) (h3 : peaches = 9.31) :
  total - (apples + peaches) = 11.08 := by
  sorry

end NUMINAMATH_CALUDE_mary_berry_spending_l3726_372643


namespace NUMINAMATH_CALUDE_multiplication_problem_solution_l3726_372647

theorem multiplication_problem_solution :
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    1000 ≤ a * b ∧ a * b < 10000 ∧
    10 ≤ a * 8 ∧ a * 8 < 100 ∧
    100 ≤ a * 9 ∧ a * 9 < 1000 ∧
    a * b = 1068 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_solution_l3726_372647


namespace NUMINAMATH_CALUDE_ballot_marking_combinations_l3726_372655

theorem ballot_marking_combinations : 
  ∀ n : ℕ, n = 10 → n.factorial = 3628800 :=
by
  sorry

end NUMINAMATH_CALUDE_ballot_marking_combinations_l3726_372655


namespace NUMINAMATH_CALUDE_trisector_inequality_l3726_372645

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Define trisectors
def trisectors (t : AcuteTriangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem trisector_inequality (t : AcuteTriangle) : 
  let (f, g) := trisectors t
  (f + g) / 2 < 2 / (1 / t.a + 1 / t.b) := by sorry

end NUMINAMATH_CALUDE_trisector_inequality_l3726_372645


namespace NUMINAMATH_CALUDE_max_table_height_l3726_372659

/-- Given a triangle DEF with side lengths 26, 28, and 34, prove that the maximum height k
    of a table formed by making right angle folds parallel to each side is 96√55/54. -/
theorem max_table_height (DE EF FD : ℝ) (h_DE : DE = 26) (h_EF : EF = 28) (h_FD : FD = 34) :
  let s := (DE + EF + FD) / 2
  let A := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_e := 2 * A / EF
  let h_f := 2 * A / FD
  let k := h_e * h_f / (h_e + h_f)
  k = 96 * Real.sqrt 55 / 54 :=
by sorry

end NUMINAMATH_CALUDE_max_table_height_l3726_372659


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3726_372637

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_properties 
  (a : ℕ → ℚ) 
  (h_geometric : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3726_372637


namespace NUMINAMATH_CALUDE_divisible_by_25_l3726_372662

theorem divisible_by_25 (n : ℕ) : ∃ k : ℤ, (2^(n+2) * 3^n + 5*n - 4 : ℤ) = 25 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_25_l3726_372662


namespace NUMINAMATH_CALUDE_geometric_sequence_example_l3726_372672

def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_example :
  is_geometric_sequence 3 (-3 * Real.sqrt 3) 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_example_l3726_372672


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l3726_372696

/-- Represents the number of shirts Kendra needs for various activities --/
structure ShirtRequirements where
  weekdaySchool : Nat
  afterSchoolClub : Nat
  spiritDay : Nat
  saturday : Nat
  sunday : Nat
  familyReunion : Nat

/-- Calculates the total number of shirts needed for a given number of weeks --/
def totalShirtsNeeded (req : ShirtRequirements) (weeks : Nat) : Nat :=
  (req.weekdaySchool + req.afterSchoolClub + req.spiritDay + req.saturday + req.sunday) * weeks + req.familyReunion

/-- Theorem stating that Kendra needs 61 shirts for 4 weeks --/
theorem kendra_shirts_theorem (req : ShirtRequirements) 
    (h1 : req.weekdaySchool = 5)
    (h2 : req.afterSchoolClub = 3)
    (h3 : req.spiritDay = 1)
    (h4 : req.saturday = 3)
    (h5 : req.sunday = 3)
    (h6 : req.familyReunion = 1) :
  totalShirtsNeeded req 4 = 61 := by
  sorry

#eval totalShirtsNeeded ⟨5, 3, 1, 3, 3, 1⟩ 4

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l3726_372696


namespace NUMINAMATH_CALUDE_alteredLucas_53_mod_5_l3726_372619

def alteredLucas : ℕ → ℕ
  | 0 => 1
  | 1 => 4
  | n + 2 => alteredLucas n + alteredLucas (n + 1)

theorem alteredLucas_53_mod_5 : alteredLucas 52 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_alteredLucas_53_mod_5_l3726_372619


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3726_372604

/-- The sum of two polynomials is equal to the simplified polynomial. -/
theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9) =
  2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3726_372604


namespace NUMINAMATH_CALUDE_inequality_proof_l3726_372690

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : d > 0) :
  d / c < (d + 4) / (c + 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3726_372690


namespace NUMINAMATH_CALUDE_f_properties_l3726_372620

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^3

theorem f_properties :
  (∃! (a b : ℝ), a ≠ b ∧ (deriv f a = 0 ∧ deriv f b = 0) ∧
    ∀ x, deriv f x = 0 → (x = a ∨ x = b)) ∧
  (∃! (a b : ℝ), deriv f a = 0 ∧ deriv f b = 0 ∧ a + b = 0) ∧
  (∃! x, f x = 0) ∧
  (¬∃ x, f x = -x ∧ deriv f x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3726_372620


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_l3726_372641

def is_prime (n : ℕ) : Prop := sorry

def is_square (n : ℕ) : Prop := sorry

def has_prime_factor_less_than (n k : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square : 
  ∀ n : ℕ, n < 3599 → 
    (is_prime n ∨ is_square n ∨ has_prime_factor_less_than n 55) ∧
    (¬ is_prime 3599 ∧ ¬ is_square 3599 ∧ ¬ has_prime_factor_less_than 3599 55) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_l3726_372641


namespace NUMINAMATH_CALUDE_hyperbola_circumradius_l3726_372667

/-- The hyperbola in the xy-plane -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The centroid of triangle F₁PF₂ -/
def G : ℝ × ℝ := sorry

/-- The incenter of triangle F₁PF₂ -/
def I : ℝ × ℝ := sorry

/-- The circumradius of triangle F₁PF₂ -/
def R : ℝ := sorry

theorem hyperbola_circumradius :
  hyperbola P.1 P.2 ∧ 
  (G.2 = I.2) →  -- GI is parallel to x-axis
  R = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_circumradius_l3726_372667


namespace NUMINAMATH_CALUDE_p_iff_q_l3726_372666

-- Define the propositions
def p (a : ℝ) : Prop := a = -1

def q (a : ℝ) : Prop := ∀ (x y : ℝ), (a * x + y + 1 = 0) ↔ (x + a * y + 2 * a - 1 = 0)

-- State the theorem
theorem p_iff_q : ∀ (a : ℝ), p a ↔ q a := by sorry

end NUMINAMATH_CALUDE_p_iff_q_l3726_372666


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l3726_372618

/-- The amount of oil leaked before engineers started fixing the pipe -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked while engineers were working -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem oil_leak_calculation :
  total_oil_leaked = 11687 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l3726_372618


namespace NUMINAMATH_CALUDE_power_multiplication_l3726_372656

theorem power_multiplication (x : ℝ) : x^5 * x^6 = x^11 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3726_372656


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3726_372632

/-- The probability of a coin with diameter 1/2 not touching any lattice lines when tossed onto a 1x1 square -/
def coin_probability : ℚ := 1 / 4

/-- The diameter of the coin -/
def coin_diameter : ℚ := 1 / 2

/-- The side length of the square -/
def square_side : ℚ := 1

theorem coin_toss_probability :
  coin_probability = (square_side - coin_diameter)^2 / square_side^2 :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3726_372632


namespace NUMINAMATH_CALUDE_math_club_composition_l3726_372652

theorem math_club_composition :
  ∀ (initial_males initial_females : ℕ),
    initial_males = initial_females →
    (3 * (initial_males + initial_females - 1) = 4 * (initial_females - 1)) →
    initial_males = 2 ∧ initial_females = 3 := by
  sorry

end NUMINAMATH_CALUDE_math_club_composition_l3726_372652


namespace NUMINAMATH_CALUDE_two_candles_burning_time_l3726_372622

/-- Proves that the time during which exactly two candles are burning simultaneously is 35 minutes -/
theorem two_candles_burning_time (t₁ t₂ t₃ : ℕ) 
  (h₁ : t₁ = 30) 
  (h₂ : t₂ = 40) 
  (h₃ : t₃ = 50) 
  (h_three : ℕ) 
  (h_three_eq : h_three = 10) 
  (h_one : ℕ) 
  (h_one_eq : h_one = 20) 
  (h_two : ℕ) 
  (h_total : h_one + 2 * h_two + 3 * h_three = t₁ + t₂ + t₃) : 
  h_two = 35 := by
  sorry

end NUMINAMATH_CALUDE_two_candles_burning_time_l3726_372622


namespace NUMINAMATH_CALUDE_democrat_ratio_is_one_third_l3726_372639

/-- Represents the number of participants in each category -/
structure Participants where
  total : ℕ
  female : ℕ
  male : ℕ
  femaleDemocrats : ℕ
  maleDemocrats : ℕ

/-- The ratio of democrats to total participants -/
def democratRatio (p : Participants) : ℚ :=
  (p.femaleDemocrats + p.maleDemocrats : ℚ) / p.total

theorem democrat_ratio_is_one_third (p : Participants) 
  (h1 : p.total = 660)
  (h2 : p.female + p.male = p.total)
  (h3 : p.femaleDemocrats = p.female / 2)
  (h4 : p.maleDemocrats = p.male / 4)
  (h5 : p.femaleDemocrats = 110) :
  democratRatio p = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_is_one_third_l3726_372639


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3726_372680

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = S seq 5)
  (h2 : seq.a 2 * seq.a 4 = S seq 4) :
  (∀ n, seq.a n = 2 * n - 6) ∧
  (∀ n < 7, S seq n ≤ seq.a n) ∧
  (S seq 7 > seq.a 7) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3726_372680


namespace NUMINAMATH_CALUDE_min_sum_of_product_l3726_372692

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 3960) :
  ∃ (x y z : ℕ+), x * y * z = 3960 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 150 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l3726_372692


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3726_372673

theorem arithmetic_sequence_count (a l d : ℤ) (h1 : a = -58) (h2 : l = 78) (h3 : d = 7) :
  ∃ n : ℕ, n > 0 ∧ l = a + (n - 1) * d ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3726_372673


namespace NUMINAMATH_CALUDE_maintenance_team_journey_l3726_372697

def walking_records : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def fuel_consumption_rate : ℝ := 3
def initial_fuel : ℝ := 180

theorem maintenance_team_journey :
  let net_distance : Int := walking_records.sum
  let total_distance : ℕ := walking_records.map (Int.natAbs) |>.sum
  let total_fuel_consumption : ℝ := (total_distance : ℝ) * fuel_consumption_rate
  let fuel_needed : ℝ := total_fuel_consumption - initial_fuel
  (net_distance = 39) ∧ 
  (total_distance = 65) ∧ 
  (total_fuel_consumption = 195) ∧ 
  (fuel_needed = 15) := by
sorry

end NUMINAMATH_CALUDE_maintenance_team_journey_l3726_372697


namespace NUMINAMATH_CALUDE_linear_function_max_min_sum_l3726_372671

theorem linear_function_max_min_sum (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a * x ≤ max (a * 0) (a * 1) ∧ min (a * 0) (a * 1) ≤ a * x) →
  max (a * 0) (a * 1) + min (a * 0) (a * 1) = 3 →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_max_min_sum_l3726_372671


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_sqrt_three_over_two_l3726_372606

theorem sin_cos_difference_equals_sqrt_three_over_two :
  Real.sin (5 * π / 180) * Real.cos (55 * π / 180) -
  Real.cos (175 * π / 180) * Real.sin (55 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_sqrt_three_over_two_l3726_372606


namespace NUMINAMATH_CALUDE_fraction_equality_l3726_372631

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 2023) :
  (w + z)/(w - z) = -1012 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3726_372631


namespace NUMINAMATH_CALUDE_statue_weight_proof_l3726_372668

def original_weight : ℝ := 190

def week1_reduction : ℝ := 0.25
def week2_reduction : ℝ := 0.15
def week3_reduction : ℝ := 0.10

def final_weight : ℝ := original_weight * (1 - week1_reduction) * (1 - week2_reduction) * (1 - week3_reduction)

theorem statue_weight_proof : final_weight = 108.9125 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_proof_l3726_372668


namespace NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l3726_372646

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2*x - 1 ∧ 2*x - 1 < 19}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

-- Theorem for (CₙA) ∩ B
theorem intersection_complement_A_B : (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l3726_372646


namespace NUMINAMATH_CALUDE_arcade_spending_equals_allowance_l3726_372601

def dress_cost : ℕ := 80
def initial_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weeks_to_save : ℕ := 3

theorem arcade_spending_equals_allowance :
  ∃ (arcade_spending : ℕ),
    arcade_spending = weekly_allowance ∧
    initial_savings + weeks_to_save * weekly_allowance - weeks_to_save * arcade_spending = dress_cost :=
by sorry

end NUMINAMATH_CALUDE_arcade_spending_equals_allowance_l3726_372601


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l3726_372670

/-- The number of chickens and rabbits in the cage satisfying the given conditions -/
theorem chicken_rabbit_problem :
  ∃ (chickens rabbits : ℕ),
    chickens + rabbits = 35 ∧
    2 * chickens + 4 * rabbits = 94 ∧
    chickens = 23 ∧
    rabbits = 12 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l3726_372670


namespace NUMINAMATH_CALUDE_constant_term_implies_n_12_l3726_372663

/-- The general term formula for the expansion of (√x - 2/x)^n -/
def generalTerm (n : ℕ) (r : ℕ) : ℚ → ℚ := 
  λ x => (n.choose r) * (-2)^r * x^((n - 3*r) / 2)

/-- The condition that the 5th term (r = 4) is the constant term -/
def fifthTermIsConstant (n : ℕ) : Prop :=
  (n - 3*4) / 2 = 0

theorem constant_term_implies_n_12 : 
  ∀ n : ℕ, fifthTermIsConstant n → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_implies_n_12_l3726_372663


namespace NUMINAMATH_CALUDE_visitor_increase_l3726_372649

theorem visitor_increase (original_fee : ℝ) (fee_reduction : ℝ) (sale_increase : ℝ) :
  original_fee = 1 →
  fee_reduction = 0.25 →
  sale_increase = 0.20 →
  let new_fee := original_fee * (1 - fee_reduction)
  let visitor_increase := (1 + sale_increase) / (1 - fee_reduction) - 1
  visitor_increase = 0.60 := by sorry

end NUMINAMATH_CALUDE_visitor_increase_l3726_372649


namespace NUMINAMATH_CALUDE_min_side_length_l3726_372609

theorem min_side_length (AB EC AC BE : ℝ) (hAB : AB = 7) (hEC : EC = 10) (hAC : AC = 15) (hBE : BE = 25) :
  ∃ (BC : ℕ), BC ≥ 15 ∧ ∀ (BC' : ℕ), (BC' ≥ 15 → BC' ≥ BC) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l3726_372609


namespace NUMINAMATH_CALUDE_problem_solution_l3726_372629

def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

theorem problem_solution :
  (∀ x : ℝ, x ≤ -1 → f 2 x = 0 ↔ x = (-1 - Real.sqrt 3) / 2) ∧
  (∀ k : ℝ, (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
    -7/2 < k ∧ k < -1) ∧
  (∀ k x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 →
    1/x₁ + 1/x₂ < 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3726_372629


namespace NUMINAMATH_CALUDE_part_one_part_two_l3726_372611

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one : 
  {x : ℝ | |x - 1| ≥ |x + 1| + 1} = {x : ℝ | x ≤ -0.5} := by sorry

-- Part II
theorem part_two :
  {a : ℝ | ∀ x ≤ -1, f a x + 3 * x ≤ 0} = {a : ℝ | -4 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3726_372611


namespace NUMINAMATH_CALUDE_common_factor_proof_l3726_372660

theorem common_factor_proof (x y : ℝ) (m n : ℕ) :
  ∃ (k : ℝ), 8 * x^m * y^(n-1) - 12 * x^(3*m) * y^n = k * (4 * x^m * y^(n-1)) ∧
              k ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3726_372660


namespace NUMINAMATH_CALUDE_unique_number_between_9_and_9_1_cube_root_l3726_372630

theorem unique_number_between_9_and_9_1_cube_root (n : ℕ+) : 
  (∃ k : ℕ, n = 21 * k) ∧ 
  (9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.1) ↔ 
  n = 735 :=
sorry

end NUMINAMATH_CALUDE_unique_number_between_9_and_9_1_cube_root_l3726_372630


namespace NUMINAMATH_CALUDE_assignment_count_assignment_count_proof_l3726_372689

theorem assignment_count : ℕ → Prop :=
  fun total_assignments =>
    ∃ (initial_hours : ℕ),
      -- Initial plan: 6 assignments per hour for initial_hours
      6 * initial_hours = total_assignments ∧
      -- New plan: 2 hours at 6 per hour, then 8 per hour for (initial_hours - 5) hours
      2 * 6 + 8 * (initial_hours - 5) = total_assignments ∧
      -- Total assignments is 84
      total_assignments = 84

-- The proof of this theorem would show that the conditions are satisfied
-- and the total number of assignments is indeed 84
theorem assignment_count_proof : assignment_count 84 := by
  sorry

#check assignment_count_proof

end NUMINAMATH_CALUDE_assignment_count_assignment_count_proof_l3726_372689


namespace NUMINAMATH_CALUDE_tank_capacities_l3726_372625

/-- Given three tanks with capacities T1, T2, and T3, prove that the total amount of water is 10850 gallons. -/
theorem tank_capacities (T1 T2 T3 : ℝ) : 
  (3/4 : ℝ) * T1 + (4/5 : ℝ) * T2 + (1/2 : ℝ) * T3 = 10850 := by
  sorry

#check tank_capacities

end NUMINAMATH_CALUDE_tank_capacities_l3726_372625


namespace NUMINAMATH_CALUDE_tan_pi_twelve_l3726_372648

theorem tan_pi_twelve : Real.tan (π / 12) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_twelve_l3726_372648


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l3726_372684

theorem shopping_tax_calculation (total_amount : ℝ) (total_amount_pos : total_amount > 0) :
  let clothing_percent : ℝ := 0.40
  let food_percent : ℝ := 0.20
  let electronics_percent : ℝ := 0.15
  let other_percent : ℝ := 0.25
  let clothing_tax : ℝ := 0.12
  let food_tax : ℝ := 0
  let electronics_tax : ℝ := 0.05
  let other_tax : ℝ := 0.20
  let total_tax := 
    clothing_percent * total_amount * clothing_tax +
    food_percent * total_amount * food_tax +
    electronics_percent * total_amount * electronics_tax +
    other_percent * total_amount * other_tax
  (total_tax / total_amount) * 100 = 10.55 := by
sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l3726_372684


namespace NUMINAMATH_CALUDE_box_balls_problem_l3726_372688

theorem box_balls_problem (balls : ℕ) (x : ℕ) : 
  balls = 57 → 
  (balls - x = 70 - balls) →
  x = 44 := by
  sorry

end NUMINAMATH_CALUDE_box_balls_problem_l3726_372688


namespace NUMINAMATH_CALUDE_intersection_limit_l3726_372607

noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 8)

theorem intersection_limit :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 
    0 < |m| ∧ |m| < δ ∧ -8 < m ∧ m < 8 → 
    |((L (-m) - L m) / m) - 1 / (2 * Real.sqrt 2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_intersection_limit_l3726_372607


namespace NUMINAMATH_CALUDE_line_segment_polar_equation_l3726_372651

/-- The polar coordinate equation of the line segment y = 1 - x (0 ≤ x ≤ 1) -/
theorem line_segment_polar_equation :
  ∀ (x y ρ θ : ℝ),
  (0 ≤ x) ∧ (x ≤ 1) ∧ (y = 1 - x) →
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (ρ = 1 / (Real.cos θ + Real.sin θ)) ∧ (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_polar_equation_l3726_372651


namespace NUMINAMATH_CALUDE_inequality_proof_l3726_372693

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a * b + a^5 + b^5) + (b * c) / (b * c + b^5 + c^5) + (c * a) / (c * a + c^5 + a^5) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3726_372693


namespace NUMINAMATH_CALUDE_max_value_of_c_l3726_372610

theorem max_value_of_c (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : 2 * a * b = 2 * a + b) (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_c_l3726_372610


namespace NUMINAMATH_CALUDE_remainder_theorem_l3726_372683

theorem remainder_theorem : (9 * 10^20 + 1^20) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3726_372683


namespace NUMINAMATH_CALUDE_square_prime_equivalence_l3726_372614

theorem square_prime_equivalence (N : ℕ) (h : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬∃ s : ℕ, 4*n*(N-n)+1 = s^2) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_square_prime_equivalence_l3726_372614


namespace NUMINAMATH_CALUDE_intersection_values_l3726_372674

-- Define the function f(x) = ax² + (3-a)x + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - a) * x + 1

-- Define the condition for intersection with x-axis at only one point
def intersects_once (a : ℝ) : Prop :=
  ∃! x, f a x = 0

-- State the theorem
theorem intersection_values : 
  ∀ a : ℝ, intersects_once a ↔ a = 0 ∨ a = 1 ∨ a = 9 := by sorry

end NUMINAMATH_CALUDE_intersection_values_l3726_372674


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_137_l3726_372669

theorem first_nonzero_digit_of_1_137 :
  ∃ (n : ℕ) (k : ℕ), 
    10^n > 137 ∧ 
    (1000 : ℚ) / 137 = k + (1000 - k * 137 : ℚ) / 137 ∧ 
    k = 7 := by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_137_l3726_372669


namespace NUMINAMATH_CALUDE_beidou_usage_scientific_notation_l3726_372695

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem beidou_usage_scientific_notation :
  scientific_notation 360000000000 = (3.6, 11) :=
sorry

end NUMINAMATH_CALUDE_beidou_usage_scientific_notation_l3726_372695


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3726_372676

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3726_372676


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3726_372665

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x - 2 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3726_372665


namespace NUMINAMATH_CALUDE_visitor_increase_percentage_l3726_372654

/-- Represents the percentage increase in visitors after implementing discounts -/
def overallPercentageIncrease (initialChildren : ℕ) (initialSeniors : ℕ) (initialAdults : ℕ)
  (childrenIncrease : ℚ) (seniorsIncrease : ℚ) : ℚ :=
  let totalInitial := initialChildren + initialSeniors + initialAdults
  let totalAfter := 
    (initialChildren * (1 + childrenIncrease)) + 
    (initialSeniors * (1 + seniorsIncrease)) + 
    initialAdults
  (totalAfter - totalInitial) / totalInitial * 100

/-- Theorem stating that the overall percentage increase in visitors is approximately 13.33% -/
theorem visitor_increase_percentage : 
  ∀ (initialChildren initialSeniors initialAdults : ℕ),
  initialChildren > 0 → initialSeniors > 0 → initialAdults > 0 →
  let childrenIncrease : ℚ := 25 / 100
  let seniorsIncrease : ℚ := 15 / 100
  abs (overallPercentageIncrease initialChildren initialSeniors initialAdults childrenIncrease seniorsIncrease - 40 / 3) < 1 / 100 :=
by
  sorry

#eval overallPercentageIncrease 100 100 100 (25 / 100) (15 / 100)

end NUMINAMATH_CALUDE_visitor_increase_percentage_l3726_372654


namespace NUMINAMATH_CALUDE_outfit_combinations_l3726_372633

def number_of_shirts : ℕ := 5
def number_of_pants : ℕ := 4
def number_of_hats : ℕ := 2

theorem outfit_combinations : 
  number_of_shirts * number_of_pants * number_of_hats = 40 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3726_372633


namespace NUMINAMATH_CALUDE_parabola_fixed_y_coordinate_l3726_372650

/-- A parabola that intersects the x-axis at only one point and passes through two specific points has a fixed y-coordinate for those points. -/
theorem parabola_fixed_y_coordinate (b c m n : ℝ) : 
  (∃ x, x^2 + b*x + c = 0 ∧ ∀ y, y ≠ x → y^2 + b*y + c ≠ 0) →  -- Parabola intersects x-axis at only one point
  (m^2 + b*m + c = n) →                                       -- Point (m, n) is on the parabola
  ((m-8)^2 + b*(m-8) + c = n) →                               -- Point (m-8, n) is on the parabola
  n = 16 := by
sorry

end NUMINAMATH_CALUDE_parabola_fixed_y_coordinate_l3726_372650


namespace NUMINAMATH_CALUDE_norm_took_110_photos_l3726_372626

/-- The number of photos taken by each photographer --/
structure PhotoCount where
  lisa : ℕ
  mike : ℕ
  norm : ℕ

/-- The conditions of the problem --/
def satisfies_conditions (p : PhotoCount) : Prop :=
  p.lisa + p.mike = p.mike + p.norm - 60 ∧
  p.norm = 2 * p.lisa + 10

/-- The theorem stating that Norm took 110 photos --/
theorem norm_took_110_photos (p : PhotoCount) 
  (h : satisfies_conditions p) : p.norm = 110 := by
  sorry

end NUMINAMATH_CALUDE_norm_took_110_photos_l3726_372626


namespace NUMINAMATH_CALUDE_probability_b_draws_red_l3726_372617

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

theorem probability_b_draws_red :
  let prob_b_red : ℚ := 
    (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) +
    (white_balls : ℚ) / total_balls * (red_balls : ℚ) / (total_balls - 1)
  prob_b_red = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_b_draws_red_l3726_372617


namespace NUMINAMATH_CALUDE_smallest_unique_sum_l3726_372621

/-- 
Given two natural numbers a and b, if their sum c can be uniquely represented 
in the form A + B = AV (where A, B, and V are distinct letters representing 
distinct digits), then the smallest possible value of c is 10.
-/
theorem smallest_unique_sum (a b : ℕ) : 
  (∃! (A B V : ℕ), A < 10 ∧ B < 10 ∧ V < 10 ∧ A ≠ B ∧ A ≠ V ∧ B ≠ V ∧ 
    a + b = c ∧ 10 * A + V = c ∧ a = A ∧ b = B) → 
  (∀ c' : ℕ, c' < c → ¬∃! (A' B' V' : ℕ), A' < 10 ∧ B' < 10 ∧ V' < 10 ∧ 
    A' ≠ B' ∧ A' ≠ V' ∧ B' ≠ V' ∧ a + b = c' ∧ 10 * A' + V' = c' ∧ a = A' ∧ b = B') →
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_unique_sum_l3726_372621


namespace NUMINAMATH_CALUDE_girls_with_rulers_l3726_372687

theorem girls_with_rulers (total_students : ℕ) (students_with_rulers : ℕ) (boys_with_set_squares : ℕ) (total_girls : ℕ) :
  total_students = 50 →
  students_with_rulers = 28 →
  boys_with_set_squares = 14 →
  total_girls = 31 →
  (total_students - students_with_rulers) = boys_with_set_squares + (total_girls - (total_students - students_with_rulers - boys_with_set_squares)) →
  total_girls - (total_students - students_with_rulers - boys_with_set_squares) = 23 :=
by sorry

end NUMINAMATH_CALUDE_girls_with_rulers_l3726_372687


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l3726_372634

theorem congruence_solutions_count :
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x > 0 ∧ x < 150 ∧ (x + 15) % 45 = 75 % 45) ∧
    (∀ x, x > 0 → x < 150 → (x + 15) % 45 = 75 % 45 → x ∈ S) ∧
    Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l3726_372634


namespace NUMINAMATH_CALUDE_squares_end_same_digit_l3726_372624

theorem squares_end_same_digit (a b : ℤ) : 
  (a + b) % 10 = 0 → a^2 % 10 = b^2 % 10 := by
  sorry

end NUMINAMATH_CALUDE_squares_end_same_digit_l3726_372624


namespace NUMINAMATH_CALUDE_inequality_proof_l3726_372602

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3726_372602


namespace NUMINAMATH_CALUDE_distribution_plans_count_l3726_372664

-- Define the number of awards and schools
def total_awards : ℕ := 7
def num_schools : ℕ := 5
def min_awards_per_special_school : ℕ := 2
def num_special_schools : ℕ := 2

-- Define the function to calculate the number of distribution plans
def num_distribution_plans : ℕ :=
  Nat.choose (total_awards - min_awards_per_special_school * num_special_schools + num_schools - 1) (num_schools - 1)

-- Theorem statement
theorem distribution_plans_count :
  num_distribution_plans = 35 :=
sorry

end NUMINAMATH_CALUDE_distribution_plans_count_l3726_372664


namespace NUMINAMATH_CALUDE_amp_composition_l3726_372623

-- Define the & operation
def amp (x : ℝ) : ℝ := 9 - x

-- Define the & operation
def amp_rev (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_composition : amp_rev (amp 15) = -15 := by sorry

end NUMINAMATH_CALUDE_amp_composition_l3726_372623


namespace NUMINAMATH_CALUDE_sum_of_integers_l3726_372612

theorem sum_of_integers : (-1) + 2 + (-3) + 1 + (-2) + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3726_372612


namespace NUMINAMATH_CALUDE_divisibility_condition_l3726_372685

theorem divisibility_condition (M : ℕ) : 
  M > 0 ∧ M < 10 → (5 ∣ 1989^M + M^1989 ↔ M = 1 ∨ M = 4) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3726_372685


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l3726_372694

def numerator : ℕ := 25 * 26 * 27 * 28 * 29 * 30
def denominator : ℕ := 1250

theorem units_digit_of_fraction : (numerator / denominator) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l3726_372694


namespace NUMINAMATH_CALUDE_function_inequality_l3726_372657

open Real

theorem function_inequality (f : ℝ → ℝ) (h : ∀ x > 0, Real.sqrt x * (deriv f x) < (1 / 2)) :
  f 9 - 1 < f 4 ∧ f 4 < f 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3726_372657


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3726_372699

theorem perfect_square_condition (p : ℕ) : 
  Nat.Prime p → (∃ (x : ℕ), 7^p - p - 16 = x^2) ↔ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3726_372699
