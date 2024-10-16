import Mathlib

namespace NUMINAMATH_CALUDE_compensation_problem_l1701_170158

/-- Represents the compensation amounts for cow, horse, and sheep respectively -/
structure Compensation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem statement -/
theorem compensation_problem (comp : Compensation) : 
  -- Total compensation is 5 measures (50 liters)
  comp.a + comp.b + comp.c = 50 →
  -- Sheep ate half as much as horse
  comp.c = (1/2) * comp.b →
  -- Horse ate half as much as cow
  comp.b = (1/2) * comp.a →
  -- Compensation is proportional to what each animal ate
  (∃ (k : ℚ), k > 0 ∧ comp.a = k * 4 ∧ comp.b = k * 2 ∧ comp.c = k * 1) →
  -- Prove that a, b, c form a geometric sequence with ratio 1/2 and c = 50/7
  (comp.b = (1/2) * comp.a ∧ comp.c = (1/2) * comp.b) ∧ comp.c = 50/7 := by
  sorry

end NUMINAMATH_CALUDE_compensation_problem_l1701_170158


namespace NUMINAMATH_CALUDE_sin_cube_identity_l1701_170174

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3*θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l1701_170174


namespace NUMINAMATH_CALUDE_binomial_expansion_103_l1701_170199

theorem binomial_expansion_103 : 102^3 + 3*(102^2) + 3*102 + 1 = 103^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_103_l1701_170199


namespace NUMINAMATH_CALUDE_equation_solutions_l1701_170171

theorem equation_solutions :
  (∃! x : ℝ, x^2 - 2*x = -1) ∧
  (∀ x : ℝ, (x + 3)^2 = 2*x*(x + 3) ↔ x = -3 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1701_170171


namespace NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l1701_170173

theorem reciprocal_sum_fourths_sixths : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l1701_170173


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_seven_min_value_exists_l1701_170113

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 1) + 8 / b = 2 → 2 * x + y ≤ 2 * a + b :=
by sorry

theorem min_value_is_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  2 * x + y ≥ 7 :=
by sorry

theorem min_value_exists (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 8 / b = 2 ∧ 2 * a + b = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_seven_min_value_exists_l1701_170113


namespace NUMINAMATH_CALUDE_circular_triangle_angle_sum_l1701_170166

/-- Represents a circular triangle --/
structure CircularTriangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  r_a : ℝ  -- radius of arc forming side a
  r_b : ℝ  -- radius of arc forming side b
  r_c : ℝ  -- radius of arc forming side c
  s_a : Int  -- sign of side a (1 or -1)
  s_b : Int  -- sign of side b (1 or -1)
  s_c : Int  -- sign of side c (1 or -1)

/-- The theorem about the sum of angles in a circular triangle --/
theorem circular_triangle_angle_sum (t : CircularTriangle) :
  t.A + t.B + t.C - (t.s_a : ℝ) * (t.a / t.r_a) - (t.s_b : ℝ) * (t.b / t.r_b) - (t.s_c : ℝ) * (t.c / t.r_c) = π :=
by sorry

end NUMINAMATH_CALUDE_circular_triangle_angle_sum_l1701_170166


namespace NUMINAMATH_CALUDE_dannys_bottle_caps_l1701_170164

/-- Calculates the total number of bottle caps in Danny's collection -/
def total_bottle_caps (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Danny's total bottle caps is 55 -/
theorem dannys_bottle_caps :
  total_bottle_caps 37 18 = 55 := by
  sorry

end NUMINAMATH_CALUDE_dannys_bottle_caps_l1701_170164


namespace NUMINAMATH_CALUDE_exam_scores_l1701_170161

theorem exam_scores (x y : ℝ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  (x + 2 = 10) ∧ ((x * y + 98 + 70) / (x + 2) = 88) :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_l1701_170161


namespace NUMINAMATH_CALUDE_image_of_four_six_l1701_170193

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem image_of_four_six : f (4, 6) = (10, -2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_four_six_l1701_170193


namespace NUMINAMATH_CALUDE_radio_show_song_time_l1701_170196

/-- Calculates the time spent on songs in a radio show -/
theorem radio_show_song_time (total_show_time : ℕ) (talking_segment_duration : ℕ) 
  (ad_break_duration : ℕ) (num_talking_segments : ℕ) (num_ad_breaks : ℕ) :
  total_show_time = 3 * 60 →
  talking_segment_duration = 10 →
  ad_break_duration = 5 →
  num_talking_segments = 3 →
  num_ad_breaks = 5 →
  total_show_time - (num_talking_segments * talking_segment_duration + num_ad_breaks * ad_break_duration) = 125 := by
  sorry

end NUMINAMATH_CALUDE_radio_show_song_time_l1701_170196


namespace NUMINAMATH_CALUDE_fran_red_macaroons_l1701_170123

/-- Represents the number of macaroons in various states --/
structure MacaroonCounts where
  green_baked : ℕ
  green_eaten : ℕ
  red_eaten : ℕ
  total_remaining : ℕ

/-- The theorem stating the number of red macaroons Fran baked --/
theorem fran_red_macaroons (m : MacaroonCounts) 
  (h1 : m.green_baked = 40)
  (h2 : m.green_eaten = 15)
  (h3 : m.red_eaten = 2 * m.green_eaten)
  (h4 : m.total_remaining = 45) :
  ∃ red_baked : ℕ, red_baked = 50 ∧ 
    red_baked = m.red_eaten + (m.total_remaining - (m.green_baked - m.green_eaten)) :=
by sorry

end NUMINAMATH_CALUDE_fran_red_macaroons_l1701_170123


namespace NUMINAMATH_CALUDE_exponential_function_property_l1701_170192

theorem exponential_function_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x ∈ Set.Icc 0 2, a^x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 2, a^x ≥ a^2) ∧
  (1 - a^2 = 3/4) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_property_l1701_170192


namespace NUMINAMATH_CALUDE_f_negative_a_l1701_170121

theorem f_negative_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + Real.sin x + 1
  f a = 2 → f (-a) = -2 := by
sorry

end NUMINAMATH_CALUDE_f_negative_a_l1701_170121


namespace NUMINAMATH_CALUDE_reptiles_per_swamp_l1701_170169

theorem reptiles_per_swamp (total_reptiles : ℕ) (num_swamps : ℕ) 
  (h1 : total_reptiles = 1424) (h2 : num_swamps = 4) :
  total_reptiles / num_swamps = 356 := by
  sorry

end NUMINAMATH_CALUDE_reptiles_per_swamp_l1701_170169


namespace NUMINAMATH_CALUDE_james_lifting_time_l1701_170143

/-- Calculates the number of days until James can lift heavy again after an injury -/
def daysUntilHeavyLifting (painSubsideDays : ℕ) (healingMultiplier : ℕ) (waitAfterHealingDays : ℕ) (waitBeforeHeavyWeeks : ℕ) : ℕ :=
  let fullHealingDays := painSubsideDays * healingMultiplier
  let totalBeforeExercise := fullHealingDays + waitAfterHealingDays
  let waitBeforeHeavyDays := waitBeforeHeavyWeeks * 7
  totalBeforeExercise + waitBeforeHeavyDays

/-- Theorem stating that given the specific conditions, James can lift heavy after 39 days -/
theorem james_lifting_time :
  daysUntilHeavyLifting 3 5 3 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_james_lifting_time_l1701_170143


namespace NUMINAMATH_CALUDE_nickys_card_value_l1701_170107

/-- Proves that if Nicky trades two cards of equal value for one card worth $21 
    and makes a profit of $5, then each of Nicky's cards is worth $8. -/
theorem nickys_card_value (card_value : ℝ) : 
  (2 * card_value + 5 = 21) → card_value = 8 := by
  sorry

end NUMINAMATH_CALUDE_nickys_card_value_l1701_170107


namespace NUMINAMATH_CALUDE_range_of_a_l1701_170180

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 ≥ 0 ∧ a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0

-- Define the theorem
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (-1 ≤ a ∧ a ≤ 1) ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1701_170180


namespace NUMINAMATH_CALUDE_inequality_relation_l1701_170184

theorem inequality_relation (n : ℕ) (hn : n > 1) :
  (1 : ℝ) / n > Real.log ((n + 1 : ℝ) / n) ∧
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l1701_170184


namespace NUMINAMATH_CALUDE_felix_lifting_capacity_l1701_170163

/-- Felix's lifting capacity problem -/
theorem felix_lifting_capacity 
  (felix_lift_ratio : ℝ) 
  (brother_weight_ratio : ℝ) 
  (brother_lift_ratio : ℝ) 
  (brother_lift_weight : ℝ) 
  (h1 : felix_lift_ratio = 1.5)
  (h2 : brother_weight_ratio = 2)
  (h3 : brother_lift_ratio = 3)
  (h4 : brother_lift_weight = 600) :
  felix_lift_ratio * (brother_lift_weight / brother_lift_ratio / brother_weight_ratio) = 150 := by
  sorry


end NUMINAMATH_CALUDE_felix_lifting_capacity_l1701_170163


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1701_170190

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Theorem stating the properties of a specific quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction)
  (point1 : f.a * (-1)^2 + f.b * (-1) + f.c = -1)
  (point2 : f.c = 1)
  (condition : f.a * (-2)^2 + f.b * (-2) + f.c > 1) :
  (f.a * f.b * f.c > 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ f.a * x^2 + f.b * x + f.c - 3 = 0 ∧ f.a * y^2 + f.b * y + f.c - 3 = 0) ∧
  (f.a + f.b + f.c > 7) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1701_170190


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_x_squared_l1701_170101

theorem integral_sqrt_minus_x_squared :
  ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - (x - 1)^2) - x^2) = π / 4 - 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_x_squared_l1701_170101


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1701_170154

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1701_170154


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1701_170149

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < s.sample_size ∧ n = s.first_element + k * s.interval

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 52)
  (h_sample : s.sample_size = 4)
  (h_5 : s.contains 5)
  (h_31 : s.contains 31)
  (h_44 : s.contains 44)
  : s.contains 18 := by
  sorry

#check systematic_sample_fourth_element

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1701_170149


namespace NUMINAMATH_CALUDE_table_height_l1701_170126

/-- The height of the table given the configurations of wooden blocks -/
theorem table_height (l w h h₃ : ℝ) 
  (config_a : l + h - w = 50)
  (config_b : w + h + h₃ - l = 44)
  (h₃_height : h₃ = 4) : h = 45 := by
  sorry

end NUMINAMATH_CALUDE_table_height_l1701_170126


namespace NUMINAMATH_CALUDE_balloon_difference_l1701_170153

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687

theorem balloon_difference : james_balloons - (amy_balloons + felix_balloons) = 22 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l1701_170153


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1701_170110

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^2 * y + y^2 = x^3 ↔ (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1701_170110


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_half_l1701_170139

theorem angle_sum_is_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : 
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_half_l1701_170139


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l1701_170172

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < 1, ∃ z, (y + 2) / (10 * y^2 - 90 * y + 20) = z) ∧ 
  ¬∃ z, (1 + 2) / (10 * 1^2 - 90 * 1 + 20) = z := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l1701_170172


namespace NUMINAMATH_CALUDE_stating_locus_of_vertex_c_l1701_170138

/-- Represents a triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of median from A to BC -/
  median_a : ℝ
  /-- Length of altitude from A to BC -/
  altitude_a : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ

/-- 
Theorem stating that the locus of vertex C in the special triangle 
is a circle with specific properties
-/
theorem locus_of_vertex_c (t : SpecialTriangle) 
  (h1 : t.ab = 6)
  (h2 : t.median_a = 4)
  (h3 : t.altitude_a = 3) :
  ∃ (c : Circle), 
    c.radius = 3 ∧ 
    c.center.1 = 4 ∧ 
    c.center.2 = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ {p | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2} ↔ 
      ∃ (triangle : SpecialTriangle), 
        triangle.ab = t.ab ∧ 
        triangle.median_a = t.median_a ∧ 
        triangle.altitude_a = t.altitude_a) :=
sorry

end NUMINAMATH_CALUDE_stating_locus_of_vertex_c_l1701_170138


namespace NUMINAMATH_CALUDE_group_interval_equals_frequency_over_height_l1701_170133

/-- Given a group [a, b] in a sampling process with frequency m and histogram height h, 
    prove that the group interval |a-b| equals m/h -/
theorem group_interval_equals_frequency_over_height 
  (a b m h : ℝ) (hm : m > 0) (hh : h > 0) : |a - b| = m / h := by
  sorry

end NUMINAMATH_CALUDE_group_interval_equals_frequency_over_height_l1701_170133


namespace NUMINAMATH_CALUDE_boys_from_maple_high_school_l1701_170195

theorem boys_from_maple_high_school (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (jonas_students : ℕ) (clay_students : ℕ) (maple_students : ℕ)
  (jonas_girls : ℕ) (clay_girls : ℕ) :
  total_students = 150 →
  total_boys = 85 →
  total_girls = 65 →
  jonas_students = 50 →
  clay_students = 70 →
  maple_students = 30 →
  jonas_girls = 25 →
  clay_girls = 30 →
  total_students = total_boys + total_girls →
  total_students = jonas_students + clay_students + maple_students →
  (maple_students - (total_girls - jonas_girls - clay_girls) : ℤ) = 20 := by
sorry

end NUMINAMATH_CALUDE_boys_from_maple_high_school_l1701_170195


namespace NUMINAMATH_CALUDE_zhang_or_beibei_probability_l1701_170191

/-- The number of singers in total -/
def total_singers : ℕ := 5

/-- The number of singers to be signed -/
def singers_to_sign : ℕ := 3

/-- The probability of signing a specific combination of singers -/
def prob_combination : ℚ := 1 / (total_singers.choose singers_to_sign)

/-- The probability that either Zhang Lei or Beibei will be signed -/
def prob_zhang_or_beibei : ℚ := 1 - ((total_singers - 2).choose singers_to_sign) * prob_combination

theorem zhang_or_beibei_probability :
  prob_zhang_or_beibei = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_zhang_or_beibei_probability_l1701_170191


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1701_170179

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum (n : ℕ) :
  geometric_sum (1/3) (1/3) n = 26/81 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1701_170179


namespace NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l1701_170144

/-- The probability that a randomly selected point (x, y) in the rectangle
    [0, 4] × [0, 8] satisfies x + y ≤ 6 is 3/8. -/
theorem probability_x_plus_y_leq_6 :
  let total_area : ℝ := 4 * 8
  let valid_area : ℝ := (1 / 2) * 4 * 6
  valid_area / total_area = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l1701_170144


namespace NUMINAMATH_CALUDE_problem_statement_l1701_170100

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 3) :
  a + b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1701_170100


namespace NUMINAMATH_CALUDE_problem_statement_l1701_170108

theorem problem_statement (x y : ℝ) 
  (h1 : 4 + x = 5 - y) 
  (h2 : 3 + y = 6 + x) : 
  4 - x = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1701_170108


namespace NUMINAMATH_CALUDE_hash_six_two_l1701_170116

-- Define the # operation
def hash (x y : ℝ) : ℝ := 4*x - 4*y

-- Theorem statement
theorem hash_six_two : hash 6 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_hash_six_two_l1701_170116


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l1701_170137

theorem sum_reciprocals_equals_one 
  (a b c d : ℝ) 
  (ω : ℂ) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω1 : ω^4 = 1) 
  (hω2 : ω ≠ 1) 
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / (ω + 1)) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l1701_170137


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_neg_two_implies_result_l1701_170145

theorem tan_pi_minus_alpha_eq_neg_two_implies_result (α : Real) 
  (h : Real.tan (Real.pi - α) = -2) : 
  1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_neg_two_implies_result_l1701_170145


namespace NUMINAMATH_CALUDE_election_votes_l1701_170152

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) 
  (a_excess_percent : ℚ) (c_percent : ℚ) 
  (h_total : total_votes = 6800)
  (h_invalid : invalid_percent = 30 / 100)
  (h_a_excess : a_excess_percent = 18 / 100)
  (h_c : c_percent = 12 / 100) : 
  ∃ (b_votes c_votes : ℕ), 
    b_votes + c_votes = 2176 ∧ 
    b_votes + c_votes + (b_votes + (a_excess_percent * total_votes).floor) = 
      (total_votes * (1 - invalid_percent)).floor ∧
    c_votes = (c_percent * total_votes).floor := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l1701_170152


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l1701_170181

theorem trig_fraction_equality (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1/2) :
  Real.cos a / (Real.sin a - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l1701_170181


namespace NUMINAMATH_CALUDE_chessboard_coloring_l1701_170136

/-- A coloring of an n × n chessboard is valid if for every i ∈ {1,2,...,n}, 
    the 2n-1 cells on i-th row and i-th column have all different colors. -/
def ValidColoring (n : ℕ) (k : ℕ) : Prop :=
  ∃ (coloring : Fin n → Fin n → Fin k),
    ∀ i : Fin n, (∀ j j' : Fin n, j ≠ j' → coloring i j ≠ coloring i j') ∧
                 (∀ i' : Fin n, i ≠ i' → coloring i i' ≠ coloring i' i)

theorem chessboard_coloring :
  (¬ ValidColoring 2001 4001) ∧
  (∀ m : ℕ, ValidColoring (2^m - 1) (2^(m+1) - 1)) :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_l1701_170136


namespace NUMINAMATH_CALUDE_solution_to_equation_l1701_170130

theorem solution_to_equation (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^4 = (15 * x)^3 ↔ x = 27 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1701_170130


namespace NUMINAMATH_CALUDE_penny_theorem_l1701_170182

def penny_problem (initial_amount : ℕ) (sock_pairs : ℕ) (sock_price : ℕ) (hat_price : ℕ) : Prop :=
  let total_spent := sock_pairs * sock_price + hat_price
  initial_amount - total_spent = 5

theorem penny_theorem : 
  penny_problem 20 4 2 7 := by
  sorry

end NUMINAMATH_CALUDE_penny_theorem_l1701_170182


namespace NUMINAMATH_CALUDE_inequality_proof_l1701_170120

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1701_170120


namespace NUMINAMATH_CALUDE_modified_triangle_sum_l1701_170175

/-- Represents the sum of numbers in the nth row of the modified triangular array -/
def f : ℕ → ℕ
  | 0 => 0  -- We define f(0) as 0 to make the function total
  | 1 => 0  -- First row starts with 0
  | (n + 2) => 2 * f (n + 1) + (n + 2) * (n + 2)

theorem modified_triangle_sum : f 100 = 2^100 - 10000 := by
  sorry

end NUMINAMATH_CALUDE_modified_triangle_sum_l1701_170175


namespace NUMINAMATH_CALUDE_ball_distribution_l1701_170177

theorem ball_distribution (n : ℕ) (k : ℕ) : 
  (∃ x y z : ℕ, x + y + z = n ∧ x ≥ 1 ∧ y ≥ 2 ∧ z ≥ 3) →
  (Nat.choose (n - 6 + k - 1) (k - 1) = 15) →
  (k = 3 ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_l1701_170177


namespace NUMINAMATH_CALUDE_hyperbola_and_ellipse_condition_l1701_170134

/-- Represents a hyperbola equation -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (1 - m) + y^2 / (m + 2) = 1

/-- Represents an ellipse equation with foci on the x-axis -/
def is_ellipse_x_foci (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2 * m) + y^2 / (2 - m) = 1

/-- Main theorem -/
theorem hyperbola_and_ellipse_condition (m : ℝ) 
  (h1 : is_hyperbola m) (h2 : is_ellipse_x_foci m) : 
  1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_ellipse_condition_l1701_170134


namespace NUMINAMATH_CALUDE_triangle_formation_with_6_and_8_l1701_170112

/-- A function that checks if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating which length among given options can form a triangle with sides 6 and 8 --/
theorem triangle_formation_with_6_and_8 :
  can_form_triangle 6 8 13 ∧
  ¬(can_form_triangle 6 8 1) ∧
  ¬(can_form_triangle 6 8 2) ∧
  ¬(can_form_triangle 6 8 14) := by
  sorry


end NUMINAMATH_CALUDE_triangle_formation_with_6_and_8_l1701_170112


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1701_170118

theorem gcd_of_polynomial_and_multiple : ∀ y : ℤ, 
  9240 ∣ y → 
  Int.gcd ((5*y+3)*(11*y+2)*(17*y+8)*(4*y+7)) y = 168 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1701_170118


namespace NUMINAMATH_CALUDE_wilsborough_savings_l1701_170150

/-- Mrs. Wilsborough's concert ticket purchase problem -/
theorem wilsborough_savings : 
  let initial_savings : ℕ := 500
  let vip_ticket_price : ℕ := 100
  let regular_ticket_price : ℕ := 50
  let vip_tickets_bought : ℕ := 2
  let regular_tickets_bought : ℕ := 3
  let total_spent : ℕ := vip_ticket_price * vip_tickets_bought + regular_ticket_price * regular_tickets_bought
  let remaining_savings : ℕ := initial_savings - total_spent
  remaining_savings = 150 := by sorry

end NUMINAMATH_CALUDE_wilsborough_savings_l1701_170150


namespace NUMINAMATH_CALUDE_division_of_fractions_l1701_170131

theorem division_of_fractions : (5 : ℚ) / 6 / (2 + 2 / 3) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1701_170131


namespace NUMINAMATH_CALUDE_black_squares_in_58th_row_l1701_170103

/-- Represents a square color in the stair-step figure -/
inductive SquareColor
| White
| Black
| Red

/-- Represents a row in the stair-step figure -/
def StairRow := List SquareColor

/-- Generates a row of the stair-step figure -/
def generateRow (n : ℕ) : StairRow :=
  sorry

/-- Counts the number of black squares in a row -/
def countBlackSquares (row : StairRow) : ℕ :=
  sorry

/-- Main theorem: The number of black squares in the 58th row is 38 -/
theorem black_squares_in_58th_row :
  countBlackSquares (generateRow 58) = 38 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_in_58th_row_l1701_170103


namespace NUMINAMATH_CALUDE_circle_intersection_angle_l1701_170140

-- Define the circle equation
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + c = 0

-- Define the center of the circle
def center : ℝ × ℝ := (2, -1)

-- Define the angle APB
def angle_APB : ℝ := 120

-- Theorem statement
theorem circle_intersection_angle (c : ℝ) :
  ∃ (A B : ℝ × ℝ),
    (A.1 = 0 ∧ circle_equation A.1 A.2 c) ∧
    (B.1 = 0 ∧ circle_equation B.1 B.2 c) ∧
    (angle_APB = 120) →
    c = -11 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_angle_l1701_170140


namespace NUMINAMATH_CALUDE_bird_fence_difference_l1701_170141

/-- Given the initial and additional numbers of sparrows and pigeons on a fence,
    and the fact that all starlings flew away, prove that there are 2 more sparrows
    than pigeons on the fence after these events. -/
theorem bird_fence_difference
  (initial_sparrows : ℕ)
  (initial_pigeons : ℕ)
  (additional_sparrows : ℕ)
  (additional_pigeons : ℕ)
  (h1 : initial_sparrows = 3)
  (h2 : initial_pigeons = 2)
  (h3 : additional_sparrows = 4)
  (h4 : additional_pigeons = 3) :
  (initial_sparrows + additional_sparrows) - (initial_pigeons + additional_pigeons) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_fence_difference_l1701_170141


namespace NUMINAMATH_CALUDE_triangles_with_integer_sides_not_exceeding_two_l1701_170156

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_sides_not_exceeding_two (a b c : ℕ) : Prop :=
  a ≤ 2 ∧ b ≤ 2 ∧ c ≤ 2

theorem triangles_with_integer_sides_not_exceeding_two :
  ∃! (S : Set (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ 
      is_valid_triangle a b c ∧ 
      triangle_sides_not_exceeding_two a b c) ∧
    S = {(1, 1, 1), (2, 2, 1), (2, 2, 2)} :=
sorry

end NUMINAMATH_CALUDE_triangles_with_integer_sides_not_exceeding_two_l1701_170156


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l1701_170102

theorem minimum_value_of_expression (x : ℝ) (h : x > 0) :
  9 * x + 1 / x^6 ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / y^6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l1701_170102


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1701_170117

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 12) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 162 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1701_170117


namespace NUMINAMATH_CALUDE_floor_multiple_implies_integer_l1701_170165

theorem floor_multiple_implies_integer (r : ℝ) : 
  r ≥ 1 →
  (∀ (m n : ℕ+), n.val % m.val = 0 → (⌊n.val * r⌋ : ℤ) % (⌊m.val * r⌋ : ℤ) = 0) →
  ∃ (k : ℤ), r = k := by
  sorry

end NUMINAMATH_CALUDE_floor_multiple_implies_integer_l1701_170165


namespace NUMINAMATH_CALUDE_factorization_equality_l1701_170189

theorem factorization_equality (a b : ℝ) : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1701_170189


namespace NUMINAMATH_CALUDE_pool_filling_time_l1701_170194

theorem pool_filling_time (R : ℝ) (h1 : R > 0) : 
  (R + 1.5 * R) * 5 = 1 → R * 12.5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1701_170194


namespace NUMINAMATH_CALUDE_f_value_at_5pi_3_l1701_170160

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_value_at_5pi_3 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : is_periodic f π)
  (h_sin : ∀ x ∈ Set.Icc 0 (π/2), f x = Real.sin x) :
  f (5*π/3) = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_5pi_3_l1701_170160


namespace NUMINAMATH_CALUDE_tech_company_work_hours_l1701_170188

/-- Calculates the total hours worked in a day for a tech company's help desk -/
theorem tech_company_work_hours :
  let total_hours : ℝ := 24
  let software_hours : ℝ := 24
  let user_help_hours : ℝ := 17
  let maintenance_percent : ℝ := 35
  let research_dev_percent : ℝ := 27
  let marketing_percent : ℝ := 15
  let multitasking_employees : ℕ := 3
  let additional_employees : ℕ := 4
  let additional_hours : ℝ := 12
  
  (maintenance_percent + research_dev_percent + marketing_percent) / 100 * total_hours +
  software_hours + user_help_hours ≥ total_hours →
  
  (max software_hours (max user_help_hours ((maintenance_percent + research_dev_percent + marketing_percent) / 100 * total_hours))) +
  additional_hours = 36 :=
by sorry

end NUMINAMATH_CALUDE_tech_company_work_hours_l1701_170188


namespace NUMINAMATH_CALUDE_fragment_sheets_l1701_170146

/-- Represents a book fragment with a first and last page number -/
structure BookFragment where
  first_page : Nat
  last_page : Nat

/-- Check if two numbers have the same digits -/
def same_digits (a b : Nat) : Prop := sorry

/-- Calculate the number of sheets in a book fragment -/
def num_sheets (fragment : BookFragment) : Nat :=
  (fragment.last_page - fragment.first_page + 1) / 2

/-- Theorem stating the number of sheets in the specific fragment -/
theorem fragment_sheets :
  ∀ (fragment : BookFragment),
    fragment.first_page = 435 →
    fragment.last_page > fragment.first_page →
    same_digits fragment.first_page fragment.last_page →
    num_sheets fragment = 50 := by
  sorry

end NUMINAMATH_CALUDE_fragment_sheets_l1701_170146


namespace NUMINAMATH_CALUDE_tan_difference_special_angle_l1701_170151

theorem tan_difference_special_angle (α : Real) :
  2 * Real.tan α = 3 * Real.tan (π / 8) →
  Real.tan (α - π / 8) = (5 * Real.sqrt 2 + 1) / 49 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_special_angle_l1701_170151


namespace NUMINAMATH_CALUDE_system_solutions_l1701_170186

def system_solution (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∧
  (x^(Real.log x) * y^(Real.log y) = 243) ∧
  ((3 / Real.log x) * x * y^(Real.log y) = 1)

theorem system_solutions :
  {(x, y) : ℝ × ℝ | system_solution x y} =
  {(9, 3), (3, 9), (1/9, 1/3), (1/3, 1/9)} := by
sorry

end NUMINAMATH_CALUDE_system_solutions_l1701_170186


namespace NUMINAMATH_CALUDE_mn_value_l1701_170124

theorem mn_value (m n : ℕ+) (h : m.val^4 - n.val^4 = 3439) : m.val * n.val = 90 := by
  sorry

end NUMINAMATH_CALUDE_mn_value_l1701_170124


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1701_170176

theorem quadratic_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1701_170176


namespace NUMINAMATH_CALUDE_kelly_games_to_give_away_l1701_170109

theorem kelly_games_to_give_away (initial_games : ℕ) (remaining_games : ℕ) : 
  initial_games - remaining_games = 15 :=
by
  sorry

#check kelly_games_to_give_away 50 35

end NUMINAMATH_CALUDE_kelly_games_to_give_away_l1701_170109


namespace NUMINAMATH_CALUDE_fewer_female_students_l1701_170128

theorem fewer_female_students (total_students : ℕ) (female_students : ℕ) 
  (h1 : total_students = 280) (h2 : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end NUMINAMATH_CALUDE_fewer_female_students_l1701_170128


namespace NUMINAMATH_CALUDE_point_outside_circle_l1701_170197

/-- A line intersects a circle if and only if the distance from the center of the circle to the line is less than the radius of the circle. -/
axiom line_intersects_circle (a b : ℝ) : 
  (∃ x y, a * x + b * y = 1 ∧ x^2 + y^2 = 1) ↔ (1 / Real.sqrt (a^2 + b^2) < 1)

/-- A point (x, y) is outside a circle centered at the origin with radius r if and only if x^2 + y^2 > r^2. -/
def outside_circle (x y r : ℝ) : Prop := x^2 + y^2 > r^2

theorem point_outside_circle (a b : ℝ) :
  (∃ x y, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → outside_circle a b 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1701_170197


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l1701_170178

theorem pencil_eraser_cost :
  ∃ (p e : ℕ), 
    p > 0 ∧ 
    e > 0 ∧ 
    7 * p + 5 * e = 130 ∧ 
    p > e ∧ 
    p + e = 22 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l1701_170178


namespace NUMINAMATH_CALUDE_max_npm_value_l1701_170168

/-- Represents a two-digit number with equal even digits -/
structure EvenTwoDigit where
  digit : Nat
  h1 : digit % 2 = 0
  h2 : digit < 10

/-- Represents a three-digit number of the form NPM -/
structure ThreeDigitNPM where
  n : Nat
  p : Nat
  m : Nat
  h1 : n > 0
  h2 : n < 10
  h3 : p < 10
  h4 : m < 10

/-- The main theorem stating the maximum value of NPM -/
theorem max_npm_value (mm : EvenTwoDigit) (m : Nat) (npm : ThreeDigitNPM) 
    (h1 : m < 10)
    (h2 : m = mm.digit)
    (h3 : m = npm.m)
    (h4 : (mm.digit * 10 + mm.digit) * m = npm.n * 100 + npm.p * 10 + npm.m) :
  npm.n * 100 + npm.p * 10 + npm.m ≤ 396 := by
  sorry

end NUMINAMATH_CALUDE_max_npm_value_l1701_170168


namespace NUMINAMATH_CALUDE_complex_combination_equality_l1701_170111

/-- Given complex numbers Q, E, D, and F, prove that their combination equals 1 + 117i -/
theorem complex_combination_equality (Q E D F : ℂ) 
  (hQ : Q = 7 + 3*I) 
  (hE : E = 2*I) 
  (hD : D = 7 - 3*I) 
  (hF : F = 1 + I) : 
  (Q * E * D) + F = 1 + 117*I := by
  sorry

end NUMINAMATH_CALUDE_complex_combination_equality_l1701_170111


namespace NUMINAMATH_CALUDE_regular_tetrahedron_sphere_ratio_l1701_170135

/-- A regular tetrahedron is a tetrahedron with four congruent equilateral triangles as faces -/
structure RegularTetrahedron where
  -- We don't need to define the structure explicitly for this problem

/-- The ratio of the radius of the circumscribed sphere to the inscribed sphere of a regular tetrahedron -/
def circumscribed_to_inscribed_ratio (t : RegularTetrahedron) : ℚ :=
  3 / 1

/-- Theorem: The ratio of the radius of the circumscribed sphere to the inscribed sphere of a regular tetrahedron is 3:1 -/
theorem regular_tetrahedron_sphere_ratio (t : RegularTetrahedron) :
  circumscribed_to_inscribed_ratio t = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_sphere_ratio_l1701_170135


namespace NUMINAMATH_CALUDE_rotation_composition_l1701_170106

/-- Represents a rotation in a plane -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Represents a translation in a plane -/
structure Translation where
  direction : ℝ × ℝ

/-- Represents the result of composing two rotations -/
inductive RotationComposition
  | IsRotation : Rotation → RotationComposition
  | IsTranslation : Translation → RotationComposition

/-- 
  Theorem: The composition of two rotations is either a rotation or a translation
  depending on the sum of their angles.
-/
theorem rotation_composition (r1 r2 : Rotation) :
  ∃ (result : RotationComposition),
    (¬ ∃ (k : ℤ), r1.angle + r2.angle = 2 * π * k → 
      ∃ (c : ℝ × ℝ), result = RotationComposition.IsRotation ⟨c, r1.angle + r2.angle⟩) ∧
    (∃ (k : ℤ), r1.angle + r2.angle = 2 * π * k → 
      ∃ (d : ℝ × ℝ), result = RotationComposition.IsTranslation ⟨d⟩) :=
by sorry


end NUMINAMATH_CALUDE_rotation_composition_l1701_170106


namespace NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l1701_170185

theorem integral_exp_plus_2x_equals_e : ∫ x in (0 : ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l1701_170185


namespace NUMINAMATH_CALUDE_smallest_k_for_mutual_criticism_l1701_170125

/-- The number of deputies in the discussion. -/
def num_deputies : ℕ := 15

/-- A function that takes the number of deputies each criticizes and returns whether
    it guarantees mutual criticism. -/
def guarantees_mutual_criticism (k : ℕ) : Prop :=
  k * num_deputies > (num_deputies * (num_deputies - 1)) / 2

/-- The theorem stating the smallest k that guarantees mutual criticism. -/
theorem smallest_k_for_mutual_criticism :
  ∃ k : ℕ, k = 8 ∧ guarantees_mutual_criticism k ∧
  ∀ m : ℕ, m < k → ¬guarantees_mutual_criticism m :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_mutual_criticism_l1701_170125


namespace NUMINAMATH_CALUDE_symmetric_points_l1701_170115

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Point P with coordinates (1, 3, 5) -/
def P : Point3D := ⟨1, 3, 5⟩

/-- Point P' with coordinates (-1, -3, -5) -/
def P' : Point3D := ⟨-1, -3, -5⟩

/-- Check if two points are symmetric with respect to the origin -/
def isSymmetricToOrigin (a b : Point3D) : Prop :=
  a.x + b.x = 0 ∧ a.y + b.y = 0 ∧ a.z + b.z = 0

/-- Theorem stating that P and P' are symmetric with respect to the origin -/
theorem symmetric_points : isSymmetricToOrigin P P' := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l1701_170115


namespace NUMINAMATH_CALUDE_circle_properties_l1701_170159

/-- Given a circle with equation 3x^2 - 4y - 12 = -3y^2 + 8x, 
    prove its center coordinates, radius, and a + 2b + r -/
theorem circle_properties : 
  ∃ (a b r : ℝ), 
    (∀ x y : ℝ, 3 * x^2 - 4 * y - 12 = -3 * y^2 + 8 * x → 
      (x - a)^2 + (y - b)^2 = r^2) ∧ 
    a = 4/3 ∧ 
    b = 2/3 ∧ 
    r = 2 * Real.sqrt 13 / 3 ∧
    a + 2 * b + r = (8 + 2 * Real.sqrt 13) / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1701_170159


namespace NUMINAMATH_CALUDE_inequality_proof_l1701_170104

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1701_170104


namespace NUMINAMATH_CALUDE_marble_redistribution_l1701_170122

/-- Proves that Tyrone giving 12.5 marbles to Eric results in Tyrone having 3 times as many marbles as Eric -/
theorem marble_redistribution (tyrone_initial : ℝ) (eric_initial : ℝ) (marbles_given : ℝ) : 
  tyrone_initial = 125 →
  eric_initial = 25 →
  marbles_given = 12.5 →
  (tyrone_initial - marbles_given) = 3 * (eric_initial + marbles_given) :=
by
  sorry

#check marble_redistribution

end NUMINAMATH_CALUDE_marble_redistribution_l1701_170122


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1701_170105

theorem floor_abs_negative_real : ⌊|(-57.6 : ℝ)|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1701_170105


namespace NUMINAMATH_CALUDE_point_on_y_axis_l1701_170129

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on the y-axis
def onYAxis (p : Point2D) : Prop := p.x = 0

-- State the theorem
theorem point_on_y_axis (m : ℝ) :
  let p := Point2D.mk (m - 1) (m + 3)
  onYAxis p → p = Point2D.mk 0 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l1701_170129


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l1701_170132

theorem lawn_mowing_problem (mary_time tom_time tom_work_time : ℝ) 
  (h1 : mary_time = 6)
  (h2 : tom_time = 4)
  (h3 : tom_work_time = 3) :
  1 - (tom_work_time / tom_time) = 1/4 := by sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l1701_170132


namespace NUMINAMATH_CALUDE_polygon_sides_l1701_170157

/-- A polygon with side length 4 and perimeter 24 has 6 sides -/
theorem polygon_sides (side_length : ℝ) (perimeter : ℝ) (num_sides : ℕ) : 
  side_length = 4 → perimeter = 24 → num_sides * side_length = perimeter → num_sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1701_170157


namespace NUMINAMATH_CALUDE_bales_equation_initial_bales_count_l1701_170198

/-- The initial number of bales in the barn -/
def initial_bales : ℕ := sorry

/-- The number of bales added to the barn -/
def added_bales : ℕ := 35

/-- The final number of bales in the barn -/
def final_bales : ℕ := 82

/-- Theorem stating that the initial number of bales plus the added bales equals the final number of bales -/
theorem bales_equation : initial_bales + added_bales = final_bales := by sorry

/-- Theorem proving that the initial number of bales was 47 -/
theorem initial_bales_count : initial_bales = 47 := by sorry

end NUMINAMATH_CALUDE_bales_equation_initial_bales_count_l1701_170198


namespace NUMINAMATH_CALUDE_tara_wrong_questions_l1701_170148

theorem tara_wrong_questions
  (total_questions : ℕ)
  (t u v w : ℕ)
  (h1 : t + u = v + w)
  (h2 : t + w = u + v + 6)
  (h3 : v = 3)
  (h4 : total_questions = 40) :
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_tara_wrong_questions_l1701_170148


namespace NUMINAMATH_CALUDE_proportion_equality_l1701_170187

theorem proportion_equality (x : ℝ) (h : (3/4) / x = 7/8) : x = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1701_170187


namespace NUMINAMATH_CALUDE_triangle_area_sqrt_3_l1701_170119

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that its area is √3 -/
theorem triangle_area_sqrt_3 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b * Real.cos C + c * Real.cos B = a * Real.cos C + c * Real.cos A)
  (h2 : b * Real.cos C + c * Real.cos B = 2)
  (h3 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  (1/2) * a * b * Real.sin C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_sqrt_3_l1701_170119


namespace NUMINAMATH_CALUDE_root_product_theorem_l1701_170162

theorem root_product_theorem (n r : ℝ) (a b : ℝ) : 
  (a^2 - n*a + 6 = 0) → 
  (b^2 - n*b + 6 = 0) → 
  ((a + 2/b)^2 - r*(a + 2/b) + s = 0) → 
  ((b + 2/a)^2 - r*(b + 2/a) + s = 0) → 
  s = 32/3 := by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1701_170162


namespace NUMINAMATH_CALUDE_total_students_in_class_l1701_170127

/-- 
Given a class where 45 students are present when 10% are absent,
prove that the total number of students in the class is 50.
-/
theorem total_students_in_class : 
  ∀ (total : ℕ), 
  (↑total * (1 - 0.1) : ℝ) = 45 → 
  total = 50 := by
sorry

end NUMINAMATH_CALUDE_total_students_in_class_l1701_170127


namespace NUMINAMATH_CALUDE_max_value_theorem_l1701_170155

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 2*a*b*Real.sqrt 3 + 2*a*c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1701_170155


namespace NUMINAMATH_CALUDE_max_value_fraction_l1701_170167

theorem max_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  (x^2 / (2 * y)) ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1701_170167


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1701_170114

theorem cube_volume_problem (s : ℝ) : 
  (s + 2)^2 * (s - 3) = s^3 + 19 → s^3 = (4 + Real.sqrt 47)^3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1701_170114


namespace NUMINAMATH_CALUDE_count_zeros_up_to_2376_l1701_170142

/-- Returns true if the given positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- Counts the number of positive integers less than or equal to n that contain the digit 0 -/
def countZeros (n : ℕ+) : ℕ :=
  sorry

/-- The number of positive integers less than or equal to 2376 that contain the digit 0 is 578 -/
theorem count_zeros_up_to_2376 : countZeros 2376 = 578 :=
  sorry

end NUMINAMATH_CALUDE_count_zeros_up_to_2376_l1701_170142


namespace NUMINAMATH_CALUDE_set_intersection_empty_iff_complement_subset_l1701_170183

universe u

theorem set_intersection_empty_iff_complement_subset {U : Type u} (A B : Set U) :
  A ∩ B = ∅ ↔ ∃ C : Set U, A ⊆ C ∧ B ⊆ Cᶜ :=
sorry

end NUMINAMATH_CALUDE_set_intersection_empty_iff_complement_subset_l1701_170183


namespace NUMINAMATH_CALUDE_hall_breadth_is_12_l1701_170170

def hall_length : ℝ := 15
def hall_volume : ℝ := 1200

theorem hall_breadth_is_12 (b h : ℝ) 
  (area_eq : 2 * (hall_length * b) = 2 * (hall_length * h + b * h))
  (volume_eq : hall_length * b * h = hall_volume) :
  b = 12 := by sorry

end NUMINAMATH_CALUDE_hall_breadth_is_12_l1701_170170


namespace NUMINAMATH_CALUDE_evaluate_expression_l1701_170147

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1701_170147
